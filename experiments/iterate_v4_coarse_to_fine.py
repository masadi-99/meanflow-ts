"""
V4 Experiment: Coarse-to-fine + variance-matched noise.

Tests the two hypothesized fixes for multi-resolution underperformance:
  1. Variance-matched noise (noise std matches target std per wavelet level)
  2. Coarse-to-fine generation (details conditioned on generated trend)

Variants:
  v3-c2f: OneFlowTSNetV3 with coarse-to-fine + variance-matched noise
  baseline: Original MeanFlow-TS v4 (reference)
"""
import os, sys, time, argparse, logging, tempfile
import numpy as np
import torch
from torch.optim import AdamW
from copy import deepcopy
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter, TestSplitSampler,
)

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oneflow_ts.model_v3 import OneFlowTSNetV3, oneflow_v3_loss, OneFlowForecasterV3
from meanflow_ts.model_v2 import (
    ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2,
    MeanFlowForecasterV2, extract_lag_features,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = "electricity_nips"
FREQ = "H"
CTX_LEN = 24
PRED_LEN = 24
MAX_LAG = 672
N_LAGS = 7
NUM_EVAL_SAMPLES = 16
TSFLOW_CRPS = 0.045


def get_data_pipeline():
    dataset = get_dataset(DATASET)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(FREQ), pred_length=PRED_LEN,
        ),
    ])
    train_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=PRED_LEN),
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed_data = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed_data), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=10000,
    )
    return dataset, transformation, train_loader


def evaluate_model(net_ema, dataset, transformation, device, is_v3):
    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )

    if is_v3:
        forecaster = OneFlowForecasterV3(
            net_ema, CTX_LEN, PRED_LEN,
            num_samples=NUM_EVAL_SAMPLES, freq=FREQ, n_lags=N_LAGS,
        ).to(device)
    else:
        forecaster = MeanFlowForecasterV2(
            net_ema, CTX_LEN, PRED_LEN, num_samples=NUM_EVAL_SAMPLES,
            freq=FREQ, n_lags=N_LAGS,
        ).to(device)

    predictor = PyTorchPredictor(
        prediction_length=PRED_LEN,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=512,
        input_transform=test_splitter, device=device,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=NUM_EVAL_SAMPLES,
    )
    forecasts = list(tqdm(forecast_it, total=len(test_transform), desc="Eval", leave=False))
    tss = list(ts_it)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    return metrics["mean_wQuantileLoss"], metrics["ND"]


def run_variant(variant, epochs, device):
    torch.manual_seed(6432)
    np.random.seed(6432)

    dataset, transformation, train_loader = get_data_pipeline()
    is_v3 = variant.startswith("v3")

    if is_v3:
        net = OneFlowTSNetV3(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            levels=2, model_channels=128, head_channels=96,
            num_ctx_blocks=2, num_head_blocks=3,
            time_emb_dim=64, dropout=0.1,
        ).to(device)
    else:
        net = ConditionalMeanFlowNetV2(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
        ).to(device)

    net_ema = deepcopy(net).eval()
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"[{variant}] Params: {params:,}")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')

    for epoch in range(epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()

        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -CTX_LEN:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
            ctx_lags = extract_lag_features(past, CTX_LEN, FREQ, N_LAGS) / loc.unsqueeze(1)
            scaled_future = future / loc

            if is_v3:
                loss = oneflow_v3_loss(net, scaled_future, ctx_lags, levels=2)
            else:
                loss = conditional_meanflow_loss_v2(net, scaled_future, ctx_lags)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
                # Also update EMA's running stats
                if is_v3:
                    for k in range(len(net.level_sizes)):
                        std_name = f'level_std_{k}'
                        mean_name = f'level_mean_{k}'
                        getattr(net_ema, std_name).lerp_(getattr(net, std_name), 0.01)
                        getattr(net_ema, mean_name).lerp_(getattr(net, mean_name), 0.01)

            epoch_loss += loss.item()
            n_b += 1

        avg_loss = epoch_loss / n_b
        elapsed = time.time() - t0

        if (epoch + 1) % 20 == 0:
            # Log running stats for V3
            extra = ""
            if is_v3:
                stds = [net.get_level_std(k).item() for k in range(len(net.level_sizes))]
                extra = f" | stds={[f'{s:.3f}' for s in stds]}"
            logger.info(f"[{variant}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s{extra}")

        if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
            crps, nd = evaluate_model(net_ema, dataset, transformation, device, is_v3)
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net_ema': net_ema.state_dict(), 'epoch': epoch+1,
                    'crps': crps, 'variant': variant,
                }, f'best_{variant}.pt')
            logger.info(f"[{variant}] Epoch {epoch+1} -> CRPS={crps:.6f} | ND={nd:.6f} | Best={best_crps:.6f} | TSFlow={TSFLOW_CRPS}")

    return best_crps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, nargs="+", default=["v3-c2f", "baseline"])
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for variant in args.variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {variant} ({args.epochs} epochs)")
        logger.info(f"{'='*60}")
        best_crps = run_variant(variant, args.epochs, device)
        results[variant] = best_crps
        logger.info(f"[{variant}] DONE — Best CRPS: {best_crps:.6f}")

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    sorted_r = sorted(results.items(), key=lambda x: x[1])
    for v, crps in sorted_r:
        marker = " <- BEST" if v == sorted_r[0][0] else ""
        logger.info(f"  {v:<30} CRPS: {crps:.6f}{marker}")
    logger.info(f"  {'TSFlow (32 NFE)':<30} CRPS: {TSFLOW_CRPS}")

    baseline = results.get("baseline", None)
    if baseline:
        for v, crps in sorted_r:
            if v != "baseline":
                pct = (crps / baseline - 1) * 100
                tag = "BETTER" if pct < 0 else "worse"
                logger.info(f"  {v}: {abs(pct):.1f}% {tag} than baseline")


if __name__ == "__main__":
    main()
