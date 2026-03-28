"""
Rapid iteration: test multiple fixes for multi-res underperformance.
Runs 100 epochs per variant, evaluates CRPS, picks the best.

Variants:
  A) flat-loss: Use oneflow_loss_simple (single JVP on full signal)
  B) level1: Use 1-level decomposition instead of 2
  C) big-heads: Increase head_channels from 64 to 128
  D) low-lr: Lower learning rate from 6e-4 to 3e-4
  E) flat-loss+big-heads: Combine A and C
  F) baseline: Single-res MeanFlow-TS v4 (for reference)
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

from oneflow_ts.model import OneFlowTSNet, OneFlowForecaster
from oneflow_ts.priors import ResolutionMatchedPrior, IsotropicPrior
from oneflow_ts.loss import oneflow_loss, oneflow_loss_simple
from oneflow_ts.wavelet import get_level_sizes
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


def evaluate_model(net_ema, dataset, transformation, device, variant_name, levels=2):
    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )

    if variant_name == "baseline":
        forecaster = MeanFlowForecasterV2(
            net_ema, CTX_LEN, PRED_LEN, num_samples=NUM_EVAL_SAMPLES,
            freq=FREQ, n_lags=N_LAGS,
        ).to(device)
    else:
        sizes = get_level_sizes(PRED_LEN, levels)
        prior = ResolutionMatchedPrior(sizes, length_scale=3.0).to(device)
        forecaster = OneFlowForecaster(
            net_ema, prior, CTX_LEN, PRED_LEN,
            num_samples=NUM_EVAL_SAMPLES, freq=FREQ, n_lags=N_LAGS,
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
    """Run a single variant and return final CRPS."""
    torch.manual_seed(6432)
    np.random.seed(6432)

    dataset, transformation, train_loader = get_data_pipeline()

    # Configure variant
    if variant == "baseline":
        net = ConditionalMeanFlowNetV2(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
        ).to(device)
        levels = 2  # unused
        lr = 6e-4
        use_flat_loss = False
    else:
        # Parse variant config
        levels = 1 if "level1" in variant else 2
        head_ch = 128 if "big-heads" in variant else 64
        lr = 3e-4 if "low-lr" in variant else 6e-4
        use_flat_loss = "flat-loss" in variant
        model_ch = 128

        net = OneFlowTSNet(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS, levels=levels,
            model_channels=model_ch, head_channels=head_ch,
            num_ctx_blocks=2, num_head_blocks=2,
            time_emb_dim=64, dropout=0.1,
        ).to(device)

    net_ema = deepcopy(net).eval()
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"[{variant}] Params: {params:,} | levels={levels if variant != 'baseline' else 'N/A'} | lr={lr}")

    sizes = get_level_sizes(PRED_LEN, levels) if variant != "baseline" else None
    prior = ResolutionMatchedPrior(sizes, length_scale=3.0).to(device) if sizes else None

    optimizer = AdamW(net.parameters(), lr=lr)
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

            if variant == "baseline":
                loss = conditional_meanflow_loss_v2(net, scaled_future, ctx_lags)
            elif use_flat_loss:
                loss = oneflow_loss_simple(net, scaled_future, ctx_lags, prior, levels=levels)
            else:
                loss = oneflow_loss(net, scaled_future, ctx_lags, prior, levels=levels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        avg_loss = epoch_loss / n_b
        elapsed = time.time() - t0

        if (epoch + 1) % 20 == 0:
            logger.info(f"[{variant}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            crps, nd = evaluate_model(
                net_ema, dataset, transformation, device, variant,
                levels=levels if variant != "baseline" else 2
            )
            if crps < best_crps:
                best_crps = crps
            logger.info(f"[{variant}] Epoch {epoch+1} → CRPS={crps:.6f} | ND={nd:.6f} | Best={best_crps:.6f}")

    return best_crps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["flat-loss", "level1", "big-heads", "flat-loss+big-heads", "baseline"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for variant in args.variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING VARIANT: {variant}")
        logger.info(f"{'='*60}")

        best_crps = run_variant(variant, args.epochs, device)
        results[variant] = best_crps

        logger.info(f"[{variant}] FINAL BEST CRPS: {best_crps:.6f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ITERATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for variant, crps in sorted_results:
        marker = " ← BEST" if variant == sorted_results[0][0] else ""
        logger.info(f"  {variant:<30} CRPS: {crps:.6f}{marker}")

    baseline_crps = results.get("baseline", None)
    if baseline_crps:
        logger.info(f"\n  Baseline CRPS: {baseline_crps:.6f}")
        for variant, crps in sorted_results:
            if variant != "baseline":
                ratio = crps / baseline_crps
                if ratio < 1.0:
                    logger.info(f"  {variant}: {(1-ratio)*100:.1f}% BETTER than baseline ✓")
                else:
                    logger.info(f"  {variant}: {(ratio-1)*100:.1f}% worse than baseline")


if __name__ == "__main__":
    main()
