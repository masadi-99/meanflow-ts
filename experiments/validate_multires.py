"""
DAY 1-2 GO/NO-GO VALIDATION: Multi-resolution vs single-resolution one-step generation.

Compares three variants on electricity_nips (200 epochs, fast evaluation):
  (a) single-res: Original MeanFlow-TS v4 (baseline)
  (b) multi-res-iso: OneFlow-TS with N(0,I) prior at all levels
  (c) multi-res-matched: OneFlow-TS with resolution-matched priors

Usage:
  python validate_multires.py --variant single-res
  python validate_multires.py --variant multi-res-iso
  python validate_multires.py --variant multi-res-matched

Run all three in PARALLEL on different GPUs for fastest results:
  CUDA_VISIBLE_DEVICES=0 python validate_multires.py --variant single-res &
  CUDA_VISIBLE_DEVICES=1 python validate_multires.py --variant multi-res-iso &
  CUDA_VISIBLE_DEVICES=2 python validate_multires.py --variant multi-res-matched &

GO/NO-GO criteria:
  multi-res CRPS < single-res CRPS → GO
  multi-res CRPS ≈ single-res CRPS (within 5%) → CONDITIONAL GO
  multi-res CRPS > single-res CRPS by >10% → NO-GO (investigate)
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

# Import OneFlow-TS components
from oneflow_ts.model import OneFlowTSNet, OneFlowForecaster
from oneflow_ts.priors import ResolutionMatchedPrior, IsotropicPrior
from oneflow_ts.loss import oneflow_loss, oneflow_loss_simple
from oneflow_ts.wavelet import get_level_sizes

# Import original MeanFlow-TS for baseline
from meanflow_ts.model_v2 import (
    ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2,
    MeanFlowForecasterV2, extract_lag_features,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset config
DATASET = "electricity_nips"
FREQ = "H"
CTX_LEN = 24
PRED_LEN = 24
MAX_LAG = 672
N_LAGS = 7
LEVELS = 2

# Training config (reduced for validation speed)
EPOCHS = 200
BATCH_SIZE = 64
BATCHES_PER_EPOCH = 128
LR = 6e-4
EMA_DECAY = 0.9999
GRAD_CLIP = 1.0
EVAL_EVERY = 50
NUM_EVAL_SAMPLES = 16


def get_data_pipeline():
    """Setup GluonTS data pipeline (shared across all variants)."""
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
        Cached(transformed_data), batch_size=BATCH_SIZE, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=BATCHES_PER_EPOCH,
        shuffle_buffer_length=10000,
    )
    return dataset, transformation, train_loader


def evaluate(net_ema, dataset, transformation, variant, device):
    """Evaluate using GluonTS."""
    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )

    if variant == "single-res":
        forecaster = MeanFlowForecasterV2(
            net_ema, CTX_LEN, PRED_LEN, num_samples=NUM_EVAL_SAMPLES,
            freq=FREQ, n_lags=N_LAGS,
        ).to(device)
    else:
        # For multi-res variants, create appropriate prior
        sizes = get_level_sizes(PRED_LEN, LEVELS)
        if variant == "multi-res-matched":
            prior = ResolutionMatchedPrior(sizes, length_scale=3.0).to(device)
        else:
            prior = IsotropicPrior(sizes).to(device)
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
    return metrics


def train_single_res(device):
    """Train original MeanFlow-TS v4 (baseline)."""
    logger.info("=== VARIANT: single-res (MeanFlow-TS v4 baseline) ===")
    dataset, transformation, train_loader = get_data_pipeline()

    net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net_ema = deepcopy(net).eval()
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"Params: {params:,}")

    optimizer = AdamW(net.parameters(), lr=LR)

    for epoch in range(EPOCHS):
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

            loss = conditional_meanflow_loss_v2(net, scaled_future, ctx_lags)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1 - EMA_DECAY)
            epoch_loss += loss.item()
            n_b += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/n_b:.4f} | {time.time()-t0:.1f}s")

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            metrics = evaluate(net_ema, dataset, transformation, "single-res", device)
            crps = metrics["mean_wQuantileLoss"]
            logger.info(f"  → CRPS={crps:.6f} | ND={metrics['ND']:.6f}")

    return metrics


def train_multi_res(variant, device):
    """Train OneFlow-TS (multi-resolution)."""
    logger.info(f"=== VARIANT: {variant} ===")
    dataset, transformation, train_loader = get_data_pipeline()

    # Model
    net = OneFlowTSNet(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS, levels=LEVELS,
        model_channels=128, head_channels=64,
        num_ctx_blocks=2, num_head_blocks=2,
        time_emb_dim=64, dropout=0.1,
    ).to(device)
    net_ema = deepcopy(net).eval()
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"Params: {params:,}")

    # Prior
    sizes = get_level_sizes(PRED_LEN, LEVELS)
    if variant == "multi-res-matched":
        prior = ResolutionMatchedPrior(sizes, length_scale=3.0).to(device)
        logger.info("Using resolution-matched priors")
    else:
        prior = IsotropicPrior(sizes).to(device)
        logger.info("Using isotropic (N(0,I)) priors at all levels")

    optimizer = AdamW(net.parameters(), lr=LR)

    for epoch in range(EPOCHS):
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

            loss = oneflow_loss(net, scaled_future, ctx_lags, prior, levels=LEVELS)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1 - EMA_DECAY)
            epoch_loss += loss.item()
            n_b += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/n_b:.4f} | {time.time()-t0:.1f}s")

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            metrics = evaluate(net_ema, dataset, transformation, variant, device)
            crps = metrics["mean_wQuantileLoss"]
            logger.info(f"  → CRPS={crps:.6f} | ND={metrics['ND']:.6f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="OneFlow-TS GO/NO-GO validation")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["single-res", "multi-res-iso", "multi-res-matched", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6432)
    np.random.seed(6432)

    if args.variant == "all":
        results = {}
        for v in ["single-res", "multi-res-iso", "multi-res-matched"]:
            torch.manual_seed(6432)
            np.random.seed(6432)
            if v == "single-res":
                metrics = train_single_res(device)
            else:
                metrics = train_multi_res(v, device)
            results[v] = metrics["mean_wQuantileLoss"]

        logger.info("\n" + "=" * 60)
        logger.info("GO/NO-GO RESULTS")
        logger.info("=" * 60)
        for v, crps in results.items():
            logger.info(f"  {v:<25} CRPS: {crps:.6f}")

        sr = results["single-res"]
        for v in ["multi-res-iso", "multi-res-matched"]:
            ratio = results[v] / sr
            if ratio < 0.95:
                logger.info(f"  {v}: {(1-ratio)*100:.1f}% BETTER → GO ✓")
            elif ratio < 1.05:
                logger.info(f"  {v}: within 5% → CONDITIONAL GO ≈")
            else:
                logger.info(f"  {v}: {(ratio-1)*100:.1f}% WORSE → NO-GO ✗")
    else:
        if args.variant == "single-res":
            train_single_res(device)
        else:
            train_multi_res(args.variant, device)


if __name__ == "__main__":
    main()
