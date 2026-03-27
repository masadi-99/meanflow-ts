"""
Compute Table 1 (2-Wasserstein) and Table 2 (LPS/CRPS) for MeanFlow-TS v4.

Matches TSFlow's evaluation protocol:
- Table 1: Exact EMD W2 via pot.emd2, on model-internal normalized scale
- Table 2: Train linear model on synthetic samples, evaluate CRPS on real test data
           using GluonTS Evaluator with proper scaling/descaling

Both tables require unconditional generation of full-length time series windows.
For our CONDITIONAL model, we generate by conditioning on real context and predicting future.
This is different from TSFlow's unconditional model — we note this in the comparison.
"""
import os, sys, argparse, tempfile, json, logging, math, time
import numpy as np
import torch
import ot as pot
from copy import deepcopy
from tqdm.auto import tqdm
from sklearn.linear_model import Ridge

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast import SampleForecast
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
from meanflow_ts.model_v2 import (
    ConditionalMeanFlowNetV2, MeanFlowForecasterV2, extract_lag_features,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
}
LAG_MAP = {"H": 672, "B": 750}

# TSFlow paper results
TSFLOW_W2 = {
    "electricity_nips": 2.090, "exchange_rate_nips": 0.029,
    "solar_nips": 4.564, "traffic_nips": 7.283, "m4_hourly": 6.509,
}
TSFLOW_LPS = {
    "electricity_nips": 0.096, "exchange_rate_nips": 0.011,
    "solar_nips": 0.616, "traffic_nips": 0.237, "m4_hourly": 0.032,
}


def exact_w2(x0, x1, max_n=2000):
    """Exact 2-Wasserstein via EMD (matching TSFlow)."""
    if len(x0) > max_n:
        x0 = x0[np.random.choice(len(x0), max_n, replace=False)]
    if len(x1) > max_n:
        x1 = x1[np.random.choice(len(x1), max_n, replace=False)]
    x0t = torch.tensor(x0, dtype=torch.float32)
    x1t = torch.tensor(x1, dtype=torch.float32)
    if x0t.dim() > 2:
        x0t = x0t.reshape(x0t.shape[0], -1)
    if x1t.dim() > 2:
        x1t = x1t.reshape(x1t.shape[0], -1)
    M = torch.cdist(x0t, x1t) ** 2
    a, b = pot.unif(len(x0)), pot.unif(len(x1))
    ret = pot.emd2(a, b, M.numpy(), numItermax=int(1e7))
    return math.sqrt(ret)


def generate_conditional_windows(net, dataset, cfg, device, n_windows=5000, n_lags=7):
    """
    Generate synthetic [context + prediction] windows using our conditional model.
    For each real context, generate a prediction and concatenate.
    Returns windows in NORMALIZED scale (divided by per-window mean).
    """
    ctx_len, pred_len, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    max_lag = LAG_MAP.get(freq, 672)

    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len,
        ),
    ])
    splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred_len),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(
        Cached(transformed), batch_size=256, stack_fn=batchify,
        transform=splitter, num_batches_per_epoch=max(1, n_windows // 256),
        shuffle_buffer_length=10000,
    )

    gen_windows_normalized = []
    real_windows_normalized = []

    with torch.no_grad():
        for batch in loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -ctx_len:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)

            # Real window (normalized)
            real_full = torch.cat([ctx, future], dim=1) / loc
            real_windows_normalized.append(real_full.cpu().numpy())

            # Generate prediction
            scaled_ctx = ctx / loc
            ctx_with_lags = extract_lag_features(past, ctx_len, freq, n_lags) / loc.unsqueeze(1)

            B = ctx.shape[0]
            z_1 = torch.randn(B, pred_len, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            gen_pred = z_1 - net(z_1, (t, h), ctx_with_lags)

            gen_full = torch.cat([scaled_ctx, gen_pred], dim=1)
            gen_windows_normalized.append(gen_full.cpu().numpy())

            if len(gen_windows_normalized) * 256 >= n_windows:
                break

    gen = np.concatenate(gen_windows_normalized)[:n_windows]
    real = np.concatenate(real_windows_normalized)[:n_windows]
    return gen, real


def compute_lps_crps(net, dataset, cfg, device, n_lags=7):
    """
    Table 2: Train Ridge on synthetic, evaluate CRPS on real test.

    Protocol (matching TSFlow):
    1. Generate synthetic [context+future] windows in normalized scale
    2. Train Ridge regression: context → future
    3. For each real test instance: normalize context, predict, denormalize, compute CRPS
    """
    ctx_len, pred_len, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    max_lag = LAG_MAP.get(freq, 672)

    # Step 1: Generate synthetic training data
    gen_windows, _ = generate_conditional_windows(
        net, dataset, cfg, device, n_windows=5000, n_lags=n_lags
    )

    # Step 2: Train Ridge on synthetic
    X_syn = gen_windows[:, :ctx_len]
    y_syn = gen_windows[:, ctx_len:ctx_len + pred_len]
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_syn, y_syn)

    # Step 3: Evaluate on real test data with proper scaling
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len,
        ),
    ])
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    test_transform = transformation.apply(dataset.test, is_train=False)

    # Build forecasts manually with proper scaling
    import pandas as pd
    from gluonts.dataset.split import split

    forecasts = []
    tss = []

    for entry in tqdm(test_transform, desc="LPS eval"):
        ts = entry["target"]
        if len(ts) < ctx_len + pred_len:
            continue

        # Get context and truth
        context = ts[-(ctx_len + pred_len):-pred_len]
        truth = ts[-pred_len:]

        # Normalize by context mean (same as training)
        loc = max(np.abs(context).mean(), 0.01)
        ctx_scaled = context / loc

        # Predict (Ridge gives point forecast in normalized space)
        pred_scaled = ridge.predict(ctx_scaled.reshape(1, -1)).flatten()

        # Denormalize
        pred_raw = pred_scaled * loc

        # Create SampleForecast (point forecast repeated as 1 sample)
        start = pd.Period("2023-01-01", freq=freq)
        forecast = SampleForecast(
            samples=pred_raw.reshape(1, -1).astype(np.float32),
            start_date=start,
        )
        forecasts.append(forecast)

        # Create truth series
        full_ts = ts[-(ctx_len + pred_len):]
        ts_series = pd.Series(
            full_ts,
            index=pd.period_range(start=start, periods=len(full_ts), freq=freq),
        )
        tss.append(ts_series)

    if not forecasts:
        return {"CRPS": float('inf')}

    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--n-lags", type=int, default=7)
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    ctx_len, pred_len, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    n_lags = args.n_lags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"best_v4_{name}.pt"
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"=== Table 1 & 2: {name} ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=pred_len, ctx_len=ctx_len, n_lags=n_lags,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net.load_state_dict(ckpt['net_ema'])
    net.eval()
    logger.info(f"Loaded from epoch {ckpt.get('epoch', '?')}, CRPS={ckpt.get('crps', '?')}")

    dataset = get_dataset(name)

    # === Table 1: W2 distance ===
    logger.info("Computing W2 distance...")
    gen_norm, real_norm = generate_conditional_windows(
        net, dataset, cfg, device, n_windows=2000, n_lags=n_lags
    )
    w2 = exact_w2(real_norm, gen_norm)
    tsflow_w2 = TSFLOW_W2.get(name, "?")
    logger.info(f"  W2 (normalized): {w2:.4f} (TSFlow: {tsflow_w2})")

    # === Table 2: LPS ===
    logger.info("Computing LPS (CRPS of linear model)...")
    lps_metrics = compute_lps_crps(net, dataset, cfg, device, n_lags=n_lags)
    lps_crps = lps_metrics.get("CRPS", float('inf'))
    lps_nd = lps_metrics.get("ND", float('inf'))
    tsflow_lps = TSFLOW_LPS.get(name, "?")
    logger.info(f"  LPS CRPS: {lps_crps:.6f} (TSFlow: {tsflow_lps})")
    logger.info(f"  LPS ND: {lps_nd:.6f}")

    # Save
    results = {
        "dataset": name, "w2": float(w2), "lps_crps": float(lps_crps),
        "lps_nd": float(lps_nd),
        "tsflow_w2": tsflow_w2, "tsflow_lps": tsflow_lps,
    }
    with open(f"table12_v4_{name}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print(f"{'='*60}")
    print(f"{'Metric':<30} | {'MeanFlow v4':>12} | {'TSFlow (OU)':>12}")
    print(f"{'-'*60}")
    print(f"{'W2 distance (Table 1)':<30} | {w2:>12.4f} | {str(tsflow_w2):>12}")
    print(f"{'LPS CRPS (Table 2)':<30} | {lps_crps:>12.6f} | {str(tsflow_lps):>12}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
