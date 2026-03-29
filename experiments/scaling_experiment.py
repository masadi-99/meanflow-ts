"""
Inference-time scaling experiment: How does forecast quality improve with more samples?

Hypothesis: MeanFlow's 1-step generation enables "inference-time compute scaling" —
generating more samples improves quality, and MeanFlow can afford orders of magnitude
more samples than multi-step methods in the same wall-clock time.

This tests a regime where MeanFlow has a QUALITATIVE advantage, not just speed:
- At 100 samples: MeanFlow ≈ TSFlow (our Table 3 results)
- At 1000+ samples: MeanFlow should improve further while TSFlow can't afford it
- Tail quantile estimation (1%, 99%) requires many samples for accuracy

Metrics:
- CRPS vs num_samples (does it keep improving?)
- Quantile calibration at extreme percentiles (1%, 5%, 95%, 99%)
- Wall-clock time vs quality Pareto frontier
- Tail risk: Value-at-Risk (VaR) and Expected Shortfall (ES) accuracy
"""
import os, sys, time, tempfile, json
import numpy as np
import torch

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import *
from tqdm.auto import tqdm

try:
    os.environ["TSFLOW_NO_KEOPS"] = "1"
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'TSFlow'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, MeanFlowForecasterV2, extract_lag_features

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "electricity_nips": {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":     {"freq": "H", "ctx": 24, "pred": 24},
}
LAG_MAP = {"H": 672, "B": 750}


def evaluate_at_n_samples(net, dataset, cfg, num_samples, device):
    """Evaluate CRPS and timing at a given sample count."""
    ctx_len, pred_len, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    max_lag = LAG_MAP.get(freq, 672)

    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred_len),
    ])
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=ctx_len + max_lag, future_length=pred_len,
        time_series_fields=["time_feat", "observed_values"],
    )

    forecaster = MeanFlowForecasterV2(
        net, ctx_len, pred_len, num_samples=num_samples, freq=freq, n_lags=7,
    ).to(device)
    predictor = PyTorchPredictor(
        prediction_length=pred_len,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=256,
        input_transform=test_splitter, device=device,
    )

    t0 = time.time()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    wall_time = time.time() - t0

    metrics, per_ts = Evaluator(num_workers=0)(tss, forecasts)
    crps = metrics["mean_wQuantileLoss"]
    nd = metrics["ND"]

    # Compute tail calibration from per-ts forecasts
    coverages = {}
    for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]:
        covered = 0
        total = 0
        for ts_series, fc in zip(tss, forecasts):
            truth = ts_series.values[-pred_len:]
            quantile_val = np.percentile(fc.samples, q * 100, axis=0)
            if q <= 0.5:
                covered += (truth >= quantile_val).mean()
            else:
                covered += (truth <= quantile_val).mean()
            total += 1
        coverages[f"coverage_{q}"] = covered / total if total > 0 else 0

    return {
        "num_samples": num_samples,
        "crps": float(crps),
        "nd": float(nd),
        "wall_time_s": wall_time,
        "time_per_forecast_ms": wall_time / len(forecasts) * 1000,
        **{k: float(v) for k, v in coverages.items()},
    }


def evaluate_tsflow_at_n_samples(dataset, num_samples, device):
    """Evaluate TSFlow at given sample count."""
    from tsflow.model import TSFlowCond
    from tsflow.dataset import get_gts_dataset
    from tsflow.utils import create_transforms
    from tsflow.utils.util import create_splitter
    from tsflow.utils.variables import get_season_length

    ckpt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'TSFlow',
                             'logs/tsflow/20260325_213559/best_checkpoint.ckpt')
    if not os.path.exists(ckpt_path):
        return None

    model = TSFlowCond(
        setting="univariate", target_dim=1, context_length=24, prediction_length=24,
        backbone_params=dict(input_dim=1, output_dim=1, step_emb=64, num_residual_blocks=3,
            residual_block="s4", hidden_dim=64, dropout=0.0, init_skip=False, feature_skip=True),
        prior_params=dict(kernel="ou", gamma=1, context_freqs=14),
        optimizer_params=dict(lr=1e-3),
        ema_params=dict(beta=0.9999, update_after_step=128, update_every=1),
        frequency="H", normalization="longmean",
        use_lags=True, use_ema=True, num_steps=32, solver="euler", matching="random",
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=True)
    model.eval()
    model.num_samples = num_samples

    dataset_gts = get_gts_dataset("electricity_nips")
    time_features = time_features_from_frequency_str("H")
    transformation = create_transforms(time_features=time_features, prediction_length=24,
        freq=get_season_length("H"), train_length=len(dataset_gts.train))
    _ = list(transformation.apply(dataset_gts.train, is_train=True))
    test_data = transformation.apply(dataset_gts.test, is_train=False)
    test_splitter = create_splitter(
        past_length=max(24 + max(model.lags_seq), model.prior_context_length),
        future_length=24, mode="test")

    batch_size = max(1, 1024 * 64 // num_samples)
    predictor = model.get_predictor(test_splitter, batch_size=batch_size, device=device)

    t0 = time.time()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data, predictor=predictor, num_samples=num_samples)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    wall_time = time.time() - t0

    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    return {
        "num_samples": num_samples,
        "crps": float(metrics["mean_wQuantileLoss"]),
        "nd": float(metrics["ND"]),
        "wall_time_s": wall_time,
        "time_per_forecast_ms": wall_time / len(forecasts) * 1000,
    }


def main():
    # Load MeanFlow v4
    ckpt = torch.load("best_v4_electricity_nips.pt", map_location=device, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=24, ctx_len=24, n_lags=7, model_channels=128, num_res_blocks=4,
    ).to(device)
    net.load_state_dict(ckpt["net_ema"])
    net.eval()

    dataset = get_dataset("electricity_nips")

    # Test MeanFlow at various sample counts
    sample_counts = [10, 25, 50, 100, 250, 500, 1000, 2500]
    mf_results = []

    logger.info("=== MeanFlow Inference Scaling ===")
    for n in sample_counts:
        logger.info(f"MeanFlow n={n}...")
        r = evaluate_at_n_samples(net, dataset, CONFIGS["electricity_nips"], n, device)
        mf_results.append(r)
        logger.info(f"  CRPS={r['crps']:.6f} | time={r['wall_time_s']:.1f}s | "
                     f"per_fc={r['time_per_forecast_ms']:.1f}ms")

    # Test TSFlow at various sample counts (limited by speed)
    tsflow_counts = [10, 25, 50, 100, 250]
    ts_results = []

    logger.info("\n=== TSFlow Inference Scaling ===")
    for n in tsflow_counts:
        logger.info(f"TSFlow n={n}...")
        r = evaluate_tsflow_at_n_samples(dataset, n, device)
        if r:
            ts_results.append(r)
            logger.info(f"  CRPS={r['crps']:.6f} | time={r['wall_time_s']:.1f}s | "
                         f"per_fc={r['time_per_forecast_ms']:.1f}ms")

    # Print comparison
    logger.info(f"\n{'='*80}")
    logger.info(f"INFERENCE SCALING COMPARISON (electricity_nips)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':<20} | {'Samples':>7} | {'CRPS':>8} | {'Time(s)':>8} | {'ms/fc':>8}")
    logger.info(f"{'-'*60}")
    for r in mf_results:
        logger.info(f"{'MeanFlow':<20} | {r['num_samples']:>7} | {r['crps']:>8.6f} | {r['wall_time_s']:>8.1f} | {r['time_per_forecast_ms']:>8.1f}")
    logger.info(f"{'-'*60}")
    for r in ts_results:
        logger.info(f"{'TSFlow':<20} | {r['num_samples']:>7} | {r['crps']:>8.6f} | {r['wall_time_s']:>8.1f} | {r['time_per_forecast_ms']:>8.1f}")

    # Tail calibration (MeanFlow only at high sample counts)
    logger.info(f"\n{'='*80}")
    logger.info(f"TAIL CALIBRATION (MeanFlow, electricity_nips)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Samples':>7} | {'Cov 1%':>8} | {'Cov 5%':>8} | {'Cov 50%':>8} | {'Cov 95%':>8} | {'Cov 99%':>8}")
    logger.info(f"{'-'*60}")
    for r in mf_results:
        logger.info(f"{r['num_samples']:>7} | {r.get('coverage_0.01',0):>8.4f} | {r.get('coverage_0.05',0):>8.4f} | "
                     f"{r.get('coverage_0.5',0):>8.4f} | {r.get('coverage_0.95',0):>8.4f} | {r.get('coverage_0.99',0):>8.4f}")

    # Save
    with open("scaling_results.json", "w") as f:
        json.dump({"meanflow": mf_results, "tsflow": ts_results}, f, indent=2)
    logger.info("\nSaved to scaling_results.json")


if __name__ == "__main__":
    main()
