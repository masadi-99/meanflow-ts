"""
Detailed per-dataset plots with multiple examples per dataset.
"""
import os, sys, tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.transform import *

try:
    os.environ["TSFLOW_NO_KEOPS"] = "1"
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24, "title": "Electricity (Hourly)"},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24, "title": "Solar Energy (Hourly)"},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24, "title": "Traffic (Hourly)"},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30, "title": "Exchange Rate (Business Daily)"},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48, "title": "M4 Hourly"},
}
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341,
    "traffic_nips": 0.082, "exchange_rate_nips": 0.005, "m4_hourly": 0.029,
}
OUR_CRPS = {
    "electricity_nips": 0.047, "solar_nips": 0.423,
    "traffic_nips": 0.086, "exchange_rate_nips": 0.010, "m4_hourly": 0.032,
}

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


def load_model(name, cfg):
    ctx, pred = cfg["ctx"], cfg["pred"]
    ckpt_path = f"best_v4_{name}.pt"
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=pred, ctx_len=ctx, n_lags=7,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net.load_state_dict(ckpt['net_ema'])
    net.eval()
    return net


def get_batch(name, cfg, n=8):
    ctx, pred, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    max_lag = LAG_MAP.get(freq, 672)
    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred),
    ])
    splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred),
        past_length=ctx + max_lag, future_length=pred,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(Cached(transformed), batch_size=n, stack_fn=batchify,
        transform=splitter, num_batches_per_epoch=1, shuffle_buffer_length=10000)
    return next(iter(loader))


def forecast(net, past, cfg, n_samples=50):
    ctx_len, pred_len, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    ctx = past[:, -ctx_len:]
    loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
    ctx_with_lags = extract_lag_features(past, ctx_len, freq, 7).to(device) / loc.unsqueeze(1)
    B = ctx.shape[0]
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(B, pred_len, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            pred = (z - net(z, (t, h), ctx_with_lags)) * loc
            samples.append(pred.cpu().numpy())
    return np.stack(samples, axis=1)


def plot_dataset_detail(name, cfg):
    net = load_model(name, cfg)
    if net is None:
        print(f"  SKIP {name}: no checkpoint")
        return

    batch = get_batch(name, cfg, n=6)
    past = batch["past_target"].to(device)
    future = batch["future_target"].to(device)
    samples = forecast(net, past, cfg, n_samples=50)

    ctx_len, pred_len = cfg["ctx"], cfg["pred"]
    n_examples = min(6, past.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'{cfg["title"]}  —  MeanFlow-TS v4 (1-step)  |  CRPS: {OUR_CRPS.get(name, "?")} (TSFlow: {TSFLOW_CRPS.get(name, "?")})',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i in range(n_examples):
        ax = axes[i]
        ctx_vals = past[i, -ctx_len:].cpu().numpy()
        truth = future[i].cpu().numpy()
        median = np.median(samples[i], axis=0)
        p5 = np.percentile(samples[i], 5, axis=0)
        p25 = np.percentile(samples[i], 25, axis=0)
        p75 = np.percentile(samples[i], 75, axis=0)
        p95 = np.percentile(samples[i], 95, axis=0)

        t_ctx = np.arange(ctx_len)
        t_pred = np.arange(ctx_len, ctx_len + pred_len)

        # Context
        ax.plot(t_ctx, ctx_vals, color='#1565C0', lw=1.8, label='Context')
        # Ground truth
        ax.plot(t_pred, truth, color='#2E7D32', lw=1.8, label='Ground Truth')
        # Prediction intervals
        ax.fill_between(t_pred, p5, p95, color='#E53935', alpha=0.1, label='90% CI')
        ax.fill_between(t_pred, p25, p75, color='#E53935', alpha=0.2, label='50% CI')
        # Median
        ax.plot(t_pred, median, color='#E53935', lw=1.8, label='Median')
        # 5 individual samples
        for s in range(5):
            ax.plot(t_pred, samples[i, s], color='#E53935', alpha=0.12, lw=0.6)

        ax.axvline(x=ctx_len - 0.5, color='gray', ls='--', alpha=0.4, lw=0.8)
        ax.set_title(f'Series {i+1}', fontsize=10)
        if i >= 3:
            ax.set_xlabel('Time step')
        if i % 3 == 0:
            ax.set_ylabel('Value')
        if i == 0:
            ax.legend(fontsize=7, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'plots/detail_{name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots/detail_{name}.png")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    for name, cfg in CONFIGS.items():
        print(f"Plotting {name}...")
        plot_dataset_detail(name, cfg)
    print("\nDone!")
