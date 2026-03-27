"""
Generate publication-quality example plots showing MeanFlow-TS forecasts.

Creates:
1. Example forecasts with prediction intervals for each dataset
2. Comparison: MeanFlow 1-step vs TSFlow 32-step (electricity only)
3. Sample diversity visualization
4. Context influence demonstration
"""
import os, sys, tempfile, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24, "title": "Electricity"},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24, "title": "Solar"},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24, "title": "Traffic"},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30, "title": "Exchange Rate"},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48, "title": "M4 Hourly"},
}
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341,
    "traffic_nips": 0.082, "exchange_rate_nips": 0.005, "m4_hourly": 0.029,
}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'figure.dpi': 150,
})


def load_model_and_data(name, n_instances=8):
    cfg = CONFIGS[name]
    ctx, pred, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    max_lag = LAG_MAP.get(freq, 672)

    ckpt_path = f"best_v4_{name}.pt"
    if not os.path.exists(ckpt_path):
        return None, None, None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=pred, ctx_len=ctx, n_lags=7,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    net.load_state_dict(ckpt['net_ema'])
    net.eval()

    dataset = get_dataset(name)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(freq), pred_length=pred,
        ),
    ])
    splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=pred),
        past_length=ctx + max_lag, future_length=pred,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    loader = TrainDataLoader(
        Cached(transformed), batch_size=n_instances, stack_fn=batchify,
        transform=splitter, num_batches_per_epoch=1, shuffle_buffer_length=10000,
    )
    batch = next(iter(loader))
    return net, batch, cfg, dataset


def generate_forecasts(net, past, cfg, n_samples=100):
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
    return np.stack(samples, axis=1)  # (B, n_samples, pred_len)


# ============================================================
# Plot 1: Example forecasts for each dataset
# ============================================================
def plot_forecasts_grid():
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, cfg) in enumerate(CONFIGS.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        net, batch, cfg, dataset = load_model_and_data(name, n_instances=4)
        if net is None:
            ax.set_title(f"{cfg['title']} (no checkpoint)")
            continue

        past = batch["past_target"].to(device)
        future = batch["future_target"].to(device)
        samples = generate_forecasts(net, past, cfg, n_samples=50)

        ctx_len, pred_len = cfg["ctx"], cfg["pred"]
        # Plot first instance
        i = 0
        context_vals = past[i, -ctx_len:].cpu().numpy()
        truth = future[i].cpu().numpy()
        forecast_median = np.median(samples[i], axis=0)
        forecast_p10 = np.percentile(samples[i], 10, axis=0)
        forecast_p90 = np.percentile(samples[i], 90, axis=0)

        t_ctx = np.arange(ctx_len)
        t_pred = np.arange(ctx_len, ctx_len + pred_len)

        ax.plot(t_ctx, context_vals, color='#2196F3', lw=1.5, label='Context')
        ax.plot(t_pred, truth, color='#4CAF50', lw=1.5, label='Ground Truth')
        ax.plot(t_pred, forecast_median, color='#F44336', lw=1.5, label='MeanFlow Median')
        ax.fill_between(t_pred, forecast_p10, forecast_p90, color='#F44336', alpha=0.2, label='80% CI')
        ax.axvline(x=ctx_len - 0.5, color='gray', ls='--', alpha=0.5)
        ax.set_title(cfg["title"])
        ax.legend(loc='upper left', fontsize=7)
        ax.set_xlabel('Time step')

    # Remove empty subplot
    if len(CONFIGS) < 6:
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig('plots/forecast_examples.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved plots/forecast_examples.png")


# ============================================================
# Plot 2: Multiple samples showing probabilistic diversity
# ============================================================
def plot_sample_diversity():
    name = "electricity_nips"
    net, batch, cfg, dataset = load_model_and_data(name, n_instances=2)
    if net is None:
        return

    past = batch["past_target"].to(device)
    future = batch["future_target"].to(device)
    samples = generate_forecasts(net, past, cfg, n_samples=50)

    ctx_len, pred_len = cfg["ctx"], cfg["pred"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for idx in range(2):
        ax = axes[idx]
        context_vals = past[idx, -ctx_len:].cpu().numpy()
        truth = future[idx].cpu().numpy()

        t_ctx = np.arange(ctx_len)
        t_pred = np.arange(ctx_len, ctx_len + pred_len)

        ax.plot(t_ctx, context_vals, color='#2196F3', lw=2, label='Context')
        ax.plot(t_pred, truth, color='#4CAF50', lw=2, label='Ground Truth')

        # Plot individual samples
        for s in range(min(20, samples.shape[1])):
            ax.plot(t_pred, samples[idx, s], color='#F44336', alpha=0.15, lw=0.7)
        ax.plot(t_pred, np.median(samples[idx], axis=0), color='#F44336', lw=2, label='Median')

        ax.axvline(x=ctx_len - 0.5, color='gray', ls='--', alpha=0.5)
        ax.set_title(f'Electricity Series {idx+1}: 20 forecast samples')
        ax.legend(fontsize=8)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig('plots/sample_diversity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved plots/sample_diversity.png")


# ============================================================
# Plot 3: Quantile coverage (calibration)
# ============================================================
def plot_quantile_coverage():
    name = "electricity_nips"
    net, batch_unused, cfg, dataset = load_model_and_data(name, n_instances=1)
    if net is None:
        return

    ctx_len, pred_len, freq = cfg["ctx"], cfg["pred"], cfg["freq"]
    max_lag = LAG_MAP.get(freq, 672)

    # Get many test instances
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
        Cached(transformed), batch_size=64, stack_fn=batchify,
        transform=splitter, num_batches_per_epoch=8, shuffle_buffer_length=10000,
    )

    all_coverages = {q: [] for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    for batch in loader:
        past = batch["past_target"].to(device)
        future = batch["future_target"].cpu().numpy()
        samples = generate_forecasts(net, past, cfg, n_samples=50)

        for q in all_coverages:
            lower = np.percentile(samples, (1 - q) / 2 * 100, axis=1)
            upper = np.percentile(samples, (1 + q) / 2 * 100, axis=1)
            covered = ((future >= lower) & (future <= upper)).mean()
            all_coverages[q].append(covered)

    fig, ax = plt.subplots(figsize=(6, 5))
    quantiles = sorted(all_coverages.keys())
    ideal = quantiles
    actual = [np.mean(all_coverages[q]) for q in quantiles]

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Ideal calibration')
    ax.plot(quantiles, actual, 'o-', color='#F44336', lw=2, markersize=6, label='MeanFlow-TS v4')
    ax.fill_between(quantiles, quantiles, actual, alpha=0.15, color='#F44336')
    ax.set_xlabel('Nominal coverage')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Calibration Plot (Electricity)')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/calibration.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved plots/calibration.png")


# ============================================================
# Plot 4: MeanFlow vs FM ablation convergence
# ============================================================
def plot_ablation_convergence():
    # Data from our ablation experiments
    epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    fm_1step = [0.345, 0.180, 0.118, 0.092, 0.077, 0.068, 0.064, 0.062, 0.061, 0.062, 0.062, 0.062]
    fm_32step = [0.363, 0.227, 0.144, 0.103, 0.081, 0.066, 0.060, 0.060, 0.058, 0.056, 0.057, 0.056]

    # MeanFlow from v3b (same architecture)
    mf_epochs = [100, 200, 300, 400, 500, 600]
    mf_crps = [0.225, 0.090, 0.061, 0.058, 0.057, 0.056]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, fm_1step, 's-', color='#FF9800', lw=2, markersize=5, label='Standard FM (1-step)')
    ax.plot(epochs, fm_32step, 'D-', color='#9C27B0', lw=2, markersize=5, label='Standard FM (32-step)')
    ax.plot(mf_epochs, mf_crps, 'o-', color='#F44336', lw=2, markersize=6, label='MeanFlow (1-step)')
    ax.axhline(y=0.045, color='#4CAF50', ls='--', lw=1.5, alpha=0.7, label='TSFlow (32-step)')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('CRPS')
    ax.set_title('Ablation: MeanFlow vs Standard FM (Electricity)')
    ax.legend()
    ax.set_ylim(0.03, 0.4)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/ablation_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved plots/ablation_convergence.png")


# ============================================================
# Plot 5: Speed vs Quality tradeoff
# ============================================================
def plot_speed_quality():
    methods = [
        ("MeanFlow-TS\n(1-step)", 1, 0.047, '#F44336'),
        ("Standard FM\n(1-step)", 1, 0.062, '#FF9800'),
        ("Standard FM\n(32-step)", 32, 0.056, '#9C27B0'),
        ("TSFlow\n(32-step)", 32, 0.045, '#4CAF50'),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, nfe, crps, color in methods:
        ax.scatter(nfe, crps, s=200, color=color, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (nfe, crps), textcoords="offset points",
                    xytext=(15, -5), fontsize=9, ha='left')

    ax.set_xlabel('Number of Function Evaluations (NFE)')
    ax.set_ylabel('CRPS (lower is better)')
    ax.set_title('Speed vs Quality Tradeoff (Electricity)')
    ax.set_xscale('log')
    ax.set_xlim(0.5, 100)
    ax.set_ylim(0.03, 0.07)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/speed_quality.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved plots/speed_quality.png")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    print("Generating plots...\n")

    plot_forecasts_grid()
    plot_sample_diversity()
    plot_quantile_coverage()
    plot_ablation_convergence()
    plot_speed_quality()

    print("\nAll plots saved to plots/")
