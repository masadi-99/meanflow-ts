"""
Controllability Evaluation: Can the model generate futures that match specified statistics?

This is NOT about improving CRPS. It's about a NEW CAPABILITY:
the model can generate diverse, realistic futures that satisfy user-specified constraints.

Tests:
1. Control accuracy: request specific stats, measure how well model hits them
2. Controllability range: sweep stat values, measure response
3. Realism: are controlled samples realistic? (compare to unconditional distribution)
4. Diversity under constraint: multiple samples with same stats should still vary

Uses the "stats" or "both" mode v3 checkpoint.
"""
import os, sys, torch, numpy as np, tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v3 import ConditionalMeanFlowNetV3
from meanflow_ts.model_v2 import extract_lag_features
from meanflow_ts.utils import extract_stats, N_STATS
from gluonts.dataset.repository.datasets import get_dataset

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

torch.manual_seed(6432)
np.random.seed(6432)

DATASET = os.environ.get("DATASET", "electricity_nips")
CONFIGS = {
    "electricity_nips": {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":     {"freq": "H", "ctx": 24, "pred": 24},
    "m4_hourly":        {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS.get(DATASET, {"freq": "H", "ctx": 24, "pred": 24})
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 50


def load_model():
    for mode in ["stats", "both"]:
        ckpt_path = f"best_v3_{mode}_{DATASET}.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            net = ConditionalMeanFlowNetV3(
                pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
                model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
            ).to(DEVICE).eval()
            net.load_state_dict(ckpt['net_ema'])
            print(f"Loaded {ckpt_path}")
            return net
    return None


@torch.no_grad()
def generate(net, n, ctx_lags, stat_vec=None):
    ctx_batch = ctx_lags.expand(n, -1, -1)
    z_1 = torch.randn(n, PRED_LEN, device=DEVICE)
    B = n
    t = torch.ones(B, device=DEVICE)
    h = torch.ones(B, device=DEVICE)
    stat_batch = stat_vec.expand(n, -1) if stat_vec is not None else None
    u = net(z_1, (t, h), ctx_batch, None, stat_batch)
    return z_1 - u


def main():
    net = load_model()
    if net is None:
        print(f"No stats/both checkpoint for {DATASET}")
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    stat_names = ["mean", "max", "min", "std", "argmax", "auc"]

    # Collect test windows
    windows = []
    for entry in test_data[:200]:
        target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
        if len(target) < CTX_LEN + max_lag + PRED_LEN:
            continue
        gt = target[-PRED_LEN:]
        past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
        ctx = past[-CTX_LEN:]
        loc = ctx.abs().mean().clamp(min=0.01)
        ctx_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc
        scaled_gt = gt / loc
        windows.append((ctx_lags, loc, scaled_gt, gt))

    print(f"Collected {len(windows)} test windows")

    # === Test 1: Control Accuracy ===
    # Request oracle stats, measure how well model hits them
    print(f"\n{'='*60}")
    print(f"TEST 1: CONTROL ACCURACY — {DATASET}")
    print(f"{'='*60}")

    stat_errors_cond = {s: [] for s in stat_names}
    stat_errors_uncond = {s: [] for s in stat_names}

    for ctx_lags, loc, scaled_gt, gt in windows:
        oracle_stats = extract_stats(scaled_gt.unsqueeze(0))

        # Conditioned generation
        preds_cond = generate(net, N_SAMPLES, ctx_lags, oracle_stats)
        gen_stats_cond = extract_stats(preds_cond)

        # Unconditional generation
        preds_uncond = generate(net, N_SAMPLES, ctx_lags, None)
        gen_stats_uncond = extract_stats(preds_uncond)

        for i, sn in enumerate(stat_names):
            err_c = (gen_stats_cond[:, i] - oracle_stats[0, i]).abs().mean().item()
            err_u = (gen_stats_uncond[:, i] - oracle_stats[0, i]).abs().mean().item()
            stat_errors_cond[sn].append(err_c)
            stat_errors_uncond[sn].append(err_u)

    print(f"\n  {'Stat':<10} | {'Uncond MAE':>12} | {'Cond MAE':>12} | {'Improvement':>12}")
    print(f"  {'-'*55}")
    for sn in stat_names:
        eu = np.mean(stat_errors_uncond[sn])
        ec = np.mean(stat_errors_cond[sn])
        imp = (eu - ec) / eu * 100 if eu > 0 else 0
        print(f"  {sn:<10} | {eu:>12.4f} | {ec:>12.4f} | {imp:>+11.1f}%")

    # === Test 2: Controllability Range — Sweep mean stat ===
    print(f"\n{'='*60}")
    print(f"TEST 2: CONTROLLABILITY RANGE — {DATASET}")
    print(f"{'='*60}")

    # Use first window, sweep the mean stat
    ctx_lags, loc, scaled_gt, gt = windows[0]
    oracle_stats = extract_stats(scaled_gt.unsqueeze(0))
    base_mean = oracle_stats[0, 0].item()

    print(f"\n  Sweeping mean around oracle ({base_mean:.2f}):")
    print(f"  {'Requested mean':>15} | {'Achieved mean':>15} | {'Achieved std':>12} | {'Diversity':>10}")
    print(f"  {'-'*60}")

    for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        target_mean = base_mean * mult
        modified_stats = oracle_stats.clone()
        modified_stats[0, 0] = target_mean

        preds = generate(net, N_SAMPLES, ctx_lags, modified_stats) * loc
        achieved_mean = preds.mean(dim=1).mean().item()
        achieved_std = preds.mean(dim=1).std().item()
        diversity = preds.std(dim=0).mean().item()

        print(f"  {target_mean * loc.item():>15.2f} | {achieved_mean:>15.2f} | {achieved_std:>12.2f} | {diversity:>10.2f}")

    # === Test 3: Diversity under constraint ===
    print(f"\n{'='*60}")
    print(f"TEST 3: DIVERSITY UNDER CONSTRAINT — {DATASET}")
    print(f"{'='*60}")

    divs_uncond = []
    divs_cond = []
    for ctx_lags, loc, scaled_gt, gt in windows[:50]:
        oracle_stats = extract_stats(scaled_gt.unsqueeze(0))

        preds_u = generate(net, N_SAMPLES, ctx_lags, None)
        preds_c = generate(net, N_SAMPLES, ctx_lags, oracle_stats)

        divs_uncond.append(preds_u.std(dim=0).mean().item())
        divs_cond.append(preds_c.std(dim=0).mean().item())

    print(f"  Unconditional diversity: {np.mean(divs_uncond):.4f}")
    print(f"  Stat-conditioned diversity: {np.mean(divs_cond):.4f}")
    ratio = np.mean(divs_cond) / np.mean(divs_uncond) if np.mean(divs_uncond) > 0 else 0
    print(f"  Ratio: {ratio:.2f} (want 0.3-0.8: focused but still diverse)")

    # === Plot: controllability visualization ===
    os.makedirs("plots", exist_ok=True)
    ctx_lags, loc, scaled_gt, gt = windows[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_idx, (mult, label) in enumerate([(0.5, "Low mean"), (1.0, "Oracle mean"), (1.5, "High mean")]):
        ax = axes[ax_idx]
        oracle_stats = extract_stats(scaled_gt.unsqueeze(0))
        oracle_stats[0, 0] *= mult

        preds = generate(net, 20, ctx_lags, oracle_stats) * loc
        ctx_vals = (ctx_lags[0, 0, :] * loc).cpu().numpy()

        t_ctx = np.arange(CTX_LEN)
        t_pred = np.arange(CTX_LEN, CTX_LEN + PRED_LEN)

        ax.plot(t_ctx, ctx_vals, 'b-', lw=2, label='Context')
        ax.plot(t_pred, gt.cpu().numpy(), 'g-', lw=2, label='Truth')
        for s in range(min(10, preds.shape[0])):
            ax.plot(t_pred, preds[s].cpu().numpy(), 'r-', alpha=0.2, lw=0.7)
        ax.plot(t_pred, preds.median(dim=0).values.cpu().numpy(), 'r-', lw=2, label='Median')
        ax.axvline(x=CTX_LEN-0.5, color='gray', ls='--', alpha=0.5)
        ax.set_title(f'{label} (×{mult})')
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f'Controllable Generation: {DATASET}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/controllability_{DATASET}.png', dpi=150)
    plt.close()
    print(f"\n  Saved plots/controllability_{DATASET}.png")


if __name__ == "__main__":
    main()
