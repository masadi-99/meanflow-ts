"""
Experiment A: Stochastic 2-step inference with midpoint noise injection

Tests whether injecting noise at the midpoint of a 2-step MeanFlow path
produces diverse-but-useful forecasts.

Sweeps σ ∈ [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
Compares CRPS of stochastic 2-step against deterministic 1-step.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features
from gluonts.dataset.repository.datasets import get_dataset

import tempfile
try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

# ============== CONFIG ==============
DATASET = os.environ.get("DATASET", "electricity_nips")
CKPT = os.environ.get("CKPT", f"best_v4_{DATASET}.pt")
CONFIGS = {
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
    "uber_tlc_hourly":    {"freq": "H", "ctx": 24, "pred": 24},
    "wiki2000_nips":      {"freq": "1D", "ctx": 30, "pred": 30},
    "kdd_cup_2018_without_missing": {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS[DATASET]
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7
# =====================================

NUM_SAMPLES = 100
N_TEST_WINDOWS = 500
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
T_MID = 0.5
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE).eval()
    net.load_state_dict(ckpt['net_ema'])
    print(f"Loaded {CKPT} (epoch {ckpt.get('epoch','?')})")
    return net


@torch.no_grad()
def generate_1step(net, z_1, ctx):
    B = z_1.shape[0]
    u = net(z_1, (torch.ones(B, device=DEVICE), torch.ones(B, device=DEVICE)), ctx)
    return z_1 - u


@torch.no_grad()
def generate_stochastic_2step(net, z_1, ctx, sigma, t_mid=0.5):
    B = z_1.shape[0]
    h_s1 = 1.0 - t_mid
    u_s1 = net(z_1, (torch.ones(B, device=DEVICE),
                      torch.full((B,), h_s1, device=DEVICE)), ctx)
    z_mid = z_1 - h_s1 * u_s1
    if sigma > 0:
        eps_fresh = torch.randn_like(z_mid)
        z_mid = (1 - sigma**2)**0.5 * z_mid + sigma * eps_fresh
    u_s2 = net(z_mid, (torch.full((B,), t_mid, device=DEVICE),
                        torch.full((B,), t_mid, device=DEVICE)), ctx)
    z_0 = z_mid - t_mid * u_s2
    return z_0


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_p = min(N, 1000)
    idx1 = torch.randint(0, N, (n_p,))
    idx2 = torch.randint(0, N, (n_p,))
    pairwise = (samples[idx1] - samples[idx2]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    results_1step = []
    results_by_sigma = {s: [] for s in SIGMAS}
    spread_by_sigma = {s: [] for s in SIGMAS}

    n_eval = 0
    for entry in test_data:
        if n_eval >= N_TEST_WINDOWS:
            break
        target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
        if len(target) < CTX_LEN + max_lag + PRED_LEN:
            continue

        ground_truth = target[-PRED_LEN:]
        past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
        ctx = past[-CTX_LEN:]
        loc = ctx.abs().mean().clamp(min=0.01)

        ctx_with_lags = extract_lag_features(
            past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS
        ).to(DEVICE) / loc
        ctx_batch = ctx_with_lags.expand(NUM_SAMPLES, -1, -1)

        z_1 = torch.randn(NUM_SAMPLES, PRED_LEN, device=DEVICE)

        z_0_1step = generate_1step(net, z_1, ctx_batch)
        samples_1step = z_0_1step * loc
        results_1step.append(crps_from_samples(samples_1step, ground_truth))

        for sigma in SIGMAS:
            z_0_stoch = generate_stochastic_2step(net, z_1, ctx_batch, sigma, T_MID)
            samples_stoch = z_0_stoch * loc
            results_by_sigma[sigma].append(crps_from_samples(samples_stoch, ground_truth))
            spread_by_sigma[sigma].append(samples_stoch.std(dim=0).mean().item())

        n_eval += 1
        if n_eval % 100 == 0:
            c1 = np.mean(results_1step)
            print(f"  [{n_eval}/{N_TEST_WINDOWS}] 1-step={c1:.5f}  " +
                  "  ".join(f"σ={s}:{np.mean(results_by_sigma[s]):.5f}"
                            for s in [0.1, 0.3, 0.5]))

    print("\n" + "=" * 80)
    print(f"EXPERIMENT A: STOCHASTIC 2-STEP INFERENCE — {DATASET}")
    print("=" * 80)
    print(f"Windows: {n_eval} | Samples: {NUM_SAMPLES} | t_mid: {T_MID}")

    c_1step = np.mean(results_1step)
    print(f"\n  {'Method':<25} | {'CRPS':>10} | {'vs 1-step':>10} | {'Spread':>10}")
    print(f"  {'-'*65}")
    print(f"  {'1-step (baseline)':<25} | {c_1step:>10.6f} | {'---':>10} | "
          f"{np.mean(spread_by_sigma[0.0]):>10.4f}")

    best_sigma = None
    best_improvement = -float('inf')
    for sigma in SIGMAS:
        c_s = np.mean(results_by_sigma[sigma])
        delta = (c_1step - c_s) / c_1step * 100
        spread = np.mean(spread_by_sigma[sigma])
        label = f"stoch-2step σ={sigma}"
        print(f"  {label:<25} | {c_s:>10.6f} | {delta:>+9.2f}% | {spread:>10.4f}")
        if delta > best_improvement:
            best_improvement = delta
            best_sigma = sigma

    print(f"\n  Best σ: {best_sigma} ({best_improvement:+.2f}%)")
    print(f"  Spread at best σ: {np.mean(spread_by_sigma[best_sigma]):.4f} "
          f"vs 1-step: {np.mean(spread_by_sigma[0.0]):.4f} "
          f"({np.mean(spread_by_sigma[best_sigma])/np.mean(spread_by_sigma[0.0])*100 - 100:+.0f}%)")

    print(f"\n  VERDICT:")
    if best_improvement > 2:
        print(f"  ✓ STOCHASTIC 2-STEP HELPS. σ={best_sigma} → {best_improvement:.2f}% CRPS improvement.")
        print(f"    Proceed to Experiment B (mixed ensemble).")
    elif best_improvement > 0.5:
        print(f"  ~ MARGINAL. Best: {best_improvement:.2f}% at σ={best_sigma}.")
        print(f"    Try Experiment B anyway — mixing with 1-step might help more.")
    else:
        print(f"  ✗ STOCHASTIC 2-STEP DOESN'T HELP at inference time.")
        print(f"    Skip to Experiment C (fine-tune with midpoint noise).")

    np.savez(f"stochastic_2step_{DATASET}.npz",
             sigmas=SIGMAS, crps_1step=results_1step,
             **{f"crps_sigma_{s}": results_by_sigma[s] for s in SIGMAS},
             **{f"spread_sigma_{s}": spread_by_sigma[s] for s in SIGMAS})
    print(f"\n  Saved: stochastic_2step_{DATASET}.npz")


if __name__ == "__main__":
    main()
