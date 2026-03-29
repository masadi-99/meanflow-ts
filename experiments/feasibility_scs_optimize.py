"""
Experiment 3: SCS-guided noise optimization

Backprop through SCS to improve z_1 before final generation.
Only run if Experiment 1 showed SCS correlates with quality.
"""
import os, sys, torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features
from gluonts.dataset.repository.datasets import get_dataset

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

N_CANDIDATES = 50
N_TEST_WINDOWS = 200
T_MID = 0.5
OPT_STEPS_LIST = [0, 1, 3, 5, 10, 20]
LR = 0.01
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import tempfile
try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE).eval()
    net.load_state_dict(ckpt['net_ema'])
    print(f"Loaded {CKPT}")
    return net


def compute_scs_differentiable(net, z_1, ctx_with_lags, t_mid):
    B = z_1.shape[0]
    u_full = net(z_1, (torch.ones(B, device=DEVICE), torch.ones(B, device=DEVICE)), ctx_with_lags)
    z_0_1step = z_1 - u_full
    h_s1 = 1.0 - t_mid
    u_s1 = net(z_1, (torch.ones(B, device=DEVICE), torch.full((B,), h_s1, device=DEVICE)), ctx_with_lags)
    z_mid = z_1 - h_s1 * u_s1
    u_s2 = net(z_mid, (torch.full((B,), t_mid, device=DEVICE), torch.full((B,), t_mid, device=DEVICE)), ctx_with_lags)
    z_0_2step = z_mid - t_mid * u_s2
    return ((z_0_1step - z_0_2step) ** 2).sum(dim=-1), z_0_1step


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_p = min(N, 500)
    pairwise = (samples[torch.randint(0,N,(n_p,))] - samples[torch.randint(0,N,(n_p,))]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    results = {s: [] for s in OPT_STEPS_LIST}
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
        ctx_batch = ctx_with_lags.expand(N_CANDIDATES, -1, -1)

        z_1_init = torch.randn(N_CANDIDATES, PRED_LEN, device=DEVICE)

        for n_opt in OPT_STEPS_LIST:
            z_1 = z_1_init.clone()
            if n_opt > 0:
                z_1.requires_grad_(True)
                for _ in range(n_opt):
                    scs, _ = compute_scs_differentiable(net, z_1, ctx_batch, T_MID)
                    scs.sum().backward()
                    with torch.no_grad():
                        z_1 = (z_1 - LR * z_1.grad).clamp(-4, 4)
                    z_1 = z_1.detach().requires_grad_(True)
                z_1 = z_1.detach()

            with torch.no_grad():
                B = z_1.shape[0]
                u = net(z_1, (torch.ones(B, device=DEVICE), torch.ones(B, device=DEVICE)), ctx_batch)
                z_0 = z_1 - u

            crps = crps_from_samples(z_0 * loc, ground_truth)
            results[n_opt].append(crps)

        n_eval += 1
        if n_eval % 50 == 0:
            print(f"  [{n_eval}/{N_TEST_WINDOWS}] " +
                  " | ".join(f"{s}opt:{np.mean(results[s]):.4f}" for s in OPT_STEPS_LIST))

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: SCS-GUIDED NOISE OPTIMIZATION — {DATASET}")
    print("=" * 70)

    crps_0 = np.mean(results[0])
    for s in OPT_STEPS_LIST:
        crps_s = np.mean(results[s])
        delta = (crps_0 - crps_s) / crps_0 * 100
        nfe = N_CANDIDATES * (1 + s * 3)
        print(f"  {s:>3} opt steps | CRPS: {crps_s:.6f} | Δ: {delta:>+6.2f}% | NFE: {nfe}")

    best_s = max(OPT_STEPS_LIST, key=lambda s: (crps_0 - np.mean(results[s])))
    best_imp = (crps_0 - np.mean(results[best_s])) / crps_0 * 100

    print(f"\n  VERDICT:")
    if best_imp > 2:
        print(f"  ✓ SCS OPTIMIZATION WORKS. Best: {best_s} steps → {best_imp:.2f}% improvement.")
    elif best_imp > 0.5:
        print(f"  ~ MARGINAL ({best_imp:.2f}%). Try LR in [0.005, 0.02, 0.05].")
    else:
        print(f"  ✗ SCS OPTIMIZATION DOES NOT HELP. Stick to SCS filtering.")

    np.savez(f"scs_optimize_{DATASET}.npz",
             opt_steps=OPT_STEPS_LIST,
             **{f"crps_{s}": results[s] for s in OPT_STEPS_LIST})
    print(f"\n  Saved: scs_optimize_{DATASET}.npz")


if __name__ == "__main__":
    main()
