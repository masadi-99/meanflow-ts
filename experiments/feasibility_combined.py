"""
Experiment 4: Combined Inference-Time Scaling — Pareto Frontier

Sweeps (k_steps, n_candidates, selection_fraction).
Run after Experiments 1 and/or 2 succeed.
"""
import os, sys, time, torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features
from gluonts.dataset.repository.datasets import get_dataset

DATASET = os.environ.get("DATASET", "electricity_nips")
CKPT = os.environ.get("CKPT", f"best_v4_{DATASET}.pt")

CONFIGS_DS = {
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
    "uber_tlc_hourly":    {"freq": "H", "ctx": 24, "pred": 24},
    "wiki2000_nips":      {"freq": "1D", "ctx": 30, "pred": 30},
    "kdd_cup_2018_without_missing": {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS_DS[DATASET]
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7

N_TEST_WINDOWS = 500
T_MID = 0.5
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = [
    (1,   50,  1.0,  "1step×50"),
    (1,  100,  1.0,  "1step×100"),
    (1,  200,  1.0,  "1step×200"),
    (1,  200,  0.5,  "1step×200→100 SCS"),
    (1,  400,  0.25, "1step×400→100 SCS"),
    (2,   50,  1.0,  "2step×50"),
    (2,  100,  1.0,  "2step×100"),
    (2,  200,  0.5,  "2step×200→100 SCS"),
    (4,   50,  1.0,  "4step×50"),
    (4,  100,  0.5,  "4step×100→50 SCS"),
    (8,   25,  1.0,  "8step×25"),
]

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
    return net


@torch.no_grad()
def meanflow_k_step(net, z_1, ctx, k):
    B = z_1.shape[0]
    z = z_1.clone()
    dt = 1.0 / k
    for i in range(k):
        t_c = 1.0 - i * dt
        u = net(z, (torch.full((B,), t_c, device=DEVICE), torch.full((B,), dt, device=DEVICE)), ctx)
        z = z - dt * u
    return z


@torch.no_grad()
def compute_scs_for_selection(net, z_1, ctx, t_mid=0.5):
    B = z_1.shape[0]
    t1 = torch.ones(B, device=DEVICE)
    u1 = net(z_1, (t1, t1), ctx)
    z_0_1step = z_1 - u1
    h_s1 = 1.0 - t_mid
    u_s1 = net(z_1, (t1, torch.full((B,), h_s1, device=DEVICE)), ctx)
    z_mid = z_1 - h_s1 * u_s1
    u_s2 = net(z_mid, (torch.full((B,), t_mid, device=DEVICE), torch.full((B,), t_mid, device=DEVICE)), ctx)
    z_0_2step = z_mid - t_mid * u_s2
    return ((z_0_1step - z_0_2step) ** 2).sum(dim=-1)


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    if N < 2:
        return float('inf')
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_p = min(N, 500)
    pairwise = (samples[torch.randint(0,N,(n_p,))] - samples[torch.randint(0,N,(n_p,))]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    results = {i: [] for i in range(len(CONFIGS))}
    wall_times = {i: [] for i in range(len(CONFIGS))}

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
        ctx_with_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc

        max_cand = max(c[1] for c in CONFIGS)
        z_1_pool = torch.randn(max_cand, PRED_LEN, device=DEVICE)

        for ci, (k, n_cand, sel_frac, label) in enumerate(CONFIGS):
            z_1 = z_1_pool[:n_cand]
            ctx_batch = ctx_with_lags.expand(n_cand, -1, -1)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            z_0 = meanflow_k_step(net, z_1, ctx_batch, k)
            samples = z_0 * loc

            if sel_frac < 1.0:
                scs = compute_scs_for_selection(net, z_1, ctx_batch, T_MID)
                n_sel = max(1, int(n_cand * sel_frac))
                samples = samples[scs.argsort()[:n_sel]]

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            results[ci].append(crps_from_samples(samples, ground_truth))
            wall_times[ci].append(elapsed)

        n_eval += 1
        if n_eval % 100 == 0:
            print(f"  [{n_eval}/{N_TEST_WINDOWS}]")

    print("\n" + "=" * 90)
    print(f"EXPERIMENT 4: COMBINED PARETO FRONTIER — {DATASET}")
    print("=" * 90)
    print(f"\n  {'Config':<30} | {'CRPS':>8} | {'NFE':>6} | {'Time(ms)':>8}")
    print(f"  {'-'*65}")

    points = []
    for ci, (k, n_cand, sel_frac, label) in enumerate(CONFIGS):
        crps = np.mean(results[ci])
        wt = np.mean(wall_times[ci]) * 1000
        nfe_gen = n_cand * k
        nfe_scs = n_cand * 2 if sel_frac < 1.0 else 0
        nfe = nfe_gen + nfe_scs
        print(f"  {label:<30} | {crps:>8.5f} | {nfe:>6} | {wt:>8.1f}")
        points.append((nfe, crps, label))

    points.sort(key=lambda x: x[0])
    front = []
    best = float('inf')
    for nfe, crps, label in points:
        if crps < best:
            front.append((nfe, crps, label))
            best = crps

    print(f"\n  Pareto-optimal configs:")
    for nfe, crps, label in front:
        marker = " ← SCS+multi" if "SCS" in label and "step" in label and "1step" not in label else ""
        print(f"    NFE={nfe:>5} | CRPS={crps:.5f} | {label}{marker}")

    has_combined = any("SCS" in l and "step" in l and "1step" not in l for _,_,l in front)
    print(f"\n  VERDICT:")
    if has_combined:
        print("  ✓ COMBINED (multi-step + SCS) IS ON THE PARETO FRONTIER.")
    else:
        print("  ~ Combined method is not dominant. Individual axes may still be useful.")

    np.savez(f"combined_feasibility_{DATASET}.npz",
             labels=[c[3] for c in CONFIGS],
             crps=[np.mean(results[i]) for i in range(len(CONFIGS))],
             wall_times=[np.mean(wall_times[i]) for i in range(len(CONFIGS))])
    print(f"\n  Saved: combined_feasibility_{DATASET}.npz")


if __name__ == "__main__":
    main()
