"""
Experiment 2: Multi-step MeanFlow inference

Tests whether k-step inference improves CRPS over 1-step.
If yes, MeanFlow has a compute-quality axis TARFVAE cannot access.
"""
import os, sys, time, torch, numpy as np

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

NUM_SAMPLES = 100
N_TEST_WINDOWS = 500
STEP_COUNTS = [1, 2, 4, 8, 16, 32]
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
    print(f"Loaded {CKPT} (epoch {ckpt.get('epoch','?')})")
    return net


@torch.no_grad()
def meanflow_k_step(net, z_1, ctx_with_lags, k_steps):
    B = z_1.shape[0]
    z = z_1.clone()
    dt = 1.0 / k_steps
    for i in range(k_steps):
        t_current = 1.0 - i * dt
        t_vec = torch.full((B,), t_current, device=DEVICE)
        h_vec = torch.full((B,), dt, device=DEVICE)
        u = net(z, (t_vec, h_vec), ctx_with_lags)
        z = z - dt * u
    return z


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_pairs = min(N, 500)
    idx1 = torch.randint(0, N, (n_pairs,))
    idx2 = torch.randint(0, N, (n_pairs,))
    pairwise = (samples[idx1] - samples[idx2]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    results = {k: [] for k in STEP_COUNTS}
    timings = {k: [] for k in STEP_COUNTS}

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

        for k in STEP_COUNTS:
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            z_0 = meanflow_k_step(net, z_1, ctx_batch, k)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            crps = crps_from_samples(z_0 * loc, ground_truth)
            results[k].append(crps)
            timings[k].append(elapsed)

        n_eval += 1
        if n_eval % 100 == 0:
            print(f"  [{n_eval}/{N_TEST_WINDOWS}] " +
                  " | ".join(f"{k}s:{np.mean(results[k]):.4f}" for k in STEP_COUNTS))

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: MULTI-STEP INFERENCE — {DATASET}")
    print("=" * 70)
    print(f"Windows: {n_eval} | Samples/window: {NUM_SAMPLES}")

    crps_1 = np.mean(results[1])
    print(f"\n  {'Steps':>6} | {'CRPS':>10} | {'vs 1-step':>10} | {'Time (ms)':>10}")
    print(f"  {'-'*50}")
    for k in STEP_COUNTS:
        crps_k = np.mean(results[k])
        delta = (crps_1 - crps_k) / crps_1 * 100
        t_ms = np.mean(timings[k]) * 1000
        print(f"  {k:>6} | {crps_k:>10.6f} | {delta:>+9.2f}% | {t_ms:>10.1f}")

    best_k = min(STEP_COUNTS, key=lambda k: np.mean(results[k]))
    best_imp = (crps_1 - np.mean(results[best_k])) / crps_1 * 100
    imp_2 = (crps_1 - np.mean(results[2])) / crps_1 * 100

    print(f"\n  VERDICT:")
    if imp_2 > 1:
        print(f"  ✓ MULTI-STEP HELPS. 2-step improves by {imp_2:.2f}% (best: {best_k}-step at {best_imp:.2f}%).")
    elif imp_2 > 0:
        print(f"  ~ MARGINAL. 2-step improves by {imp_2:.2f}%. May combine with SCS.")
    else:
        print(f"  ✗ MULTI-STEP DOES NOT HELP. Focus on SCS selection instead.")

    np.savez(f"multistep_feasibility_{DATASET}.npz",
             step_counts=STEP_COUNTS,
             **{f"crps_{k}": results[k] for k in STEP_COUNTS},
             **{f"time_{k}": timings[k] for k in STEP_COUNTS})
    print(f"\n  Saved: multistep_feasibility_{DATASET}.npz")


if __name__ == "__main__":
    main()
