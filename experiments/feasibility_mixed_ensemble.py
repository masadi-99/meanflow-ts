"""
Experiment B: Mixed Ensemble — 1-step + stochastic 2-step

The core test: does mixing accurate 1-step samples with diverse
stochastic 2-step samples beat pure 1-step at equal compute?
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

SIGMAS_TO_TEST = [0.1, 0.2, 0.3, 0.5]
N_TEST_WINDOWS = 500
T_MID = 0.5
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_configs(sigma):
    return [
        (100,  0, 0,     "pure 1-step ×100 (100 NFE)"),
        (150,  0, 0,     "pure 1-step ×150 (150 NFE)"),
        (200,  0, 0,     "pure 1-step ×200 (200 NFE)"),
        (300,  0, 0,     "pure 1-step ×300 (300 NFE)"),
        (50,  50, sigma,  f"mix 50+50 σ={sigma} (150 NFE)"),
        (100, 25, sigma,  f"mix 100+25 σ={sigma} (150 NFE)"),
        (100, 50, sigma,  f"mix 100+50 σ={sigma} (200 NFE)"),
        (50, 75,  sigma,  f"mix 50+75 σ={sigma} (200 NFE)"),
        (100,100, sigma,  f"mix 100+100 σ={sigma} (300 NFE)"),
        (0,  100, sigma,  f"pure stoch-2step ×100 σ={sigma} (200 NFE)"),
    ]


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE).eval()
    net.load_state_dict(ckpt['net_ema'])
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
        eps = torch.randn_like(z_mid)
        z_mid = (1 - sigma**2)**0.5 * z_mid + sigma * eps
    u_s2 = net(z_mid, (torch.full((B,), t_mid, device=DEVICE),
                        torch.full((B,), t_mid, device=DEVICE)), ctx)
    return z_mid - t_mid * u_s2


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    if N < 2: return float('inf')
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_p = min(N, 1000)
    pairwise = (samples[torch.randint(0,N,(n_p,))] - samples[torch.randint(0,N,(n_p,))]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    for sigma in SIGMAS_TO_TEST:
        configs = make_configs(sigma)
        results = {i: [] for i in range(len(configs))}

        n_eval = 0
        for entry in test_data:
            if n_eval >= N_TEST_WINDOWS: break
            target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
            if len(target) < CTX_LEN + max_lag + PRED_LEN: continue

            ground_truth = target[-PRED_LEN:]
            past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
            ctx = past[-CTX_LEN:]
            loc = ctx.abs().mean().clamp(min=0.01)
            ctx_with_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc

            max_n = max(c[0] + c[1] for c in configs)
            z_pool = torch.randn(max_n, PRED_LEN, device=DEVICE)

            for ci, (n_1, n_2, sig, label) in enumerate(configs):
                all_samples = []
                if n_1 > 0:
                    z_1 = z_pool[:n_1]
                    ctx_b = ctx_with_lags.expand(n_1, -1, -1)
                    all_samples.append(generate_1step(net, z_1, ctx_b) * loc)
                if n_2 > 0:
                    z_1_2 = z_pool[n_1:n_1+n_2] if n_1+n_2 <= max_n else torch.randn(n_2, PRED_LEN, device=DEVICE)
                    ctx_b = ctx_with_lags.expand(n_2, -1, -1)
                    all_samples.append(generate_stochastic_2step(net, z_1_2, ctx_b, sig, T_MID) * loc)
                samples = torch.cat(all_samples, dim=0)
                results[ci].append(crps_from_samples(samples, ground_truth))

            n_eval += 1
            if n_eval % 100 == 0: print(f"  [{n_eval}/{N_TEST_WINDOWS}] σ={sigma}")

        print(f"\n{'='*85}")
        print(f"EXPERIMENT B: MIXED ENSEMBLE — {DATASET} — σ={sigma}")
        print(f"{'='*85}")
        print(f"  {'Config':<45} | {'CRPS':>10} | {'NFE':>5}")
        print(f"  {'-'*65}")

        for ci, (n_1, n_2, sig, label) in enumerate(configs):
            c = np.mean(results[ci])
            nfe = n_1 + 2 * n_2
            print(f"  {label:<45} | {c:>10.6f} | {nfe:>5}")

        print(f"\n  KEY COMPARISONS:")
        pure_150_idx = next(i for i, (n1,n2,s,l) in enumerate(configs) if n1==150 and n2==0)
        mix_150_idxs = [i for i, (n1,n2,s,l) in enumerate(configs) if n1+2*n2==150 and n2>0]
        for mi in mix_150_idxs:
            c_pure = np.mean(results[pure_150_idx])
            c_mix = np.mean(results[mi])
            delta = (c_pure - c_mix) / c_pure * 100
            print(f"    150 NFE: {configs[mi][3]} vs pure-1step → {delta:+.2f}%")

        pure_200_idx = next(i for i, (n1,n2,s,l) in enumerate(configs) if n1==200 and n2==0)
        mix_200_idxs = [i for i, (n1,n2,s,l) in enumerate(configs) if n1+2*n2==200 and n2>0]
        for mi in mix_200_idxs:
            c_pure = np.mean(results[pure_200_idx])
            c_mix = np.mean(results[mi])
            delta = (c_pure - c_mix) / c_pure * 100
            print(f"    200 NFE: {configs[mi][3]} vs pure-1step → {delta:+.2f}%")

    print(f"\n  VERDICT:")
    print(f"    If any mixed ensemble beats pure 1-step at equal NFE by >1%:")
    print(f"    → Stochastic multi-scale is a real contribution. Proceed to Experiment C.")


if __name__ == "__main__":
    main()
