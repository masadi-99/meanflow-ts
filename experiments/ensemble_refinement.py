"""
Ensemble Self-Refinement: Use ensemble consensus as coarse conditioning.

Round 2 pivot: single-sample self-refinement failed because one sample is too noisy.
Ensemble median of N samples should be a much better coarse signal.

Pipeline:
1. Generate N samples unconditionally (N NFE)
2. Take ensemble median → downsample to coarse
3. Generate N NEW samples conditioned on the coarse (N more NFE)
4. Total: 2N NFE

Compare at EQUAL NFE:
- 2N unconditional samples (2N NFE) vs N+N ensemble-refined samples (2N NFE)
"""
import os, sys, torch, numpy as np, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v3 import ConditionalMeanFlowNetV3
from meanflow_ts.model_v2 import extract_lag_features
from meanflow_ts.utils import downsample, upsample
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
    "electricity_nips":   {"freq": "H", "ctx": 24, "pred": 24},
    "solar_nips":         {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "exchange_rate_nips": {"freq": "B", "ctx": 30, "pred": 30},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS[DATASET]
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WINDOWS = 500


def load_model():
    for mode in ["multires", "both"]:
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
    print("No checkpoint found!")
    return None


@torch.no_grad()
def generate_batch(net, n_samples, ctx_lags, coarse_up=None):
    ctx_batch = ctx_lags.expand(n_samples, -1, -1)
    z_1 = torch.randn(n_samples, PRED_LEN, device=DEVICE)
    B = z_1.shape[0]
    t = torch.ones(B, device=DEVICE)
    h = torch.ones(B, device=DEVICE)
    u = net(z_1, (t, h), ctx_batch, coarse_up)
    return z_1 - u  # scaled predictions


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    if N < 2: return float('inf')
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_p = min(N, 1000)
    idx1 = torch.randint(0, N, (n_p,))
    idx2 = torch.randint(0, N, (n_p,))
    pairwise = (samples[idx1] - samples[idx2]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    if net is None:
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)
    valid_r = [r for r in [2, 4] if PRED_LEN % r == 0]

    # Test configurations: (N_first_pass, N_second_pass, use_refinement, label)
    # Total NFE = N_first + N_second
    configs = [
        (50, 0, False, "50 uncond (50 NFE)"),
        (100, 0, False, "100 uncond (100 NFE)"),
        (200, 0, False, "200 uncond (200 NFE)"),
        (50, 50, True, "50+50 ensemble-refine (100 NFE)"),
        (100, 100, True, "100+100 ensemble-refine (200 NFE)"),
        (25, 75, True, "25+75 ensemble-refine (100 NFE)"),
    ]

    best_r = valid_r[0] if valid_r else 2

    results = {i: [] for i in range(len(configs))}

    n_eval = 0
    for entry in test_data:
        if n_eval >= N_WINDOWS:
            break
        target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
        if len(target) < CTX_LEN + max_lag + PRED_LEN:
            continue

        gt = target[-PRED_LEN:]
        past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
        ctx = past[-CTX_LEN:]
        loc = ctx.abs().mean().clamp(min=0.01)
        ctx_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc

        for ci, (n1, n2, refine, label) in enumerate(configs):
            # First pass: unconditional
            preds_1 = generate_batch(net, n1, ctx_lags) * loc

            if refine and n2 > 0 and valid_r:
                # Ensemble median as coarse
                ensemble_median = preds_1.median(dim=0).values / loc
                coarse = downsample(ensemble_median, best_r)
                coarse_up = upsample(coarse, best_r, PRED_LEN).unsqueeze(0).expand(n2, -1)

                # Second pass: conditioned on ensemble coarse
                residuals = generate_batch(net, n2, ctx_lags, coarse_up)
                preds_2 = (residuals + coarse_up) * loc

                # Combine first and second pass samples
                all_preds = torch.cat([preds_1, preds_2], dim=0)
            else:
                all_preds = preds_1

            results[ci].append(crps_from_samples(all_preds, gt))

        n_eval += 1
        if n_eval % 100 == 0:
            c_base = np.mean(results[1])  # 100 uncond
            c_ref = np.mean(results[3])   # 50+50 refine
            print(f"  [{n_eval}/{N_WINDOWS}] 100-uncond={c_base:.5f} 50+50-refine={c_ref:.5f} "
                  f"({(c_base-c_ref)/c_base*100:+.1f}%)")

    print(f"\n{'='*80}")
    print(f"ENSEMBLE SELF-REFINEMENT — {DATASET} (r={best_r})")
    print(f"{'='*80}")
    print(f"  {'Config':<40} | {'CRPS':>10} | {'NFE':>5}")
    print(f"  {'-'*60}")
    for ci, (n1, n2, refine, label) in enumerate(configs):
        c = np.mean(results[ci])
        nfe = n1 + n2
        print(f"  {label:<40} | {c:>10.5f} | {nfe:>5}")

    # Key comparison at equal NFE
    print(f"\n  KEY COMPARISONS (equal NFE):")
    # 100 NFE: 100 uncond vs 50+50 refine
    c_100u = np.mean(results[1])
    c_5050 = np.mean(results[3])
    c_2575 = np.mean(results[5])
    print(f"    100 NFE: 100-uncond={c_100u:.5f} vs 50+50-refine={c_5050:.5f} → {(c_100u-c_5050)/c_100u*100:+.2f}%")
    print(f"    100 NFE: 100-uncond={c_100u:.5f} vs 25+75-refine={c_2575:.5f} → {(c_100u-c_2575)/c_100u*100:+.2f}%")

    # 200 NFE: 200 uncond vs 100+100 refine
    c_200u = np.mean(results[2])
    c_100100 = np.mean(results[4])
    print(f"    200 NFE: 200-uncond={c_200u:.5f} vs 100+100-refine={c_100100:.5f} → {(c_200u-c_100100)/c_200u*100:+.2f}%")

    best_refine = min([3, 4, 5], key=lambda i: np.mean(results[i]))
    best_uncond_at_same_nfe = {3: 1, 4: 2, 5: 1}[best_refine]
    imp = (np.mean(results[best_uncond_at_same_nfe]) - np.mean(results[best_refine])) / np.mean(results[best_uncond_at_same_nfe]) * 100

    print(f"\n  VERDICT:")
    if imp > 2:
        print(f"  ✓ ENSEMBLE REFINEMENT WORKS! Best config: {configs[best_refine][3]}")
        print(f"    Beats equal-NFE unconditional by {imp:.1f}%")
        print(f"    This is a genuine contribution: ensemble consensus → coarse → refine")
    elif imp > 0.5:
        print(f"  ~ MARGINAL ({imp:.1f}%). Try on more datasets.")
    else:
        print(f"  ✗ ENSEMBLE REFINEMENT DOESN'T BEAT UNCOND at equal NFE ({imp:.1f}%)")

    np.savez(f"ensemble_refine_{DATASET}.npz",
             **{f"crps_{i}": results[i] for i in range(len(configs))})


if __name__ == "__main__":
    main()
