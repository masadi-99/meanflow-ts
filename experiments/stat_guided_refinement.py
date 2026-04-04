"""
Round 4: Stat-Guided Self-Refinement

Pipeline:
1. Generate N unconditional samples (N NFE)
2. Compute ensemble statistics: mean, max, min, std, argmax, auc from the ensemble
3. Re-generate N samples conditioned on ensemble stats (N more NFE)
4. Total: 2N NFE

Why this should work where coarse self-refinement failed:
- Stats are 6 numbers (robust ensemble estimate) vs 12-24 for coarse trajectory
- Stats anchor the distribution's LOCATION without constraining DIVERSITY
- The model learned stat conditioning with dropout, so unconditional is the fallback

Compare at EQUAL NFE: 2N unconditional vs N + N stat-guided
"""
import os, sys, torch, numpy as np, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v3 import ConditionalMeanFlowNetV3
from meanflow_ts.model_v2 import extract_lag_features
from meanflow_ts.utils import extract_stats
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
    "traffic_nips":       {"freq": "H", "ctx": 24, "pred": 24},
    "m4_hourly":          {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS.get(DATASET, {"freq": "H", "ctx": 24, "pred": 24})
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WINDOWS = 500


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
        print(f"No stats/both checkpoint for {DATASET}")
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    # Configs: (n_first, n_second, use_stat_guide, label)
    configs = [
        (50,  0,   False, "50 uncond (50 NFE)"),
        (100, 0,   False, "100 uncond (100 NFE)"),
        (200, 0,   False, "200 uncond (200 NFE)"),
        (50,  50,  True,  "50+50 stat-guided (100 NFE)"),
        (100, 100, True,  "100+100 stat-guided (200 NFE)"),
        (25,  75,  True,  "25+75 stat-guided (100 NFE)"),
        (75,  25,  True,  "75+25 stat-guided (100 NFE)"),
    ]

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
        scaled_gt = gt / loc

        for ci, (n1, n2, guided, label) in enumerate(configs):
            # First pass: unconditional
            preds_1 = generate(net, n1, ctx_lags)  # scaled space

            if guided and n2 > 0:
                # Compute ensemble stats from first pass
                ensemble_stats = extract_stats(preds_1)  # (n1, 6)
                # Use MEDIAN of per-sample stats as the consensus
                consensus_stats = ensemble_stats.median(dim=0).values.unsqueeze(0)  # (1, 6)

                # Second pass: conditioned on consensus stats
                preds_2 = generate(net, n2, ctx_lags, consensus_stats)

                # Combine both passes
                all_preds = torch.cat([preds_1, preds_2], dim=0) * loc
            else:
                all_preds = preds_1 * loc

            results[ci].append(crps_from_samples(all_preds, gt))

        n_eval += 1
        if n_eval % 100 == 0:
            c_100u = np.mean(results[1])
            c_5050 = np.mean(results[3])
            imp = (c_100u - c_5050) / c_100u * 100
            print(f"  [{n_eval}/{N_WINDOWS}] 100-uncond={c_100u:.5f} 50+50-stat={c_5050:.5f} ({imp:+.1f}%)")

    print(f"\n{'='*80}")
    print(f"STAT-GUIDED SELF-REFINEMENT — {DATASET}")
    print(f"{'='*80}")
    print(f"  {'Config':<40} | {'CRPS':>10} | {'NFE':>5}")
    print(f"  {'-'*60}")
    for ci, (n1, n2, guided, label) in enumerate(configs):
        c = np.mean(results[ci])
        nfe = n1 + n2
        print(f"  {label:<40} | {c:>10.5f} | {nfe:>5}")

    # Key comparisons at equal NFE
    print(f"\n  KEY COMPARISONS (equal NFE):")
    c_100u = np.mean(results[1])
    for ci in [3, 5, 6]:  # 50+50, 25+75, 75+25
        c_ref = np.mean(results[ci])
        nfe = configs[ci][0] + configs[ci][1]
        imp = (c_100u - c_ref) / c_100u * 100
        print(f"    100 NFE: {configs[ci][3]:>35} vs 100-uncond → {imp:+.2f}%")

    c_200u = np.mean(results[2])
    c_100100 = np.mean(results[4])
    imp_200 = (c_200u - c_100100) / c_200u * 100
    print(f"    200 NFE: 100+100-stat vs 200-uncond → {imp_200:+.2f}%")

    # Best config
    refine_configs = [3, 4, 5, 6]
    best_ci = min(refine_configs, key=lambda i: np.mean(results[i]))
    equal_nfe_uncond = {3: 1, 4: 2, 5: 1, 6: 1}[best_ci]
    c_best = np.mean(results[best_ci])
    c_comp = np.mean(results[equal_nfe_uncond])
    imp_best = (c_comp - c_best) / c_comp * 100

    print(f"\n  VERDICT:")
    if imp_best > 2:
        print(f"  ✓ STAT-GUIDED REFINEMENT WORKS! {configs[best_ci][3]}")
        print(f"    Beats equal-NFE unconditional by {imp_best:.1f}%")
    elif imp_best > 0.5:
        print(f"  ~ MARGINAL ({imp_best:.1f}%). {configs[best_ci][3]}")
    else:
        print(f"  ✗ STAT-GUIDED REFINEMENT DOESN'T BEAT UNCOND ({imp_best:.1f}%)")

    np.savez(f"stat_guided_{DATASET}.npz",
             **{f"crps_{i}": results[i] for i in range(len(configs))})


if __name__ == "__main__":
    main()
