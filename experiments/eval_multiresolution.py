"""
Evaluate multi-resolution refinement and stat conditioning capabilities.

NOTE: These are ORACLE/CEILING experiments. The coarse trajectory and statistics
are derived from the ground-truth future. Results measure the model's ABILITY to
condition, not realistic forecasting performance. In a real application, coarse
trajectories would come from a coarser model's prediction, and statistics would
be user-specified targets.

Tests:
1. Refinement: give oracle coarse future → does fine detail improve?
2. Stat control: give oracle stats → does model hit them?
3. Coarse consistency: downsample(generated) ≈ given coarse?
4. Diversity under fixed conditioning

IMPORTANT: CRPS values computed here use energy CRPS (unnormalized), NOT
GluonTS mean_wQuantileLoss. These absolute values are NOT comparable to
the normalized CRPS in Table 3. Only relative comparisons within this
script (e.g., refinement vs unconditional) are valid.
"""
import os, sys, torch, numpy as np, tempfile

torch.manual_seed(6432)
np.random.seed(6432)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v3 import ConditionalMeanFlowNetV3
from meanflow_ts.model_v2 import extract_lag_features
from meanflow_ts.utils import downsample, upsample, extract_stats, N_STATS
from gluonts.dataset.repository.datasets import get_dataset

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

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
N_SAMPLES = 50
N_WINDOWS = 500


def load_model(mode):
    ckpt_path = f"best_v3_{mode}_{DATASET}.pt"
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    net = ConditionalMeanFlowNetV3(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE).eval()
    net.load_state_dict(ckpt['net_ema'])
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch','?')}, CRPS={ckpt.get('crps','?'):.4f})")
    return net


@torch.no_grad()
def generate(net, z_1, ctx_lags, coarse_up=None, stat_vec=None):
    B = z_1.shape[0]
    t = torch.ones(B, device=DEVICE)
    h = torch.ones(B, device=DEVICE)
    u = net(z_1, (t, h), ctx_lags, coarse_up, stat_vec)
    return z_1 - u


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    if N < 2: return float('inf')
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_p = min(N, 500)
    pairwise = (samples[torch.randint(0,N,(n_p,))] - samples[torch.randint(0,N,(n_p,))]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def eval_refinement(net):
    """Test 1: Does oracle coarse conditioning improve forecasts?"""
    print("\n  === REFINEMENT EVALUATION ===")
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    results = {"uncond": [], "r2": [], "r4": []}
    coarse_consistency = {"r2": [], "r4": []}

    n_eval = 0
    for entry in test_data:
        if n_eval >= N_WINDOWS: break
        target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
        if len(target) < CTX_LEN + max_lag + PRED_LEN: continue

        gt = target[-PRED_LEN:]
        past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
        ctx = past[-CTX_LEN:]
        loc = ctx.abs().mean().clamp(min=0.01)
        ctx_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc
        ctx_batch = ctx_lags.expand(N_SAMPLES, -1, -1)
        scaled_gt = gt / loc

        z_1 = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)

        # Unconditional
        pred_uncond = generate(net, z_1, ctx_batch) * loc
        results["uncond"].append(crps_from_samples(pred_uncond, gt))

        # With oracle coarse at r=2, r=4
        for r in [2, 4]:
            if PRED_LEN % r != 0: continue
            key = f"r{r}"
            coarse = downsample(scaled_gt, r)
            coarse_up = upsample(coarse, r, PRED_LEN)
            coarse_up_batch = coarse_up.unsqueeze(0).expand(N_SAMPLES, -1)

            # Generate residual, add back coarse
            residual = generate(net, z_1, ctx_batch, coarse_up_batch)
            pred = (residual + coarse_up_batch) * loc

            results[key].append(crps_from_samples(pred, gt))

            # Coarse consistency: downsample(pred) vs coarse
            pred_coarse = downsample(pred.mean(0) / loc, r)
            consistency = (pred_coarse - coarse).abs().mean().item()
            coarse_consistency[key].append(consistency)

        n_eval += 1

    print(f"  Windows: {n_eval}")
    for key in results:
        if results[key]:
            c = np.mean(results[key])
            print(f"  CRPS ({key:>8}): {c:.6f}")
    for key in coarse_consistency:
        if coarse_consistency[key]:
            print(f"  Coarse consistency ({key}): {np.mean(coarse_consistency[key]):.6f}")

    # Key comparison
    c_uncond = np.mean(results["uncond"])
    for r_key in ["r2", "r4"]:
        if results[r_key]:
            c_ref = np.mean(results[r_key])
            imp = (c_uncond - c_ref) / c_uncond * 100
            print(f"  Refinement {r_key} vs uncond: {imp:+.2f}%")

    return results


def eval_stat_control(net):
    """Test 2: Does stat conditioning control the generated futures?"""
    print("\n  === STAT CONTROL EVALUATION ===")
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    stat_errors = {s: [] for s in ["mean", "max", "min", "std", "argmax", "auc"]}
    crps_uncond = []
    crps_conditioned = []

    n_eval = 0
    for entry in test_data:
        if n_eval >= N_WINDOWS: break
        target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
        if len(target) < CTX_LEN + max_lag + PRED_LEN: continue

        gt = target[-PRED_LEN:]
        past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
        ctx = past[-CTX_LEN:]
        loc = ctx.abs().mean().clamp(min=0.01)
        ctx_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc
        ctx_batch = ctx_lags.expand(N_SAMPLES, -1, -1)
        scaled_gt = gt / loc

        # Oracle stats from ground truth
        oracle_stats = extract_stats(scaled_gt.unsqueeze(0))  # (1, n_stats)
        stat_batch = oracle_stats.expand(N_SAMPLES, -1)

        z_1 = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)

        # Unconditional
        pred_uncond = generate(net, z_1, ctx_batch) * loc
        crps_uncond.append(crps_from_samples(pred_uncond, gt))

        # With oracle stats
        pred_cond = generate(net, z_1, ctx_batch, stat_vec=stat_batch) * loc
        crps_conditioned.append(crps_from_samples(pred_cond, gt))

        # Measure stat accuracy on generated samples
        gen_stats = extract_stats(pred_cond / loc)  # (N_SAMPLES, n_stats)
        oracle_expanded = oracle_stats.expand(N_SAMPLES, -1)
        stat_names = ["mean", "max", "min", "std", "argmax", "auc"]
        for i, sn in enumerate(stat_names):
            err = (gen_stats[:, i] - oracle_expanded[:, i]).abs().mean().item()
            stat_errors[sn].append(err)

        n_eval += 1

    print(f"  Windows: {n_eval}")
    c_u = np.mean(crps_uncond)
    c_c = np.mean(crps_conditioned)
    imp = (c_u - c_c) / c_u * 100
    print(f"  CRPS (unconditional): {c_u:.6f}")
    print(f"  CRPS (oracle stats):  {c_c:.6f}")
    print(f"  Improvement: {imp:+.2f}%")

    print(f"\n  Stat control accuracy (MAE, lower=better):")
    for sn in stat_names:
        print(f"    {sn:>8}: {np.mean(stat_errors[sn]):.4f}")

    return crps_uncond, crps_conditioned, stat_errors


def eval_diversity(net):
    """Test 3: Are samples diverse under fixed conditioning?"""
    print("\n  === DIVERSITY EVALUATION ===")
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    div_uncond = []
    div_coarse = []
    div_stats = []

    n_eval = 0
    for entry in test_data[:100]:
        target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
        if len(target) < CTX_LEN + max_lag + PRED_LEN: continue

        gt = target[-PRED_LEN:]
        past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
        ctx = past[-CTX_LEN:]
        loc = ctx.abs().mean().clamp(min=0.01)
        ctx_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc
        ctx_batch = ctx_lags.expand(N_SAMPLES, -1, -1)
        scaled_gt = gt / loc

        z_1 = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)

        # Unconditional
        pred_u = generate(net, z_1, ctx_batch)
        div_uncond.append(pred_u.std(dim=0).mean().item())

        # With coarse r=2
        if PRED_LEN % 2 == 0:
            coarse = downsample(scaled_gt, 2)
            coarse_up = upsample(coarse, 2, PRED_LEN).unsqueeze(0).expand(N_SAMPLES, -1)
            pred_c = generate(net, z_1, ctx_batch, coarse_up)
            div_coarse.append(pred_c.std(dim=0).mean().item())

        # With stats
        stats = extract_stats(scaled_gt.unsqueeze(0)).expand(N_SAMPLES, -1)
        pred_s = generate(net, z_1, ctx_batch, stat_vec=stats)
        div_stats.append(pred_s.std(dim=0).mean().item())

        n_eval += 1

    print(f"  Diversity (std across {N_SAMPLES} samples, avg over time):")
    print(f"    Unconditional:   {np.mean(div_uncond):.4f}")
    if div_coarse:
        print(f"    Coarse (r=2):    {np.mean(div_coarse):.4f}")
    print(f"    Stat-conditioned: {np.mean(div_stats):.4f}")
    ratio = np.mean(div_stats) / np.mean(div_uncond) if np.mean(div_uncond) > 0 else 0
    print(f"    Stat/Uncond ratio: {ratio:.2f} (want 0.3-0.8: diverse but more focused)")


def main():
    # Try each mode
    for mode in ["multires", "stats", "both"]:
        net = load_model(mode)
        if net is None:
            print(f"\n{'='*60}\nSKIP mode={mode}: no checkpoint\n{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"EVALUATING mode={mode} on {DATASET}")
        print(f"{'='*60}")

        if mode in ("multires", "both"):
            eval_refinement(net)
        if mode in ("stats", "both"):
            eval_stat_control(net)
        eval_diversity(net)


if __name__ == "__main__":
    main()
