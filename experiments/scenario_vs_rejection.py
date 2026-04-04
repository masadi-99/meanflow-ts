"""
Round 7: Scenario Generation vs Rejection Sampling Baseline

The reviewer's devastating critique: at 1 NFE/sample, you can generate 20K samples
and rejection-sample. Does conditioning STILL help?

Tests:
1. Rejection sampling with budget scaling (200, 2K, 20K unconditional samples)
2. Blind stat conditioning (no oracle — stats predicted from context only)
3. Quality comparison: are conditioned samples more realistic than rejection-sampled?
"""
import os, sys, torch, numpy as np, tempfile, time

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
    "electricity_nips": {"freq": "H", "ctx": 24, "pred": 24},
    "traffic_nips":     {"freq": "H", "ctx": 24, "pred": 24},
    "m4_hourly":        {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS.get(DATASET, {"freq": "H", "ctx": 24, "pred": 24})
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TARGET = 50  # we want 50 valid scenario samples


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
    t = torch.ones(n, device=DEVICE)
    h = torch.ones(n, device=DEVICE)
    stat_batch = stat_vec.expand(n, -1) if stat_vec is not None else None
    u = net(z_1, (t, h), ctx_batch, None, stat_batch)
    return z_1 - u


def main():
    net = load_model()
    if net is None:
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    # Compute historical stats for blind conditioning
    print("Computing historical stat distributions...")
    hist_stats = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        for i in range(len(ts) // PRED_LEN):
            w = ts[i*PRED_LEN:(i+1)*PRED_LEN]
            loc = max(np.abs(w).mean(), 0.01)
            scaled = w / loc
            hist_stats.append([scaled.mean(), scaled.max(), scaled.min(), scaled.std(),
                              np.argmax(scaled) / PRED_LEN, scaled.sum() / PRED_LEN])
    hist_stats = np.array(hist_stats)

    # Scenario: high peak (90th percentile)
    threshold = np.percentile(hist_stats[:, 1], 90)
    stat_idx = 1
    print(f"Scenario: high peak > {threshold:.3f} (90th pctile)")

    N_EVAL = 100
    budgets = [200, 1000, 5000, 20000]

    # Results: for each method, how many NFE to get N_TARGET valid samples?
    results_rejection = {b: {"nfe": [], "n_valid": [], "time": [], "quality": []} for b in budgets}
    results_conditioned = {"nfe": [], "n_valid": [], "time": [], "quality": []}
    results_blind = {"nfe": [], "n_valid": [], "time": [], "quality": []}

    n_eval = 0
    for entry in test_data:
        if n_eval >= N_EVAL:
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
        oracle_stats = extract_stats(scaled_gt.unsqueeze(0))

        # === Method 1: Rejection sampling at various budgets ===
        for budget in budgets:
            t0 = time.perf_counter()
            preds = generate(net, budget, ctx_lags, None)
            stats = extract_stats(preds)
            valid_mask = stats[:, stat_idx] >= threshold
            n_valid = valid_mask.sum().item()
            elapsed = time.perf_counter() - t0

            # Quality: MAE of valid samples to ground truth
            if n_valid > 0:
                valid_preds = preds[valid_mask]
                mae = (valid_preds.mean(0) - scaled_gt).abs().mean().item()
            else:
                mae = float('inf')

            results_rejection[budget]["nfe"].append(budget)
            results_rejection[budget]["n_valid"].append(n_valid)
            results_rejection[budget]["time"].append(elapsed)
            results_rejection[budget]["quality"].append(mae)

        # === Method 2: Oracle-stat conditioning (200 samples) ===
        target_stats = oracle_stats.clone()
        target_stats[0, stat_idx] = max(threshold * 1.2, target_stats[0, stat_idx])

        t0 = time.perf_counter()
        preds_cond = generate(net, 200, ctx_lags, target_stats)
        stats_cond = extract_stats(preds_cond)
        valid_cond = (stats_cond[:, stat_idx] >= threshold).sum().item()
        elapsed_cond = time.perf_counter() - t0

        if valid_cond > 0:
            mae_cond = (preds_cond[stats_cond[:, stat_idx] >= threshold].mean(0) - scaled_gt).abs().mean().item()
        else:
            mae_cond = float('inf')

        results_conditioned["nfe"].append(200)
        results_conditioned["n_valid"].append(valid_cond)
        results_conditioned["time"].append(elapsed_cond)
        results_conditioned["quality"].append(mae_cond)

        # === Method 3: Blind conditioning (stats predicted from context) ===
        # Use context statistics as proxy for future stats
        ctx_scaled = ctx_lags[0, 0, :]  # (ctx_len,)
        ctx_stats_vals = extract_stats(ctx_scaled.unsqueeze(0))  # stats of CONTEXT
        blind_stats = ctx_stats_vals.clone()
        blind_stats[0, stat_idx] = max(threshold * 1.2, blind_stats[0, stat_idx])

        t0 = time.perf_counter()
        preds_blind = generate(net, 200, ctx_lags, blind_stats)
        stats_blind = extract_stats(preds_blind)
        valid_blind = (stats_blind[:, stat_idx] >= threshold).sum().item()
        elapsed_blind = time.perf_counter() - t0

        if valid_blind > 0:
            mae_blind = (preds_blind[stats_blind[:, stat_idx] >= threshold].mean(0) - scaled_gt).abs().mean().item()
        else:
            mae_blind = float('inf')

        results_blind["nfe"].append(200)
        results_blind["n_valid"].append(valid_blind)
        results_blind["time"].append(elapsed_blind)
        results_blind["quality"].append(mae_blind)

        n_eval += 1
        if n_eval % 20 == 0:
            print(f"  [{n_eval}/{N_EVAL}] rej@5K={np.mean(results_rejection[5000]['n_valid']):.0f} "
                  f"cond={np.mean(results_conditioned['n_valid']):.0f} "
                  f"blind={np.mean(results_blind['n_valid']):.0f}")

    # === REPORT ===
    print(f"\n{'='*80}")
    print(f"SCENARIO VS REJECTION SAMPLING — {DATASET}")
    print(f"Scenario: high peak (stat > {threshold:.3f})")
    print(f"{'='*80}")

    print(f"\n  {'Method':<30} | {'NFE':>6} | {'Valid/200':>10} | {'Time(ms)':>10} | {'MAE':>8}")
    print(f"  {'-'*75}")

    for budget in budgets:
        r = results_rejection[budget]
        nfe = budget
        valid = np.mean(r["n_valid"])
        t_ms = np.mean(r["time"]) * 1000
        mae = np.mean([q for q in r["quality"] if q != float('inf')])
        print(f"  {'Reject@' + str(budget):<30} | {nfe:>6} | {valid:>10.1f} | {t_ms:>10.1f} | {mae:>8.4f}")

    r_c = results_conditioned
    valid_c = np.mean(r_c["n_valid"])
    t_c = np.mean(r_c["time"]) * 1000
    mae_c = np.mean([q for q in r_c["quality"] if q != float('inf')])
    print(f"  {'Cond (oracle stats)':<30} | {'200':>6} | {valid_c:>10.1f} | {t_c:>10.1f} | {mae_c:>8.4f}")

    r_b = results_blind
    valid_b = np.mean(r_b["n_valid"])
    t_b = np.mean(r_b["time"]) * 1000
    mae_b = np.mean([q for q in r_b["quality"] if q != float('inf')])
    print(f"  {'Cond (blind/context stats)':<30} | {'200':>6} | {valid_b:>10.1f} | {t_b:>10.1f} | {mae_b:>8.4f}")

    # Find equivalent budget: how many rejection samples to match conditioning?
    for budget in budgets:
        if np.mean(results_rejection[budget]["n_valid"]) >= valid_c:
            print(f"\n  Rejection sampling needs ~{budget} NFE to match conditioned ({valid_c:.0f} valid)")
            print(f"  Conditioning efficiency: {budget / 200:.0f}x fewer NFE for same yield")
            break

    # Quality comparison: are conditioned samples MORE realistic than rejection-sampled?
    print(f"\n  QUALITY (MAE to ground truth of valid samples):")
    print(f"  Conditioned:  {mae_c:.4f}")
    print(f"  Blind:        {mae_b:.4f}")
    for budget in budgets:
        mae_r = np.mean([q for q in results_rejection[budget]["quality"] if q != float('inf')])
        print(f"  Reject@{budget}: {mae_r:.4f}")

    print(f"\n  VERDICT:")
    # Does conditioning produce BETTER samples (lower MAE) than rejection?
    best_rej_mae = min(np.mean([q for q in results_rejection[b]["quality"] if q != float('inf')]) for b in budgets)
    if mae_c < best_rej_mae * 0.95:
        print(f"  ✓ Conditioned samples are MORE REALISTIC than rejection-sampled")
        print(f"    ({mae_c:.4f} MAE vs {best_rej_mae:.4f} best rejection)")
    else:
        print(f"  ✗ Conditioned samples are NOT more realistic than rejection-sampled")
        print(f"    ({mae_c:.4f} vs {best_rej_mae:.4f})")

    # Does blind conditioning work?
    if valid_b > np.mean(results_rejection[200]["n_valid"]) * 1.5:
        print(f"  ✓ Blind conditioning ({valid_b:.0f} valid) beats uncond@200 ({np.mean(results_rejection[200]['n_valid']):.0f})")
    else:
        print(f"  ✗ Blind conditioning ({valid_b:.0f}) ≈ uncond@200 ({np.mean(results_rejection[200]['n_valid']):.0f})")


if __name__ == "__main__":
    main()
