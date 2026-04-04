"""
Round 8: Fully Prospective Scenario Evaluation (No Oracle Information)

Addresses Round 7 reviewer's key concern: the previous evaluation leaks
oracle info by modifying oracle stats. This version is fully prospective:

1. Define scenarios from TRAINING distribution (no future information)
2. Generate conditioned and rejection-sampled scenarios
3. Evaluate against whatever future actually occurs
4. Add distributional quality metrics (autocorrelation, energy score)

Protocol:
- Scenario spec: "peak > 90th pctile of TRAINING peaks" (fixed threshold, no oracle)
- Conditioning: target stat vector with only the scenario-relevant dimension set,
  all other dimensions set to TRAINING DISTRIBUTION MEANS (not oracle)
- Rejection: generate N samples, filter by same criterion
- Both evaluated against actual future on MAE, autocorrelation error, energy score
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


def autocorrelation_error(samples, truth, max_lag=5):
    """MAE between autocorrelation of samples and truth at lags 1..max_lag."""
    truth_np = truth.cpu().numpy()
    acf_truth = [np.corrcoef(truth_np[:-k], truth_np[k:])[0, 1] for k in range(1, max_lag + 1)]

    acf_errors = []
    for s in samples:
        s_np = s.cpu().numpy()
        acf_s = [np.corrcoef(s_np[:-k], s_np[k:])[0, 1] if len(s_np) > k else 0 for k in range(1, max_lag + 1)]
        acf_errors.append(np.mean(np.abs(np.array(acf_s) - np.array(acf_truth))))
    return np.mean(acf_errors)


def energy_score(samples, truth):
    """Energy score: E||X-y|| - 0.5*E||X-X'||."""
    N = samples.shape[0]
    mae = (samples - truth.unsqueeze(0)).norm(dim=1).mean().item()
    n_p = min(N, 500)
    idx1 = torch.randint(0, N, (n_p,))
    idx2 = torch.randint(0, N, (n_p,))
    spread = (samples[idx1] - samples[idx2]).norm(dim=1).mean().item()
    return mae - 0.5 * spread


def main():
    net = load_model()
    if net is None:
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    # === Step 1: Compute TRAINING distribution statistics ===
    print("Computing training stat distributions (NO test/future info)...")
    train_stats = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        for i in range(len(ts) // PRED_LEN):
            w = ts[i * PRED_LEN:(i + 1) * PRED_LEN]
            loc = max(np.abs(w).mean(), 0.01)
            scaled = w / loc
            train_stats.append([scaled.mean(), scaled.max(), scaled.min(), scaled.std(),
                                np.argmax(scaled) / PRED_LEN, scaled.sum() / PRED_LEN])
    train_stats = np.array(train_stats)
    train_mean_stats = torch.tensor(train_stats.mean(axis=0), dtype=torch.float32, device=DEVICE)
    print(f"  Training windows: {len(train_stats)}")
    print(f"  Training stat means: {train_mean_stats.cpu().numpy().round(3)}")

    # === Step 2: Define scenario thresholds from TRAINING data ===
    threshold_high_peak = np.percentile(train_stats[:, 1], 90)
    threshold_low_vol = np.percentile(train_stats[:, 3], 10)
    print(f"  High peak threshold (90th pctile of training): {threshold_high_peak:.3f}")
    print(f"  Low volatility threshold (10th pctile of training): {threshold_low_vol:.3f}")

    N_EVAL = 200
    N_GEN_COND = 200
    budgets_rej = [200, 2000, 20000]

    scenarios = [
        ("high_peak", 1, threshold_high_peak, "above", threshold_high_peak * 1.2),
        ("low_volatility", 3, threshold_low_vol, "below", threshold_low_vol * 0.8),
    ]

    for sc_name, stat_idx, threshold, direction, target_val in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {sc_name} — stat[{stat_idx}] {direction} {threshold:.3f}")
        print(f"{'='*80}")

        results = {}
        for method in ["cond_blind"] + [f"reject_{b}" for b in budgets_rej]:
            results[method] = {"n_valid": [], "mae": [], "acf_err": [], "energy": []}

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

            # === BLIND conditioning: use TRAINING MEANS for all stats ===
            # Only set the target dimension to the scenario value
            blind_stats = train_mean_stats.unsqueeze(0).clone()
            blind_stats[0, stat_idx] = target_val

            preds_cond = generate(net, N_GEN_COND, ctx_lags, blind_stats)
            stats_cond = extract_stats(preds_cond)

            if direction == "above":
                valid_mask = stats_cond[:, stat_idx] >= threshold
            else:
                valid_mask = stats_cond[:, stat_idx] <= threshold

            n_valid = valid_mask.sum().item()
            results["cond_blind"]["n_valid"].append(n_valid)

            if n_valid > 0:
                valid_samples = preds_cond[valid_mask]
                results["cond_blind"]["mae"].append(
                    (valid_samples.mean(0) - scaled_gt).abs().mean().item())
                results["cond_blind"]["acf_err"].append(
                    autocorrelation_error(valid_samples[:20], scaled_gt))
                results["cond_blind"]["energy"].append(
                    energy_score(valid_samples, scaled_gt))

            # === Rejection sampling at various budgets ===
            for budget in budgets_rej:
                preds_rej = generate(net, budget, ctx_lags, None)
                stats_rej = extract_stats(preds_rej)

                if direction == "above":
                    valid_rej = stats_rej[:, stat_idx] >= threshold
                else:
                    valid_rej = stats_rej[:, stat_idx] <= threshold

                n_valid_r = valid_rej.sum().item()
                results[f"reject_{budget}"]["n_valid"].append(n_valid_r)

                if n_valid_r > 0:
                    valid_r = preds_rej[valid_rej]
                    results[f"reject_{budget}"]["mae"].append(
                        (valid_r.mean(0) - scaled_gt).abs().mean().item())
                    results[f"reject_{budget}"]["acf_err"].append(
                        autocorrelation_error(valid_r[:20], scaled_gt))
                    results[f"reject_{budget}"]["energy"].append(
                        energy_score(valid_r, scaled_gt))

            n_eval += 1
            if n_eval % 50 == 0:
                cv = np.mean(results["cond_blind"]["n_valid"])
                rv = np.mean(results["reject_2000"]["n_valid"])
                print(f"  [{n_eval}/{N_EVAL}] cond_valid={cv:.0f} rej@2K_valid={rv:.0f}")

        # === REPORT ===
        print(f"\n  {'Method':<25} | {'Valid/200':>10} | {'MAE':>8} | {'ACF err':>8} | {'Energy':>8}")
        print(f"  {'-'*70}")
        for method in ["cond_blind"] + [f"reject_{b}" for b in budgets_rej]:
            r = results[method]
            valid = np.mean(r["n_valid"])
            mae = np.mean(r["mae"]) if r["mae"] else float('inf')
            acf = np.mean(r["acf_err"]) if r["acf_err"] else float('inf')
            eng = np.mean(r["energy"]) if r["energy"] else float('inf')
            label = method.replace("_", " ").replace("reject", "Reject@")
            print(f"  {label:<25} | {valid:>10.1f} | {mae:>8.4f} | {acf:>8.4f} | {eng:>8.4f}")

    print(f"\n  NOTE: ALL comparisons are FULLY PROSPECTIVE.")
    print(f"  - Thresholds from TRAINING data only")
    print(f"  - Conditioning uses TRAINING MEAN stats (not oracle)")
    print(f"  - No future information used in scenario specification")


if __name__ == "__main__":
    main()
