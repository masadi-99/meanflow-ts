"""
Round 5: Scenario Generation for Decision Support

Key experiment: Generate futures that satisfy business-relevant constraints.
Show that the stat-conditioned model produces realistic constrained scenarios
that an unconditional model cannot.

Scenarios:
1. "High peak" — futures where max > 90th percentile of historical peaks
2. "Low volatility" — futures where std < 10th percentile of historical stds
3. "Late peak" — futures where argmax > 0.7 (peak in last 30% of horizon)
4. "Demand spike" — futures where max/mean > 2.0

For each scenario:
- Generate with stat conditioning (targeted)
- Generate unconditionally and filter (rejection sampling)
- Compare: how many samples satisfy the constraint? How realistic are they?
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


def main():
    net = load_model()
    if net is None:
        print(f"No checkpoint for {DATASET}")
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    # First, compute historical stat distributions from training data
    print("Computing historical stat distributions...")
    all_stats = []
    for entry in dataset.train:
        ts = np.array(entry["target"], dtype=np.float32)
        for i in range(len(ts) // PRED_LEN):
            window = ts[i*PRED_LEN:(i+1)*PRED_LEN]
            loc = max(np.abs(window).mean(), 0.01)
            scaled = window / loc
            all_stats.append([scaled.mean(), scaled.max(), scaled.min(), scaled.std(),
                              np.argmax(scaled) / PRED_LEN, scaled.sum() / PRED_LEN])
    all_stats = np.array(all_stats)
    print(f"  Collected {len(all_stats)} historical windows")

    # Define scenarios based on percentiles
    scenarios = {
        "high_peak": {"stat_idx": 1, "target_percentile": 90, "direction": "above",
                      "desc": "Peak value > 90th pctile"},
        "low_volatility": {"stat_idx": 3, "target_percentile": 10, "direction": "below",
                           "desc": "Volatility < 10th pctile"},
        "late_peak": {"stat_idx": 4, "target_value": 0.75, "direction": "above",
                      "desc": "Peak timing in last 25%"},
    }

    # Compute threshold values
    for name, sc in scenarios.items():
        idx = sc["stat_idx"]
        if "target_percentile" in sc:
            sc["threshold"] = np.percentile(all_stats[:, idx], sc["target_percentile"])
        else:
            sc["threshold"] = sc["target_value"]
        print(f"  {name}: stat[{idx}] {sc['direction']} {sc['threshold']:.3f}")

    # Evaluate each scenario on test windows
    N_EVAL = 100
    N_GEN = 200  # samples per window

    print(f"\n{'='*80}")
    print(f"SCENARIO GENERATION — {DATASET}")
    print(f"{'='*80}")

    for sc_name, sc in scenarios.items():
        print(f"\n--- Scenario: {sc_name} ({sc['desc']}) ---")
        stat_idx = sc["stat_idx"]
        threshold = sc["threshold"]
        direction = sc["direction"]

        hit_rates_cond = []
        hit_rates_uncond = []
        realism_cond = []   # MAE of other stats vs oracle
        realism_uncond = []
        nfe_cond = []
        nfe_uncond = []

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

            # === Method 1: Stat-conditioned generation ===
            # Modify the target stat to be in the scenario region
            target_stats = oracle_stats.clone()
            if direction == "above":
                target_stats[0, stat_idx] = max(threshold * 1.2, target_stats[0, stat_idx])
            else:
                target_stats[0, stat_idx] = min(threshold * 0.8, target_stats[0, stat_idx])

            preds_cond = generate(net, N_GEN, ctx_lags, target_stats)
            stats_cond = extract_stats(preds_cond)

            # Count hits
            if direction == "above":
                hits_c = (stats_cond[:, stat_idx] >= threshold).float().mean().item()
            else:
                hits_c = (stats_cond[:, stat_idx] <= threshold).float().mean().item()
            hit_rates_cond.append(hits_c)
            nfe_cond.append(N_GEN)

            # === Method 2: Unconditional + rejection sampling ===
            preds_uncond = generate(net, N_GEN, ctx_lags, None)
            stats_uncond = extract_stats(preds_uncond)

            if direction == "above":
                hits_u = (stats_uncond[:, stat_idx] >= threshold).float().mean().item()
            else:
                hits_u = (stats_uncond[:, stat_idx] <= threshold).float().mean().item()
            hit_rates_uncond.append(hits_u)
            nfe_uncond.append(N_GEN)

            n_eval += 1

        avg_hit_cond = np.mean(hit_rates_cond)
        avg_hit_uncond = np.mean(hit_rates_uncond)
        improvement = avg_hit_cond / max(avg_hit_uncond, 0.001)

        print(f"  Hit rate (conditioned):   {avg_hit_cond:.3f} ({avg_hit_cond*100:.1f}%)")
        print(f"  Hit rate (unconditional): {avg_hit_uncond:.3f} ({avg_hit_uncond*100:.1f}%)")
        print(f"  Improvement factor:       {improvement:.1f}x")
        print(f"  NFE per scenario:         {N_GEN} (both methods)")

    print(f"\n{'='*80}")
    print(f"VERDICT:")
    print(f"  If conditioned hit rate >> unconditional hit rate:")
    print(f"  → The model enables targeted scenario generation.")
    print(f"  → Users can request specific future properties and get them.")
    print(f"  → This is a NEW CAPABILITY, not a CRPS improvement.")


if __name__ == "__main__":
    main()
