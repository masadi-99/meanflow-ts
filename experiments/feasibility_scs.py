"""
Experiment 1: Self-Consistency Score (SCS) Feasibility

Tests whether SCS correlates with actual sample quality.
If yes, MeanFlow can be its own verifier — no other 1-step method can do this.

Outputs:
- Spearman correlation between SCS and per-sample error
- CRPS improvement from SCS-based selection vs random
- Comparison of 1-step vs 2-step sample quality
"""
import os, sys, torch, numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features
from gluonts.dataset.repository.datasets import get_dataset

# ============== CONFIG — CHANGE THESE ==============
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
# ====================================================

N_CANDIDATES = 200
N_TEST_WINDOWS = 500
T_MIDS = [0.3, 0.5, 0.7]
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
    print(f"Loaded {CKPT} (epoch {ckpt.get('epoch','?')}, train CRPS={ckpt.get('crps','?')})")
    return net


@torch.no_grad()
def generate_onestep(net, z_1, ctx_with_lags):
    B = z_1.shape[0]
    t1 = torch.ones(B, device=DEVICE)
    h1 = torch.ones(B, device=DEVICE)
    u = net(z_1, (t1, h1), ctx_with_lags)
    return z_1 - u


@torch.no_grad()
def generate_twostep(net, z_1, ctx_with_lags, t_mid):
    B = z_1.shape[0]
    t_s1 = torch.ones(B, device=DEVICE)
    h_s1 = torch.full((B,), 1.0 - t_mid, device=DEVICE)
    u_s1 = net(z_1, (t_s1, h_s1), ctx_with_lags)
    z_mid = z_1 - (1.0 - t_mid) * u_s1
    t_s2 = torch.full((B,), t_mid, device=DEVICE)
    h_s2 = torch.full((B,), t_mid, device=DEVICE)
    u_s2 = net(z_mid, (t_s2, h_s2), ctx_with_lags)
    z_0 = z_mid - t_mid * u_s2
    return z_0


@torch.no_grad()
def compute_scs(net, z_1, z_0_onestep, ctx_with_lags, t_mids):
    scs_total = torch.zeros(z_1.shape[0], device=DEVICE)
    for t_mid in t_mids:
        z_0_twostep = generate_twostep(net, z_1, ctx_with_lags, t_mid)
        scs_total += ((z_0_onestep - z_0_twostep) ** 2).sum(dim=-1)
    return scs_total / len(t_mids)


def crps_from_samples(samples, truth):
    N = samples.shape[0]
    mae = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    n_pairs = min(N, 1000)
    idx1 = torch.randint(0, N, (n_pairs,))
    idx2 = torch.randint(0, N, (n_pairs,))
    pairwise = (samples[idx1] - samples[idx2]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model()
    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    corrs_spearman = []
    corrs_pearson = []
    crps_all_list = []
    crps_scs_best_list = []
    crps_scs_worst_list = []
    crps_twostep_list = []
    crps_random_half_list = []

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

        z_1 = torch.randn(N_CANDIDATES, PRED_LEN, device=DEVICE)
        scaled_gt = ground_truth / loc

        z_0_onestep = generate_onestep(net, z_1, ctx_batch)
        scs = compute_scs(net, z_1, z_0_onestep, ctx_batch, T_MIDS)
        errors = (z_0_onestep - scaled_gt.unsqueeze(0)).abs().sum(dim=-1)

        scs_np = scs.cpu().numpy()
        err_np = errors.cpu().numpy()
        if scs_np.std() > 1e-10 and err_np.std() > 1e-10:
            corrs_spearman.append(stats.spearmanr(scs_np, err_np).statistic)
            corrs_pearson.append(stats.pearsonr(scs_np, err_np).statistic)

        samples_1step = z_0_onestep * loc
        crps_all_list.append(crps_from_samples(samples_1step, ground_truth))

        n_select = N_CANDIDATES // 2
        best_idx = scs.argsort()[:n_select]
        crps_scs_best_list.append(crps_from_samples(samples_1step[best_idx], ground_truth))

        worst_idx = scs.argsort()[n_select:]
        crps_scs_worst_list.append(crps_from_samples(samples_1step[worst_idx], ground_truth))

        rand_idx = torch.randperm(N_CANDIDATES)[:n_select]
        crps_random_half_list.append(crps_from_samples(samples_1step[rand_idx], ground_truth))

        z_0_twostep = generate_twostep(net, z_1, ctx_batch, 0.5)
        crps_twostep_list.append(crps_from_samples(z_0_twostep * loc, ground_truth))

        n_eval += 1
        if n_eval % 50 == 0:
            print(f"  [{n_eval}/{N_TEST_WINDOWS}] "
                  f"spearman={np.mean(corrs_spearman):.3f}  "
                  f"crps_all={np.mean(crps_all_list):.5f}  "
                  f"crps_scs_best={np.mean(crps_scs_best_list):.5f}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 1: SELF-CONSISTENCY SCORE — {DATASET}")
    print("=" * 70)
    print(f"Windows: {n_eval} | Candidates/window: {N_CANDIDATES} | t_mids: {T_MIDS}")

    sp = np.mean(corrs_spearman)
    print(f"\n  Spearman(SCS, error):  {sp:.4f}  (std {np.std(corrs_spearman):.4f})")
    print(f"  Pearson(SCS, error):   {np.mean(corrs_pearson):.4f}")
    print(f"    → Positive = SCS predicts quality. Need > 0.1 useful, > 0.3 strong.")

    c_all = np.mean(crps_all_list)
    c_best = np.mean(crps_scs_best_list)
    c_worst = np.mean(crps_scs_worst_list)
    c_rand = np.mean(crps_random_half_list)
    c_2step = np.mean(crps_twostep_list)

    print(f"\n  CRPS (all {N_CANDIDATES}):           {c_all:.6f}")
    print(f"  CRPS (random 50%):            {c_rand:.6f}")
    print(f"  CRPS (SCS-best 50%):          {c_best:.6f}")
    print(f"  CRPS (SCS-worst 50%):         {c_worst:.6f}")
    print(f"  CRPS (2-step samples):        {c_2step:.6f}")

    imp_vs_rand = (c_rand - c_best) / c_rand * 100
    imp_vs_all  = (c_all  - c_best) / c_all  * 100
    print(f"\n  SCS-select vs random-half:    {imp_vs_rand:+.2f}%")
    print(f"  SCS-select vs all:            {imp_vs_all:+.2f}%")
    print(f"  SCS-best vs SCS-worst gap:    {c_best - c_worst:.6f}  (negative = SCS works)")
    print(f"  2-step vs 1-step:             {(c_all - c_2step)/c_all*100:+.2f}%")

    print("\n  VERDICT:")
    if sp > 0.2 and imp_vs_rand > 1.5:
        print("  ✓ SCS IS A VIABLE VERIFIER. Proceed to Experiments 3 and 4.")
    elif sp > 0.08 or imp_vs_rand > 0.5:
        print("  ~ MARGINAL. Potentially useful. Run on more datasets to confirm.")
    else:
        print("  ✗ SCS does not predict quality on this dataset.")

    np.savez(f"scs_feasibility_{DATASET}.npz",
             corrs_spearman=corrs_spearman, corrs_pearson=corrs_pearson,
             crps_all=crps_all_list, crps_scs_best=crps_scs_best_list,
             crps_scs_worst=crps_scs_worst_list, crps_random_half=crps_random_half_list,
             crps_twostep=crps_twostep_list)
    print(f"\n  Saved: scs_feasibility_{DATASET}.npz")


if __name__ == "__main__":
    main()
