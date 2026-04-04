"""
Self-Refinement via Multi-Resolution Cascade.

The key idea: use the model's OWN 1-step output as coarse conditioning for a second pass.
No oracle information needed. Total cost: 2 NFE (still 16x cheaper than TSFlow's 32).

Pipeline:
1. Generate N unconditional 1-step forecasts → take median as coarse
2. Downsample the coarse to factor r
3. Generate N new samples conditioned on the coarse
4. Compare CRPS: unconditional vs self-refined

Also test iterative refinement: refine → downsample → refine again (3 NFE total).
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
    "uber_tlc_hourly":    {"freq": "H", "ctx": 24, "pred": 24},
    "wiki2000_nips":      {"freq": "1D", "ctx": 30, "pred": 30},
    "kdd_cup_2018_without_missing": {"freq": "H", "ctx": 48, "pred": 48},
}
cfg = CONFIGS[DATASET]
CTX_LEN, PRED_LEN, FREQ = cfg["ctx"], cfg["pred"], cfg["freq"]
N_LAGS = 7
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 50
N_WINDOWS = 500


def load_model(mode="multires"):
    ckpt_path = f"best_v3_{mode}_{DATASET}.pt"
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint: {ckpt_path}")
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
    n_p = min(N, 1000)
    idx1 = torch.randint(0, N, (n_p,))
    idx2 = torch.randint(0, N, (n_p,))
    pairwise = (samples[idx1] - samples[idx2]).abs().mean(dim=0)
    return (mae - 0.5 * pairwise).mean().item()


def main():
    net = load_model("multires")
    if net is None:
        net = load_model("both")
    if net is None:
        print("No multires or both checkpoint found!")
        return

    dataset = get_dataset(DATASET)
    test_data = list(dataset.test)
    max_lag = LAG_MAP.get(FREQ, 672)

    # Valid downsample factors
    valid_r = [r for r in [2, 4] if PRED_LEN % r == 0]
    if not valid_r:
        print(f"No valid downsample factors for pred_len={PRED_LEN}")
        return

    crps_baseline = []       # unconditional 1-step
    crps_oracle = {r: [] for r in valid_r}   # oracle coarse (ceiling)
    crps_self_refine = {r: [] for r in valid_r}  # self-refinement (our method!)
    crps_iter_refine = {r: [] for r in valid_r}  # iterative self-refinement (2 rounds)

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
        ctx_batch = ctx_lags.expand(N_SAMPLES, -1, -1)
        scaled_gt = gt / loc

        # === Baseline: unconditional 1-step ===
        z_1 = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)
        pred_uncond = generate(net, z_1, ctx_batch) * loc
        crps_baseline.append(crps_from_samples(pred_uncond, gt))

        for r in valid_r:
            # === Oracle: coarse from ground truth ===
            coarse_oracle = downsample(scaled_gt, r)
            coarse_up_oracle = upsample(coarse_oracle, r, PRED_LEN).unsqueeze(0).expand(N_SAMPLES, -1)
            z_1_ref = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)
            residual_oracle = generate(net, z_1_ref, ctx_batch, coarse_up_oracle)
            pred_oracle = (residual_oracle + coarse_up_oracle) * loc
            crps_oracle[r].append(crps_from_samples(pred_oracle, gt))

            # === Self-refinement: coarse from model's OWN prediction ===
            # Use median of unconditional predictions as coarse
            model_coarse = pred_uncond.median(dim=0).values / loc  # (pred_len,)
            coarse_self = downsample(model_coarse, r)
            coarse_up_self = upsample(coarse_self, r, PRED_LEN).unsqueeze(0).expand(N_SAMPLES, -1)
            z_1_self = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)
            residual_self = generate(net, z_1_self, ctx_batch, coarse_up_self)
            pred_self = (residual_self + coarse_up_self) * loc
            crps_self_refine[r].append(crps_from_samples(pred_self, gt))

            # === Iterative self-refinement: refine → downsample → refine ===
            model_coarse_2 = pred_self.median(dim=0).values / loc
            coarse_iter = downsample(model_coarse_2, r)
            coarse_up_iter = upsample(coarse_iter, r, PRED_LEN).unsqueeze(0).expand(N_SAMPLES, -1)
            z_1_iter = torch.randn(N_SAMPLES, PRED_LEN, device=DEVICE)
            residual_iter = generate(net, z_1_iter, ctx_batch, coarse_up_iter)
            pred_iter = (residual_iter + coarse_up_iter) * loc
            crps_iter_refine[r].append(crps_from_samples(pred_iter, gt))

        n_eval += 1
        if n_eval % 100 == 0:
            c_base = np.mean(crps_baseline)
            c_self = np.mean(crps_self_refine[valid_r[0]])
            print(f"  [{n_eval}/{N_WINDOWS}] baseline={c_base:.5f} self-refine(r={valid_r[0]})={c_self:.5f} "
                  f"({(c_base-c_self)/c_base*100:+.1f}%)")

    # === REPORT ===
    print(f"\n{'='*80}")
    print(f"SELF-REFINEMENT EXPERIMENT — {DATASET}")
    print(f"{'='*80}")
    print(f"Windows: {n_eval} | Samples/window: {N_SAMPLES}")

    c_base = np.mean(crps_baseline)
    print(f"\n  {'Method':<35} | {'CRPS':>10} | {'vs baseline':>12} | {'NFE':>5}")
    print(f"  {'-'*70}")
    print(f"  {'Baseline (1-step uncond)':<35} | {c_base:>10.5f} | {'---':>12} | {'1':>5}")

    for r in valid_r:
        c_oracle = np.mean(crps_oracle[r])
        c_self = np.mean(crps_self_refine[r])
        c_iter = np.mean(crps_iter_refine[r])

        imp_oracle = (c_base - c_oracle) / c_base * 100
        imp_self = (c_base - c_self) / c_base * 100
        imp_iter = (c_base - c_iter) / c_base * 100

        print(f"  {'Oracle coarse r=' + str(r):<35} | {c_oracle:>10.5f} | {imp_oracle:>+11.2f}% | {'2':>5}")
        print(f"  {'Self-refine r=' + str(r):<35} | {c_self:>10.5f} | {imp_self:>+11.2f}% | {'2':>5}")
        print(f"  {'Iter self-refine r=' + str(r):<35} | {c_iter:>10.5f} | {imp_iter:>+11.2f}% | {'3':>5}")

    # Key comparison: does self-refinement work?
    best_r = min(valid_r, key=lambda r: np.mean(crps_self_refine[r]))
    c_self_best = np.mean(crps_self_refine[best_r])
    imp_best = (c_base - c_self_best) / c_base * 100

    print(f"\n  VERDICT:")
    if imp_best > 5:
        print(f"  ✓ SELF-REFINEMENT WORKS! r={best_r} → {imp_best:.1f}% improvement over baseline.")
        print(f"    Cost: 2 NFE (baseline is 1 NFE, TSFlow is 32 NFE).")
        print(f"    This is a GENUINE contribution: the model improves its own output")
        print(f"    via multi-resolution self-conditioning. No oracle needed.")
    elif imp_best > 1:
        print(f"  ~ MARGINAL. r={best_r} → {imp_best:.1f}% improvement.")
        print(f"    May be statistically significant with enough windows.")
    else:
        print(f"  ✗ SELF-REFINEMENT DOESN'T HELP. r={best_r} → {imp_best:.1f}%.")

    # Also check: does iterative help over single refinement?
    c_iter_best = np.mean(crps_iter_refine[best_r])
    iter_vs_self = (c_self_best - c_iter_best) / c_self_best * 100
    print(f"\n  Iterative vs single refinement: {iter_vs_self:+.2f}% (positive = iterative better)")

    np.savez(f"self_refinement_{DATASET}.npz",
             crps_baseline=crps_baseline,
             **{f"crps_oracle_{r}": crps_oracle[r] for r in valid_r},
             **{f"crps_self_{r}": crps_self_refine[r] for r in valid_r},
             **{f"crps_iter_{r}": crps_iter_refine[r] for r in valid_r})
    print(f"\n  Saved: self_refinement_{DATASET}.npz")


if __name__ == "__main__":
    main()
