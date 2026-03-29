"""
sanity_check.py — Run this on every checkpoint before using it for feasibility experiments.
Prints PASS/FAIL. Do not use a checkpoint that fails.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features
from gluonts.dataset.repository.datasets import get_dataset
import argparse

TSFLOW_CRPS = {
    "electricity_nips": 0.045, "solar_nips": 0.341, "traffic_nips": 0.082,
    "exchange_rate_nips": 0.005, "m4_hourly": 0.029, "uber_tlc_hourly": 0.154,
    "wiki2000_nips": 0.207, "kdd_cup_2018_without_missing": 0.288,
}
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
LAG_MAP = {"H": 672, "B": 750, "1D": 750}

def check(dataset_name):
    cfg = CONFIGS[dataset_name]
    CTX_LEN, PRED_LEN, FREQ, N_LAGS = cfg["ctx"], cfg["pred"], cfg["freq"], 7
    CKPT = f"best_v4_{dataset_name}.pt"
    TSFLOW_REF = TSFLOW_CRPS[dataset_name]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_lag = LAG_MAP.get(FREQ, 672)

    if not os.path.exists(CKPT):
        print(f"{dataset_name}: SKIP — no checkpoint")
        return False

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE).eval()
    net.load_state_dict(ckpt['net_ema'])

    dataset = get_dataset(dataset_name)
    test_data = list(dataset.test)

    errors, diversities = [], []
    with torch.no_grad():
        for entry in test_data[:100]:
            target = torch.tensor(entry["target"], dtype=torch.float32, device=DEVICE)
            if len(target) < CTX_LEN + max_lag + PRED_LEN:
                continue
            ground_truth = target[-PRED_LEN:]
            past = target[-(CTX_LEN + max_lag + PRED_LEN):-PRED_LEN]
            ctx = past[-CTX_LEN:]
            loc = ctx.abs().mean().clamp(min=0.01)
            ctx_with_lags = extract_lag_features(past.unsqueeze(0), CTX_LEN, FREQ, N_LAGS).to(DEVICE) / loc
            preds = []
            for _ in range(20):
                z_1 = torch.randn(1, PRED_LEN, device=DEVICE)
                u = net(z_1, (torch.ones(1, device=DEVICE), torch.ones(1, device=DEVICE)), ctx_with_lags)
                preds.append((z_1 - u).squeeze(0) * loc)
            preds = torch.stack(preds)
            mae = (preds.mean(0) - ground_truth).abs().mean().item()
            scale = ground_truth.abs().mean().item()
            errors.append(mae / max(scale, 1e-6))
            diversities.append(preds.std(dim=0).mean().item())

    nd = np.mean(errors)
    div = np.mean(diversities)
    passed = True
    status = []
    if nd > 2 * TSFLOW_REF:
        status.append(f"FAIL: ND={nd:.4f} > 2×TSFlow={2*TSFLOW_REF:.4f}")
        passed = False
    if div < 1e-4:
        status.append(f"FAIL: diversity={div:.6f} collapsed")
        passed = False
    result = "PASS" if passed else "FAIL"
    print(f"{dataset_name}: {result} | ND={nd:.4f} | div={div:.4f} | epoch={ckpt.get('epoch','?')} | crps={ckpt.get('crps','?')}")
    for s in status:
        print(f"  {s}")
    return passed

if __name__ == "__main__":
    import tempfile
    try:
        import pykeops
        tmp = tempfile.mkdtemp(prefix="pykeops_build_")
        pykeops.set_build_folder(tmp)
        pykeops.clean_pykeops()
    except: pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(CONFIGS.keys()))
    args = parser.parse_args()

    passing = []
    for ds in args.datasets:
        if check(ds):
            passing.append(ds)
    print(f"\nPassing: {passing}")
