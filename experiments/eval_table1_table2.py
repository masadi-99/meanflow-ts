"""
Evaluate unconditional MeanFlow for Table 1 (2-Wasserstein) and Table 2 (LPS).

Table 1: 2-Wasserstein distance between real and generated samples.
Table 2: Linear Predictive Score — train linear model on synthetic, test on real.

Usage: python eval_table1_table2.py <dataset_name>
"""
import os, sys, argparse, tempfile, json, logging
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from gluonts.dataset.repository.datasets import get_dataset

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model import UnconditionalMeanFlowNet, meanflow_sample

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {
    "electricity_nips": {"freq": "H", "seq_len": 24},
    "solar_nips":       {"freq": "H", "seq_len": 24},
    "traffic_nips":     {"freq": "H", "seq_len": 24},
    "exchange_rate_nips": {"freq": "B", "seq_len": 30},
    "m4_hourly":        {"freq": "H", "seq_len": 48},
}

# TSFlow paper Table 1 results (OU kernel, NFE=4)
TSFLOW_W2 = {
    "electricity_nips": 2.090, "exchange_rate_nips": 0.029,
    "solar_nips": 4.564, "traffic_nips": 7.283, "m4_hourly": 6.509,
}

# TSFlow paper Table 2 results (OU kernel)
TSFLOW_LPS = {
    "electricity_nips": 0.096, "exchange_rate_nips": 0.011,
    "solar_nips": 0.616, "traffic_nips": 0.237, "m4_hourly": 0.032,
}


def load_windows(dataset_name, seq_len, split="train"):
    """Extract non-overlapping windows."""
    dataset = get_dataset(dataset_name)
    data = dataset.train if split == "train" else dataset.test
    windows = []
    for entry in data:
        ts = np.array(entry["target"], dtype=np.float32)
        n = len(ts) // seq_len
        for i in range(n):
            windows.append(ts[i * seq_len : (i + 1) * seq_len])
    return np.stack(windows)


def compute_w2_distance(real_windows, gen_windows, n_projections=1000):
    """
    Sliced 2-Wasserstein distance between two sets of time series.
    Projects to 1D via random directions, computes 1D Wasserstein, averages.
    """
    d = real_windows.shape[1]
    w2_vals = []
    for _ in range(n_projections):
        direction = np.random.randn(d)
        direction /= np.linalg.norm(direction)
        proj_real = real_windows @ direction
        proj_gen = gen_windows @ direction
        w2 = wasserstein_distance(proj_real, proj_gen)
        w2_vals.append(w2 ** 2)
    return np.sqrt(np.mean(w2_vals))


def compute_lps(real_train, synthetic, real_test, pred_steps=1):
    """
    Linear Predictive Score.
    Train Ridge regression on synthetic data to predict next step from history.
    Evaluate on real test data. Lower MSE = better generation quality.
    Returns MSE (lower is better, following TSFlow's convention of reporting this as LPS).
    """
    seq_len = synthetic.shape[1]
    ctx_len = seq_len - pred_steps

    # Build features/targets from synthetic
    X_syn = synthetic[:, :ctx_len]
    y_syn = synthetic[:, ctx_len:]

    # Build features/targets from real test
    X_test = real_test[:, :ctx_len]
    y_test = real_test[:, ctx_len:]

    model = Ridge(alpha=1.0)
    model.fit(X_syn, y_syn)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--n-gen", type=int, default=5000, help="Number of samples to generate")
    args = parser.parse_args()

    name = args.dataset
    cfg = CONFIGS[name]
    seq_len = cfg["seq_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"uncond_meanflow_{name}.pt"
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}. Run train_unconditional.py first.")
        return

    logger.info(f"=== Evaluating {name} ===")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = UnconditionalMeanFlowNet(
        seq_len=seq_len, model_channels=128, num_res_blocks=4,
    ).to(device)
    net.load_state_dict(ckpt['net_ema'])
    net.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_scale'])
    scaler.var_ = scaler.scale_ ** 2

    # Load real data
    real_train = load_windows(name, seq_len, "train")
    real_test = load_windows(name, seq_len, "test")

    # Normalize real data with same scaler
    real_train_scaled = scaler.transform(real_train.reshape(-1, 1)).reshape(real_train.shape)
    real_test_scaled = scaler.transform(real_test.reshape(-1, 1)).reshape(real_test.shape)

    # Generate synthetic samples (one-step MeanFlow)
    logger.info(f"Generating {args.n_gen} synthetic samples...")
    n_batches = (args.n_gen + 511) // 512
    gen_all = []
    with torch.no_grad():
        for _ in range(n_batches):
            bs = min(512, args.n_gen - len(gen_all) * 0 if gen_all else 512)
            samples = meanflow_sample(net, (512, seq_len), device)
            gen_all.append(samples.cpu().numpy())
    gen_scaled = np.concatenate(gen_all)[:args.n_gen]

    # Inverse transform for real-scale comparisons
    gen_real_scale = scaler.inverse_transform(gen_scaled.reshape(-1, 1)).reshape(gen_scaled.shape)

    logger.info(f"Generated shape: {gen_scaled.shape}")
    logger.info(f"Real train: {real_train.shape}, Real test: {real_test.shape}")
    logger.info(f"Gen stats: mean={gen_scaled.mean():.3f} std={gen_scaled.std():.3f}")
    logger.info(f"Real stats: mean={real_train_scaled.mean():.3f} std={real_train_scaled.std():.3f}")

    # === Table 1: 2-Wasserstein distance ===
    logger.info("\nComputing 2-Wasserstein distance...")
    # Use scaled data (same as TSFlow)
    w2 = compute_w2_distance(real_train_scaled[:5000], gen_scaled[:5000])
    tsflow_w2 = TSFLOW_W2.get(name, "?")
    logger.info(f"  W2 distance: {w2:.4f} (TSFlow: {tsflow_w2})")

    # === Table 2: Linear Predictive Score ===
    logger.info("\nComputing Linear Predictive Score...")
    lps = compute_lps(real_train_scaled, gen_scaled, real_test_scaled)
    tsflow_lps = TSFLOW_LPS.get(name, "?")
    logger.info(f"  LPS (MSE): {lps:.6f} (TSFlow: {tsflow_lps})")

    # Save results
    results = {
        "dataset": name,
        "w2_distance": float(w2),
        "lps_mse": float(lps),
        "tsflow_w2": tsflow_w2,
        "tsflow_lps": tsflow_lps,
        "n_generated": args.n_gen,
        "gen_mean": float(gen_scaled.mean()),
        "gen_std": float(gen_scaled.std()),
    }
    outfile = f"table12_{name}.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {outfile}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} | {'MeanFlow':>12} | {'TSFlow (OU)':>12}")
    print(f"{'-'*55}")
    print(f"{'W2 distance (Table 1)':<25} | {w2:>12.4f} | {tsflow_w2:>12}")
    print(f"{'LPS / MSE (Table 2)':<25} | {lps:>12.6f} | {tsflow_lps:>12}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
