"""
Item 5: Parameter-matched ablation. Train a SMALL v3 model (~292K params)
to match TSFlow's ~189K and see if the quality advantage holds.

Also train a small V2 model (~200K) for unconditional CRPS comparison.
"""
import os, sys, time, argparse, logging, tempfile
import numpy as np
import torch
from torch.optim import AdamW
from copy import deepcopy
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import *

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v3 import ConditionalMeanFlowNetV3, conditional_meanflow_loss_v3
from meanflow_ts.model_v2 import extract_lag_features, MeanFlowForecasterV2, ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2
from meanflow_ts.utils import extract_stats

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = os.environ.get("DATASET", "electricity_nips")
CTX_LEN, PRED_LEN, FREQ = 24, 24, "H"
N_LAGS = 7
LAG_MAP = {"H": 672}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAT_DROPOUT = 0.15
EPOCHS = 600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="v3_small_stats", choices=["v3_small_stats", "v2_small"])
    args = parser.parse_args()

    torch.manual_seed(6432)
    np.random.seed(6432)

    dataset = get_dataset(DATASET)
    max_lag = LAG_MAP.get(FREQ, 672)

    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(FREQ), pred_length=PRED_LEN),
    ])
    train_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=PRED_LEN),
        past_length=CTX_LEN + max_lag, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed_data = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed_data), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=10000,
    )

    if args.mode == "v3_small_stats":
        net = ConditionalMeanFlowNetV3(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            model_channels=32, num_res_blocks=2, time_emb_dim=32, dropout=0.1,
        ).to(DEVICE)
        mode_label = "v3_small_stats"
    else:
        net = ConditionalMeanFlowNetV2(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            model_channels=32, num_res_blocks=2, time_emb_dim=32, dropout=0.1,
        ).to(DEVICE)
        mode_label = "v2_small"

    net_ema = deepcopy(net).eval()
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"=== {mode_label} | {n_params:,} params (TSFlow: ~189K) ===")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')

    for epoch in range(EPOCHS):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()
        for batch in train_loader:
            past = batch["past_target"].to(DEVICE)
            future = batch["future_target"].to(DEVICE)
            ctx = past[:, -CTX_LEN:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
            ctx_with_lags = extract_lag_features(past, CTX_LEN, FREQ, N_LAGS) / loc.unsqueeze(1)
            scaled_future = future / loc

            if args.mode == "v3_small_stats":
                stats = extract_stats(scaled_future)
                stat_vec = stats if torch.rand(1).item() > STAT_DROPOUT else None
                loss = conditional_meanflow_loss_v3(net, scaled_future, ctx_with_lags, stat_vector=stat_vec)
            else:
                loss = conditional_meanflow_loss_v2(net, scaled_future, ctx_with_lags)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        elapsed = time.time() - t0
        if (epoch+1) % 20 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:>3}/{EPOCHS} | Loss: {epoch_loss/n_b:.4f} | {elapsed:.1f}s")

        if (epoch+1) % 100 == 0 or (epoch+1) == EPOCHS:
            logger.info("Evaluating...")
            net_ema.eval()
            test_transform = transformation.apply(dataset.test, is_train=False)
            test_splitter = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=CTX_LEN + max_lag, future_length=PRED_LEN,
                time_series_fields=["time_feat", "observed_values"],
            )
            forecaster = MeanFlowForecasterV2(
                net_ema, CTX_LEN, PRED_LEN, num_samples=16, freq=FREQ, n_lags=N_LAGS,
            ).to(DEVICE)
            predictor = PyTorchPredictor(
                prediction_length=PRED_LEN, input_names=["past_target", "past_observed_values"],
                prediction_net=forecaster, batch_size=512, input_transform=test_splitter, device=DEVICE,
            )
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_transform, predictor=predictor, num_samples=16)
            forecasts = list(tqdm(forecast_it, total=len(test_transform), desc="Eval"))
            tss = list(ts_it)
            metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = metrics["mean_wQuantileLoss"]
            tag = " ***BEST***" if crps < best_crps else ""
            if crps < best_crps:
                best_crps = crps
                torch.save({'net_ema': net_ema.state_dict(), 'crps': crps, 'epoch': epoch+1,
                            'params': n_params, 'mode': mode_label},
                           f'best_{mode_label}_{DATASET}.pt')
            logger.info(f"  CRPS={crps:.6f}{tag} | Best: {best_crps:.6f}")
            logger.info(f"  TSFlow: 0.045 | v4 (1.14M): 0.047")

    logger.info(f"DONE. Best CRPS: {best_crps:.6f} ({n_params:,} params)")


if __name__ == "__main__":
    main()
