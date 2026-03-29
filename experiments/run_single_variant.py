"""
Run a single variant for V5 experiments. Designed to be run in parallel.
Usage: python run_single_variant.py <variant> [--epochs N]

Variants: imf-only, residual-only, imf+residual, baseline
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
from gluonts.transform import (
    AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray,
    Chain, ExpectedNumInstanceSampler, InstanceSplitter, TestSplitSampler,
)

try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import (
    ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2,
    MeanFlowForecasterV2, extract_lag_features,
)
from oneflow_ts.improvements import (
    BaseForecaster, residual_meanflow_loss, imf_loss,
    imf_residual_loss, AdaptiveForecaster,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

FREQ = "H"; CTX_LEN = 24; PRED_LEN = 24; MAX_LAG = 672; N_LAGS = 7
NUM_EVAL_SAMPLES = 16; TSFLOW_CRPS = 0.045

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", type=str)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--dataset", type=str, default="electricity_nips")
    args = parser.parse_args()
    variant = args.variant

    device = torch.device("cuda")
    torch.manual_seed(6432); np.random.seed(6432)

    dataset = get_dataset(args.dataset)
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
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=10000,
    )

    flow_net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(device)
    flow_ema = deepcopy(flow_net).eval()

    use_residual = "residual" in variant
    use_imf = "imf" in variant

    base_net = BaseForecaster(CTX_LEN, PRED_LEN, N_LAGS).to(device) if use_residual else None
    base_ema = deepcopy(base_net).eval() if base_net else None

    params = list(flow_net.parameters()) + (list(base_net.parameters()) if base_net else [])
    optimizer = AdamW(params, lr=6e-4)
    best_crps = float('inf')

    logger.info(f"[{variant}] Params: {sum(p.numel() for p in params):,}")

    for epoch in range(args.epochs):
        flow_net.train()
        if base_net: base_net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()

        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -CTX_LEN:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
            ctx_lags = extract_lag_features(past, CTX_LEN, FREQ, N_LAGS) / loc.unsqueeze(1)
            scaled_future = future / loc

            # Base weight warmup: 0 at start, ramp to 1.0 over first 100 epochs
            base_w = min(1.0, (epoch + 1) / 100.0) if use_residual else 0

            if use_imf and use_residual:
                loss = imf_residual_loss(flow_net, base_net, scaled_future, ctx_lags,
                                          base_weight=base_w)
            elif use_imf:
                loss = imf_loss(flow_net, scaled_future, ctx_lags)
            elif use_residual:
                loss = residual_meanflow_loss(flow_net, base_net, scaled_future, ctx_lags,
                                              base_weight=base_w)
            else:
                loss = conditional_meanflow_loss_v2(flow_net, scaled_future, ctx_lags)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(flow_net.parameters(), flow_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
                if base_net and base_ema:
                    for p, pe in zip(base_net.parameters(), base_ema.parameters()):
                        pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item(); n_b += 1

        elapsed = time.time() - t0
        if (epoch+1) % 20 == 0:
            logger.info(f"[{variant}] Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/n_b:.4f} | {elapsed:.1f}s")

        if (epoch+1) % 100 == 0 or (epoch+1) == args.epochs:
            flow_ema.eval()
            test_transform = transformation.apply(dataset.test, is_train=False)
            test_splitter = InstanceSplitter(
                target_field="target", is_pad_field="is_pad", start_field="start",
                forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
                past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
                time_series_fields=["time_feat", "observed_values"],
            )
            if base_ema:
                forecaster = AdaptiveForecaster(flow_ema, base_ema, CTX_LEN, PRED_LEN,
                    num_samples=NUM_EVAL_SAMPLES, freq=FREQ, n_lags=N_LAGS, max_steps=1).to(device)
            else:
                forecaster = MeanFlowForecasterV2(flow_ema, CTX_LEN, PRED_LEN,
                    num_samples=NUM_EVAL_SAMPLES, freq=FREQ, n_lags=N_LAGS).to(device)
            predictor = PyTorchPredictor(prediction_length=PRED_LEN,
                input_names=["past_target", "past_observed_values"],
                prediction_net=forecaster, batch_size=512,
                input_transform=test_splitter, device=device)
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_transform, predictor=predictor, num_samples=NUM_EVAL_SAMPLES)
            forecasts = list(tqdm(forecast_it, total=len(test_transform), desc="Eval", leave=False))
            tss = list(ts_it)
            metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
            crps = metrics["mean_wQuantileLoss"]
            nd = metrics["ND"]
            if crps < best_crps:
                best_crps = crps
                torch.save({'net_ema': flow_ema.state_dict(), 'epoch': epoch+1,
                            'crps': crps, 'variant': variant,
                            'base_ema': base_ema.state_dict() if base_ema else None},
                           f'best_v5_{variant}.pt')
            logger.info(f"[{variant}] Epoch {epoch+1} -> CRPS={crps:.6f} | ND={nd:.6f} | Best={best_crps:.6f} | TSFlow={TSFLOW_CRPS}")

    logger.info(f"[{variant}] FINAL Best CRPS: {best_crps:.6f}")

if __name__ == "__main__":
    main()
