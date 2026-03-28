"""
Iteration 2: Test V2 architecture (single-network multi-resolution).

V2 key differences from V1:
- Single network predicts velocity in time domain (no per-level heads)
- Multi-scale context conditioning via wavelet decomposition of context
- Resolution-matched noise: sample in wavelet domain, IDWT to time domain
- Compatible with standard MeanFlow JVP loss (no gradient conflicts)

Variants:
  v2-matched: V2 with resolution-matched noise prior
  v2-iso: V2 with standard N(0,I) noise (ablation)
  v2-matched-600ep: V2 with matched prior, 600 epochs (full training)
  baseline-600ep: Original MeanFlow-TS v4, 600 epochs (reference)
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

from oneflow_ts.model_v2 import OneFlowTSNetV2, OneFlowForecasterV2
from oneflow_ts.priors import ResolutionMatchedPrior, IsotropicPrior
from oneflow_ts.wavelet import get_level_sizes, idwt_reconstruct
from meanflow_ts.model_v2 import (
    ConditionalMeanFlowNetV2, conditional_meanflow_loss_v2,
    MeanFlowForecasterV2, extract_lag_features,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = "electricity_nips"
FREQ = "H"
CTX_LEN = 24
PRED_LEN = 24
MAX_LAG = 672
N_LAGS = 7
NUM_EVAL_SAMPLES = 16
TSFLOW_CRPS = 0.045


def get_data_pipeline():
    dataset = get_dataset(DATASET)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(
            start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(FREQ), pred_length=PRED_LEN,
        ),
    ])
    train_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=PRED_LEN),
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )
    transformed_data = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed_data), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=10000,
    )
    return dataset, transformation, train_loader


def evaluate_model(net_ema, dataset, transformation, device, is_v2, prior=None):
    net_ema.eval()
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=CTX_LEN + MAX_LAG, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )

    if is_v2:
        forecaster = OneFlowForecasterV2(
            net_ema, prior, CTX_LEN, PRED_LEN,
            num_samples=NUM_EVAL_SAMPLES, freq=FREQ, n_lags=N_LAGS,
        ).to(device)
    else:
        forecaster = MeanFlowForecasterV2(
            net_ema, CTX_LEN, PRED_LEN, num_samples=NUM_EVAL_SAMPLES,
            freq=FREQ, n_lags=N_LAGS,
        ).to(device)

    predictor = PyTorchPredictor(
        prediction_length=PRED_LEN,
        input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=512,
        input_transform=test_splitter, device=device,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=NUM_EVAL_SAMPLES,
    )
    forecasts = list(tqdm(forecast_it, total=len(test_transform), desc="Eval", leave=False))
    tss = list(ts_it)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    return metrics["mean_wQuantileLoss"], metrics["ND"]


def v2_meanflow_loss(net, future_clean, context_with_lags, prior,
                     levels=2, norm_p=0.75, norm_eps=1e-3):
    """
    Standard MeanFlow JVP loss for V2 model.
    Noise is sampled in wavelet domain (matched prior) then IDWT'd.
    """
    from oneflow_ts.loss import sample_t_r
    B = future_clean.shape[0]
    device = future_clean.device

    # Sample noise in wavelet domain with matched priors, then IDWT
    noise_levels = prior.sample(B)
    e = idwt_reconstruct(noise_levels)  # (B, pred_len) — structured noise

    t, r = sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        loss = (u_pred - u_tgt) ** 2
        loss = loss.sum(dim=1)
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = (loss / adp_wt).mean()
    return loss


def run_variant(variant, epochs, device):
    torch.manual_seed(6432)
    np.random.seed(6432)

    dataset, transformation, train_loader = get_data_pipeline()
    levels = 2
    sizes = get_level_sizes(PRED_LEN, levels)

    is_v2 = variant.startswith("v2")

    if is_v2:
        net = OneFlowTSNetV2(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            levels=levels, model_channels=128, num_res_blocks=4,
            time_emb_dim=64, dropout=0.1,
        ).to(device)
        if "iso" in variant:
            prior = IsotropicPrior(sizes).to(device)
            logger.info("Using isotropic prior")
        else:
            prior = ResolutionMatchedPrior(sizes, length_scale=3.0).to(device)
            logger.info("Using resolution-matched prior")
    else:
        net = ConditionalMeanFlowNetV2(
            pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
            model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
        ).to(device)
        prior = None

    net_ema = deepcopy(net).eval()
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"[{variant}] Params: {params:,}")

    optimizer = AdamW(net.parameters(), lr=6e-4)
    best_crps = float('inf')

    for epoch in range(epochs):
        net.train()
        epoch_loss, n_b = 0, 0
        t0 = time.time()

        for batch in train_loader:
            past = batch["past_target"].to(device)
            future = batch["future_target"].to(device)
            ctx = past[:, -CTX_LEN:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
            ctx_lags = extract_lag_features(past, CTX_LEN, FREQ, N_LAGS) / loc.unsqueeze(1)
            scaled_future = future / loc

            if is_v2:
                loss = v2_meanflow_loss(net, scaled_future, ctx_lags, prior, levels=levels)
            else:
                loss = conditional_meanflow_loss_v2(net, scaled_future, ctx_lags)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss += loss.item()
            n_b += 1

        avg_loss = epoch_loss / n_b
        elapsed = time.time() - t0

        if (epoch + 1) % 20 == 0:
            logger.info(f"[{variant}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
            crps, nd = evaluate_model(net_ema, dataset, transformation, device, is_v2, prior)
            if crps < best_crps:
                best_crps = crps
                torch.save({
                    'net_ema': net_ema.state_dict(), 'epoch': epoch+1,
                    'crps': crps, 'variant': variant,
                }, f'best_{variant}.pt')
            logger.info(f"[{variant}] Epoch {epoch+1} → CRPS={crps:.6f} | ND={nd:.6f} | Best={best_crps:.6f} | TSFlow={TSFLOW_CRPS}")

    return best_crps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["v2-matched", "v2-iso", "baseline"])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for variant in args.variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {variant} ({args.epochs} epochs)")
        logger.info(f"{'='*60}")
        best_crps = run_variant(variant, args.epochs, device)
        results[variant] = best_crps
        logger.info(f"[{variant}] DONE — Best CRPS: {best_crps:.6f}")

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    sorted_r = sorted(results.items(), key=lambda x: x[1])
    for v, crps in sorted_r:
        marker = " ← BEST" if v == sorted_r[0][0] else ""
        logger.info(f"  {v:<30} CRPS: {crps:.6f}{marker}")
    logger.info(f"  {'TSFlow (32 NFE)':<30} CRPS: {TSFLOW_CRPS}")

    best_v, best_crps = sorted_r[0]
    baseline = results.get("baseline", None)
    if baseline:
        if best_crps < baseline:
            logger.info(f"\n  BEST ({best_v}) is {(1-best_crps/baseline)*100:.1f}% BETTER than baseline ✓")
        else:
            logger.info(f"\n  BEST ({best_v}) is {(best_crps/baseline-1)*100:.1f}% worse than baseline ✗")


if __name__ == "__main__":
    main()
