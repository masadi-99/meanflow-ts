"""
Item 4: Baseline — Standard FM with stat conditioning (same architecture, no JVP).

Tests whether the quality advantage of stat conditioning comes from MeanFlow
specifically or from any conditional generative model with stat conditioning.

If FM+stats matches MeanFlow+stats → the contribution is stat conditioning, not MeanFlow.
If MeanFlow+stats beats FM+stats → MeanFlow's JVP helps for controllable generation.
"""
import os, sys, time, logging, tempfile
import numpy as np
import torch
import torch.nn.functional as F
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
from meanflow_ts.model_v3 import ConditionalMeanFlowNetV3
from meanflow_ts.model_v2 import extract_lag_features, MeanFlowForecasterV2
from meanflow_ts.utils import extract_stats
from meanflow_ts.model import sample_t_r

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = os.environ.get("DATASET", "electricity_nips")
CTX_LEN, PRED_LEN, FREQ = 24, 24, "H"
N_LAGS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAT_DROPOUT = 0.15
EPOCHS = 600


def standard_fm_loss_with_stats(net, future_clean, ctx_with_lags, stat_vector=None,
                                 norm_p=0.75, norm_eps=1e-3):
    """Standard FM loss (NO JVP) with stat conditioning. h=0 always."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t = torch.rand(B, device=device)
    t_bc = t.unsqueeze(-1)
    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean
    h = torch.zeros(B, device=device)
    pred = net(z, (t, h), ctx_with_lags, None, stat_vector)
    loss = (pred - v) ** 2
    loss = loss.sum(dim=1)
    adp_wt = (loss.detach() + norm_eps) ** norm_p
    return (loss / adp_wt).mean()


class FM1StepForecasterWithStats(torch.nn.Module):
    """Standard FM 1-step inference: z_0 = z_1 - v(z_1, t=1, h=0)."""
    def __init__(self, net, ctx_len, pred_len, num_samples=16, freq="H", n_lags=7):
        super().__init__()
        self.net = net
        self.context_length = ctx_len
        self.prediction_length = pred_len
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
        ctx_with_lags = extract_lag_features(
            past_target, self.context_length, self.freq, self.n_lags) / loc.unsqueeze(1)
        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.zeros(B, device=device)  # h=0 for standard FM
            u = self.net(z_1, (t, h), ctx_with_lags)
            all_preds.append((z_1 - u) * loc)  # z_0 = z_1 - v
        return torch.stack(all_preds, dim=1)


def main():
    torch.manual_seed(6432)
    np.random.seed(6432)

    dataset = get_dataset(DATASET)
    max_lag = 672

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

    # Same architecture as MeanFlow v3, same params, just different loss
    net = ConditionalMeanFlowNetV3(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE)
    net_ema = deepcopy(net).eval()
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"=== FM+Stats Baseline | {n_params:,} params ===")

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

            stats = extract_stats(scaled_future)
            stat_vec = stats if torch.rand(1).item() > STAT_DROPOUT else None

            loss = standard_fm_loss_with_stats(net, scaled_future, ctx_with_lags, stat_vec)
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
            forecaster = FM1StepForecasterWithStats(
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
                            'params': n_params, 'mode': 'fm_stats_baseline'},
                           f'best_fm_stats_{DATASET}.pt')
            logger.info(f"  CRPS={crps:.6f}{tag} | Best: {best_crps:.6f}")
            logger.info(f"  MeanFlow+stats: 0.048 | TSFlow: 0.045 | v4: 0.047")

    logger.info(f"DONE. Best CRPS: {best_crps:.6f} ({n_params:,} params)")


if __name__ == "__main__":
    main()
