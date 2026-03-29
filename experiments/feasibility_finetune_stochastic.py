"""
Experiment C: Fine-tune MeanFlow with stochastic midpoint noise

Adds a reconstruction loss for stochastic 2-step paths alongside JVP loss.
After fine-tuning, evaluates mixed ensemble vs pure 1-step.
"""
import os, sys, time, torch, numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v2 import ConditionalMeanFlowNetV2, extract_lag_features
from meanflow_ts.model_v2 import conditional_meanflow_loss_v2
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.transform import *
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.predictor import PyTorchPredictor
from tqdm.auto import tqdm

import tempfile
try:
    import pykeops
    tmp = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(tmp)
    pykeops.clean_pykeops()
except: pass

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

FINETUNE_EPOCHS = 100
SIGMA_TRAIN = 0.3
LAMBDA_STOCH = 0.5
LR = 1e-4
T_MID = 0.5
LAG_MAP = {"H": 672, "B": 750, "1D": 750}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stochastic_path_loss(net, future_clean, ctx_with_lags, sigma=0.3, t_mid=0.5):
    B = future_clean.shape[0]
    z_1 = torch.randn_like(future_clean)
    h_s1 = 1.0 - t_mid
    u_s1 = net(z_1, (torch.ones(B, device=DEVICE),
                      torch.full((B,), h_s1, device=DEVICE)), ctx_with_lags)
    z_mid = z_1 - h_s1 * u_s1
    eps = torch.randn_like(z_mid)
    z_mid_noisy = (1 - sigma**2)**0.5 * z_mid + sigma * eps
    u_s2 = net(z_mid_noisy, (torch.full((B,), t_mid, device=DEVICE),
                              torch.full((B,), t_mid, device=DEVICE)), ctx_with_lags)
    z_0_pred = z_mid_noisy - t_mid * u_s2
    return ((z_0_pred - future_clean) ** 2).mean()


class StochasticMeanFlowForecaster(torch.nn.Module):
    def __init__(self, net, ctx_len, pred_len, num_samples=16,
                 freq="H", n_lags=7, sigma=0.3, frac_stochastic=0.5):
        super().__init__()
        self.net = net
        self.context_length = ctx_len
        self.prediction_length = pred_len
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags
        self.sigma = sigma
        self.n_stoch = int(num_samples * frac_stochastic)
        self.n_det = num_samples - self.n_stoch

    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
        ctx_with_lags = extract_lag_features(
            past_target, self.context_length, self.freq, self.n_lags) / loc.unsqueeze(1)

        all_preds = []
        for _ in range(self.n_det):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            u = self.net(z_1, (torch.ones(B, device=device), torch.ones(B, device=device)), ctx_with_lags)
            all_preds.append((z_1 - u) * loc)

        t_mid = 0.5
        for _ in range(self.n_stoch):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            h_s1 = 1.0 - t_mid
            u_s1 = self.net(z_1, (torch.ones(B, device=device),
                                   torch.full((B,), h_s1, device=device)), ctx_with_lags)
            z_mid = z_1 - h_s1 * u_s1
            eps = torch.randn_like(z_mid)
            z_mid = (1 - self.sigma**2)**0.5 * z_mid + self.sigma * eps
            u_s2 = self.net(z_mid, (torch.full((B,), t_mid, device=device),
                                     torch.full((B,), t_mid, device=device)), ctx_with_lags)
            z_0 = z_mid - t_mid * u_s2
            all_preds.append(z_0 * loc)

        return torch.stack(all_preds, dim=1)


def evaluate_model(net_ema, dataset, num_samples=16, sigma=0, frac_stochastic=0):
    max_lag = LAG_MAP.get(FREQ, 672)
    transformation = Chain([
        AsNumpyArray(field="target", expected_ndim=1),
        AddObservedValuesIndicator(target_field="target", output_field="observed_values"),
        AddTimeFeatures(start_field="start", target_field="target", output_field="time_feat",
            time_features=time_features_from_frequency_str(FREQ), pred_length=PRED_LEN),
    ])
    test_transform = transformation.apply(dataset.test, is_train=False)
    test_splitter = InstanceSplitter(
        target_field="target", is_pad_field="is_pad", start_field="start",
        forecast_start_field="forecast_start", instance_sampler=TestSplitSampler(),
        past_length=CTX_LEN + max_lag, future_length=PRED_LEN,
        time_series_fields=["time_feat", "observed_values"],
    )
    forecaster = StochasticMeanFlowForecaster(
        net_ema, CTX_LEN, PRED_LEN, num_samples=num_samples,
        freq=FREQ, n_lags=N_LAGS, sigma=sigma, frac_stochastic=frac_stochastic,
    ).to(DEVICE)
    predictor = PyTorchPredictor(
        prediction_length=PRED_LEN, input_names=["past_target", "past_observed_values"],
        prediction_net=forecaster, batch_size=512, input_transform=test_splitter, device=DEVICE,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_transform, predictor=predictor, num_samples=num_samples)
    forecasts = list(tqdm(forecast_it, total=len(test_transform), desc="Eval"))
    tss = list(ts_it)
    metrics, _ = Evaluator(num_workers=0)(tss, forecasts)
    return metrics["mean_wQuantileLoss"]


def main():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    net = ConditionalMeanFlowNetV2(
        pred_len=PRED_LEN, ctx_len=CTX_LEN, n_lags=N_LAGS,
        model_channels=128, num_res_blocks=4, time_emb_dim=64, dropout=0.1,
    ).to(DEVICE)
    net.load_state_dict(ckpt['net_ema'])
    net_ema = deepcopy(net).eval()
    print(f"Loaded {CKPT} (epoch {ckpt.get('epoch','?')}, CRPS={ckpt.get('crps','?')})")

    dataset = get_dataset(DATASET)
    max_lag = LAG_MAP.get(FREQ, 672)

    print("\n--- Evaluating BEFORE fine-tuning ---")
    crps_before_pure = evaluate_model(net_ema, dataset, num_samples=50, sigma=0, frac_stochastic=0)
    crps_before_mixed = evaluate_model(net_ema, dataset, num_samples=50, sigma=SIGMA_TRAIN, frac_stochastic=0.5)
    print(f"  CRPS (pure 1-step):  {crps_before_pure:.6f}")
    print(f"  CRPS (mixed σ={SIGMA_TRAIN}): {crps_before_mixed:.6f}")

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
    transformed = transformation.apply(dataset.train, is_train=True)
    train_loader = TrainDataLoader(
        Cached(transformed), batch_size=64, stack_fn=batchify,
        transform=train_splitter, num_batches_per_epoch=128, shuffle_buffer_length=10000,
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)

    print(f"\n--- Fine-tuning for {FINETUNE_EPOCHS} epochs ---")
    for epoch in range(FINETUNE_EPOCHS):
        net.train()
        epoch_loss_jvp, epoch_loss_stoch, n_b = 0, 0, 0
        for batch in train_loader:
            past = batch["past_target"].to(DEVICE)
            future = batch["future_target"].to(DEVICE)
            ctx = past[:, -CTX_LEN:]
            loc = ctx.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
            ctx_with_lags = extract_lag_features(past, CTX_LEN, FREQ, N_LAGS) / loc.unsqueeze(1)
            scaled_future = future / loc

            loss_jvp = conditional_meanflow_loss_v2(net, scaled_future, ctx_with_lags)
            loss_stoch = stochastic_path_loss(net, scaled_future, ctx_with_lags, SIGMA_TRAIN, T_MID)
            loss = loss_jvp + LAMBDA_STOCH * loss_stoch

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for p, pe in zip(net.parameters(), net_ema.parameters()):
                    pe.data.lerp_(p.data, 1e-4)
            epoch_loss_jvp += loss_jvp.item()
            epoch_loss_stoch += loss_stoch.item()
            n_b += 1

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{FINETUNE_EPOCHS} | JVP: {epoch_loss_jvp/n_b:.4f} | Stoch: {epoch_loss_stoch/n_b:.4f}")

        if (epoch+1) % 25 == 0 or (epoch+1) == FINETUNE_EPOCHS:
            net_ema.eval()
            crps_pure = evaluate_model(net_ema, dataset, num_samples=50, sigma=0, frac_stochastic=0)
            crps_mixed = evaluate_model(net_ema, dataset, num_samples=50, sigma=SIGMA_TRAIN, frac_stochastic=0.5)
            print(f"    CRPS (pure):  {crps_pure:.6f} (before: {crps_before_pure:.6f})")
            print(f"    CRPS (mixed): {crps_mixed:.6f} (before: {crps_before_mixed:.6f})")
            torch.save({'net_ema': net_ema.state_dict(), 'epoch': epoch+1,
                        'crps_pure': crps_pure, 'crps_mixed': crps_mixed},
                       f'finetuned_stochastic_{DATASET}.pt')

    print("\n" + "=" * 70)
    print(f"EXPERIMENT C: FINE-TUNING WITH STOCHASTIC PATH — {DATASET}")
    print("=" * 70)

    net_ema.eval()
    crps_after_pure = evaluate_model(net_ema, dataset, num_samples=50, sigma=0, frac_stochastic=0)
    crps_after_mixed = evaluate_model(net_ema, dataset, num_samples=50, sigma=SIGMA_TRAIN, frac_stochastic=0.5)

    print(f"\n  {'Method':<35} | {'Before':>10} | {'After':>10} | {'Change':>10}")
    print(f"  {'-'*70}")
    print(f"  {'Pure 1-step':<35} | {crps_before_pure:>10.6f} | {crps_after_pure:>10.6f} | "
          f"{(crps_before_pure-crps_after_pure)/crps_before_pure*100:>+9.2f}%")
    print(f"  {'Mixed (σ=' + str(SIGMA_TRAIN) + ')':<35} | {crps_before_mixed:>10.6f} | {crps_after_mixed:>10.6f} | "
          f"{(crps_before_mixed-crps_after_mixed)/crps_before_mixed*100:>+9.2f}%")

    mix_vs_pure = (crps_after_pure - crps_after_mixed) / crps_after_pure * 100
    print(f"\n  Mixed vs Pure (after FT):  {mix_vs_pure:+.2f}%")

    print(f"\n  VERDICT:")
    if mix_vs_pure > 2 and crps_after_mixed < crps_before_pure:
        print(f"  ✓ STOCHASTIC FINE-TUNING WORKS.")
    elif crps_after_mixed < crps_before_mixed:
        print(f"  ~ FINE-TUNING HELPS but mixed still doesn't beat pure 1-step.")
    else:
        print(f"  ✗ FINE-TUNING DOESN'T HELP.")


if __name__ == "__main__":
    main()
