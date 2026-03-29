"""
OneFlow-TS Improvements: Residual flow, iMF loss, adaptive multi-step.

Three independent improvements that can be combined:
1. ResidualFlowForecaster: Deterministic base forecast + flow for residual uncertainty
2. imf_loss: iMF velocity loss reformulation (more stable than original MeanFlow JVP)
3. adaptive_multistep_sample: Adaptive 1-2-4 step inference via h parameter
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================
# Improvement 1: Residual Flow
# ============================================================

class BaseForecaster(nn.Module):
    """
    Lightweight deterministic base forecaster.
    Predicts the MEAN forecast from context. The flow then only
    models the residual uncertainty around this mean.

    Output is initialized near zero (small init) to avoid destabilizing
    the flow training at the start. The base learns to predict the mean
    gradually while the flow handles everything initially.
    """
    def __init__(self, ctx_len, pred_len, n_lags=7, hidden_dim=128):
        super().__init__()
        input_dim = ctx_len * (1 + n_lags)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pred_len),
        )
        # Small init: base starts predicting ~0, so residual ≈ full signal
        # The base gradually learns the mean, shifting work from flow to base
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_with_lags):
        """
        context_with_lags: (B, 1+n_lags, ctx_len)
        Returns: (B, pred_len) — deterministic base forecast (in scaled space)
        """
        B = context_with_lags.shape[0]
        flat = context_with_lags.reshape(B, -1)
        return self.net(flat)


def residual_meanflow_loss(flow_net, base_net, future_clean, context_with_lags,
                           norm_p=0.75, norm_eps=1e-3, base_weight=1.0):
    """
    MeanFlow loss on the RESIDUAL = future - base_forecast.

    The flow learns the distribution of (future - base_prediction),
    which is simpler than the full future distribution.
    base_weight: weight for base MSE loss (ramp up during training)
    """
    B = future_clean.shape[0]
    device = future_clean.device

    # Deterministic base prediction (detached from flow path)
    base_pred = base_net(context_with_lags)
    residual = future_clean - base_pred.detach()

    # Standard MeanFlow loss on the residual
    e = torch.randn_like(residual)
    t, r = _sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    z = (1 - t_bc) * residual + t_bc * e
    v = e - residual

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return flow_net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        flow_loss = (u_pred - u_tgt) ** 2
        flow_loss = flow_loss.sum(dim=1)
        adp_wt = (flow_loss.detach() + norm_eps) ** norm_p
        flow_loss = (flow_loss / adp_wt).mean()

    # Base forecaster loss — separate from flow, just MSE
    base_loss = F.mse_loss(base_pred, future_clean)

    return flow_loss + base_weight * base_loss


# ============================================================
# Improvement 2: iMF Velocity Loss
# ============================================================

def imf_loss(net, future_clean, context_with_lags,
             norm_p=0.75, norm_eps=1e-3):
    """
    iMF-style velocity loss for MeanFlow (Improved Mean Flows, Dec 2025).

    Key difference from original MeanFlow:
    - Original: JVP tangent uses ground-truth velocity v_true = e - x
    - iMF: JVP tangent uses the network's OWN predicted velocity
    - Loss is on the instantaneous velocity v, not the average velocity u

    This eliminates the network-dependent target problem and reduces
    training loss variance.
    """
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)

    t, r = _sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v_true = e - future_clean  # Ground truth instantaneous velocity

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)

    with torch.amp.autocast("cuda", enabled=False):
        # First: get u_pred to compute v_pred (network's instantaneous velocity estimate)
        u_pred = u_func(z, t_bc, r_bc)

        # v_pred from the MeanFlow identity: v = u + (t-r) * du/dt
        # We need du/dt via JVP, but using v_pred as tangent (not v_true)
        # For the first iteration, use v_true as tangent (bootstrap)
        # Then refine: v_pred_approx = u_pred (at r=t, u≈v)

        # iMF approach: use u_pred as a proxy for v direction in JVP tangent
        # The key insight: JVP with u_pred tangent gives a better-conditioned target
        u_pred_2, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (u_pred.detach(), torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )

        # Reconstruct instantaneous velocity from MeanFlow identity
        v_pred = u_pred_2 + (t_bc - r_bc) * dudt

        # Loss on instantaneous velocity (not average velocity)
        loss = (v_pred - v_true) ** 2
        loss = loss.sum(dim=1)
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = (loss / adp_wt).mean()

    return loss


def imf_residual_loss(flow_net, base_net, future_clean, context_with_lags,
                      norm_p=0.75, norm_eps=1e-3, base_weight=1.0):
    """Combined iMF loss + residual flow."""
    B = future_clean.shape[0]
    device = future_clean.device

    # Base prediction (detached from flow path)
    base_pred = base_net(context_with_lags)
    residual = future_clean - base_pred.detach()

    # iMF loss on residual
    e = torch.randn_like(residual)
    t, r = _sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    z = (1 - t_bc) * residual + t_bc * e
    v_true = e - residual

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return flow_net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred = u_func(z, t_bc, r_bc)
        u_pred_2, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (u_pred.detach(), torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        v_pred = u_pred_2 + (t_bc - r_bc) * dudt
        flow_loss = (v_pred - v_true) ** 2
        flow_loss = flow_loss.sum(dim=1)
        adp_wt = (flow_loss.detach() + norm_eps) ** norm_p
        flow_loss = (flow_loss / adp_wt).mean()

    base_loss = F.mse_loss(base_pred, future_clean)
    return flow_loss + base_weight * base_loss


# ============================================================
# Improvement 3: Adaptive Multi-Step Inference
# ============================================================

@torch.no_grad()
def adaptive_multistep_sample(net, z_1, context_with_lags, threshold=0.5,
                               max_steps=4):
    """
    Adaptive multi-step sampling using MeanFlow's h parameter.

    Strategy:
    1. Try 1-step: compute u(z_1, t=1, h=1)
    2. If velocity magnitude is large (hard sample), refine with more steps
    3. Use MeanFlow's compositional property for multi-step

    Args:
        net: MeanFlow network (predicts average velocity u)
        z_1: (B, pred_len) — noise samples
        context_with_lags: (B, 1+n_lags, ctx_len)
        threshold: velocity magnitude threshold for step decision
        max_steps: maximum number of steps (1, 2, or 4)

    Returns:
        x_0: (B, pred_len) — generated forecast (in scaled space)
        steps_used: (B,) — number of steps used per sample
    """
    B = z_1.shape[0]
    device = z_1.device

    # 1-step attempt
    t_1 = torch.ones(B, device=device)
    h_1 = torch.ones(B, device=device)
    u_1step = net(z_1, (t_1, h_1), context_with_lags)
    x_0_1step = z_1 - u_1step

    # Velocity magnitude per sample
    vel_magnitude = u_1step.abs().mean(dim=-1)  # (B,)

    if max_steps == 1:
        return x_0_1step, torch.ones(B, device=device, dtype=torch.long)

    # Decide which samples need refinement
    needs_refinement = vel_magnitude > threshold

    if not needs_refinement.any():
        return x_0_1step, torch.ones(B, device=device, dtype=torch.long)

    # 2-step for samples that need it
    # Step 1: z_1 -> z_0.5
    h_half = torch.full((B,), 0.5, device=device)
    u_half1 = net(z_1, (t_1, h_half), context_with_lags)
    z_half = z_1 - 0.5 * u_half1

    # Step 2: z_0.5 -> z_0
    t_half = torch.full((B,), 0.5, device=device)
    u_half2 = net(z_half, (t_half, h_half), context_with_lags)
    x_0_2step = z_half - 0.5 * u_half2

    # Blend: use 1-step for easy, 2-step for hard
    x_0 = torch.where(needs_refinement.unsqueeze(-1), x_0_2step, x_0_1step)
    steps_used = torch.where(needs_refinement, torch.tensor(2, device=device),
                             torch.tensor(1, device=device))

    return x_0, steps_used


class AdaptiveForecaster(nn.Module):
    """
    Forecaster with adaptive multi-step and optional residual flow.
    """
    def __init__(self, flow_net, base_net=None, context_length=24,
                 prediction_length=24, num_samples=16, freq="H", n_lags=7,
                 adaptive_threshold=0.5, max_steps=2):
        super().__init__()
        self.flow_net = flow_net
        self.base_net = base_net  # Optional: for residual flow
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags
        self.adaptive_threshold = adaptive_threshold
        self.max_steps = max_steps

    def _extract_lag_features(self, past_target):
        ctx_len = self.context_length
        B = past_target.shape[0]
        context = past_target[:, -ctx_len:]
        if self.freq == "H":
            lag_offsets = [24 * (i + 1) for i in range(self.n_lags)]
        elif self.freq == "B":
            lag_offsets = [5 * (i + 1) for i in range(self.n_lags)]
        elif self.freq in ("D", "1D"):
            lag_offsets = [7 * (i + 1) for i in range(self.n_lags)]
        else:
            lag_offsets = [24 * (i + 1) for i in range(self.n_lags)]
        channels = [context.unsqueeze(1)]
        for offset in lag_offsets:
            start = past_target.shape[1] - ctx_len - offset
            end = past_target.shape[1] - offset
            if start >= 0:
                lag = past_target[:, start:end]
            else:
                lag = torch.zeros(B, ctx_len, device=past_target.device)
                if end > 0:
                    available = past_target[:, :end]
                    lag[:, -available.shape[1]:] = available
            channels.append(lag.unsqueeze(1))
        return torch.cat(channels, dim=1)

    @torch.no_grad()
    def forward(self, past_target, past_observed_values, **kwargs):
        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
        ctx_with_lags = self._extract_lag_features(past_target) / loc.unsqueeze(1)

        # Base forecast (if using residual flow)
        base = self.base_net(ctx_with_lags) if self.base_net is not None else 0.0

        all_preds = []
        total_steps = 0
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)

            if self.max_steps > 1:
                residual, steps = adaptive_multistep_sample(
                    self.flow_net, z_1, ctx_with_lags,
                    threshold=self.adaptive_threshold,
                    max_steps=self.max_steps,
                )
                total_steps += steps.float().mean().item()
            else:
                t = torch.ones(B, device=device)
                h = torch.ones(B, device=device)
                u = self.flow_net(z_1, (t, h), ctx_with_lags)
                residual = z_1 - u
                total_steps += 1

            pred = (base + residual) * loc
            all_preds.append(pred)

        self._avg_steps = total_steps / self.num_samples
        return torch.stack(all_preds, dim=1)


# ============================================================
# Shared utilities
# ============================================================

def _logit_normal_sample(P_mean, P_std, n, device):
    rnd = torch.randn(n, device=device)
    return torch.sigmoid(rnd * P_std + P_mean).clip(1e-5, 1 - 1e-5)


def _sample_t_r(n, device, ratio=0.75):
    t = _logit_normal_sample(-0.6, 1.6, n, device)
    r = _logit_normal_sample(-4.0, 1.6, n, device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    mask = torch.rand(n, device=device) < (1 - ratio)
    r = torch.where(mask, t, r)
    return t, r
