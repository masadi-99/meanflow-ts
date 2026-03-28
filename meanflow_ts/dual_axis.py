"""
Dual-Axis Self-Consistency for Temporal Flow Matching.

Generation-axis: MeanFlow JVP identity  u = v − (s−r)·du/ds
Temporal-axis:   Shift-equivariance     v^{t+1}(z,s) = v^t(T·z, s)

The temporal identity is exact for stationary time series and approximate
for non-stationary. It enforces that the velocity field respects the
temporal structure of the data without requiring the outputs at adjacent
positions to be similar (which would be mere smoothness), but rather
that the FUNCTION mapping input→velocity is equivariant to temporal shifts.
"""
import torch
import torch.nn.functional as F
from .model import sample_t_r
from .model_v2 import ConditionalMeanFlowNetV2, extract_lag_features


def temporal_shift(z, direction=1):
    """
    Shift time series by 1 position.
    direction=1: shift right (T operator), direction=-1: shift left.
    Pads with edge values to maintain length.
    """
    if direction == 1:
        return torch.cat([z[:, :1], z[:, :-1]], dim=1)
    else:
        return torch.cat([z[:, 1:], z[:, -1:]], dim=1)


def temporal_equivariance_loss(net, z, time_steps, context_with_lags):
    """
    Temporal shift-equivariance loss.

    For stationary time series, the true velocity field satisfies:
        v^{t+1}(z, s) = v^t(T·z, s)

    where T is the temporal shift operator. This means: the velocity at
    position t+1 should equal the velocity at position t evaluated on
    shifted input.

    We compute this by:
    1. v = net(z, time_steps, ctx)           — standard forward pass
    2. v_shifted = net(T·z, time_steps, ctx) — forward pass on shifted input
    3. Loss = ||v[:, 1:] - v_shifted[:, :-1]||^2

    v[:, 1:] gives positions 1..T-1 of original output.
    v_shifted[:, :-1] gives positions 0..T-2 of shifted-input output.
    If equivariant: v^{t+1}(z) = v^t(Tz), so these should match.
    """
    v_original = net(z, time_steps, context_with_lags)
    z_shifted = temporal_shift(z, direction=1)
    v_shifted = net(z_shifted, time_steps, context_with_lags)

    # v_original[:, 1:] = [v^1, v^2, ..., v^{T-1}] at positions 1..T-1
    # v_shifted[:, :-1] = [v^0(Tz), v^1(Tz), ..., v^{T-2}(Tz)] at positions 0..T-2
    # Equivariance: v^{t+1}(z) = v^t(Tz), so v_original[:, 1:] ≈ v_shifted[:, :-1]
    loss = F.mse_loss(v_original[:, 1:], v_shifted[:, :-1].detach())
    return loss


def dual_axis_meanflow_loss(net, future_clean, context_with_lags,
                             lambda_temporal=0.1, norm_p=0.75, norm_eps=1e-3):
    """
    Dual-axis loss: MeanFlow JVP + temporal shift-equivariance.

    Args:
        net: ConditionalMeanFlowNetV2
        future_clean: (B, pred_len) — scaled target
        context_with_lags: (B, 1+n_lags, ctx_len) — scaled context with lags
        lambda_temporal: weight for temporal equivariance term
    """
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    # === Generation-axis: MeanFlow JVP loss ===
    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        mf_loss = (u_pred - u_tgt) ** 2
        mf_loss = mf_loss.sum(dim=1)
        adp_wt = (mf_loss.detach() + norm_eps) ** norm_p
        mf_loss = (mf_loss / adp_wt).mean()

    # === Temporal-axis: Shift-equivariance loss ===
    # v^{t+1}(z, s) should equal v^t(shift(z), s) for stationary series.
    # Requires one additional forward pass on shifted input.
    z_shifted = temporal_shift(z.detach(), direction=1)
    h_vals = (t - r)
    v_on_shifted = net(z_shifted, (t, h_vals), context_with_lags)
    # u_pred[:, 1:] = v at positions 1..T-1
    # v_on_shifted[:, :-1] = v at positions 0..T-2 on shifted input
    te_loss = F.mse_loss(u_pred[:, 1:].detach(), v_on_shifted[:, :-1])

    return mf_loss + lambda_temporal * te_loss, mf_loss.item(), te_loss.item()
