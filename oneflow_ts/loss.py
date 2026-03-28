"""
OneFlow-TS: Multi-resolution MeanFlow loss.

Per-level JVP self-consistency loss applied in wavelet space,
with resolution-specific weighting.
"""
import torch
import torch.nn as nn
from typing import List, Optional

from .wavelet import dwt_decompose


def logit_normal_sample(P_mean, P_std, n, device):
    rnd = torch.randn(n, device=device)
    return torch.sigmoid(rnd * P_std + P_mean).clip(1e-5, 1 - 1e-5)


def sample_t_r(n, device, ratio=0.75):
    t = logit_normal_sample(-0.6, 1.6, n, device)
    r = logit_normal_sample(-4.0, 1.6, n, device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    mask = torch.rand(n, device=device) < (1 - ratio)
    r = torch.where(mask, t, r)
    return t, r


def oneflow_loss(net, future_clean, context_with_lags, prior,
                 levels=2, norm_p=0.75, norm_eps=1e-3,
                 level_weights=None):
    """
    Multi-resolution MeanFlow self-consistency loss via JVP.

    For each wavelet level k:
      1. Decompose future_clean into wavelet coefficients
      2. Sample noise from resolution-matched prior
      3. Interpolate: z_k = (1-t)*x_k + t*e_k
      4. Compute JVP self-consistency loss
      5. Weight and sum across levels

    Args:
        net: OneFlowTSNet
        future_clean: (B, pred_len) — scaled future target
        context_with_lags: (B, 1+n_lags, ctx_len) — scaled context with lags
        prior: ResolutionMatchedPrior or IsotropicPrior
        levels: number of wavelet decomposition levels
        norm_p: adaptive weighting exponent
        norm_eps: adaptive weighting epsilon
        level_weights: optional list of per-level weights

    Returns:
        loss: scalar
    """
    B = future_clean.shape[0]
    device = future_clean.device

    # 1. Decompose targets into wavelet levels
    target_levels = dwt_decompose(future_clean, levels)

    # 2. Sample matched noise per level
    noise_levels = prior.sample(B)

    # 3. Sample (t, r) — shared across all levels for consistency
    t, r = sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    # 4. Build interpolated noisy inputs per level
    noisy_levels = []
    velocity_targets = []  # v_k = e_k - x_k
    for x_k, e_k in zip(target_levels, noise_levels):
        z_k = (1 - t_bc) * x_k + t_bc * e_k
        v_k = e_k - x_k
        noisy_levels.append(z_k)
        velocity_targets.append(v_k)

    # 5. JVP self-consistency loss per level
    # We need JVP of the network output w.r.t. (z, t, r)
    # Tangent vectors: (v_k, 1, 0) for each level
    def u_func(noisy_list, t_bc, r_bc):
        h_bc = t_bc - r_bc
        ts = (t_bc.squeeze(-1), h_bc.squeeze(-1))
        return net(noisy_list, ts, context_with_lags)

    # Pack all levels for JVP computation
    # We compute JVP for the entire network at once
    with torch.amp.autocast("cuda", enabled=False):
        # Forward: u_pred = net(noisy_levels, (t, h), ctx)
        h_bc = t_bc - r_bc
        u_preds = net(noisy_levels, (t_bc.squeeze(-1), h_bc.squeeze(-1)),
                      context_with_lags)

        # For JVP, we need du/dt for each level
        # We compute this by running the forward pass with tangent vectors
        # tangent_z_k = v_k (velocity direction)
        # tangent_t = 1, tangent_r = 0

        # Use torch.func.jvp per level (applied to each head independently)
        total_loss = torch.tensor(0.0, device=device)
        for k in range(len(target_levels)):
            x_k = target_levels[k]
            e_k = noise_levels[k]
            v_k = velocity_targets[k]
            z_k = noisy_levels[k]

            def u_func_k(z_in, t_in, r_in):
                h_in = t_in - r_in
                # Build full noisy_levels list with z_in at position k
                nl = list(noisy_levels)
                nl[k] = z_in
                ts = (t_in.squeeze(-1), h_in.squeeze(-1))
                outputs = net(nl, ts, context_with_lags)
                return outputs[k]

            u_pred_k, dudt_k = torch.func.jvp(
                u_func_k,
                (z_k, t_bc, r_bc),
                (v_k, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
            )

            # Self-consistency target: u = v - (t-r) * du/dt
            u_tgt_k = (v_k - (t_bc - r_bc) * dudt_k).detach()

            # Per-level loss with adaptive weighting
            loss_k = (u_pred_k - u_tgt_k) ** 2
            loss_k = loss_k.sum(dim=1)  # Sum over level dimension

            # Adaptive weighting
            adp_wt = (loss_k.detach() + norm_eps) ** norm_p
            loss_k = (loss_k / adp_wt).mean()

            # Apply level weight
            weight = level_weights[k] if level_weights is not None else 1.0
            total_loss = total_loss + weight * loss_k

    num_levels = len(target_levels)
    if level_weights is None:
        total_loss = total_loss / num_levels

    return total_loss


def oneflow_loss_simple(net, future_clean, context_with_lags, prior,
                        levels=2, norm_p=0.75, norm_eps=1e-3):
    """
    Simplified version: apply original MeanFlow loss in time domain
    using the flat interface. Uses DWT/IDWT inside the network.

    This is a fallback if the per-level JVP has issues.
    """
    B = future_clean.shape[0]
    device = future_clean.device
    e_flat = torch.randn_like(future_clean)

    t, r = sample_t_r(B, device)
    t_bc = t.unsqueeze(-1)
    r_bc = r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e_flat
    v = e_flat - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net.forward_flat(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)),
                                context_with_lags)

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
