"""
MeanFlow-TS v3: Multi-resolution refinement + future-statistic conditioning.

Extends ConditionalMeanFlowNetV2 with two new conditioning paths:
1. Coarse trajectory conditioning (for multi-resolution refinement)
2. Future-statistic conditioning (for controllable generation)

Both use classifier-free dropout so the model works unconditionally too.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SinusoidalPosEmb, ResBlock1D, sample_t_r
from .utils import downsample, upsample, extract_stats, N_STATS


class ConditionalMeanFlowNetV3(nn.Module):
    """
    MeanFlow backbone with three conditioning paths:
    1. Past context with lag features (from v2)
    2. Coarse future trajectory (multi-resolution)
    3. Future statistics vector (controllable generation)

    Each conditioning path can be independently dropped (classifier-free).
    """

    def __init__(self, pred_len=24, ctx_len=24, n_lags=7, model_channels=128,
                 num_res_blocks=4, time_emb_dim=64, dropout=0.1,
                 n_stats=N_STATS):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.n_lags = n_lags
        self.n_stats = n_stats

        # Dual time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # === Path 1: Context encoder (same as v2) ===
        self.ctx_proj = nn.Conv1d(1 + n_lags, model_channels, 1)
        self.ctx_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout) for _ in range(2)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(model_channels, emb_dim),
        )
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Conv1d(model_channels, model_channels, 1)

        # === Path 2: Coarse trajectory encoder ===
        # Processes upsampled coarse trajectory (pred_len,) → spatial features
        self.coarse_proj = nn.Conv1d(1, model_channels, 1)
        self.coarse_block = ResBlock1D(model_channels, emb_dim, dropout)
        self.coarse_feat_proj = nn.Conv1d(model_channels, model_channels, 1)

        # === Path 3: Statistic encoder ===
        # Maps stat vector (n_stats,) → embedding added to time embedding
        self.stat_encoder = nn.Sequential(
            nn.Linear(n_stats, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # === Prediction pathway ===
        self.pred_proj = nn.Conv1d(1, model_channels, 1)
        self.pred_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout) for _ in range(num_res_blocks)
        ])

        # Output (zero-init)
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags,
                coarse_upsampled=None, stat_vector=None):
        """
        noisy_pred: (B, pred_len)
        time_steps: (t, h) each (B,)
        context_with_lags: (B, 1+n_lags, ctx_len)
        coarse_upsampled: (B, pred_len) or None — upsampled coarse trajectory
        stat_vector: (B, n_stats) or None — future statistics
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        # Path 1: Context
        ctx = self.ctx_proj(context_with_lags)
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.ctx_pool(ctx)
        ctx_spatial = self.ctx_feat_proj(self.ctx_to_pred(ctx))

        # Path 2: Coarse trajectory (if provided)
        coarse_spatial = torch.zeros_like(ctx_spatial)
        if coarse_upsampled is not None:
            c = self.coarse_proj(coarse_upsampled.unsqueeze(1))
            c = self.coarse_block(c, emb)
            coarse_spatial = self.coarse_feat_proj(c)

        # Path 3: Statistics (if provided)
        if stat_vector is not None:
            emb = emb + self.stat_encoder(stat_vector)

        # Prediction pathway
        pred = self.pred_proj(noisy_pred.unsqueeze(1)) + ctx_spatial + coarse_spatial
        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(1)


def conditional_meanflow_loss_v3(net, future_clean, context_with_lags,
                                  coarse_upsampled=None, stat_vector=None,
                                  norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss for v3 model with optional coarse + stat conditioning."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean)
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)

    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)),
                   context_with_lags, coarse_upsampled, stat_vector)

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


class MeanFlowForecasterV3(nn.Module):
    """GluonTS-compatible forecaster with optional coarse + stat conditioning."""

    def __init__(self, net, context_length, prediction_length, num_samples=16,
                 freq="H", n_lags=7):
        super().__init__()
        self.net = net
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags

    def forward(self, past_target, past_observed_values,
                coarse_future=None, coarse_r=None, stat_vector=None, **kwargs):
        """
        coarse_future: (B, pred_len // r) or None
        coarse_r: int, downsampling factor
        stat_vector: (B, n_stats) or None
        """
        from .model_v2 import extract_lag_features

        device = past_target.device
        B = past_target.shape[0]
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
        ctx_with_lags = extract_lag_features(
            past_target, self.context_length, self.freq, self.n_lags) / loc.unsqueeze(1)

        # Prepare coarse conditioning
        coarse_up = None
        if coarse_future is not None and coarse_r is not None:
            coarse_up = upsample(coarse_future / loc, coarse_r, self.prediction_length)

        # Scale stat vector if provided
        # Stats [mean, max, min, std, argmax/L, area/L]:
        # mean, max, min, std, area should be divided by loc (scale-dependent)
        # argmax/L should NOT be divided (it's a position index in [0,1])
        stat_scaled = None
        if stat_vector is not None:
            scale = loc.squeeze(1).unsqueeze(1)  # (B, 1)
            stat_scaled = stat_vector.clone()
            stat_scaled[:, :4] = stat_vector[:, :4] / scale  # mean, max, min, std
            # stat_scaled[:, 4] stays as-is (argmax, dimensionless)
            stat_scaled[:, 5:] = stat_vector[:, 5:] / scale  # area/L

        all_preds = []
        for _ in range(self.num_samples):
            z_1 = torch.randn(B, self.prediction_length, device=device)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), ctx_with_lags, coarse_up, stat_scaled)
            all_preds.append((z_1 - u) * loc)

        return torch.stack(all_preds, dim=1)
