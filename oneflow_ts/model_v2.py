"""
OneFlow-TS v2: Single-network multi-resolution model.

Instead of separate per-level heads, this model:
1. Takes the full noisy prediction as input (like original MeanFlow-TS)
2. Internally decomposes context into wavelet features for multi-scale conditioning
3. Predicts velocity in the ORIGINAL time domain
4. Uses resolution-matched prior for noise initialization

This avoids the gradient conflict issue of per-level JVP while still
leveraging multi-resolution structure for conditioning.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .wavelet import dwt_decompose, idwt_reconstruct, get_level_sizes


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""
    def __init__(self, channels, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, channels * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class MultiScaleContextEncoder(nn.Module):
    """
    Encode context at multiple wavelet scales.
    Provides richer conditioning than single-scale encoding.
    """
    def __init__(self, ctx_len, n_lags, model_channels, emb_dim,
                 levels=2, dropout=0.1):
        super().__init__()
        self.levels = levels
        ctx_in = 1 + n_lags

        # Full-resolution context encoder
        self.ctx_proj = nn.Conv1d(ctx_in, model_channels, 1)
        self.ctx_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout) for _ in range(2)
        ])

        # Multi-scale context: encode context at coarser scales too
        # Wavelet decompose the context, encode each level, fuse
        self.level_sizes = get_level_sizes(ctx_len, levels)
        self.scale_encoders = nn.ModuleList()
        for size in self.level_sizes:
            self.scale_encoders.append(nn.Sequential(
                nn.Linear(size, ctx_len),  # project to full resolution
                nn.SiLU(),
            ))

        # Fusion: combine multi-scale features
        n_scales = len(self.level_sizes)
        self.scale_fusion = nn.Conv1d(model_channels * (1 + n_scales), model_channels, 1)

        # Global pooling for embedding
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(model_channels, emb_dim),
        )

    def forward(self, context_with_lags, emb):
        """
        context_with_lags: (B, 1+n_lags, ctx_len)
        emb: (B, emb_dim)

        Returns:
            ctx_features: (B, model_channels, ctx_len)
            ctx_emb: (B, emb_dim) — global context embedding
        """
        B = context_with_lags.shape[0]

        # Full-resolution encoding
        ctx = self.ctx_proj(context_with_lags)
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)

        # Multi-scale encoding: decompose context channel 0 (main series)
        main_ctx = context_with_lags[:, 0, :]  # (B, ctx_len)
        ctx_levels = dwt_decompose(main_ctx, self.levels)

        scale_features = []
        for k, (level_coeffs, encoder) in enumerate(zip(ctx_levels, self.scale_encoders)):
            # Project each level back to full ctx_len
            projected = encoder(level_coeffs)  # (B, ctx_len)
            # Expand to model_channels
            scale_features.append(
                projected.unsqueeze(1).expand(-1, ctx.shape[1], -1)
            )

        # Fuse: concatenate all scales + original, then project
        all_features = torch.cat([ctx] + scale_features, dim=1)
        ctx_fused = self.scale_fusion(all_features)

        ctx_emb = self.ctx_pool(ctx_fused)
        return ctx_fused, ctx_emb


class OneFlowTSNetV2(nn.Module):
    """
    Single-network multi-resolution model.

    Key differences from V1:
    - Single velocity output in time domain (no per-level heads)
    - Multi-scale context conditioning via wavelet decomposition of context
    - Compatible with standard MeanFlow JVP loss (no per-level JVP)
    - Prior can still be resolution-matched (noise init in wavelet domain)
    """

    def __init__(self, pred_len=24, ctx_len=24, n_lags=7,
                 levels=2, model_channels=128, num_res_blocks=4,
                 time_emb_dim=64, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.levels = levels

        emb_dim = time_emb_dim * 4

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # Multi-scale context encoder
        self.ctx_encoder = MultiScaleContextEncoder(
            ctx_len, n_lags, model_channels, emb_dim, levels, dropout
        )

        # Context-to-prediction projection
        self.ctx_to_pred = nn.Linear(ctx_len, pred_len)
        self.ctx_feat_proj = nn.Conv1d(model_channels, model_channels, 1)

        # Prediction pathway (same as original MeanFlow-TS)
        self.pred_proj = nn.Conv1d(1, model_channels, 1)
        self.pred_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout)
            for _ in range(num_res_blocks)
        ])

        # Output
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_pred, time_steps, context_with_lags):
        """
        noisy_pred: (B, pred_len) — noisy future in time domain
        time_steps: (t, h) each (B,)
        context_with_lags: (B, 1+n_lags, ctx_len)

        Returns:
            velocity: (B, pred_len) — predicted average velocity in time domain
        """
        t, h = time_steps
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)

        # Multi-scale context encoding
        ctx_features, ctx_emb = self.ctx_encoder(context_with_lags, emb)
        emb = emb + ctx_emb

        # Project context features to prediction space
        ctx_spatial = self.ctx_feat_proj(self.ctx_to_pred(ctx_features))

        # Prediction pathway
        pred = self.pred_proj(noisy_pred.unsqueeze(1)) + ctx_spatial
        for block in self.pred_blocks:
            pred = block(pred, emb)

        return self.out_proj(F.silu(self.out_norm(pred))).squeeze(1)


class OneFlowForecasterV2(nn.Module):
    """
    Forecaster using V2 model with resolution-matched prior.

    Noise is sampled in wavelet domain (per-level matched priors),
    then IDWT-reconstructed to time domain before passing to the network.
    """

    def __init__(self, net, prior, context_length, prediction_length,
                 num_samples=16, freq="H", n_lags=7):
        super().__init__()
        self.net = net
        self.prior = prior
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags

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

        all_preds = []
        for _ in range(self.num_samples):
            # Sample noise in wavelet domain with matched priors
            noise_levels = self.prior.sample(B)
            # Reconstruct to time domain
            z_1 = idwt_reconstruct(noise_levels)

            # One-step generation in time domain
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            u = self.net(z_1, (t, h), ctx_with_lags)
            pred = (z_1 - u) * loc
            all_preds.append(pred)

        return torch.stack(all_preds, dim=1)
