"""
OneFlow-TS: Multi-resolution one-step flow map for time series forecasting.

Decomposes forecast targets into wavelet resolution levels and learns a
one-step flow map per level with a shared context encoder backbone.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .wavelet import dwt_decompose, idwt_reconstruct, get_level_sizes, get_level_names


# ============================================================
# Building blocks (reused from MeanFlow-TS)
# ============================================================

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


# ============================================================
# Per-level velocity head
# ============================================================

class LevelHead(nn.Module):
    """
    Small network for per-level velocity prediction.
    Predicts average velocity u_k for one wavelet resolution level.
    """
    def __init__(self, level_size, model_channels, emb_dim,
                 num_blocks=2, dropout=0.1):
        super().__init__()
        self.level_size = level_size
        self.input_proj = nn.Conv1d(1, model_channels, 1)
        self.ctx_proj = nn.Conv1d(model_channels, model_channels, 1)
        self.blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_proj = nn.Conv1d(model_channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy: torch.Tensor, emb: torch.Tensor,
                ctx_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy: (B, level_size) — noisy wavelet coefficients
            emb: (B, emb_dim) — time + context embedding
            ctx_features: (B, model_channels, level_size) — projected context

        Returns:
            velocity: (B, level_size) — predicted average velocity
        """
        x = self.input_proj(noisy.unsqueeze(1))  # (B, C, level_size)
        x = x + self.ctx_proj(ctx_features)       # add context
        for block in self.blocks:
            x = block(x, emb)
        return self.out_proj(F.silu(self.out_norm(x))).squeeze(1)


# ============================================================
# Main model
# ============================================================

class OneFlowTSNet(nn.Module):
    """
    Multi-resolution one-step flow map for time series forecasting.

    Architecture:
      Context → [Shared Backbone] → context_embedding, per-level context features
      For each wavelet level k:
        noise_k → [LevelHead_k](conditioned on context + time) → velocity_k

    All levels share the backbone context encoder but have independent heads.
    """

    def __init__(self, pred_len: int = 24, ctx_len: int = 24, n_lags: int = 7,
                 levels: int = 2, model_channels: int = 128,
                 head_channels: int = 64, num_ctx_blocks: int = 2,
                 num_head_blocks: int = 2, time_emb_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.n_lags = n_lags
        self.levels = levels
        self.level_sizes = get_level_sizes(pred_len, levels)
        self.level_names = get_level_names(levels)

        emb_dim = time_emb_dim * 4

        # --- Dual time embedding (t, h) ---
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # --- Shared context encoder ---
        ctx_in_channels = 1 + n_lags
        self.ctx_proj = nn.Conv1d(ctx_in_channels, model_channels, 1)
        self.ctx_blocks = nn.ModuleList([
            ResBlock1D(model_channels, emb_dim, dropout)
            for _ in range(num_ctx_blocks)
        ])
        self.ctx_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_channels, emb_dim),
        )

        # --- Per-level context projections ---
        # Project context features (B, model_channels, ctx_len) to each level's size
        self.ctx_to_level = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ctx_len, size),
            )
            for size in self.level_sizes
        ])

        # --- Per-level heads ---
        self.level_heads = nn.ModuleList([
            LevelHead(
                level_size=size,
                model_channels=head_channels,
                emb_dim=emb_dim,
                num_blocks=num_head_blocks,
                dropout=dropout,
            )
            for size in self.level_sizes
        ])

        # Channel adaptation: backbone uses model_channels, heads use head_channels
        self.channel_adapt = nn.ModuleList([
            nn.Conv1d(model_channels, head_channels, 1)
            for _ in self.level_sizes
        ])

    def forward(self, noisy_levels: List[torch.Tensor],
                time_steps: Tuple[torch.Tensor, torch.Tensor],
                context_with_lags: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            noisy_levels: list of (B, level_size_k) — noisy wavelet coefficients per level
            time_steps: (t, h) each (B,) — MeanFlow dual time
            context_with_lags: (B, 1+n_lags, ctx_len) — context with lag features

        Returns:
            velocities: list of (B, level_size_k) — predicted avg velocity per level
        """
        t, h = time_steps

        # Time embedding
        emb = torch.cat([self.time_emb(t), self.time_emb(h)], dim=-1)
        emb = self.time_mlp(emb)  # (B, emb_dim)

        # Context encoding (shared)
        ctx = self.ctx_proj(context_with_lags)  # (B, model_channels, ctx_len)
        for block in self.ctx_blocks:
            ctx = block(ctx, emb)
        emb = emb + self.ctx_pool(ctx)  # Add global context to embedding

        # Per-level velocity prediction
        velocities = []
        for k, (head, ctx_proj, ch_adapt) in enumerate(
            zip(self.level_heads, self.ctx_to_level, self.channel_adapt)
        ):
            # Project context to level's spatial size
            ctx_k = ctx_proj(ctx)              # (B, model_channels, level_size_k)
            ctx_k = ch_adapt(ctx_k)            # (B, head_channels, level_size_k)
            vel_k = head(noisy_levels[k], emb, ctx_k)
            velocities.append(vel_k)

        return velocities

    def forward_flat(self, noisy_flat: torch.Tensor,
                     time_steps: Tuple[torch.Tensor, torch.Tensor],
                     context_with_lags: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: takes flat noisy input, decomposes, and returns flat velocity.
        Useful for compatibility with existing training code.

        Args:
            noisy_flat: (B, pred_len) — noisy prediction in time domain
            time_steps: (t, h)
            context_with_lags: (B, 1+n_lags, ctx_len)

        Returns:
            velocity_flat: (B, pred_len) — velocity in time domain
        """
        noisy_levels = dwt_decompose(noisy_flat, self.levels)
        vel_levels = self.forward(noisy_levels, time_steps, context_with_lags)
        return idwt_reconstruct(vel_levels)


# ============================================================
# Forecaster (GluonTS-compatible)
# ============================================================

class OneFlowForecaster(nn.Module):
    """
    Wraps OneFlowTSNet for GluonTS evaluation.
    Generates probabilistic forecasts via per-level one-step sampling + IDWT.
    """

    def __init__(self, net: OneFlowTSNet, prior,
                 context_length: int, prediction_length: int,
                 num_samples: int = 16, freq: str = "H", n_lags: int = 7):
        super().__init__()
        self.net = net
        self.prior = prior
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.freq = freq
        self.n_lags = n_lags

    def _extract_lag_features(self, past_target):
        """Extract lag features — reuse logic from model_v2."""
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

        # Normalize
        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)

        # Extract and scale lag features
        ctx_with_lags = self._extract_lag_features(past_target) / loc.unsqueeze(1)

        # Generate samples
        all_preds = []
        for _ in range(self.num_samples):
            # Sample per-level noise from prior
            noise_levels = self.prior.sample(B)

            # One-step generation per level
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            vel_levels = self.net(noise_levels, (t, h), ctx_with_lags)

            # x_k = z_k - u_k (one-step flow map per level)
            pred_levels = [z - u for z, u in zip(noise_levels, vel_levels)]

            # Reconstruct from wavelet coefficients
            pred = idwt_reconstruct(pred_levels)

            # Unscale
            all_preds.append(pred * loc)

        return torch.stack(all_preds, dim=1)

    @torch.no_grad()
    def generate_per_level(self, past_target, num_samples=None):
        """
        Generate per-level samples (for scale-decomposed uncertainty analysis).

        Returns:
            per_level_samples: list of (B, S, level_size_k) per level
            full_samples: (B, S, pred_len) reconstructed forecasts
        """
        num_samples = num_samples or self.num_samples
        device = past_target.device
        B = past_target.shape[0]

        context = past_target[:, -self.context_length:]
        loc = context.abs().mean(dim=1, keepdim=True).clamp(min=0.01)
        ctx_with_lags = self._extract_lag_features(past_target) / loc.unsqueeze(1)

        per_level_samples = [[] for _ in self.net.level_sizes]
        full_samples = []

        for _ in range(num_samples):
            noise_levels = self.prior.sample(B)
            t = torch.ones(B, device=device)
            h = torch.ones(B, device=device)
            vel_levels = self.net(noise_levels, (t, h), ctx_with_lags)

            pred_levels = [z - u for z, u in zip(noise_levels, vel_levels)]
            for k, p in enumerate(pred_levels):
                per_level_samples[k].append(p * loc)

            pred = idwt_reconstruct(pred_levels) * loc
            full_samples.append(pred)

        per_level_samples = [torch.stack(s, dim=1) for s in per_level_samples]
        full_samples = torch.stack(full_samples, dim=1)
        return per_level_samples, full_samples
