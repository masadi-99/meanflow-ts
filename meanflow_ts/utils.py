"""
Utility functions for multi-resolution and statistic conditioning.
"""
import torch
import torch.nn.functional as F


def downsample(y, r):
    """
    Downsample time series by factor r using average pooling.
    y: (B, L) or (L,)
    Returns: (B, L//r) or (L//r,)
    """
    squeeze = y.dim() == 1
    if squeeze:
        y = y.unsqueeze(0)
    # Use avg_pool1d: needs (B, C, L) format
    out = F.avg_pool1d(y.unsqueeze(1), kernel_size=r, stride=r).squeeze(1)
    if squeeze:
        out = out.squeeze(0)
    return out


def upsample(c, r, target_len=None):
    """
    Upsample coarse time series by factor r using linear interpolation.
    c: (B, L_coarse) or (L_coarse,)
    Returns: (B, L_coarse * r) or (L_coarse * r,)
    """
    squeeze = c.dim() == 1
    if squeeze:
        c = c.unsqueeze(0)
    if target_len is None:
        target_len = c.shape[-1] * r
    # Use interpolate: needs (B, C, L) format
    out = F.interpolate(c.unsqueeze(1), size=target_len, mode='linear', align_corners=True).squeeze(1)
    if squeeze:
        out = out.squeeze(0)
    return out


def extract_stats(y):
    """
    Extract summary statistics from a time series.
    y: (B, L)
    Returns: (B, n_stats) where n_stats = 6
    Stats: [mean, max, min, std, argmax/L, area_under_curve/L]
    All normalized to be roughly O(1).
    """
    B, L = y.shape
    stats = []
    stats.append(y.mean(dim=1, keepdim=True))           # mean
    stats.append(y.max(dim=1, keepdim=True).values)      # max
    stats.append(y.min(dim=1, keepdim=True).values)      # min
    stats.append(y.std(dim=1, keepdim=True))              # std
    stats.append(y.argmax(dim=1, keepdim=True).float() / L)  # argmax (normalized)
    stats.append(y.sum(dim=1, keepdim=True) / L)         # area / L
    return torch.cat(stats, dim=1)  # (B, 6)


N_STATS = 6
