"""
OneFlow-TS: Haar wavelet decomposition/reconstruction via PyTorch convolutions.

No external dependencies (no pywt). Exact reconstruction for any even-length input.
Supports multi-level decomposition.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import math


SQRT2_INV = 1.0 / math.sqrt(2.0)


def haar_dwt_1level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-level Haar DWT on last dimension.

    Args:
        x: (B, L) where L is even

    Returns:
        approx: (B, L//2) — low-frequency approximation coefficients
        detail: (B, L//2) — high-frequency detail coefficients
    """
    assert x.shape[-1] % 2 == 0, f"Input length must be even, got {x.shape[-1]}"
    even = x[..., 0::2]
    odd = x[..., 1::2]
    approx = (even + odd) * SQRT2_INV
    detail = (even - odd) * SQRT2_INV
    return approx, detail


def haar_idwt_1level(approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """
    Single-level inverse Haar DWT.

    Args:
        approx: (B, L//2)
        detail: (B, L//2)

    Returns:
        x: (B, L)
    """
    even = (approx + detail) * SQRT2_INV
    odd = (approx - detail) * SQRT2_INV
    B = even.shape[0]
    L = even.shape[-1] * 2
    x = torch.empty(B, L, device=even.device, dtype=even.dtype)
    x[..., 0::2] = even
    x[..., 1::2] = odd
    return x


def dwt_decompose(x: torch.Tensor, levels: int = 2) -> List[torch.Tensor]:
    """
    Multi-level Haar DWT decomposition.

    Args:
        x: (B, pred_len) where pred_len must be divisible by 2^levels

    Returns:
        coeffs: [A_K, D_K, D_{K-1}, ..., D_1]
            A_K: (B, pred_len // 2^K) — coarsest approximation
            D_k: (B, pred_len // 2^k) — detail at level k

    Example (pred_len=24, levels=2):
        [A2: (B,6), D2: (B,6), D1: (B,12)]
    """
    assert x.shape[-1] % (2 ** levels) == 0, (
        f"pred_len={x.shape[-1]} must be divisible by 2^levels={2**levels}"
    )
    details = []
    current = x
    for _ in range(levels):
        current, d = haar_dwt_1level(current)
        details.append(d)
    # Return [A_K, D_K, D_{K-1}, ..., D_1]
    return [current] + details[::-1]


def idwt_reconstruct(coeffs: List[torch.Tensor]) -> torch.Tensor:
    """
    Multi-level inverse Haar DWT reconstruction.

    Args:
        coeffs: [A_K, D_K, D_{K-1}, ..., D_1] as returned by dwt_decompose

    Returns:
        x: (B, pred_len) — reconstructed signal
    """
    # coeffs[0] = A_K, coeffs[1] = D_K, coeffs[2] = D_{K-1}, ..., coeffs[-1] = D_1
    current = coeffs[0]  # Start with coarsest approximation
    # Details are stored as [D_K, D_{K-1}, ..., D_1]
    # We reconstruct from coarsest to finest
    for detail in coeffs[1:]:
        current = haar_idwt_1level(current, detail)
    return current


def get_level_sizes(pred_len: int, levels: int = 2) -> List[int]:
    """
    Return the size of each coefficient level.

    Args:
        pred_len: prediction length
        levels: number of decomposition levels

    Returns:
        sizes: [size_A_K, size_D_K, ..., size_D_1]

    Example: get_level_sizes(24, 2) -> [6, 6, 12]
    """
    sizes = []
    current = pred_len
    detail_sizes = []
    for _ in range(levels):
        current = current // 2
        detail_sizes.append(current)
    # [A_K size, D_K size, ..., D_1 size]
    return [current] + detail_sizes[::-1]


def get_level_names(levels: int = 2) -> List[str]:
    """Return human-readable names for each level."""
    names = [f"A{levels}"]
    for k in range(levels, 0, -1):
        names.append(f"D{k}")
    return names
