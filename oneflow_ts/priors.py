"""
OneFlow-TS: Resolution-matched temporal priors.

Per-level source distributions matching the spectral structure at each wavelet scale.
Coarsest level gets a correlated prior (smooth), finest level gets N(0,I).
"""
import torch
import torch.nn as nn
import math
from typing import List, Optional


class ResolutionMatchedPrior(nn.Module):
    """
    Per-level noise sampling with resolution-matched temporal correlation.

    For K decomposition levels [A_K, D_K, ..., D_1]:
      - Level A_K (coarsest): Exponential autocorrelation with length_scale
      - Intermediate D_k: Exponential autocorrelation with decreasing length_scale
      - Level D_1 (finest): Standard N(0,I)

    The prior is designed so that transport distance W_2 from prior to target
    is minimized at each resolution level. Coarse levels benefit most from
    correlated priors since their targets are smooth temporal signals.

    Sampling: z_k = L_k @ eps, where eps ~ N(0,I) and Sigma_k = L_k @ L_k^T
    """

    def __init__(self, level_sizes: List[int], length_scale: float = 3.0,
                 scale_decay: float = 0.5, device: Optional[torch.device] = None):
        """
        Args:
            level_sizes: [size_A_K, size_D_K, ..., size_D_1] from get_level_sizes()
            length_scale: correlation length for the coarsest level
            scale_decay: multiply length_scale by this factor for each finer level
            device: torch device
        """
        super().__init__()
        self.level_sizes = level_sizes
        self.num_levels = len(level_sizes)

        # Pre-compute Cholesky factors for each level
        cholesky_factors = []
        current_ls = length_scale
        for k, size in enumerate(level_sizes):
            if k == self.num_levels - 1:
                # Finest level: identity (N(0,I))
                L = torch.eye(size)
            elif current_ls < 0.1:
                # Length scale too small → effectively N(0,I)
                L = torch.eye(size)
            else:
                # Build exponential autocorrelation kernel
                idx = torch.arange(size, dtype=torch.float32)
                dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
                K = torch.exp(-dist / current_ls)
                # Add small jitter for numerical stability
                K = K + 1e-5 * torch.eye(size)
                L = torch.linalg.cholesky(K)
                current_ls *= scale_decay  # Decay for next level

            cholesky_factors.append(L)

        # Register as buffers (move with model, not parameters)
        for k, L in enumerate(cholesky_factors):
            self.register_buffer(f'cholesky_{k}', L)

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """
        Sample noise for each wavelet level.

        Args:
            batch_size: number of samples

        Returns:
            noise_levels: list of (B, level_size_k) tensors
        """
        noise_levels = []
        for k, size in enumerate(self.level_sizes):
            L = getattr(self, f'cholesky_{k}')
            eps = torch.randn(batch_size, size, device=L.device)
            z = eps @ L.T  # (B, size) @ (size, size) -> (B, size)
            noise_levels.append(z)
        return noise_levels

    def log_prob(self, noise_levels: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute log probability of noise samples under the prior.
        Useful for density evaluation.

        Returns:
            log_p: (B,) total log probability
        """
        log_p = torch.zeros(noise_levels[0].shape[0], device=noise_levels[0].device)
        for k, z in enumerate(noise_levels):
            L = getattr(self, f'cholesky_{k}')
            # z = L @ eps, so eps = L^{-1} @ z
            eps = torch.linalg.solve_triangular(L, z.T, upper=False).T
            # log p(z) = log p(eps) - log |det L|
            # log p(eps) = -0.5 * ||eps||^2 - 0.5 * d * log(2pi)
            d = z.shape[-1]
            log_p_eps = -0.5 * (eps ** 2).sum(dim=-1) - 0.5 * d * math.log(2 * math.pi)
            log_det_L = L.diagonal().log().sum()
            log_p += log_p_eps - log_det_L
        return log_p


class IsotropicPrior(nn.Module):
    """Standard N(0,I) prior at all levels. Used as baseline for ablation."""

    def __init__(self, level_sizes: List[int], device: Optional[torch.device] = None):
        super().__init__()
        self.level_sizes = level_sizes
        # Register a dummy buffer so .device works
        self.register_buffer('_device_tracker', torch.zeros(1))

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        device = self._device_tracker.device
        return [torch.randn(batch_size, size, device=device) for size in self.level_sizes]
