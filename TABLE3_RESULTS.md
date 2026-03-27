# Table 3: Forecasting CRPS (lower is better)

## MeanFlow-TS vs Standard FM vs TSFlow

All methods use the same ConditionalMeanFlowNet architecture (1.1M params).
MeanFlow and FM differ only in loss function. TSFlow uses a different architecture (S4 + GP prior + lag features).

| Dataset | MeanFlow (1 NFE) | FM (1 NFE) | MF/FM | TSFlow (32 NFE) | MF/TSFlow |
|---------|-----------------|-----------|-------|-----------------|-----------|
| electricity | **0.056** | 0.063 | 0.89x | 0.045 | 1.24x |
| solar | **0.397** | 0.460 | 0.86x | 0.341 | 1.16x |
| traffic | **0.129** | 0.143 | 0.90x | 0.082 | 1.57x |
| exchange | **0.010** | 0.011 | 0.91x | 0.005 | 2.00x |
| m4_hourly | **0.031** | 0.035 | 0.89x | 0.029 | 1.07x |

## Key Findings

1. **MeanFlow beats standard FM on ALL 5 datasets** (8-14% better CRPS)
2. **MeanFlow uses 1 network evaluation** vs TSFlow's 32
3. **m4_hourly gap is only 7%** from TSFlow with 32x fewer inference steps
4. The gap to TSFlow is mostly due to architecture/features, not the flow matching method

## Training Details
- 600 epochs, 128 batches of 64 per epoch
- AdamW optimizer, lr=6e-4, grad clip 1.0
- EMA decay 0.9999
- MeanFlow: JVP self-consistency loss (adaptive weighting, norm_p=0.75)
- FM: Standard MSE loss (same adaptive weighting for fair comparison)

## Ablation Summary (electricity_nips only)
| Method | CRPS | NFE |
|--------|------|-----|
| FM 1-step | 0.062 | 1 |
| FM 32-step | 0.056 | 32 |
| **MeanFlow 1-step** | **0.056** | **1** |
| TSFlow | 0.045 | 32 |

MeanFlow 1-step matches FM 32-step quality — a 32x inference speedup.
