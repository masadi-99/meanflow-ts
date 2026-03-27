# Ablation: MeanFlow vs Standard FM (Corrected)

## Setup
- Same architecture: ConditionalMeanFlowNet (1.1M params)
- Same dataset: electricity_nips (ctx=24, pred=24)
- Same optimizer: AdamW, lr=6e-4, grad clip 1.0
- Same adaptive weighting (norm_p=0.75)
- Same EMA (0.9999)
- 600 epochs, 128 batches of 64 per epoch
- Difference: MeanFlow uses JVP self-consistency loss; FM uses standard MSE loss

## Corrected Results (epoch 600)

| Method | CRPS | ND | NRMSE | NFE |
|--------|------|-----|-------|-----|
| FM 1-step | 0.0619 | 0.0659 | 0.499 | 1 |
| **FM 32-step** | **0.0564** | 0.0674 | 0.536 | 32 |
| **MeanFlow 1-step** | **0.0558** | 0.0673 | 0.526 | **1** |
| TSFlow (paper) | 0.0446 | 0.0552 | 0.455 | 32 |

## Key Findings

1. **MeanFlow 1-step ≈ FM 32-step at convergence** (0.056 vs 0.056)
   - MeanFlow achieves in 1 network evaluation what FM needs 32 for
   - This is a **32x inference speedup** for equal quality

2. **FM 1-step is worse** (0.062 vs 0.056)
   - Standard FM can do 1-step via z_0 = z_1 - v(z_1, t=1), but quality degrades
   - MeanFlow's JVP loss genuinely improves 1-step generation

3. **Neither matches TSFlow** (0.045)
   - Gap is due to missing features: lag values, GP prior, S4 backbone, per-series normalization
   - Not a MeanFlow vs FM difference — it's an architecture/features difference

## Full Convergence Trajectory

| Epoch | FM 1-step | FM 32-step | 32/1 ratio |
|-------|----------|-----------|------------|
| 50 | 0.345 | 0.363 | 1.05x |
| 100 | 0.180 | 0.227 | 1.26x |
| 150 | 0.118 | 0.144 | 1.22x |
| 200 | 0.092 | 0.103 | 1.11x |
| 250 | 0.077 | 0.081 | 1.05x |
| 300 | 0.068 | 0.066 | 0.97x ← 32-step overtakes |
| 400 | 0.062 | 0.060 | 0.97x |
| 500 | 0.062 | 0.056 | 0.91x |
| 600 | 0.062 | 0.056 | 0.91x |

## Previous Bug
The original ablation showed "MeanFlow 5x better" — this was caused by a reversed ODE direction in the FM inference code. After fixing (integrate t=1→0, not t=0→1), FM performs much better.
