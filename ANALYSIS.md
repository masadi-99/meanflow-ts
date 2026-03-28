# Deep Analysis: Why Multi-Resolution Didn't Work

## Bug Audit

### Bugs Found

**1. Redundant forward pass in `oneflow_loss()` (minor, lines 88-91)**
```python
u_preds = net(noisy_levels, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_with_lags)
```
This forward pass computes predictions but the result `u_preds` is NEVER USED. The actual computation happens in the per-level JVP loop below. This wastes ~30% of computation per training step in V1.

**Impact**: Slows training by ~30% but doesn't affect correctness or final quality.

### No-Bugs Confirmed

**Per-level JVP is mathematically correct.** Tested by comparing per-level JVP loss against flat JVP loss with identical noise — they match exactly (53.82 vs 53.82). This is because the V1 architecture has truly independent level heads (no cross-level dependency through the noisy input), so per-level partial derivatives equal the full derivative.

**Wavelet decomposition is correct.** Reconstruction error < 1e-7, Parseval energy conservation verified.

**Prior sampling is correct.** Marginal variance = 1.0, correlation structure matches the exponential kernel.

---

## Root Cause Analysis

### Finding 1: Catastrophic Noise-to-Signal Ratio Imbalance

The electricity_nips data, after normalization and 2-level Haar wavelet decomposition:

| Level | Target Mean | Target Std | Noise Std | Noise/Signal Ratio |
|-------|-----------|-----------|-----------|-------------------|
| A2 (trend) | 2.024 | 0.712 | 1.0 | 1.4x |
| D2 (seasonal) | -0.005 | 0.214 | 1.0 | **4.7x** |
| D1 (detail) | -0.003 | 0.105 | 1.0 | **9.5x** |

The fine-level noise is **5-10x larger than the target signal**. This means the fine-level flow maps must "compress" the noise distribution by an enormous factor. With equal parameter allocation across levels, most capacity is wasted fighting the noise imbalance rather than learning the data distribution.

**98.5% of signal energy is at level A2** (the trend). The model spends equal parameters on D2 (1.0% energy) and D1 (0.5% energy) as on A2 (98.5% energy).

### Finding 2: Massive Transport Distance Mismatch

The Wasserstein-2 transport distance from prior to target at each level:

| Level | W2 with N(0,I) noise | W2 with variance-matched noise | Ratio |
|-------|---------------------|-------------------------------|-------|
| A2 (trend) | 2.044 | 2.024 | 1.0x (no difference) |
| D2 (seasonal) | 0.786 | 0.005 | **171x harder** |
| D1 (detail) | 0.895 | 0.003 | **276x harder** |

The fine levels have almost zero signal (mean ≈ 0, std ≈ 0.1), yet we're asking the flow map to transport from N(0,1) to something very close to a point mass. With variance-matched noise N(0, 0.01), the transport would be trivial (just predict zero). But with N(0,1) noise, the model must learn to suppress 95% of the input — a hard task that dominates the loss.

**This is the core design flaw.** The resolution-matched prior matched CORRELATION structure but NOT VARIANCE. It should have matched both.

### Finding 3: Cross-Level Correlations Are Significant but Ignored

| Correlation | Mean |corr| | Max |corr| |
|-------------|------------|------------|
| A2 ↔ D2 | 0.288 | 0.697 |
| A2 ↔ D1 | 0.203 | 0.585 |
| D2 ↔ D1 | 0.184 | 0.629 |

Cross-level correlations are STRONG (up to 0.70). The V1 architecture generates each level independently (conditioned only on context, not on other levels). This throws away information — knowing the trend (A2) should inform the detail pattern (D1), but V1 can't use this.

### Finding 4: The Backbone Already Captures Multi-Scale Features

- Conv1D kernel=3, 4 ResBlocks (8 conv layers) → receptive field = 17
- Prediction length = 24 → RF covers 71% of the sequence
- For level A2 (6 coefficients), RF > sequence length — the backbone can see everything

The Conv1D backbone with FiLM conditioning ALREADY learns multi-scale temporal features. The wavelet decomposition doesn't provide new information — it just rearranges existing information into a harder-to-learn format.

### Finding 5: High Autocorrelation Makes Time-Domain Learning Easy

Time-domain autocorrelation of normalized electricity targets:
```
Lag 0: 1.000
Lag 1: 0.928
Lag 2: 0.830
Lag 3: 0.725
Lag 4: 0.607
Lag 5: 0.482
```

ACF of 0.93 at lag-1 means consecutive time points are highly correlated. The Conv1D backbone naturally exploits this through local convolutions. Adding wavelet decomposition doesn't help because the backbone already "sees" temporal smoothness through the high autocorrelation structure.

### Finding 6: V3 (Noise-Only) Tells Us the Noise Distribution Doesn't Matter

All three noise distributions converge to the same ~0.047 CRPS:
- noise-wavelet: 0.0466
- baseline N(0,I): 0.0469
- noise-corr: 0.0472

This means the MeanFlow training is robust to the noise distribution. The model learns to transport from ANY reasonable noise distribution to the target. Changing the noise changes the early training trajectory but not the final result. This is expected from flow matching theory — the optimal velocity field adapts to any source distribution.

---

## Why Each Approach Failed

### V1 (Per-Level Heads): Three Compounding Problems
1. **Noise imbalance**: Fine levels have 5-10x more noise than signal
2. **No cross-level flow**: Trend can't inform details
3. **Capacity fragmentation**: Equal params for 98.5% vs 0.5% energy levels

### V2 (Multi-Scale Context): Redundant Information
The multi-scale context encoder decomposes the CONTEXT window into wavelets. But the context encoder already has 17-point receptive field on a 24-point window — it already captures all scales. The wavelet decomposition adds 1.2M params but no new information.

### V3 (Structured Noise): Correct but Ineffective
The noise distribution doesn't affect converged quality because MeanFlow adapts. The slight early-convergence advantage of correlated noise (10% better CRPS at epoch 100) vanishes by epoch 300.

---

## Hypotheses for Fixing It

### Hypothesis 1: Variance-Matched Per-Level Noise (High Confidence)

**The Fix**: Scale noise at each level to match target variance:
```python
# Instead of: e_k ~ N(0, I)
# Use: e_k ~ N(0, sigma_k^2 * I) where sigma_k = target_std_k
# Level A2: sigma = 0.71
# Level D2: sigma = 0.21
# Level D1: sigma = 0.11
```

**Why It Should Work**: Reduces W2 transport distance by 170-280x at fine levels. The flow maps no longer need to learn "noise compression" — they focus on learning the DATA distribution.

**Risk**: Needs adaptation during training (target std varies per series). Could use running statistics or per-batch estimation.

**Estimated Impact**: Could reduce V1's deficit from 13% to 2-5%.

### Hypothesis 2: Coarse-to-Fine Generation (High Confidence)

**The Fix**: Generate A2 (trend) first, then condition D2 and D1 on the generated A2:
```python
# Step 1: Generate trend
a2_pred = z_a2 - u_a2(z_a2, context)
# Step 2: Generate details conditioned on trend
d2_pred = z_d2 - u_d2(z_d2, context, a2_pred)
d1_pred = z_d1 - u_d1(z_d1, context, a2_pred, d2_pred)
```

**Why It Should Work**: Cross-level correlations are strong (up to 0.70). Knowing the trend significantly reduces uncertainty in the details. This is what MG-TSD and mr-Diff do (and they work well).

**Risk**: Makes inference sequential (3 steps instead of 1), partially defeating the speed advantage. But each step is on a smaller sequence, so total compute is similar.

### Hypothesis 3: Longer Prediction Horizons (Medium Confidence)

**The Fix**: Test on pred_len=336 or 720 (ETT-style long-horizon forecasting).

**Why It Should Work**: At pred_len=24, the Conv1D backbone covers 71% of the sequence. At pred_len=336, it covers only 5%. Multi-resolution decomposition would provide ESSENTIAL global structure that the backbone can't learn locally.

**Risk**: The current MeanFlow-TS architecture wasn't designed for long horizons. Would need architectural changes beyond just adding wavelets.

### Hypothesis 4: Different Backbone (Medium Confidence)

**The Fix**: Replace Conv1D with a backbone that has limited receptive field (e.g., local attention, windowed transformer). Then add wavelet decomposition to provide global context.

**Why It Should Work**: If the backbone CAN'T see all scales, the wavelet decomposition becomes NECESSARY. Currently it's redundant.

### Hypothesis 5: Different Datasets (Medium Confidence)

**The Fix**: Test on solar_nips and exchange_rate_nips where the baseline has larger gaps to TSFlow.

**Why It Should Work**: Electricity_nips has very high autocorrelation (0.93 at lag-1), making it easy for any architecture. Solar and exchange have different temporal structures that might benefit more from explicit multi-scale modeling.

**Risk**: The same baseline may also struggle more, not just the multi-res variant.

### Hypothesis 6: Adaptive Level Weighting (Low Confidence)

**The Fix**: Weight the per-level loss by energy fraction:
```python
level_weights = [0.985, 0.010, 0.005]  # Proportional to energy
```

**Why It Might Work**: Currently equal weighting means 50% of gradient signal comes from levels with 1.5% of energy. Energy-proportional weighting would focus capacity where it matters.

**Risk**: The adaptive weighting (norm_p=0.75) already partially addresses this by downweighting easy levels. May not add much.

---

## Recommendation for Next Steps

**Highest-impact, lowest-risk experiment**: Test Hypothesis 1 (variance-matched noise) + Hypothesis 2 (coarse-to-fine) together. This addresses the two biggest problems (noise imbalance + missing cross-level information) simultaneously.

**If that doesn't work**: The multi-resolution approach is fundamentally unsuited to short-horizon (pred_len=24) forecasting with a backbone that already has near-full receptive field. Consider:
- Pivoting to longer horizons where multi-resolution IS needed
- Pivoting to a completely different innovation angle for the NeurIPS submission
