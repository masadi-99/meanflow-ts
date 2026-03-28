# MeanFlow-TS: One-Step Time Series Forecasting via MeanFlow

First application of MeanFlow (Geng et al., 2025) to time series forecasting. Achieves probabilistic forecasting with **1-step inference** (32x fewer network evaluations than TSFlow).

## Results (Table 3 — CRPS, lower is better)

| Method | NFE | Electricity | Exchange | Solar | Traffic | M4 (H) |
|--------|-----|------------|----------|-------|---------|--------|
| TSFlow (OU) | 32 | **0.045** | **0.005** | **0.341** | **0.082** | 0.029 |
| **MeanFlow-TS** | **1** | 0.055 | 0.011 | 0.377 | 0.132 | **0.027** |

## Known Limitations

- **Normalization mismatch**: We normalize by context window mean; TSFlow uses global per-series means cached during training. This may account for part of the CRPS gap.
- **No lag features**: TSFlow passes lagged values (same-hour from 1-28 days ago) as features. Our model only sees the immediate context window.
- **No GP prior**: TSFlow initializes from a Gaussian Process posterior; we initialize from standard Gaussian noise.
- **Simpler backbone**: 1D convolutions vs TSFlow's S4 (Structured State Space) blocks.

## Quick Start

```bash
# Train conditional forecasting on a dataset
python experiments/train_forecasting.py electricity_nips --epochs 600

# Evaluate all metrics
python experiments/eval_forecasting.py

# Train unconditional generation (for Table 1 & 2)
python experiments/train_unconditional.py electricity_nips --epochs 200

# Evaluate W2 distance and LPS
python experiments/eval_table1_table2.py electricity_nips

# Run ablation: standard FM vs MeanFlow
python experiments/ablation_standard_fm.py
```

## Architecture

- **Conditional MeanFlow**: Context encoder (2 ResBlocks + pooling) -> Prediction decoder (4 ResBlocks with FiLM conditioning)
- **Dual time embedding**: (t, h=t-r) with sinusoidal positional encoding, concatenated (same as original MeanFlow)
- **JVP self-consistency loss**: Exact MeanFlow training objective via `torch.func.jvp`
- **1-step inference**: `z_0 = z_1 - u(z_1, t=1, h=1, context)`

## References

- MeanFlow: Geng et al., "Mean Flows for One-step Generative Modeling", 2025
- TSFlow: Kollovieh et al., "Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting", ICLR 2025

---

# Multi-Resolution Experiment (this branch)

This branch documents a systematic investigation into whether **multi-resolution decomposition** can improve MeanFlow-TS quality. The hypothesis was that decomposing forecasts into wavelet resolution levels (trend, seasonal, detail) and applying one-step flow maps per level would make the generation task easier and improve CRPS.

## Motivation

The core idea was to build **OneFlow-TS** — a method for structured probabilistic time series forecasting via multi-resolution flow maps. The key hypotheses:

1. **Per-level generation is easier**: Each wavelet level has simpler structure (trends are smooth, details are Gaussian-like), so per-level one-step flow maps should be more accurate than a monolithic one-step map.
2. **Resolution-matched priors help**: Using correlated noise (matched to the spectral structure of each level) should reduce the transport distance W2, making one-step generation more accurate.
3. **Structured uncertainty**: Multi-resolution decomposition would enable scale-decomposed uncertainty quantification and hierarchical scenario generation.

## New Code

### `oneflow_ts/` package

| File | Purpose |
|------|---------|
| `wavelet.py` | Haar DWT/IDWT via PyTorch convolutions. Zero dependencies, exact reconstruction, Parseval-verified. |
| `priors.py` | Resolution-matched temporal priors. Per-level noise sampling with exponential autocorrelation kernels + Cholesky factorization. |
| `model.py` | **V1**: Multi-resolution flow map with shared backbone + separate per-level velocity heads (LevelHead). 848K params. |
| `model_v2.py` | **V2**: Single-network model with multi-scale context encoder (wavelet decomposition of context for conditioning). 1.2M params. |
| `loss.py` | Per-level MeanFlow JVP loss + flat-interface fallback loss. |
| `__init__.py` | Package exports. |

### Experiment scripts

| Script | Purpose |
|--------|---------|
| `experiments/validate_multires.py` | Day 1-2 GO/NO-GO validation comparing single-res baseline vs multi-res variants. |
| `experiments/iterate_fixes.py` | Iteration 1: Tests flat-loss, big-heads, and baseline at 100 epochs. |
| `experiments/iterate_v2.py` | Iteration 2: Tests V2 architecture (multi-scale context) at 200 epochs. |
| `experiments/iterate_v3_noise_only.py` | Iteration 3: Tests structured noise only (no architecture change) at 600 epochs. |

## Experiments and Results

### Iteration 1: V1 Per-Level Heads (100 epochs, electricity_nips)

**Approach**: Decompose forecast targets into 2-level Haar wavelet coefficients [A2, D2, D1]. Each level has its own velocity head. Shared context encoder backbone. Three loss variants tested.

| Variant | CRPS @100ep | vs Baseline | Params |
|---------|------------|-------------|--------|
| **baseline** (single-res MeanFlow v4) | **0.208** | — | 1.1M |
| flat-loss (V1 multi-res, single JVP) | 0.235 | 13% worse | 848K |
| flat-loss+big-heads (V1, 128ch heads) | 0.243 | 17% worse | 1.3M |
| per-level JVP (V1, original) | 0.239 | 15% worse | 848K |

**Finding**: Per-level decomposition makes the problem HARDER, not easier. The per-level heads receive conflicting gradients, and the loss oscillates during training. Larger heads don't help. The flat-loss (single JVP on full reconstructed signal) is slightly better but still behind baseline.

### Iteration 2: V2 Multi-Scale Context (200 epochs, electricity_nips)

**Approach**: Keep a single-network architecture (same forward signature as original MeanFlow-TS) but add multi-scale context conditioning: wavelet-decompose the context window and project each scale back to the full resolution for richer conditioning. Also test resolution-matched noise initialization (sample noise in wavelet domain, IDWT to time domain).

| Variant | CRPS @100ep | CRPS @200ep | vs Baseline @200ep |
|---------|------------|------------|-------------------|
| **baseline** | 0.225 | **0.092** | — |
| v2-matched (multi-scale ctx + matched noise) | 0.225 | 0.101 | 10% worse |
| v2-iso (multi-scale ctx + N(0,I) noise) | 0.242 | 0.105 | 14% worse |

**Finding**: Multi-scale context encoding adds parameters (1.2M vs 1.1M) and computation but doesn't improve quality. The extra wavelet-decomposed context features don't provide information that the standard context encoder can't learn on its own.

### Iteration 3: Noise-Only (600 epochs, electricity_nips)

**Approach**: The cleanest test — use the EXACT same architecture (ConditionalMeanFlowNetV2, 1.1M params) for all variants. Only change the noise distribution used during training and inference:
- **noise-wavelet**: Sample noise in wavelet domain with per-level matched priors, IDWT to time domain
- **noise-corr**: Sample noise from N(0, K) with exponential autocorrelation kernel (length_scale=4)
- **baseline**: Standard N(0, I)

Full convergence trajectory (CRPS, lower is better):

| Epoch | noise-wavelet | noise-corr | baseline |
|-------|-------------|-----------|----------|
| 100 | 0.207 | **0.190** | 0.212 |
| 200 | 0.087 | **0.086** | 0.088 |
| 300 | **0.056** | 0.057 | 0.054 |
| 400 | 0.050 | 0.051 | 0.050 |
| 500 | **0.047** | 0.049 | 0.048 |
| **600** | **0.0466** | 0.0472 | 0.0469 |

**Final results at 600 epochs:**

| Variant | CRPS | vs Baseline | vs TSFlow (32 NFE) |
|---------|------|-------------|-------------------|
| **noise-wavelet** | **0.04664** | 0.6% better | 3.6% worse |
| baseline | 0.04692 | — | 4.3% worse |
| noise-corr | 0.04717 | 0.5% worse | 4.8% worse |
| TSFlow | 0.04500 | — | — |

**Finding**: All three variants converge to essentially the same CRPS (~0.047) at 600 epochs. The structured noise provides no statistically significant improvement. An interesting training dynamics effect: noise-corr converges ~10% faster in early epochs (100-200) but the advantage vanishes by epoch 300.

## Key Takeaways

### What didn't work

1. **Per-level wavelet decomposition** (V1): Separate flow maps per resolution level creates gradient conflicts and oscillating loss. The decomposed problem is harder to optimize, not easier.

2. **Multi-scale context conditioning** (V2): Wavelet-decomposing the context window and feeding multi-scale features to the model adds complexity without quality gain. The standard context encoder already captures multi-scale temporal structure.

3. **Structured noise initialization** (V3): Resolution-matched priors and correlated noise change early training dynamics but converge to the same final quality as N(0,I).

### What we learned

1. **MeanFlow-TS already achieves ~0.047 CRPS on electricity_nips with 1 step** — only 4.3% behind TSFlow's 0.045 with 32 steps. The baseline is already near-optimal for this architecture.

2. **The Conv1D + ResBlock architecture is the bottleneck**, not the noise distribution or multi-resolution structure. To meaningfully improve, the backbone needs upgrading (S4/Mamba, as in TSFlow).

3. **Multi-resolution decomposition in wavelet space is theoretically appealing but empirically unnecessary** for 1D time series forecasting at this scale (prediction length 24-48). The model learns multi-scale features internally. This may differ for longer prediction horizons or higher-dimensional data.

4. **Correlated noise accelerates early convergence** (~10% better CRPS at epoch 100-200) but this advantage disappears with sufficient training. Could be useful for scenarios with limited training budget.

## Reproducing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the V3 noise-only experiment (most comprehensive, ~3h on 1 GPU)
PYTHONUNBUFFERED=1 python experiments/iterate_v3_noise_only.py \
  --variants noise-wavelet noise-corr baseline \
  --epochs 600

# Run the V1 per-level heads experiment (~1.5h on 1 GPU)
PYTHONUNBUFFERED=1 python experiments/iterate_fixes.py \
  --variants flat-loss flat-loss+big-heads baseline \
  --epochs 100

# Run the V2 multi-scale context experiment (~3h on 1 GPU)
PYTHONUNBUFFERED=1 python experiments/iterate_v2.py \
  --variants v2-matched v2-iso baseline \
  --epochs 200
```

## Files in this branch

```
oneflow_ts/                          # New multi-resolution package
  wavelet.py                         # Haar DWT/IDWT
  priors.py                          # Resolution-matched priors
  model.py                           # V1: per-level flow map heads
  model_v2.py                        # V2: multi-scale context encoder
  loss.py                            # Per-level and flat losses
  __init__.py

experiments/
  validate_multires.py               # GO/NO-GO validation
  iterate_fixes.py                   # V1 iteration (flat-loss, big-heads)
  iterate_v2.py                      # V2 iteration (multi-scale context)
  iterate_v3_noise_only.py           # V3 iteration (noise-only, 600ep)
  validate_matched.log               # V1 validation log
  iterate_log.txt                    # V1 iteration log
  iterate_v2_log.txt                 # V2 iteration log
  iterate_v3_log.txt                 # V3 iteration log
```
