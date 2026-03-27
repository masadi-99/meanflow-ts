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

- **Conditional MeanFlow**: Context encoder (2 ResBlocks + pooling) → Prediction decoder (4 ResBlocks with FiLM conditioning)
- **Dual time embedding**: (t, h=t-r) with sinusoidal positional encoding, concatenated (same as original MeanFlow)
- **JVP self-consistency loss**: Exact MeanFlow training objective via `torch.func.jvp`
- **1-step inference**: `z_0 = z_1 - u(z_1, t=1, h=1, context)`

## References

- MeanFlow: Geng et al., "Mean Flows for One-step Generative Modeling", 2025
- TSFlow: Kollovieh et al., "Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting", ICLR 2025
