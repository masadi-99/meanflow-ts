# MeanFlow-TS: One-Step Time Series Forecasting via MeanFlow

First application of MeanFlow (Geng et al., 2025) to time series forecasting. Achieves probabilistic forecasting with **1-step inference** (32x fewer network evaluations than TSFlow).

## Results (Table 3 — CRPS, lower is better)

v4 model with 7 daily lag features, 1.14M params, 1-step inference:

| Dataset | MeanFlow-TS v4 (1 NFE) | TSFlow (32 NFE) | Gap |
|---------|----------------------|-----------------|-----|
| electricity | 0.047 | **0.045** | 4% |
| solar | 0.423 | **0.341** | 24% |
| traffic | 0.086 | **0.082** | 5% |
| exchange | 0.010 | **0.005** | 100% |
| m4_hourly | 0.032 | **0.029** | 10% |
| uber_tlc | 0.159 | **0.154** | 3% |
| wiki2000 | 0.208 | **0.207** | 0% |
| kdd_cup | 0.293 | **0.288** | 2% |

TSFlow is better on all datasets. MeanFlow-TS trades quality for 32x fewer inference steps.

## Known Limitations

- **Model size**: MeanFlow-TS has 1.14M params vs TSFlow's ~189K (6x larger)
- **Normalization mismatch**: We normalize by context window mean; TSFlow uses global per-series means cached during training
- **No GP prior**: TSFlow initializes from a Gaussian Process posterior; we use standard Gaussian noise
- **Simpler backbone**: 1D convolutions vs TSFlow's S4 (Structured State Space) blocks

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
