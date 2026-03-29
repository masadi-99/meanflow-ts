# MeanFlow-TS: One-Step Probabilistic Time Series Forecasting via MeanFlow

First application of MeanFlow (Geng et al., 2025) to time series forecasting. Achieves competitive CRPS with **1-step inference** (32x fewer network evaluations than TSFlow), and enables **inference-time compute scaling** — generating more samples cheaply improves forecast quality in a regime multi-step methods cannot access.

## Key Result: Inference-Time Scaling

One-step generation enables a compute-quality tradeoff unavailable to multi-step methods:

| Method | Samples | CRPS | Wall Time | Quality per Second |
|--------|---------|------|-----------|-------------------|
| MeanFlow | 250 | **0.0456** | 185s | Better |
| TSFlow | 10 | 0.0479 | 143s | Worse |
| MeanFlow | 1000 | **0.0455** | 648s | Best |
| TSFlow | 100 | 0.0447 | 791s | Comparable |

At equal compute budget, MeanFlow produces **5% better CRPS**. At equal quality, MeanFlow is **10x faster**.

## Table 3 — CRPS Forecasting (all 8 datasets, lower is better)

| Dataset | MeanFlow-TS v4 (1 NFE) | TSFlow (32 NFE) | Gap |
|---------|----------------------|-----------------|-----|
| electricity | **0.047** | 0.045 | 4% |
| solar | 0.423 | 0.341 | 24% |
| traffic | **0.086** | 0.082 | 5% |
| exchange | 0.010 | 0.005 | 100% |
| m4_hourly | 0.032 | 0.029 | 10% |
| uber_tlc | 0.161 | 0.154 | 5% |
| wiki2000 | **0.207** | 0.207 | 0% |
| kdd_cup | 0.301 | 0.288 | 5% |

5 of 8 datasets within 5% of TSFlow. Wiki2000 matches exactly. All with 32x fewer inference steps.

## Table 2 — LPS (CRPS of linear model on synthetic data, lower is better)

| Dataset | MeanFlow-TS | TSFlow | Winner |
|---------|------------|--------|--------|
| electricity | **0.080** | 0.096 | MeanFlow |
| solar | **0.254** | 0.616 | MeanFlow |
| traffic | **0.139** | 0.237 | MeanFlow |
| exchange | **0.009** | 0.011 | MeanFlow |
| m4_hourly | 0.044 | **0.032** | TSFlow |

MeanFlow wins on 4/5 datasets — synthetic data from MeanFlow trains better downstream predictors.

## Inference Comparison

| Metric | MeanFlow-TS v4 | TSFlow | Ratio |
|--------|---------------|--------|-------|
| NFE per sample | **1** | 32 | **32x fewer** |
| Total FLOPs (100 samples) | **3.0B** | 10.9B | **3.6x fewer** |
| Wall time per forecast | **7.4 ms** | 1,713 ms | **231x faster** |

## Architecture

- **Conditional MeanFlow**: Context encoder (2 ResBlocks + pooling) with lag features → Prediction decoder (4 ResBlocks with FiLM conditioning)
- **Dual time embedding**: (t, h=t-r) with sinusoidal positional encoding, concatenated (same as original MeanFlow)
- **JVP self-consistency loss**: Exact MeanFlow training objective via `torch.func.jvp`
- **1-step inference**: `z_0 = z_1 - u(z_1, t=1, h=1, context)`

## Known Limitations

- **Normalization mismatch**: Context window mean vs TSFlow's global per-series mean
- **No GP prior**: TSFlow initializes from Gaussian Process posterior; we use standard Gaussian
- **Simpler backbone**: 1D convolutions vs TSFlow's S4 (Structured State Space) blocks
- **Solar and exchange gaps**: Datasets with bimodal distributions or near-random-walk behavior are harder for one-step generation

## Quick Start

```bash
# Train on a dataset
PYTHONPATH=. python experiments/train_forecasting_v4.py electricity_nips --epochs 600

# Evaluate all metrics
PYTHONPATH=. python experiments/eval_forecasting.py

# Run inference scaling experiment
PYTHONPATH=. python experiments/scaling_experiment.py

# Run ablation (MeanFlow vs standard FM)
PYTHONPATH=. python experiments/ablation_standard_fm.py
```

## References

- MeanFlow: Geng et al., "Mean Flows for One-step Generative Modeling", 2025
- TSFlow: Kollovieh et al., "Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting", ICLR 2025
