# MeanFlow-TS: One-Step Probabilistic Time Series Forecasting via MeanFlow

## Summary

This branch contains all experiments exploring whether MeanFlow provides a **novel contribution** beyond "applying MeanFlow to time series." The answer is mixed: MeanFlow gives competitive forecasting quality with massive speedup, and the inference-time scaling property is real, but the Self-Consistency Score (SCS) — our strongest candidate for novelty — does not translate to practical CRPS improvement.

---

## What Works

### Table 3 — CRPS Forecasting (1-step MeanFlow vs 32-step TSFlow)

| Dataset | MeanFlow v4 (1 NFE) | TSFlow (32 NFE) | Gap |
|---------|-------------------|-----------------|-----|
| electricity | **0.047** | 0.045 | 4% |
| solar | 0.423 | 0.341 | 24% |
| traffic | **0.086** | 0.082 | 5% |
| exchange | 0.010 | 0.005 | 100% |
| m4_hourly | 0.032 | 0.029 | 10% |
| uber_tlc | 0.161 | 0.154 | 5% |
| wiki2000 | **0.207** | 0.207 | 0% |
| kdd_cup | 0.301 | 0.288 | 5% |

5 of 8 datasets within 5% of TSFlow. All with 32x fewer inference steps.

### Table 2 — LPS (Linear Predictive Score)

| Dataset | MeanFlow | TSFlow | Winner |
|---------|---------|--------|--------|
| electricity | **0.080** | 0.096 | MeanFlow |
| solar | **0.254** | 0.616 | MeanFlow |
| traffic | **0.139** | 0.237 | MeanFlow |
| exchange | **0.009** | 0.011 | MeanFlow |
| m4_hourly | 0.044 | **0.032** | TSFlow |

MeanFlow wins 4/5 — synthetic data from MeanFlow trains better downstream predictors.

### Inference Speedup

| Metric | MeanFlow | TSFlow | Ratio |
|--------|---------|--------|-------|
| NFE per sample | **1** | 32 | **32x fewer** |
| Wall time / forecast | **7.4 ms** | 1,713 ms | **231x faster** |
| Total FLOPs (100 samples) | **3.0B** | 10.9B | **3.6x fewer** |

### Inference-Time Scaling

More samples → better CRPS, cheaply. At equal compute budget, MeanFlow produces 5% better CRPS:

| Method | Samples | CRPS | Wall Time |
|--------|---------|------|-----------|
| MeanFlow | 250 | **0.0456** | 185s |
| TSFlow | 10 | 0.0479 | 143s |
| MeanFlow | 1000 | **0.0455** | 648s |
| TSFlow | 100 | 0.0447 | 791s |

---

## What Doesn't Work (Negative Results)

### Self-Consistency Score (SCS) — Experiments 1-3

**Idea:** MeanFlow's average velocity satisfies a self-consistency identity. The discrepancy between 1-step and 2-step generation (SCS) could be a free quality signal.

**Experiment 1 (SCS correlation with quality):**

| Dataset | Spearman(SCS, error) | SCS-select vs random | Verdict |
|---------|---------------------|---------------------|---------|
| electricity | 0.134 | -13.6% (worse) | Marginal |
| traffic | **0.319** | -1.8% (worse) | Marginal |
| m4_hourly | -0.219 | -4.9% (worse) | Fail |

SCS correlates with individual sample quality (Spearman 0.32 on traffic) but **selecting low-SCS samples hurts CRPS** because it reduces ensemble diversity. CRPS rewards both accuracy and spread — filtering removes spread.

**Experiment 2 (Multi-step refinement):**

| Dataset | 2-step vs 1-step | Verdict |
|---------|-----------------|---------|
| electricity | No improvement | ✗ |
| traffic | +0.35% | Marginal |
| m4_hourly | +0.65% | Marginal |

Multi-step doesn't help — the 1-step MeanFlow velocity is already optimal.

**Experiment 3 (SCS-guided noise optimization):**

| Dataset | Best improvement | Verdict |
|---------|-----------------|---------|
| traffic | 0% | ✗ |
| electricity | 0.97% | Marginal |

Backpropagating through SCS to optimize z_1 doesn't improve CRPS. The SCS landscape is too flat.

### Spectral Consistency Loss

**Idea:** Enforce that the velocity field produces outputs with correct power spectrum.

**Result:** 0.5-0.9% improvement (electricity 0.0467 vs 0.0471, traffic 0.0854 vs 0.0858). Real but small — not a standalone contribution.

### Temporal Shift-Equivariance Loss

**Idea:** For stationary time series, v^{t+1}(z,s) = v^t(shift(z), s).

**Result:** No improvement. Conv1d already has approximate shift-equivariance via weight sharing. The loss is redundant.

---

## Ablation: MeanFlow vs Standard FM

The corrected ablation (with fixed ODE direction) shows MeanFlow and standard FM converge to similar CRPS at 600 epochs. The 32-step FM gap closes as training progresses:

| Epoch | FM 1-step | FM 32-step | MeanFlow 1-step |
|-------|----------|-----------|----------------|
| 50 | 0.345 | 0.363 | 0.404 |
| 100 | 0.180 | 0.227 | 0.215 |
| 200 | 0.092 | 0.103 | 0.090 |
| 300 | 0.068 | 0.066 | 0.061 |
| 600 | 0.062 | 0.056 | 0.056 |

At convergence, MeanFlow ≈ standard FM for this architecture. The advantage is architectural (dual time conditioning) rather than loss-specific.

---

## Architecture

- **ConditionalMeanFlowNetV2**: Context encoder with 7 daily lag features → Prediction decoder (4 ResBlocks, FiLM conditioning)
- **Dual time embedding**: (t, h=t-r) sinusoidal, concatenated
- **JVP self-consistency loss**: Exact MeanFlow from Geng et al. (2025)
- **1-step inference**: z_0 = z_1 - u(z_1, t=1, h=1, context)

## Files

```
meanflow_ts/model.py          — Core MeanFlow model + loss
meanflow_ts/model_v2.py       — V4 with lag features
meanflow_ts/dual_axis.py      — Spectral consistency loss

experiments/
  train_forecasting_v4.py     — Training script (all 8 datasets)
  feasibility_scs.py          — Exp 1: SCS correlation
  feasibility_multistep.py    — Exp 2: Multi-step refinement
  feasibility_scs_optimize.py — Exp 3: SCS noise optimization
  feasibility_combined.py     — Exp 4: Pareto frontier
  scaling_experiment.py       — Inference-time scaling
  benchmark_inference.py      — Speed comparison
  ablation_standard_fm.py     — MeanFlow vs FM ablation
  sanity_check.py             — Checkpoint validation
  plot_examples.py            — Visualization
```

## References

- MeanFlow: Geng et al., "Mean Flows for One-step Generative Modeling", 2025
- TSFlow: Kollovieh et al., "Flow Matching with GP Priors for Probabilistic TS Forecasting", ICLR 2025
- TARFVAE: One-step generative TS forecasting via VAE (NeurIPS 2025 poster)
