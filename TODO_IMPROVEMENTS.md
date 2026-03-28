# MeanFlow-TS: Improvement Hypotheses

Ranked by novelty × expected impact × feasibility. Based on the deep analysis of why multi-resolution didn't work and what the actual quality gap (4.3% CRPS vs TSFlow) consists of.

## Priority 1: Residual Flow (Highest Potential)

**Status**: TODO

**Idea**: Instead of generating the full forecast from noise, predict a deterministic base forecast `f(context)` and only model the RESIDUAL uncertainty with the flow map:
```
forecast = f(context) + (z_1 - u(z_1, t, h, context))
```
The flow map learns the error/uncertainty distribution, not the full signal.

**Why it should work**: Most of CRPS comes from the mean prediction accuracy. If f(context) captures the mean well (which a simple MLP can do), the flow only needs to model the remaining uncertainty — a much simpler distribution closer to zero-mean Gaussian.

**Expected impact**: 2-5% CRPS improvement
**Novelty**: HIGH — no one combines deterministic base forecast + one-step residual flow for TS
**Similar to**: CGFM (Jul 2025) uses auxiliary model predictions to guide flow matching, but with multi-step and external model. Ours is integrated and one-step.

---

## Priority 2: iMF Velocity Loss Reformulation (Lowest Risk)

**Status**: TODO

**Idea**: Replace the original MeanFlow JVP loss with iMF's reformulated velocity loss. From the iMF paper (Dec 2025):
- Original MeanFlow: loss target depends on the network itself (problematic regression)
- iMF: reformulates as loss on instantaneous velocity v, using the network's average velocity u in the JVP tangent
- Result on ImageNet: FID 3.43 → 1.72 (50% improvement)

**Implementation**:
```python
# Original MeanFlow:
u_pred, dudt = jvp(u_func, (z, t, r), (v_true, 1, 0))
u_target = (v_true - (t-r) * dudt).detach()
loss = ||u_pred - u_target||^2

# iMF reformulation:
u_pred, dudt = jvp(u_func, (z, t, r), (v_pred, 1, 0))  # Use v_pred instead of v_true
v_pred_full = u_pred + (t-r) * dudt  # Reconstructed instantaneous velocity
loss = ||v_pred_full - v_true||^2  # Loss on v, not u
```

Key change: JVP tangent uses the network's OWN predicted velocity `v_pred` instead of ground truth `v_true`. This eliminates the network-dependent target problem.

**Expected impact**: 1-5% CRPS improvement
**Novelty**: MEDIUM — iMF exists but hasn't been applied to time series
**Risk**: VERY LOW — just a loss function change, same architecture

---

## Priority 3: Adaptive Multi-Step Inference (Most Novel)

**Status**: TODO

**Idea**: MeanFlow learns `u(z, t, h)` for ALL h values during training. At inference, instead of always using h=1 (1-step), adaptively choose the number of steps per sample:

```python
# 1-step attempt
u_1step = model(z_1, t=1, h=1, context)
quality_score = some_criterion(u_1step)

if quality_score > threshold:
    # 1-step is sufficient
    forecast = z_1 - u_1step
else:
    # Use 2-step via compositional property
    u_half1 = model(z_1, t=1, h=0.5, context)
    z_0.5 = z_1 - 0.5 * u_half1
    u_half2 = model(z_0.5, t=0.5, h=0.5, context)
    forecast = z_0.5 - 0.5 * u_half2
```

**Quality criteria options**:
- `||u_1step||` — velocity magnitude (large = hard sample)
- Entropy of generated distribution (high = uncertain)
- Learned quality predictor (small auxiliary network)

**Why it's novel**: Nobody has explored adaptive step selection in MeanFlow. The h parameter naturally supports this but everyone uses fixed h=1.

**Expected impact**: 1-3% (recovers multi-step quality for hard cases)
**Novelty**: VERY HIGH — completely unexplored in the literature
**Average NFE**: ~1.1-1.3 (most samples use 1 step)

---

## Priority 4: Learned Data-Dependent Prior

**Status**: TODO

**Idea**: Instead of fixed `z_1 ~ N(0, I)`, learn a context-dependent prior:
```python
mu_prior, log_sigma_prior = prior_net(context)  # Small MLP
z_1 = mu_prior + sigma_prior * eps,  eps ~ N(0, I)
```

The flow map then transports from a distribution that's ALREADY close to the target (because the prior network predicts the approximate forecast shape). The remaining transport is just refinement.

**Why it should work**: TSFlow's GP prior works precisely because it reduces transport distance. A LEARNED prior can be even better — it adapts to each specific context, not just a generic temporal correlation structure.

**Difference from TSFlow**: TSFlow uses a FIXED GP kernel (Ornstein-Uhlenbeck). Ours is LEARNED and context-dependent. TSFlow still needs 32 ODE steps. Ours is 1 step.

**Expected impact**: 1-3%
**Novelty**: HIGH — learned priors for one-step flow matching don't exist
**Risk**: Need to ensure the prior doesn't collapse (mu_prior = exact forecast, sigma_prior = 0). Regularize by adding KL(prior || N(0,I)) to loss.

---

## Priority 5: Backbone Upgrade (S4/Mamba)

**Status**: TODO

**Idea**: Replace Conv1D ResBlocks with S4 or Mamba blocks for infinite receptive field and better long-range temporal modeling.

**Why it should work**: TSFlow uses S4 and it's one of the main architectural advantages. For longer prediction horizons (48+), Conv1D's RF=17 becomes a serious limitation.

**Expected impact**: 2-3%
**Novelty**: LOW — TSFlow already uses S4
**Feasibility**: HIGH — S4/Mamba implementations exist (pip install mamba-ssm)
**Note**: This is engineering, not a research contribution. Do last, only to close the remaining gap for the paper's results table.

---

## Experiment Plan

1. **Quick test (1h each)**: Test hypotheses 5 (residual flow) and 3 (iMF loss) individually on electricity_nips, 200 epochs
2. **Combine winners**: Stack the improvements that work
3. **Full training (600ep)**: Run the combined model for full convergence
4. **Multi-dataset**: Evaluate on all 5 GluonTS datasets
5. **Backbone upgrade**: Add S4/Mamba if needed to close remaining gap
