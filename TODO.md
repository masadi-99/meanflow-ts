# Multi-Resolution Conditional Refinement + Future-Statistic Conditioning

This branch implements two extensions to MeanFlow-TS that exploit MeanFlow's one-step generation for capabilities multi-step methods cannot replicate cheaply.

---

## Idea 1: Multi-Resolution Conditional Refinement

### What
Given past context and a **coarse version of the future**, generate a full-resolution future consistent with that coarse trajectory. Each refinement stage is a single MeanFlow forward pass.

### Why
Turns MeanFlow into a resolution-aware generator. Multi-stage refinement (coarse → fine) stays cheap because each stage is 1 NFE. With diffusion or multi-step ODE, repeated refinement would be expensive. Also enables longer-horizon generation: predict coarse over a long window, then refine.

### Implementation Steps

**Step 1: Downsampling/upsampling operators**
- Add `downsample(y, r)` = average pooling with factor r ∈ {2, 4, 8}
- Add `upsample(c, r)` = linear interpolation back to original length
- These are deterministic, no learnable params

**Step 2: Modify ConditionalMeanFlowNetV2**
- Add a coarse-condition encoder: small Conv1d + ResBlock that processes the upsampled coarse trajectory `U_r(c)`
- Fuse with existing context encoding via concatenation or addition
- The model predicts the **residual** `r_f = y_f - U_r(c)` instead of the full future
- When no coarse input is given (dropout or unconditional mode), predict the full future as before

**Step 3: Training**
- For each training batch:
  - Sample a random downsampling factor r from {2, 4, 8}
  - Compute c = downsample(future, r)
  - With probability 0.1, drop the coarse condition (unconditional fallback)
  - Compute residual target: r_f = future - upsample(c, r)
  - Train MeanFlow JVP loss on the residual, conditioned on (past, c)

**Step 4: Evaluation**
- Refinement mode: give model oracle coarse future, measure if it recovers fine detail
- Long-horizon mode: predict coarse 2x-length future, then refine each half
- Compare CRPS at full resolution vs baseline (no coarse input)
- Measure coarse consistency: downsample(generated) vs given coarse
- Measure spectral recovery: compare power spectra

**Step 5: Run on all datasets**
- electricity, traffic, m4, solar, exchange, uber, wiki, kdd

### Success Criteria
- Refinement improves CRPS over baseline forecaster
- Coarse consistency is high (generated output matches given coarse input when downsampled)
- Model still works as baseline forecaster when coarse is dropped

---

## Idea 2: Future-Statistic Conditioning

### What
Condition generation on a vector of **future path statistics** (mean, max, min, volatility, time-to-peak, etc.) so the model generates futures satisfying high-level requirements.

### Why
Gives an interpretable control interface unique to generative forecasters. The user specifies "I want a future with peak value X at time Y" and the model produces diverse trajectories satisfying that. MeanFlow's 1-step generation makes sampling many controlled futures cheap.

### Implementation Steps

**Step 1: Define statistic extraction**
- `extract_stats(y_f)` → vector of [mean, max, min, std, argmax, area_under_curve]
- All computed from the future trajectory, all differentiable (or approximately so)
- Start with mean, max, min, std (4 values). Add argmax and AUC later.

**Step 2: Modify ConditionalMeanFlowNetV2**
- Add a stat encoder: MLP that maps stat vector → embedding
- Fuse with existing conditioning (add to the time+context embedding)
- Classifier-free dropout: with probability 0.15, zero out the stat embedding during training

**Step 3: Training**
- For each training batch:
  - Compute stats from the actual future: s = extract_stats(future)
  - With probability 0.15, set s = 0 (dropout for unconditional mode)
  - Train MeanFlow JVP loss conditioned on (past, s)
- Optional auxiliary loss: after generating a sample, check if extract_stats(sample) ≈ s

**Step 4: Evaluation**
- **Control accuracy**: give oracle stats from held-out future, generate samples, measure how close generated stats are to requested stats
- **Controllable generation**: vary one stat (e.g., sweep peak value) while keeping others fixed, verify the model responds
- **Forecast quality**: when using oracle stats, does CRPS improve over unconditional baseline?
- **Diversity**: multiple samples with same stats should vary in trajectory shape

**Step 5: Synthetic diagnostics**
- Bimodal futures: test if stat-conditioning can select the desired mode
- Rare events: condition on high peak, verify model generates spikes
- Volatility control: condition on high vs low std, verify

### Success Criteria
- Model hits requested statistics better than unconditional baseline
- CRPS with oracle stats is better than unconditional CRPS
- Samples are diverse (not collapsed) under fixed stats
- Model still works unconditionally when stats are dropped

---

## Implementation Order

1. Start with Idea 1 (multi-resolution) — it's simpler architecturally (just add a conditioning path)
2. Test on electricity first (fastest iteration)
3. If it works, add Idea 2 (stat conditioning) — same architectural pattern (add another conditioning path)
4. Both ideas share the same training framework — can be combined in a single model

## Files to Create/Modify

| File | Change |
|------|--------|
| `meanflow_ts/model_v3.py` | New model with coarse + stat conditioning |
| `meanflow_ts/utils.py` | Downsampling/upsampling operators, stat extraction |
| `experiments/train_multiresolution.py` | Training script for multi-res |
| `experiments/train_stat_conditioning.py` | Training script for stat conditioning |
| `experiments/eval_multiresolution.py` | Eval: refinement quality, coarse consistency |
| `experiments/eval_stat_control.py` | Eval: control accuracy, diversity |
