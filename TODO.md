# Dual-Axis Self-Consistency for Temporal Flow Matching

## Core Idea

MeanFlow enforces self-consistency along the **generation axis** (ODE time s ∈ [0,1]): the average velocity over an interval [r,s] is consistent with instantaneous velocities via the MeanFlow identity `u = v − (s−r)·du/ds`.

Time series have a **second axis** — the temporal axis of the data (data time t ∈ {1,...,T}). Current methods treat the forecast vector [x₁,...,xₜ] as a monolithic object and ignore that adjacent components x_t and x_{t+1} are temporally correlated.

This branch introduces a **dual-axis self-consistency** framework: enforce consistency along both the generation axis AND the temporal axis simultaneously. This is new mathematics, not an application of existing tools.

## What Needs to Be Done

### 1. Formalize the Temporal Identity
- The velocity field v_θ(z_s, s) produces a T-dimensional output. Decompose it into per-position components: v_θ = [v¹_θ, ..., vᵀ_θ].
- Define what "temporal consistency" means mathematically: the flow generating position t should be predictable from the flow generating position t−1, because adjacent values are correlated.
- Derive an identity relating the per-position velocity components, analogous to how MeanFlow relates average and instantaneous velocity.
- The key challenge: the generation axis is continuous and the temporal axis is discrete. The identity will likely involve a discrete analogue of the JVP — e.g., finite differences along the temporal axis.

### 2. Design a Training Loss
- The loss should have two terms: (a) standard MeanFlow JVP loss along the generation axis, (b) a new temporal consistency term along the data axis.
- The temporal term should penalize violations of the temporal identity — i.e., cases where the flow for position t is inconsistent with the flow for position t+1.
- Consider whether this can be computed cheaply (e.g., via finite differences on the network output) rather than requiring additional JVP calls.

### 3. Implement and Test
- Start with the existing ConditionalMeanFlowNetV2 architecture.
- Add the temporal consistency loss term.
- Test on electricity_nips first (our strongest dataset, best for fast iteration).
- Compare: (a) MeanFlow only, (b) temporal consistency only, (c) dual-axis (both).
- Measure CRPS, but also temporal coherence metrics: autocorrelation error, spectral divergence.

### 4. Theoretical Analysis
- Prove (or characterize) when the dual-axis identity holds exactly vs approximately.
- Show that the temporal consistency term provides a tighter bound on generation quality than MeanFlow alone.
- Connect to GP priors: show that TSFlow's GP prior approximately enforces temporal consistency in the initial conditions, but not throughout the generation process. Our method enforces it throughout.

## Key Questions to Resolve
- Is the temporal identity a hard constraint (always satisfied by the true velocity field) or a soft regularizer (approximately satisfied)?
- Does enforcing temporal consistency help more for datasets with strong temporal structure (electricity, traffic) vs weak structure (exchange_rate)?
- Can the temporal identity be derived from the stochastic interpolant framework, or does it require new theory?
