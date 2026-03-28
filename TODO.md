# Meta Flow Matching on the Temporal Wasserstein Manifold

## Core Idea

Meta Flow Matching (Tong et al., ICLR 2025) amortizes flow models over *populations* on the Wasserstein manifold — given a new distribution (represented by samples), it generates from that distribution without retraining. It uses a GNN to embed populations and learns geodesics on the Wasserstein manifold.

Standard Wasserstein distance treats samples as unordered sets. But time series values have a **temporal ordering** — permuting [x₁, x₂, x₃] to [x₃, x₁, x₂] changes the time series entirely but doesn't change the Wasserstein distance between two sets of such vectors.

This branch introduces a **Temporal Wasserstein distance** that respects the ordering of values within a time series. Geodesics on this temporal Wasserstein manifold naturally produce flows that preserve temporal structure. Combined with Meta Flow Matching, this enables **few-shot adaptation** to new time series domains — embed a few examples, immediately generate forecasts.

## What Needs to Be Done

### 1. Define Temporal Wasserstein Distance
- Standard 2-Wasserstein: W₂(μ, ν) = inf E[||X - Y||²] over couplings.
- Temporal Wasserstein: replace Euclidean cost ||x - y||² with a **temporally-weighted cost** that penalizes mismatches at adjacent time positions more than distant ones. For example, a cost based on DTW (dynamic time warping), or a Toeplitz-structured cost matrix that encodes temporal locality.
- The key mathematical question: for which temporal cost functions does the resulting Wasserstein space have useful geometric properties (geodesic completeness, curvature bounds)?

### 2. Derive Geodesics on Temporal Wasserstein Manifold
- Standard Wasserstein geodesics (McCann interpolation) produce linear displacement interpolants. With a temporal cost, the geodesics will be different — they should produce interpolations that respect temporal ordering.
- Compute or approximate these geodesics. They define the "natural" flow paths for time series generation.
- Show that flowing along temporal Wasserstein geodesics produces smoother, more temporally coherent samples than standard Wasserstein geodesics.

### 3. Build Meta Flow Matching for Time Series
- Follow Meta FM architecture: embed a *context distribution* (a few example time series from a domain) using a permutation-invariant encoder (set transformer or GNN).
- Condition the flow model on this distributional embedding.
- Train on multiple time series datasets/domains simultaneously.
- At test time: given a few examples from a new domain + observed context, generate forecasts in one step.

### 4. Experiments
- Multi-domain training: train on all 8 TSFlow benchmark datasets jointly.
- Few-shot evaluation: hold out one dataset, embed 5-10 examples from it, generate forecasts. Compare to training from scratch.
- Zero-shot evaluation: test on a completely new dataset (not in training) with only a few examples.
- Compare temporal Wasserstein vs standard Wasserstein embeddings — show that temporal structure matters.

## Key Questions to Resolve
- Is the temporal Wasserstein distance a proper metric? Under what conditions?
- Can temporal Wasserstein geodesics be computed in closed form, or do we need numerical approximation?
- Does the Meta FM framework (GNN embedding + Wasserstein geodesic regression) transfer to time series, or does the temporal structure require different embedding architecture (e.g., temporal attention instead of GNN)?
- How many examples are needed for useful few-shot adaptation? Does temporal Wasserstein help with smaller support sizes?
