# What Does the Chern Number Actually Measure?

## Unified Interpretation Report

**Status**: Template - Run experiments to populate results

---

## Executive Summary

This report synthesizes findings from four parallel investigation tracks to answer the fundamental question: **What does the Chern number actually measure in financial markets?**

### Key Hypothesis

The Chern number measures the **topological twisting of the market's correlation structure** - specifically, how the geometry of asset relationships changes as you move through different market states. It is NOT primarily measuring volatility.

### Findings Overview

| Track | Focus | Key Question | Status |
|-------|-------|--------------|--------|
| A | Interpretation | Does Chern correlate with correlation structure? | Pending |
| B | TDA Baseline | Does persistent homology work better? | Pending |
| C | Ensemble | Does Chern add unique information? | Pending |
| D | Improvements | Can we reduce false positives? | Pending |

---

## Track A: Chern Interpretation Test

**File**: `experiments/chern_interpretation_test.py`

### Methodology

1. **Correlation Eigenstructure Comparison**
   - Computed rolling correlation matrix eigenvalues across 10 assets
   - Tracked eigenvalue ratio (largest/smallest) as "correlation stability"
   - Measured correlation between Chern spikes and eigenvalue shifts

2. **Information Geometry Connection**
   - Computed Fisher Information Matrix from return distributions
   - Compared FIM behavior to Berry curvature magnitude
   - Validates "twisted probability manifold" interpretation

3. **Lead/Lag Analysis**
   - Cross-correlation between |ΔChern| and correlation breakdown indicator
   - Determined if Chern leads, lags, or coincides with structural shifts

### Results

*Run `python experiments/chern_interpretation_test.py` to populate*

| Metric | Value | Interpretation |
|--------|-------|----------------|
| |ΔChern| ↔ |Δeigenvalue_ratio| correlation | TBD | > 0.3 supports hypothesis |
| Spike-shift alignment rate | TBD | > 50% supports hypothesis |
| Fisher ↔ Berry correlation | TBD | Shows geometric connection |
| Optimal lag (days) | TBD | Positive = Chern leads |
| FP explanation rate | TBD | > 50% validates "false positives" |

### Interpretation

- [ ] **Chern measures correlation structure changes**: TBD
- [ ] **Chern leads correlation breakdown**: TBD
- [ ] **"False positives" are actually correlation events**: TBD

---

## Track B: Persistent Homology Baseline

**File**: `experiments/persistent_homology_baseline.py`

### Methodology

1. **Takens Embedding**
   - Delay: 1 day
   - Embedding dimension: 3
   - Creates point cloud from time series

2. **Persistent Homology**
   - Using ripser (if available) or simplified H0 computation
   - Extracts topological features (persistence entropy, norms)

3. **Comparison**
   - Same crisis detection task as Chern
   - Precision/recall/F1 comparison

### Results

*Run `python experiments/persistent_homology_baseline.py` to populate*

| Method | Spikes | Precision | Recall | F1 |
|--------|--------|-----------|--------|-----|
| TDA (Persistent Homology) | TBD | TBD | TBD | TBD |
| Chern Number | TBD | TBD | TBD | TBD |

| Metric | Value |
|--------|-------|
| Agreement rate | TBD |
| Unique TDA detections | TBD |
| Unique Chern detections | TBD |

### Interpretation

- [ ] **TDA performs better/worse/similar to Chern**: TBD
- [ ] **Methods capture different aspects of topology**: TBD
- [ ] **Consider ensemble of TDA + Chern**: TBD

---

## Track C: Chern Ensemble Signal

**File**: `experiments/chern_ensemble_signal.py`

### Methodology

1. **Signal Components**
   - Chern number (topological)
   - Realized volatility (20-day)
   - Correlation stability (eigenvalue ratio)
   - VIX (if available)

2. **Ensemble Rule**
   - Signal fires when ≥2 components agree

3. **Ablation Test**
   - Ensemble without Chern to measure Chern's contribution

### Results

*Run `python experiments/chern_ensemble_signal.py` to populate*

| Signal | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Chern alone | TBD | TBD | TBD |
| Volatility alone | TBD | TBD | TBD |
| Correlation alone | TBD | TBD | TBD |
| **Ensemble** | TBD | TBD | TBD |
| Ablated (no Chern) | TBD | TBD | TBD |

| Metric | Value |
|--------|-------|
| Chern lift (F1 improvement) | TBD |
| Unique Chern detections | TBD |
| False positive rate | TBD |

### Interpretation

- [ ] **Ensemble outperforms individual signals**: TBD
- [ ] **Chern provides unique information**: TBD
- [ ] **Signal is tradeable (< 50% FP rate)**: TBD

---

## Track D: Improved Chern Computation

**File**: `qcml/improved_chern.py`

### Methodology

1. **Threshold Methods**
   - Fixed 3σ (vs. original 2σ)
   - Adaptive (regime-dependent)
   - Quantile (95th percentile)

2. **Multi-Scale Analysis**
   - Windows: 10, 20, 50 days
   - Require agreement across scales

3. **Confirmation Logic**
   - Require 2+ consecutive days

### Results

*Run `python -m qcml.improved_chern` to populate*

| Method | Spikes | Precision | Recall | F1 | Δ vs Original |
|--------|--------|-----------|--------|-----|---------------|
| Original (2σ) | TBD | TBD | TBD | TBD | baseline |
| Fixed 3σ | TBD | TBD | TBD | TBD | TBD |
| Adaptive | TBD | TBD | TBD | TBD | TBD |
| Quantile | TBD | TBD | TBD | TBD | TBD |
| Full improved | TBD | TBD | TBD | TBD | TBD |

### Interpretation

- [ ] **At least one method improves precision without killing recall**: TBD
- [ ] **Best method**: TBD
- [ ] **Maintains quantum/topological interpretation**: Yes (by design)

---

## Unified Conclusions

### What Chern Actually Measures

Based on Track A results:

1. **Correlation to eigenvalue shifts**: TBD
2. **Lead time**: TBD days
3. **FP explanation**: TBD of "false positives" are correlation events

**Interpretation**: TBD

### Best Detection Approach

Based on Tracks B, C, D:

1. **Individual method**: TBD
2. **Ensemble approach**: TBD
3. **Key improvement**: TBD

### The Narrative for Papers/Interviews

If validation succeeds, the recommended narrative is:

> "The Chern number computes a topological invariant of the market's probability manifold. When correlation structures undergo fundamental restructuring, this manifests as a discontinuity in the Chern number - analogous to a phase transition in condensed matter physics. Unlike volatility measures which capture magnitude, the Chern number captures the *geometry* of market relationships. Our experiments show it detects structural deterioration [X] days before crisis manifestation, with [Y]% recall on major market events."

This preserves:
- ✓ Quantum geometry framing (Berry curvature, Chern number)
- ✓ Topological interpretation (phase transitions, invariants)
- ✓ Novel contribution (different from volatility)
- ✓ Practical relevance (lead time, regime detection)

---

## Recommendations

### For Research Paper

1. **Frame**: Chern as "correlation structure health index"
2. **Validation**: Show correlation with eigenvalue shifts (Track A)
3. **Comparison**: Include TDA baseline (Track B)
4. **Practical**: Present ensemble approach (Track C)

### For Trading Signal

1. **Use improved computation** (Track D best method)
2. **Combine with** volatility and correlation signals (Track C)
3. **Target**: Regime identification, not point-in-time crisis prediction

### For Interviews

1. **Lead with interpretation**: "Measures geometric twisting of probability manifold"
2. **Show validation**: Correlation with correlation structure changes
3. **Acknowledge limitations**: High false positive rate standalone, better in ensemble
4. **Highlight novelty**: Orthogonal to volatility measures

---

## How to Run Experiments

```bash
# Track A: Interpretation test (run first)
python experiments/chern_interpretation_test.py

# Track B: TDA baseline
python experiments/persistent_homology_baseline.py

# Track C: Ensemble signal
python experiments/chern_ensemble_signal.py

# Track D: Improved computation
python -m qcml.improved_chern
```

Results will be saved to `experiments/outputs/track_*_results.json`

---

## Version History

- **v1.0** (2024-02-04): Initial template created
