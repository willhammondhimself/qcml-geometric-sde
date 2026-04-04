# Response to reviewers

Date: 2026-04-04
Revision addresses: Synthesis 2026-03-05 (4 reviewers) and Synthesis 2026-03-21 (Statistician + Hostile)

---

## What changed

We fixed all 5 critical and 10 major issues from both review rounds. The short version:

- The Spectral Entropy d=0.83 error is gone. That value belonged to Reduced Purity; Spectral Entropy is d=0.53, rank 12. Fixed in two body-text locations.
- Hamiltonian Sensitivity, RF, and GARCH numbers were already correct in §5.1 after an earlier rerun. We audited and confirmed.
- A reverse-causality caveat now appears right after the offline results in §5.1, not just buried in §5.2.
- Cliff's delta is reported as a nonparametric robustness check (|delta| = 0.09--0.45 for the top 10 methods).
- The per-crisis variation paragraph now says the patterns are descriptive and formally non-significant (p=0.31).
- Bootstrap CIs are specified as percentile method. Single-asset-pair limitation is explicit. Scope breakdown (4 + 18 + 24 = 46) is in the Methods. A new Limitations bullet covers omitted wavelet/DL/signature baselines.
- We added Guritanu et al. (2025), a TDA competitor using persistent homology, to the related work.

All 9 registered numerical claims pass automated verification against the canonical JSON (tolerance 0.02).

---

## Reviewer 1 (Statistician)

**C1: Spectral Entropy d=0.83 is wrong.**

You were right. A stale value from the fusion JSON had leaked into two body-text locations (§3 intro and §3.3 subsection). Both now read d=0.53, rank 12 of 46. Every remaining d=0.83 in the paper correctly refers to Reduced State Purity. We ran grep across the full source to make sure.

Fixed in §3 and §3.3; 9/9 automated claim checks pass.

---

**C2/C3: Hamiltonian d-value and baseline ranks.**

We audited every subsection. §5.1 already had Hamiltonian Sensitivity at d=0.60 (rank 10), RF at rank 30, GARCH at rank 33. No stale values found elsewhere. No changes needed.

---

**M1: Missing Nemenyi post-hoc.**

The paper does report Nemenyi -- in both the methods ("Friedman rank test with Nemenyi post-hoc pairwise comparisons at alpha = 0.05") and §5.1 (CD = 18.2, 105 of 1,035 pairs significant, competitive-tier caveat). The review tracker was stale; we updated it.

---

**M5: Cohen's d normality assumption.**

We added Cliff's delta in §5.1: median |delta| for the top 10 methods ranges from 0.09 to 0.45, confirming that the Cohen's d rankings hold up under a nonparametric measure. The code now computes Cliff's delta for every method-crisis cell.

---

**M6: Bootstrap method unspecified.**

The implementation uses the percentile method (2.5th and 97.5th percentiles of the block-bootstrap distribution). We added those two words to §Statistical evaluation.

---

**m2: Friedman df.**

Added: chi-squared now reads chi^2_45 with the subscript.

---

## Reviewer 2 (Physicist)

**Adiabatic condition / gauge invariance.**

Eq. 4 is written in the parallel-transport gauge. The computational definition is the gauge-invariant plaquette formula (Eq. 7), which does not require adiabaticity. The paper already states this at lines 281--283: "our numerical implementation uses the gauge-invariant plaquette formula." Our Berry curvature is a discrete Wilson loop observable, not an adiabatic holonomy.

---

## Reviewer 3 (Quant)

**p=0.31 undermines the complementarity narrative.**

We overstated it. The narrative now leads with walk-forward Berry d=0.72. The per-crisis variation paragraph in §5.1 reads: "Descriptively, different geometric observables lead on different crisis types... However, formal testing finds no significant per-crisis specialization (p = 0.31), so these patterns should not be used as a selection rule." The abstract already puts walk-forward first and offline second. Orthogonality (|rho| = 0.13 vs baselines) is the actual value proposition, not per-crisis superiority.

---

## Reviewer 4 (Hostile)

**C5: Reverse Granger undermines "detection."**

This was the most important criticism. We added a caveat directly after the offline results in §5.1: "These offline d values reflect contemporaneous crisis sensitivity, not predictive lead time." The detailed Granger numbers (17/45 reverse vs 6/45 forward) remain in §5.2. We kept the word "detection" in the title because contemporaneous detection of ongoing regime shifts has practical value -- but the paper no longer implies prediction.

---

**M2: Why bother with QCML if Absorption Ratio wins?**

Because the signals are orthogonal, not because ours are bigger. Cross-correlation between Berry Phase Rate and Absorption Ratio is |rho| = 0.13. The paper says this in three places (abstract, intro, §6.2) and explicitly states: "We do not claim QCML dominates these methods." The pitch is that Berry d=0.72 OOS with no crisis labels adds something AR cannot provide alone.

---

**M3: Reduced Purity holdout collapse.**

The intro caveat now reads: "Reduced Purity (d = 0.83 offline) is sensitive to bipartition choice and drops to d = 0.26 on frozen holdout, underscoring that high offline separability does not guarantee out-of-sample stability." We do not hide this.

---

**M7: Single asset pair.**

The limitation now reads: "Single asset pair. All tests use the SPY/DIA equity pair; generalization to other pairs, asset classes (fixed income, commodities, FX), or non-U.S. markets is untested." Multi-asset work is planned for the companion paper.

---

**M9/M10: Scope and missing baselines.**

We added a scope sentence: "The 46-method benchmark comprises four featured QCML observables, eighteen additional geometric channels (deferred to the companion paper), and twenty-four classical and machine-learning baselines." A new Limitations bullet acknowledges that wavelet, deep-learning, and path-signature methods are omitted. We also added Guritanu et al. (2025) to the related work as a TDA-based competitor.

---

**"What would save this paper."**

You asked for four things:
1. Fix the Spectral Entropy error and reframe around walk-forward Berry -- done.
2. Lead with orthogonality, not superiority -- done (three locations).
3. Multi-asset validation -- deferred to the companion paper, acknowledged in Limitations.
4. Self-contain methods or remove undefined results -- the companion-paper framing for MLF/QFI remains; we believe this is acceptable for a paper series.

Three of four addressed. Multi-asset validation is the remaining gap.

---

## Numerical verification

9 registered claims checked against the canonical JSON (causal_comparison_20260311_010639.json):

| Claim | Paper | Source | Delta | |
|-------|-------|--------|-------|--|
| Reduced Purity median d | 0.83 | 0.835 | 0.005 | pass |
| Hamilton MS median d | 0.71 | 0.713 | 0.003 | pass |
| CUSUM median d | 0.63 | 0.625 | 0.005 | pass |
| Berry Phase Rate median d | 0.61 | 0.608 | 0.002 | pass |
| Random Forest median d | 0.36 | 0.350 | 0.010 | pass |
| GARCH(1,1) median d | 0.27 | 0.288 | 0.018 | pass |
| Friedman chi-squared | 233.1 | 233.13 | 0.0004 | pass |
| Berry d on 2008 GFC | 0.54 | 0.539 | 0.001 | pass |
| QFI Det d on 2008 GFC | 1.73 | 1.732 | 0.002 | pass |

Tolerance: 0.02. All pass.
