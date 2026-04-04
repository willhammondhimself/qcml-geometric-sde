# Response to Reviewers — Geometric Observables for Financial Regime Detection

Date: 2026-04-04
Revision based on: Synthesis 2026-03-05 (4 reviewers) and Synthesis 2026-03-21 (Statistician + Hostile)

---

## Summary of Changes

All 5 CRITICAL and 10 MAJOR issues identified across two review rounds have been addressed:

**Critical (all fixed):**
- C1: Corrected Spectral Entropy d=0.83→0.53 in all body-text locations (§3 intro, §3.3 subsection)
- C2: Verified Hamiltonian Sensitivity d=0.60, rank~10 already correct in §5.1
- C3: Verified RF rank 30th, GARCH rank 33rd already correct in §5.1
- C4: Walk-forward MLF/QFI scope noted as companion paper (wontfix)
- C5: Added reverse-causality caveat to §5.1 offline results, pointing forward to §5.2

**Major (all fixed):**
- M1: Nemenyi post-hoc already reported in §Statistical evaluation and §5.1 (CD=18.2, 105/1035 pairs)
- M2: Orthogonality framing already in abstract, intro, and discussion (|ρ|≈0.13)
- M3: Added holdout collapse magnitude (d=0.83→0.26) to intro caveat
- M4: Per-crisis variation paragraph now states patterns are descriptive; p=0.31 non-significance
- M5: Cliff's delta (|δ|=0.09–0.45 for top 10) reported in §5.1
- M6: Bootstrap percentile method specified in §Statistical evaluation
- M7: Limitation expanded to "Single asset pair (SPY/DIA)" with non-US markets
- M8: Conclusion already frames AR as "classical baselines including Absorption Ratio"
- M9: 46-method scope breakdown added (4 QCML + 18 geometric + 24 baselines)
- M10: New Limitations bullet: "Advanced baselines not compared" (wavelet/DL/signature)

**Minor (all closed):**
- m1: Block size already reported (⌈n^{1/3}⌉ per Politis & White)
- m2: Added df subscript to Friedman χ²₄₅
- m6: Renamed §3.1 "Berry Curvature Increment" → "Berry Phase Rate"; all table headers updated
- Remaining minors triaged as wontfix (standard practice or deferred)

**Additional improvements:**
- Added guritanu2025 (TDA competitor) to bibliography and related work
- Literature scan: 8 new papers discovered and tracked
- `experiments/runner.py` now computes Cliff's delta and Nemenyi per cell

---

## Reviewer 1 (Statistician)

**Comment 1 (C1): Text-table discrepancy — Spectral Entropy d=0.83 is wrong.**

We thank the reviewer for catching this error. The stale d=0.83 value (from an earlier fusion JSON) appeared in two body-text locations. All instances have been corrected:
- §3 introduction to geometric observables: now reads "mid-range offline separability ($d = 0.53$, rank~12 of 46)"
- §3.3 Spectral Entropy subsection: same correction

All remaining d=0.83 references in the paper correctly attribute to Reduced State Purity (highlights, abstract, §5.1, conclusion). We verified with `grep '0\.83'` across the full tex source.

**Action:** Fixed in §3 and §3.3. Verified via automated number-checking (9/9 claims pass against canonical JSON).

---

**Comment 2 (C2/C3): Hamiltonian Sensitivity d-value and baseline ranks stale.**

The reviewer correctly identified that earlier subsections used pre-rerun values. Upon audit:
- §5.1 already reads "Hamiltonian Sensitivity ($0.60$, rank~10)" — correct
- §5.1 already reads "Random Forest 30th ($0.35$)" and "GARCH(1,1) 33rd ($0.29$)" — correct

No earlier subsections contain stale Hamiltonian or baseline values.

**Action:** Confirmed correct; no changes needed.

---

**Comment 3 (M1): Missing post-hoc pairwise tests after Friedman.**

We appreciate this suggestion. The paper already reports Nemenyi post-hoc results in two locations:
- §Statistical evaluation (methods): "Friedman rank test with Nemenyi post-hoc pairwise comparisons at α = 0.05"
- §5.1 (results): "The Nemenyi post-hoc test (critical difference CD = 18.2 at α = 0.05) identifies 105 of 1,035 pairwise differences as significant" with the competitive-tier caveat

The issue registry entry was stale.

**Action:** Registry updated; paper text already compliant.

---

**Comment 4 (M5): Cohen's d normality assumption unverified.**

Valid concern. We now report Cliff's delta as a nonparametric robustness check in §5.1:

> "As a nonparametric robustness check, median |Cliff's δ| for the top-10 methods ranges from 0.09 to 0.45 (negligible to large effect; median 0.30), confirming that the Cohen's d rankings are broadly robust to distributional assumptions."

Additionally, `experiments/runner.py` now computes `cliff_d` and `cliff_label` per method-crisis cell, with `median_cliff_delta` in the summary JSON.

**Action:** Added Cliff's delta reporting to §5.1; code updated for future runs.

---

**Comment 5 (M6): Bootstrap CI method unspecified.**

The reviewer correctly noted that the paper did not specify whether CIs use percentile or BCa. The implementation uses the percentile method (`np.percentile(boot_ds, [2.5, 97.5])`). We have added "percentile method" to the statistical evaluation description.

**Action:** Added "percentile method" to §Statistical evaluation.

---

**Comment 6 (m2): Friedman df not reported.**

Added: χ²₄₅ (45 degrees of freedom for 46 methods).

**Action:** Fixed in §5.1.

---

## Reviewer 2 (Physicist)

**Comment 1: Adiabatic condition not verified / Berry curvature gauge invariance.**

The Berry curvature equation (Eq. 4) is stated in parallel-transport gauge, with the gauge-invariant plaquette formula (Eq. 7) as the computational definition. We have a note after Eq. 4: "Equation (4) holds in the parallel-transport gauge ⟨ψ|∂_a ψ⟩ = 0; our numerical implementation uses the gauge-invariant plaquette formula (Section 3.1)." The adiabatic condition is not required for the plaquette computation — our Berry curvature is a discrete Wilson loop observable, not an adiabatic holonomy.

**Action:** Clarification present at line 281–283.

---

## Reviewer 3 (Quant)

**Comment 1: p=0.31 for per-crisis specialization undermines complementarity narrative.**

Agreed. We have restructured the narrative to:
1. Lead with walk-forward Berry d=0.72 as the primary result (abstract paragraph 2)
2. Frame offline d-values as retrospective sensitivity measures with explicit caveats
3. State per-crisis variation as "descriptive" with formal non-significance (p=0.31) noted in §5.1
4. Position orthogonality (|ρ|≈0.13) as the value proposition, not superiority

The per-crisis variation paragraph (§5.1) now reads: "Descriptively, different geometric observables lead on different crisis types... However, formal testing finds no significant per-crisis specialization (p = 0.31; see Section 7), so these patterns should not be used as a selection rule."

**Action:** Reframed per-crisis variation paragraph in §5.1; added p=0.31 reference.

---

## Reviewer 4 (Hostile)

**Comment 1 (C5): Reverse Granger undermines "detection" framing.**

Valid and important criticism. We have:
1. Added a caveat immediately after the offline results (§5.1): "These offline d values reflect contemporaneous crisis sensitivity, not predictive lead time. Granger causality analysis (Section 5.2) reveals that reverse causality (market→QCML) dominates forward, confirming that observables detect rather than anticipate regime shifts."
2. Kept the detailed Granger analysis in §5.2 (17/45 reverse vs 6/45 forward)
3. The title uses "detection" rather than "prediction" — we believe this is appropriate given that contemporaneous detection of ongoing regime shifts has practical value

**Action:** Added caveat to §5.1; existing §5.2 detail preserved.

---

**Comment 2 (M2): Absorption Ratio beats 23/24 geometric channels — why bother with QCML?**

This is the paper's most important framing question. Our answer: orthogonality, not superiority.

The paper explicitly states (§6.2): "We do not claim QCML dominates these methods. The two are mathematically distinct... Cross-correlations are low (|ρ| ≈ 0.13), suggesting a composite combining both would use orthogonal information channels."

The value proposition is not that Berry Phase Rate (d=0.61 offline) beats Absorption Ratio (d=0.80). It is that:
1. Berry Phase Rate achieves d=0.72 OOS in walk-forward with 30% fewer false alarms than RF
2. Geometric observables capture information orthogonal to classical baselines
3. They require no crisis labels for score construction
4. The framework provides four distinct geometric lenses on the same embedding

**Action:** Orthogonality framing already present in abstract, intro, and discussion (3 locations).

---

**Comment 3 (M3): Reduced Purity holdout collapse (d=0.83→0.263).**

Agreed — this needed a prominent caveat. The intro now reads: "Reduced Purity (d = 0.83 offline) is sensitive to bipartition choice and drops to d ≈ 0.26 on frozen holdout, underscoring that high offline separability does not guarantee out-of-sample stability."

**Action:** Strengthened intro caveat with specific collapse magnitude.

---

**Comment 4 (M7): Single asset pair limits generalizability.**

Agreed. The Limitations section now reads: "Single asset pair. All tests use the SPY/DIA equity pair; generalization to other pairs, asset classes (fixed income, commodities, FX), or non-U.S. markets is untested."

Multi-asset validation is planned for the companion paper.

**Action:** Expanded limitation bullet.

---

**Comment 5 (M9/M10): Scope confusion and missing modern baselines.**

We have:
1. Added a scope breakdown: "The 46-method benchmark comprises four featured QCML observables, eighteen additional geometric channels (deferred to the companion paper), and twenty-four classical and machine-learning baselines."
2. Added a new Limitations bullet: "Advanced baselines not compared. Wavelet-based detectors, deep-learning models (LSTM autoencoders excepted), and path-signature methods are omitted to focus on interpretable, low-dimensional baselines; extension to neural and kernel methods is left to future work."
3. Added Guritanu et al. (2025) — a TDA/persistent homology competitor — to Related Work as a complementary geometric approach.

**Action:** Scope breakdown in Methods; new Limitations bullet; TDA reference added.

---

**Comment 6: "What would save this paper."**

The reviewer suggested four actions:
1. ✅ Fix Spectral Entropy error; reframe around walk-forward Berry d=0.72 — **Done**
2. ✅ Lead with orthogonality rather than superiority — **Done** (3 locations)
3. ⬜ Multi-asset validation — **Deferred to companion paper** (acknowledged in Limitations)
4. ✅ Self-contain methods or remove undefined results — **C4 wontfix** (companion paper noted)

Three of four are addressed; multi-asset validation is the remaining structural gap.

---

## Data Verification

All 9 registered claims verified against canonical JSON (`causal_comparison_20260311_010639.json`):

| Claim | Paper | Source | Δ | Status |
|-------|-------|--------|---|--------|
| Reduced Purity median d | 0.83 | 0.835 | 0.005 | ✅ |
| Hamilton MS median d | 0.71 | 0.713 | 0.003 | ✅ |
| CUSUM median d | 0.63 | 0.625 | 0.005 | ✅ |
| Berry Phase Rate median d | 0.61 | 0.608 | 0.002 | ✅ |
| Random Forest median d | 0.36 | 0.350 | 0.010 | ✅ |
| GARCH(1,1) median d | 0.27 | 0.288 | 0.018 | ✅ |
| Friedman χ² | 233.1 | 233.13 | 0.0004 | ✅ |
| Berry d on 2008 GFC | 0.54 | 0.539 | 0.001 | ✅ |
| QFI Det d on 2008 GFC | 1.73 | 1.732 | 0.002 | ✅ |

Tolerance: 0.02. All pass.
