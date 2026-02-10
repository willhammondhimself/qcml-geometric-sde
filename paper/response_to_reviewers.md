# Response to Reviewer Comments

## Issue 1: "Quantum" Terminology

**Concern:** The "Quantum" branding may trigger unnecessary skepticism given that the mathematics is spectral geometry, not quantum physics.

**Changes made:**

- **Abstract (line ~68):** Added parenthetical disclaimer after first QCML mention: "(hereafter QCML; the name is historical---the mathematics is spectral geometry, not quantum physics)"
- **Keywords (line ~93):** Changed "quantum metric" to "Fubini-Study metric"
- **Section 6.3 "On the Quantum Label":** Reordered to lead with "Berry phase is geometric (Pancharatnam 1956, Simon 1983), not quantum"
- **README.md:** Title changed from "Quantum Geometric Observables" to "Geometric Observables for Financial Regime Detection"; removed ~5 other "Quantum" references
- **Code:** Renamed `QuantumIndicatorSuite` to `GeometricIndicatorSuite` in `qcml_geometry/indicators.py` with backward-compat alias in `__init__.py`
- **Tests:** Updated `tests/test_quantum_indicators.py` to use new class name

**Files modified:** `paper/qcml_geometric_sde.tex`, `README.md`, `qcml_geometry/indicators.py`, `qcml_geometry/__init__.py`, `tests/test_quantum_indicators.py`

---

## Issue 2: Value Proposition vs RF

**Concern:** The paper needs a clearer answer to "why not just use Random Forest?"

**Changes made:**

- **Section 5.2 (Temporal OOS):** Added `\paragraph{Per-crisis wins.}` summarizing that Berry wins 8/11 crises in-sample; QCML methods achieve mean d=0.52 vs RF d=0.31 on post-2020 novel crises; hybrid ensemble reaches d=0.60
- **Section 6.1 (Discussion):** Strengthened with explicit OOS reversal emphasis: "This reversal illustrates why temporal OOS validation is the more informative test"
- **Conclusion:** Reordered key findings to lead with temporal OOS + hybrid as finding (i), moved in-sample results to finding (ii)

**Files modified:** `paper/qcml_geometric_sde.tex`

---

## Issue 3: Supervised HPO for Unsupervised Scores

**Concern:** Using cross-validated hyperparameter selection for unsupervised methods may appear to conflate unsupervised and supervised paradigms.

**Changes made:**

- **Section 4.2 (Methods Protocol):** Added `\paragraph{On supervised hyperparameter selection for unsupervised scores.}` normalizing the practice by citing precedents: bandwidth selection for kernel density estimation (Silverman 1986), threshold calibration for autoencoders, contamination fraction tuning for isolation forests
- **New Appendix F:** "Hyperparameter Sensitivity: Fixed vs. Tuned" with Table comparing fixed defaults (h=8, p=15, pca\_inspired, w=20) vs LOCO-tuned results.
- **New experiment:** `experiments/fixed_hp_ablation.py` --- runs 3 QCML methods x 12 crises with pre-specified defaults, outputs JSON + summary
- **New reference:** Silverman (1986) added to bibliography
- **Actual results (experiment completed):**
  - Berry: fixed d=0.32 vs LOCO d=0.93
  - QFI: fixed d=0.63 vs LOCO d=0.93
  - MLF: fixed d=0.45 vs LOCO d=0.84
  - Honest narrative: fixed defaults produce small-to-medium effects; QFI (d=0.63) still beats BOCPD/IsolationForest baselines; LOCO approximately doubles effect sizes

**Files created:** `experiments/fixed_hp_ablation.py`
**Files modified:** `paper/qcml_geometric_sde.tex`

---

## Issue 4: Walk-Forward Thresholds

**Concern:** The fixed z > 2.0 threshold creates an unfair comparison across methods whose score distributions differ in scale.

**Changes made:**

- **Section 3.4 (Z-Score Thresholding):** Added Algorithm 2 (`\label{alg:far}`) for FAR-calibrated threshold via binary search on training data, targeting alpha = 1 alarm/year
- **Section 5.4 (Walk-Forward Detection):** Rewrote narrative around FAR-calibrated results as primary; replaced single-threshold table with dual-column table (FAR-calibrated + fixed z > 2 for transparency)
- **Code:** Modified `experiments/walk_forward_evaluation.py`:
  - Added `calibrate_threshold()` function (binary search on training z-scores)
  - Added `--far-target` CLI argument (default: None for z>2 only; recommended: 1.0)
  - Modified `_evaluate_method()` to compute both z>2 and FAR-calibrated detection results
  - Updated summary computation to track both threshold types

**Files modified:** `experiments/walk_forward_evaluation.py`, `paper/qcml_geometric_sde.tex`

**Actual results (experiment completed, FAR target = 1.0/yr):**

| Method | FAR-cal Det. | FAR-cal Delay | FAR-cal FAR | z>2 Det. | z>2 Delay |
|--------|-------------|--------------|-------------|----------|-----------|
| Berry  | 0/9 | --- | 0.0/yr | 0/9 | --- |
| QFI    | 3/9 | 44d | 0.0/yr | 0/9 | --- |
| MLF    | 2/9 | 14.5d | 1.2/yr* | 1/9 | 26d |
| RolVol | 3/9 | 28d | 0.0/yr | 4/9 | 20.5d |
| BOCPD  | 0/9 | --- | 0.0/yr | 0/9 | --- |
| RF     | 7/9 | 8d | 1.5/yr* | 0/9 | --- |

\* Minimum threshold (τ=0.5) still exceeds 1.0/yr target.

RF dominates walk-forward detection (7/9 crises detected). Among QCML methods, QFI detects 3/9 and MLF detects 2/9 under FAR calibration. Berry Phase Rate detects 0/9 in walk-forward (its strength is in-sample separability, not real-time alerting).

---

## Issue 5: Null Interaction Test

**Concern:** The ANOVA interaction test yields a null result, but the paper previously framed this as evidence of "complementarity."

**Changes made:**

- **Section 5.5:** Changed title from "Interaction Test: Complementarity" to "Interaction Test" (dropped loaded term)
- **Section 5.5 content:** Rewrote as explicit null: "We find no evidence of differential advantage: F_interaction(1,116) = 0.38, p = 0.54, partial eta^2 = 0.003." Recharacterized as "additive" rather than synergistic complementarity
- **Section 6.1 (Discussion):** Updated paragraph: "The interaction test yields a null result (Section 5.5), consistent with additive rather than synergistic complementarity. This is precisely the condition under which forecast combination is most valuable (Bates & Granger 1969)"
- **Conclusion:** Softened language from "largely orthogonal" to "additive diversity"; from "differential advantage" to "per-crisis strength"

**Files modified:** `paper/qcml_geometric_sde.tex`

---

## Issue 6: Operator Construction

**Concern:** The operator construction is heuristic; the paper should demonstrate it matters and quantify the design space.

**Changes made:**

- **New Section 5.6 "Operator Ablation":** Table comparing 3 conditions (random, PCA-inspired, learned-scaling) x 3 methods with mean Cohen's d
- **Section 2 (Framework):** Updated operator construction paragraph to reference the ablation: "An ablation comparing random, PCA-inspired, and learned-scaling operators appears in Section 5.6"
- **Section 6 (Limitations item 4):** Updated with honest, observable-dependent findings
- **New experiment:** `experiments/operator_ablation.py` --- 3 conditions x 3 methods x 11 crises, with learned scaling optimized on pre-2020 crises via Nelder-Mead (3 scalar weights, bounded [0.1, 10])
- **Actual results (experiment completed):**

| Observable | Random | PCA-Inspired | Learned Scaling |
|-----------|--------|-------------|-----------------|
| Berry     | 0.58   | 0.33        | 0.33            |
| QFI       | 0.65   | 0.67        | 0.65            |
| MLF       | 0.37   | 0.49        | 0.49            |

Key finding: Effect is observable-dependent. PCA-inspired helps MLF (+32%) but hurts Berry (-43%); QFI is tied. Learned scaling converges to near-unity weights (no improvement). Narrative updated honestly in paper.

**Files created:** `experiments/operator_ablation.py`
**Files modified:** `paper/qcml_geometric_sde.tex`

---

## Verification Summary

| Check | Status |
|-------|--------|
| Tests pass (92/92, excluding pre-existing failures) | PASS |
| Paper compiles (pdflatex x3, no errors) | PASS |
| No undefined references | PASS |
| All new Python scripts compile | PASS |
| README no longer says "Quantum" in title | PASS |
| `QuantumIndicatorSuite` renamed with backward compat | PASS |
| Walk-forward table shows FAR-calibrated as primary | PASS |
| Interaction section states explicit null | PASS |
| Operator ablation table added (3 conditions) | PASS |
| Fixed-HP ablation table in Appendix F | PASS |
| `response_to_reviewers.md` documents all 6 issues | PASS |

## Outstanding TBD Values

All TBD placeholders have been filled with actual experiment results. Zero `[TBD]` entries remain in the paper.
