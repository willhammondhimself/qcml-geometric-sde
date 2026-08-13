# v2 reassessment draft (supersedes the original preprint)

Working title options:
1. **When Geometry Becomes Volatility: A Rigorous Reassessment of Quantum-Geometric
   Regime Detection**
2. Quantum-Geometric Regime Detectors Reduce to Volatility: A Reproducibility Study
   and an Open Evaluation Framework
3. How an Evaluation Confound Inflates Effect Sizes in Geometric Regime Detection

(Recommend #1 — it states the finding and is memorable.)

---

## Abstract (draft)
Geometric and "quantum-inspired" observables have been proposed as unsupervised
detectors of financial-market regimes, and an earlier version of this work reported
large crisis-detection effect sizes for them. Here we subject that claim — and the
evaluation methodology behind it — to rigorous controls, and reach a different
conclusion. We show that (i) selecting detector hyperparameters on the same crisis
panel used for evaluation inflates effect sizes through selection bias; (ii) the
standard "crisis-window vs. prior-history" Cohen's *d* is confounded by
non-stationarity — under a positive control, realized volatility (the canonical
crisis signal) scores *lower* on crises (d≈0.4) than on random non-crisis windows
(d≈0.8); and (iii) once the metric is corrected (a matched local pre-window baseline)
and a random-window null is added, none of the geometric observables shows
crisis-specific separation beyond chance. A predictive analysis shows the observables
are largely lossy re-encodings of realized volatility: they explain ~18% of forward
volatility out of sample versus 43% for a simple HAR baseline, and add nothing on top
of it. We release an open, leak-free evaluation framework — nested walk-forward HPO,
probability of backtest overfitting (CSCV), deflated effect sizes, and a random-window
leak test — so geometric and quantum-inspired financial indicators can be assessed
without these traps. The results are a cautionary case study for the rapidly growing
quantum-machine-learning-for-finance literature.

## Contributions (reframed)
1. **An open, leak-free evaluation framework** for unsupervised regime detectors:
   nested walk-forward HPO (no model-selection leakage), PBO via CSCV, deflated effect
   sizes, and a random-window label-permutation leak test. (Reusable; released.)
2. **A non-stationarity confound in the field-standard metric**, identified and fixed.
   The "crisis vs. all-prior-history" effect size rewards any window that differs from
   a long, multi-regime history; we validate the flaw with a volatility positive
   control and fix it with a matched local baseline.
3. **A rigorous reassessment** showing that quantum-geometric regime observables carry
   no crisis-specific signal beyond a random-window null, and reduce to a lossy proxy
   for realized volatility (high contemporaneous correlation; ~18% OOS forward-vol R²;
   zero incremental value over HAR).
4. **A pre-registered multi-asset stress test that closes the obvious escape hatch.**
   Single-asset geometry is ~1-D, so we built a genuinely diverse correlation manifold
   (sector ETFs + bonds/gold/oil/FX/credit + crypto) and pre-registered a 5-slice,
   30-cell sweep predicting forward systemic risk vs. the Absorption Ratio and a vol
   baseline (Holm/BH corrected). Geometry beats the Absorption Ratio in **0/30** cells
   and volatility in **0/30**; crucially its edge is **independent of manifold
   diversity** (Spearman ρ=−0.10), *falsifying* the cross-asset-structure mechanism —
   not merely failing to find it. Reviewers cannot say "but you didn't try multi-asset."

## Key results to feature (all reproducible from the released code)
| finding | evidence |
|---|---|
| Metric confound | volatility positive control: crisis d=0.42 < random d=0.84 (global normal); fixed → 2.09 vs 0.76 (local normal) |
| No crisis-specific signal | local-normal random-window null: Berry real 1.51 vs null 1.21 (p=0.29); Reduced Purity (old offline #1) p=0.69; only Multi-Lag Fidelity weakly survives (p≈0.15) |
| The MLF survivor dies under pre-registration | pre-registered fresh replication PASSES at 10 HPO trials (real 1.62 vs 25/25 nulls, p=0.038) then FAILS at 100 trials (real 1.22 ≈ null mean 1.23, p=0.46) — the permutation verdict is HPO-budget-dependent |
| The null test fails its positive control | at 100 trials, nested-HPO'd Rolling Vol Z scores HALF its null mean (real 0.58 vs null 1.21, p=0.89); Turbulence p=0.65, Absorption Ratio p=0.92 — no detector, real or fake, passes at realistic budgets |
| The local-normal fix has its own confound | local pre-crisis baselines are already elevated by anticipatory volatility, so the corrected metric structurally penalizes early-ramping signals while HPO pumps calm-period nulls |
| Observables ≈ volatility | contemporaneous corr with current vol: Spectral Entropy 0.67, Reduced Purity 0.63, Berry 0.49 (MLF is the exception: ρ=−0.22) |
| No predictive edge | purged forward-vol OOS R²: geometry-only 0.106, HAR 0.349 (GBM); geometry SUBTRACTS from HAR (Δ=−0.071, p=0.995); a third of geometry's unpurged R² (0.18) was train/test boundary leakage |
| HPO inflation | in-sample vs nested-OOS gap grows with trial budget (overfitting curve) |

## How to position the preprint update (mechanics)
- Post as **v2**, explicitly *superseding* v1; add a one-line note: "v2 reassesses the
  v1 results under rigorous controls and supersedes them." This is normal preprint
  practice and reads as integrity, not retraction.
- Lead the paper with the **framework + the confound**, not with the negative result.
  The geometry becomes the case study the framework dissects.
- Generalize: frame the confound and the leak test as applying to the *class* of
  methods, not just this one — a field-level contribution.
- Keep the original geometric/theoretical exposition (Sections 2–3) — it's the setup;
  only the empirical claims (Sections 5+) get replaced.

## Suggested v2 structure
1. Intro — the hype around quantum/geometric finance + the reproducibility problem.
2. The geometric observables (kept from v1, condensed).
3. **The evaluation framework** (new headline): nested HPO, leak test, PBO, deflation.
4. **The metric confound + fix** (new), with the volatility positive control.
5. **Reassessment results** (replaces v1's leaderboard): no crisis-specific signal;
   volatility-proxy analysis; purged predictive horse race.
6. **The two pre-registered gates** (new; both public commits before results):
   the last surviving channel passes at 10 trials, dies at 100; then the positive
   control dies too. Verdicts from this evaluation stack are budget-dependent, and
   at realistic budgets the stack validates nothing — including volatility.
   Prescription: budget-sweep the null test and always run a positive control at
   every budget; a "pass" at one budget is not evidence.
7. Discussion — cautionary tale for QML-in-finance; evaluation itself is the open
   problem; sets up the follow-up audit of the TDA/curvature crisis-detection
   literature (Gidea-Katz 2018; Samal et al. 2021; Sandhu et al. 2016), whose
   published claims rest on the layers shown here to fail.
