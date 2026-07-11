# Pre-registration: final gate for the geometric-observables program

**Registered:** 2026-07-11, committed to the public repo before either experiment runs.
**Decides:** whether any geometric channel carries signal worth building on, or the
program concludes as a methodology / negative-results paper (`paper/v2_reassessment_draft.md`).

## Honest disclosure — what has already been observed

This is not a blind pre-registration. The following results were observed before this
document was written, and motivate it:

- Crisis-detection headline void under the local-normal metric; Berry Phase Rate fails
  the random-window leak test (d=0.60 mid-null, p=0.71, `leak_test_berry.json`).
- Salvage sweep at n_perm=12 (`leak_test_salvage.json`): only Multi-Lag Fidelity is
  borderline — real d=1.66 vs null mean 1.13, q95 1.64, p=0.154. All other channels dead.
- Multi-asset systemic sweep: geometry 0/30 cells vs Absorption Ratio and vs volatility;
  H3 (diversity scaling) falsified (`research/multi_asset_preregistration.md` run).
- Forward-vol horse race (UNPURGED walk-forward): geometry-only OOS R² ≈ 0.18 vs
  HAR ≈ 0.43; HAR+geometry ≤ HAR.

What is genuinely unobserved, and therefore registrable:
1. Multi-Lag Fidelity under a fresh seed with a null sample large enough to resolve
   p < 0.05 (prior run: 12 permutations, min resolvable p = 1/13).
2. The vol horse race under a purged (embargoed) walk-forward, with a significance test
   on the incremental ΔR² (prior run had a train/test boundary leak — labels spanning
   the test window — and reported no p-value).

## E1 — Multi-Lag Fidelity replication (crisis-specificity)

- Command: `python -m experiments.negative_controls --methods "Multi-Lag Fidelity"
  --n-trials 10 --n-perm 25 --seed 7 --normal-mode local --null-seed-base 2000
  --out experiments/outputs/regime_detection/overfitting/leak_test_mlf_prereg.json`
- Fully fresh replication: HPO seed 7 (prior run: 42) and null-window placements
  seeded 2000..2024 (prior run: 1000..1011, which `--seed` does not control — the
  `--null-seed-base` flag was added for this registration so no null placement is
  a reuse of an already-observed draw).
- Protocol otherwise identical to the prior salvage run: 9 OOS crisis windows,
  10 Optuna trials per expanding window, local-normal metric.
- **Pass criterion:** permutation p < 0.05, i.e. p = (1 + #{null medians ≥ real
  median}) / 26 — the real run must beat all 25 nulls. Anything else (including
  p in [0.05, 0.10]) is a FAIL. `null_q95` is reported for context only.

## E2 — Purged forward-vol horse race (incremental predictive value)

- Command: `python -m experiments.volatility_forecasting` at the registered commit
  (purge = HORIZON = 20 trading days between the expanding train window and each
  test block; refit every 63d; SPY+DIA 1995–2024; 6 causal geometry channels).
- **Primary cell:** GBM (HistGradientBoosting), the nonlinear model — the surviving
  hypothesis is that geometry carries nonlinear/multivariate information vol misses.
- **Pass criterion:** ΔR² = OOS R²(HAR+geometry) − OOS R²(HAR) > 0 with one-sided
  block-bootstrap p < 0.05 (n_boot=2000, block=63, seed=0, as implemented in
  `delta_r2_pvalue`). Ridge is reported as a secondary cell and does not affect
  the verdict.

## Gate rule

- **Gate PASSES** iff E1 passes OR E2 passes. The surviving channel/result becomes
  the seed of the next phase (application selection per the roadmap's
  decide-at-gate rule).
- **Gate FAILS** iff both fail. The program's empirical claim is then closed:
  the deliverable is the reassessment paper (v2), reframing the work as a
  positive-control-validated evaluation methodology with a rigorous negative result.
- No re-runs, seed changes, threshold adjustments, or added channels after results
  are observed. Deviations, if forced (e.g. crash), will be documented in this file
  above the results, before re-running.

## Results (appended 2026-07-11, same day as registration; no protocol deviations)

**E1 — PASS.** `leak_test_mlf_prereg.json`: real nested-OOS median d = **1.617**;
25/25 fresh nulls below it (max 1.430, q95 1.374, mean 0.893); permutation
p = **0.038** (< 0.05). Note: the JSON's `passes_leak_test: false` field is the
legacy `null_mean < 0.2` heuristic from the original harness-leak check, not this
registration's criterion; it is reported for completeness and does not bear on
the registered verdict.

**E2 — FAIL.** Purged forward-vol horse race (purge=20d):

| model | HAR | HAR+geo | geo-only | Δ(geo adds) | p(Δ≤0) |
|---|---|---|---|---|---|
| GBM (primary) | 0.349 | 0.277 | 0.106 | **−0.071** | 0.995 |
| Ridge (secondary) | 0.430 | 0.413 | 0.174 | −0.016 | 0.963 |

Geometry subtracts predictive value on the primary cell. Purging also shrank
geometry-only R² from the previously observed 0.18 to 0.106 — roughly a third of
its apparent forward-vol information was the boundary leak.

**Gate verdict: PASSES via E1.** The surviving thread is Multi-Lag Fidelity's
crisis-specific separation (vol-orthogonal, contemporaneous corr ≈ −0.02). The
program's predictive claims (forward vol, forward systemic risk) remain closed
per the disclosures above; what survives is a detection claim for one channel.
Next verification per the roadmap: 100-trial confirmation + classical baseline
parity for MLF before anything is built on it.
