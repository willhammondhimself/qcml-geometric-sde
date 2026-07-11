# Pre-registration: Multi-Lag Fidelity hardening (round 2)

**Registered:** 2026-07-11, committed before any of E3-E5 runs.
**Context:** the phase-1 gate (`research/phase1_gate_preregistration.md`) passed via
E1 alone — MLF beat 25/25 fresh nulls at 10 HPO trials, p = 0.038. That is minimum
clearance from one replication. This round decides whether MLF is a foundation
(worth building applications and a revised Paper 2 on) or a fluke.

## Honest disclosure — already observed

- E1 (10 trials, seed 7, nulls 2000-2024): MLF real d = 1.617, null mean 0.893,
  max 1.430, p = 0.038.
- Prior seed-42 run at 12 nulls: real 1.66, p = 0.154.
- Old fixed-param estimate of MLF-vol contemporaneous correlation ≈ −0.02
  (research/honest_hpo_findings.md salvage table).
- The overfitting curve shows in-sample/nested-OOS gaps grow with HPO budget;
  E3 tests whether MLF's nested-OOS d survives a 10× budget.

## E3 — 100-trial confirmation (primary; kill test)

- Command: `python -m experiments.negative_controls --methods "Multi-Lag Fidelity"
  --n-trials 100 --n-perm 25 --seed 11 --normal-mode local --null-seed-base 3000
  --out experiments/outputs/regime_detection/overfitting/leak_test_mlf_100t.json`
- Everything fresh again: HPO seed 11, null placements 3000-3024 (never drawn before).
- **Pass criterion:** permutation p < 0.05 (real must beat all 25 nulls). FAIL otherwise.

## E4 — classical-baseline parity (context; not a kill test)

- Same command shape, same seed/nulls (`--seed 11 --null-seed-base 3000`, shared
  null placements → paired comparison), `--n-trials 100`, for:
  `"Rolling Vol Z"`, `"Turbulence Index"`, `"Absorption Ratio"` — search spaces
  added to `SEARCH_SPACES` at this commit so classical methods get the same HPO
  budget as the geometry has always had.
- Registered readouts: each baseline's real d and permutation p; MLF real d vs each.
- Interpretation rule, fixed now: baselines passing does NOT kill MLF (volatility is
  a positive control — it SHOULD pass). What matters is (a) whether MLF's d is in the
  same class as the passing classical detectors, and (b) E5.

## E5 — vol-orthogonality (kill test for the "adds anything" claim)

- Command: `python -m experiments.mlf_vol_orthogonality` (default MLF params, causal
  scores, post-fit region, SPY+DIA panel).
- **Pass criterion:** |Spearman ρ(MLF, Rolling Vol Z)| < 0.3. If MLF is a vol proxy,
  its surviving the null test is uninteresting (vol also survives) and the channel
  is not a foundation.

## Hardening verdict

- **MLF HARDENED** iff E3 passes AND E5 passes. E4 shapes the paper positioning
  (complement vs. redundant), not the verdict.
- **MLF KILLED** if E3 fails or E5 fails. The program then concludes as the v2
  reassessment paper with MLF's round-1 pass reported honestly as a
  non-replicating borderline.
- No re-runs, seed changes, or threshold adjustments after results are observed.
  Forced deviations documented above the results before re-running.

## Results (to be appended after the runs — empty at registration)

*(pending)*
