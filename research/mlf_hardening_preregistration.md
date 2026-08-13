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

## Deviations

- 2026-07-11, before any E4 result was observed: the first E4 launch crashed to
  all-NaN because `crisis_cohens_d` passed `causal_fit_length=` to every detector
  constructor and the classical baselines don't accept it (TypeError swallowed by
  the invalid-config guard). Fixed by passing the kwarg only to constructors that
  declare it (the baselines are causal by construction); verified end-to-end on
  2020_covid (vol d=2.28, turbulence 2.87, AR 1.83, all finite). E4 re-launched
  with the identical registered command. E3/E5 unaffected (geometric path
  unchanged; E5 doesn't use `crisis_cohens_d`).

## Results (appended 2026-07-13; registration 2026-07-11; one documented deviation above)

**E5 — PASS.** MLF vs Rolling Vol Z, causal scores, 4,720 post-fit days:
Spearman ρ = −0.220, Pearson −0.132. Under the registered |ρ| < 0.3 bar, though
notably less orthogonal than the fixed-param estimate (−0.02) suggested.

**E3 — FAIL.** `leak_test_mlf_100t.json`: real nested-OOS median d = 1.221 vs
null mean 1.227 (q95 1.790, max 1.980), permutation p = 0.462. The round-1 pass
(10 trials, p = 0.038) did not survive a 10× HPO budget: real d fell 1.62 → 1.22
while the null mean rose 0.89 → 1.23.

**E4 — the positive control FAILS.** Identical protocol, same null placements:

| method | real d | null mean | null q95 | p |
|---|---|---|---|---|
| Multi-Lag Fidelity | 1.221 | 1.227 | 1.790 | 0.462 |
| Rolling Vol Z (positive control) | 0.577 | 1.206 | 2.194 | 0.885 |
| Turbulence Index | 0.617 | 0.678 | 1.001 | 0.654 |
| Absorption Ratio | 0.872 | 1.771 | 2.651 | 0.923 |

## Verdict

**Formal (per the registered rule): MLF is NOT hardened.** E3 failed; nothing is
built on the channel. The rule was fixed before the runs and stands.

**Scientific interpretation (registered rule cannot capture what E4 revealed):**
at 100 trials the test instrument fails its own positive control — realized
volatility, the canonical crisis signal, scores *half* its null mean. Given
enough HPO budget, random calm-period windows can be tuned to separate from
their pre-windows better than real crises separate from theirs (real crises
have anticipatory volatility, so their local baselines are already elevated —
the local-normal metric structurally penalizes signals that ramp up early).
E3's failure is therefore uninformative about MLF specifically: **no detector,
real or fake, passes this test at realistic HPO budgets.**

The compounding finding across both pre-registered rounds: (i) the global
metric rewards non-stationarity; (ii) the local-normal fix penalizes
anticipatory signals; (iii) the random-window permutation null degrades with
HPO budget until it cannot validate volatility itself. Every layer of the
standard evaluation stack for unsupervised crisis detectors has a demonstrated
failure mode. This — not any single detector's death — is the result.
