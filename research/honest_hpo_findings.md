# Honest nested-HPO findings & paper reframing

**Status:** concluded. Numbers below are from the full run (9 OOS windows, 50 trials,
SPY+DIA, `overfitting_stats_full.json`) + a Berry gauge A/B + the random-window
leak/null test (`negative_controls.py`; outputs in
`experiments/outputs/regime_detection/overfitting/leak_test_*.json`).

## TL;DR (final, after negative controls)
Three compounding problems inflated the geometric detectors' apparent crisis-
detection performance, and **under proper controls the edge does not survive**:
1. **In-sample HPO** (selection look-ahead) — fixed by nested walk-forward HPO.
2. **Circular provenance** — the headline d=0.72 was hardcoded from the LaTeX.
3. **A confounded metric (the decisive one):** "crisis-window vs all-prior-history"
   Cohen's d is corrupted by non-stationarity. Positive control: *volatility*, the
   textbook crisis signal, scores **lower on crises (0.42) than on random windows
   (0.84)** under it. Fix = a **local** pre-window baseline (vol then behaves:
   crisis **2.09** vs random **0.76**).

Under the corrected local-normal metric + a random-window null (n_trials=10):
| detector | real (crises) | null (random) | crisis-specific? |
|---|---|---|---|
| Volatility (control) | 2.09 | 0.76 | **yes** |
| Berry Phase Rate | 1.51 | mean 1.21 (0.80–1.91) | **no** (p=0.29, mid-null) |
| Multi-Lag Fidelity | 1.66 | mean 1.10 | maybe (beats 6/6 nulls, p=0.14, underpowered) |

**Berry — the paper's headline — shows no crisis-specific signal even after fixing
the metric.** The detectors fire on *any* window (high background separability), so
random windows separate from their baseline ~as well as crises. The earlier "Berry
survives at d≈0.93" was itself under the broken global-normal metric and is void.

### Salvage sweep (local-normal, random-window null) + what the observables *are*
Tested every plausible channel for crisis-specific signal under the corrected metric:
| channel | real | null_q95 | verdict | contemp corr w/ vol |
|---|---|---|---|---|
| Multi-Lag Fidelity | 1.66 | 1.64 | weak/borderline (p=0.15) | **−0.02 (vol-orthogonal)** |
| Berry Phase Rate | 1.51 | 1.78 | none (p=0.29) | +0.49 |
| Spectral Entropy | 1.12 | 1.83 | none (p=0.46) | **+0.67 (vol proxy)** |
| Spectral Gap | 0.71 | 1.09 | none (p=0.62) | +0.40 |
| Reduced Purity (offline #1) | 0.71 | 1.21 | none (p=0.69) | **+0.63 (vol proxy)** |

**Interpretability finding (the constructive one):** most geometric observables are
**dressed-up contemporaneous volatility** — Spectral Entropy (+0.67), Reduced Purity
(+0.63), Berry (+0.49) all strongly track current realized vol; that is *why* they
"detect" crises (vol is high in crises) and why they fail the crisis-specific null.
**None** adds predictive value beyond vol (forward-vol/drawdown partial corr ≈0, all
p>0.18, while vol itself predicts forward vol at r=0.58). The lone exception is
**Multi-Lag Fidelity**, the only vol-orthogonal observable (−0.02) and the only one to
beat the random-window null — but weak (p≈0.15) and with no forward-predictive content.

### Volatility-forecasting horse race (the fairest test of "geometry predicts vol")
Honest expanding-window OOS forecast of forward-20d log realized vol; HAR-RV baseline
vs HAR+geometry vs geometry-only; Ridge and gradient boosting:
| model | HAR (vol only) | HAR + geometry | geometry only | Δ geometry adds |
|---|---|---|---|---|
| Ridge | 0.430 | 0.416 | 0.178 | −0.014 |
| GBM | 0.356 | 0.309 | 0.135 | −0.048 |
Geometry alone predicts ~18% of forward-vol variance (so it *does* carry vol info),
but HAR (plain rolling vol) more than doubles that, and geometry adds **nothing**
on top of HAR (Δ ≤ 0). Conclusion: the geometry is a lossy, redundant re-encoding of
realized volatility. Single-asset, it does not beat or augment vol — confirmed across
three independent fair tests (crisis detection, linear incremental, nonlinear OOS).
Untested frontier: **multi-asset / cross-correlation** geometry (contagion), which is
a genuinely different object than single-asset vol.

### Superseded intermediate readings (audit trail)
A 5-window demo first suggested Berry overfits to d→0.10 (pessimistic subset); the
full 9-window run then showed d≈0.93 under global-normal. Both are artifacts of the
confounded metric; the local-normal leak test is the authoritative result.

## The setup that was wrong
`optuna_hpo.py` selects hyperparameters by maximizing median d over `OPT_CRISES`
— the *same* crisis panel the paper reports on (`optuna_hpo.py:269`). That is
look-ahead in *model selection*; the per-crisis causal cutoff only prevents
within-day preprocessing leakage. The honest fix is nested walk-forward HPO
(`walk_forward_hpo.py`): tune on chronologically-prior crises only, evaluate OOS
on the held-out crisis.

## Full-run results (9 windows, 50 trials)
| Detector | in-sample d | nested-OOS d | gap | gap slope/log₂ | PBO | deflated d |
|---|---|---|---|---|---|---|
| **Berry Phase Rate** | 0.91 | **0.93** | −0.02 | −0.01 (flat) | 0.39 | **0.68** |
| Geometric Phase Rate | 0.88 | 0.48 | +0.41 | 0.16 | 0.60 | 0.31 |
| Multi-Lag Fidelity | 0.75 | 0.65 | +0.10 | −0.11 | 0.61 | 0.46 |
| Spectral Gap | 0.67 | 0.34 | +0.33 | 0.06 | 0.21 | 0.22 |

Berry per-window OOS d: Volmageddon 2.24, COVID 1.79, Euro 1.17, Rates 1.01,
Repo 0.93, Flash 0.64, Q4 0.56, **China 0.12, SVB 0.01** (median 0.93).

## The gauge fix is a no-op for the headline (and we proved it)
Berry with `berry_aggregation="f01"` already uses the **gauge-invariant Wilson
loop** (`berry_curvature_2d`). A/B over 9 windows: gauge-tunable **0.928** vs
gauge-forced-off **0.928** — identical per window. The opt-in `gauge_fix` only
matters for the `np.angle`-based **Geometric Phase Rate** (still overfits,
deflated 0.31). So physics change #1 is correct but inert for the headline; the
geometry isn't where the headline's leverage is.

## What it means for the paper
The central empirical claim (geometric observables detect financial crises) does
**not** survive rigorous negative controls. The apparent effect was the compound of
HPO selection bias + a non-stationarity-confounded metric + the detectors' generic
high background separability. The honest options:
1. **Reframe as a methodology / negative-results paper** (recommended): a rigorous,
   positive-control-validated evaluation framework (nested HPO, PBO/CSCV, deflated
   d, random-window leak test, the local-normal metric fix) that shows the geometric
   crisis-detection edge is an artifact. This is genuine, reusable, and publishable.
2. **Salvage specific channels** only if they survive: Multi-Lag Fidelity is the one
   candidate (beats 6/6 nulls) — re-run with ≥20 permutations under local-normal to
   confirm or kill it; check QFI / Spectral Gap likewise.
3. **Pivot the geometry's role** away from crisis-vs-window separation (e.g., the
   topological/Chern narrative, or geometric features inside a supervised model with
   its own honest CV).
Baseline parity (HPO the classical baselines the same way) is moot for the main
claim now, but still needed if any channel is salvaged.

## Multi-asset correlation-manifold test (the strongest, most on-thesis shot)
Pre-registered (`research/multi_asset_preregistration.md`), then run: 5 diversity-ordered
asset slices (sectors / macro / combined / +crypto), correlation-manifold features fed to
5 cross-asset-sensitive detectors, forward systemic-risk prediction vs the Absorption Ratio
and a volatility baseline, expanding-window OOS R², Holm/BH across the 30-cell grid.
- **Geometry beats Absorption Ratio: 0/30 cells** (BH q<0.05). The lone uncorrected glimmer
  (sectors+macro, p=0.034) → BH p=0.259.
- **Geometry beats volatility: 0/30.** Vol is the best forward-systemic-risk predictor on
  every slice; AR and geometry add nothing over it.
- **H3 falsified:** geometry's edge does NOT scale with manifold diversity — Spearman
  ρ=−0.10 (p=0.87). The mechanism the thesis rests on (more correlation structure → more
  geometric edge) is absent. This is a *falsification*, not just a non-finding.
Conclusion: even given a genuinely diverse multi-asset manifold — the method's intended use
case — the quantum geometry carries no edge over simple systemic-risk measures, and forward
systemic concentration is best predicted by volatility. The door is closed, rigorously.

## Leak test — landed, negative
Random non-crisis windows through the same nested protocol (`leak_test_berry.json`,
2026-06-22): Berry real nested-OOS median d = 0.60 vs null mean 0.65 (q95 0.87),
p = 0.71, `passes_leak_test: false`. Random windows separate as well as crises —
consistent with the high-background-separability finding above; the nested-OOS d
is not crisis-specific signal. (The 0.93 it was meant to gate was already void on
independent grounds — broken global-normal metric.)

## Still open
- Multi-Lag Fidelity ≥20-permutation re-run under local-normal (the one salvage candidate).
- Full 100-trial / all-detector confirmation + baseline parity, only if a channel survives.

## Methods & citations (grounded, verified)
- PBO via CSCV — Bailey, Borwein, López de Prado & Zhu (2017), *J. Comp. Finance* 20(4):39–69.
- Deflated effect size — Bailey & López de Prado (2014), *J. Portfolio Mgmt* 40(5):94–107.
- Purged/embargoed CV — López de Prado (2018), *Advances in Financial ML*, Ch. 7.
- Nested-CV selection bias — Cawley & Talbot (2010), *JMLR* 11:2079–2107.
- Multiplicity — Holm (1979); Benjamini-Hochberg (1995); White (2000); Hansen (2005).
- Gauge-invariant Chern / parallel transport — Fukui-Hatsugai-Suzuki (2005), *JPSJ* 74(6):1674;
  Wilczek-Zee (1984), *PRL* 52:2111.

## Tooling added
`experiments/walk_forward_hpo.py` (nested HPO), `overfitting_stats.py` (PBO/CSCV,
deflated d, overfitting curve, gap significance, multiplicity), `hpo_cache.py`
(memoization → 15,000× warm), `negative_controls.py`
(label-permutation leak test). Opt-in physics flag `gauge_fix` in `core.py`.
Tests: `tests/test_{walk_forward_hpo,overfitting_stats,hpo_cache,physics_variants}.py`.
