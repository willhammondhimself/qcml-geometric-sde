# Pre-registration: multi-asset correlation-geometry vs the Absorption Ratio

**Committed before any evaluation run.** This fixes the universe, slices, detectors,
baselines, targets, models, and success criteria so the slice/subset sweep cannot be
fished for a lucky winner. All p-values are corrected for multiplicity across the entire
slices×detectors grid (Holm-Bonferroni for FWER and Benjamini-Hochberg for FDR). We
commit to *believing the result either way*.

## Motivation (one line)
Prior work fed the geometry SPY+DIA (~95% correlated) — no real correlation manifold. Here
the geometry gets a genuinely diverse, high-dimensional manifold; we test whether its
observables beat the Absorption Ratio at characterizing/predicting cross-asset regime risk.

## Universe (master panel, yfinance, daily)
- Equity sectors (11): XLK XLF XLE XLV XLY XLP XLI XLU XLB XLRE XLC
- Macro / cross-asset (9): TLT IEF HYG LQD GLD DBC UUP EEM VNQ
- Crypto (2): BTC-USD ETH-USD

## Pre-registered slices (fixed; no additions post-hoc)
| slice | members | approx start |
|---|---|---|
| `equity_sectors` | 9 original SPDR sectors (excl. XLRE/XLC) | 2007 |
| `macro_crossasset` | TLT IEF HYG LQD GLD DBC UUP EEM VNQ | 2007 |
| `sectors+macro` | above two combined | 2007 |
| `full` | sectors+macro + XLRE + XLC | 2018 |
| `full+crypto` | full + BTC + ETH | 2018 |
Each slice uses its max common history; #crises (from `ALL_CRISES`) recorded per slice.
**Diversity metric** (pre-registered): `div = 1 − mean|corr|` over the slice (higher = more
diverse manifold).

## Geometry detectors under test (cross-asset-sensitive; fixed list)
ReducedPurity (bipartition entanglement = contagion), DimensionalityCollapse,
SectionalCurvature, QFI-Determinant, Berry Phase Rate. Input = correlation-manifold features
(rolling pairwise correlations + corr-matrix eigen-summaries) → PCA → geometry. Causal:
operators fit on an early prefix; scores expanding-z.

## Baselines to beat (already implemented in baselines.py)
Absorption Ratio (primary), Turbulence Index, Cross-Sectional Dispersion, Transfer Entropy;
plus a panel realized-volatility baseline.

## Targets
- **Primary (prediction headline):** forward 20-trading-day max drawdown of the equal-weight
  panel. Secondary: forward avg pairwise correlation; forward Absorption-Ratio change.
- **Detection:** `ALL_CRISES` windows, evaluated with the corrected **local-normal** metric.

## Hypotheses + success criteria (falsifiable, pre-committed)
- **H1 (prediction, headline).** Expanding-window OOS R² (Ridge + HistGBM): `AR + geometry`
  beats `AR-only` (ΔR² > 0) with block-bootstrap p < 0.05 **after BH** across the grid.
  Primary slice for the headline = `full+crypto`; all slices reported.
- **H2 (detection).** A geometry detector's local-normal nested-OOS d must exceed BOTH the
  random-window null q95 (leak test, ≥12 perms) AND the Absorption Ratio's d on the same
  crises — after BH.
- **H3 (diversity scaling).** Spearman ρ between slice diversity `div` and geometry's edge
  (ΔR² for H1 / detection margin for H2) across slices is **positive** (report ρ, p).

## Multiplicity (the anti-fishing rule)
Grid = slices (5) × geometry detectors (5) × {prediction, detection} = 50 primary tests.
Every reported p is Holm- and BH-adjusted across this grid. A result "counts" only if it
survives BH at q=0.05. No slice or detector is added after seeing results.

## The commitment
If, after correction, `AR + geometry ≤ AR-only` and no detector beats AR + the null, that is
the result — it strengthens the reproducibility paper. A surviving positive is a real finding
and a real paper. We do not p-hack the slice sweep; the correction above is the guard.
