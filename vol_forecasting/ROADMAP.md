# Roadmap — QCML Volatility Forecasting

Fluid. Phases can be reordered, split, or dropped. Update this when priorities shift.
Mark phases ✅ done, 🔄 in progress, ⏸ paused, ❌ abandoned (with reason).

---

## Phase 1 — Foundation & Sanity Check
**Goal**: Verify the core hypothesis is worth pursuing before building anything complex.
**Success criteria**: QCML features Granger-cause realized vol at ≥1 lag, p < 0.05.

- [ ] Build realized vol target series (close-to-close, 5d and 21d forward)
- [ ] Extract QCML z-score features (QFI, Berry, SpectralGap) using existing detectors
- [ ] Scatter plots: QCML feature t vs RV t+21 — visual sanity check
- [ ] Granger causality test: do any QCML features lead RV? (if all fail, rethink)
- [ ] Cross-correlation function plots (CCF) at lags 1–30d
- [ ] Save outputs to `results/phase1_granger.json` and `figures/phase1_*.pdf`

**Estimate**: 2–3 days

---

## Phase 2 — Tail Volatility Forecasting ✅
**Goal**: Predict extreme vol (90th percentile) where the QCML signal lives.
**Result**: Clean null. QCML does not improve tail vol forecasting (DM p=0.36).

- [x] Walk-forward engine: 15 folds (2010-2024), expanding window from 2004
- [x] 4 models: Quantile HAR, Quantile HAR+QCML, RF HAR, RF HAR+QCML
- [x] Evaluation: pinball loss, exceedance rate, Christoffersen conditional coverage
- [x] Diebold-Mariano tests with Holm-Bonferroni correction
- [x] Regime-conditional analysis: calm vs stressed regimes
- [x] 4 publication figures

**Script**: `vol_forecasting/experiments/phase2_tail_vol_forecasting.py`

---

## Phase 2b — Vol Dynamics Forecasting ✅
**Goal**: Predict vol *change* (delta_rv_5d) and vol *spikes* with richer features.
**Result**: Comprehensive null. Vol change is unpredictable; all models R^2 < 0 vs naive.

Key insight tested: HAR predicts where vol IS; QCML detects where vol is GOING.
If QCML captures transitions, it should predict vol change, not vol level.

- [x] 9 delta-RV models: HAR-delta OLS, RS-HAR, GBM × 3 tiers, GBM quantiles, GeoARCH, GARCH
- [x] 2 vol-spike logistic models
- [x] 3 pre-registered DM tests with Holm-Bonferroni
- [x] Feature tiers: T1=6 (HAR-delta), T2=10 (+QCML), T3=13 (+interactions)
- [x] Novel Geometric ARCH (QCML in variance equation)
- [x] Regime-conditional analysis (pre-transition vs continuation)
- [x] 4 publication figures
- [x] GBM feature importances per fold

**Key findings:**
- ALL models have R^2 < 0 vs naive (delta=0). Vol change is a random walk at weekly scale.
- GeoARCH vs GARCH: DM p_raw=0.04 (closest to significance), p_adj=0.12 after Bonferroni.
- QCML features dominate GBM importance but model doesn't beat naive (overfitting).
- Vol spike: AUC-ROC +0.016 with QCML (not significant, DM p=0.78).

**Script**: `vol_forecasting/experiments/phase2b_vol_dynamics.py`

---

## Phase 3 — QCML Vol Models
**Goal**: Add QCML features to baselines and measure incremental improvement.

- [ ] QCML-HAR: HAR-RV + QCML features (OLS, then elastic net for regularization)
- [ ] Feature selection: which QCML observable matters most?
  - QFI susceptibility
  - Berry phase rate
  - Spectral gap
  - Multi-lag fidelity
- [ ] Regime-conditional analysis: does QCML add more in high-vol regimes?
- [ ] Ablation: single feature vs all features vs HAR-only
- [ ] Diebold-Mariano test: QCML-HAR vs HAR-only (null: equal predictive accuracy)
- [ ] Mincer-Zarnowitz regressions for each model

**Estimate**: 1 week

---

## Phase 4 — WRDS Extension (Optional, High Value)
**Goal**: Add OptionMetrics IV surface as features. Variance risk premium = IV² - RV².

- [ ] Get WRDS credentials working (`WRDS_USERNAME` in `.env`, test connection)
- [ ] Fetch SPY IV surface: `SPY_iv30`, `SPY_iv90`, `SPY_term_slope`
- [ ] Add VRP (variance risk premium) as feature: VRP_t = IV30_t² - RV_{t-21:t}²
- [ ] QCML-HAR-IV model: HAR + QCML + IV features
- [ ] Compare: how much does IV add on top of QCML? Are they complementary?
- [ ] Note: results with WRDS may not be reproducible for reviewers without access

**Estimate**: 3–4 days (if credentials work smoothly)

---

## Phase 5 — Geometric ARCH (Speculative / Stretch)
**Goal**: Can we replace the GARCH variance equation with a QCML geometric update?

The idea: GARCH(1,1) says `σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}`.
A geometric ARCH might say `σ²_t = f(curvature_t, spectral_gap_t)` — the variance
is determined by the geometry of the state space, not lagged squared residuals.

This is genuinely novel but speculative. Only pursue if Phases 1–3 show strong results.

- [ ] Theoretical motivation: write 1-page derivation connecting QFI to conditional variance
- [ ] Implementation: two-stage (fit QCML geometry, then regress σ² on geometric observables)
- [ ] Test: does this outperform standard GARCH on tail events?

**Estimate**: 1–2 weeks (risky, may not pan out)

---

## Phase 6 — Paper / APS
**Goal**: Clean writeup of results for APS poster and/or arXiv draft.

- [ ] Update `RESEARCH_LOG.md` with all confirmed findings
- [ ] Generate publication-quality figures (4–6 key plots)
- [ ] Write abstract and introduction
- [ ] Update poster `vol_forecasting` section (or add QR code to GitHub results)
- [ ] arXiv draft: even a short 10-page note is worth posting before APS

**Estimate**: 1 week (after Phase 3 complete)

---

## Abandoned / Deprioritized

*(Move ideas here when we decide not to pursue them, with a brief reason.)*

---

## Key Milestones

| Milestone | Target |
|-----------|--------|
| Phase 1 done (hypothesis validated or killed) | ~1 week |
| HAR-RV baseline running | ~2 weeks |
| QCML-HAR OOS results | ~3 weeks |
| Draft figures for APS poster | ~5 weeks |
| arXiv draft | ~7 weeks |
