"""
Dead Signal Resurrection Smoke Tests — Q29, Q30, Q31.

Tests whether simple fixes can rescue three low-d detectors:
  Q29: GeometricConsensusDetector (baseline d=0.075) — weighted/OR variants
  Q30: GeometricEnsembleDetector (baseline d=0.013) — focused top-performer ensemble
  Q31: BOCPDDetector (baseline d=0.057) — prior tuning + pre-whitening

Smoke test protocol:
  - 4 crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
  - SPY data, 2005-2025
  - Cohen's d per crisis (block bootstrap, n=500 for speed)
  - Verdict KEEP if median d > 0.3
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 500      # Speed over precision for smoke test
CONTEXT_DAYS = 60      # Pre-crisis "normal" window
KEEP_THRESHOLD = 0.3   # Median d threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_detector(scores, dates, crises=TEST_CRISES, n_boot=N_BOOTSTRAP):
    """Return per-crisis Cohen's d dict and median d."""
    dates_idx = pd.DatetimeIndex(dates)
    results = {}
    for key in crises:
        c = ALL_CRISES[key]
        cs = pd.Timestamp(c['start'])
        ce = pd.Timestamp(c['end'])
        ns = cs - pd.Timedelta(days=CONTEXT_DAYS)

        cmask = (dates_idx >= cs) & (dates_idx <= ce)
        nmask = (dates_idx >= ns) & (dates_idx < cs)

        cv = scores[np.asarray(cmask)]
        nv = scores[np.asarray(nmask)]
        cv = cv[~np.isnan(cv)]
        nv = nv[~np.isnan(nv)]

        if len(cv) >= 2 and len(nv) >= 2:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(cv, nv, n_bootstrap=n_boot, seed=42)
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan

        results[key] = {'d': float(d) if not np.isnan(d) else None,
                        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
                        'n_crisis': int(len(cv)), 'n_normal': int(len(nv))}
    d_vals = [r['d'] for r in results.values() if r['d'] is not None]
    median_d = float(np.median(d_vals)) if d_vals else None
    return results, median_d


def expanding_zscore(vals, rolling_window=20, min_expanding=60):
    """Expanding-window z-score (absolute) with rolling pre-smoothing."""
    T = len(vals)
    smoothed = pd.Series(vals).rolling(rolling_window, min_periods=1).mean().values
    z = np.full(T, 0.0)
    for t in range(min_expanding, T):
        past = smoothed[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma = np.std(past_valid, ddof=1)
        if sigma > 1e-12:
            z[t] = abs((smoothed[t] - mu) / sigma)
    return z


def print_results(label, results, median_d):
    verdict = 'KEEP' if (median_d is not None and median_d > KEEP_THRESHOLD) else 'REJECT'
    print(f"\n{label}")
    print(f"  {'Crisis':<20}  {'d':>6}  {'CI':>20}  n_crisis  n_normal")
    for key, r in results.items():
        d = r['d']
        ci = (f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              if r['ci_lo'] is not None else '   N/A   ')
        d_str = f"{d:.3f}" if d is not None else '  nan'
        print(f"  {key:<20}  {d_str:>6}  {ci:>20}  {r['n_crisis']:>8}  {r['n_normal']:>8}")
    md_str = f"{median_d:.3f}" if median_d is not None else 'nan'
    print(f"  Median d: {md_str}   Verdict: {verdict}")


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------

def load_data():
    logger.info(f"Fetching {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices: {close_prices.shape}")
    features, dates = create_feature_matrix(close_prices)
    logger.info(f"Feature matrix: {features.shape}")
    return features, dates, close_prices


# ===========================================================================
# Q29: Geometric Consensus — weighted / OR / adaptive variants
# ===========================================================================

def run_q29(features, dates):
    """Test three variants of GeometricConsensusDetector."""
    logger.info("Q29: Computing Berry curvature and metric determinant signals...")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from qcml_geometry.core import QCMLGeometry

    # Build shared geometry (same params as original GeometricConsensusDetector)
    HILBERT_DIM = 8
    N_PCA = 15
    ROLLING = 20
    MIN_EXP = 60

    np.random.seed(42)
    scaler = StandardScaler()
    scaler.fit(features)
    pca = PCA(n_components=min(N_PCA, features.shape[1]))
    pca.fit(scaler.transform(features))

    X_pca_raw = pca.transform(scaler.transform(features))
    norms = np.linalg.norm(X_pca_raw, axis=1, keepdims=True)
    X_pca = X_pca_raw / np.maximum(norms, 1e-12)

    geometry = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=HILBERT_DIM)
    # Use pca_inspired to match original; note: produces low-variance Berry
    # but we test whether combining changes matter regardless
    geometry.fit_operators(X_pca, method='pca_inspired')

    T = len(X_pca)
    berry = np.empty(T)
    log_det = np.empty(T)

    logger.info("  Computing raw geometric signals...")
    for t in range(T):
        F_01 = geometry.berry_curvature_2d(X_pca[t], indices=(0, 1))
        berry[t] = abs(F_01)
        g = geometry.quantum_metric(X_pca[t])
        eigs = np.linalg.eigvalsh(g)
        pos_eigs = eigs[eigs > 1e-10]
        log_det[t] = np.sum(np.log(pos_eigs)) if len(pos_eigs) > 0 else -20.0

    berry_rate = np.abs(np.diff(berry, prepend=berry[0]))

    z_berry = expanding_zscore(berry_rate, ROLLING, MIN_EXP)
    z_det = expanding_zscore(log_det, ROLLING, MIN_EXP)

    valid_mask = ~(z_berry == 0) & ~(z_det == 0)

    # --- Variant (a): Weighted average (berry has 2x weight as primary signal) ---
    logger.info("  Q29a: Weighted voting (berry 2x weight)...")
    w_berry, w_det = 2.0, 1.0
    scores_weighted = (w_berry * z_berry + w_det * z_det) / (w_berry + w_det)
    res_a, med_a = evaluate_detector(scores_weighted, dates)

    # --- Variant (b): OR logic — fire if EITHER exceeds threshold ---
    logger.info("  Q29b: OR logic (max of z-scores)...")
    scores_or = np.maximum(z_berry, z_det)
    res_b, med_b = evaluate_detector(scores_or, dates)

    # --- Variant (c): Adaptive thresholds — use 80th percentile instead of 90th ---
    # AND logic but with lower quantile threshold to be less conservative
    logger.info("  Q29c: Adaptive thresholds (80th pctile AND)...")
    berry_thresh = np.nanpercentile(z_berry[valid_mask], 80) if np.any(valid_mask) else 1.5
    det_thresh = np.nanpercentile(z_det[valid_mask], 80) if np.any(valid_mask) else 1.5
    scores_adapt = np.zeros(T)
    for t in range(T):
        if z_berry[t] > berry_thresh and z_det[t] > det_thresh:
            scores_adapt[t] = np.sqrt(z_berry[t] * z_det[t])
    res_c, med_c = evaluate_detector(scores_adapt, dates)

    return {
        'weighted': (res_a, med_a),
        'or_logic': (res_b, med_b),
        'adaptive_thresh': (res_c, med_c),
    }


# ===========================================================================
# Q30: Geometric Ensemble — top-performer focused ensemble
# ===========================================================================

def run_q30(features, dates):
    """Compare full GeometricEnsemble vs focused top-performer ensemble."""
    from qcml_geometry.observables import (
        BerryPhaseRateDetector,
        SpectralGapDetector,
        ReducedPurityDetector,
        GeometricEnsembleDetector,
    )

    # --- Variant: Full ensemble (baseline, re-run for reference) ---
    logger.info("  Q30 baseline: Full GeometricEnsembleDetector...")
    full_det = GeometricEnsembleDetector(
        hilbert_dim=8, n_pca_components=15, operator_method='random',
        rolling_window=20, min_expanding=60, seed=42,
    )
    full_det.fit(features)
    scores_full = full_det.compute_regime_scores(features)
    res_full, med_full = evaluate_detector(scores_full, dates)

    # --- Variant (a): Top-3 focused ensemble (BerryPhaseRate + SpectralGap + ReducedPurity) ---
    logger.info("  Q30a: Focused top-3 ensemble...")

    # Run each individually, z-score, combine as RMS
    logger.info("    Fitting BerryPhaseRateDetector...")
    bpr = BerryPhaseRateDetector(
        hilbert_dim=8, n_pca_components=15, operator_method='random',
        rolling_window=20, min_expanding=60, seed=42,
    )
    bpr.fit(features)
    s_bpr = bpr.compute_regime_scores(features)

    logger.info("    Fitting SpectralGapDetector...")
    sg = SpectralGapDetector(
        hilbert_dim=8, n_pca_components=15, operator_method='random',
        rolling_window=20, min_expanding=60, seed=42,
    )
    sg.fit(features)
    s_sg = sg.compute_regime_scores(features)

    logger.info("    Fitting ReducedPurityDetector...")
    rp = ReducedPurityDetector(
        hilbert_dim=8, n_pca_components=15, operator_method='random',
        rolling_window=20, min_expanding=60, seed=42,
    )
    rp.fit(features)
    s_rp = rp.compute_regime_scores(features)

    # Stack and compute RMS z-score
    T = len(s_bpr)

    def robust_zscore_series(arr):
        """Global z-score (not expanding), skipping NaN."""
        valid = arr[~np.isnan(arr)]
        if len(valid) < 10:
            return np.zeros_like(arr)
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma < 1e-12:
            return np.zeros_like(arr)
        z = (arr - mu) / sigma
        z = np.where(np.isnan(z), 0.0, np.abs(z))
        return z

    z_bpr = robust_zscore_series(s_bpr)
    z_sg = robust_zscore_series(s_sg)
    z_rp = robust_zscore_series(s_rp)

    # RMS combination
    z_stack = np.column_stack([z_bpr, z_sg, z_rp])
    scores_top3 = np.sqrt(np.mean(z_stack ** 2, axis=1))
    res_top3, med_top3 = evaluate_detector(scores_top3, dates)

    # --- Variant (b): Best single — ReducedPurity alone (d=0.834 in Paper 1) ---
    logger.info("  Q30b: ReducedPurity alone (best known single)...")
    res_rp_solo, med_rp_solo = evaluate_detector(s_rp, dates)

    return {
        'full_ensemble': (res_full, med_full),
        'top3_focused': (res_top3, med_top3),
        'reduced_purity_solo': (res_rp_solo, med_rp_solo),
    }


# ===========================================================================
# Q31: BOCPD — prior tuning + pre-whitening variants
# ===========================================================================

def run_q31(features, dates, close_prices):
    """Test BOCPD variants: shorter hazard, wider variance, GARCH pre-whitening."""
    from experiments.baselines import BOCPDDetector

    # Use log returns of SPY as the univariate input (standard for BOCPD)
    spy_close = close_prices['SPY']
    spy_returns = np.log(spy_close / spy_close.shift(1)).dropna()
    spy_dates = spy_returns.index
    spy_arr = spy_returns.values.reshape(-1, 1)

    # --- Variant (a): Shorter hazard_lambda (faster changepoint prior) ---
    # Original: hazard_rate=250 (prior mean run length = 250 days ~ 1 year)
    # New: hazard_rate=63 (prior mean run length = 63 days ~ 1 quarter)
    logger.info("  Q31a: Shorter hazard (hazard_rate=63, ~1 quarter)...")
    bocpd_fast = BOCPDDetector(hazard_rate=63, min_expanding=30, max_run_length=500)
    bocpd_fast.fit(spy_arr)
    scores_fast = bocpd_fast.compute_regime_scores(spy_arr)
    res_fast, med_fast = evaluate_detector(scores_fast, spy_dates)

    # --- Variant (b): Very short hazard (hazard_rate=21, ~1 month) ---
    logger.info("  Q31b: Very short hazard (hazard_rate=21, ~1 month)...")
    bocpd_vfast = BOCPDDetector(hazard_rate=21, min_expanding=30, max_run_length=500)
    bocpd_vfast.fit(spy_arr)
    scores_vfast = bocpd_vfast.compute_regime_scores(spy_arr)
    res_vfast, med_vfast = evaluate_detector(scores_vfast, spy_dates)

    # --- Variant (c): GARCH pre-whitening + original hazard ---
    # Standardize returns by a rolling EWMA volatility estimate (poor-man's GARCH)
    # This removes heteroskedasticity so BOCPD's Gaussian assumption is better met
    logger.info("  Q31c: GARCH pre-whitening (EWMA vol, hazard_rate=63)...")
    ret_series = pd.Series(spy_arr.ravel(), index=spy_dates)
    ewma_vol = ret_series.ewm(span=21).std()
    whitened_series = (ret_series / ewma_vol.clip(lower=1e-8)).dropna()
    whitened_dates = whitened_series.index
    whitened = whitened_series.values.reshape(-1, 1)

    bocpd_white = BOCPDDetector(hazard_rate=63, min_expanding=30, max_run_length=500)
    bocpd_white.fit(whitened)
    scores_white = bocpd_white.compute_regime_scores(whitened)
    res_white, med_white = evaluate_detector(scores_white, whitened_dates)

    # --- Variant (d): GARCH pre-whitening + very short hazard ---
    logger.info("  Q31d: GARCH pre-whitening + very short hazard (hazard_rate=21)...")
    bocpd_white_vfast = BOCPDDetector(hazard_rate=21, min_expanding=30, max_run_length=500)
    bocpd_white_vfast.fit(whitened)
    scores_white_vfast = bocpd_white_vfast.compute_regime_scores(whitened)
    res_white_vfast, med_white_vfast = evaluate_detector(scores_white_vfast, whitened_dates)

    return {
        'hazard_63': (res_fast, med_fast),
        'hazard_21': (res_vfast, med_vfast),
        'whitened_hazard_63': (res_white, med_white),
        'whitened_hazard_21': (res_white_vfast, med_white_vfast),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()

    features, dates, close_prices = load_data()

    all_results = {}

    # ------------------------------------------------------------------
    # Q29
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q29: GEOMETRIC CONSENSUS — Dead signal resurrection test")
    print("=" * 70)
    t_q29 = time.time()
    q29 = run_q29(features, dates)
    logger.info(f"Q29 done in {time.time() - t_q29:.1f}s")

    for variant, (res, med) in q29.items():
        print_results(f"  Variant: {variant}", res, med)

    # Pick best Q29 variant
    best_q29_key = max(q29, key=lambda k: q29[k][1] if q29[k][1] is not None else -999)
    best_q29_res, best_q29_med = q29[best_q29_key]
    all_results['Q29'] = {
        'variants': {k: {'per_crisis': v[0], 'median_d': v[1]} for k, v in q29.items()},
        'best_variant': best_q29_key,
        'best_median_d': best_q29_med,
        'verdict': 'KEEP' if (best_q29_med is not None and best_q29_med > KEEP_THRESHOLD) else 'REJECT',
    }

    # ------------------------------------------------------------------
    # Q30
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q30: GEOMETRIC ENSEMBLE — Focused top-performer ensemble test")
    print("=" * 70)
    t_q30 = time.time()
    q30 = run_q30(features, dates)
    logger.info(f"Q30 done in {time.time() - t_q30:.1f}s")

    for variant, (res, med) in q30.items():
        print_results(f"  Variant: {variant}", res, med)

    best_q30_key = max(q30, key=lambda k: q30[k][1] if q30[k][1] is not None else -999)
    best_q30_res, best_q30_med = q30[best_q30_key]
    all_results['Q30'] = {
        'variants': {k: {'per_crisis': v[0], 'median_d': v[1]} for k, v in q30.items()},
        'best_variant': best_q30_key,
        'best_median_d': best_q30_med,
        'verdict': 'KEEP' if (best_q30_med is not None and best_q30_med > KEEP_THRESHOLD) else 'REJECT',
    }

    # ------------------------------------------------------------------
    # Q31
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q31: BOCPD — Informative priors + pre-whitening test")
    print("=" * 70)
    t_q31 = time.time()
    q31 = run_q31(features, dates, close_prices)
    logger.info(f"Q31 done in {time.time() - t_q31:.1f}s")

    for variant, (res, med) in q31.items():
        print_results(f"  Variant: {variant}", res, med)

    best_q31_key = max(q31, key=lambda k: q31[k][1] if q31[k][1] is not None else -999)
    best_q31_res, best_q31_med = q31[best_q31_key]
    all_results['Q31'] = {
        'variants': {k: {'per_crisis': v[0], 'median_d': v[1]} for k, v in q31.items()},
        'best_variant': best_q31_key,
        'best_median_d': best_q31_med,
        'verdict': 'KEEP' if (best_q31_med is not None and best_q31_med > KEEP_THRESHOLD) else 'REJECT',
    }

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for q_label, qr in all_results.items():
        med = qr['best_median_d']
        med_str = f"{med:.3f}" if med is not None else "nan"
        print(f"  {q_label}: best_variant={qr['best_variant']}, "
              f"median_d={med_str}, verdict={qr['verdict']}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 70)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return all_results


if __name__ == '__main__':
    main()
