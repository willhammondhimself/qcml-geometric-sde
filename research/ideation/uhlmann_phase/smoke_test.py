"""
Smoke test for Uhlmann Phase Detector on 4 financial crises.

Tests 4 variants:
  1. Uhlmann phase rate (abs) for (2,4) bipartition
  2. Uhlmann phase rate (abs) for (4,2) bipartition
  3. Uhlmann fidelity (1-F) for (2,4) bipartition (infidelity)
  4. Pure-state Berry phase rate (abs) for comparison

Also computes correlation with existing Berry Phase Rate and Reduced Purity
for redundancy check.

Data: SPY, DIA from 2005-01-01 to 2024-12-31
Metric: Cohen's d (crisis vs normal) with bootstrap CI (n=2000)
Crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
"""

import json
import sys
import time
import os
import warnings
import logging

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.uhlmann_phase.detector import (
    compute_uhlmann_phase_series,
    compute_pure_berry_phase_series,
    compute_purity_series,
)
from qcml_geometry.core import QCMLGeometry
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# Configuration
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2024-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 2000
CONTEXT_DAYS = 60
HILBERT_DIM = 8
N_PCA = 8
OPERATOR_METHOD = 'random'
ROLLING_WINDOW = 20
MIN_EXPANDING = 60


def build_states(features: np.ndarray) -> np.ndarray:
    """Build QCML pure state sequence from feature matrix.

    Args:
        features: Feature matrix (T, d).

    Returns:
        states: Array of normalized state vectors, shape (T, HILBERT_DIM).
    """
    np.random.seed(42)

    n_components = min(N_PCA, features.shape[1])

    scaler = StandardScaler()
    scaler.fit(features)
    X_scaled = scaler.transform(features)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # Soft normalization (consistent with existing detectors)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    median_norm = np.median(norms)
    X_norm = X_pca / (norms + median_norm)

    geom = QCMLGeometry(n_features=X_norm.shape[1], hilbert_dim=HILBERT_DIM)
    geom.fit_operators(X_norm, method=OPERATOR_METHOD)

    T = len(X_norm)
    states = np.empty((T, HILBERT_DIM), dtype=np.complex128)
    for t in range(T):
        states[t] = geom.quasi_coherent_state(X_norm[t])

    return states


def score_to_zscore(raw: np.ndarray, rolling_window: int, min_expanding: int) -> np.ndarray:
    """Convert raw time series to rolling-smoothed z-score.

    Args:
        raw: Raw values, shape (T,).
        rolling_window: Smoothing window.
        min_expanding: Minimum samples for z-score.

    Returns:
        z_scores: Z-scored values with NaN padding, shape (T,).
    """
    smoothed = pd.Series(raw).rolling(window=rolling_window, min_periods=1).mean().values
    n = len(smoothed)
    z = np.full(n, np.nan)
    for t in range(min_expanding, n):
        mu = np.mean(smoothed[:t])
        sigma = np.std(smoothed[:t], ddof=1)
        if sigma > 1e-12:
            z[t] = (smoothed[t] - mu) / sigma
        else:
            z[t] = 0.0
    return z


def evaluate_crisis(
    scores: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_key: str,
) -> dict:
    """Compute Cohen's d for a single crisis.

    Args:
        scores: 1-D regime score array.
        dates: DatetimeIndex aligned with scores.
        crisis_key: Key into ALL_CRISES.

    Returns:
        dict with cohens_d, ci_lo, ci_hi, crisis_n, normal_n.
    """
    crisis = ALL_CRISES[crisis_key]
    crisis_start = pd.Timestamp(crisis['start'])
    crisis_end = pd.Timestamp(crisis['end'])
    normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

    crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
    normal_mask = (dates >= normal_start) & (dates < crisis_start)

    crisis_scores = scores[np.asarray(crisis_mask)]
    normal_scores = scores[np.asarray(normal_mask)]

    crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
    normal_valid = normal_scores[~np.isnan(normal_scores)]

    if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_valid, normal_valid,
            n_bootstrap=N_BOOTSTRAP, seed=42,
        )
    else:
        d, ci_lo, ci_hi = np.nan, np.nan, np.nan

    return {
        'cohens_d': round(float(d), 3) if not np.isnan(d) else None,
        'ci_lo': round(float(ci_lo), 3) if not np.isnan(ci_lo) else None,
        'ci_hi': round(float(ci_hi), 3) if not np.isnan(ci_hi) else None,
        'crisis_n': int(len(crisis_valid)),
        'normal_n': int(len(normal_valid)),
    }


def main():
    t0 = time.time()

    # =========================================================================
    # Step 1: Fetch data and build features + states
    # =========================================================================
    logger.info(f"Fetching data for {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices shape: {close_prices.shape}")

    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    dates = pd.DatetimeIndex(dates)
    logger.info(f"Feature matrix shape: {features.shape}, dates: {dates[0]} to {dates[-1]}")

    logger.info("Building QCML states (d=8)...")
    t_states = time.time()
    states = build_states(features)
    dt_states = time.time() - t_states
    logger.info(f"Built {len(states)} states in {dt_states:.1f}s")

    # =========================================================================
    # Step 2: Compute all variants
    # =========================================================================
    logger.info("Computing Uhlmann phase series for (2,4) bipartition...")
    t_uhl = time.time()
    uhl_phases_24, uhl_fid_24 = compute_uhlmann_phase_series(states, dim_A=2, dim_B=4)
    dt_uhl_24 = time.time() - t_uhl
    logger.info(f"  (2,4) done in {dt_uhl_24:.1f}s")

    logger.info("Computing Uhlmann phase series for (4,2) bipartition...")
    t_uhl = time.time()
    uhl_phases_42, uhl_fid_42 = compute_uhlmann_phase_series(states, dim_A=4, dim_B=2)
    dt_uhl_42 = time.time() - t_uhl
    logger.info(f"  (4,2) done in {dt_uhl_42:.1f}s")

    logger.info("Computing pure-state Berry phase series...")
    berry_phases = compute_pure_berry_phase_series(states)

    logger.info("Computing purity series for redundancy check...")
    purities_24 = compute_purity_series(states, dim_A=2, dim_B=4)
    purities_42 = compute_purity_series(states, dim_A=4, dim_B=2)

    # Build scores: abs(phase rate) and infidelity (1-F), then z-score
    # Dates for phase-rate series are offset by 1 (consecutive differences)
    dates_diff = dates[1:]  # T-1 dates

    variants = {
        'uhlmann_phase_24': np.abs(uhl_phases_24),
        'uhlmann_phase_42': np.abs(uhl_phases_42),
        'uhlmann_infidelity_24': 1.0 - uhl_fid_24,
        'pure_berry_phase': np.abs(berry_phases),
    }

    # Z-score all variants
    zscored = {}
    for name, raw in variants.items():
        zscored[name] = score_to_zscore(raw, ROLLING_WINDOW, MIN_EXPANDING)

    # Also z-score purity for correlation (use full-length dates)
    purity_rate_24 = np.abs(np.diff(purities_24))
    purity_rate_42 = np.abs(np.diff(purities_42))
    purity_zscore_24 = score_to_zscore(purity_rate_24, ROLLING_WINDOW, MIN_EXPANDING)
    purity_zscore_42 = score_to_zscore(purity_rate_42, ROLLING_WINDOW, MIN_EXPANDING)

    # =========================================================================
    # Step 3: Evaluate per crisis
    # =========================================================================
    all_results = {}
    for variant_name, scores in zscored.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {variant_name}")
        logger.info(f"{'='*60}")

        per_crisis = {}
        for crisis_key in TEST_CRISES:
            result = evaluate_crisis(scores, dates_diff, crisis_key)
            per_crisis[crisis_key] = result
            d = result['cohens_d']
            ci = (f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]"
                  if result['ci_lo'] is not None else "N/A")
            logger.info(f"  {crisis_key:20s}  d={d}  CI={ci}")

        d_values = [r['cohens_d'] for r in per_crisis.values() if r['cohens_d'] is not None]
        median_d = round(float(np.median(d_values)), 3) if d_values else None
        max_d = round(float(np.max(d_values)), 3) if d_values else None
        n_above_08 = sum(1 for d in d_values if d > 0.8)

        all_results[variant_name] = {
            'per_crisis': per_crisis,
            'summary': {
                'median_d': median_d,
                'max_d': max_d,
                'n_above_0.8': n_above_08,
                'passes_threshold_0.3': median_d is not None and median_d > 0.3,
            },
        }

    # =========================================================================
    # Step 4: Redundancy checks (Spearman correlations)
    # =========================================================================
    logger.info("\nComputing redundancy correlations...")
    from scipy.stats import spearmanr

    correlations = {}

    # Uhlmann phase vs Berry phase
    for uhl_name in ['uhlmann_phase_24', 'uhlmann_phase_42']:
        mask = ~np.isnan(zscored[uhl_name]) & ~np.isnan(zscored['pure_berry_phase'])
        if np.sum(mask) >= 30:
            rho, _ = spearmanr(zscored[uhl_name][mask], zscored['pure_berry_phase'][mask])
            correlations[f'{uhl_name}_vs_berry'] = round(float(rho), 3)
            logger.info(f"  {uhl_name} vs berry: rho={rho:.3f}")

    # Uhlmann phase vs purity rate
    for uhl_name, pur_z in [('uhlmann_phase_24', purity_zscore_24),
                              ('uhlmann_phase_42', purity_zscore_42)]:
        mask = ~np.isnan(zscored[uhl_name]) & ~np.isnan(pur_z)
        if np.sum(mask) >= 30:
            rho, _ = spearmanr(zscored[uhl_name][mask], pur_z[mask])
            correlations[f'{uhl_name}_vs_purity_rate'] = round(float(rho), 3)
            logger.info(f"  {uhl_name} vs purity_rate: rho={rho:.3f}")

    # Uhlmann infidelity vs purity rate
    mask = ~np.isnan(zscored['uhlmann_infidelity_24']) & ~np.isnan(purity_zscore_24)
    if np.sum(mask) >= 30:
        rho, _ = spearmanr(zscored['uhlmann_infidelity_24'][mask], purity_zscore_24[mask])
        correlations['infidelity_24_vs_purity_rate_24'] = round(float(rho), 3)
        logger.info(f"  infidelity_24 vs purity_rate_24: rho={rho:.3f}")

    # Cross-partition correlation
    mask = ~np.isnan(zscored['uhlmann_phase_24']) & ~np.isnan(zscored['uhlmann_phase_42'])
    if np.sum(mask) >= 30:
        rho, _ = spearmanr(zscored['uhlmann_phase_24'][mask], zscored['uhlmann_phase_42'][mask])
        correlations['phase_24_vs_42'] = round(float(rho), 3)
        logger.info(f"  phase_24 vs phase_42: rho={rho:.3f}")

    # =========================================================================
    # Step 5: Diagnostic statistics
    # =========================================================================
    diagnostics = {}
    for name, scores in zscored.items():
        valid = scores[~np.isnan(scores)]
        diagnostics[name] = {
            'n_valid': int(len(valid)),
            'mean': round(float(np.mean(valid)), 4) if len(valid) > 0 else None,
            'std': round(float(np.std(valid)), 4) if len(valid) > 0 else None,
            'min': round(float(np.min(valid)), 4) if len(valid) > 0 else None,
            'max': round(float(np.max(valid)), 4) if len(valid) > 0 else None,
        }

    # Purity statistics (are states actually mixed?)
    purity_stats = {
        'partition_24': {
            'mean_purity': round(float(np.mean(purities_24)), 4),
            'std_purity': round(float(np.std(purities_24)), 4),
            'min_purity': round(float(np.min(purities_24)), 4),
            'max_purity': round(float(np.max(purities_24)), 4),
        },
        'partition_42': {
            'mean_purity': round(float(np.mean(purities_42)), 4),
            'std_purity': round(float(np.std(purities_42)), 4),
            'min_purity': round(float(np.min(purities_42)), 4),
            'max_purity': round(float(np.max(purities_42)), 4),
        },
    }
    logger.info(f"\nPurity stats (2,4): mean={purity_stats['partition_24']['mean_purity']:.4f}, "
                f"range=[{purity_stats['partition_24']['min_purity']:.4f}, "
                f"{purity_stats['partition_24']['max_purity']:.4f}]")
    logger.info(f"Purity stats (4,2): mean={purity_stats['partition_42']['mean_purity']:.4f}, "
                f"range=[{purity_stats['partition_42']['min_purity']:.4f}, "
                f"{purity_stats['partition_42']['max_purity']:.4f}]")

    # =========================================================================
    # Step 6: Assemble and save results
    # =========================================================================
    total_time = time.time() - t0

    summary = {
        'detector': 'Uhlmann Phase',
        'question': 'Q0128: Can the Uhlmann phase (mixed-state Berry phase) improve detection for noisy/mixed market states?',
        'config': {
            'hilbert_dim': HILBERT_DIM,
            'n_pca_components': N_PCA,
            'operator_method': OPERATOR_METHOD,
            'normalization': 'soft',
            'rolling_window': ROLLING_WINDOW,
            'n_bootstrap': N_BOOTSTRAP,
            'context_days': CONTEXT_DAYS,
        },
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'n_states': len(states),
        },
        'variants': all_results,
        'correlations': correlations,
        'purity_stats': purity_stats,
        'diagnostics': diagnostics,
        'timing': {
            'state_build_seconds': round(dt_states, 1),
            'uhlmann_24_seconds': round(dt_uhl_24, 1),
            'uhlmann_42_seconds': round(dt_uhl_42, 1),
            'total_seconds': round(total_time, 1),
        },
    }

    # Determine best variant
    best_name = None
    best_median = -1.0
    for vname, vdata in all_results.items():
        md = vdata['summary']['median_d']
        if md is not None and md > best_median:
            best_median = md
            best_name = vname
    summary['best_variant'] = best_name
    summary['best_median_d'] = best_median

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("UHLMANN PHASE DETECTOR -- SMOKE TEST RESULTS")
    print("=" * 70)

    for variant_name, vdata in all_results.items():
        print(f"\n--- {variant_name} ---")
        for crisis_key, r in vdata['per_crisis'].items():
            d = r['cohens_d']
            ci = (f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                  if r['ci_lo'] is not None else "N/A")
            print(f"  {crisis_key:20s}  d={d}  CI={ci}  "
                  f"(n_crisis={r['crisis_n']}, n_normal={r['normal_n']})")
        md = vdata['summary']['median_d']
        mx = vdata['summary']['max_d']
        na = vdata['summary']['n_above_0.8']
        print(f"  Median d: {md}  |  Max d: {mx}  |  "
              f"N above 0.8: {na}/4  |  Pass (>0.3): {vdata['summary']['passes_threshold_0.3']}")

    print(f"\n--- Purity Statistics (are states mixed?) ---")
    for part, stats in purity_stats.items():
        print(f"  {part}: mean={stats['mean_purity']:.4f}  "
              f"range=[{stats['min_purity']:.4f}, {stats['max_purity']:.4f}]")

    print(f"\n--- Redundancy Correlations ---")
    for pair, rho in correlations.items():
        print(f"  {pair}: rho={rho:.3f}")

    print(f"\n--- Overall ---")
    print(f"  Best variant: {best_name} (median d = {best_median})")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    main()
