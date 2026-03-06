"""
Smoke test for Full QGT (Quantum Geometric Tensor) Detector.

Tests all four QGT observables across 4 financial crises:
  2008_gfc, 2020_covid, 2022_rates, 2023_svb

Data: SPY, DIA 2005-2025 (yfinance, real data)
Metric: Cohen's d (crisis vs prior-60-day normal) with bootstrap CI (n=1000)
Primary signal: off-diagonal coupling (sum_{a!=b} |Q_ab|^2)

Output: smoke_results.json in the same directory.
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
from research.ideation.qgt_full.detector import QGTFullDetector

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# Configuration
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60  # Normal baseline: 60 days before each crisis


def evaluate_signal_per_crisis(scores, dates_series, test_crises):
    """Compute Cohen's d for each crisis vs prior normal window.

    Args:
        scores: 1-D array of regime scores (may contain NaN).
        dates_series: DatetimeIndex aligned with scores.
        test_crises: List of crisis keys from ALL_CRISES.

    Returns:
        Dict mapping crisis_key -> {'cohens_d', 'ci_lo', 'ci_hi', ...}
    """
    results = {}
    for crisis_key in test_crises:
        crisis = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis['start'])
        crisis_end = pd.Timestamp(crisis['end'])
        normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

        crisis_mask = (dates_series >= crisis_start) & (dates_series <= crisis_end)
        normal_mask = (dates_series >= normal_start) & (dates_series < crisis_start)

        crisis_scores = scores[np.asarray(crisis_mask)]
        normal_scores = scores[np.asarray(normal_mask)]

        crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
        normal_valid = normal_scores[~np.isnan(normal_scores)]

        logger.info(
            f"  {crisis_key}: crisis_n={len(crisis_valid)}, normal_n={len(normal_valid)}"
        )

        if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_valid, normal_valid,
                n_bootstrap=N_BOOTSTRAP, seed=42,
            )
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan

        results[crisis_key] = {
            'cohens_d': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'crisis_n': int(len(crisis_valid)),
            'normal_n': int(len(normal_valid)),
            'crisis_mean': float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
            'normal_mean': float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
        }
        d_str = f"{d:.3f}" if not np.isnan(d) else "nan"
        logger.info(f"    d={d_str}")

    return results


def main():
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Fetch data
    # -------------------------------------------------------------------------
    logger.info(f"Fetching data for {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)

    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices shape: {close_prices.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Create feature matrix
    # -------------------------------------------------------------------------
    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    dates_series = pd.DatetimeIndex(dates)
    logger.info(f"Feature matrix shape: {features.shape}, {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 3: Initialize and fit detector
    # -------------------------------------------------------------------------
    logger.info("Initializing QGT Full Detector...")
    detector = QGTFullDetector(
        hilbert_dim=4,
        n_pca_components=8,
        operator_method='random',
        normalization='soft',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        adaptive_epsilon=True,
    )

    logger.info("Fitting detector on full feature matrix...")
    detector.fit(features)

    # -------------------------------------------------------------------------
    # Step 4: Compute primary signal (off-diagonal coupling) + all observables
    # -------------------------------------------------------------------------
    logger.info("Computing primary regime scores (off-diagonal coupling)...")
    t_scores = time.time()
    primary_scores = detector.compute_regime_scores(features)
    dt_primary = time.time() - t_scores

    n_valid = np.sum(~np.isnan(primary_scores))
    logger.info(f"Primary scores: {n_valid}/{len(primary_scores)} valid, took {dt_primary:.1f}s")

    logger.info("Computing all QGT observables...")
    t_all = time.time()
    all_obs = detector.compute_all_observables(features)
    dt_all = time.time() - t_all
    logger.info(f"All observables computed in {dt_all:.1f}s")

    # -------------------------------------------------------------------------
    # Step 5: Evaluate per-crisis Cohen's d for primary signal
    # -------------------------------------------------------------------------
    logger.info("\nEvaluating PRIMARY signal (off-diagonal coupling) per crisis:")
    primary_results = evaluate_signal_per_crisis(primary_scores, dates_series, TEST_CRISES)

    # -------------------------------------------------------------------------
    # Step 6: Evaluate all four observables
    # -------------------------------------------------------------------------
    all_obs_results = {}
    for obs_name, obs_scores in all_obs.items():
        logger.info(f"\nEvaluating {obs_name}:")
        all_obs_results[obs_name] = evaluate_signal_per_crisis(
            obs_scores, dates_series, TEST_CRISES
        )

    # -------------------------------------------------------------------------
    # Step 7: Summary statistics
    # -------------------------------------------------------------------------
    d_values_primary = [
        r['cohens_d'] for r in primary_results.values() if r['cohens_d'] is not None
    ]
    median_d = float(np.median(d_values_primary)) if d_values_primary else None
    max_d = float(np.max(d_values_primary)) if d_values_primary else None
    passes_threshold = median_d is not None and median_d > 0.3

    total_time = time.time() - t0

    # Per-observable median d
    obs_medians = {}
    for obs_name, obs_res in all_obs_results.items():
        dvals = [r['cohens_d'] for r in obs_res.values() if r['cohens_d'] is not None]
        obs_medians[obs_name] = float(np.median(dvals)) if dvals else None

    summary = {
        'detector': 'QGT Full (Off-Diagonal Coupling)',
        'config': {
            'hilbert_dim': 4,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'adaptive_epsilon': True,
            'rolling_window': 20,
            'min_expanding': 60,
            'n_bootstrap': N_BOOTSTRAP,
        },
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'n_valid_scores': int(n_valid),
        },
        'primary_signal': 'off_diagonal_coupling',
        'per_crisis': primary_results,
        'all_observables': all_obs_results,
        'summary': {
            'cohens_d_per_crisis': {k: r['cohens_d'] for k, r in primary_results.items()},
            'median_d': median_d,
            'max_d': max_d,
            'passes_threshold': passes_threshold,
            'observable_medians': obs_medians,
            'best_observable': max(obs_medians, key=lambda k: obs_medians[k] or -np.inf),
        },
        'timing': {
            'total_seconds': round(total_time, 1),
            'primary_scores_seconds': round(dt_primary, 1),
            'all_observables_seconds': round(dt_all, 1),
        },
    }

    # -------------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------------
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("QGT FULL DETECTOR — SMOKE TEST RESULTS")
    print("=" * 70)
    print("\nPrimary signal: off-diagonal coupling (sum_{a!=b} |Q_ab|^2)")
    print()
    for crisis_key, r in primary_results.items():
        d = r['cohens_d']
        if d is not None and r['ci_lo'] is not None:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"  {crisis_key:20s}  d={d:.3f}  CI={ci}"
                  f"  (n_crisis={r['crisis_n']}, n_normal={r['normal_n']})")
        else:
            print(f"  {crisis_key:20s}  d=nan  (insufficient data)")

    print(f"\nPrimary signal summary:")
    print(f"  Median d: {median_d:.3f}" if median_d is not None else "  Median d: None")
    print(f"  Max d:    {max_d:.3f}" if max_d is not None else "  Max d: None")
    print(f"  Passes threshold (d>0.3): {passes_threshold}")

    print("\nAll observables — median Cohen's d:")
    for obs_name, med in obs_medians.items():
        med_str = f"{med:.3f}" if med is not None else "None"
        marker = " <-- primary" if obs_name == 'off_diagonal_coupling' else ""
        print(f"  {obs_name:35s} median_d={med_str}{marker}")

    best = summary['summary']['best_observable']
    print(f"\nBest observable: {best} (median_d={obs_medians.get(best, None):.3f})")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    main()
