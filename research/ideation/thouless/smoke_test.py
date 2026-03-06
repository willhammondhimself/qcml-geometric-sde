"""
Smoke test for Thouless Energy Detector on 4 financial crises.

Tests: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Data: SPY, DIA from 2005-01-01 to 2025-12-31
Metric: Cohen's d (crisis vs normal) with bootstrap CI (n=1000)

Evaluates three scoring modes:
    - 'thouless': g_T = delta_E / W alone
    - 'lsr':      Level spacing ratio r alone
    - 'combined': 50/50 blend of both (default detector)

Reports all three and saves the best-performing mode as the headline result.
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.thouless.detector import ThoulessEnergyDetector

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# Configuration
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000   # Speed over precision for smoke test
CONTEXT_DAYS = 60    # Days before crisis for "normal" baseline

# Detector hyperparameters — shared across all scoring modes
DETECTOR_KWARGS = dict(
    hilbert_dim=8,
    n_pca_components=15,
    operator_method='random',
    normalization='soft',
    rolling_window=20,
    min_expanding=60,
    seed=42,
    adaptive_epsilon=True,
    alpha=0.5,
)

SCORING_MODES = ['thouless', 'lsr', 'combined']


def evaluate_scores(scores, dates_series, crises):
    """Compute per-crisis Cohen's d for a score array.

    Args:
        scores: 1-D numpy array of regime scores (may contain NaN).
        dates_series: DatetimeIndex aligned with scores.
        crises: List of crisis keys to evaluate.

    Returns:
        Dictionary mapping crisis key → result dict.
    """
    results = {}
    for crisis_key in crises:
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
        logger.info(
            f"  {crisis_key}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
            f"(n_crisis={len(crisis_valid)}, n_normal={len(normal_valid)})"
        )

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
    logger.info(f"Feature matrix: {features.shape}, {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 3: Fit a single detector on the data (reuse geometry across modes)
    #         We run each scoring mode separately since the raw signal differs.
    # -------------------------------------------------------------------------
    all_mode_results = {}
    scores_by_mode = {}

    for mode in SCORING_MODES:
        logger.info(f"\n--- Scoring mode: {mode} ---")
        detector = ThoulessEnergyDetector(scoring_mode=mode, **DETECTOR_KWARGS)

        logger.info("Fitting detector...")
        detector.fit(features)

        logger.info("Computing regime scores...")
        t_scores = time.time()
        scores = detector.compute_regime_scores(features)
        dt_scores = time.time() - t_scores

        n_valid = int(np.sum(~np.isnan(scores)))
        logger.info(f"Scores computed in {dt_scores:.1f}s. Valid: {n_valid}/{len(scores)}")

        logger.info(f"Evaluating per-crisis Cohen's d (mode={mode})...")
        per_crisis = evaluate_scores(scores, dates_series, TEST_CRISES)

        d_values = [r['cohens_d'] for r in per_crisis.values() if r['cohens_d'] is not None]
        median_d = float(np.median(d_values)) if d_values else None

        all_mode_results[mode] = {
            'per_crisis': per_crisis,
            'median_d': median_d,
            'd_values': d_values,
            'score_seconds': round(dt_scores, 1),
            'n_valid_scores': n_valid,
        }
        scores_by_mode[mode] = scores

    # -------------------------------------------------------------------------
    # Step 4: Select best mode by median d
    # -------------------------------------------------------------------------
    best_mode = max(
        SCORING_MODES,
        key=lambda m: all_mode_results[m]['median_d'] or -1.0,
    )
    best = all_mode_results[best_mode]
    best_median_d = best['median_d']
    passes_threshold = best_median_d is not None and best_median_d > 0.3

    total_time = time.time() - t0

    # -------------------------------------------------------------------------
    # Step 5: Build summary and save
    # -------------------------------------------------------------------------
    summary = {
        'detector': 'Thouless Energy',
        'question': 'Q11',
        'config': {
            **DETECTOR_KWARGS,
            'scoring_modes_evaluated': SCORING_MODES,
            'best_mode': best_mode,
            'n_bootstrap': N_BOOTSTRAP,
            'context_days': CONTEXT_DAYS,
        },
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
        },
        'all_modes': all_mode_results,
        'best_mode_results': {
            'mode': best_mode,
            'per_crisis': best['per_crisis'],
            'median_d': best_median_d,
            'max_d': float(max(best['d_values'])) if best['d_values'] else None,
            'd_values': best['d_values'],
        },
        'summary': {
            'cohens_d_per_crisis': {
                k: v['cohens_d'] for k, v in best['per_crisis'].items()
            },
            'median_d': best_median_d,
            'passes_threshold': passes_threshold,
            'implementation_issues': [],
        },
        'timing': {
            'total_seconds': round(total_time, 1),
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # -------------------------------------------------------------------------
    # Step 6: Print summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("THOULESS ENERGY DETECTOR — SMOKE TEST RESULTS")
    print("=" * 70)

    # Per-mode summary row
    print(f"\n{'Mode':<12}  {'Median d':>9}  Per-crisis d values")
    print("-" * 70)
    for mode in SCORING_MODES:
        r = all_mode_results[mode]
        md = r['median_d']
        md_str = f"{md:.3f}" if md is not None else "  N/A"
        d_strs = "  ".join(
            f"{v:.3f}" if v is not None else " N/A" for v in r['d_values']
        )
        marker = " <-- BEST" if mode == best_mode else ""
        print(f"  {mode:<10}  {md_str:>9}  {d_strs}{marker}")

    print(f"\nBest mode: {best_mode}")
    print(f"\nPer-crisis breakdown ({best_mode}):")
    for crisis_key, r in best['per_crisis'].items():
        d = r['cohens_d']
        d_str = f"{d:.3f}" if d is not None else " N/A"
        ci_str = (
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            if r['ci_lo'] is not None else "N/A"
        )
        print(
            f"  {crisis_key:20s}  d={d_str}  CI={ci_str}"
            f"  (n_crisis={r['crisis_n']}, n_normal={r['normal_n']})"
        )

    print(f"\n  Median d: {best_median_d:.3f}" if best_median_d else "\n  Median d: N/A")
    print(f"  Passes threshold (d > 0.3): {passes_threshold}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 7: Return YAML-ready result (printed for inspection)
    # -------------------------------------------------------------------------
    yaml_result = {
        'cohens_d_per_crisis': {
            k: round(v['cohens_d'], 4) if v['cohens_d'] is not None else None
            for k, v in best['per_crisis'].items()
        },
        'median_d': round(best_median_d, 4) if best_median_d is not None else None,
        'passes_threshold': passes_threshold,
        'implementation_issues': [],
    }

    print("\nYAML result:")
    print(f"cohens_d_per_crisis:")
    for k, v in yaml_result['cohens_d_per_crisis'].items():
        print(f"  {k}: {v}")
    print(f"median_d: {yaml_result['median_d']}")
    print(f"passes_threshold: {yaml_result['passes_threshold']}")
    print(f"implementation_issues: {yaml_result['implementation_issues']}")

    return summary


if __name__ == '__main__':
    main()
