"""
Smoke test for Husimi Q-function Detector on 4 financial crises.

Tests: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Data:  SPY, DIA from 2005-01-01 to 2025-12-31
Metric: Cohen's d (crisis vs 60-day pre-crisis normal) with bootstrap CI (n=1000)

Saves results to research/ideation/husimi/smoke_results.json
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.husimi.detector import HusimiQDetector

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60   # pre-crisis "normal" window in calendar days

DETECTOR_CONFIG = dict(
    hilbert_dim=4,
    n_pca_components=8,
    operator_method='random',
    normalization='soft',
    rolling_window=20,
    min_expanding=60,
    seed=42,
    n_probes=512,
    signal='wehrl',
)


def main() -> dict:
    t0 = time.time()

    # -------------------------------------------------------------------------
    # 1. Fetch data
    # -------------------------------------------------------------------------
    logger.info(f"Fetching data for {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)

    # raw_df has MultiIndex (symbol, date); close prices per symbol
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices shape: {close_prices.shape}")

    # -------------------------------------------------------------------------
    # 2. Feature matrix
    # -------------------------------------------------------------------------
    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    logger.info(
        f"Feature matrix shape: {features.shape}, "
        f"dates: {dates[0]} to {dates[-1]}"
    )

    # -------------------------------------------------------------------------
    # 3. Fit detector
    # -------------------------------------------------------------------------
    logger.info(f"Initializing HusimiQDetector with config: {DETECTOR_CONFIG}")
    detector = HusimiQDetector(**DETECTOR_CONFIG)

    logger.info("Fitting detector on full feature matrix...")
    detector.fit(features)

    # -------------------------------------------------------------------------
    # 4. Compute regime scores
    # -------------------------------------------------------------------------
    logger.info("Computing regime scores...")
    t_scores = time.time()
    scores = detector.compute_regime_scores(features)
    dt_scores = time.time() - t_scores
    logger.info(f"Scores computed in {dt_scores:.1f}s")

    n_valid = int(np.sum(~np.isnan(scores)))
    logger.info(f"Valid scores: {n_valid}/{len(scores)}")

    # -------------------------------------------------------------------------
    # 5. Per-crisis Cohen's d
    # -------------------------------------------------------------------------
    dates_idx = pd.DatetimeIndex(dates)
    results = {}
    implementation_issues = []

    for crisis_key in TEST_CRISES:
        crisis = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis['start'])
        crisis_end = pd.Timestamp(crisis['end'])
        normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

        crisis_mask = (dates_idx >= crisis_start) & (dates_idx <= crisis_end)
        normal_mask = (dates_idx >= normal_start) & (dates_idx < crisis_start)

        crisis_scores = scores[np.asarray(crisis_mask)]
        normal_scores = scores[np.asarray(normal_mask)]

        crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
        normal_valid = normal_scores[~np.isnan(normal_scores)]

        logger.info(
            f"{crisis_key}: crisis_n={len(crisis_valid)}, "
            f"normal_n={len(normal_valid)}"
        )

        if len(crisis_valid) < 2:
            implementation_issues.append(
                f"{crisis_key}: too few crisis scores ({len(crisis_valid)})"
            )
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan
        elif len(normal_valid) < 2:
            implementation_issues.append(
                f"{crisis_key}: too few normal scores ({len(normal_valid)})"
            )
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan
        else:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_valid, normal_valid,
                n_bootstrap=N_BOOTSTRAP, seed=42,
            )

        results[crisis_key] = {
            'cohens_d': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'crisis_n': int(len(crisis_valid)),
            'normal_n': int(len(normal_valid)),
            'crisis_mean': float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
            'normal_mean': float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
        }

        d_str = f"{d:.3f}" if not np.isnan(d) else "NaN"
        ci_str = (
            f"[{ci_lo:.3f}, {ci_hi:.3f}]"
            if not (np.isnan(ci_lo) or np.isnan(ci_hi))
            else "N/A"
        )
        logger.info(f"  d={d_str}  CI={ci_str}")

    # -------------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------------
    d_values = [
        r['cohens_d']
        for r in results.values()
        if r['cohens_d'] is not None
    ]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None
    passes_threshold = median_d is not None and median_d > 0.3

    total_time = time.time() - t0

    # Check for any negative d values (wrong direction)
    if any(v < 0 for v in d_values):
        implementation_issues.append(
            "Some crises show negative Cohen's d "
            "(crisis scores lower than normal — direction inversion)."
        )

    summary = {
        'detector': 'Husimi Q-function',
        'config': DETECTOR_CONFIG,
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'n_valid_scores': n_valid,
        },
        'cohens_d_per_crisis': {k: v['cohens_d'] for k, v in results.items()},
        'per_crisis': results,
        'summary': {
            'median_d': median_d,
            'max_d': max_d,
            'passes_threshold': passes_threshold,
            'd_values': d_values,
        },
        'implementation_issues': implementation_issues,
        'timing': {
            'total_seconds': round(total_time, 1),
            'score_computation_seconds': round(dt_scores, 1),
        },
    }

    # -------------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------------
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # -------------------------------------------------------------------------
    # 8. Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("HUSIMI Q-FUNCTION DETECTOR — SMOKE TEST RESULTS")
    print("=" * 65)
    for crisis_key, r in results.items():
        d = r['cohens_d']
        ci_lo, ci_hi = r['ci_lo'], r['ci_hi']
        d_str = f"{d:.3f}" if d is not None else "NaN"
        ci_str = (
            f"[{ci_lo:.3f}, {ci_hi:.3f}]"
            if ci_lo is not None and ci_hi is not None
            else "N/A"
        )
        print(
            f"  {crisis_key:20s}  d={d_str}  CI={ci_str}"
            f"  (n_crisis={r['crisis_n']}, n_normal={r['normal_n']})"
        )
    print(f"\n  Median d:              {median_d:.3f}" if median_d is not None else "\n  Median d: NaN")
    print(f"  Max d:                 {max_d:.3f}" if max_d is not None else "  Max d:    NaN")
    print(f"  Passes threshold (>0.3): {passes_threshold}")
    print(f"  Total time: {total_time:.1f}s")

    if implementation_issues:
        print("\n  Implementation issues:")
        for issue in implementation_issues:
            print(f"    - {issue}")
    else:
        print("\n  Implementation issues: none")

    print("=" * 65)

    return summary


if __name__ == '__main__':
    main()
