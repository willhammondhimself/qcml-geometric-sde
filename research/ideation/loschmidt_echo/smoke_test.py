"""
Smoke test for Loschmidt Echo Detector on 4 financial crises.

Tests: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Data: SPY, DIA from 2005-01-01 to 2025-12-31
Metric: Cohen's d (crisis vs normal) with bootstrap CI (n=1000)
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
from research.ideation.loschmidt_echo.detector import LoschmidtEchoDetector

# Suppress noisy warnings
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
N_BOOTSTRAP = 1000  # Speed over precision for smoke test
CONTEXT_DAYS = 60  # Days before crisis for "normal" baseline


def main():
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Fetch data
    # -------------------------------------------------------------------------
    logger.info(f"Fetching data for {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)

    # Extract close prices: raw_df has MultiIndex (symbol, date), columns include 'close'
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices shape: {close_prices.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Create feature matrix
    # -------------------------------------------------------------------------
    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    logger.info(f"Feature matrix shape: {features.shape}, dates: {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 3: Initialize and fit detector
    # -------------------------------------------------------------------------
    logger.info("Initializing Loschmidt Echo Detector...")
    detector = LoschmidtEchoDetector(
        hilbert_dim=4,
        n_pca_components=8,
        operator_method='random',
        normalization='soft',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        perturbation_scale=0.01,
        tau_values=np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
    )

    logger.info("Fitting detector on full feature matrix...")
    detector.fit(features)

    # -------------------------------------------------------------------------
    # Step 4: Compute regime scores
    # -------------------------------------------------------------------------
    logger.info("Computing regime scores (this may take a while)...")
    t_scores = time.time()
    scores = detector.compute_regime_scores(features)
    dt_scores = time.time() - t_scores
    logger.info(f"Regime scores computed in {dt_scores:.1f}s")

    n_valid = np.sum(~np.isnan(scores))
    logger.info(f"Valid scores: {n_valid}/{len(scores)}")

    # -------------------------------------------------------------------------
    # Step 5: Evaluate per-crisis Cohen's d
    # -------------------------------------------------------------------------
    results = {}
    import pandas as pd
    dates_series = pd.DatetimeIndex(dates)

    for crisis_key in TEST_CRISES:
        crisis = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis['start'])
        crisis_end = pd.Timestamp(crisis['end'])
        normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

        # Crisis period mask
        crisis_mask = (dates_series >= crisis_start) & (dates_series <= crisis_end)
        # Normal period: CONTEXT_DAYS before crisis
        normal_mask = (dates_series >= normal_start) & (dates_series < crisis_start)

        crisis_scores = scores[np.asarray(crisis_mask)]
        normal_scores = scores[np.asarray(normal_mask)]

        # Filter NaN
        crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
        normal_valid = normal_scores[~np.isnan(normal_scores)]

        logger.info(
            f"{crisis_key}: crisis_n={len(crisis_valid)}, normal_n={len(normal_valid)}"
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
        logger.info(f"  d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # -------------------------------------------------------------------------
    # Step 6: Summary statistics
    # -------------------------------------------------------------------------
    d_values = [r['cohens_d'] for r in results.values() if r['cohens_d'] is not None]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None
    passes_threshold = median_d is not None and median_d > 0.3

    total_time = time.time() - t0

    summary = {
        'detector': 'Loschmidt Echo',
        'config': {
            'hilbert_dim': 4,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'perturbation_scale': 0.01,
            'tau_values': [0.1, 0.5, 1.0, 2.0, 5.0],
            'n_bootstrap': N_BOOTSTRAP,
        },
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'n_valid_scores': int(n_valid),
        },
        'per_crisis': results,
        'summary': {
            'median_d': median_d,
            'max_d': max_d,
            'passes_threshold_0.3': passes_threshold,
            'd_values': d_values,
        },
        'timing': {
            'total_seconds': round(total_time, 1),
            'score_computation_seconds': round(dt_scores, 1),
        },
    }

    # -------------------------------------------------------------------------
    # Step 7: Save results
    # -------------------------------------------------------------------------
    output_path = os.path.join(
        os.path.dirname(__file__), 'smoke_results.json'
    )
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("LOSCHMIDT ECHO DETECTOR — SMOKE TEST RESULTS")
    print("=" * 60)
    for crisis_key, r in results.items():
        d = r['cohens_d']
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r['ci_lo'] is not None else "N/A"
        print(f"  {crisis_key:20s}  d={d:.3f}  CI={ci}  (n_crisis={r['crisis_n']}, n_normal={r['normal_n']})")
    print(f"\n  Median d: {median_d:.3f}")
    print(f"  Max d:    {max_d:.3f}")
    print(f"  Passes threshold (d>0.3): {passes_threshold}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 60)

    return summary


if __name__ == '__main__':
    main()
