"""
Smoke test for WassersteinStateDetector on 4 financial crises.

Tests: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Data: SPY, DIA from 2005-01-01 to 2025-12-31
Metric: Cohen's d (crisis vs normal) with bootstrap CI (n=1000)

Evaluates all three distance modes:
  - trace distance         (T = sqrt(1 - F))
  - Bures distance         (B = sqrt(2(1-sqrt(F))))
  - eigenvalue_wasserstein (W_1 on eigenvalue spectra — diagnostic only)
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
from research.ideation.wasserstein.detector import WassersteinStateDetector

# Suppress noisy warnings from yfinance / statsmodels
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# Configuration
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000   # Speed-over-precision for smoke test
CONTEXT_DAYS = 60    # Trading days before crisis start = "normal" baseline

# Detector configurations to evaluate
DETECTOR_CONFIGS = [
    {
        'name': 'trace',
        'params': dict(
            distance_mode='trace',
            hilbert_dim=4,
            n_pca_components=8,
            operator_method='random',
            normalization='soft',
            rolling_window=20,
            min_expanding=60,
            adaptive_epsilon=True,
            seed=42,
        ),
    },
    {
        'name': 'bures',
        'params': dict(
            distance_mode='bures',
            hilbert_dim=4,
            n_pca_components=8,
            operator_method='random',
            normalization='soft',
            rolling_window=20,
            min_expanding=60,
            adaptive_epsilon=True,
            seed=42,
        ),
    },
    {
        'name': 'eigenvalue_wasserstein',
        'params': dict(
            distance_mode='eigenvalue_wasserstein',
            hilbert_dim=4,
            n_pca_components=8,
            operator_method='random',
            normalization='soft',
            rolling_window=20,
            min_expanding=60,
            adaptive_epsilon=True,
            seed=42,
        ),
    },
]


def evaluate_detector(detector, scores, dates, crisis_key):
    """Compute Cohen's d for one detector/crisis pair.

    Args:
        detector: Fitted detector (unused after scoring; kept for future diagnostics).
        scores: 1-D score array aligned with `dates`.
        dates: DatetimeIndex of score timestamps.
        crisis_key: Key into ALL_CRISES dict.

    Returns:
        dict with cohens_d, ci_lo, ci_hi, crisis_n, normal_n, crisis_mean, normal_mean.
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
            crisis_valid, normal_valid, n_bootstrap=N_BOOTSTRAP, seed=42,
        )
    else:
        d, ci_lo, ci_hi = np.nan, np.nan, np.nan

    return {
        'cohens_d': float(d) if not np.isnan(d) else None,
        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        'crisis_n': int(len(crisis_valid)),
        'normal_n': int(len(normal_valid)),
        'crisis_mean': float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
        'normal_mean': float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
    }


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
    dates_index = pd.DatetimeIndex(dates)
    logger.info(f"Feature matrix: {features.shape}, {dates_index[0]} to {dates_index[-1]}")

    # -------------------------------------------------------------------------
    # Step 3: Evaluate each distance mode
    # -------------------------------------------------------------------------
    all_results = {}
    implementation_issues = []

    for cfg in DETECTOR_CONFIGS:
        mode_name = cfg['name']
        logger.info(f"\n--- Evaluating mode: {mode_name} ---")

        try:
            detector = WassersteinStateDetector(**cfg['params'])
            logger.info(f"  Fitting detector ({mode_name})...")
            t_fit = time.time()
            detector.fit(features)
            dt_fit = time.time() - t_fit
            logger.info(f"  Fit complete in {dt_fit:.1f}s")

            logger.info(f"  Computing regime scores ({mode_name})...")
            t_scores = time.time()
            scores = detector.compute_regime_scores(features)
            dt_scores = time.time() - t_scores
            logger.info(f"  Scores computed in {dt_scores:.1f}s")

            n_valid = int(np.sum(~np.isnan(scores)))
            logger.info(f"  Valid scores: {n_valid}/{len(scores)}")

            mode_results = {}
            for crisis_key in TEST_CRISES:
                r = evaluate_detector(detector, scores, dates_index, crisis_key)
                mode_results[crisis_key] = r
                d_str = f"{r['cohens_d']:.3f}" if r['cohens_d'] is not None else "N/A"
                ci_str = (
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                    if r['ci_lo'] is not None else "N/A"
                )
                logger.info(
                    f"  {crisis_key}: d={d_str}  CI={ci_str}  "
                    f"(crisis_n={r['crisis_n']}, normal_n={r['normal_n']})"
                )

            d_vals = [r['cohens_d'] for r in mode_results.values() if r['cohens_d'] is not None]
            median_d = float(np.median(d_vals)) if d_vals else None
            logger.info(f"  Median d ({mode_name}): {median_d:.3f}" if median_d is not None else "  Median d: N/A")

            all_results[mode_name] = {
                'per_crisis': mode_results,
                'median_d': median_d,
                'timing': {
                    'fit_seconds': round(dt_fit, 1),
                    'score_seconds': round(dt_scores, 1),
                },
                'n_valid_scores': n_valid,
            }

            # Sanity check for eigenvalue_wasserstein mode (expected: d=0 for pure states)
            if mode_name == 'eigenvalue_wasserstein' and median_d is not None and median_d > 1e-6:
                issue = (
                    f"eigenvalue_wasserstein returned non-zero median_d={median_d:.4f} "
                    "for pure states — unexpected. Check implementation."
                )
                logger.warning(issue)
                implementation_issues.append(issue)

        except Exception as e:
            import traceback
            msg = f"Mode '{mode_name}' failed: {e}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            implementation_issues.append(msg)
            all_results[mode_name] = {
                'error': str(e),
                'per_crisis': {},
                'median_d': None,
            }

    # -------------------------------------------------------------------------
    # Step 4: Select best mode and compute summary
    # -------------------------------------------------------------------------
    # Best = highest median_d among non-degenerate modes
    best_mode = None
    best_median_d = -np.inf
    for mode_name, mode_res in all_results.items():
        if mode_name == 'eigenvalue_wasserstein':
            continue  # Skip degenerate mode for selection
        md = mode_res.get('median_d')
        if md is not None and md > best_median_d:
            best_median_d = md
            best_mode = mode_name

    # Aggregate: per-crisis Cohen's d for best mode
    best_per_crisis = {}
    if best_mode is not None:
        for crisis_key, r in all_results[best_mode]['per_crisis'].items():
            best_per_crisis[crisis_key] = r.get('cohens_d')

    # Document mathematical relationship for pure states
    math_note = (
        "For pure states: trace_d = sqrt(1-F), bures_d = sqrt(2(1-sqrt(F))). "
        "Both are exact quantum Wasserstein analogs. "
        "eigenvalue_wasserstein = 0 for all pure states (degenerate). "
        "fidelity-based detectors use F directly; trace/bures use 1-F transforms."
    )

    total_time = time.time() - t0

    summary = {
        'detector': 'Wasserstein State Detector',
        'question': (
            'Q10: Does Wasserstein distance between consecutive quantum states '
            'work better than fidelity?'
        ),
        'config': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'n_bootstrap': N_BOOTSTRAP,
            'context_days': CONTEXT_DAYS,
            'test_crises': TEST_CRISES,
        },
        'modes_evaluated': list(DETECTOR_CONFIGS[i]['name'] for i in range(len(DETECTOR_CONFIGS))),
        'all_mode_results': all_results,
        'best_mode': best_mode,
        'cohens_d_per_crisis': best_per_crisis,
        'median_d': float(best_median_d) if best_mode is not None else None,
        'passes_threshold': best_median_d > 0.3 if best_mode is not None else False,
        'implementation_issues': implementation_issues,
        'math_note': math_note,
        'timing': {'total_seconds': round(total_time, 1)},
    }

    # -------------------------------------------------------------------------
    # Step 5: Save results
    # -------------------------------------------------------------------------
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # -------------------------------------------------------------------------
    # Step 6: Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WASSERSTEIN STATE DETECTOR — SMOKE TEST RESULTS")
    print("=" * 70)
    for mode_name, mode_res in all_results.items():
        print(f"\n  Mode: {mode_name}")
        if 'error' in mode_res:
            print(f"    ERROR: {mode_res['error']}")
            continue
        for crisis_key, r in mode_res.get('per_crisis', {}).items():
            d = r.get('cohens_d')
            ci_lo, ci_hi = r.get('ci_lo'), r.get('ci_hi')
            d_str = f"{d:.3f}" if d is not None else "N/A"
            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else "N/A"
            print(
                f"    {crisis_key:20s}  d={d_str:7s}  CI={ci_str}"
                f"  (crisis_n={r['crisis_n']}, normal_n={r['normal_n']})"
            )
        md = mode_res.get('median_d')
        print(f"    Median d: {md:.3f}" if md is not None else "    Median d: N/A")

    print(f"\n  Best mode: {best_mode}")
    print(f"  Best median d: {best_median_d:.3f}" if best_mode is not None else "  Best median d: N/A")
    print(f"  Passes threshold (d > 0.3): {best_median_d > 0.3 if best_mode is not None else False}")
    print(f"  Total time: {total_time:.1f}s")
    if implementation_issues:
        print(f"\n  Implementation issues:")
        for issue in implementation_issues:
            print(f"    - {issue}")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    main()
