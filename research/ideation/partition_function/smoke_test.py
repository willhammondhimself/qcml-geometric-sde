"""
Smoke test for PartitionFunctionDetector on 4 financial crises.

Tests: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Data: SPY, DIA from 2005-01-01 to 2025-12-31
Metric: Cohen's d (crisis vs. 60-day pre-crisis normal) with bootstrap CI (n=1000)

Output: research/ideation/partition_function/smoke_results.json
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.partition_function.detector import PartitionFunctionDetector

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000   # Smoke-test speed; paper uses 10 000
CONTEXT_DAYS = 60    # Days before crisis start used as "normal" baseline
BETA = 1.0           # Inverse temperature

np.random.seed(42)


def main():
    t0 = time.time()
    implementation_issues = []

    # -------------------------------------------------------------------------
    # 1. Fetch and prepare data
    # -------------------------------------------------------------------------
    logger.info(f"Fetching {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices shape: {close_prices.shape}")

    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    logger.info(f"Feature matrix: {features.shape}, dates {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # 2. Fit detector
    # -------------------------------------------------------------------------
    logger.info("Initializing and fitting PartitionFunctionDetector...")
    detector = PartitionFunctionDetector(
        hilbert_dim=4,
        n_pca_components=8,
        operator_method='random',   # avoids Kramers degeneracy
        normalization='soft',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        beta=BETA,
    )
    detector.fit(features)

    # -------------------------------------------------------------------------
    # 3. Compute regime scores
    # -------------------------------------------------------------------------
    logger.info("Computing specific heat scores...")
    t_scores = time.time()
    try:
        scores = detector.compute_regime_scores(features)
    except Exception as exc:
        msg = f"compute_regime_scores failed: {exc}"
        logger.error(msg)
        implementation_issues.append(msg)
        scores = np.full(len(features), np.nan)

    dt_scores = time.time() - t_scores
    n_valid = int(np.sum(~np.isnan(scores)))
    logger.info(f"Scores computed in {dt_scores:.1f}s — valid: {n_valid}/{len(scores)}")

    if n_valid == 0:
        implementation_issues.append("All scores are NaN — check Hamiltonian diagonalization.")

    # -------------------------------------------------------------------------
    # 4. Diagnostic: raw specific heat sanity check
    # -------------------------------------------------------------------------
    logger.info("Running diagnostic on raw specific heat series...")
    try:
        C_raw = detector.compute_specific_heat_series(features)
        n_nonnan = np.sum(~np.isnan(C_raw))
        c_mean = float(np.nanmean(C_raw))
        c_std = float(np.nanstd(C_raw))
        c_min = float(np.nanmin(C_raw))
        c_max = float(np.nanmax(C_raw))
        logger.info(
            f"  Raw C: n_nonnan={n_nonnan}, mean={c_mean:.4f}, "
            f"std={c_std:.4f}, min={c_min:.4f}, max={c_max:.4f}"
        )
        if c_std < 1e-10:
            implementation_issues.append(
                "Raw specific heat has near-zero variance — signal may be degenerate."
            )
        if c_min < 0:
            implementation_issues.append(
                f"Specific heat has negative values (min={c_min:.6f}). "
                "Numerical instability in energy variance."
            )
    except Exception as exc:
        msg = f"Raw specific heat diagnostic failed: {exc}"
        logger.warning(msg)
        implementation_issues.append(msg)
        C_raw = None
        c_mean = c_std = c_min = c_max = None

    # -------------------------------------------------------------------------
    # 5. Per-crisis Cohen's d
    # -------------------------------------------------------------------------
    results = {}
    dates_series = pd.DatetimeIndex(dates)

    for crisis_key in TEST_CRISES:
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
            f"{crisis_key}: crisis_n={len(crisis_valid)}, normal_n={len(normal_valid)}"
        )

        if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
            try:
                d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                    crisis_valid, normal_valid,
                    n_bootstrap=N_BOOTSTRAP, seed=42,
                )
            except Exception as exc:
                msg = f"Bootstrap CI failed for {crisis_key}: {exc}"
                logger.warning(msg)
                implementation_issues.append(msg)
                d, ci_lo, ci_hi = np.nan, np.nan, np.nan
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan
            implementation_issues.append(
                f"{crisis_key}: insufficient valid scores "
                f"(crisis={len(crisis_valid)}, normal={len(normal_valid)})"
            )

        results[crisis_key] = {
            'cohens_d': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'crisis_n': int(len(crisis_valid)),
            'normal_n': int(len(normal_valid)),
            'crisis_mean_score': float(np.nanmean(crisis_scores)) if len(crisis_valid) > 0 else None,
            'normal_mean_score': float(np.nanmean(normal_scores)) if len(normal_valid) > 0 else None,
        }
        d_str = f"{d:.3f}" if not np.isnan(d) else "NaN"
        logger.info(f"  {crisis_key}: d={d_str}")

    # -------------------------------------------------------------------------
    # 6. Summary statistics
    # -------------------------------------------------------------------------
    d_values = [r['cohens_d'] for r in results.values() if r['cohens_d'] is not None]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None
    passes_threshold = median_d is not None and median_d > 0.3

    total_time = time.time() - t0

    summary = {
        'detector': 'Partition Function (Specific Heat)',
        'question': 'Q7: Can we define a partition function / free energy for phase transition detection?',
        'config': {
            'hilbert_dim': 4,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'beta': BETA,
            'rolling_window': 20,
            'min_expanding': 60,
            'n_bootstrap': N_BOOTSTRAP,
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
        },
        'data': {
            'feature_shape': list(features.shape),
            'n_valid_scores': n_valid,
        },
        'raw_specific_heat': {
            'mean': c_mean,
            'std': c_std,
            'min': c_min,
            'max': c_max,
        },
        'cohens_d_per_crisis': {k: v['cohens_d'] for k, v in results.items()},
        'per_crisis_detail': results,
        'median_d': median_d,
        'max_d': max_d,
        'passes_threshold': passes_threshold,
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
    # 8. Print human-readable summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 65)
    print("PARTITION FUNCTION DETECTOR (Q7) — SMOKE TEST RESULTS")
    print(f"  beta (inverse temperature) = {BETA}")
    print("=" * 65)
    for crisis_key, r in results.items():
        d = r['cohens_d']
        label = ALL_CRISES[crisis_key]['label']
        if d is not None:
            ci_str = (
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                if r['ci_lo'] is not None else "CI N/A"
            )
            print(
                f"  {crisis_key:22s}  d={d:+.3f}  {ci_str}  "
                f"(n={r['crisis_n']}/{r['normal_n']})"
            )
        else:
            print(f"  {crisis_key:22s}  d=NaN  (n={r['crisis_n']}/{r['normal_n']})")

    print()
    print(f"  Median Cohen's d : {median_d:.3f}" if median_d is not None else "  Median d: N/A")
    print(f"  Max Cohen's d    : {max_d:.3f}" if max_d is not None else "  Max d: N/A")
    print(f"  Passes threshold (d > 0.3): {passes_threshold}")
    print(f"  Total runtime    : {total_time:.1f}s")
    if implementation_issues:
        print(f"\n  Implementation issues ({len(implementation_issues)}):")
        for issue in implementation_issues:
            print(f"    - {issue}")
    else:
        print("\n  No implementation issues.")
    print("=" * 65)

    return summary


if __name__ == '__main__':
    main()
