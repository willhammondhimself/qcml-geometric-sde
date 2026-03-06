"""
Smoke test for Entanglement Entropy Detector on 4 financial crises.

Q4: "Does entanglement entropy of subsector partitions (tech vs. financials)
     predict contagion?"

Sector ETFs used (preferred): XLK (tech), XLF (financials), XLE (energy), XLV (healthcare)
Partition: A = [XLK, XLE] (tech + energy), B = [XLF, XLV] (financials + healthcare)
Fallback: SPY, DIA (if sector ETFs unavailable for full period)

Test crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Metric: Cohen's d with n_bootstrap=1000 (block bootstrap)
Output: smoke_results.json
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
from research.ideation.entanglement_entropy.detector import EntanglementEntropyDetector

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary: sector ETFs (available since 1998-12-22 for XLK, XLF, XLE, XLV)
SECTOR_SYMBOLS = ['XLK', 'XLF', 'XLE', 'XLV']
# Fallback: broad market ETFs
FALLBACK_SYMBOLS = ['SPY', 'DIA']

START_DATE = '2005-01-01'
END_DATE = '2025-12-31'

TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60  # "normal" baseline: 60 trading days before crisis start

# Hilbert space: 2-qubit (dim=4), split as dim_A=2, dim_B=2
# Avoids Kramers degeneracy by using method='random'
HILBERT_DIM = 4
DIM_A = 2  # subsystem A (half of Hilbert space)
DIM_B = 2  # subsystem B


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def try_fetch_sectors():
    """Attempt to fetch sector ETFs; fall back to SPY/DIA if unavailable."""
    logger.info(f"Attempting sector ETFs: {SECTOR_SYMBOLS}")
    try:
        df = fetch_data(SECTOR_SYMBOLS, START_DATE, END_DATE)
        close = df['close'].unstack('symbol').dropna()
        if close.shape[1] < 4:
            raise RuntimeError(f"Only {close.shape[1]} symbols available, expected 4")
        # Verify we have enough data (at least 2 years before first crisis)
        if len(close) < 500:
            raise RuntimeError(f"Insufficient data: {len(close)} rows")
        logger.info(f"Sector ETFs OK: {close.shape}, {close.index[0]} to {close.index[-1]}")
        return close, SECTOR_SYMBOLS, 'sector_etfs'
    except Exception as e:
        logger.warning(f"Sector ETF fetch failed: {e}. Falling back to SPY/DIA.")
        df = fetch_data(FALLBACK_SYMBOLS, START_DATE, END_DATE)
        close = df['close'].unstack('symbol').dropna()
        logger.info(f"Fallback data OK: {close.shape}")
        return close, FALLBACK_SYMBOLS, 'spy_dia_fallback'


def evaluate_crisis(
    scores: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_key: str,
) -> dict:
    """Evaluate Cohen's d for a single crisis.

    Args:
        scores: Regime score time series of length T.
        dates: Corresponding timestamps, length T.
        crisis_key: Key into ALL_CRISES dict.

    Returns:
        dict with cohens_d, ci_lo, ci_hi, crisis_n, normal_n, etc.
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
        logger.warning(f"  {crisis_key}: insufficient data for Cohen's d")

    return {
        'cohens_d': float(d) if not np.isnan(d) else None,
        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        'crisis_n': int(len(crisis_valid)),
        'normal_n': int(len(normal_valid)),
        'crisis_mean': float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
        'normal_mean': float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    implementation_issues = []

    # -------------------------------------------------------------------------
    # Step 1: Fetch data
    # -------------------------------------------------------------------------
    close_prices, symbols_used, data_source = try_fetch_sectors()
    n_sectors = close_prices.shape[1]
    logger.info(f"Using {n_sectors} symbols: {list(close_prices.columns)}")

    # -------------------------------------------------------------------------
    # Step 2: Create feature matrix
    # -------------------------------------------------------------------------
    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    dates = pd.DatetimeIndex(dates)
    logger.info(f"Feature matrix: {features.shape}, {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 3: Initialize detector
    # -------------------------------------------------------------------------
    # Use hilbert_dim=4 (2-qubit), dim_A=2, dim_B=2 for 4→(2,2) bipartition
    # operator_method='random' to avoid Kramers degeneracy
    logger.info("Initializing Entanglement Entropy Detector...")
    logger.info(f"  hilbert_dim={HILBERT_DIM}, dim_A={DIM_A}, dim_B={DIM_B}")
    logger.info(f"  operator_method=random (avoids Kramers degeneracy)")

    detector = EntanglementEntropyDetector(
        hilbert_dim=HILBERT_DIM,
        dim_A=DIM_A,
        dim_B=DIM_B,
        n_pca_components=8,
        operator_method='random',
        normalization='soft',
        rolling_window=20,
        min_expanding=60,
        seed=42,
    )

    # -------------------------------------------------------------------------
    # Step 4: Fit detector
    # -------------------------------------------------------------------------
    logger.info("Fitting detector...")
    t_fit = time.time()
    detector.fit(features)
    dt_fit = time.time() - t_fit
    logger.info(f"Fit complete in {dt_fit:.1f}s")

    # -------------------------------------------------------------------------
    # Step 5: Compute regime scores
    # -------------------------------------------------------------------------
    logger.info("Computing entanglement entropy scores...")
    t_scores = time.time()
    scores = detector.compute_regime_scores(features)
    dt_scores = time.time() - t_scores

    n_valid = int(np.sum(~np.isnan(scores)))
    logger.info(f"Scores computed in {dt_scores:.1f}s. Valid: {n_valid}/{len(scores)}")

    if n_valid == 0:
        implementation_issues.append("All scores are NaN — computation failed")
        logger.error("All scores are NaN!")

    # -------------------------------------------------------------------------
    # Step 6: Per-crisis Cohen's d
    # -------------------------------------------------------------------------
    logger.info("Evaluating per-crisis Cohen's d...")
    per_crisis = {}
    for crisis_key in TEST_CRISES:
        per_crisis[crisis_key] = evaluate_crisis(scores, dates, crisis_key)
        r = per_crisis[crisis_key]
        if r['cohens_d'] is not None:
            logger.info(
                f"  {crisis_key}: d={r['cohens_d']:.3f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            )
        else:
            logger.warning(f"  {crisis_key}: d=NaN")
            implementation_issues.append(f"{crisis_key}: Cohen's d could not be computed")

    # -------------------------------------------------------------------------
    # Step 7: Summary
    # -------------------------------------------------------------------------
    d_values = [r['cohens_d'] for r in per_crisis.values() if r['cohens_d'] is not None]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None
    passes_threshold = median_d is not None and median_d > 0.3
    total_time = time.time() - t0

    # -------------------------------------------------------------------------
    # Step 8: Diagnostic — check for Kramers degeneracy signature
    # -------------------------------------------------------------------------
    # If all d ~ 0.0, likely degeneracy issue. With method='random' this should
    # not occur but we flag it for transparency.
    if d_values and max_d < 0.05:
        implementation_issues.append(
            "All d values near zero — possible ground state degeneracy. "
            "Consider increasing hilbert_dim or checking operator_method='random'."
        )

    summary = {
        'knight': 2,
        'role': 'empirical_test',
        'detector': 'Entanglement Entropy',
        'config': {
            'hilbert_dim': HILBERT_DIM,
            'dim_A': DIM_A,
            'dim_B': DIM_B,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'n_bootstrap': N_BOOTSTRAP,
        },
        'data': {
            'symbols_used': symbols_used,
            'data_source': data_source,
            'n_sectors': n_sectors,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'n_valid_scores': n_valid,
        },
        'cohens_d_per_crisis': {
            k: v['cohens_d'] for k, v in per_crisis.items()
        },
        'per_crisis_detail': per_crisis,
        'median_d': median_d,
        'max_d': max_d,
        'passes_threshold': passes_threshold,
        'implementation_issues': implementation_issues,
        'timing': {
            'total_seconds': round(total_time, 1),
            'fit_seconds': round(dt_fit, 1),
            'score_computation_seconds': round(dt_scores, 1),
        },
    }

    # -------------------------------------------------------------------------
    # Step 9: Save results
    # -------------------------------------------------------------------------
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("ENTANGLEMENT ENTROPY DETECTOR — SMOKE TEST RESULTS")
    print("=" * 65)
    print(f"  Data source: {data_source}  |  Symbols: {symbols_used}")
    print(f"  Hilbert dim: {HILBERT_DIM} ({DIM_A}x{DIM_B} bipartition)  |  method=random")
    print()
    for crisis_key in TEST_CRISES:
        r = per_crisis[crisis_key]
        d = r['cohens_d']
        if d is not None:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(
                f"  {crisis_key:22s}  d={d:.3f}  CI={ci}"
                f"  (n_c={r['crisis_n']}, n_n={r['normal_n']})"
            )
        else:
            print(f"  {crisis_key:22s}  d=NaN")
    print()
    print(f"  Median d: {median_d:.3f}" if median_d is not None else "  Median d: NaN")
    print(f"  Max d:    {max_d:.3f}" if max_d is not None else "  Max d:    NaN")
    print(f"  Passes threshold (median d > 0.3): {passes_threshold}")
    print(f"  Total time: {total_time:.1f}s")
    if implementation_issues:
        print(f"\n  Issues: {implementation_issues}")
    print("=" * 65)

    return summary


if __name__ == '__main__':
    main()
