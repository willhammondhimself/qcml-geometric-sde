"""
Central 17-method × 12-crisis regime detection comparison pipeline.

Usage:
    python experiments/regime_comparison.py
    python experiments/regime_comparison.py --causal
    python experiments/regime_comparison.py --causal --window-size 20
    python experiments/regime_comparison.py --quick  # 4 crises only

Outputs:
    experiments/outputs/regime_detection/comparison_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import (
    fetch_polygon_data, create_feature_matrix, ALL_CRISES,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    BOCPDDetector,
    IsolationForestDetector,
    RandomForestRegimeDetector,
)
from experiments.additional_detectors import (
    QCMLChernDetector,
    GeometricConsensusDetector,
)
from experiments.evaluation import (
    compute_cohens_d_with_ci,
    welch_t_test,
    holm_bonferroni_correction,
    friedman_test,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# Helpers (also imported by tests)
# =============================================================================

def apply_persistence_filter(mask, min_persistence=3):
    """Remove runs of True shorter than min_persistence.

    Args:
        mask: Boolean array.
        min_persistence: Minimum run length to keep.

    Returns:
        Filtered boolean array.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    T = len(mask)
    result = np.zeros(T, dtype=bool)

    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if (j - i) >= min_persistence:
                result[i:j] = True
            i = j
        else:
            i += 1

    return result


def compute_adaptive_threshold(scores, min_expanding=20, quantile=0.95):
    """Compute expanding-window quantile threshold.

    Args:
        scores: 1-D array of regime scores.
        min_expanding: Minimum history before computing threshold.
        quantile: Quantile for threshold (e.g. 0.95).

    Returns:
        thresholds: Array of same length as scores.
    """
    T = len(scores)
    thresholds = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = scores[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) > 0:
            thresholds[t] = np.nanpercentile(past_valid, quantile * 100)
    return thresholds


# =============================================================================
# Build Detectors
# =============================================================================

def build_qcml_detectors(causal=False, expanding_interval=252):
    """Build all QCML-based detectors.

    Args:
        causal: If True, use expanding window refit (no future data).
        expanding_interval: Refit interval in trading days.

    Returns:
        List of (name, detector) tuples.
    """
    shared = dict(hilbert_dim=8, n_pca_components=15, rolling_window=20, seed=42)
    if causal:
        shared['expanding_refit_interval'] = expanding_interval

    # Berry uses random operators (d=0.59 vs 0.28 pca_inspired, operator_ablation.py).
    # QFI and MLF use pca_inspired operators.
    detectors = [
        ('Berry Phase Rate', BerryPhaseRateDetector(
            **shared, operator_method='random')),
        ('QFI Determinant', QFIDeterminantDetector(
            **shared, operator_method='pca_inspired')),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector(
            **shared, operator_method='pca_inspired')),
    ]

    # Chern and Consensus don't support expanding window
    detectors.append(('QCML Chern', QCMLChernDetector(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired', seed=42,
    )))
    detectors.append(('Geometric Consensus', GeometricConsensusDetector(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired', seed=42,
    )))

    return detectors


def build_classical_detectors():
    """Build all classical baseline detectors (excluding RF)."""
    return [
        ('Rolling Vol Z', RollingVolatilityDetector(vol_window=20, min_expanding=60)),
        ('CUSUM', CUSUMDetector(burn_in=60)),
        ('HMM', HMMRegimeDetector(n_iter=100, seed=42)),
        ('BOCPD', BOCPDDetector(hazard_rate=250.0)),
        ('Isolation Forest', IsolationForestDetector(n_estimators=100, seed=42)),
    ]


# =============================================================================
# Main Comparison Pipeline
# =============================================================================

def run_comparison(
    causal=False,
    window_size=10,
    quick=False,
    n_bootstrap=10000,
):
    """Run full 17-method × 12-crisis comparison.

    Args:
        causal: Use expanding-window PCA/operators (no lookahead).
        window_size: Crisis window extension in trading days (±).
        quick: Only run on 4 representative crises.
        n_bootstrap: Bootstrap resamples for CIs.

    Returns:
        dict with all results.
    """
    logger.info("=" * 70)
    logger.info(f"Regime Comparison Pipeline (causal={causal}, window=±{window_size}d)")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1] Fetching data from Polygon...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"  Feature matrix: {X.shape}, dates: {dates[0]} to {dates[-1]}")

    # Build enriched features
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Enriched features: {X_enriched.shape}")

    # ---- Crisis selection ----
    if quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = list(ALL_CRISES.keys())

    crises = {k: ALL_CRISES[k] for k in crisis_keys}
    logger.info(f"  Evaluating {len(crises)} crises")

    # ---- Build detectors ----
    logger.info("\n[2] Building detectors...")
    qcml_detectors = build_qcml_detectors(causal=causal)
    classical_detectors = build_classical_detectors()

    # ---- Fit and score unsupervised methods ----
    all_scores = {}

    logger.info("\n[3] Fitting QCML detectors...")
    for name, det in qcml_detectors:
        logger.info(f"  {name}...")
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        all_scores[name] = scores

    logger.info("\n[4] Fitting classical detectors...")
    for name, det in classical_detectors:
        logger.info(f"  {name}...")
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        all_scores[name] = scores

    # ---- RF with leave-one-crisis-out ----
    logger.info("\n[5] Fitting RF (leave-one-crisis-out)...")
    rf_scores_per_crisis = {}
    for held_out_key in crises:
        # Build labels aligned with raw X (fit_with_labels trims first 19 rows internally)
        y = np.zeros(len(X))
        for ck, ci in crises.items():
            if ck == held_out_key:
                continue
            cs = pd.Timestamp(ci['start'])
            ce = pd.Timestamp(ci['end'])
            mask = (dates >= cs) & (dates <= ce)
            y[mask] = 1.0

        rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
        rf.fit_with_labels(X, y)

        # Score on enriched-length data
        scores = rf.compute_regime_scores(X)
        # Trim to match enriched dates
        rf_scores_per_crisis[held_out_key] = scores[19:] if len(scores) > len(dates_enriched) else scores

    # ---- Compute Cohen's d for each method × crisis ----
    logger.info("\n[6] Computing Cohen's d with bootstrap CIs...")
    results = {}

    for method_name, scores in all_scores.items():
        method_results = {}
        for ck, ci in crises.items():
            cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
            ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)

            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            crisis_scores = scores[crisis_mask]
            normal_scores = scores[normal_mask]

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_scores, normal_scores, n_bootstrap=n_bootstrap
            )
            method_results[ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }

        results[method_name] = method_results

    # RF results (per-crisis scores)
    rf_results = {}
    for ck, ci in crises.items():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)

        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        rf_sc = rf_scores_per_crisis.get(ck)
        if rf_sc is not None:
            # Ensure same length
            if len(rf_sc) != len(dates_enriched):
                rf_sc = rf_sc[:len(dates_enriched)]
            crisis_scores = rf_sc[crisis_mask]
            normal_scores = rf_sc[normal_mask]
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_scores, normal_scores, n_bootstrap=n_bootstrap
            )
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan

        rf_results[ck] = {
            'd': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        }

    results['Random Forest'] = rf_results

    # ---- Summary statistics ----
    logger.info("\n[7] Computing summary statistics...")

    # Build d-value matrix for Friedman test
    method_names = list(results.keys())
    n_methods = len(method_names)
    n_crises = len(crises)
    d_matrix = np.full((n_crises, n_methods), np.nan)

    for j, mname in enumerate(method_names):
        for i, ck in enumerate(crises.keys()):
            val = results[mname].get(ck, {}).get('d')
            if val is not None:
                d_matrix[i, j] = val

    # Friedman test
    chi_sq, p_val, mean_ranks = friedman_test(d_matrix)

    # Median d per method
    median_d = {}
    for j, mname in enumerate(method_names):
        col = d_matrix[:, j]
        valid = col[~np.isnan(col)]
        median_d[mname] = float(np.median(valid)) if len(valid) > 0 else None

    # ---- Print summary ----
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    sorted_methods = sorted(median_d.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    for rank, (mname, md) in enumerate(sorted_methods, 1):
        logger.info(f"  {rank:2d}. {mname:25s}  median d = {md:.3f}" if md else f"  {rank:2d}. {mname:25s}  median d = N/A")

    logger.info(f"\n  Friedman chi-sq = {chi_sq:.2f}, p = {p_val:.4f}" if not np.isnan(chi_sq) else "  Friedman test: insufficient data")

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'causal': causal,
            'window_size': window_size,
            'n_bootstrap': n_bootstrap,
            'quick': quick,
            'n_crises': n_crises,
            'n_methods': n_methods,
        },
        'results': results,
        'summary': {
            'median_d': median_d,
            'friedman_chi_sq': float(chi_sq) if not np.isnan(chi_sq) else None,
            'friedman_p': float(p_val) if not np.isnan(p_val) else None,
            'mean_ranks': {mname: float(mean_ranks[j]) for j, mname in enumerate(method_names)} if not np.any(np.isnan(mean_ranks)) else None,
        },
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'causal_' if causal else ''
    out_path = out_dir / f'{tag}comparison_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Regime detection comparison pipeline')
    parser.add_argument('--causal', action='store_true',
                        help='Use expanding-window PCA/operators (no lookahead)')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Crisis window extension ± days (default: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Only run on 4 representative crises')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Bootstrap resamples for CIs (default: 10000)')
    args = parser.parse_args()

    run_comparison(
        causal=args.causal,
        window_size=args.window_size,
        quick=args.quick,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == '__main__':
    main()
