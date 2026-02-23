"""
Central regime detection comparison pipeline with causal preprocessing.

Per-crisis causal fitting: for each crisis, scaler/PCA/operators are fitted
only on data *before* that crisis window, eliminating lookahead bias.

Usage:
    python experiments/regime_comparison.py
    python experiments/regime_comparison.py --quick  # 4 crises only
    python experiments/regime_comparison.py --full    # all 16 crises

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
# HPO-Optimized QCML Detector Configs
# =============================================================================

# Source: experiments/outputs/regime_detection/honest_hpo/hpo_results_20260221_220739.json
# Train: pre-2020 crises, Val: post-2020 crises, Consistency-penalized objective
HPO_CONFIGS = {
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
        ),
    },
    'QFI Determinant': {
        'class': QFIDeterminantDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
        ),
    },
    'Multi-Lag Fidelity': {
        'class': MultiLagFidelityDetector,
        'params': dict(
            hilbert_dim=4, n_pca_components=8, rolling_window=20,
            operator_method='pca_inspired', seed=42,
        ),
    },
    'QCML Chern': {
        'class': QCMLChernDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15,
            operator_method='random', seed=42,
        ),
    },
    'Geometric Consensus': {
        'class': GeometricConsensusDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15,
            operator_method='random', seed=42,
        ),
    },
}


# =============================================================================
# Helpers
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
# Classical Baseline Factories
# =============================================================================

CLASSICAL_CONFIGS = {
    'Rolling Vol Z': {
        'class': RollingVolatilityDetector,
        'params': dict(vol_window=20, min_expanding=60),
    },
    'CUSUM': {
        'class': CUSUMDetector,
        'params': dict(burn_in=60),
    },
    'HMM': {
        'class': HMMRegimeDetector,
        'params': dict(n_iter=100, seed=42),
    },
    'BOCPD': {
        'class': BOCPDDetector,
        'params': dict(hazard_rate=250.0),
    },
    'Isolation Forest': {
        'class': IsolationForestDetector,
        'params': dict(n_estimators=100, seed=42),
    },
}


# =============================================================================
# Main Comparison Pipeline
# =============================================================================

def run_comparison(
    window_size=10,
    quick=False,
    full=False,
    n_bootstrap=10000,
):
    """Run comparison with per-crisis causal preprocessing.

    For each crisis, all detectors fit scaler/PCA/operators on data
    strictly before the crisis window. No future data is used in
    preprocessing.

    Args:
        window_size: Crisis window extension in trading days (±).
        quick: Only run on 4 representative crises.
        full: Run on all 16 crises (including pre-2005).
        n_bootstrap: Bootstrap resamples for CIs.

    Returns:
        dict with all results.
    """
    logger.info("=" * 70)
    logger.info("Regime Comparison Pipeline (per-crisis causal preprocessing)")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1] Fetching data from Polygon...")
    symbols = ['SPY', 'DIA']
    start = '1995-01-01' if full else '1995-01-01'
    raw = fetch_polygon_data(symbols, start, '2024-12-31')
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
    elif full:
        crisis_keys = list(ALL_CRISES.keys())
    else:
        # Default: post-2005 crises only (Polygon data available)
        crisis_keys = [k for k in ALL_CRISES if int(k[:4]) >= 2005]

    crises = {k: ALL_CRISES[k] for k in crisis_keys if k in ALL_CRISES}
    logger.info(f"  Evaluating {len(crises)} crises")

    # ---- Per-crisis causal evaluation ----
    results = {}

    logger.info("\n[2] Per-crisis causal fitting and evaluation...")
    for ck, ci in crises.items():
        crisis_start = pd.Timestamp(ci['start'])
        crisis_end = pd.Timestamp(ci['end'])

        # Causal cutoff: fit only on data before crisis window
        cutoff_date = crisis_start - pd.Timedelta(days=window_size)
        fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

        if fit_end_idx < 100:
            logger.warning(f"  Skipping {ck}: insufficient pre-crisis data ({fit_end_idx} rows)")
            continue

        logger.info(f"\n  --- {ck} ({ci['label']}) ---")
        logger.info(f"  Causal cutoff: {dates_enriched[fit_end_idx - 1].date()} "
                     f"({fit_end_idx} rows for fitting)")

        # Define crisis and normal masks
        cs = crisis_start - pd.Timedelta(days=window_size)
        ce = crisis_end + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        # --- QCML detectors (per-crisis causal fit) ---
        for method_name, config in HPO_CONFIGS.items():
            params = {**config['params'], 'causal_fit_length': fit_end_idx}
            det = config['class'](**params)
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
            )

            if method_name not in results:
                results[method_name] = {}
            results[method_name][ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }
            logger.info(f"    {method_name:25s}  d = {d:.3f}" if not np.isnan(d) else
                        f"    {method_name:25s}  d = N/A")

        # --- Classical baselines (causal fit on pre-crisis data) ---
        for method_name, config in CLASSICAL_CONFIGS.items():
            det = config['class'](**config['params'])
            det.fit(X_enriched[:fit_end_idx])
            scores = det.compute_regime_scores(X_enriched)

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
            )

            if method_name not in results:
                results[method_name] = {}
            results[method_name][ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }
            logger.info(f"    {method_name:25s}  d = {d:.3f}" if not np.isnan(d) else
                        f"    {method_name:25s}  d = N/A")

    # ---- RF with leave-one-crisis-out ----
    logger.info("\n[3] Fitting RF (leave-one-crisis-out, causal per-crisis)...")
    rf_results = {}
    for held_out_key in crises:
        ci = crises[held_out_key]
        crisis_start = pd.Timestamp(ci['start'])
        crisis_end = pd.Timestamp(ci['end'])

        # Build labels excluding held-out crisis
        y = np.zeros(len(X))
        for train_ck, train_ci in crises.items():
            if train_ck == held_out_key:
                continue
            tc_start = pd.Timestamp(train_ci['start'])
            tc_end = pd.Timestamp(train_ci['end'])
            mask = (dates >= tc_start) & (dates <= tc_end)
            y[mask] = 1.0

        # Causal: only train on data before held-out crisis
        cutoff_date = crisis_start - pd.Timedelta(days=window_size)
        fit_end_raw = int(np.searchsorted(dates, cutoff_date))
        if fit_end_raw < 100:
            logger.warning(f"  RF skipping {held_out_key}: insufficient pre-crisis data")
            continue

        rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
        rf.fit_with_labels(X[:fit_end_raw], y[:fit_end_raw])

        scores = rf.compute_regime_scores(X)
        # Trim to enriched length
        rf_scores = scores[19:] if len(scores) > len(dates_enriched) else scores
        if len(rf_scores) > len(dates_enriched):
            rf_scores = rf_scores[:len(dates_enriched)]

        # Compute Cohen's d
        cs = crisis_start - pd.Timedelta(days=window_size)
        ce = crisis_end + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        # Ensure length alignment
        if len(rf_scores) == len(dates_enriched):
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                rf_scores[crisis_mask], rf_scores[normal_mask], n_bootstrap=n_bootstrap,
            )
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan

        rf_results[held_out_key] = {
            'd': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        }
        logger.info(f"    RF on {held_out_key:20s}  d = {d:.3f}" if not np.isnan(d) else
                    f"    RF on {held_out_key:20s}  d = N/A")

    results['Random Forest'] = rf_results

    # ---- Summary statistics ----
    logger.info("\n[4] Computing summary statistics...")

    method_names = list(results.keys())
    n_methods = len(method_names)
    crisis_list = list(crises.keys())
    n_crises = len(crisis_list)

    # Build d-value matrix for Friedman test
    d_matrix = np.full((n_crises, n_methods), np.nan)
    for j, mname in enumerate(method_names):
        for i, ck in enumerate(crisis_list):
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
    logger.info("RESULTS SUMMARY (per-crisis causal preprocessing)")
    logger.info("=" * 70)

    sorted_methods = sorted(
        median_d.items(),
        key=lambda x: x[1] if x[1] is not None else -1,
        reverse=True,
    )
    for rank, (mname, md) in enumerate(sorted_methods, 1):
        if md is not None:
            logger.info(f"  {rank:2d}. {mname:25s}  median d = {md:.3f}")
        else:
            logger.info(f"  {rank:2d}. {mname:25s}  median d = N/A")

    if not np.isnan(chi_sq):
        logger.info(f"\n  Friedman chi-sq = {chi_sq:.2f}, p = {p_val:.4f}")
    else:
        logger.info("  Friedman test: insufficient data")

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'causal': True,
            'causal_method': 'per_crisis_cutoff',
            'window_size': window_size,
            'n_bootstrap': n_bootstrap,
            'quick': quick,
            'full': full,
            'n_crises': n_crises,
            'n_methods': n_methods,
            'hpo_source': 'honest_hpo/hpo_results_20260221_220739.json',
        },
        'hpo_params': {
            name: config['params']
            for name, config in HPO_CONFIGS.items()
        },
        'results': results,
        'summary': {
            'median_d': median_d,
            'friedman_chi_sq': float(chi_sq) if not np.isnan(chi_sq) else None,
            'friedman_p': float(p_val) if not np.isnan(p_val) else None,
            'mean_ranks': (
                {mname: float(mean_ranks[j]) for j, mname in enumerate(method_names)}
                if not np.any(np.isnan(mean_ranks)) else None
            ),
        },
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'causal_comparison_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Regime detection comparison pipeline')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Crisis window extension ± days (default: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Only run on 4 representative crises')
    parser.add_argument('--full', action='store_true',
                        help='Run on all 16 crises (including pre-2005)')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Bootstrap resamples for CIs (default: 10000)')
    args = parser.parse_args()

    run_comparison(
        window_size=args.window_size,
        quick=args.quick,
        full=args.full,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == '__main__':
    main()
