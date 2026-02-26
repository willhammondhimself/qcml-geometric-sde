"""
Operator construction ablation: random vs PCA-inspired vs learned-scaling.

Tests whether operator choice matters and whether simple learned scaling
improves beyond PCA-inspired defaults.

Usage:
    python experiments/operator_ablation.py
    python experiments/operator_ablation.py --quick
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry import (
    QCMLGeometry,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

EXTENSION_DAYS = 10


def get_crisis_scores(scores, dates, crisis_key):
    """Extract crisis and normal scores for a given crisis.

    Args:
        scores: 1-D array of regime scores.
        dates: DatetimeIndex aligned with scores.
        crisis_key: Key into ALL_CRISES dict.

    Returns:
        (crisis_scores, normal_scores): Arrays of scores in/out of crisis window.
    """
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def evaluate_condition(detector_class, common_params, X_enriched, dates_enriched,
                       crisis_keys, operator_method='pca_inspired',
                       operator_scales=None):
    """Run one condition: fit, score, compute d for each crisis.

    Args:
        detector_class: One of BerryPhaseRateDetector, QFIDeterminantDetector,
            MultiLagFidelityDetector.
        common_params: Dict of shared hyperparameters.
        X_enriched: Enriched feature matrix (T, d).
        dates_enriched: DatetimeIndex aligned with X_enriched.
        crisis_keys: List of crisis keys to evaluate.
        operator_method: 'random' or 'pca_inspired'.
        operator_scales: Optional array of per-operator scalar weights.

    Returns:
        Dict mapping crisis_key -> {d, ci_lo, ci_hi}.
    """
    det = detector_class(**common_params, operator_method=operator_method)
    det.fit(X_enriched)

    # Apply learned scaling if provided
    if operator_scales is not None and det._geometry is not None:
        for i, op in enumerate(det._geometry.operators):
            if i < len(operator_scales):
                det._geometry.operators[i] = op * operator_scales[i]
        # Clear ground state cache after modifying operators
        det._geometry._ground_state_cache = {}

    scores = det.compute_regime_scores(X_enriched)

    results = {}
    for ck in crisis_keys:
        crisis_s, normal_s = get_crisis_scores(scores, dates_enriched, ck)
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=5000)
        results[ck] = {'d': float(d), 'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi)}

    return results


def learn_operator_scales(detector_class, X_enriched, dates_enriched,
                          train_crisis_keys, common_params):
    """Learn per-operator scalar weights to maximize mean d on training crises.

    Optimizes 3 scalar weights (for the top 3 PCA-inspired operators) via
    Nelder-Mead on the negative mean Cohen's d across training crises.

    Args:
        detector_class: Detector class to optimize.
        X_enriched: Enriched feature matrix.
        dates_enriched: DatetimeIndex aligned with X_enriched.
        train_crisis_keys: Crisis keys used for optimization.
        common_params: Shared hyperparameters.

    Returns:
        optimal_scales: Array of 3 learned scale factors.
    """
    def objective(log_scales):
        scales = np.exp(log_scales)  # Ensure positive
        try:
            det = detector_class(**common_params, operator_method='pca_inspired')
            det.fit(X_enriched)

            # Scale operators
            for i, op in enumerate(det._geometry.operators):
                if i < len(scales):
                    det._geometry.operators[i] = op * scales[i]
            det._geometry._ground_state_cache = {}

            scores = det.compute_regime_scores(X_enriched)

            ds = []
            for ck in train_crisis_keys:
                crisis_s, normal_s = get_crisis_scores(scores, dates_enriched, ck)
                d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=500)
                if not np.isnan(d):
                    ds.append(d)

            return -np.mean(ds) if ds else 0.0
        except Exception as e:
            logger.warning(f"  Optimization error: {e}")
            return 0.0

    # Only optimize first 3 scales (most impactful PCA dims)
    x0 = np.zeros(3)  # log-scale, so 0 = scale of 1.0
    bounds = [(-2.3, 2.3)] * 3  # exp(-2.3)=0.1, exp(2.3)=10

    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': 50, 'xatol': 0.1, 'fatol': 0.01},
    )

    optimal_scales = np.exp(result.x)
    logger.info(f"  Learned scales: {optimal_scales}")
    return optimal_scales


def run_ablation(quick=False):
    """Run operator ablation experiment.

    Compares 3 operator conditions x 3 methods x 12 crises:
    1. Random operators (operator_method='random')
    2. PCA-inspired operators (operator_method='pca_inspired')
    3. Learned scaling (PCA-inspired + optimized per-operator scalar weights)

    Args:
        quick: If True, use only 4 representative crises.

    Returns:
        Dict of all results keyed by "method|condition|crisis".
    """
    logger.info("=" * 60)
    logger.info("OPERATOR ABLATION: Random vs PCA-Inspired vs Learned Scaling")
    logger.info("=" * 60)

    # Fetch data
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_full, dates_full = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X_full, lookback=20)
    dates_enriched = dates_full[19:]

    common_params = dict(
        hilbert_dim=8, n_pca_components=15, rolling_window=20, seed=42,
    )

    if quick:
        crisis_keys = ['2008_gfc', '2010_flash', '2020_covid', '2023_svb']
    else:
        crisis_keys = [k for k in ALL_CRISES.keys() if k != '2024_carry']

    # Training crises for learned scaling (pre-2020)
    train_crises = [k for k in crisis_keys if int(k[:4]) < 2020]

    methods = [
        ('Berry Phase Rate', BerryPhaseRateDetector),
        ('QFI Determinant', QFIDeterminantDetector),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector),
    ]

    all_results = {}

    for method_name, detector_class in methods:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Method: {method_name}")
        logger.info(f"{'=' * 40}")

        # Condition 1: Random operators
        logger.info("\n  [1] Random operators")
        r1 = evaluate_condition(
            detector_class, common_params,
            X_enriched, dates_enriched, crisis_keys,
            operator_method='random',
        )

        # Condition 2: PCA-inspired operators
        logger.info("\n  [2] PCA-inspired operators")
        r2 = evaluate_condition(
            detector_class, common_params,
            X_enriched, dates_enriched, crisis_keys,
            operator_method='pca_inspired',
        )

        # Condition 3: Learned scaling
        logger.info("\n  [3] Learned scaling (optimizing on pre-2020 crises)")
        scales = learn_operator_scales(
            detector_class, X_enriched, dates_enriched,
            train_crises, common_params,
        )
        r3 = evaluate_condition(
            detector_class, common_params,
            X_enriched, dates_enriched, crisis_keys,
            operator_method='pca_inspired',
            operator_scales=scales,
        )

        for ck in crisis_keys:
            all_results[f"{method_name}|random|{ck}"] = {
                'method': method_name, 'condition': 'random',
                'crisis': ck, **r1.get(ck, {}),
            }
            all_results[f"{method_name}|pca_inspired|{ck}"] = {
                'method': method_name, 'condition': 'pca_inspired',
                'crisis': ck, **r2.get(ck, {}),
            }
            all_results[f"{method_name}|learned_scaling|{ck}"] = {
                'method': method_name, 'condition': 'learned_scaling',
                'crisis': ck, **r3.get(ck, {}),
            }

        # Summary per method
        for cond_name, r in [('random', r1), ('pca_inspired', r2), ('learned_scaling', r3)]:
            ds = [v['d'] for v in r.values() if not np.isnan(v.get('d', float('nan')))]
            logger.info(
                f"  {cond_name:20s}: mean d = {np.mean(ds):.3f}, "
                f"median d = {np.median(ds):.3f}"
            )

    # Save
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'operator_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': ts,
        'config': common_params,
        'crisis_keys': crisis_keys,
        'results': all_results,
    }
    out_path = out_dir / f'operator_ablation_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Operator construction ablation: random vs PCA-inspired vs learned-scaling."
    )
    parser.add_argument('--quick', action='store_true',
                        help="Use 4 representative crises instead of all 11.")
    args = parser.parse_args()
    run_ablation(quick=args.quick)


if __name__ == '__main__':
    main()
