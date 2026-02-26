"""
Optuna hyperparameter optimization for QCML detectors on the honest pipeline.

No lookahead bias: global PCA fit once, pre-2020 crises for training,
post-2020 crises for validation.

Usage:
    python experiments/honest_hpo_sweep.py
    python experiments/honest_hpo_sweep.py --n-trials 50 --quick
    python experiments/honest_hpo_sweep.py --detector berry
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

import optuna
from optuna.samplers import TPESampler

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
    SpectralGapDetector,
    MetricConditionDetector,
    GeometricEnsembleDetector,
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXTENSION_DAYS = 10

# Pre-2020 crises (training) — all post-2005 crises before 2020
TRAIN_CRISES = [k for k in ALL_CRISES if int(k[:4]) < 2020 and int(k[:4]) >= 2005]
# Post-2020 crises (validation)
VAL_CRISES = [k for k in ALL_CRISES if int(k[:4]) >= 2020]

DETECTOR_CLASSES = {
    'berry': ('Berry Phase Rate', BerryPhaseRateDetector),
    'qfi': ('QFI Determinant', QFIDeterminantDetector),
    'mlf': ('Multi-Lag Fidelity', MultiLagFidelityDetector),
    'spectral_gap': ('Spectral Gap', SpectralGapDetector),
    'metric_cond': ('Metric Condition', MetricConditionDetector),
    'geo_ensemble': ('Geometric Ensemble', GeometricEnsembleDetector),
}


def get_crisis_scores(scores, dates, crisis_key):
    """Extract crisis and normal scores."""
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def evaluate_detector(detector_class, params, X_enriched, dates_enriched, crisis_keys):
    """Fit detector and compute mean Cohen's d across given crises.

    Args:
        detector_class: Detector class to instantiate.
        params: Dict of hyperparameters.
        X_enriched: Enriched feature matrix.
        dates_enriched: DatetimeIndex.
        crisis_keys: List of crisis keys to evaluate.

    Returns:
        mean_d: Mean Cohen's d across crises (NaN-safe).
        per_crisis: Dict of per-crisis d values.
    """
    try:
        det = detector_class(**params)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
    except Exception as e:
        logger.debug(f"  Detector failed: {e}")
        return 0.0, {}

    per_crisis = {}
    ds = []
    for ck in crisis_keys:
        try:
            crisis_s, normal_s = get_crisis_scores(scores, dates_enriched, ck)
            d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=500)
            if not np.isnan(d):
                ds.append(d)
                per_crisis[ck] = float(d)
        except Exception:
            pass

    mean_d = float(np.mean(ds)) if ds else 0.0
    return mean_d, per_crisis


CONSISTENCY_PENALTY = 0.3  # alpha: penalizes std(d) across training crises

# Fixed operator methods based on prior analysis (pca_inspired wins for QFI/MLF,
# random marginally better for Berry — fixing these removes one axis of overfitting)
OPERATOR_METHODS = {
    'berry': 'random',
    'qfi': 'pca_inspired',
    'mlf': 'pca_inspired',
    'spectral_gap': 'random',
    'metric_cond': 'random',
    'geo_ensemble': 'random',
}


def create_objective(detector_key, X_enriched, dates_enriched):
    """Create Optuna objective for a specific detector.

    Objective = mean_d - CONSISTENCY_PENALTY * std_d across training crises.
    This rewards params that detect consistently across diverse crisis types,
    not just on average, which improves OOS generalization.

    Args:
        detector_key: One of 'berry', 'qfi', 'mlf'.
        X_enriched: Enriched feature matrix.
        dates_enriched: DatetimeIndex.

    Returns:
        Callable objective function for Optuna.
    """
    _, detector_class = DETECTOR_CLASSES[detector_key]
    operator_method = OPERATOR_METHODS[detector_key]

    def objective(trial):
        # Tighter search space — fewer params = less to overfit
        hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 6, 8])
        n_pca = trial.suggest_categorical('n_pca_components', [5, 8, 10, 15])
        rolling_window = trial.suggest_categorical('rolling_window', [10, 15, 20])

        # Normalization and adaptive epsilon (new axes)
        normalization = trial.suggest_categorical('normalization', ['sphere', 'none', 'soft', 'clip'])
        adaptive_epsilon = (normalization != 'sphere')

        # Validate n_pca < feature dimension to avoid errors
        n_features = X_enriched.shape[1]
        if n_pca > n_features:
            n_pca = n_features

        params = dict(
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca,
            rolling_window=rolling_window,
            operator_method=operator_method,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            seed=42,
        )

        # Detector-specific aggregation modes
        if detector_key == 'berry':
            berry_agg = trial.suggest_categorical('berry_aggregation', ['f01', 'frobenius', 'max'])
            params['berry_aggregation'] = berry_agg
        elif detector_key == 'qfi':
            qfi_mode = trial.suggest_categorical('qfi_mode', ['logdet', 'trace', 'max_eig', 'condition', 'entropy'])
            params['qfi_mode'] = qfi_mode

        mean_d, per_crisis = evaluate_detector(
            detector_class, params, X_enriched, dates_enriched, TRAIN_CRISES,
        )

        # Penalize inconsistency across crises — consistent detection generalizes better
        if len(per_crisis) > 1:
            std_d = float(np.std(list(per_crisis.values())))
            return mean_d - CONSISTENCY_PENALTY * std_d
        return mean_d

    return objective


def run_hpo(n_trials=100, quick=False, detector_filter=None):
    """Run Optuna HPO for all 3 QCML detectors.

    Args:
        n_trials: Number of Optuna trials per detector.
        quick: Use fewer trials (25) and bootstrap (500).
        detector_filter: If set, only optimize this detector ('berry', 'qfi', 'mlf').

    Returns:
        Dict with best params and validation results per detector.
    """
    if quick:
        n_trials = min(n_trials, 25)

    logger.info("=" * 70)
    logger.info("HONEST HPO SWEEP")
    logger.info(f"  Trials per detector: {n_trials}")
    logger.info(f"  Train crises: {TRAIN_CRISES}")
    logger.info(f"  Validation crises: {VAL_CRISES}")
    logger.info("=" * 70)

    # Fetch and prepare data (once)
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Features: {X_enriched.shape}, dates: {dates_enriched[0]} to {dates_enriched[-1]}")

    detectors_to_run = (
        [detector_filter] if detector_filter else list(DETECTOR_CLASSES.keys())
    )

    all_results = {}

    for det_key in detectors_to_run:
        det_name, det_class = DETECTOR_CLASSES[det_key]
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Optimizing: {det_name}")
        logger.info(f"{'=' * 50}")

        objective = create_objective(det_key, X_enriched, dates_enriched)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f'honest_hpo_{det_key}',
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        logger.info(f"\n  Best trial #{best.number}: mean d = {best.value:.4f}")
        logger.info(f"  Best params: {best.params}")

        # Validate on held-out post-2020 crises
        best_params = dict(
            hilbert_dim=best.params['hilbert_dim'],
            n_pca_components=best.params['n_pca_components'],
            rolling_window=best.params['rolling_window'],
            operator_method=OPERATOR_METHODS[det_key],
            normalization=best.params.get('normalization', 'sphere'),
            adaptive_epsilon=(best.params.get('normalization', 'sphere') != 'sphere'),
            seed=42,
        )
        if 'berry_aggregation' in best.params:
            best_params['berry_aggregation'] = best.params['berry_aggregation']
        if 'qfi_mode' in best.params:
            best_params['qfi_mode'] = best.params['qfi_mode']

        val_d, val_per_crisis = evaluate_detector(
            det_class, best_params, X_enriched, dates_enriched, VAL_CRISES,
        )
        logger.info(f"  Validation mean d = {val_d:.4f}")
        logger.info(f"  Validation per-crisis: {val_per_crisis}")

        # Also compute full (all crises) performance
        all_crisis_keys = [k for k in ALL_CRISES if int(k[:4]) >= 2005]
        full_d, full_per_crisis = evaluate_detector(
            det_class, best_params, X_enriched, dates_enriched, all_crisis_keys,
        )
        logger.info(f"  Full (all crises) mean d = {full_d:.4f}")

        # Top 5 trials for reference
        top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]

        all_results[det_key] = {
            'name': det_name,
            'best_params': best_params,
            'train_mean_d': float(best.value),
            'val_mean_d': float(val_d),
            'val_per_crisis': val_per_crisis,
            'full_mean_d': float(full_d),
            'full_per_crisis': full_per_crisis,
            'n_trials': n_trials,
            'top_5': [
                {'trial': t.number, 'value': float(t.value), 'params': t.params}
                for t in top_trials
            ],
        }

    # Save results
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'honest_hpo'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'hpo_results_{ts}.json'

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': n_trials,
            'train_crises': TRAIN_CRISES,
            'val_crises': VAL_CRISES,
            'symbols': symbols,
            'consistency_penalty': CONSISTENCY_PENALTY,
            'operator_methods': OPERATOR_METHODS,
            'search_space': {
                'hilbert_dim': [4, 6, 8],
                'n_pca_components': [5, 8, 10, 15],
                'rolling_window': [10, 15, 20],
                'normalization': ['sphere', 'none', 'soft', 'clip'],
                'berry_aggregation': ['f01', 'frobenius', 'max'],
                'qfi_mode': ['logdet', 'trace', 'max_eig', 'condition', 'entropy'],
            },
        },
        'results': all_results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("HPO SUMMARY")
    logger.info(f"{'=' * 70}")
    for det_key, res in all_results.items():
        logger.info(f"  {res['name']:25s}  train d={res['train_mean_d']:.3f}  "
                     f"val d={res['val_mean_d']:.3f}  full d={res['full_mean_d']:.3f}")
        logger.info(f"    Best: {res['best_params']}")

    logger.info(f"\nSaved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Honest HPO sweep for QCML detectors')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optuna trials per detector (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with 25 trials')
    parser.add_argument('--detector',
                        choices=['berry', 'qfi', 'mlf', 'spectral_gap', 'metric_cond', 'geo_ensemble'],
                        help='Only optimize a single detector')
    args = parser.parse_args()

    run_hpo(
        n_trials=args.n_trials,
        quick=args.quick,
        detector_filter=args.detector,
    )


if __name__ == '__main__':
    main()
