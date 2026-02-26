"""
Nested Leave-One-Crisis-Out HPO: Gold-standard unbiased evaluation.

For each held-out crisis k:
  1. Inner HPO: Run Optuna on the remaining N-1 crises (train/val split within)
  2. Evaluate: Compute Cohen's d on held-out crisis k using inner-selected config

This eliminates HPO leakage: no crisis ever sees the config that was tuned on it.

Optionally includes baseline HPO (CUSUM, HMM, BOCPD, IsolationForest) with
the same Optuna budget per fold, ensuring symmetric evaluation.

Usage:
    python experiments/nested_loco_hpo.py
    python experiments/nested_loco_hpo.py --n-trials 50 --quick
    python experiments/nested_loco_hpo.py --include-baselines
    python experiments/nested_loco_hpo.py --full  # all 16 crises
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
)
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.baselines import (
    CUSUMDetector,
    HMMRegimeDetector,
    BOCPDDetector,
    IsolationForestDetector,
)
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
CONSISTENCY_PENALTY = 0.3

# Fixed operator methods (from prior ablation)
OPERATOR_METHODS = {
    'berry': 'random',
    'qfi': 'pca_inspired',
    'mlf': 'pca_inspired',
}

QCML_DETECTORS = {
    'berry': ('Berry Phase Rate', BerryPhaseRateDetector),
    'qfi': ('QFI Determinant', QFIDeterminantDetector),
    'mlf': ('Multi-Lag Fidelity', MultiLagFidelityDetector),
}

BASELINE_DETECTORS = {
    'cusum': ('CUSUM', CUSUMDetector),
    'hmm': ('HMM 2-state', HMMRegimeDetector),
    'bocpd': ('BOCPD', BOCPDDetector),
    'isoforest': ('Isolation Forest', IsolationForestDetector),
}


def get_crisis_scores(scores, dates, crisis_key):
    """Extract crisis and normal scores for a given crisis."""
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def evaluate_detector_on_crises(detector_class, params, X_enriched, dates_enriched,
                                crisis_keys, n_bootstrap=500):
    """Fit detector and compute per-crisis Cohen's d."""
    try:
        det = detector_class(**params)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
    except Exception as e:
        logger.debug(f"  Detector failed: {e}")
        return {}, 0.0

    per_crisis = {}
    ds = []
    for ck in crisis_keys:
        try:
            crisis_s, normal_s = get_crisis_scores(scores, dates_enriched, ck)
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=n_bootstrap)
            if not np.isnan(d):
                ds.append(d)
                per_crisis[ck] = {'d': float(d), 'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi)}
        except Exception:
            pass

    mean_d = float(np.mean(ds)) if ds else 0.0
    return per_crisis, mean_d


def create_qcml_objective(detector_key, X_enriched, dates_enriched, train_keys):
    """Create Optuna objective for a QCML detector on given training crises."""
    _, detector_class = QCML_DETECTORS[detector_key]
    operator_method = OPERATOR_METHODS[detector_key]

    def objective(trial):
        hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 6, 8])
        n_pca = trial.suggest_categorical('n_pca_components', [5, 8, 10, 15])
        rolling_window = trial.suggest_categorical('rolling_window', [10, 15, 20])
        normalization = trial.suggest_categorical('normalization', ['sphere', 'none', 'soft', 'clip'])
        adaptive_epsilon = (normalization != 'sphere')

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

        if detector_key == 'berry':
            berry_agg = trial.suggest_categorical('berry_aggregation', ['f01', 'frobenius', 'max'])
            params['berry_aggregation'] = berry_agg
        elif detector_key == 'qfi':
            qfi_mode = trial.suggest_categorical('qfi_mode', ['logdet', 'trace', 'max_eig', 'condition', 'entropy'])
            params['qfi_mode'] = qfi_mode

        per_crisis, mean_d = evaluate_detector_on_crises(
            detector_class, params, X_enriched, dates_enriched, train_keys,
        )

        if len(per_crisis) > 1:
            std_d = float(np.std([v['d'] for v in per_crisis.values()]))
            return mean_d - CONSISTENCY_PENALTY * std_d
        return mean_d

    return objective


def create_baseline_objective(detector_key, X_enriched, dates_enriched, train_keys):
    """Create Optuna objective for a baseline detector."""
    _, detector_class = BASELINE_DETECTORS[detector_key]

    def objective(trial):
        if detector_key == 'cusum':
            params = dict(
                k=trial.suggest_float('k', 0.1, 2.0),
                burn_in=trial.suggest_int('burn_in', 20, 120),
            )
        elif detector_key == 'hmm':
            params = dict(
                n_states=trial.suggest_categorical('n_states', [2, 3]),
                n_iter=trial.suggest_categorical('n_iter', [50, 100, 200]),
                seed=42,
            )
        elif detector_key == 'bocpd':
            params = dict(
                hazard_rate=trial.suggest_float('hazard_rate', 50.0, 1000.0, log=True),
                min_expanding=trial.suggest_int('min_expanding', 10, 60),
            )
        elif detector_key == 'isoforest':
            params = dict(
                n_estimators=trial.suggest_categorical('n_estimators', [50, 100, 200]),
                contamination=trial.suggest_float('contamination', 0.01, 0.15),
                seed=42,
            )
        else:
            return 0.0

        per_crisis, mean_d = evaluate_detector_on_crises(
            detector_class, params, X_enriched, dates_enriched, train_keys,
        )

        if len(per_crisis) > 1:
            std_d = float(np.std([v['d'] for v in per_crisis.values()]))
            return mean_d - CONSISTENCY_PENALTY * std_d
        return mean_d

    return objective


def extract_best_params(study, detector_key, is_baseline=False):
    """Extract best params from Optuna study."""
    best = study.best_trial.params

    if is_baseline:
        if detector_key == 'cusum':
            return dict(k=best['k'], burn_in=best['burn_in'])
        elif detector_key == 'hmm':
            return dict(n_states=best['n_states'], n_iter=best['n_iter'], seed=42)
        elif detector_key == 'bocpd':
            return dict(hazard_rate=best['hazard_rate'], min_expanding=best['min_expanding'])
        elif detector_key == 'isoforest':
            return dict(n_estimators=best['n_estimators'],
                        contamination=best['contamination'], seed=42)
    else:
        params = dict(
            hilbert_dim=best['hilbert_dim'],
            n_pca_components=best['n_pca_components'],
            rolling_window=best['rolling_window'],
            operator_method=OPERATOR_METHODS[detector_key],
            normalization=best.get('normalization', 'sphere'),
            adaptive_epsilon=(best.get('normalization', 'sphere') != 'sphere'),
            seed=42,
        )
        if 'berry_aggregation' in best:
            params['berry_aggregation'] = best['berry_aggregation']
        if 'qfi_mode' in best:
            params['qfi_mode'] = best['qfi_mode']
        return params


def run_nested_loco(n_trials=100, quick=False, include_baselines=False,
                    full=False, n_bootstrap=5000):
    """Run nested leave-one-crisis-out HPO.

    Args:
        n_trials: Optuna trials per inner fold per detector.
        quick: Reduce trials and bootstrap.
        include_baselines: Also HPO-tune CUSUM, HMM, BOCPD, IsoForest.
        full: Use all 16 crises (including pre-2005).
        n_bootstrap: Bootstrap resamples for final CIs.
    """
    if quick:
        n_trials = min(n_trials, 25)
        n_bootstrap = 500

    # Crisis selection
    if full:
        crisis_keys = list(ALL_CRISES.keys())
    else:
        crisis_keys = [k for k in ALL_CRISES if int(k[:4]) >= 2005]

    n_crises = len(crisis_keys)

    logger.info("=" * 70)
    logger.info("NESTED LEAVE-ONE-CRISIS-OUT HPO")
    logger.info(f"  Crises: {n_crises}")
    logger.info(f"  Trials per inner fold: {n_trials}")
    logger.info(f"  Include baselines: {include_baselines}")
    logger.info(f"  Bootstrap: {n_bootstrap}")
    logger.info("=" * 70)

    # Fetch and prepare data
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    start_date = '1993-01-01' if full else '2005-01-01'
    raw = fetch_data(symbols, start_date, '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Features: {X_enriched.shape}, dates: {dates_enriched[0]} to {dates_enriched[-1]}")

    # Build detector list
    detectors = dict(QCML_DETECTORS)
    if include_baselines:
        detectors.update(BASELINE_DETECTORS)

    # Nested LOCO loop
    all_results = {}

    for det_key in detectors:
        det_name = detectors[det_key][0]
        det_class = detectors[det_key][1]
        is_baseline = det_key in BASELINE_DETECTORS

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Detector: {det_name} ({'baseline' if is_baseline else 'QCML'})")
        logger.info(f"{'=' * 60}")

        fold_results = {}
        inner_configs = {}

        for fold_idx, held_out_key in enumerate(crisis_keys):
            inner_keys = [k for k in crisis_keys if k != held_out_key]

            logger.info(f"\n  Fold {fold_idx + 1}/{n_crises}: held-out = {held_out_key}")
            logger.info(f"    Inner crises: {len(inner_keys)}")

            # Inner HPO on N-1 crises
            if is_baseline:
                obj = create_baseline_objective(det_key, X_enriched, dates_enriched, inner_keys)
            else:
                obj = create_qcml_objective(det_key, X_enriched, dates_enriched, inner_keys)

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42 + fold_idx),
                study_name=f'nested_loco_{det_key}_fold{fold_idx}',
            )
            study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

            best_params = extract_best_params(study, det_key, is_baseline)
            inner_configs[held_out_key] = best_params
            logger.info(f"    Best inner objective: {study.best_value:.4f}")
            logger.info(f"    Best params: {best_params}")

            # Evaluate on held-out crisis
            per_crisis, _ = evaluate_detector_on_crises(
                det_class, best_params, X_enriched, dates_enriched,
                [held_out_key], n_bootstrap=n_bootstrap,
            )

            if held_out_key in per_crisis:
                fold_results[held_out_key] = per_crisis[held_out_key]
                logger.info(f"    Held-out d = {per_crisis[held_out_key]['d']:.4f} "
                            f"[{per_crisis[held_out_key]['ci_lo']:.3f}, "
                            f"{per_crisis[held_out_key]['ci_hi']:.3f}]")
            else:
                fold_results[held_out_key] = {'d': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan')}
                logger.warning(f"    Held-out crisis {held_out_key}: no valid d")

        # Summary statistics
        ds = [v['d'] for v in fold_results.values() if not np.isnan(v['d'])]
        median_d = float(np.median(ds)) if ds else float('nan')
        mean_d = float(np.mean(ds)) if ds else float('nan')
        std_d = float(np.std(ds, ddof=1)) if len(ds) > 1 else float('nan')

        logger.info(f"\n  {'─' * 40}")
        logger.info(f"  {det_name} NESTED LOCO SUMMARY:")
        logger.info(f"    Median d = {median_d:.4f}")
        logger.info(f"    Mean d = {mean_d:.4f} +/- {std_d:.4f}")
        logger.info(f"    N valid = {len(ds)} / {n_crises}")
        logger.info(f"  {'─' * 40}")

        all_results[det_key] = {
            'name': det_name,
            'type': 'baseline' if is_baseline else 'qcml',
            'per_crisis': fold_results,
            'inner_configs': {k: _serialize(v) for k, v in inner_configs.items()},
            'summary': {
                'median_d': median_d,
                'mean_d': mean_d,
                'std_d': std_d,
                'n_valid': len(ds),
                'n_crises': n_crises,
            },
        }

    # Save results
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'nested_loco'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'nested_loco_{ts}.json'

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': n_trials,
            'n_crises': n_crises,
            'crisis_keys': crisis_keys,
            'include_baselines': include_baselines,
            'full': full,
            'n_bootstrap': n_bootstrap,
            'consistency_penalty': CONSISTENCY_PENALTY,
        },
        'results': all_results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {out_path}")

    # Print comparison table
    logger.info(f"\n{'=' * 70}")
    logger.info("NESTED LOCO-CV COMPARISON TABLE")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Method':<25} {'Type':<10} {'Median d':>10} {'Mean d':>10} {'Std d':>8}")
    logger.info(f"{'─' * 63}")
    for det_key in sorted(all_results, key=lambda k: -all_results[k]['summary']['median_d']):
        r = all_results[det_key]
        s = r['summary']
        logger.info(f"{r['name']:<25} {r['type']:<10} {s['median_d']:>10.4f} {s['mean_d']:>10.4f} {s['std_d']:>8.4f}")

    return output


def _serialize(obj):
    """Make params JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(
        description='Nested Leave-One-Crisis-Out HPO (gold-standard unbiased evaluation)'
    )
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optuna trials per inner fold (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 25 trials, 500 bootstrap')
    parser.add_argument('--include-baselines', action='store_true',
                        help='Also HPO-tune CUSUM, HMM, BOCPD, IsoForest')
    parser.add_argument('--full', action='store_true',
                        help='All 16 crises (including pre-2005)')
    parser.add_argument('--n-bootstrap', type=int, default=5000,
                        help='Bootstrap resamples for CIs (default: 5000)')
    args = parser.parse_args()

    run_nested_loco(
        n_trials=args.n_trials,
        quick=args.quick,
        include_baselines=args.include_baselines,
        full=args.full,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == '__main__':
    main()
