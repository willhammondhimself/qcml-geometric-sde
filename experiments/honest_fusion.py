"""
Fusion of 3 QCML detectors into a single regime score.

Tests max-z, weighted mean, and rank average fusion strategies.
Weights optimized on pre-2020 crises, validated on post-2020.

Usage:
    python experiments/honest_fusion.py
    python experiments/honest_fusion.py --quick
    python experiments/honest_fusion.py --hpo-config experiments/outputs/regime_detection/honest_hpo/hpo_results_*.json
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

TRAIN_CRISES = [k for k in ALL_CRISES if int(k[:4]) < 2020 and int(k[:4]) >= 2005]
VAL_CRISES = [k for k in ALL_CRISES if int(k[:4]) >= 2020]


def get_crisis_scores(scores, dates, crisis_key):
    """Extract crisis and normal scores."""
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def compute_mean_d(scores, dates, crisis_keys, n_bootstrap=500):
    """Compute mean Cohen's d across crises for a score array.

    Args:
        scores: 1-D array of regime scores.
        dates: DatetimeIndex.
        crisis_keys: List of crisis keys.
        n_bootstrap: Bootstrap resamples.

    Returns:
        mean_d: Mean d across crises.
        per_crisis: Dict of per-crisis d values.
    """
    ds = []
    per_crisis = {}
    for ck in crisis_keys:
        try:
            crisis_s, normal_s = get_crisis_scores(scores, dates, ck)
            d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=n_bootstrap)
            if not np.isnan(d):
                ds.append(d)
                per_crisis[ck] = float(d)
        except Exception:
            pass
    return (float(np.mean(ds)) if ds else 0.0), per_crisis


def fuse_max_z(score_arrays):
    """Max-z fusion: take the maximum z-score across methods per timepoint."""
    stacked = np.column_stack(score_arrays)
    return np.nanmax(stacked, axis=1)


def fuse_weighted_mean(score_arrays, weights):
    """Weighted mean fusion."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    stacked = np.column_stack(score_arrays)
    return np.nansum(stacked * weights[None, :], axis=1)


def fuse_rank_average(score_arrays):
    """Rank-average fusion: average the rank of each method's score per timepoint."""
    from scipy.stats import rankdata
    stacked = np.column_stack(score_arrays)
    ranked = np.apply_along_axis(rankdata, 0, stacked)
    return ranked.mean(axis=1)


def z_normalize(scores):
    """Z-normalize scores using expanding window (causal)."""
    T = len(scores)
    z = np.full(T, np.nan)
    for t in range(30, T):
        past = scores[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) > 10:
            mu = np.mean(past_valid)
            sigma = np.std(past_valid)
            if sigma > 1e-12:
                z[t] = (scores[t] - mu) / sigma
    return z


def run_fusion(quick=False, hpo_config_path=None):
    """Run fusion experiment.

    Args:
        quick: Use 4 crises only.
        hpo_config_path: Path to HPO results JSON with optimized params.

    Returns:
        Dict with fusion results.
    """
    logger.info("=" * 70)
    logger.info("HONEST FUSION: Combining 3 QCML Detectors")
    logger.info("=" * 70)

    # Load HPO-optimized params if available
    hpo_params = {}
    if hpo_config_path:
        with open(hpo_config_path) as f:
            hpo_data = json.load(f)
        for det_key, res in hpo_data.get('results', {}).items():
            hpo_params[det_key] = res['best_params']
        logger.info(f"  Loaded HPO params from {hpo_config_path}")

    # Default params (with Berry=random from operator ablation)
    default_params = {
        'berry': dict(hilbert_dim=8, n_pca_components=15, rolling_window=20,
                      operator_method='random', seed=42),
        'qfi': dict(hilbert_dim=8, n_pca_components=15, rolling_window=20,
                     operator_method='pca_inspired', seed=42),
        'mlf': dict(hilbert_dim=8, n_pca_components=15, rolling_window=20,
                     operator_method='pca_inspired', seed=42),
    }

    # Use HPO params where available, fall back to defaults
    params = {k: hpo_params.get(k, default_params[k]) for k in default_params}

    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Features: {X_enriched.shape}")

    # Fit individual detectors
    logger.info("\n[2] Fitting individual detectors...")
    detectors = {
        'berry': ('Berry Phase Rate', BerryPhaseRateDetector(**params['berry'])),
        'qfi': ('QFI Determinant', QFIDeterminantDetector(**params['qfi'])),
        'mlf': ('Multi-Lag Fidelity', MultiLagFidelityDetector(**params['mlf'])),
    }

    raw_scores = {}
    z_scores = {}
    for key, (name, det) in detectors.items():
        logger.info(f"  {name}...")
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        raw_scores[key] = scores
        z_scores[key] = z_normalize(scores)

    all_crisis_keys = TRAIN_CRISES + VAL_CRISES if not quick else ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']

    # Individual detector performance
    logger.info("\n[3] Individual detector performance...")
    individual_results = {}
    for key, (name, _) in detectors.items():
        train_d, _ = compute_mean_d(raw_scores[key], dates_enriched, TRAIN_CRISES)
        val_d, _ = compute_mean_d(raw_scores[key], dates_enriched, VAL_CRISES)
        full_d, full_pc = compute_mean_d(raw_scores[key], dates_enriched, all_crisis_keys)
        individual_results[key] = {
            'name': name, 'train_d': train_d, 'val_d': val_d,
            'full_d': full_d, 'per_crisis': full_pc,
        }
        logger.info(f"  {name:25s}  train d={train_d:.3f}  val d={val_d:.3f}  full d={full_d:.3f}")

    # Fusion strategies
    logger.info("\n[4] Testing fusion strategies...")
    z_arrays = [z_scores['berry'], z_scores['qfi'], z_scores['mlf']]
    raw_arrays = [raw_scores['berry'], raw_scores['qfi'], raw_scores['mlf']]

    fusion_results = {}

    # (a) Max-z
    fused_max = fuse_max_z(z_arrays)
    train_d, _ = compute_mean_d(fused_max, dates_enriched, TRAIN_CRISES)
    val_d, _ = compute_mean_d(fused_max, dates_enriched, VAL_CRISES)
    full_d, full_pc = compute_mean_d(fused_max, dates_enriched, all_crisis_keys)
    fusion_results['max_z'] = {
        'train_d': train_d, 'val_d': val_d, 'full_d': full_d, 'per_crisis': full_pc,
    }
    logger.info(f"  Max-z:            train d={train_d:.3f}  val d={val_d:.3f}  full d={full_d:.3f}")

    # (b) Equal-weight mean
    fused_eq = fuse_weighted_mean(z_arrays, [1/3, 1/3, 1/3])
    train_d, _ = compute_mean_d(fused_eq, dates_enriched, TRAIN_CRISES)
    val_d, _ = compute_mean_d(fused_eq, dates_enriched, VAL_CRISES)
    full_d, full_pc = compute_mean_d(fused_eq, dates_enriched, all_crisis_keys)
    fusion_results['equal_weight'] = {
        'train_d': train_d, 'val_d': val_d, 'full_d': full_d, 'per_crisis': full_pc,
    }
    logger.info(f"  Equal-weight:     train d={train_d:.3f}  val d={val_d:.3f}  full d={full_d:.3f}")

    # (c) Rank average
    fused_rank = fuse_rank_average(raw_arrays)
    train_d, _ = compute_mean_d(fused_rank, dates_enriched, TRAIN_CRISES)
    val_d, _ = compute_mean_d(fused_rank, dates_enriched, VAL_CRISES)
    full_d, full_pc = compute_mean_d(fused_rank, dates_enriched, all_crisis_keys)
    fusion_results['rank_average'] = {
        'train_d': train_d, 'val_d': val_d, 'full_d': full_d, 'per_crisis': full_pc,
    }
    logger.info(f"  Rank average:     train d={train_d:.3f}  val d={val_d:.3f}  full d={full_d:.3f}")

    # (d) Optuna-optimized weights
    logger.info("\n[5] Optimizing fusion weights on training crises...")

    def weight_objective(trial):
        w_berry = trial.suggest_float('w_berry', 0.0, 1.0)
        w_qfi = trial.suggest_float('w_qfi', 0.0, 1.0)
        w_mlf = trial.suggest_float('w_mlf', 0.0, 1.0)
        fused = fuse_weighted_mean(z_arrays, [w_berry, w_qfi, w_mlf])
        d, _ = compute_mean_d(fused, dates_enriched, TRAIN_CRISES)
        return d

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='fusion_weights',
    )
    study.optimize(weight_objective, n_trials=100, show_progress_bar=True)

    best_weights = [
        study.best_params['w_berry'],
        study.best_params['w_qfi'],
        study.best_params['w_mlf'],
    ]
    w_sum = sum(best_weights)
    best_weights_norm = [w / w_sum for w in best_weights]
    logger.info(f"  Best weights (normalized): Berry={best_weights_norm[0]:.3f}, "
                f"QFI={best_weights_norm[1]:.3f}, MLF={best_weights_norm[2]:.3f}")

    fused_opt = fuse_weighted_mean(z_arrays, best_weights)
    train_d, _ = compute_mean_d(fused_opt, dates_enriched, TRAIN_CRISES)
    val_d, _ = compute_mean_d(fused_opt, dates_enriched, VAL_CRISES)
    full_d, full_pc = compute_mean_d(fused_opt, dates_enriched, all_crisis_keys, n_bootstrap=5000)
    fusion_results['optimized_weight'] = {
        'weights': best_weights_norm,
        'train_d': train_d, 'val_d': val_d, 'full_d': full_d, 'per_crisis': full_pc,
    }
    logger.info(f"  Optimized-weight: train d={train_d:.3f}  val d={val_d:.3f}  full d={full_d:.3f}")

    # Save
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'honest_fusion'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'fusion_results_{ts}.json'

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'symbols': symbols,
            'detector_params': {k: v for k, v in params.items()},
            'train_crises': TRAIN_CRISES,
            'val_crises': VAL_CRISES,
        },
        'individual': individual_results,
        'fusion': fusion_results,
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert_numpy)

    logger.info(f"\n{'=' * 70}")
    logger.info("FUSION SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info("  Individual:")
    for key, res in individual_results.items():
        logger.info(f"    {res['name']:25s}  full d={res['full_d']:.3f}")
    logger.info("  Fusion:")
    for fname, res in fusion_results.items():
        logger.info(f"    {fname:25s}  full d={res['full_d']:.3f}  val d={res['val_d']:.3f}")

    logger.info(f"\nSaved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Honest fusion of QCML detectors')
    parser.add_argument('--quick', action='store_true', help='Quick run with 4 crises')
    parser.add_argument('--hpo-config', type=str, help='Path to HPO results JSON')
    args = parser.parse_args()

    run_fusion(quick=args.quick, hpo_config_path=args.hpo_config)


if __name__ == '__main__':
    main()
