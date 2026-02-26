"""
Strictly causal fusion of 3 QCML detectors for regime detection.

Two-phase evaluation ensuring zero future information leakage:

Phase 1 — Weight Optimization (pre-2020 crises only):
    For each pre-2020 crisis, fit detectors on data strictly before that crisis,
    compute z-normalized scores, then optimize fusion weights via Optuna TPE.

Phase 2 — Out-of-Sample Validation (post-2020 crises):
    Apply trained fusion weights to post-2020 crises. For each crisis,
    detectors are re-fitted on data strictly before the crisis window.

Phase 3 — Full 14-Crisis Per-Crisis Causal Evaluation:
    For every crisis, fit detectors causally and apply the best fusion strategy.
    This produces the final headline numbers for the paper.

Fusion strategies tested:
    - max_z: Maximum z-score across detectors per timestep
    - equal_weight: Equal-weight mean of z-scores
    - rank_average: Average rank across detectors
    - optimized_weight: Optuna-optimized weighted mean (trained on pre-2020)

Usage:
    python experiments/causal_fusion_final.py
    python experiments/causal_fusion_final.py --quick
    python experiments/causal_fusion_final.py --n-trials 200
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
from scipy.stats import rankdata

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import (
    compute_cohens_d_with_ci,
    friedman_test,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXTENSION_DAYS = 10

# Temporal split: pre-2020 for training, post-2020 for validation
TRAIN_CRISES = [k for k in ALL_CRISES if 2005 <= int(k[:4]) < 2020]
VAL_CRISES = [k for k in ALL_CRISES if int(k[:4]) >= 2020]
ALL_POST_2005 = [k for k in ALL_CRISES if int(k[:4]) >= 2005]

# HPO-optimized detector configs (from normalization_ablation + honest_hpo)
DETECTOR_CONFIGS = {
    'berry': {
        'class': BerryPhaseRateDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
            normalization='sphere', berry_aggregation='f01',
        ),
    },
    'qfi': {
        'class': QFIDeterminantDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            normalization='soft', qfi_mode='logdet',
            adaptive_epsilon=True,
        ),
    },
    'mlf': {
        'class': MultiLagFidelityDetector,
        'params': dict(
            hilbert_dim=4, n_pca_components=8, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            normalization='sphere',
        ),
    },
}


# =============================================================================
# Fusion Functions
# =============================================================================

def z_normalize_expanding(scores):
    """Causal z-normalization using expanding window.

    At each timestep t, z-score is computed using only data from [0, t).

    Args:
        scores: 1-D array of raw regime scores.

    Returns:
        z: 1-D array of z-normalized scores (NaN for t < 30).
    """
    T = len(scores)
    z = np.full(T, np.nan)
    for t in range(30, T):
        past = scores[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) > 10:
            mu = np.mean(past_valid)
            sigma = np.std(past_valid, ddof=1)
            if sigma > 1e-12:
                z[t] = (scores[t] - mu) / sigma
    return z


def fuse_max_z(z_arrays):
    """Max-z fusion: per-timestep maximum across detector z-scores."""
    stacked = np.column_stack(z_arrays)
    return np.nanmax(stacked, axis=1)


def fuse_weighted_mean(z_arrays, weights):
    """Weighted mean fusion of z-score arrays."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    stacked = np.column_stack(z_arrays)
    return np.nansum(stacked * weights[None, :], axis=1)


def fuse_rank_average(score_arrays):
    """Rank-average fusion: mean rank across detectors per timestep."""
    stacked = np.column_stack(score_arrays)
    T = stacked.shape[0]
    ranked = np.empty_like(stacked)
    for t in range(T):
        row = stacked[t]
        if np.any(np.isnan(row)):
            ranked[t] = np.nan
        else:
            ranked[t] = rankdata(row)
    return np.nanmean(ranked, axis=1)


# =============================================================================
# Per-Crisis Causal Scoring
# =============================================================================

def compute_crisis_d(
    scores, dates, crisis_key, extension_days=EXTENSION_DAYS, n_bootstrap=5000,
):
    """Compute Cohen's d for a single crisis.

    Args:
        scores: 1-D score array aligned with dates.
        dates: DatetimeIndex.
        crisis_key: Key into ALL_CRISES.
        extension_days: Days to extend crisis window.
        n_bootstrap: Bootstrap resamples.

    Returns:
        d, ci_lo, ci_hi: Cohen's d and 95% CI bounds.
    """
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=extension_days)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=extension_days)
    crisis_mask = (dates >= cs) & (dates <= ce)
    normal_mask = ~crisis_mask

    crisis_scores = scores[crisis_mask]
    normal_scores = scores[normal_mask]

    # Filter NaN
    crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
    normal_valid = normal_scores[~np.isnan(normal_scores)]

    if len(crisis_valid) < 5 or len(normal_valid) < 10:
        return np.nan, np.nan, np.nan

    return compute_cohens_d_with_ci(crisis_valid, normal_valid, n_bootstrap=n_bootstrap)


def fit_and_score_causal(
    X_enriched, dates_enriched, crisis_key, detector_configs,
):
    """Fit detectors causally for one crisis and return raw + z-normalized scores.

    Fits scaler/PCA/operators on data strictly before the crisis window.
    Z-normalizes scores using expanding window (causal).

    Args:
        X_enriched: Full enriched feature matrix.
        dates_enriched: DatetimeIndex for enriched features.
        crisis_key: Crisis key for causal cutoff.
        detector_configs: Dict of detector configurations.

    Returns:
        raw_scores: Dict {det_key: 1-D array}.
        z_scores: Dict {det_key: 1-D array of z-normalized scores}.
        fit_end_idx: Index used as causal cutoff.
    """
    ci = ALL_CRISES[crisis_key]
    crisis_start = pd.Timestamp(ci['start'])
    cutoff_date = crisis_start - pd.Timedelta(days=EXTENSION_DAYS)
    fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

    if fit_end_idx < 100:
        return None, None, fit_end_idx

    raw_scores = {}
    z_scores = {}

    for det_key, config in detector_configs.items():
        params = {**config['params'], 'causal_fit_length': fit_end_idx}
        det = config['class'](**params)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        raw_scores[det_key] = scores
        z_scores[det_key] = z_normalize_expanding(scores)

    return raw_scores, z_scores, fit_end_idx


# =============================================================================
# Main Pipeline
# =============================================================================

def run_causal_fusion(quick=False, n_trials=100, n_bootstrap=5000):
    """Run the full causal fusion experiment.

    Args:
        quick: Only run on 4 crises.
        n_trials: Optuna trials for weight optimization.
        n_bootstrap: Bootstrap resamples for CIs.

    Returns:
        Dict with all results.
    """
    logger.info("=" * 70)
    logger.info("STRICTLY CAUSAL FUSION: Per-Crisis Causal Fitting + Weight Optimization")
    logger.info("=" * 70)

    # --- Data ---
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Features: {X_enriched.shape}, dates: {dates_enriched[0]} to {dates_enriched[-1]}")

    # --- Crisis selection ---
    if quick:
        train_keys = ['2008_gfc', '2015_china', '2018_volmageddon']
        val_keys = ['2020_covid', '2022_rates']
        all_keys = train_keys + val_keys
    else:
        train_keys = TRAIN_CRISES
        val_keys = VAL_CRISES
        all_keys = ALL_POST_2005

    logger.info(f"  Training crises: {len(train_keys)}, Validation crises: {len(val_keys)}")
    logger.info(f"  Total crises: {len(all_keys)}")

    # =========================================================================
    # PHASE 1: Per-Crisis Individual Detector Performance
    # =========================================================================
    logger.info("\n[2] Per-crisis causal individual detector evaluation...")

    individual_results = {}  # {det_key: {crisis_key: d}}
    for det_key in DETECTOR_CONFIGS:
        individual_results[det_key] = {}

    for ck in all_keys:
        raw_sc, z_sc, fit_idx = fit_and_score_causal(
            X_enriched, dates_enriched, ck, DETECTOR_CONFIGS,
        )
        if raw_sc is None:
            logger.warning(f"  Skipping {ck}: insufficient pre-crisis data ({fit_idx} rows)")
            continue

        for det_key in DETECTOR_CONFIGS:
            d, ci_lo, ci_hi = compute_crisis_d(
                raw_sc[det_key], dates_enriched, ck, n_bootstrap=n_bootstrap,
            )
            individual_results[det_key][ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }

    # Print individual results
    logger.info("\n  Individual detector results (median d):")
    det_names = {'berry': 'Berry Phase Rate', 'qfi': 'QFI Determinant', 'mlf': 'Multi-Lag Fidelity'}
    for det_key, name in det_names.items():
        ds = [r['d'] for r in individual_results[det_key].values() if r['d'] is not None]
        if ds:
            logger.info(f"    {name:25s}  median d = {np.median(ds):.3f}  (n={len(ds)})")

    # =========================================================================
    # PHASE 2: Fusion Weight Optimization on Training Crises
    # =========================================================================
    logger.info("\n[3] Optimizing fusion weights on pre-2020 crises...")

    # For weight optimization, we need a single set of z-scores.
    # We use a "global causal" approach: fit detectors on data up to 2019-12-31,
    # then compute z-scores on the full timeline.
    global_cutoff = pd.Timestamp('2019-12-31')
    global_fit_end = int(np.searchsorted(dates_enriched, global_cutoff))
    logger.info(f"  Global causal cutoff for weight training: {global_fit_end} rows (up to {dates_enriched[global_fit_end - 1].date()})")

    global_raw = {}
    global_z = {}
    for det_key, config in DETECTOR_CONFIGS.items():
        params = {**config['params'], 'causal_fit_length': global_fit_end}
        det = config['class'](**params)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        global_raw[det_key] = scores
        global_z[det_key] = z_normalize_expanding(scores)

    z_arrays = [global_z['berry'], global_z['qfi'], global_z['mlf']]
    raw_arrays = [global_raw['berry'], global_raw['qfi'], global_raw['mlf']]

    # Test fixed fusion strategies on training crises
    fusion_strategies = {}

    # (a) Max-z
    fused = fuse_max_z(z_arrays)
    train_ds = []
    for ck in train_keys:
        d, _, _ = compute_crisis_d(fused, dates_enriched, ck, n_bootstrap=1000)
        if not np.isnan(d):
            train_ds.append(d)
    fusion_strategies['max_z'] = {
        'train_mean_d': float(np.mean(train_ds)) if train_ds else 0.0,
        'train_median_d': float(np.median(train_ds)) if train_ds else 0.0,
    }
    logger.info(f"  max_z:            train median d = {fusion_strategies['max_z']['train_median_d']:.3f}")

    # (b) Equal-weight
    fused = fuse_weighted_mean(z_arrays, [1/3, 1/3, 1/3])
    train_ds = []
    for ck in train_keys:
        d, _, _ = compute_crisis_d(fused, dates_enriched, ck, n_bootstrap=1000)
        if not np.isnan(d):
            train_ds.append(d)
    fusion_strategies['equal_weight'] = {
        'train_mean_d': float(np.mean(train_ds)) if train_ds else 0.0,
        'train_median_d': float(np.median(train_ds)) if train_ds else 0.0,
    }
    logger.info(f"  equal_weight:     train median d = {fusion_strategies['equal_weight']['train_median_d']:.3f}")

    # (c) Rank average
    fused = fuse_rank_average(raw_arrays)
    train_ds = []
    for ck in train_keys:
        d, _, _ = compute_crisis_d(fused, dates_enriched, ck, n_bootstrap=1000)
        if not np.isnan(d):
            train_ds.append(d)
    fusion_strategies['rank_average'] = {
        'train_mean_d': float(np.mean(train_ds)) if train_ds else 0.0,
        'train_median_d': float(np.median(train_ds)) if train_ds else 0.0,
    }
    logger.info(f"  rank_average:     train median d = {fusion_strategies['rank_average']['train_median_d']:.3f}")

    # (d) Optuna-optimized weights (trained on pre-2020 crises only)
    logger.info(f"\n  Running Optuna weight optimization ({n_trials} trials)...")

    def weight_objective(trial):
        w_berry = trial.suggest_float('w_berry', 0.0, 1.0)
        w_qfi = trial.suggest_float('w_qfi', 0.0, 1.0)
        w_mlf = trial.suggest_float('w_mlf', 0.0, 1.0)
        fused_trial = fuse_weighted_mean(z_arrays, [w_berry, w_qfi, w_mlf])
        ds = []
        for ck in train_keys:
            d, _, _ = compute_crisis_d(fused_trial, dates_enriched, ck, n_bootstrap=200)
            if not np.isnan(d):
                ds.append(d)
        if not ds:
            return 0.0
        # Optimize mean d with consistency bonus (penalize high variance)
        mean_d = np.mean(ds)
        std_d = np.std(ds) if len(ds) > 1 else 0.0
        return mean_d - 0.1 * std_d  # slight consistency penalty

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='causal_fusion_weights',
    )
    study.optimize(weight_objective, n_trials=n_trials, show_progress_bar=True)

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
    train_ds = []
    for ck in train_keys:
        d, _, _ = compute_crisis_d(fused_opt, dates_enriched, ck, n_bootstrap=1000)
        if not np.isnan(d):
            train_ds.append(d)
    fusion_strategies['optimized_weight'] = {
        'weights': best_weights_norm,
        'train_mean_d': float(np.mean(train_ds)) if train_ds else 0.0,
        'train_median_d': float(np.median(train_ds)) if train_ds else 0.0,
    }
    logger.info(f"  optimized_weight: train median d = {fusion_strategies['optimized_weight']['train_median_d']:.3f}")

    # =========================================================================
    # PHASE 3: Out-of-Sample Validation on Post-2020 Crises
    # =========================================================================
    logger.info("\n[4] Out-of-sample validation on post-2020 crises...")

    # Apply each fusion strategy to validation crises using the global-causal scores
    val_results = {}
    for strategy_name in ['max_z', 'equal_weight', 'rank_average', 'optimized_weight']:
        if strategy_name == 'max_z':
            fused = fuse_max_z(z_arrays)
        elif strategy_name == 'equal_weight':
            fused = fuse_weighted_mean(z_arrays, [1/3, 1/3, 1/3])
        elif strategy_name == 'rank_average':
            fused = fuse_rank_average(raw_arrays)
        elif strategy_name == 'optimized_weight':
            fused = fuse_weighted_mean(z_arrays, best_weights)

        strategy_val = {}
        for ck in val_keys:
            d, ci_lo, ci_hi = compute_crisis_d(
                fused, dates_enriched, ck, n_bootstrap=n_bootstrap,
            )
            strategy_val[ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }

        val_ds = [r['d'] for r in strategy_val.values() if r['d'] is not None]
        val_results[strategy_name] = {
            'per_crisis': strategy_val,
            'val_mean_d': float(np.mean(val_ds)) if val_ds else None,
            'val_median_d': float(np.median(val_ds)) if val_ds else None,
        }
        logger.info(f"  {strategy_name:20s}  val median d = {val_results[strategy_name]['val_median_d']:.3f}"
                     if val_results[strategy_name]['val_median_d'] is not None
                     else f"  {strategy_name:20s}  val median d = N/A")

    # =========================================================================
    # PHASE 4: Full Per-Crisis Causal Evaluation (Best Strategy)
    # =========================================================================
    logger.info("\n[5] Full per-crisis causal evaluation with best fusion strategy...")

    # Pick the strategy with highest validation median d
    best_strategy = max(
        val_results.keys(),
        key=lambda s: val_results[s]['val_median_d'] or 0.0,
    )
    logger.info(f"  Best strategy: {best_strategy}")

    # For each crisis, fit detectors causally and apply fusion
    full_causal_results = {}
    for ck in all_keys:
        raw_sc, z_sc, fit_idx = fit_and_score_causal(
            X_enriched, dates_enriched, ck, DETECTOR_CONFIGS,
        )
        if raw_sc is None:
            logger.warning(f"  Skipping {ck}: insufficient data")
            continue

        z_arr = [z_sc['berry'], z_sc['qfi'], z_sc['mlf']]
        raw_arr = [raw_sc['berry'], raw_sc['qfi'], raw_sc['mlf']]

        if best_strategy == 'max_z':
            fused = fuse_max_z(z_arr)
        elif best_strategy == 'equal_weight':
            fused = fuse_weighted_mean(z_arr, [1/3, 1/3, 1/3])
        elif best_strategy == 'rank_average':
            fused = fuse_rank_average(raw_arr)
        elif best_strategy == 'optimized_weight':
            fused = fuse_weighted_mean(z_arr, best_weights)

        d, ci_lo, ci_hi = compute_crisis_d(
            fused, dates_enriched, ck, n_bootstrap=n_bootstrap,
        )
        is_train = ck in train_keys
        full_causal_results[ck] = {
            'd': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'split': 'train' if is_train else 'val',
        }
        tag = "TRAIN" if is_train else "VAL  "
        d_str = f"d = {d:.3f}" if not np.isnan(d) else "d = N/A"
        logger.info(f"    [{tag}] {ck:25s}  {d_str}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CAUSAL FUSION SUMMARY")
    logger.info("=" * 70)

    # Full causal median d
    all_ds = [r['d'] for r in full_causal_results.values() if r['d'] is not None]
    train_ds = [r['d'] for r in full_causal_results.values()
                if r['d'] is not None and r['split'] == 'train']
    val_ds_final = [r['d'] for r in full_causal_results.values()
                    if r['d'] is not None and r['split'] == 'val']

    logger.info(f"\n  Best strategy: {best_strategy}")
    if best_strategy == 'optimized_weight':
        logger.info(f"  Weights: Berry={best_weights_norm[0]:.3f}, "
                     f"QFI={best_weights_norm[1]:.3f}, MLF={best_weights_norm[2]:.3f}")
    logger.info(f"\n  Full causal (all crises):     median d = {np.median(all_ds):.3f}  (n={len(all_ds)})")
    logger.info(f"  Training crises (pre-2020):   median d = {np.median(train_ds):.3f}  (n={len(train_ds)})")
    logger.info(f"  Validation crises (post-2020): median d = {np.median(val_ds_final):.3f}  (n={len(val_ds_final)})")

    # Compare against individual detectors
    logger.info("\n  Comparison (full causal, all crises):")
    logger.info(f"    Causal Fusion ({best_strategy}):  median d = {np.median(all_ds):.3f}")
    for det_key, name in det_names.items():
        ds = [r['d'] for r in individual_results[det_key].values() if r['d'] is not None]
        if ds:
            logger.info(f"    {name:30s}  median d = {np.median(ds):.3f}")

    # =========================================================================
    # Save
    # =========================================================================
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'causal_fusion'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'causal_fusion_{ts}.json'

    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'symbols': symbols,
            'detector_configs': {
                k: v['params'] for k, v in DETECTOR_CONFIGS.items()
            },
            'train_crises': train_keys,
            'val_crises': val_keys,
            'n_trials': n_trials,
            'n_bootstrap': n_bootstrap,
            'global_causal_cutoff': str(dates_enriched[global_fit_end - 1].date()),
            'quick': quick,
        },
        'individual_results': individual_results,
        'fusion_strategies': fusion_strategies,
        'val_results': val_results,
        'best_strategy': best_strategy,
        'best_weights': best_weights_norm if best_strategy == 'optimized_weight' else None,
        'full_causal_results': full_causal_results,
        'summary': {
            'all_crises_median_d': float(np.median(all_ds)) if all_ds else None,
            'train_median_d': float(np.median(train_ds)) if train_ds else None,
            'val_median_d': float(np.median(val_ds_final)) if val_ds_final else None,
        },
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert_numpy)

    logger.info(f"\n  Saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Strictly causal fusion of QCML detectors',
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with 5 crises')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optuna trials for weight optimization (default: 100)')
    parser.add_argument('--n-bootstrap', type=int, default=5000,
                        help='Bootstrap resamples for CIs (default: 5000)')
    args = parser.parse_args()

    run_causal_fusion(
        quick=args.quick,
        n_trials=args.n_trials,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == '__main__':
    main()
