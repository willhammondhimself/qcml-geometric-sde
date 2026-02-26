"""
Walk-forward evaluation with nested Optuna HPO at each window.

At each expanding window (2005→eval_year-1), re-optimizes hyperparameters
using ONLY crises available up to that point, then evaluates on the next year.
This eliminates ALL hyperparameter look-ahead bias.

Estimated runtime: ~6.6 hours for 100 trials × 3 detectors × 14 windows.

Usage:
    caffeinate -i python experiments/walk_forward_hpo.py
    caffeinate -i python experiments/walk_forward_hpo.py --n-trials 50
    caffeinate -i python experiments/walk_forward_hpo.py --quick
"""

import argparse
import json
import logging
import sys
import time
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
CONSISTENCY_PENALTY = 0.3

DETECTOR_CLASSES = {
    'berry': ('Berry Phase Rate', BerryPhaseRateDetector),
    'qfi': ('QFI Determinant', QFIDeterminantDetector),
    'mlf': ('Multi-Lag Fidelity', MultiLagFidelityDetector),
}

OPERATOR_METHODS = {
    'berry': 'random',
    'qfi': 'pca_inspired',
    'mlf': 'pca_inspired',
}


def get_crisis_scores(scores, dates, crisis_key):
    """Extract crisis and normal scores for a given crisis."""
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def evaluate_detector(detector_class, params, X_enriched, dates_enriched, crisis_keys):
    """Fit detector and compute mean Cohen's d across given crises."""
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


def create_hpo_objective(detector_key, detector_class, X_enriched, dates_enriched, train_crises):
    """Create Optuna objective for a detector at a specific walk-forward window.

    Args:
        detector_key: One of 'berry', 'qfi', 'mlf'.
        detector_class: Detector class to instantiate.
        X_enriched: Enriched feature matrix (training period only).
        dates_enriched: DatetimeIndex for training period.
        train_crises: Crisis keys available for training at this window.

    Returns:
        Callable objective function for Optuna.
    """
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

        mean_d, per_crisis = evaluate_detector(
            detector_class, params, X_enriched, dates_enriched, train_crises,
        )

        if len(per_crisis) > 1:
            std_d = float(np.std(list(per_crisis.values())))
            return mean_d - CONSISTENCY_PENALTY * std_d
        return mean_d

    return objective


def extract_best_params(detector_key, best_trial):
    """Extract full parameter dict from Optuna best trial."""
    params = dict(
        hilbert_dim=best_trial.params['hilbert_dim'],
        n_pca_components=best_trial.params['n_pca_components'],
        rolling_window=best_trial.params['rolling_window'],
        operator_method=OPERATOR_METHODS[detector_key],
        normalization=best_trial.params.get('normalization', 'sphere'),
        adaptive_epsilon=(best_trial.params.get('normalization', 'sphere') != 'sphere'),
        seed=42,
    )
    if 'berry_aggregation' in best_trial.params:
        params['berry_aggregation'] = best_trial.params['berry_aggregation']
    if 'qfi_mode' in best_trial.params:
        params['qfi_mode'] = best_trial.params['qfi_mode']
    return params


def find_training_crises(eval_year):
    """Return crisis keys whose windows END before eval_year starts.

    Only includes post-2005 crises (our data starts 2005).
    """
    cutoff = pd.Timestamp(f'{eval_year}-01-01')
    matching = []
    for ck, ci in ALL_CRISES.items():
        year = int(ck[:4])
        if year < 2005:
            continue
        ce = pd.Timestamp(ci['end'])
        if ce < cutoff:
            matching.append(ck)
    return matching


def find_crises_in_year(year):
    """Return crisis keys whose windows overlap the given year."""
    start = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp(f'{year}-12-31')
    matching = []
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        if cs <= end and ce >= start:
            matching.append(ck)
    return matching


def run_walk_forward_hpo(n_trials=100, quick=False):
    """Run walk-forward evaluation with nested HPO at each window.

    For each expanding window:
      1. Identify training crises (ended before eval year)
      2. Run Optuna HPO on training crises (n_trials per detector)
      3. Fit detector with best HP on training data
      4. Evaluate on eval year crises

    Args:
        n_trials: Optuna trials per detector per window.
        quick: Use 3 windows and 25 trials.
    """
    if quick:
        n_trials = min(n_trials, 25)

    logger.info("=" * 70)
    logger.info("WALK-FORWARD HPO (Nested Optimization)")
    logger.info(f"  Trials per detector per window: {n_trials}")
    logger.info(f"  Consistency penalty: {CONSISTENCY_PENALTY}")
    logger.info("=" * 70)

    # Fetch full data range once
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_full, dates_full = create_feature_matrix(prices_df)
    logger.info(f"  Full feature matrix: {X_full.shape}")
    logger.info(f"  Date range: {dates_full[0]} to {dates_full[-1]}")

    # Walk-forward windows
    if quick:
        eval_years = [2010, 2015, 2020]
    else:
        eval_years = list(range(2010, 2024))

    # Pre-compute which crises are available at each window
    for yr in eval_years:
        tc = find_training_crises(yr)
        ec = find_crises_in_year(yr)
        logger.info(f"  {yr}: train_crises={len(tc)} {tc}, eval_crises={ec or 'none'}")

    all_window_results = []
    all_hp_choices = {}  # detector -> list of (year, params)
    total_start = time.time()

    for window_idx, eval_year in enumerate(eval_years):
        window_start = time.time()
        logger.info(f"\n{'=' * 70}")
        logger.info(f"WINDOW {window_idx + 1}/{len(eval_years)}: "
                     f"Train 2005-{eval_year - 1}, Eval {eval_year}")
        logger.info(f"{'=' * 70}")

        # Slice data: training = 2005 to eval_year-1
        train_end = pd.Timestamp(f'{eval_year - 1}-12-31')
        eval_start = pd.Timestamp(f'{eval_year}-01-01')
        eval_end = pd.Timestamp(f'{eval_year}-12-31')

        train_mask = dates_full <= train_end
        eval_mask = (dates_full >= eval_start) & (dates_full <= eval_end)
        window_mask = dates_full <= eval_end

        X_train = X_full[train_mask]
        X_window = X_full[window_mask]
        dates_train = dates_full[train_mask]
        dates_window = dates_full[window_mask]
        dates_eval = dates_full[eval_mask]

        if len(X_train) < 100:
            logger.info(f"  Skipping (insufficient training data: {len(X_train)})")
            continue

        # Build enriched features for training and full window
        X_train_enriched = BaseRegimeDetector.build_enriched_features(X_train, lookback=20)
        dates_train_enriched = dates_train[19:]
        X_window_enriched = BaseRegimeDetector.build_enriched_features(X_window, lookback=20)
        dates_window_enriched = dates_window[19:]

        # Where eval period starts in the window-enriched array
        eval_start_idx = np.searchsorted(dates_window_enriched, eval_start)

        # Identify crises
        train_crises = find_training_crises(eval_year)
        eval_crises = find_crises_in_year(eval_year)

        logger.info(f"  Training crises ({len(train_crises)}): {train_crises}")
        logger.info(f"  Eval crises: {eval_crises or 'none'}")
        logger.info(f"  Training features: {X_train_enriched.shape}")

        if len(train_crises) < 2:
            logger.info(f"  Skipping HPO (need >=2 training crises, have {len(train_crises)})")
            continue

        window_result = {
            'eval_year': eval_year,
            'n_train_crises': len(train_crises),
            'train_crises': train_crises,
            'eval_crises': eval_crises,
            'detectors': {},
        }

        # Run HPO for each detector
        for det_key in DETECTOR_CLASSES:
            det_name, det_class = DETECTOR_CLASSES[det_key]
            hpo_start = time.time()
            logger.info(f"\n  --- HPO: {det_name} ({n_trials} trials) ---")

            # Create objective using ONLY training data and training crises
            objective = create_hpo_objective(
                det_key, det_class, X_train_enriched, dates_train_enriched, train_crises,
            )

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42 + window_idx),
                study_name=f'wf_hpo_{det_key}_{eval_year}',
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best = study.best_trial
            best_params = extract_best_params(det_key, best)
            hpo_elapsed = time.time() - hpo_start

            logger.info(f"    Best trial #{best.number}: obj={best.value:.4f} "
                         f"({hpo_elapsed:.1f}s)")
            logger.info(f"    Params: {best.params}")

            # Track HP choices across windows
            if det_key not in all_hp_choices:
                all_hp_choices[det_key] = []
            all_hp_choices[det_key].append({
                'year': eval_year,
                'params': best.params,
                'train_obj': float(best.value),
            })

            # Now evaluate on the eval year using best HP
            # Fit on training data, score on full window, extract eval portion
            eval_d_values = {}
            try:
                det = det_class(**best_params)
                det.fit(X_train_enriched)
                scores = det.compute_regime_scores(X_window_enriched)
                eval_scores = scores[eval_start_idx:]

                for ck in eval_crises:
                    ci = ALL_CRISES[ck]
                    cs = pd.Timestamp(ci['start'])
                    ce = pd.Timestamp(ci['end'])

                    crisis_mask = (dates_window_enriched[eval_start_idx:] >= cs) & \
                                  (dates_window_enriched[eval_start_idx:] <= ce)
                    normal_mask = ~crisis_mask

                    crisis_scores = eval_scores[crisis_mask]
                    normal_scores = eval_scores[normal_mask]

                    if len(crisis_scores) >= 5 and len(normal_scores) >= 10:
                        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                            crisis_scores, normal_scores, n_bootstrap=5000
                        )
                        if not np.isnan(d):
                            eval_d_values[ck] = {
                                'd': float(d),
                                'ci_lo': float(ci_lo),
                                'ci_hi': float(ci_hi),
                            }
                            logger.info(f"    {ck}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

            except Exception as e:
                logger.warning(f"    Eval failed: {e}")

            # Also compute training performance with best HP for reference
            train_d, train_per_crisis = evaluate_detector(
                det_class, best_params, X_train_enriched, dates_train_enriched, train_crises,
            )

            window_result['detectors'][det_key] = {
                'name': det_name,
                'best_params': best_params,
                'train_objective': float(best.value),
                'train_mean_d': float(train_d),
                'train_per_crisis': train_per_crisis,
                'eval_per_crisis': eval_d_values,
                'eval_mean_d': float(np.mean([v['d'] for v in eval_d_values.values()])) if eval_d_values else None,
                'hpo_seconds': round(hpo_elapsed, 1),
                'n_trials': n_trials,
            }

        window_elapsed = time.time() - window_start
        logger.info(f"\n  Window {eval_year} complete in {window_elapsed / 60:.1f} min")
        all_window_results.append(window_result)

        # Save intermediate results after each window (crash recovery)
        _save_results(all_window_results, all_hp_choices, n_trials, symbols,
                      eval_years, intermediate=True)

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TOTAL TIME: {total_elapsed / 3600:.1f} hours")
    logger.info(f"{'=' * 70}")

    # Final summary
    _print_summary(all_window_results, all_hp_choices)

    # Save final results
    out_path = _save_results(all_window_results, all_hp_choices, n_trials, symbols,
                             eval_years, intermediate=False)
    return out_path


def _print_summary(all_window_results, all_hp_choices):
    """Print summary of walk-forward HPO results."""
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD HPO SUMMARY")
    logger.info("=" * 70)

    # Collect all eval d-values per detector
    for det_key in DETECTOR_CLASSES:
        det_name = DETECTOR_CLASSES[det_key][0]
        all_eval_ds = []
        per_crisis_ds = {}

        for wr in all_window_results:
            if det_key in wr.get('detectors', {}):
                dr = wr['detectors'][det_key]
                for ck, v in dr.get('eval_per_crisis', {}).items():
                    all_eval_ds.append(v['d'])
                    if ck not in per_crisis_ds:
                        per_crisis_ds[ck] = []
                    per_crisis_ds[ck].append(v['d'])

        if all_eval_ds:
            median_d = np.median(all_eval_ds)
            mean_d = np.mean(all_eval_ds)
            logger.info(f"\n  {det_name}:")
            logger.info(f"    OOS eval: median d={median_d:.3f}, mean d={mean_d:.3f}, "
                         f"n={len(all_eval_ds)} crisis-windows")
            for ck in sorted(per_crisis_ds.keys()):
                ds = per_crisis_ds[ck]
                logger.info(f"      {ck}: d={np.mean(ds):.3f} (n={len(ds)})")
        else:
            logger.info(f"\n  {det_name}: No eval crises encountered")

    # HP stability analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("HP STABILITY ACROSS WINDOWS")
    logger.info("=" * 70)
    for det_key, choices in all_hp_choices.items():
        det_name = DETECTOR_CLASSES[det_key][0]
        logger.info(f"\n  {det_name}:")
        if not choices:
            continue

        # Track how often each categorical param value was chosen
        param_counts = {}
        for c in choices:
            for k, v in c['params'].items():
                if k not in param_counts:
                    param_counts[k] = {}
                v_str = str(v)
                param_counts[k][v_str] = param_counts[k].get(v_str, 0) + 1

        for param_name, counts in param_counts.items():
            total = sum(counts.values())
            most_common = max(counts, key=counts.get)
            pct = counts[most_common] / total * 100
            logger.info(f"    {param_name}: most_common={most_common} ({pct:.0f}%), "
                         f"dist={dict(sorted(counts.items(), key=lambda x: -x[1]))}")


def _save_results(all_window_results, all_hp_choices, n_trials, symbols,
                  eval_years, intermediate=False):
    """Save results to JSON."""
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'walk_forward_hpo'
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = '_intermediate' if intermediate else ''
    out_path = out_dir / f'wf_hpo_{ts}{suffix}.json'

    # Compute aggregate stats
    aggregate = {}
    for det_key in DETECTOR_CLASSES:
        det_name = DETECTOR_CLASSES[det_key][0]
        all_eval_ds = []
        all_train_ds = []
        for wr in all_window_results:
            if det_key in wr.get('detectors', {}):
                dr = wr['detectors'][det_key]
                for v in dr.get('eval_per_crisis', {}).values():
                    all_eval_ds.append(v['d'])
                if dr.get('train_mean_d') is not None:
                    all_train_ds.append(dr['train_mean_d'])

        aggregate[det_key] = {
            'name': det_name,
            'n_eval_observations': len(all_eval_ds),
            'eval_median_d': float(np.median(all_eval_ds)) if all_eval_ds else None,
            'eval_mean_d': float(np.mean(all_eval_ds)) if all_eval_ds else None,
            'eval_std_d': float(np.std(all_eval_ds)) if all_eval_ds else None,
            'train_median_d': float(np.median(all_train_ds)) if all_train_ds else None,
            'train_mean_d': float(np.mean(all_train_ds)) if all_train_ds else None,
            'overfitting_gap': (
                float(np.median(all_train_ds)) - float(np.median(all_eval_ds))
                if all_train_ds and all_eval_ds else None
            ),
        }

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials_per_detector_per_window': n_trials,
            'consistency_penalty': CONSISTENCY_PENALTY,
            'symbols': symbols,
            'eval_years': eval_years,
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
        'aggregate': aggregate,
        'hp_choices': all_hp_choices,
        'windows': all_window_results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    if not intermediate:
        logger.info(f"\nResults saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward evaluation with nested Optuna HPO',
    )
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optuna trials per detector per window (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: 3 windows, 25 trials')
    args = parser.parse_args()

    run_walk_forward_hpo(n_trials=args.n_trials, quick=args.quick)


if __name__ == '__main__':
    main()
