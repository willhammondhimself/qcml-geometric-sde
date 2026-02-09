#!/usr/bin/env python3
"""
Fused QCML Optuna Optimization

Phase A: Optimize fusion strategy (max/weighted_mean/rank_mean) and shared
         hyperparameters for FusedQCMLDetector combining Berry Phase Rate,
         QFI Determinant, and Multi-Lag Fidelity.

Phase B: Optimize operator scale factors and fusion weights for
         GeometryOptimizedDetector (single geometry, 5 observables).

Objective: Maximize median Cohen's d across 12-fold leave-one-crisis-out.

Usage:
    python experiments/fused_qcml_optimization.py --phase A --n-trials 200
    python experiments/fused_qcml_optimization.py --phase B --n-trials 300
    python experiments/fused_qcml_optimization.py --phase all --n-trials 200

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    evaluate_method,
    prepare_data,
)
from qcml.regime.fused_detector import FusedQCMLDetector, GeometryOptimizedDetector

load_dotenv(project_root / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / 'experiments' / 'outputs' / 'regime_detection' / 'fused'


def prepare_all_crisis_data(
    enriched_lookback: int = 20,
) -> List[Dict[str, Any]]:
    """Pre-fetch and prepare data for all 12 crises.

    Returns list of dicts with keys: crisis, X, X_enriched, times, crisis_idx,
    times_enriched, crisis_idx_enriched.
    """
    config = get_default_validation_config()
    crisis_data = []

    for crisis in DATA_AVAILABLE_CRISES:
        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is None:
            logger.warning(f"Skipping {crisis.name}: no data")
            continue

        trim = enriched_lookback - 1
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        crisis_data.append({
            'crisis': crisis,
            'X': X,
            'X_enriched': X_enriched,
            'times': times,
            'crisis_idx': crisis_idx,
            'times_enriched': times_enriched,
            'crisis_idx_enriched': crisis_idx_enriched,
        })

    logger.info(f"Prepared data for {len(crisis_data)} crises")
    return crisis_data


def evaluate_detector_across_crises(
    detector_factory,
    crisis_data: List[Dict[str, Any]],
    seed: int = 42,
    n_bootstrap: int = 1000,
    n_permutations: int = 500,
) -> List[float]:
    """Evaluate a detector across all crises, return list of Cohen's d values.

    Args:
        detector_factory: Callable() -> detector instance (fresh for each crisis).
        crisis_data: Pre-fetched crisis data from prepare_all_crisis_data().

    Returns:
        List of Cohen's d (effect_size_d_normalized) values, one per crisis.
    """
    config = get_default_validation_config()
    d_values = []

    for cd in crisis_data:
        try:
            detector = detector_factory()
            detector.fit(cd['X_enriched'])
            result = evaluate_method(
                detector,
                cd['X_enriched'],
                cd['times_enriched'],
                cd['crisis_idx_enriched'],
                cd['crisis'],
                config,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                seed=seed,
            )
            d = result.get('effect_size_d_normalized', 0.0)
            d_values.append(d)
        except Exception as e:
            logger.warning(f"  {cd['crisis'].name} failed: {e}")
            d_values.append(0.0)

    return d_values


# ---------------------------------------------------------------------------
# Phase A: FusedQCMLDetector optimization
# ---------------------------------------------------------------------------

def phase_a_objective(trial: optuna.Trial, crisis_data: List[Dict]) -> float:
    """Optuna objective for Phase A: FusedQCMLDetector."""
    fusion_method = trial.suggest_categorical(
        'fusion_method', ['max', 'weighted_mean', 'rank_mean']
    )

    hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 8, 12])
    n_pca_components = trial.suggest_categorical('n_pca_components', [8, 10, 15])
    operator_method = trial.suggest_categorical(
        'operator_method', ['random', 'pca_inspired']
    )
    min_expanding = trial.suggest_categorical('min_expanding', [40, 60, 80])
    rolling_window = trial.suggest_int('rolling_window', 10, 40, step=5)

    weights = None
    if fusion_method == 'weighted_mean':
        w_berry = trial.suggest_float('w_berry', 0.0, 1.0)
        w_qfi = trial.suggest_float('w_qfi', 0.0, 1.0)
        w_multilag = trial.suggest_float('w_multilag', 0.0, 1.0)
        total = w_berry + w_qfi + w_multilag
        if total < 1e-8:
            return 0.0  # degenerate
        weights = [w_berry / total, w_qfi / total, w_multilag / total]

    def factory():
        return FusedQCMLDetector(
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            seed=42,
            fusion_method=fusion_method,
            weights=weights,
            min_expanding=min_expanding,
            rolling_window=rolling_window,
        )

    d_values = evaluate_detector_across_crises(factory, crisis_data)
    median_d = float(np.median(d_values))

    trial.set_user_attr('d_values', [float(d) for d in d_values])
    trial.set_user_attr('mean_d', float(np.mean(d_values)))
    trial.set_user_attr('median_d', median_d)
    trial.set_user_attr('min_d', float(np.min(d_values)))

    logger.info(
        f"Trial {trial.number}: {fusion_method} | "
        f"h{hilbert_dim} pca{n_pca_components} | "
        f"median_d={median_d:.3f} mean_d={np.mean(d_values):.3f}"
    )

    return median_d


def run_phase_a(n_trials: int = 200, crisis_data: Optional[List] = None):
    """Run Phase A Optuna optimization."""
    print("\n" + "=" * 70)
    print("PHASE A: FusedQCMLDetector Optimization")
    print("=" * 70)

    if crisis_data is None:
        crisis_data = prepare_all_crisis_data()

    study = optuna.create_study(
        direction='maximize',
        study_name='fused_qcml_phase_a',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    start = time.time()
    study.optimize(
        lambda trial: phase_a_objective(trial, crisis_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    best = study.best_trial
    print(f"\nPhase A completed in {elapsed:.0f}s ({n_trials} trials)")
    print(f"Best median d: {best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")
    print(f"Per-crisis d: {best.user_attrs.get('d_values', [])}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'phase': 'A',
        'best_value': best.value,
        'best_params': best.params,
        'best_user_attrs': best.user_attrs,
        'n_trials': n_trials,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
        'top_10_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'd_values': t.user_attrs.get('d_values', []),
            }
            for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)[:10]
        ],
    }

    out_path = OUTPUT_DIR / 'phase_a_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    return study, crisis_data


# ---------------------------------------------------------------------------
# Phase B: GeometryOptimizedDetector optimization
# ---------------------------------------------------------------------------

def phase_b_objective(trial: optuna.Trial, crisis_data: List[Dict]) -> float:
    """Optuna objective for Phase B: GeometryOptimizedDetector."""
    hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 8, 12])
    n_pca = trial.suggest_categorical('n_pca_components', [6, 8, 10])
    rolling_window = trial.suggest_int('rolling_window', 10, 40, step=5)

    # Operator scale factors (log-uniform in [0.1, 5.0])
    n_scales = n_pca
    operator_scales = np.array([
        trial.suggest_float(f'op_scale_{k}', 0.1, 5.0, log=True)
        for k in range(n_scales)
    ])

    # Fusion weights for 5 observables
    raw_weights = np.array([
        trial.suggest_float(f'fw_{k}', 0.0, 1.0)
        for k in range(5)
    ])
    total = raw_weights.sum()
    if total < 1e-8:
        return 0.0
    fusion_weights = raw_weights / total

    def factory():
        return GeometryOptimizedDetector(
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca,
            operator_scales=operator_scales.copy(),
            fusion_weights=fusion_weights.copy(),
            rolling_window=rolling_window,
            min_expanding=60,
            seed=42,
        )

    d_values = evaluate_detector_across_crises(factory, crisis_data)
    median_d = float(np.median(d_values))

    trial.set_user_attr('d_values', [float(d) for d in d_values])
    trial.set_user_attr('mean_d', float(np.mean(d_values)))
    trial.set_user_attr('median_d', median_d)

    logger.info(
        f"Trial {trial.number}: h{hilbert_dim} pca{n_pca} rw{rolling_window} | "
        f"median_d={median_d:.3f} mean_d={np.mean(d_values):.3f}"
    )

    return median_d


def run_phase_b(n_trials: int = 300, crisis_data: Optional[List] = None):
    """Run Phase B Optuna optimization."""
    print("\n" + "=" * 70)
    print("PHASE B: GeometryOptimizedDetector Optimization")
    print("=" * 70)

    if crisis_data is None:
        crisis_data = prepare_all_crisis_data()

    study = optuna.create_study(
        direction='maximize',
        study_name='fused_qcml_phase_b',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    start = time.time()
    study.optimize(
        lambda trial: phase_b_objective(trial, crisis_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    best = study.best_trial
    print(f"\nPhase B completed in {elapsed:.0f}s ({n_trials} trials)")
    print(f"Best median d: {best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")
    print(f"Per-crisis d: {best.user_attrs.get('d_values', [])}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'phase': 'B',
        'best_value': best.value,
        'best_params': best.params,
        'best_user_attrs': best.user_attrs,
        'n_trials': n_trials,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
        'top_10_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'd_values': t.user_attrs.get('d_values', []),
            }
            for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)[:10]
        ],
    }

    out_path = OUTPUT_DIR / 'phase_b_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    return study, crisis_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fused QCML Optuna Optimization')
    parser.add_argument(
        '--phase', choices=['A', 'B', 'all'], default='all',
        help='Which phase to run (A=score fusion, B=operator optimization, all=both)',
    )
    parser.add_argument(
        '--n-trials', type=int, default=200,
        help='Number of Optuna trials per phase',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Pre-fetch data once (shared across phases)
    crisis_data = prepare_all_crisis_data()

    if args.phase in ('A', 'all'):
        n_a = args.n_trials
        run_phase_a(n_trials=n_a, crisis_data=crisis_data)

    if args.phase in ('B', 'all'):
        n_b = args.n_trials if args.phase == 'B' else min(args.n_trials, 300)
        run_phase_b(n_trials=n_b, crisis_data=crisis_data)

    print("\nOptimization complete.")


if __name__ == '__main__':
    main()
