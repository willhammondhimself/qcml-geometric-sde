#!/usr/bin/env python3
"""
Optuna-Based Causal Regime Detection Optimization

Intelligent hyperparameter optimization with causal constraint (no lookahead bias).
Replaces exhaustive grid search from causal_sensitivity_analysis.py with TPE sampling.

Phase A: Optimize base hyperparameters (hilbert_dim, n_pca_components,
         operator_method, rolling_window) under causal constraint.
         Search space: 4 × 4 × 2 × 3 = 96 combinations (but Optuna explores intelligently).

Phase B (Optional): Optimize expanding window parameters on top of Phase A results.

Objective: Maximize median Cohen's d across crises (more robust than mean).

Usage:
    # Quick test (4 crises, 50 trials)
    python experiments/optuna_causal_regime.py --phase A --n-trials 50 --crises quick

    # Full Phase A (12 crises, 200 trials, ~45-60 min)
    python experiments/optuna_causal_regime.py --phase A --n-trials 200 --crises all --storage sqlite:///optuna_causal.db

    # Full Phase B (150 trials, ~30 min)
    python experiments/optuna_causal_regime.py --phase B --n-trials 150 --crises all --storage sqlite:///optuna_causal.db

    # Both phases
    python experiments/optuna_causal_regime.py --phase all --n-trials 200

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
    ValidationConfig,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    evaluate_method,
    prepare_data,
    seed_everything,
)
from qcml.regime.classical_baselines import (
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
)

load_dotenv(project_root / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

# Representative crisis subset for quick mode
STAGE1_CRISES = [
    "2008_crisis",
    "2018_volmageddon",
    "2020_covid",
    "2023_svb",
]

# Method factories
METHOD_FACTORIES = {
    "Berry Phase Rate": BerryPhaseRateDetector,
    "QFI Determinant": QFIDeterminantDetector,
    "Multi-Lag Fidelity": MultiLagFidelityDetector,
}

OUTPUT_DIR = project_root / 'experiments' / 'outputs' / 'regime_detection' / 'optuna_causal'


def _preload_crisis_data(
    crisis_names: List[str],
    config: ValidationConfig,
    enriched_lookback: int = 20,
) -> Dict[str, Dict]:
    """Pre-load and cache crisis data for the specified crises."""
    all_crises = DATA_AVAILABLE_CRISES
    crisis_map = {c.name: c for c in all_crises}
    crisis_data = {}

    for name in crisis_names:
        if name not in crisis_map:
            logger.warning(f"Crisis {name} not in DATA_AVAILABLE_CRISES, skipping")
            continue
        crisis = crisis_map[name]
        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is not None:
            trim = enriched_lookback - 1
            times_enriched = times[trim:]
            crisis_idx_enriched = max(0, crisis_idx - trim)
            crisis_data[name] = {
                'crisis': crisis,
                'X': X,
                'X_enriched': X_enriched,
                'times': times,
                'crisis_idx': crisis_idx,
                'times_enriched': times_enriched,
                'crisis_idx_enriched': crisis_idx_enriched,
            }
            logger.info(f"  Loaded {name}: X_enriched.shape={X_enriched.shape}, crisis_idx_enriched={crisis_idx_enriched}")

    logger.info(f"Cached {len(crisis_data)} crises")
    return crisis_data


def define_search_space_phase_a(trial: optuna.Trial, method_name: str) -> Dict:
    """Define Phase A search space for base hyperparameters.

    Search space mirrors causal_sensitivity_analysis.py grid:
    - hilbert_dim: [4, 8, 12, 16]
    - n_pca_components: [8, 10, 15, 20]
    - operator_method: ['random', 'pca_inspired']
    - rolling_window: [10, 20, 30]
    - min_expanding: [40, 60, 80] (for context, not used without expanding_refit_interval)
    """
    return {
        'hilbert_dim': trial.suggest_categorical('hilbert_dim', [4, 8, 12, 16]),
        'n_pca_components': trial.suggest_categorical('n_pca_components', [8, 10, 15, 20]),
        'operator_method': trial.suggest_categorical('operator_method', ['random', 'pca_inspired']),
        'rolling_window': trial.suggest_int('rolling_window', 10, 30, step=10),
        'min_expanding': trial.suggest_categorical('min_expanding', [40, 60, 80]),
    }


def define_search_space_phase_b(trial: optuna.Trial, base_params: Dict) -> Dict:
    """Define Phase B search space for expanding window optimization.

    Refines base params from Phase A and optimizes expanding_refit_interval.
    """
    params = base_params.copy()

    # Optimize expanding window interval
    params['expanding_refit_interval'] = trial.suggest_int('expanding_refit_interval', 10, 50, step=5)

    # Optionally refine base params with narrower ranges around Phase A optimum
    # (For now, keep base params fixed from Phase A for clearer A/B comparison)

    return params


def objective_phase_a(
    trial: optuna.Trial,
    method_name: str,
    DetectorClass,
    crisis_data: Dict,
    config: ValidationConfig,
) -> float:
    """Objective: maximize median Cohen's d across crises with causal constraint.

    Args:
        trial: Optuna trial object.
        method_name: Method name (for logging).
        DetectorClass: Detector class to instantiate.
        crisis_data: Pre-loaded crisis data dict.
        config: Validation configuration.

    Returns:
        Median Cohen's d across crises (higher is better).
    """
    params = define_search_space_phase_a(trial, method_name)

    d_values = []
    for crisis_name, cd in crisis_data.items():
        try:
            # KEY: causal_fit_length enforces no lookahead
            # PCA/scaler/operators are fit only on pre-crisis data
            detector = DetectorClass(
                **params,
                seed=42,
                causal_fit_length=cd['crisis_idx_enriched'],  # Causal constraint
            )
            detector.fit(cd['X_enriched'])

            result = evaluate_method(
                detector,
                cd['X_enriched'],
                cd['times_enriched'],
                cd['crisis_idx_enriched'],
                cd['crisis'],
                config,
                n_bootstrap=100,  # Quick for optimization (full=1000 in final eval)
                n_permutations=100,  # Quick for optimization (full=500 in final eval)
                seed=42,
            )
            d = result.get('effect_size_d_normalized', 0.0)
            d_values.append(d)

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed on {crisis_name}: {e}")
            d_values.append(0.0)

    # Objective: median d across crises (more robust than mean)
    median_d = float(np.median(d_values)) if d_values else 0.0

    # Store detailed metrics for analysis
    trial.set_user_attr('d_values', [float(v) for v in d_values])
    trial.set_user_attr('mean_d', float(np.mean(d_values)))
    trial.set_user_attr('std_d', float(np.std(d_values)))
    trial.set_user_attr('min_d', float(np.min(d_values)))
    trial.set_user_attr('max_d', float(np.max(d_values)))

    return median_d


def objective_phase_b(
    trial: optuna.Trial,
    method_name: str,
    DetectorClass,
    crisis_data: Dict,
    config: ValidationConfig,
    base_params: Dict,
) -> float:
    """Objective: maximize median Cohen's d with expanding window optimization.

    Args:
        trial: Optuna trial object.
        method_name: Method name (for logging).
        DetectorClass: Detector class to instantiate.
        crisis_data: Pre-loaded crisis data dict.
        config: Validation configuration.
        base_params: Base hyperparameters from Phase A.

    Returns:
        Median Cohen's d across crises (higher is better).
    """
    params = define_search_space_phase_b(trial, base_params)

    d_values = []
    for crisis_name, cd in crisis_data.items():
        try:
            # Causal constraint + expanding window
            detector = DetectorClass(
                **params,
                seed=42,
                causal_fit_length=cd['crisis_idx_enriched'],
            )
            detector.fit(cd['X_enriched'])

            result = evaluate_method(
                detector,
                cd['X_enriched'],
                cd['times_enriched'],
                cd['crisis_idx_enriched'],
                cd['crisis'],
                config,
                n_bootstrap=100,
                n_permutations=100,
                seed=42,
            )
            d = result.get('effect_size_d_normalized', 0.0)
            d_values.append(d)

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed on {crisis_name}: {e}")
            d_values.append(0.0)

    median_d = float(np.median(d_values)) if d_values else 0.0

    trial.set_user_attr('d_values', [float(v) for v in d_values])
    trial.set_user_attr('mean_d', float(np.mean(d_values)))
    trial.set_user_attr('std_d', float(np.std(d_values)))
    trial.set_user_attr('min_d', float(np.min(d_values)))
    trial.set_user_attr('max_d', float(np.max(d_values)))

    return median_d


def optimize_method(
    method_name: str,
    DetectorClass,
    crisis_data: Dict,
    config: ValidationConfig,
    phase: str,
    n_trials: int,
    storage: Optional[str],
    base_params: Optional[Dict] = None,
) -> optuna.Study:
    """Run Optuna optimization for one method.

    Args:
        method_name: Method name.
        DetectorClass: Detector class to optimize.
        crisis_data: Pre-loaded crisis data.
        config: Validation configuration.
        phase: 'A' or 'B'.
        n_trials: Number of Optuna trials.
        storage: SQLite storage URL for resume capability (optional).
        base_params: Base hyperparameters from Phase A (required for Phase B).

    Returns:
        Optuna study object.
    """
    study_name = f"causal_{method_name.replace(' ', '_').lower()}_phase_{phase.lower()}"

    sampler = optuna.samplers.TPESampler(seed=42)
    # No pruner for regime detection (evaluation is already fast enough)

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    logger.info(f"=== Optimizing {method_name} Phase {phase} ===")
    logger.info(f"Study: {study_name}")
    logger.info(f"Crises: {len(crisis_data)}")
    logger.info(f"Trials: {n_trials}")

    if phase == 'A':
        objective_fn = lambda trial: objective_phase_a(
            trial, method_name, DetectorClass, crisis_data, config
        )
    else:  # Phase B
        if base_params is None:
            raise ValueError("Phase B requires base_params from Phase A")
        objective_fn = lambda trial: objective_phase_b(
            trial, method_name, DetectorClass, crisis_data, config, base_params
        )

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(f"Best median_d: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study


def export_results(study: optuna.Study, method_name: str, phase: str, output_dir: Path) -> Path:
    """Export Optuna results in format compatible with evaluation script.

    Args:
        study: Optuna study object.
        method_name: Method name.
        phase: 'A' or 'B'.
        output_dir: Output directory.

    Returns:
        Path to exported JSON file.
    """
    best_trial = study.best_trial

    results = {
        'timestamp': datetime.now().isoformat(),
        'method': method_name,
        'phase': phase,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': best_trial.number,
        'user_attrs': dict(best_trial.user_attrs),
        'top_10_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'user_attrs': dict(t.user_attrs),
            }
            for t in sorted(
                study.trials,
                key=lambda t: t.value if t.value is not None else -np.inf,
                reverse=True
            )[:10]
        ],
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method_name.replace(' ', '_').lower()}_phase_{phase.lower()}_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results exported to: {output_path}")
    return output_path


def load_phase_a_params(method_name: str, output_dir: Path) -> Dict:
    """Load best parameters from Phase A results.

    Args:
        method_name: Method name.
        output_dir: Output directory containing Phase A results.

    Returns:
        Best parameters from Phase A.
    """
    phase_a_file = output_dir / f"{method_name.replace(' ', '_').lower()}_phase_a_results.json"

    if not phase_a_file.exists():
        raise FileNotFoundError(f"Phase A results not found: {phase_a_file}")

    with open(phase_a_file) as f:
        phase_a_data = json.load(f)

    return phase_a_data['best_params']


def main():
    parser = argparse.ArgumentParser(
        description="Optuna causal regime detection optimization"
    )
    parser.add_argument(
        '--phase',
        choices=['A', 'B', 'all'],
        default='A',
        help='Optimization phase: A (base params), B (expanding window), or all',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=200,
        help='Number of Optuna trials (default: 200)',
    )
    parser.add_argument(
        '--crises',
        choices=['quick', 'all'],
        default='quick',
        help='quick: 4 representative crises, all: 12 crises',
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='SQLite storage for resume: sqlite:///optuna_causal.db',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['Berry', 'QFI', 'MultiLag', 'all'],
        default=['all'],
        help='Methods to optimize (default: all)',
    )
    args = parser.parse_args()

    seed_everything(42)
    config = get_default_validation_config()

    # Load crisis data (once, cached)
    crisis_names = STAGE1_CRISES if args.crises == 'quick' else [c.name for c in DATA_AVAILABLE_CRISES]
    logger.info(f"Loading {len(crisis_names)} crises...")
    crisis_data = _preload_crisis_data(crisis_names, config)

    # Determine which methods to run
    if args.methods == ['all']:
        methods_to_run = METHOD_FACTORIES.items()
    else:
        method_map = {
            'Berry': 'Berry Phase Rate',
            'QFI': 'QFI Determinant',
            'MultiLag': 'Multi-Lag Fidelity',
        }
        methods_to_run = [
            (method_map[m], METHOD_FACTORIES[method_map[m]])
            for m in args.methods if m != 'all'
        ]

    # Run optimization for each method
    for method_name, DetectorClass in methods_to_run:
        logger.info(f"\n{'='*80}")
        logger.info(f"Method: {method_name}")
        logger.info(f"{'='*80}")

        # Phase A
        if args.phase in ['A', 'all']:
            study_a = optimize_method(
                method_name,
                DetectorClass,
                crisis_data,
                config,
                phase='A',
                n_trials=args.n_trials,
                storage=args.storage,
            )
            export_results(study_a, method_name, 'A', OUTPUT_DIR)

        # Phase B
        if args.phase in ['B', 'all']:
            # Load best params from Phase A
            if args.phase == 'all':
                base_params = study_a.best_params
            else:
                base_params = load_phase_a_params(method_name, OUTPUT_DIR)

            logger.info(f"\nPhase B base params: {base_params}")

            study_b = optimize_method(
                method_name,
                DetectorClass,
                crisis_data,
                config,
                phase='B',
                n_trials=args.n_trials // 2,  # Fewer trials for Phase B
                storage=args.storage,
                base_params=base_params,
            )
            export_results(study_b, method_name, 'B', OUTPUT_DIR)

    logger.info(f"\n{'='*80}")
    logger.info("Optimization complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
