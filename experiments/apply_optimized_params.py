#!/usr/bin/env python3
"""
Apply Optimized Multi-Scale Chern Parameters

Helper module to load Optuna optimization results and create
MultiScaleChernDetector with optimal hyperparameters.

Usage:
    from experiments.apply_optimized_params import create_optimized_detector

    detector = create_optimized_detector(
        params_file='experiments/outputs/multiscale_optuna/best_params.json',
        seed=42
    )

Author: QCML Research
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from experiments.additional_detectors import MultiScaleChernDetector

logger = logging.getLogger(__name__)

# Scale preset mapping (must match optuna_multiscale_chern.py)
SCALE_PRESETS = {
    'short_5': [5, 10, 15, 20, 30],
    'default_5': [10, 20, 30, 50, 100],
    'long_5': [30, 50, 75, 100, 150],
    'mixed_5': [10, 25, 50, 100, 200],
    'sparse_3': [20, 60, 120],
    'dense_7': [10, 15, 20, 30, 50, 75, 100],
}


def load_best_params(params_file: str) -> Dict[str, Any]:
    """
    Load best hyperparameters from Optuna results JSON.

    Args:
        params_file: Path to best_params.json from optimization

    Returns:
        Dictionary with best_params and summary metrics

    Raises:
        FileNotFoundError: If params file doesn't exist
        ValueError: If JSON is invalid or missing required fields
    """
    params_path = Path(params_file)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Optimized params not found at {params_file}. "
            "Run optuna_multiscale_chern.py first."
        )

    with open(params_path, 'r') as f:
        data = json.load(f)

    if 'summary' not in data or 'best_params' not in data['summary']:
        raise ValueError(
            f"Invalid params file format. Expected 'summary.best_params' key."
        )

    return data


def optuna_params_to_detector_params(
    optuna_params: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Convert Optuna trial parameters to MultiScaleChernDetector kwargs.

    Args:
        optuna_params: Best params dict from Optuna study
        seed: Random seed for reproducibility

    Returns:
        kwargs dict for MultiScaleChernDetector.__init__()
    """
    # 1. Extract and convert scales
    scales_preset = optuna_params['scales_preset']
    scales = SCALE_PRESETS[scales_preset]

    # 2. Extract per-scale weights and normalize
    n_scales = len(scales)
    weights_raw = [
        optuna_params[f'weight_{i}']
        for i in range(n_scales)
    ]
    weights_sum = sum(weights_raw)
    weights = [w / weights_sum for w in weights_raw]

    # 3. Extract other params
    detector_params = {
        'hilbert_dim': optuna_params['hilbert_dim'],
        'n_pca_components': optuna_params['n_pca_components'],
        'scales': scales,
        'weights': weights,
        'consensus_threshold': optuna_params['consensus_threshold'],
        'normalization_strategy': optuna_params['normalization_strategy'],
        'normalization_window': optuna_params.get('normalization_window'),
        'operator_method': optuna_params['operator_method'],
        'seed': seed,
    }

    return detector_params


def create_optimized_detector(
    params_file: str = 'experiments/outputs/multiscale_optuna/best_params.json',
    seed: int = 42,
    verbose: bool = True
) -> MultiScaleChernDetector:
    """
    Create MultiScaleChernDetector with optimized hyperparameters.

    Args:
        params_file: Path to best_params.json from Optuna optimization
        seed: Random seed for reproducibility
        verbose: Print loaded parameters

    Returns:
        Configured MultiScaleChernDetector instance

    Example:
        >>> detector = create_optimized_detector()
        >>> detector.fit(X_train)
        >>> scores = detector.compute_regime_scores(X_test)
    """
    # Load optimization results
    data = load_best_params(params_file)
    optuna_params = data['summary']['best_params']

    # Convert to detector params
    detector_params = optuna_params_to_detector_params(optuna_params, seed=seed)

    if verbose:
        logger.info("Creating optimized Multi-Scale Chern detector")
        logger.info(f"  Scales: {detector_params['scales']}")
        logger.info(f"  Weights: {[f'{w:.3f}' for w in detector_params['weights']]}")
        logger.info(f"  Consensus threshold: {detector_params['consensus_threshold']:.3f}")
        logger.info(f"  Normalization: {detector_params['normalization_strategy']}")
        if detector_params['normalization_window']:
            logger.info(f"  Normalization window: {detector_params['normalization_window']}")
        logger.info(f"  Hilbert dim: {detector_params['hilbert_dim']}")
        logger.info(f"  PCA components: {detector_params['n_pca_components']}")
        logger.info(f"  Operator method: {detector_params['operator_method']}")

        # Show optimization performance
        if 'best_user_attrs' in data['summary']:
            attrs = data['summary']['best_user_attrs']
            logger.info(f"\nOptimization performance (walk-forward CV):")
            logger.info(f"  Avg F1: {attrs.get('avg_f1', 0):.4f}")
            logger.info(f"  Avg Recall: {attrs.get('avg_recall', 0):.4f}")
            logger.info(f"  Avg Lead Time: {attrs.get('avg_lead_time', 0):.4f}")
            logger.info(f"  Avg FPR: {attrs.get('avg_fpr', 0):.4f}")
            logger.info(f"  Composite objective: {data['summary']['best_value']:.4f}")

    # Create detector
    detector = MultiScaleChernDetector(**detector_params)

    return detector


def create_baseline_detector(seed: int = 42) -> MultiScaleChernDetector:
    """
    Create baseline Multi-Scale Chern detector (equal weights, default params).

    For comparison with optimized version.

    Args:
        seed: Random seed

    Returns:
        Baseline MultiScaleChernDetector with default configuration
    """
    return MultiScaleChernDetector(
        hilbert_dim=8,
        n_pca_components=15,
        scales=[10, 20, 30, 50, 100],
        weights=None,  # Equal weights
        consensus_threshold=0.6,
        normalization_strategy='rolling_adaptive',
        normalization_window=None,
        operator_method='random',
        seed=seed,
    )


def get_optimization_summary(
    params_file: str = 'experiments/outputs/multiscale_optuna/best_params.json'
) -> Dict[str, Any]:
    """
    Get summary statistics from Optuna optimization.

    Args:
        params_file: Path to best_params.json

    Returns:
        Dictionary with n_trials, best_value, metrics, etc.
    """
    data = load_best_params(params_file)
    summary = data['summary']

    result = {
        'n_trials': summary['n_trials_total'],
        'n_completed': summary['n_completed'],
        'n_pruned': summary['n_pruned'],
        'best_value': summary['best_value'],
        'best_trial_number': summary['best_trial_number'],
        'timestamp': data['timestamp'],
    }

    if 'best_user_attrs' in summary:
        result['metrics'] = summary['best_user_attrs']

    return result


if __name__ == "__main__":
    # Demo usage
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Optimized Multi-Scale Chern Detector")
    print("=" * 70)

    try:
        # Show optimization summary
        summary = get_optimization_summary()
        print(f"\nOptimization Study Summary:")
        print(f"  Trials: {summary['n_completed']} completed, {summary['n_pruned']} pruned")
        print(f"  Best composite objective: {summary['best_value']:.4f}")
        print(f"  Best trial: #{summary['best_trial_number']}")

        # Create optimized detector
        print("\n" + "=" * 70)
        detector = create_optimized_detector(verbose=True)

        print("\n" + "=" * 70)
        print("✓ Optimized detector created successfully")
        print("  Use in regime_comparison.py or other experiments")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nRun optimization first:")
        print("  python experiments/optuna_multiscale_chern.py --n-trials 200")
        sys.exit(1)
