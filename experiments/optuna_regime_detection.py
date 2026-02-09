#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for QCML Regime Detection

Phase 1 of Academic Research Plan: Systematic search for optimal QCML
configuration for crisis detection using walk-forward validation.

Walk-Forward Validation Strategy (no temporal leakage):
  Fold 1: Train pre-2008 -> Test 2008 crisis window
  Fold 2: Train pre-2020 (incl 2008) -> Test 2020 COVID window
  Fold 3: Train pre-2022 (incl 2008, 2020) -> Test 2022 rates window

Composite Objective:
  obj = 0.35 * F1 + 0.25 * recall + 0.20 * norm_lead_time
      + 0.10 * (1 - FPR) + 0.10 * statistical_significance

Usage:
    # Smoke test (10 trials)
    python experiments/optuna_regime_detection.py --n-trials 10

    # Full optimization
    python experiments/optuna_regime_detection.py --n-trials 200

    # Resume from saved study
    python experiments/optuna_regime_detection.py --n-trials 100 --storage sqlite:///regime_optuna.db

Author: QCML Research
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna not installed. Run: pip install optuna>=3.5.0")
    sys.exit(1)

from qcml_geometry import QCMLGeometry
from qcml_geometry.topology import TopologicalRegimeDetector
# Minimal QCMLDataset replacement (original archived in archive/data_pipeline/)
from dataclasses import dataclass as _dataclass
@_dataclass
class QCMLDataset:
    """Minimal dataset wrapper for QCML experiments."""
    features: object
    prices: object
    times: object
    metadata: dict

    @property
    def X(self):
        """Feature matrix as numpy array."""
        return self.features.values if hasattr(self.features, 'values') else self.features

from experiments.crisis_config import (
    CrisisDefinition,
    ValidationConfig,
    ALL_CRISES,
    CRISIS_2008,
    CRISIS_2020,
    CRISIS_2022,
    get_default_validation_config,
)
from experiments.crisis_metrics import (
    compute_statistical_significance,
    compute_precision_recall,
    compute_lead_time,
)

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


@dataclass
class WalkForwardFold:
    """
    Definition of a walk-forward validation fold.

    Attributes:
        name: Fold identifier
        train_crisis_list: Crises included in training (for synthetic data generation)
        test_crisis: The held-out crisis for testing
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date
        test_end: Test period end date
    """
    name: str
    train_crisis_list: List[CrisisDefinition]
    test_crisis: CrisisDefinition
    train_start: str
    train_end: str
    test_start: str
    test_end: str


# Walk-forward fold definitions ensuring no temporal leakage
WALK_FORWARD_FOLDS = [
    WalkForwardFold(
        name="fold_2008",
        train_crisis_list=[],  # No prior crises available
        test_crisis=CRISIS_2008,
        train_start="2005-01-01",
        train_end="2008-03-14",  # 6 months before crisis
        test_start="2008-03-15",
        test_end="2009-03-15",
    ),
    WalkForwardFold(
        name="fold_2020",
        train_crisis_list=[CRISIS_2008],
        test_crisis=CRISIS_2020,
        train_start="2005-01-01",
        train_end="2019-09-15",  # 6 months before crisis
        test_start="2019-09-16",
        test_end="2020-09-16",
    ),
    WalkForwardFold(
        name="fold_2022",
        train_crisis_list=[CRISIS_2008, CRISIS_2020],
        test_crisis=CRISIS_2022,
        train_start="2005-01-01",
        train_end="2021-09-15",  # 6 months before crisis
        test_start="2021-09-16",
        test_end="2022-09-16",
    ),
]


def create_synthetic_fold_data(
    fold: WalkForwardFold,
    n_features: int = 10,
    seed: int = 42
) -> Tuple[QCMLDataset, QCMLDataset]:
    """
    Create synthetic train/test datasets for a fold.

    Training data uses the period before the test crisis. If previous crises
    are included, the training data contains regime changes at those dates.

    Args:
        fold: WalkForwardFold definition
        n_features: Number of features to generate
        seed: Random seed

    Returns:
        train_dataset: Training QCMLDataset
        test_dataset: Test QCMLDataset
    """
    np.random.seed(seed)

    # Generate test data centered on the crisis
    test_dataset = _create_crisis_synthetic(
        fold.test_crisis, n_features, seed=seed + 1
    )

    # Generate training data
    # If we have prior crises, include them in training
    train_dates = pd.bdate_range(start=fold.train_start, end=fold.train_end)
    n_train = len(train_dates)

    # Normal regime training data with some structural variation
    train_features = np.zeros((n_train, n_features))
    for i in range(n_features):
        phase = 2 * np.pi * i / n_features
        theta = np.linspace(0, 4 * np.pi, n_train)
        train_features[:, i] = np.cos(theta + phase) + 0.15 * np.random.randn(n_train)

    # If we have prior crises in training, inject regime changes
    for crisis in fold.train_crisis_list:
        crisis_ts = pd.Timestamp(crisis.crisis_date)
        if crisis_ts >= pd.Timestamp(fold.train_start) and crisis_ts <= pd.Timestamp(fold.train_end):
            crisis_idx = (train_dates <= crisis_ts).sum()
            # Inject regime change at crisis point
            n_after = n_train - crisis_idx
            for i in range(n_features):
                phase = 2 * np.pi * i / n_features
                theta = np.linspace(0, 4 * np.pi, n_after)
                train_features[crisis_idx:, i] = (
                    2.0 * np.sin(2 * theta + phase) + 0.3 * np.random.randn(n_after)
                )

    train_prices = 100 * np.exp(np.cumsum(np.random.randn(n_train) * 0.01))
    train_features_df = pd.DataFrame(
        train_features,
        columns=[f"feature_{i}" for i in range(n_features)],
        index=train_dates
    )
    train_prices_series = pd.Series(train_prices, name='close', index=train_dates)
    train_metadata = {
        'type': 'synthetic_train',
        'fold': fold.name,
        'train_crises': [c.name for c in fold.train_crisis_list],
    }

    train_dataset = QCMLDataset(
        train_features_df, train_prices_series, train_dates, train_metadata
    )

    return train_dataset, test_dataset


def _create_crisis_synthetic(
    crisis: CrisisDefinition,
    n_features: int = 10,
    seed: int = 42
) -> QCMLDataset:
    """Create synthetic dataset with regime change at crisis date."""
    np.random.seed(seed)

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    crisis_idx = (dates <= crisis_date).sum()

    # Pre-crisis: spherical geometry
    theta1 = np.linspace(0, 2 * np.pi, crisis_idx)
    features1 = np.zeros((crisis_idx, n_features))
    for i in range(n_features):
        phase = 2 * np.pi * i / n_features
        features1[:, i] = np.cos(theta1 + phase) + 0.1 * np.random.randn(crisis_idx)

    # Post-crisis: different geometry (torus-like)
    n_after = n_days - crisis_idx
    theta2 = np.linspace(0, 4 * np.pi, n_after)
    features2 = np.zeros((n_after, n_features))
    for i in range(n_features):
        phase = 2 * np.pi * i / n_features
        features2[:, i] = 2.0 * np.sin(2 * theta2 + phase) + 0.3 * np.random.randn(n_after)

    # Correlation structure change
    for i in range(0, n_features - 1, 2):
        features1[:, i + 1] += 0.7 * features1[:, i]
        features2[:, i + 1] -= 0.7 * features2[:, i]

    features = np.vstack([features1, features2])
    features += np.random.randn(n_days, n_features) * 0.1

    # Synthetic prices
    returns = np.zeros(n_days)
    returns[:crisis_idx] = np.random.randn(crisis_idx) * 0.01 + 0.0002
    returns[crisis_idx:] = np.random.randn(n_after) * 0.025 - 0.001
    prices = 100 * np.exp(np.cumsum(returns))

    features_df = pd.DataFrame(
        features,
        columns=[f"feature_{i}" for i in range(n_features)],
        index=dates
    )
    prices_series = pd.Series(prices, name='close', index=dates)
    metadata = {
        'type': 'synthetic_test',
        'crisis': crisis.name,
        'crisis_date': crisis.crisis_date,
        'regime_change_idx': crisis_idx,
    }

    return QCMLDataset(features_df, prices_series, dates, metadata)


def evaluate_config_on_fold(
    config: ValidationConfig,
    fold: WalkForwardFold,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate a configuration on a single walk-forward fold.

    Pipeline:
    1. Create synthetic data for fold
    2. Fit QCML geometry on training data
    3. Compute rolling Chern on test data
    4. Evaluate detection metrics

    Args:
        config: ValidationConfig hyperparameters
        fold: Walk-forward fold definition
        seed: Random seed

    Returns:
        Dictionary with F1, recall, lead_time, FPR, t_statistic, p_value
    """
    # Create data
    train_dataset, test_dataset = create_synthetic_fold_data(
        fold, n_features=10, seed=seed
    )

    # Prepare features
    X_train_raw = train_dataset.X
    X_test_raw = test_dataset.X

    # Standardize using training statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # PCA fit on training, transform test
    n_components = min(config.n_pca_components, X_train_raw.shape[1])
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    # Normalize to unit sphere
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)

    # Fit geometry on training data
    geometry = QCMLGeometry(
        n_features=X_train.shape[1],
        hilbert_dim=config.hilbert_dim
    )
    geometry.fit_operators(X_train, method=config.operator_method)

    # Compute rolling Chern on test data
    detector = TopologicalRegimeDetector(
        geometry=geometry,
        window_size=config.window_size,
        chern_threshold=config.chern_threshold,
        smoothing_window=config.smoothing_window
    )

    chern_series = detector.rolling_chern_number(X_test, window=config.window_size)

    # Align times
    test_times = test_dataset.times[config.window_size - 1:]
    if len(test_times) > len(chern_series):
        test_times = test_times[:len(chern_series)]

    # Find crisis index in test data
    crisis_ts = pd.Timestamp(fold.test_crisis.crisis_date)
    crisis_idx = (test_times >= crisis_ts).argmax()

    if crisis_idx == 0 and test_times[0] >= crisis_ts:
        crisis_idx = 0
    elif crisis_idx == 0:
        # Crisis not in range
        return {
            'f1': 0.0, 'recall': 0.0, 'lead_time_norm': 0.0,
            'fpr': 1.0, 't_statistic': 0.0, 'p_value': 1.0,
        }

    # Statistical significance
    window_days = config.analysis_window_days
    before_start = max(0, crisis_idx - window_days)
    after_end = min(len(chern_series), crisis_idx + window_days)

    chern_before = chern_series[before_start:crisis_idx]
    chern_after = chern_series[crisis_idx:after_end]

    if len(chern_before) < 2 or len(chern_after) < 2:
        return {
            'f1': 0.0, 'recall': 0.0, 'lead_time_norm': 0.0,
            'fpr': 1.0, 't_statistic': 0.0, 'p_value': 1.0,
        }

    significance = compute_statistical_significance(chern_before, chern_after)

    # Detect transitions
    transitions = detector.detect_transitions(X_test, times=np.arange(len(X_test)))
    transition_indices = [t.start_idx for t in transitions]

    # Precision/recall
    true_crisis_idx = crisis_idx + config.window_size - 1
    pr_metrics = compute_precision_recall(
        detected_transitions=transition_indices,
        true_crisis_idx=true_crisis_idx,
        tolerance_days=window_days
    )

    # Lead time
    lead_time = compute_lead_time(
        chern_series=chern_series,
        times=test_times.values,
        crisis_date=fold.test_crisis.crisis_date,
        threshold=config.chern_threshold
    )

    # Normalize lead time: 0 if None, otherwise lead_time / expected_lead_days
    expected = fold.test_crisis.expected_lead_days
    lead_time_norm = min(lead_time / expected, 2.0) if lead_time else 0.0

    # False positive rate
    n_total_transitions = len(transition_indices)
    n_tp = pr_metrics['n_true_positives']
    n_fp = n_total_transitions - n_tp
    total_non_crisis = max(len(chern_series) - 2 * window_days, 1)
    fpr = n_fp / total_non_crisis

    # Statistical significance score (bounded)
    t_stat = min(significance['t_statistic'], 10.0)
    stat_sig = 1.0 if significance['p_value'] < 0.05 else (
        0.5 if significance['p_value'] < 0.10 else 0.0
    )

    return {
        'f1': pr_metrics['f1_score'],
        'recall': pr_metrics['recall'],
        'lead_time_norm': lead_time_norm,
        'fpr': fpr,
        't_statistic': t_stat,
        'p_value': significance['p_value'],
        'effect_size': significance['effect_size'],
        'stat_sig': stat_sig,
    }


def create_objective(
    folds: List[WalkForwardFold],
    seed: int = 42
):
    """
    Create Optuna objective function.

    Composite objective:
      obj = 0.35 * F1 + 0.25 * recall + 0.20 * norm_lead_time
          + 0.10 * (1 - FPR) + 0.10 * stat_significance

    Args:
        folds: Walk-forward folds
        seed: Random seed

    Returns:
        Optuna objective function
    """
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 8, 12, 16, 24])
        window_size = trial.suggest_categorical('window_size', [10, 20, 30, 50, 100])
        chern_threshold = trial.suggest_float('chern_threshold', 0.1, 1.0, log=True)
        smoothing_window = trial.suggest_categorical('smoothing_window', [3, 5, 10])
        n_pca_components = trial.suggest_int('n_pca_components', 5, 10)
        operator_method = trial.suggest_categorical(
            'operator_method', ['random', 'pca_inspired', 'pauli']
        )
        base_threshold_std = trial.suggest_float('base_threshold_std', 2.0, 4.0)
        analysis_window_days = trial.suggest_int('analysis_window_days', 5, 20)

        config = ValidationConfig(
            hilbert_dim=hilbert_dim,
            window_size=window_size,
            chern_threshold=chern_threshold,
            smoothing_window=smoothing_window,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            normalize_features=True,
            analysis_window_days=analysis_window_days,
        )

        # Evaluate on each fold
        fold_metrics = []

        for fold_idx, fold in enumerate(folds):
            try:
                metrics = evaluate_config_on_fold(
                    config, fold, seed=seed + fold_idx
                )
                fold_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Fold {fold.name} failed: {e}")
                fold_metrics.append({
                    'f1': 0.0, 'recall': 0.0, 'lead_time_norm': 0.0,
                    'fpr': 1.0, 'stat_sig': 0.0,
                })

            # Report for pruning
            if fold_metrics:
                avg_f1 = np.mean([m['f1'] for m in fold_metrics])
                trial.report(avg_f1, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not fold_metrics:
            return 0.0

        # Compute composite objective
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])
        avg_recall = np.mean([m['recall'] for m in fold_metrics])
        avg_lead = np.mean([m['lead_time_norm'] for m in fold_metrics])
        avg_fpr = np.mean([m['fpr'] for m in fold_metrics])
        avg_sig = np.mean([m['stat_sig'] for m in fold_metrics])

        composite = (
            0.35 * avg_f1
            + 0.25 * avg_recall
            + 0.20 * avg_lead
            + 0.10 * (1.0 - avg_fpr)
            + 0.10 * avg_sig
        )

        # Store per-fold metrics
        trial.set_user_attr('avg_f1', float(avg_f1))
        trial.set_user_attr('avg_recall', float(avg_recall))
        trial.set_user_attr('avg_lead_time', float(avg_lead))
        trial.set_user_attr('avg_fpr', float(avg_fpr))
        trial.set_user_attr('avg_stat_sig', float(avg_sig))

        for i, (fold, metrics) in enumerate(zip(folds, fold_metrics)):
            trial.set_user_attr(f'{fold.name}_f1', float(metrics['f1']))
            trial.set_user_attr(f'{fold.name}_recall', float(metrics['recall']))

        return composite

    return objective


def run_study(
    n_trials: int = 100,
    storage: Optional[str] = None,
    seed: int = 42,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run the Optuna study.

    Args:
        n_trials: Number of optimization trials
        storage: SQLite storage URL for persistence
        seed: Random seed
        timeout: Timeout in seconds
        n_jobs: Parallel jobs

    Returns:
        study: Completed Optuna study
        summary: Summary dictionary
    """
    seed_everything(seed)

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name="qcml_regime_detection",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = create_objective(WALK_FORWARD_FOLDS, seed=seed)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Summarize
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]

    summary = {
        'n_trials_total': len(study.trials),
        'n_completed': len(completed),
        'n_pruned': len(pruned),
        'best_value': study.best_value if completed else None,
        'best_params': study.best_params if completed else None,
        'best_trial_number': study.best_trial.number if completed else None,
    }

    if completed:
        best = study.best_trial
        summary['best_user_attrs'] = dict(best.user_attrs)

    return study, summary


def save_results(
    study: optuna.Study,
    summary: Dict[str, Any],
    output_dir: str = "experiments/outputs/regime_detection/results"
) -> str:
    """
    Save study results to JSON.

    Args:
        study: Completed study
        summary: Summary dict
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_regime_study_{timestamp}.json"

    # Build output
    output = {
        'timestamp': timestamp,
        'study_name': study.study_name,
        'summary': summary,
        'top_10_trials': [],
    }

    # Top 10 trials
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)

    for trial in sorted_trials[:10]:
        output['top_10_trials'].append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'user_attrs': dict(trial.user_attrs),
        })

    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


def print_results(study: optuna.Study, summary: Dict[str, Any]) -> None:
    """Print formatted study results."""
    print("\n" + "=" * 70)
    print("OPTUNA REGIME DETECTION - RESULTS")
    print("=" * 70)

    print(f"\nTrials: {summary['n_completed']} completed, "
          f"{summary['n_pruned']} pruned, "
          f"{summary['n_trials_total']} total")

    if summary['best_value'] is not None:
        print(f"\nBest composite objective: {summary['best_value']:.4f}")
        print(f"Best trial: #{summary['best_trial_number']}")

        print("\nBest Parameters:")
        for key, value in summary['best_params'].items():
            print(f"  {key}: {value}")

        if 'best_user_attrs' in summary:
            attrs = summary['best_user_attrs']
            print("\nBest Trial Metrics:")
            print(f"  Avg F1:        {attrs.get('avg_f1', 'N/A'):.4f}")
            print(f"  Avg Recall:    {attrs.get('avg_recall', 'N/A'):.4f}")
            print(f"  Avg Lead Time: {attrs.get('avg_lead_time', 'N/A'):.4f}")
            print(f"  Avg FPR:       {attrs.get('avg_fpr', 'N/A'):.4f}")
            print(f"  Avg Stat Sig:  {attrs.get('avg_stat_sig', 'N/A'):.4f}")

            print("\nPer-Fold F1:")
            for fold in WALK_FORWARD_FOLDS:
                f1 = attrs.get(f'{fold.name}_f1', 'N/A')
                recall = attrs.get(f'{fold.name}_recall', 'N/A')
                print(f"  {fold.name}: F1={f1:.4f}, Recall={recall:.4f}")

        # Check success criteria
        best_f1 = summary['best_user_attrs'].get('avg_f1', 0) if 'best_user_attrs' in summary else 0
        if best_f1 > 0.7:
            print(f"\nSUCCESS: F1 > 0.7 achieved ({best_f1:.4f})")
        else:
            print(f"\nF1 target not yet met: {best_f1:.4f} < 0.7")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for QCML Regime Detection"
    )
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials")
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite storage URL (e.g., sqlite:///regime_optuna.db)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/outputs/regime_detection/results",
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 70)
    print("QCML Regime Detection - Optuna Hyperparameter Optimization")
    print("=" * 70)
    print(f"Trials: {args.n_trials}")
    print(f"Folds: {[f.name for f in WALK_FORWARD_FOLDS]}")
    print(f"Storage: {args.storage or 'in-memory'}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    study, summary = run_study(
        n_trials=args.n_trials,
        storage=args.storage,
        seed=args.seed,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )

    print_results(study, summary)

    filepath = save_results(study, summary, output_dir=args.output_dir)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
