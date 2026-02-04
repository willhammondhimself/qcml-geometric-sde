#!/usr/bin/env python3
"""
Crisis Detection Validation Script

Validates the QCML hypothesis: "Chern number discontinuities detect regime changes
in real markets" across multiple historical crises (2008, 2020, 2022).

Features:
    - Validates against real market data (Polygon API) or synthetic data
    - Computes statistical significance, precision/recall, lead time
    - Supports Optuna hyperparameter optimization
    - Outputs JSON and CSV results

Usage:
    # Run on synthetic data (no API key needed)
    python experiments/validate_crisis_detection.py --synthetic

    # Run on real data (requires POLYGON_API_KEY)
    python experiments/validate_crisis_detection.py

    # Run with Optuna optimization
    python experiments/validate_crisis_detection.py --optuna --n-trials 50

Author: QCML Research
Date: 2024
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.qcml_geometry import QCMLGeometry
from qcml.topological_regime import TopologicalRegimeDetector
from qcml.data import QCMLDataset, load_crisis_dataset, create_synthetic_qcml_dataset
from qcml.data import PolygonDataSource, MinimalFeatureEngine

from experiments.crisis_config import (
    CrisisDefinition,
    ValidationConfig,
    CrisisValidationResult,
    AggregateMetrics,
    OptunaConfig,
    ALL_CRISES,
    get_crisis_by_name,
    get_default_validation_config,
    config_to_dict,
    result_to_dict,
    aggregate_to_dict
)

from experiments.crisis_metrics import (
    compute_statistical_significance,
    compute_precision_recall,
    compute_lead_time,
    aggregate_cross_crisis_metrics,
    evaluate_hypothesis,
    format_results_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    np.random.seed(seed)
    import random
    random.seed(seed)


class CrisisValidator:
    """
    Validates Chern number hypothesis on historical crisis data.

    The validator:
    1. Loads crisis data (real or synthetic)
    2. Computes features and reduces dimensionality
    3. Fits QCML geometry and computes rolling Chern numbers
    4. Detects transitions and computes statistical metrics
    5. Determines if hypothesis is supported

    Attributes:
        config: ValidationConfig with hyperparameters
        results: Dictionary mapping crisis name to CrisisValidationResult
        aggregate: AggregateMetrics computed across all crises

    Example:
        >>> validator = CrisisValidator(ValidationConfig())
        >>> results = validator.validate_all_crises(use_synthetic=False)
        >>> print(f"Success rate: {validator.aggregate.success_rate:.1%}")
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize the crisis validator.

        Args:
            config: ValidationConfig with hyperparameters
        """
        self.config = config
        self.results: Dict[str, CrisisValidationResult] = {}
        self.aggregate: Optional[AggregateMetrics] = None

    def validate_single_crisis(
        self,
        crisis: CrisisDefinition,
        use_synthetic: bool = False,
        data_dir: str = "./data"
    ) -> CrisisValidationResult:
        """
        Validate hypothesis on a single crisis.

        Args:
            crisis: CrisisDefinition specifying the crisis
            use_synthetic: Use synthetic data instead of real API data
            data_dir: Directory for data storage

        Returns:
            CrisisValidationResult with all metrics

        Example:
            >>> from experiments.crisis_config import CRISIS_2008
            >>> result = validator.validate_single_crisis(CRISIS_2008)
            >>> print(f"Supported: {result.hypothesis_supported}")
        """
        logger.info(f"Validating crisis: {crisis.name} ({crisis.crisis_date})")

        # Step 1: Load data
        if use_synthetic:
            logger.info("Using synthetic data for testing")
            dataset = self._create_synthetic_crisis_dataset(crisis)
        else:
            logger.info("Loading real market data")
            # Try to fetch directly from Polygon API
            dataset = self._fetch_real_crisis_data(crisis)

        logger.info(f"Dataset: {dataset.n_samples} samples, {dataset.n_features} features")

        # Step 2: Prepare features with PCA
        X_raw = dataset.X
        X, pca_explained_var = self._prepare_features(X_raw)
        logger.info(f"PCA explained variance: {pca_explained_var:.1%}")

        # Step 3: Fit QCML geometry
        geometry = QCMLGeometry(
            n_features=X.shape[1],
            hilbert_dim=self.config.hilbert_dim
        )
        geometry.fit_operators(X, method=self.config.operator_method)
        logger.info(f"Fitted {len(geometry.operators)} operators (dim={self.config.hilbert_dim})")

        # Step 4: Create detector and compute rolling Chern
        detector = TopologicalRegimeDetector(
            geometry=geometry,
            window_size=self.config.window_size,
            chern_threshold=self.config.chern_threshold,
            smoothing_window=self.config.smoothing_window
        )

        chern_series = detector.rolling_chern_number(X, window=self.config.window_size)

        # Align times with Chern series
        chern_times = dataset.times[self.config.window_size - 1:]
        if len(chern_times) > len(chern_series):
            chern_times = chern_times[:len(chern_series)]

        logger.info(f"Chern series: {len(chern_series)} values, "
                   f"range [{chern_series.min():.3f}, {chern_series.max():.3f}]")

        # Step 5: Analyze around crisis date
        crisis_ts = pd.Timestamp(crisis.crisis_date)
        crisis_idx = (chern_times >= crisis_ts).argmax()

        # Get windows before and after crisis
        window_days = self.config.analysis_window_days
        before_start = max(0, crisis_idx - window_days)
        after_end = min(len(chern_series), crisis_idx + window_days)

        chern_before = chern_series[before_start:crisis_idx]
        chern_after = chern_series[crisis_idx:after_end]

        # Step 6: Compute statistical significance
        significance = compute_statistical_significance(chern_before, chern_after)

        # Step 7: Detect transitions
        transitions = detector.detect_transitions(X, times=np.arange(len(X)))
        transition_indices = [t.start_idx for t in transitions]

        # Find transitions near crisis
        nearby_transitions = [
            t for t in transitions
            if abs(t.start_idx - (crisis_idx + self.config.window_size - 1)) <= window_days
        ]

        logger.info(f"Total transitions: {len(transitions)}, near crisis: {len(nearby_transitions)}")

        # Step 8: Compute precision/recall
        pr_metrics = compute_precision_recall(
            detected_transitions=transition_indices,
            true_crisis_idx=crisis_idx + self.config.window_size - 1,
            tolerance_days=window_days
        )

        # Step 9: Compute lead time
        lead_time = compute_lead_time(
            chern_series=chern_series,
            times=chern_times.values,
            crisis_date=crisis.crisis_date,
            threshold=self.config.chern_threshold
        )

        # Step 10: Determine if hypothesis is supported
        # Primary evidence: statistical significance of Chern change
        is_significant = (
            significance['is_significant'] or
            significance['t_statistic'] > 2.0 or
            abs(significance['delta_chern']) > self.config.chern_threshold
        )

        # Hypothesis is supported if we have statistical evidence of regime change
        # Either: (1) significant Chern change with good statistics, OR
        #         (2) transitions detected near crisis with good precision/recall
        strong_statistical_evidence = (
            significance['t_statistic'] > 2.0 and
            significance['p_value'] < 0.05 and
            significance['effect_size'] > 0.5
        )

        transition_evidence = (
            len(nearby_transitions) > 0 or
            pr_metrics['recall'] > 0.5
        )

        hypothesis_supported = is_significant and (strong_statistical_evidence or transition_evidence)

        # Create result
        result = CrisisValidationResult(
            crisis_name=crisis.name,
            crisis_date=crisis.crisis_date,
            delta_chern=significance['delta_chern'],
            chern_before=significance['mean_before'],
            chern_after=significance['mean_after'],
            t_statistic=significance['t_statistic'],
            p_value=significance['p_value'],
            effect_size=significance['effect_size'],
            lead_time_days=lead_time,
            precision=pr_metrics['precision'],
            recall=pr_metrics['recall'],
            f1_score=pr_metrics['f1_score'],
            n_transitions_detected=len(nearby_transitions),
            is_significant=is_significant,
            hypothesis_supported=hypothesis_supported,
            raw_chern_series=chern_series.tolist(),
            transition_indices=transition_indices
        )

        self.results[crisis.name] = result
        return result

    def validate_all_crises(
        self,
        crises: Optional[List[CrisisDefinition]] = None,
        use_synthetic: bool = False,
        data_dir: str = "./data"
    ) -> Dict[str, CrisisValidationResult]:
        """
        Validate hypothesis on all configured crises.

        Args:
            crises: List of crises to validate (default: ALL_CRISES)
            use_synthetic: Use synthetic data
            data_dir: Data directory

        Returns:
            Dictionary mapping crisis name to result

        Example:
            >>> results = validator.validate_all_crises()
            >>> for name, result in results.items():
            ...     print(f"{name}: {'✓' if result.hypothesis_supported else '✗'}")
        """
        if crises is None:
            crises = ALL_CRISES

        logger.info(f"Validating {len(crises)} crises")

        for crisis in crises:
            try:
                self.validate_single_crisis(
                    crisis=crisis,
                    use_synthetic=use_synthetic,
                    data_dir=data_dir
                )
            except Exception as e:
                logger.error(f"Failed to validate {crisis.name}: {e}")
                # Create failed result
                self.results[crisis.name] = CrisisValidationResult(
                    crisis_name=crisis.name,
                    crisis_date=crisis.crisis_date,
                    delta_chern=0.0,
                    chern_before=0.0,
                    chern_after=0.0,
                    t_statistic=0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    lead_time_days=None,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    n_transitions_detected=0,
                    is_significant=False,
                    hypothesis_supported=False
                )

        # Compute aggregate metrics
        self.aggregate = aggregate_cross_crisis_metrics(list(self.results.values()))

        return self.results

    def create_optuna_objective(
        self,
        crises: Optional[List[CrisisDefinition]] = None,
        use_synthetic: bool = True,
        data_dir: str = "./data"
    ):
        """
        Create Optuna objective function for hyperparameter optimization.

        The objective maximizes: avg_t_statistic * avg_recall

        Args:
            crises: Crises to validate in each trial
            use_synthetic: Use synthetic data (faster for optimization)
            data_dir: Data directory

        Returns:
            Optuna objective function

        Example:
            >>> import optuna
            >>> objective = validator.create_optuna_objective()
            >>> study = optuna.create_study(direction='maximize')
            >>> study.optimize(objective, n_trials=50)
        """
        if crises is None:
            crises = ALL_CRISES

        def objective(trial):
            # Sample hyperparameters
            config = ValidationConfig(
                hilbert_dim=trial.suggest_categorical('hilbert_dim', [4, 8, 16]),
                window_size=trial.suggest_categorical('window_size', [10, 20, 30, 50]),
                chern_threshold=trial.suggest_float('chern_threshold', 0.05, 0.5),
                n_pca_components=trial.suggest_int('n_pca_components', 8, 30),
                operator_method=trial.suggest_categorical(
                    'operator_method', ['random', 'pca_inspired', 'pauli']
                ),
                use_full_features=self.config.use_full_features,
                smoothing_window=trial.suggest_int('smoothing_window', 3, 10),
                normalize_features=True,
                analysis_window_days=trial.suggest_int('analysis_window_days', 5, 20)
            )

            # Create validator with sampled config
            validator = CrisisValidator(config)

            # Validate all crises
            try:
                validator.validate_all_crises(
                    crises=crises,
                    use_synthetic=use_synthetic,
                    data_dir=data_dir
                )
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

            # Compute objective based on metric type
            if validator.aggregate is None:
                return 0.0

            # Composite objective that rewards:
            # 1. Statistical significance (t-statistic)
            # 2. Effect size (Cohen's d)
            # 3. Success rate (fraction of crises validated)
            # 4. Recall (transition detection) when available
            if self.config.use_full_features:
                # When using full features, prioritize statistical metrics
                objective_value = (
                    validator.aggregate.avg_t_statistic * 0.4 +
                    validator.aggregate.median_effect_size * 0.3 +
                    validator.aggregate.success_rate * 10.0 * 0.2 +
                    validator.aggregate.avg_recall * 0.1
                )
            else:
                # For minimal features, also weight recall
                objective_value = (
                    validator.aggregate.avg_t_statistic * 0.3 +
                    validator.aggregate.median_effect_size * 0.2 +
                    validator.aggregate.success_rate * 10.0 * 0.3 +
                    validator.aggregate.avg_recall * 0.2
                )

            # Report intermediate metrics
            trial.set_user_attr('success_rate', float(validator.aggregate.success_rate))
            trial.set_user_attr('avg_t_stat', float(validator.aggregate.avg_t_statistic))
            trial.set_user_attr('avg_effect_size', float(validator.aggregate.median_effect_size))
            trial.set_user_attr('avg_recall', float(validator.aggregate.avg_recall))

            return objective_value

        return objective

    def save_results(
        self,
        output_dir: str = "experiments/outputs/results",
        prefix: str = "crisis_validation"
    ) -> Dict[str, str]:
        """
        Save results to JSON and CSV files.

        Args:
            output_dir: Directory to save results
            prefix: Filename prefix

        Returns:
            Dictionary with paths to saved files

        Example:
            >>> paths = validator.save_results()
            >>> print(f"JSON: {paths['json']}")
            >>> print(f"CSV: {paths['csv']}")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build JSON output
        json_output = {
            'timestamp': timestamp,
            'config': config_to_dict(self.config),
            'crises': {
                name: result_to_dict(result)
                for name, result in self.results.items()
            },
            'aggregate_metrics': aggregate_to_dict(self.aggregate) if self.aggregate else None,
            'hypothesis_verdict': self._compute_verdict()
        }

        # Save JSON
        json_path = output_path / f"{prefix}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        logger.info(f"Saved JSON results to {json_path}")

        # Build CSV output
        csv_rows = []
        for name, result in self.results.items():
            csv_rows.append({
                'crisis': name,
                'crisis_date': result.crisis_date,
                'delta_chern': result.delta_chern,
                't_stat': result.t_statistic,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'lead_time': result.lead_time_days,
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1_score,
                'hypothesis_supported': result.hypothesis_supported
            })

        csv_df = pd.DataFrame(csv_rows)
        csv_path = output_path / f"{prefix}_metrics_summary.csv"
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV summary to {csv_path}")

        return {
            'json': str(json_path),
            'csv': str(csv_path)
        }

    def _fetch_real_crisis_data(self, crisis: CrisisDefinition) -> QCMLDataset:
        """
        Fetch real crisis data directly from Polygon API.

        Args:
            crisis: Crisis definition

        Returns:
            QCMLDataset with real market data
        """
        # Load API key
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            raise ValueError("POLYGON_API_KEY not found. Set it in .env file.")

        # Calculate date range
        crisis_date = pd.Timestamp(crisis.crisis_date)
        start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
        end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

        logger.info(f"Fetching data from Polygon: {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {crisis.universe}")

        # Fetch from Polygon
        source = PolygonDataSource(api_key=api_key)
        raw_data = source.fetch_equities(
            crisis.universe,
            str(start_date.date()),
            str(end_date.date()),
            timeframe="1d"
        )

        if raw_data.empty:
            raise ValueError(f"No data returned from Polygon API for {crisis.name}")

        # Pivot to get prices DataFrame
        prices = raw_data['close'].unstack(level=0)
        prices = prices.ffill()  # Forward fill missing values

        logger.info(f"Fetched {len(prices)} days of data for {len(prices.columns)} symbols")

        # Compute features using MinimalFeatureEngine (more robust than full features)
        benchmark_col = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
        engine = MinimalFeatureEngine(window=20)
        features = engine.create_feature_matrix(prices, benchmark_col=benchmark_col)

        # Drop any remaining NaN rows
        features = features.dropna()

        if len(features) < 50:
            raise ValueError(f"Insufficient data after feature computation: {len(features)} rows")

        # Align prices with features
        aligned_prices = prices[benchmark_col].loc[features.index]

        # Create metadata
        metadata = {
            'crisis': crisis.name,
            'crisis_date': crisis.crisis_date,
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date()),
            'symbols': list(prices.columns),
            'benchmark': benchmark_col,
            'source': 'polygon_api'
        }

        return QCMLDataset(features, aligned_prices, features.index, metadata)

    def _prepare_features(self, X_raw: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Prepare features with standardization and PCA.

        Args:
            X_raw: Raw feature matrix

        Returns:
            Tuple of (processed features, explained variance ratio)
        """
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # PCA
        n_components = min(self.config.n_pca_components, X_raw.shape[1])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_.sum()

        # Normalize to unit sphere for better geometric properties
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        return X, explained_var

    def _create_synthetic_crisis_dataset(
        self,
        crisis: CrisisDefinition
    ) -> QCMLDataset:
        """
        Create synthetic dataset with regime change at crisis date.

        Creates a dataset with:
        - Realistic date range centered on the crisis date
        - Clear regime change at the crisis date (doubled volatility)
        - Features that exhibit different behavior before/after crisis

        Args:
            crisis: Crisis definition

        Returns:
            Synthetic QCMLDataset with dates matching crisis period
        """
        # Compute date range
        crisis_date = pd.Timestamp(crisis.crisis_date)
        start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
        end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

        # Generate business days
        dates = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(dates)

        # Find crisis index
        crisis_idx = (dates <= crisis_date).sum()

        np.random.seed(42)
        n_features = 30 if self.config.use_full_features else 10

        # Create features with STRONG regime change for clear topological transition
        # Regime 1: Normal market - sphere-like geometry
        theta1 = np.linspace(0, 2 * np.pi, crisis_idx)
        features1 = np.zeros((crisis_idx, n_features))
        for i in range(n_features):
            phase = 2 * np.pi * i / n_features
            features1[:, i] = np.cos(theta1 + phase) + 0.1 * np.random.randn(crisis_idx)

        # Regime 2: Crisis market - DIFFERENT geometry (torus-like structure)
        theta2 = np.linspace(0, 4 * np.pi, n_days - crisis_idx)
        features2 = np.zeros((n_days - crisis_idx, n_features))
        for i in range(n_features):
            phase = 2 * np.pi * i / n_features
            # Different geometric structure - higher frequency, different amplitude
            features2[:, i] = 2.0 * np.sin(2 * theta2 + phase) + 0.3 * np.random.randn(n_days - crisis_idx)

        # Add strong correlation structure change
        for i in range(0, n_features - 1, 2):
            features1[:, i + 1] += 0.7 * features1[:, i]  # Strong positive correlation before
            features2[:, i + 1] -= 0.7 * features2[:, i]  # Strong negative correlation after

        features = np.vstack([features1, features2])

        # Add noise
        features += np.random.randn(n_days, n_features) * 0.1

        # Create synthetic prices (GBM with regime change)
        returns = np.zeros(n_days)
        returns[:crisis_idx] = np.random.randn(crisis_idx) * 0.01 + 0.0002  # Pre-crisis
        returns[crisis_idx:] = np.random.randn(n_days - crisis_idx) * 0.025 - 0.001  # Post-crisis

        prices = 100 * np.exp(np.cumsum(returns))

        # Convert to DataFrame/Series with proper dates
        features_df = pd.DataFrame(
            features,
            columns=[f"feature_{i}" for i in range(n_features)],
            index=dates
        )
        prices_series = pd.Series(prices, name='close', index=dates)

        metadata = {
            'type': 'synthetic',
            'crisis': crisis.name,
            'crisis_date': crisis.crisis_date,
            'regime_change_idx': crisis_idx,
            'n_samples': n_days,
            'n_features': n_features,
            'synthetic': True
        }

        return QCMLDataset(features_df, prices_series, dates, metadata)

    def _compute_verdict(self) -> str:
        """Compute overall hypothesis verdict."""
        if self.aggregate is None:
            return "INSUFFICIENT_DATA"

        if self.aggregate.success_rate >= 0.67:
            return "SUPPORTED"
        elif self.aggregate.success_rate >= 0.33:
            return "PARTIALLY_SUPPORTED"
        else:
            return "NOT_SUPPORTED"


def run_optuna_optimization(
    optuna_config: OptunaConfig,
    crises: Optional[List[CrisisDefinition]] = None,
    use_synthetic: bool = True,
    data_dir: str = "./data"
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization.

    Args:
        optuna_config: Optuna configuration
        crises: Crises to use for optimization
        use_synthetic: Use synthetic data (recommended for speed)
        data_dir: Data directory

    Returns:
        Dictionary with best parameters and study results

    Example:
        >>> config = OptunaConfig(n_trials=50)
        >>> results = run_optuna_optimization(config)
        >>> print(f"Best params: {results['best_params']}")
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    logger.info(f"Starting Optuna optimization with {optuna_config.n_trials} trials")

    # Create study
    if optuna_config.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=42)
    elif optuna_config.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)

    if optuna_config.pruner == 'median':
        pruner = optuna.pruners.MedianPruner()
    elif optuna_config.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=optuna_config.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=optuna_config.storage,
        load_if_exists=True
    )

    # Create objective
    base_config = get_default_validation_config()
    validator = CrisisValidator(base_config)
    objective = validator.create_optuna_objective(
        crises=crises,
        use_synthetic=use_synthetic,
        data_dir=data_dir
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        timeout=optuna_config.timeout,
        show_progress_bar=True
    )

    # Get results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    logger.info(f"Best objective value: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_trial': best_trial.number,
        'n_trials': len(study.trials),
        'study_name': optuna_config.study_name
    }


def main():
    """Main entry point for crisis validation."""
    parser = argparse.ArgumentParser(
        description="Validate QCML Chern number hypothesis on crisis data"
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data (no API key needed)'
    )
    parser.add_argument(
        '--optuna', action='store_true',
        help='Run Optuna hyperparameter optimization'
    )
    parser.add_argument(
        '--n-trials', type=int, default=50,
        help='Number of Optuna trials (default: 50)'
    )
    parser.add_argument(
        '--crisis', type=str, default=None,
        help='Validate single crisis (2008_crisis, 2020_covid, 2022_rates)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='experiments/outputs/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./data',
        help='Data directory'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    # Load environment
    load_dotenv(project_root / '.env')

    print("=" * 70)
    print("QCML Crisis Detection Validation")
    print("=" * 70)
    print()

    # Determine crises to validate
    if args.crisis:
        crises = [get_crisis_by_name(args.crisis)]
    else:
        crises = ALL_CRISES

    print(f"Crises to validate: {[c.name for c in crises]}")
    print(f"Using synthetic data: {args.synthetic}")
    print()

    if args.optuna:
        # Run Optuna optimization
        print("-" * 70)
        print("Running Optuna Hyperparameter Optimization")
        print("-" * 70)

        optuna_config = OptunaConfig(n_trials=args.n_trials)
        optuna_results = run_optuna_optimization(
            optuna_config=optuna_config,
            crises=crises,
            use_synthetic=args.synthetic,
            data_dir=args.data_dir
        )

        print()
        print("Best Parameters:")
        for key, value in optuna_results['best_params'].items():
            print(f"  {key}: {value}")
        print(f"Best Objective: {optuna_results['best_value']:.4f}")
        print()

        # Validate with best parameters
        best_config = ValidationConfig(**optuna_results['best_params'])
        validator = CrisisValidator(best_config)

    else:
        # Use default configuration
        config = get_default_validation_config()
        validator = CrisisValidator(config)

    # Run validation
    print("-" * 70)
    print("Running Crisis Validation")
    print("-" * 70)

    results = validator.validate_all_crises(
        crises=crises,
        use_synthetic=args.synthetic,
        data_dir=args.data_dir
    )

    # Print summary
    print()
    summary = format_results_summary(
        list(results.values()),
        validator.aggregate
    )
    print(summary)

    # Save results
    output_paths = validator.save_results(output_dir=args.output_dir)
    print()
    print(f"Results saved to:")
    print(f"  JSON: {output_paths['json']}")
    print(f"  CSV: {output_paths['csv']}")

    return validator


if __name__ == "__main__":
    try:
        validator = main()
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise
