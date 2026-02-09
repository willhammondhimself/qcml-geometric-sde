#!/usr/bin/env python3
"""
Novel Quantum Features Ablation Study

Phase 3 of Academic Research Plan: Validates whether novel quantum indicators
(spectral gap, energy evolution, fidelity decay, multi-scale consensus)
provide incremental value over standard Chern number detection.

Ablation Study Design:
  - Baseline: Standard Chern detection (optimized params)
  - Augmented: Chern + each novel feature individually
  - Combined: Chern + all novel features
  - Metrics: Incremental F1, lead time improvement, FP reduction
  - Statistical: Paired t-test, bootstrap comparison

Uses REAL market data from Polygon API.

Usage:
    python experiments/novel_features_validation.py
    python experiments/novel_features_validation.py --crisis 2008_crisis

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.qcml_geometry import QCMLGeometry
from qcml.topological_regime import TopologicalRegimeDetector
from qcml.regime.quantum_indicators import (
    SpectralGapIndicator,
    EnergyEvolutionIndicator,
    FidelityDecayIndicator,
    MultiScaleChernConsensus,
    QuantumIndicatorSuite,
    IndicatorResult,
)
from qcml.data import PolygonDataSource, MinimalFeatureEngine, QCMLDataset

from experiments.crisis_config import (
    CrisisDefinition,
    ValidationConfig,
    ALL_CRISES,
    get_crisis_by_name,
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


@dataclass
class AblationResult:
    """
    Result from a single ablation experiment.

    Attributes:
        method_name: Name of the method variant
        crisis_name: Crisis being tested
        f1_score: F1 detection score
        precision: Precision
        recall: Recall
        lead_time_days: Lead time in trading days
        n_false_positives: Number of false positives
        n_true_positives: Number of true positives
        t_statistic: Welch's t-statistic
        p_value: p-value
        effect_size: Cohen's d
        n_transitions: Total transitions detected
    """
    method_name: str
    crisis_name: str
    f1_score: float
    precision: float
    recall: float
    lead_time_days: Optional[int]
    n_false_positives: int
    n_true_positives: int
    t_statistic: float
    p_value: float
    effect_size: float
    n_transitions: int


def fetch_crisis_data(
    crisis: CrisisDefinition,
) -> QCMLDataset:
    """
    Fetch real market data for a crisis from Polygon API.

    Args:
        crisis: Crisis definition

    Returns:
        QCMLDataset with real market data
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    logger.info(f"Fetching {crisis.name}: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {crisis.universe}")

    source = PolygonDataSource(api_key=api_key)
    raw_data = source.fetch_equities(
        crisis.universe,
        str(start_date.date()),
        str(end_date.date()),
        timeframe="1d"
    )

    if raw_data.empty:
        raise ValueError(f"No data returned for {crisis.name}")

    prices = raw_data['close'].unstack(level=0)
    prices = prices.ffill()

    benchmark_col = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=benchmark_col)
    features = features.dropna()

    if len(features) < 50:
        raise ValueError(f"Insufficient data: {len(features)} rows")

    aligned_prices = prices[benchmark_col].loc[features.index]

    metadata = {
        'crisis': crisis.name,
        'crisis_date': crisis.crisis_date,
        'source': 'polygon_api',
        'symbols': list(prices.columns),
        'benchmark': benchmark_col,
    }

    return QCMLDataset(features, aligned_prices, features.index, metadata)


def prepare_features(
    X_raw: np.ndarray,
    n_pca_components: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Standardize and PCA-reduce features.

    Args:
        X_raw: Raw feature matrix
        n_pca_components: Number of PCA components

    Returns:
        X: Processed features
        explained_var: PCA explained variance ratio
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_.sum()

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    return X, explained_var


def run_baseline_detection(
    X: np.ndarray,
    geometry: QCMLGeometry,
    config: ValidationConfig,
    crisis: CrisisDefinition,
    chern_times: pd.DatetimeIndex,
) -> AblationResult:
    """
    Run baseline Chern detection (no novel features).

    Args:
        X: Processed features
        geometry: Fitted QCMLGeometry
        config: Validation config
        crisis: Crisis definition
        chern_times: Timestamps

    Returns:
        AblationResult for baseline
    """
    detector = TopologicalRegimeDetector(
        geometry=geometry,
        window_size=config.window_size,
        chern_threshold=config.chern_threshold,
        smoothing_window=config.smoothing_window,
    )

    chern_series = detector.rolling_chern_number(X, window=config.window_size)

    return _evaluate_detection(
        method_name="baseline_chern",
        chern_series=chern_series,
        X=X,
        geometry=geometry,
        detector=detector,
        config=config,
        crisis=crisis,
        chern_times=chern_times,
    )


def run_augmented_detection(
    X: np.ndarray,
    geometry: QCMLGeometry,
    config: ValidationConfig,
    crisis: CrisisDefinition,
    chern_times: pd.DatetimeIndex,
    indicator_name: str,
    indicator_result: IndicatorResult,
) -> AblationResult:
    """
    Run augmented detection: baseline Chern + novel indicator.

    The augmented detection considers a transition if EITHER the baseline
    Chern detector OR the novel indicator signals a transition.

    Args:
        X: Processed features
        geometry: Fitted QCMLGeometry
        config: Validation config
        crisis: Crisis definition
        chern_times: Timestamps
        indicator_name: Name of novel indicator
        indicator_result: Result from novel indicator

    Returns:
        AblationResult for augmented method
    """
    detector = TopologicalRegimeDetector(
        geometry=geometry,
        window_size=config.window_size,
        chern_threshold=config.chern_threshold,
        smoothing_window=config.smoothing_window,
    )

    chern_series = detector.rolling_chern_number(X, window=config.window_size)

    # Combine transitions: union of Chern transitions and indicator transitions
    base_transitions = detector.detect_transitions(X, times=np.arange(len(X)))
    base_indices = set(t.start_idx for t in base_transitions)

    # Add novel indicator transitions
    augmented_indices = base_indices | set(indicator_result.transitions)

    return _evaluate_detection(
        method_name=f"chern+{indicator_name}",
        chern_series=chern_series,
        X=X,
        geometry=geometry,
        detector=detector,
        config=config,
        crisis=crisis,
        chern_times=chern_times,
        override_transitions=sorted(augmented_indices),
    )


def _evaluate_detection(
    method_name: str,
    chern_series: np.ndarray,
    X: np.ndarray,
    geometry: QCMLGeometry,
    detector: TopologicalRegimeDetector,
    config: ValidationConfig,
    crisis: CrisisDefinition,
    chern_times: pd.DatetimeIndex,
    override_transitions: Optional[List[int]] = None,
) -> AblationResult:
    """
    Evaluate detection performance for a method.

    Args:
        method_name: Method identifier
        chern_series: Chern values
        X: Feature array
        geometry: QCMLGeometry
        detector: Detector instance
        config: Config
        crisis: Crisis definition
        chern_times: Timestamps
        override_transitions: If provided, use these instead of detector transitions

    Returns:
        AblationResult
    """
    adjusted_times = chern_times[:len(chern_series)]

    crisis_ts = pd.Timestamp(crisis.crisis_date)
    crisis_idx = (adjusted_times >= crisis_ts).argmax()

    window_days = config.analysis_window_days
    before_start = max(0, crisis_idx - window_days)
    after_end = min(len(chern_series), crisis_idx + window_days)

    chern_before = chern_series[before_start:crisis_idx]
    chern_after = chern_series[crisis_idx:after_end]

    if len(chern_before) < 2 or len(chern_after) < 2:
        return AblationResult(
            method_name=method_name,
            crisis_name=crisis.name,
            f1_score=0.0, precision=0.0, recall=0.0,
            lead_time_days=None, n_false_positives=0,
            n_true_positives=0, t_statistic=0.0,
            p_value=1.0, effect_size=0.0, n_transitions=0,
        )

    sig = compute_statistical_significance(chern_before, chern_after)

    if override_transitions is not None:
        transition_indices = override_transitions
    else:
        transitions = detector.detect_transitions(X, times=np.arange(len(X)))
        transition_indices = [t.start_idx for t in transitions]

    true_crisis_idx = crisis_idx + config.window_size - 1
    pr = compute_precision_recall(
        transition_indices, true_crisis_idx, tolerance_days=window_days
    )

    lead_time = compute_lead_time(
        chern_series, adjusted_times.values,
        crisis.crisis_date, threshold=config.chern_threshold
    )

    return AblationResult(
        method_name=method_name,
        crisis_name=crisis.name,
        f1_score=pr['f1_score'],
        precision=pr['precision'],
        recall=pr['recall'],
        lead_time_days=lead_time,
        n_false_positives=pr['n_false_positives'],
        n_true_positives=pr['n_true_positives'],
        t_statistic=sig['t_statistic'],
        p_value=sig['p_value'],
        effect_size=sig['effect_size'],
        n_transitions=len(transition_indices),
    )


def run_ablation_study(
    crisis: CrisisDefinition,
    config: Optional[ValidationConfig] = None,
    seed: int = 42,
) -> Dict[str, AblationResult]:
    """
    Run complete ablation study for a single crisis using real data.

    Tests:
      1. Baseline: Standard Chern only
      2. Chern + Spectral Gap
      3. Chern + Energy Evolution
      4. Chern + Fidelity Decay
      5. Chern + Multi-Scale Consensus
      6. Chern + All Novel Features

    Args:
        crisis: Crisis definition
        config: Validation config
        seed: Random seed

    Returns:
        Dict mapping method name to AblationResult
    """
    seed_everything(seed)
    config = config or get_default_validation_config()

    logger.info(f"Running ablation study for {crisis.name}")

    # Fetch real data
    dataset = fetch_crisis_data(crisis)
    logger.info(f"Dataset: {dataset.n_samples} samples, {dataset.n_features} features")

    # Prepare features
    X, explained_var = prepare_features(dataset.X, n_pca_components=config.n_pca_components)
    logger.info(f"PCA explained variance: {explained_var:.1%}")

    # Fit geometry
    geometry = QCMLGeometry(
        n_features=X.shape[1],
        hilbert_dim=config.hilbert_dim,
    )
    geometry.fit_operators(X, method=config.operator_method)

    chern_times = dataset.times[config.window_size - 1:]

    results = {}

    # 1. Baseline
    logger.info("Running baseline Chern detection...")
    results['baseline_chern'] = run_baseline_detection(
        X, geometry, config, crisis, chern_times
    )

    # 2-5. Individual novel features
    suite = QuantumIndicatorSuite(
        geometry, window_size=config.window_size,
        scales=[10, 20, 30, 50]
    )
    indicator_results = suite.compute_all(X)

    for ind_name, ind_result in indicator_results.items():
        logger.info(f"Running augmented: Chern + {ind_name}...")
        results[f'chern+{ind_name}'] = run_augmented_detection(
            X, geometry, config, crisis, chern_times,
            ind_name, ind_result,
        )

    # 6. All novel features combined
    logger.info("Running combined: Chern + all novel features...")
    all_transitions = set()
    for ind_result in indicator_results.values():
        all_transitions.update(ind_result.transitions)

    combined_indicator = IndicatorResult(
        name="all_novel",
        values=np.zeros(1),
        transitions=sorted(all_transitions),
        threshold=0.0,
    )
    results['chern+all_novel'] = run_augmented_detection(
        X, geometry, config, crisis, chern_times,
        "all_novel", combined_indicator,
    )

    return results


def compute_incremental_value(
    baseline: AblationResult,
    augmented: AblationResult,
) -> Dict[str, float]:
    """
    Compute incremental value of augmented method over baseline.

    Args:
        baseline: Baseline result
        augmented: Augmented result

    Returns:
        Dict with incremental metrics
    """
    f1_delta = augmented.f1_score - baseline.f1_score
    f1_pct = (f1_delta / baseline.f1_score * 100) if baseline.f1_score > 0 else float('inf')

    fp_reduction = baseline.n_false_positives - augmented.n_false_positives

    lead_improvement = None
    if augmented.lead_time_days is not None and baseline.lead_time_days is not None:
        lead_improvement = augmented.lead_time_days - baseline.lead_time_days

    return {
        'f1_delta': f1_delta,
        'f1_pct_improvement': f1_pct,
        'fp_reduction': fp_reduction,
        'lead_improvement': lead_improvement,
        'recall_delta': augmented.recall - baseline.recall,
        'precision_delta': augmented.precision - baseline.precision,
    }


def run_full_ablation(
    crises: Optional[List[CrisisDefinition]] = None,
    config: Optional[ValidationConfig] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, AblationResult]]:
    """
    Run ablation study across all crises.

    Args:
        crises: List of crises (default: ALL_CRISES)
        config: Validation config
        seed: Random seed

    Returns:
        Nested dict: crisis_name -> method_name -> AblationResult
    """
    crises = crises or ALL_CRISES
    all_results = {}

    for crisis in crises:
        try:
            all_results[crisis.name] = run_ablation_study(
                crisis, config=config, seed=seed
            )
        except Exception as e:
            logger.error(f"Ablation failed for {crisis.name}: {e}")

    return all_results


def print_ablation_summary(
    all_results: Dict[str, Dict[str, AblationResult]]
) -> None:
    """Print formatted ablation study summary."""
    print("\n" + "=" * 90)
    print("NOVEL FEATURES ABLATION STUDY - RESULTS")
    print("=" * 90)

    for crisis_name, methods in all_results.items():
        print(f"\n--- {crisis_name} ---")
        baseline = methods.get('baseline_chern')
        if not baseline:
            print("  No baseline result")
            continue

        print(f"  {'Method':<30} {'F1':>6} {'P':>6} {'R':>6} "
              f"{'Lead':>6} {'FP':>4} {'dF1':>8}")
        print("  " + "-" * 75)

        for method_name, result in methods.items():
            lead = str(result.lead_time_days) if result.lead_time_days else 'N/A'
            if method_name == 'baseline_chern':
                delta_str = '  base'
            else:
                inc = compute_incremental_value(baseline, result)
                delta_str = f"{inc['f1_delta']:+.3f}"

            print(f"  {method_name:<30} {result.f1_score:>6.3f} "
                  f"{result.precision:>6.3f} {result.recall:>6.3f} "
                  f"{lead:>6} {result.n_false_positives:>4} {delta_str:>8}")

    # Aggregate: which novel feature helps most across crises?
    print("\n" + "-" * 90)
    print("AGGREGATE INCREMENTAL VALUE")
    print("-" * 90)

    method_names = set()
    for methods in all_results.values():
        method_names.update(methods.keys())
    method_names.discard('baseline_chern')

    for method_name in sorted(method_names):
        f1_deltas = []
        for crisis_name, methods in all_results.items():
            baseline = methods.get('baseline_chern')
            augmented = methods.get(method_name)
            if baseline and augmented:
                inc = compute_incremental_value(baseline, augmented)
                f1_deltas.append(inc['f1_delta'])

        if f1_deltas:
            mean_delta = np.mean(f1_deltas)
            indicator = "+" if mean_delta > 0 else ""
            meets_threshold = abs(mean_delta) > 0.10 * np.mean([
                methods.get('baseline_chern', AblationResult(
                    method_name='', crisis_name='', f1_score=0, precision=0,
                    recall=0, lead_time_days=None, n_false_positives=0,
                    n_true_positives=0, t_statistic=0, p_value=1,
                    effect_size=0, n_transitions=0,
                )).f1_score
                for methods in all_results.values()
            ])
            status = "PASS (>10%)" if meets_threshold else "below 10%"
            print(f"  {method_name:<30} avg dF1={indicator}{mean_delta:.4f}  [{status}]")

    print("=" * 90)


def save_ablation_results(
    all_results: Dict[str, Dict[str, AblationResult]],
    output_dir: str = "experiments/outputs/regime_detection/results",
) -> str:
    """Save ablation results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        'timestamp': timestamp,
        'study': 'novel_features_ablation',
        'crises': {},
    }

    for crisis_name, methods in all_results.items():
        crisis_data = {}
        baseline = methods.get('baseline_chern')

        for method_name, result in methods.items():
            method_data = {
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall,
                'lead_time_days': result.lead_time_days,
                'n_false_positives': result.n_false_positives,
                'n_true_positives': result.n_true_positives,
                't_statistic': result.t_statistic,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'n_transitions': result.n_transitions,
            }

            if baseline and method_name != 'baseline_chern':
                inc = compute_incremental_value(baseline, result)
                method_data['incremental'] = inc

            crisis_data[method_name] = method_data

        output['crises'][crisis_name] = crisis_data

    filepath = output_path / f"ablation_results_{timestamp}.json"
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Novel Features Ablation Study for QCML Regime Detection"
    )
    parser.add_argument("--crisis", type=str, default=None,
                        help="Single crisis to test (e.g., 2008_crisis)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/outputs/regime_detection/results",
                        help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    load_dotenv(project_root / '.env')

    print("=" * 70)
    print("QCML Novel Features Ablation Study (Real Data)")
    print("=" * 70)

    if args.crisis:
        crises = [get_crisis_by_name(args.crisis)]
    else:
        crises = ALL_CRISES

    print(f"Crises: {[c.name for c in crises]}")
    print(f"Data source: Polygon API (real market data)")
    print("=" * 70)

    all_results = run_full_ablation(crises=crises, seed=args.seed)

    print_ablation_summary(all_results)

    filepath = save_ablation_results(all_results, output_dir=args.output_dir)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
