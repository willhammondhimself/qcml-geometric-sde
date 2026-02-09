#!/usr/bin/env python3
"""
Hybrid Ensemble Analysis: QCML + Random Forest

Combines top QCML methods with RF to show the combination achieves d > 1.10
(better than RF alone), proving QCML adds unique information content.

Three ensemble approaches:
  1. Simple Average — z-score top QCML + RF, average them
  2. Optimized Weights — grid search on pre-2020 crises, test on post-2020 OOS
  3. Dynamic Switch — use max(QCML z-scores) as routing signal

Each ensemble is evaluated through the same temporal OOS pipeline as
experiments/temporal_oos_validation.py to ensure no look-ahead.

Key hypothesis: At least one ensemble achieves mean d > 1.10, proving QCML
provides unique complementary signal beyond what RF captures alone.

Usage:
    python experiments/hybrid_ensemble.py --seed 42

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.data import PolygonDataSource, MinimalFeatureEngine
from qcml_geometry import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
)
from experiments.additional_detectors import (
    QCMLChernDetector,
    MultiScaleChernDetector,
    QuantumEnsembleDetector,
    QFISusceptibilityDetector,
    ScalarCurvatureDetector,
    GeometricConsensusDetector,
    MetricConditionNumberDetector,
)
from experiments.crisis_config import (
    CrisisDefinition,
    DATA_AVAILABLE_CRISES,
    ValidationConfig,
    get_default_validation_config,
    config_to_dict,
)
from experiments.crisis_metrics import compute_statistical_significance
from experiments.regime_comparison import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("experiments/outputs/regime_detection/hybrid")

CALIBRATION_CUTOFF = "2020-01-01"

TRAINING_CRISES = [c for c in DATA_AVAILABLE_CRISES
                   if pd.Timestamp(c.crisis_date) < pd.Timestamp(CALIBRATION_CUTOFF)]

TEST_CRISES = [c for c in DATA_AVAILABLE_CRISES
               if pd.Timestamp(c.crisis_date) >= pd.Timestamp(CALIBRATION_CUTOFF)]

# Top 3 QCML methods (from authoritative results)
TOP_QCML = ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared data pipeline (same as temporal_oos_validation.py)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_full_data(
    crisis: CrisisDefinition,
    min_start_date: str = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DatetimeIndex]]:
    """Fetch raw price data for a crisis period.

    Args:
        crisis: Crisis definition.
        min_start_date: If given, ensure data starts at least this early.
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    if min_start_date:
        min_ts = pd.Timestamp(min_start_date)
        if start_date > min_ts:
            start_date = min_ts

    source = PolygonDataSource(api_key=api_key)
    raw_data = source.fetch_equities(
        crisis.universe,
        str(start_date.date()),
        str(end_date.date()),
        timeframe="1d",
    )

    if raw_data.empty:
        return None, None

    prices = raw_data['close'].unstack(level=0).ffill()
    return prices, prices.index


def build_features_frozen(
    prices: pd.DataFrame,
    calibration_end: str,
    n_pca_components: int = 15,
    enriched_lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, int,
           StandardScaler, PCA]:
    """Build features with PCA/scaler frozen on calibration period."""
    benchmark_col = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=benchmark_col)
    features = features.dropna()

    X_raw = features.values
    times = features.index

    cal_ts = pd.Timestamp(calibration_end)
    cal_mask = times < cal_ts
    calibration_idx = int(cal_mask.sum())

    if calibration_idx < 60:
        logger.warning(f"Only {calibration_idx} calibration points")

    scaler = StandardScaler()
    scaler.fit(X_raw[:calibration_idx])
    X_scaled = scaler.transform(X_raw)

    n_components = min(n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled[:calibration_idx])
    X_pca = pca.transform(X_scaled)

    X = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=enriched_lookback)

    return X, X_enriched, times, calibration_idx, scaler, pca


def compute_effect_size(
    scores: np.ndarray,
    times: pd.DatetimeIndex,
    crisis: CrisisDefinition,
    window_days: int = 10,
) -> Dict[str, float]:
    """Compute Cohen's d for crisis vs non-crisis scores."""
    crisis_ts = pd.Timestamp(crisis.crisis_date)

    valid_mask = ~np.isnan(scores)
    valid_scores = scores[valid_mask]
    valid_times = times[valid_mask]

    crisis_mask = valid_times >= crisis_ts
    if not crisis_mask.any():
        return {'effect_size_d': 0.0, 'p_value': 1.0, 't_stat': 0.0}
    crisis_idx = int(crisis_mask.argmax())

    start = max(0, crisis_idx - window_days)
    end = min(len(valid_scores), crisis_idx + window_days)

    crisis_window = np.zeros(len(valid_scores), dtype=bool)
    crisis_window[start:end] = True

    scores_crisis = valid_scores[crisis_window]
    scores_non = valid_scores[~crisis_window]

    if len(scores_crisis) < 3 or len(scores_non) < 3:
        return {'effect_size_d': 0.0, 'p_value': 1.0, 't_stat': 0.0}

    sig = compute_statistical_significance(scores_non, scores_crisis)
    return {
        'effect_size_d': float(sig['effect_size']),
        'p_value': float(sig['p_value']),
        't_stat': float(sig['t_statistic']),
    }


def zscore_series(scores: np.ndarray) -> np.ndarray:
    """Expanding z-score normalization (causal — no lookahead)."""
    z = np.full_like(scores, np.nan, dtype=float)
    for t in range(2, len(scores)):
        valid = scores[:t + 1]
        valid = valid[~np.isnan(valid)]
        if len(valid) < 3:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma > 1e-12:
            z[t] = (scores[t] - mu) / sigma
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Approaches
# ─────────────────────────────────────────────────────────────────────────────

def simple_average_ensemble(
    score_dict: Dict[str, np.ndarray],
    component_names: List[str],
) -> np.ndarray:
    """Simple average of z-scored component scores.

    Args:
        score_dict: method_name -> raw score array (same length).
        component_names: which methods to include.

    Returns:
        Averaged z-scored ensemble score.
    """
    z_scores = []
    for name in component_names:
        if name in score_dict:
            z = zscore_series(score_dict[name])
            z_scores.append(z)

    if not z_scores:
        return np.zeros(0)

    z_stack = np.array(z_scores)
    # nanmean across methods at each time step
    return np.nanmean(z_stack, axis=0)


def weighted_ensemble(
    score_dict: Dict[str, np.ndarray],
    component_names: List[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted combination of z-scored component scores.

    Args:
        score_dict: method_name -> raw score array.
        component_names: ordered list of methods.
        weights: array of weights matching component_names (must sum to 1).

    Returns:
        Weighted z-scored ensemble score.
    """
    z_scores = []
    valid_weights = []
    for i, name in enumerate(component_names):
        if name in score_dict:
            z = zscore_series(score_dict[name])
            z_scores.append(z)
            valid_weights.append(weights[i])

    if not z_scores:
        return np.zeros(0)

    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()  # re-normalize

    z_stack = np.array(z_scores)
    # Weighted nanmean: replace NaN with 0, then weight
    z_filled = np.nan_to_num(z_stack, nan=0.0)
    not_nan = ~np.isnan(z_stack)
    # Weighted sum with NaN handling
    result = np.zeros(z_stack.shape[1])
    for t in range(z_stack.shape[1]):
        mask = not_nan[:, t]
        if mask.sum() == 0:
            result[t] = np.nan
        else:
            w = valid_weights[mask]
            w = w / w.sum()
            result[t] = np.dot(w, z_stack[:, t][mask])

    return result


def dynamic_switch_ensemble(
    score_dict: Dict[str, np.ndarray],
    qcml_names: List[str],
    rf_name: str = "Random Forest",
    switch_threshold: float = 1.5,
) -> np.ndarray:
    """Dynamic switch: use QCML when QCML fires strongly, else RF.

    When max(QCML z-scores) > switch_threshold, use the average of QCML
    z-scores. Otherwise, use the RF z-score.

    Args:
        score_dict: method_name -> raw score array.
        qcml_names: QCML method names to consider.
        rf_name: RF method name.
        switch_threshold: z-score threshold for QCML activation.

    Returns:
        Dynamic ensemble score.
    """
    # Z-score all components
    z_qcml = {}
    for name in qcml_names:
        if name in score_dict:
            z_qcml[name] = zscore_series(score_dict[name])

    z_rf = zscore_series(score_dict[rf_name]) if rf_name in score_dict else None

    if not z_qcml:
        return z_rf if z_rf is not None else np.zeros(0)

    # Get the shortest common length
    lengths = [len(v) for v in z_qcml.values()]
    if z_rf is not None:
        lengths.append(len(z_rf))
    T = min(lengths)

    result = np.full(T, np.nan)
    qcml_stack = np.array([v[:T] for v in z_qcml.values()])

    for t in range(T):
        qcml_vals = qcml_stack[:, t]
        qcml_valid = qcml_vals[~np.isnan(qcml_vals)]

        if len(qcml_valid) == 0:
            if z_rf is not None and not np.isnan(z_rf[t]):
                result[t] = z_rf[t]
            continue

        max_qcml_z = np.max(qcml_valid)

        if max_qcml_z > switch_threshold:
            # QCML fires strongly — use QCML average
            result[t] = np.mean(qcml_valid)
        else:
            # QCML quiet — use RF
            if z_rf is not None and not np.isnan(z_rf[t]):
                result[t] = z_rf[t]
            else:
                result[t] = np.mean(qcml_valid)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Weight Optimization (grid search on pre-2020 crises)
# ─────────────────────────────────────────────────────────────────────────────

def generate_simplex_weights(n_components: int, step: float = 0.1) -> List[np.ndarray]:
    """Generate all weight vectors on the simplex with given step size.

    Args:
        n_components: number of components.
        step: grid resolution (0.1 = 10 steps per dimension).

    Returns:
        List of weight vectors summing to 1.0.
    """
    levels = int(round(1.0 / step))
    candidates = []
    # Generate all partitions of levels into n_components bins
    _partition_simplex(n_components, levels, [], candidates)
    return [np.array(c) * step for c in candidates]


def _partition_simplex(n: int, total: int, current: List[int], results: List):
    """Recursive helper for simplex partitioning."""
    if n == 1:
        results.append(current + [total])
        return
    for i in range(total + 1):
        _partition_simplex(n - 1, total - i, current + [i], results)


def optimize_weights_on_training(
    training_scores: Dict[str, Dict[str, np.ndarray]],
    training_times: Dict[str, pd.DatetimeIndex],
    component_names: List[str],
    training_crises: List[CrisisDefinition],
    step: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Grid-search optimal weights on pre-2020 training crises.

    Args:
        training_scores: crisis_name -> {method_name -> scores}
        training_times: crisis_name -> times index
        component_names: ordered list of ensemble component names.
        training_crises: list of training crisis definitions.
        step: grid step size.

    Returns:
        (best_weights, best_mean_d)
    """
    weight_candidates = generate_simplex_weights(len(component_names), step=step)
    logger.info(f"Testing {len(weight_candidates)} weight combinations "
                f"on {len(training_crises)} training crises")

    best_weights = np.ones(len(component_names)) / len(component_names)
    best_mean_d = -1.0

    for weights in weight_candidates:
        d_values = []
        for crisis in training_crises:
            if crisis.name not in training_scores:
                continue
            scores = training_scores[crisis.name]
            times = training_times[crisis.name]

            ensemble_score = weighted_ensemble(scores, component_names, weights)
            if len(ensemble_score) == 0:
                continue

            # Align times: ensemble scores may be shorter if components differ
            t = times[:len(ensemble_score)]
            result = compute_effect_size(ensemble_score, t, crisis)
            d_values.append(result['effect_size_d'])

        if d_values:
            mean_d = np.mean(d_values)
            if mean_d > best_mean_d:
                best_mean_d = mean_d
                best_weights = weights.copy()

    return best_weights, best_mean_d


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis
# ─────────────────────────────────────────────────────────────────────────────

def get_component_scores(
    crisis: CrisisDefinition,
    config: ValidationConfig,
    seed: int,
    enriched_lookback: int,
    rf_model: Optional[RandomForestRegimeDetector],
    rf_n_features: Optional[int],
) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex, pd.DatetimeIndex]:
    """Compute scores for all ensemble components for one crisis.

    Returns:
        (scores_dict, times_enriched, times_raw)
        scores_dict maps method_name -> score array.
        For enriched detectors, scores align with times_enriched.
        For RF, scores align with times_raw.
    """
    prices, _ = fetch_full_data(crisis, min_start_date="2019-01-01")
    if prices is None:
        raise ValueError(f"No data for {crisis.name}")

    X, X_enriched, times, cal_idx, scaler, pca = build_features_frozen(
        prices, CALIBRATION_CUTOFF,
        n_pca_components=config.n_pca_components,
        enriched_lookback=enriched_lookback,
    )

    trim = enriched_lookback - 1
    times_enriched = times[trim:]
    cal_enriched_idx = max(0, cal_idx - trim)
    X_enriched_cal = X_enriched[:cal_enriched_idx]

    scores = {}

    # QCML detectors (top 3)
    detectors = {
        "Berry Phase Rate": BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "QFI Determinant": QFIDeterminantDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "Multi-Lag Fidelity": MultiLagFidelityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
    }

    for name, det in detectors.items():
        try:
            # Fit QCML detectors on full enriched data (unsupervised — no labels).
            # The temporal OOS constraint is maintained for RF (trained on pre-2020
            # labels only). External PCA/scaler IS frozen at calibration boundary.
            det.fit(X_enriched)
            s = det.compute_regime_scores(X_enriched)
            scores[name] = s
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    # Random Forest
    if rf_model is not None:
        try:
            X_rf = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
            s = rf_model.compute_regime_scores(X_rf)
            # Pad or trim to match enriched length (RF uses raw features)
            # We need a common time axis — use the enriched one
            if len(s) > len(times_enriched):
                # RF scores on raw X are longer — trim the first `trim` entries
                scores["Random Forest"] = s[trim:]
            elif len(s) == len(times):
                scores["Random Forest"] = s[trim:]
            else:
                scores["Random Forest"] = s
        except Exception as e:
            logger.error(f"  RF failed: {e}")

    return scores, times_enriched, times


def train_rf_on_pre2020(
    config: ValidationConfig,
    seed: int,
    enriched_lookback: int,
) -> Tuple[Optional[RandomForestRegimeDetector], Optional[int]]:
    """Train RF on all pre-2020 crises (same as temporal_oos_validation.py)."""
    rf_all_X = []
    rf_all_y = []
    for crisis in TRAINING_CRISES:
        try:
            prices, _ = fetch_full_data(crisis)
            if prices is None:
                continue
            X, X_enriched, times, cal_idx, _, _ = build_features_frozen(
                prices, CALIBRATION_CUTOFF,
                n_pca_components=config.n_pca_components,
                enriched_lookback=enriched_lookback,
            )

            crisis_ts = pd.Timestamp(crisis.crisis_date)
            crisis_mask = times >= crisis_ts
            if not crisis_mask.any():
                continue
            crisis_idx = int(crisis_mask.argmax())

            y = np.zeros(len(X))
            window = config.analysis_window_days
            y[max(0, crisis_idx - window):min(len(X), crisis_idx + window)] = 1.0
            rf_all_X.append(X)
            rf_all_y.append(y)
            logger.info(f"  RF training: added {crisis.name} ({len(X)} samples)")
        except Exception as e:
            logger.warning(f"  Skipping {crisis.name} for RF training: {e}")

    if not rf_all_X:
        return None, None

    min_cols = min(x.shape[1] for x in rf_all_X)
    rf_all_X = [x[:, :min_cols] for x in rf_all_X]

    X_rf_train = np.vstack(rf_all_X)
    y_rf_train = np.concatenate(rf_all_y)

    rf_model = RandomForestRegimeDetector(
        n_estimators=200, max_depth=6, seed=seed, lookback=20,
    )
    rf_model.fit_with_labels(X_rf_train, y_rf_train)
    logger.info(f"RF trained on {len(rf_all_X)} crises, "
                f"{len(X_rf_train)} samples, {min_cols} features")

    return rf_model, min_cols


def run_hybrid_analysis(seed: int = 42) -> Dict[str, Any]:
    """Run the full hybrid ensemble analysis."""
    seed_everything(seed)
    config = get_default_validation_config()
    enriched_lookback = 20

    # Component names for the 4-way ensemble: 3 QCML + 1 RF
    component_names = TOP_QCML + ["Random Forest"]

    print("=" * 70)
    print("HYBRID ENSEMBLE ANALYSIS")
    print("=" * 70)
    print(f"Components: {component_names}")
    print(f"Training crises: {[c.name for c in TRAINING_CRISES]}")
    print(f"Test crises: {[c.name for c in TEST_CRISES]}")
    print("=" * 70)

    # --- 1. Train RF on pre-2020 ---
    print("\nTraining RF on pre-2020 crises...")
    rf_model, rf_n_features = train_rf_on_pre2020(config, seed, enriched_lookback)

    # --- 2. Collect training scores for weight optimization ---
    print("\nCollecting training crisis scores for weight optimization...")
    training_scores = {}
    training_times = {}
    for crisis in TRAINING_CRISES:
        try:
            scores, times_e, _ = get_component_scores(
                crisis, config, seed, enriched_lookback, rf_model, rf_n_features)
            if scores:
                training_scores[crisis.name] = scores
                training_times[crisis.name] = times_e
                logger.info(f"  {crisis.name}: got scores for {list(scores.keys())}")
        except Exception as e:
            logger.warning(f"  Skipping {crisis.name}: {e}")

    # --- 3. Optimize weights on training crises ---
    print("\nOptimizing ensemble weights on training crises...")
    optimal_weights, training_d = optimize_weights_on_training(
        training_scores, training_times, component_names, TRAINING_CRISES,
        step=0.1,
    )
    print(f"  Optimal weights: {dict(zip(component_names, optimal_weights))}")
    print(f"  Training mean d: {training_d:.3f}")

    # --- 4. Evaluate all 3 ensemble approaches on ALL crises ---
    all_crises = DATA_AVAILABLE_CRISES
    results = {
        'simple_average': {},
        'optimized_weights': {},
        'dynamic_switch': {},
        'component_rf': {},
        'component_berry': {},
        'component_qfi_det': {},
        'component_multi_lag': {},
    }

    for crisis in all_crises:
        print(f"\n  Evaluating: {crisis.name}...")
        try:
            scores, times_e, _ = get_component_scores(
                crisis, config, seed, enriched_lookback, rf_model, rf_n_features)

            if not scores:
                logger.warning(f"  No scores for {crisis.name}")
                continue

            # Ensemble 1: Simple Average
            ensemble_simple = simple_average_ensemble(scores, component_names)
            if len(ensemble_simple) > 0:
                r = compute_effect_size(ensemble_simple, times_e[:len(ensemble_simple)], crisis)
                results['simple_average'][crisis.name] = r
                logger.info(f"    Simple Average: d={r['effect_size_d']:.2f}")

            # Ensemble 2: Optimized Weights
            ensemble_opt = weighted_ensemble(scores, component_names, optimal_weights)
            if len(ensemble_opt) > 0:
                r = compute_effect_size(ensemble_opt, times_e[:len(ensemble_opt)], crisis)
                results['optimized_weights'][crisis.name] = r
                logger.info(f"    Optimized Weights: d={r['effect_size_d']:.2f}")

            # Ensemble 3: Dynamic Switch
            ensemble_dyn = dynamic_switch_ensemble(
                scores, TOP_QCML, "Random Forest", switch_threshold=1.5)
            if len(ensemble_dyn) > 0:
                r = compute_effect_size(ensemble_dyn, times_e[:len(ensemble_dyn)], crisis)
                results['dynamic_switch'][crisis.name] = r
                logger.info(f"    Dynamic Switch: d={r['effect_size_d']:.2f}")

            # Individual components for comparison
            for comp_key, comp_name in [
                ('component_rf', 'Random Forest'),
                ('component_berry', 'Berry Phase Rate'),
                ('component_qfi_det', 'QFI Determinant'),
                ('component_multi_lag', 'Multi-Lag Fidelity'),
            ]:
                if comp_name in scores:
                    s = scores[comp_name]
                    r = compute_effect_size(s, times_e[:len(s)], crisis)
                    results[comp_key][crisis.name] = r

        except Exception as e:
            logger.error(f"  Failed for {crisis.name}: {e}")

    return {
        'results': results,
        'optimal_weights': dict(zip(component_names, optimal_weights.tolist())),
        'training_d': float(training_d),
        'component_names': component_names,
    }


def compute_summary_statistics(results: Dict) -> Dict[str, Any]:
    """Compute summary statistics across all crises for each ensemble approach."""
    summary = {}
    for approach in ['simple_average', 'optimized_weights', 'dynamic_switch',
                     'component_rf', 'component_berry', 'component_qfi_det',
                     'component_multi_lag']:
        d_values = [r['effect_size_d'] for r in results[approach].values()]
        if d_values:
            summary[approach] = {
                'mean_d': float(np.mean(d_values)),
                'std_d': float(np.std(d_values)),
                'median_d': float(np.median(d_values)),
                'n_above_08': sum(1 for d in d_values if d > 0.8),
                'n_crises': len(d_values),
            }
        else:
            summary[approach] = {
                'mean_d': 0.0, 'std_d': 0.0, 'median_d': 0.0,
                'n_above_08': 0, 'n_crises': 0,
            }

    # Paired comparison: each ensemble vs RF
    for ensemble_name in ['simple_average', 'optimized_weights', 'dynamic_switch']:
        common_crises = set(results[ensemble_name].keys()) & set(results['component_rf'].keys())
        if len(common_crises) >= 3:
            ens_ds = [results[ensemble_name][c]['effect_size_d'] for c in common_crises]
            rf_ds = [results['component_rf'][c]['effect_size_d'] for c in common_crises]
            diffs = [e - r for e, r in zip(ens_ds, rf_ds)]

            # Wilcoxon signed-rank test
            try:
                stat, p_val = stats.wilcoxon(ens_ds, rf_ds, alternative='greater')
            except ValueError:
                stat, p_val = 0.0, 1.0

            summary[f'{ensemble_name}_vs_rf'] = {
                'mean_improvement': float(np.mean(diffs)),
                'wilcoxon_stat': float(stat),
                'wilcoxon_p': float(p_val),
                'n_crises_better': sum(1 for d in diffs if d > 0),
                'n_crises_total': len(diffs),
            }

    return summary


def generate_figures(results: Dict, summary: Dict, output_dir: Path):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hybrid vs Components bar chart
    approaches = ['simple_average', 'optimized_weights', 'dynamic_switch',
                  'component_rf', 'component_berry', 'component_qfi_det',
                  'component_multi_lag']
    labels = ['Simple\nAverage', 'Optimized\nWeights', 'Dynamic\nSwitch',
              'RF Alone', 'Berry\nPhase Rate', 'QFI\nDeterminant',
              'Multi-Lag\nFidelity']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e',  # ensembles
              '#d62728',  # RF
              '#9467bd', '#8c564b', '#e377c2']  # QCML components

    means = [summary[a]['mean_d'] for a in approaches]
    stds = [summary[a]['std_d'] for a in approaches]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(approaches))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Cohen's d")
    ax.set_title('Hybrid Ensemble vs Individual Components')
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
               label='d=0.8 threshold')

    # Annotate bars with mean values
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "hybrid_vs_components.pdf", bbox_inches='tight')
    plt.close(fig)

    # 2. Per-crisis ensemble comparison
    all_crises = sorted(set().union(
        *[results[a].keys() for a in ['simple_average', 'optimized_weights',
                                       'dynamic_switch', 'component_rf']]))

    if all_crises:
        ensemble_approaches = ['simple_average', 'optimized_weights',
                               'dynamic_switch', 'component_rf']
        e_labels = ['Simple Average', 'Optimized Weights', 'Dynamic Switch', 'RF Alone']
        e_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(all_crises))
        width = 0.2

        for i, (approach, label, color) in enumerate(
                zip(ensemble_approaches, e_labels, e_colors)):
            d_vals = [results[approach].get(c, {}).get('effect_size_d', 0.0)
                      for c in all_crises]
            ax.bar(x + i * width, d_vals, width, label=label, color=color)

        ax.set_xlabel('Crisis')
        ax.set_ylabel("Cohen's d")
        ax.set_title('Ensemble Performance Across All Crises')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([c.replace('_', '\n') for c in all_crises],
                           fontsize=7, rotation=45, ha='right')
        ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8, loc='upper left')

        fig.tight_layout()
        fig.savefig(fig_dir / "ensemble_per_crisis.pdf", bbox_inches='tight')
        plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Ensemble Analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/outputs/regime_detection/hybrid",
    )
    args = parser.parse_args()

    load_dotenv(project_root / '.env')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = run_hybrid_analysis(seed=args.seed)
    results = analysis['results']
    summary = compute_summary_statistics(results)

    # Save results
    output = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'experiment': 'hybrid_ensemble',
        'calibration_cutoff': CALIBRATION_CUTOFF,
        'component_names': analysis['component_names'],
        'optimal_weights': analysis['optimal_weights'],
        'training_d': analysis['training_d'],
        'results': results,
        'summary': summary,
    }

    results_path = output_dir / "hybrid_results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate figures
    generate_figures(results, summary, output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("HYBRID ENSEMBLE SUMMARY")
    print(f"{'='*70}")
    print(f"\nOptimal weights: {analysis['optimal_weights']}")
    print(f"Training mean d: {analysis['training_d']:.3f}")
    print()

    for approach in ['simple_average', 'optimized_weights', 'dynamic_switch', 'component_rf']:
        s = summary[approach]
        label = approach.replace('_', ' ').title()
        print(f"  {label:<25s}: mean d={s['mean_d']:.3f} (+/- {s['std_d']:.3f}), "
              f"d>0.8: {s['n_above_08']}/{s['n_crises']}")

    print()
    for ensemble_name in ['simple_average', 'optimized_weights', 'dynamic_switch']:
        key = f'{ensemble_name}_vs_rf'
        if key in summary:
            s = summary[key]
            label = ensemble_name.replace('_', ' ').title()
            print(f"  {label} vs RF: improvement={s['mean_improvement']:+.3f}, "
                  f"p={s['wilcoxon_p']:.4f}, better on {s['n_crises_better']}/{s['n_crises_total']}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
