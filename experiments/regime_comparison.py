#!/usr/bin/env python3
"""
Head-to-Head Regime Detection Method Comparison

Runs 16 regime detection methods on up to 15 historical crises using the SAME
statistical pipeline (Welch's t-test, bootstrap CI, permutation test,
Bayes factor, Cohen's d raw + rank-normalized, F1/precision/recall) for an
apples-to-apples comparison.

Methods:
  1. QCML Chern — rolling Chern number (our method, unsupervised)
  2. Rolling Volatility Z-score — classical baseline (unsupervised)
  3. CUSUM — cumulative sum change-point detector (unsupervised)
  4. HMM (2-state Gaussian) — probabilistic regime model (unsupervised)
  5. Random Forest — supervised ML baseline (leave-one-crisis-out)
  6. Oracle RF — supervised oracle trained on ALL crises (in-sample)
  7. Multi-Scale Chern — multi-scale consensus (unsupervised)
  8. Quantum Ensemble — 4 quantum indicators combined (unsupervised)
  9. QFI Susceptibility — tr(quantum metric tensor) z-scored (unsupervised)
 10. Scalar Curvature — Ricci scalar of quantum metric manifold (unsupervised)
 11. Geometric Consensus — persistence + voting across 4 geometric methods (unsupervised)
 12. Adaptive Ensemble — crisis-type classifier + Fast/Slow/Shock detectors (unsupervised)
 13. QFI Determinant — log(det(quantum Fisher info matrix)) (unsupervised)
 14. Berry Phase Rate — rate of change of Berry curvature (unsupervised)
 15. Multi-Lag Fidelity — weighted fidelity across lags [1,3,5,10] (unsupervised)
 16. Metric Condition Number — anisotropy of quantum metric tensor (unsupervised)

Usage:
    python experiments/regime_comparison.py --crises all --seed 42
    python experiments/regime_comparison.py --crises extended --seed 42
    python experiments/regime_comparison.py --crises full --seed 42
    python experiments/regime_comparison.py --crises 2008_crisis --seed 42

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
# from qcml.regime.adaptive_ensemble import AdaptiveRegimeEnsemble  # archived
from experiments.crisis_config import (
    CrisisDefinition,
    ValidationConfig,
    ALL_CRISES,
    EXTENDED_CRISES,
    FULL_CRISES,
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
    get_crisis_by_name,
    config_to_dict,
)
from experiments.crisis_metrics import (
    compute_statistical_significance,
    compute_precision_recall,
    compute_lead_time,
)
from experiments.rigorous_crisis_validation import (
    bootstrap_confidence_interval,
    permutation_test,
    bayesian_t_test,
    fetch_real_crisis_data,
)

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    crisis: CrisisDefinition,
    config: ValidationConfig,
    enriched_lookback: int = 20,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DatetimeIndex], Optional[int]]:
    """Fetch data, build features, PCA, normalize. Also build enriched features.

    Returns:
        X: Feature matrix (T, n_pca_components), normalized. None if no data.
        X_enriched: Enriched feature matrix (T_trimmed, 4*n_pca_components),
            with rolling mean/std/min/max over lookback window. None if no data.
            Note: T_trimmed = T - enriched_lookback + 1.
        times: DatetimeIndex of length T (original). None if no data.
        crisis_idx: Integer index of the crisis date in ``times``. None if no data.
    """
    try:
        dataset = fetch_real_crisis_data(crisis)
    except (ValueError, Exception) as e:
        logger.warning(f"Skipping {crisis.name}: {e}")
        return None, None, None, None

    X_raw = dataset.X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(config.n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Build enriched features: rolling mean/std/min/max of PCA components
    from qcml_geometry import BaseRegimeDetector
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=enriched_lookback)

    times = dataset.times
    crisis_ts = pd.Timestamp(crisis.crisis_date)
    crisis_idx = int((times >= crisis_ts).argmax())

    logger.info(
        f"  {crisis.name}: T={len(X)}, features={X.shape[1]}, "
        f"enriched={X_enriched.shape}, crisis_idx={crisis_idx}"
    )
    return X, X_enriched, times, crisis_idx


# ---------------------------------------------------------------------------
# Adaptive threshold & persistence helpers
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(
    scores: np.ndarray,
    min_expanding: int = 60,
    quantile: float = 0.95,
) -> np.ndarray:
    """Expanding-window quantile threshold per time step.

    Adapts to each method's score distribution (HMM in [0,1] vs unbounded
    CUSUM) by computing the ``quantile``-th percentile over all scores up
    to each time step.

    Args:
        scores: 1-D regime score array (may contain NaN at start).
        min_expanding: Minimum observations before threshold is computed.
            Output is NaN for the first ``min_expanding`` valid entries.
        quantile: Percentile to use as threshold (0-1).

    Returns:
        1-D threshold array of the same length as *scores*.  NaN where
        not enough data is available.
    """
    n = len(scores)
    thresholds = np.full(n, np.nan)
    valid_count = 0
    for t in range(n):
        if np.isnan(scores[t]):
            continue
        valid_count += 1
        if valid_count >= min_expanding:
            past = scores[:t + 1]
            past_valid = past[~np.isnan(past)]
            thresholds[t] = np.percentile(past_valid, quantile * 100)
    return thresholds


def apply_persistence_filter(
    detected_mask: np.ndarray,
    min_persistence: int = 3,
) -> np.ndarray:
    """Keep only runs of consecutive True values >= ``min_persistence``.

    Removes single-day noise spikes from a boolean detection array.
    Modelled on ImprovedChernDetector.confirm_spikes().

    Args:
        detected_mask: Boolean 1-D array of raw detections.
        min_persistence: Minimum consecutive True values to retain.

    Returns:
        Boolean 1-D array with the same shape; isolated spikes removed.
    """
    out = np.zeros_like(detected_mask, dtype=bool)
    n = len(detected_mask)
    run_start = None

    for i in range(n):
        if detected_mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_persistence:
                    out[run_start:i] = True
                run_start = None

    # Handle run that extends to the end of the array
    if run_start is not None:
        run_len = n - run_start
        if run_len >= min_persistence:
            out[run_start:n] = True

    return out


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_method(
    detector: BaseRegimeDetector,
    X: np.ndarray,
    times: pd.DatetimeIndex,
    crisis_idx: int,
    crisis: CrisisDefinition,
    config: ValidationConfig,
    n_bootstrap: int = 10000,
    n_permutations: int = 5000,
    seed: int = 42,
    threshold_method: str = 'adaptive',
    min_persistence: int = 3,
) -> Dict[str, Any]:
    """Run the full statistical pipeline on one method's output.

    Args:
        threshold_method: How to threshold scores for detection events.
            'fixed' — original mean + 1.5*std (backward compat).
            'adaptive' — expanding-window 95th percentile per method.
            'hybrid' — max(fixed, adaptive) at each time step.
        min_persistence: Minimum consecutive days a detection must persist
            to count.  Set to 1 to disable persistence filtering.

    Returns a dict with method_name, delta_score, t_stat, p_value,
    bonferroni_p, effect_size_d, bootstrap_ci, permutation_p,
    bayes_factor, f1, precision, recall, lead_time_days, n_false_positives.
    """
    scores = detector.compute_regime_scores(X)

    # Mask NaNs for windowing
    valid_mask = ~np.isnan(scores)
    valid_scores = scores[valid_mask]
    valid_times = times[valid_mask]

    # Re-locate crisis index in valid-only series
    crisis_ts = pd.Timestamp(crisis.crisis_date)
    crisis_mask = valid_times >= crisis_ts
    if not crisis_mask.any():
        logger.warning(f"  {detector.name}: crisis date after data range")
        return _empty_result(detector.name)

    valid_crisis_idx = int(crisis_mask.argmax())

    window_days = config.analysis_window_days

    # Define crisis window: [crisis_idx - window : crisis_idx + window]
    crisis_window_start = max(0, valid_crisis_idx - window_days)
    crisis_window_end = min(len(valid_scores), valid_crisis_idx + window_days)

    # Create crisis mask
    crisis_mask_indices = np.zeros(len(valid_scores), dtype=bool)
    crisis_mask_indices[crisis_window_start:crisis_window_end] = True

    # Separate crisis vs. non-crisis scores
    scores_crisis = valid_scores[crisis_mask_indices]
    scores_non_crisis = valid_scores[~crisis_mask_indices]

    if len(scores_crisis) < 3 or len(scores_non_crisis) < 3:
        logger.warning(f"  {detector.name}: insufficient crisis/non-crisis data")
        return _empty_result(detector.name)

    # 1. Welch's t-test + Cohen's d (via existing helper)
    # Compare non-crisis (baseline) to crisis (elevated)
    sig = compute_statistical_significance(scores_non_crisis, scores_crisis)
    bonferroni_p = min(sig['p_value'] * 3, 1.0)  # 3 crises

    # 1b. Rank-normalized Cohen's d for fair cross-method comparison
    # Probability integral transform maps all methods to [0, 1] via empirical CDF
    ranked_all = rankdata(valid_scores) / len(valid_scores)
    ranked_crisis = ranked_all[crisis_mask_indices]
    ranked_non_crisis = ranked_all[~crisis_mask_indices]
    sig_norm = compute_statistical_significance(ranked_non_crisis, ranked_crisis)

    # 2. Bootstrap CI for delta score
    boot = bootstrap_confidence_interval(
        scores_non_crisis, scores_crisis,
        n_bootstrap=n_bootstrap, seed=seed,
    )

    # 3. Permutation test
    perm = permutation_test(
        scores_non_crisis, scores_crisis,
        n_permutations=n_permutations, seed=seed,
    )

    # 4. Bayes factor
    bf = bayesian_t_test(scores_non_crisis, scores_crisis)

    # 5. Detection metrics (threshold-based) — adaptive / fixed / hybrid
    fixed_threshold = np.nanmean(valid_scores) + 1.5 * np.nanstd(valid_scores)

    if threshold_method == 'fixed':
        detected_raw = valid_scores > fixed_threshold
        lead_threshold = fixed_threshold * 0.8
    elif threshold_method == 'adaptive':
        adaptive_thresholds = compute_adaptive_threshold(
            valid_scores, min_expanding=60, quantile=0.95,
        )
        detected_raw = valid_scores > adaptive_thresholds
        # Use adaptive threshold at crisis index for lead-time computation
        lead_threshold_val = adaptive_thresholds[valid_crisis_idx]
        lead_threshold = (lead_threshold_val * 0.8
                          if not np.isnan(lead_threshold_val)
                          else fixed_threshold * 0.8)
    else:  # 'hybrid'
        adaptive_thresholds = compute_adaptive_threshold(
            valid_scores, min_expanding=60, quantile=0.95,
        )
        hybrid_thresholds = np.where(
            np.isnan(adaptive_thresholds),
            fixed_threshold,
            np.maximum(fixed_threshold, adaptive_thresholds),
        )
        detected_raw = valid_scores > hybrid_thresholds
        lead_threshold_val = hybrid_thresholds[valid_crisis_idx]
        lead_threshold = lead_threshold_val * 0.8

    # Apply persistence filter
    detected_filtered = apply_persistence_filter(detected_raw, min_persistence)
    detected_indices = np.where(detected_filtered)[0].tolist()

    pr = compute_precision_recall(
        detected_indices, valid_crisis_idx, tolerance_days=window_days
    )

    # 6. Lead time
    lead_time = compute_lead_time(
        valid_scores, valid_times.values,
        crisis.crisis_date, threshold=lead_threshold,
    )

    return {
        'method_name': detector.name,
        'delta_score': float(sig['delta_chern']),
        't_stat': float(sig['t_statistic']),
        'p_value': float(sig['p_value']),
        'bonferroni_p': float(bonferroni_p),
        'effect_size_d': float(sig['effect_size']),
        'effect_size_d_normalized': float(sig_norm['effect_size']),
        'delta_score_normalized': float(sig_norm['delta_chern']),
        'p_value_normalized': float(sig_norm['p_value']),
        'bootstrap_ci': [float(boot.ci_lower), float(boot.ci_upper)],
        'bootstrap_se': float(boot.se),
        'permutation_p': float(perm.p_value),
        'bayes_factor': float(bf.bayes_factor),
        'bf_interpretation': bf.interpretation,
        'f1': float(pr['f1_score']),
        'precision': float(pr['precision']),
        'recall': float(pr['recall']),
        'n_false_positives': int(pr['n_false_positives']),
        'lead_time_days': lead_time,
        'threshold_method': threshold_method,
        'min_persistence': min_persistence,
    }


def _empty_result(method_name: str) -> Dict[str, Any]:
    return {
        'method_name': method_name,
        'delta_score': 0.0,
        't_stat': 0.0,
        'p_value': 1.0,
        'bonferroni_p': 1.0,
        'effect_size_d': 0.0,
        'effect_size_d_normalized': 0.0,
        'delta_score_normalized': 0.0,
        'p_value_normalized': 1.0,
        'bootstrap_ci': [0.0, 0.0],
        'bootstrap_se': 0.0,
        'permutation_p': 1.0,
        'bayes_factor': 1.0,
        'bf_interpretation': 'insufficient data',
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'n_false_positives': 0,
        'lead_time_days': None,
    }


# ---------------------------------------------------------------------------
# Random Forest training data (leave-one-crisis-out)
# ---------------------------------------------------------------------------

def _align_features(all_X: List[np.ndarray]) -> List[np.ndarray]:
    """Pad or truncate feature matrices to a common column count.

    Different crises may produce different PCA dimensions (because universe
    sizes differ).  We align to the minimum column count across all matrices
    so that ``np.vstack`` succeeds.
    """
    if not all_X:
        return all_X
    min_cols = min(x.shape[1] for x in all_X)
    return [x[:, :min_cols] for x in all_X]


def prepare_rf_training_data(
    test_crisis: CrisisDefinition,
    all_crises: List[CrisisDefinition],
    config: ValidationConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Leave-one-out: train RF on the other crises.

    Labels: 1 for analysis_window_days around each crisis, 0 otherwise.

    Returns (X_train, y_train, n_features) where n_features is the aligned
    column count (needed to truncate test data to match).
    """
    train_crises = [c for c in all_crises if c.name != test_crisis.name]
    all_X = []
    all_y = []

    for crisis in train_crises:
        X, _X_enriched, times, crisis_idx = prepare_data(crisis, config)

        # Skip if no data
        if X is None:
            logger.warning(f"Skipping {crisis.name} for RF training: no data available")
            continue

        y = np.zeros(len(X))
        window = config.analysis_window_days
        start = max(0, crisis_idx - window)
        end = min(len(X), crisis_idx + window)
        y[start:end] = 1
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No crisis data available for RF training")

    all_X = _align_features(all_X)
    n_features = all_X[0].shape[1]
    return np.vstack(all_X), np.concatenate(all_y), n_features


def prepare_oracle_rf_data(
    all_crises: List[CrisisDefinition],
    config: ValidationConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Oracle: train RF on ALL crises (fully in-sample).

    This is an intentionally unfair baseline to show that QCML (unsupervised)
    is competitive with a supervised oracle that has access to all crisis labels.

    Labels: 1 for analysis_window_days around each crisis, 0 otherwise.

    Returns (X_train, y_train, n_features) where n_features is the aligned
    column count (needed to truncate test data to match).
    """
    all_X = []
    all_y = []

    for crisis in all_crises:
        try:
            X, _X_enriched, times, crisis_idx = prepare_data(crisis, config)

            # Skip if no data
            if X is None:
                logger.warning(f"Skipping {crisis.name} for oracle RF: no data available")
                continue

            y = np.zeros(len(X))
            window = config.analysis_window_days
            start = max(0, crisis_idx - window)
            end = min(len(X), crisis_idx + window)
            y[start:end] = 1
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            logger.warning(f"Skipping {crisis.name} for oracle RF: {e}")
            continue

    if not all_X:
        raise ValueError("No crisis data available for oracle RF training")

    all_X = _align_features(all_X)
    n_features = all_X[0].shape[1]
    return np.vstack(all_X), np.concatenate(all_y), n_features


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

def run_full_comparison(
    crises: List[CrisisDefinition],
    config: ValidationConfig,
    n_bootstrap: int = 10000,
    n_permutations: int = 5000,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Run all 11 methods on all crises.

    Methods:
      1. QCML Chern (unsupervised, single-scale)
      2. Rolling Vol Z-score (unsupervised)
      3. CUSUM (unsupervised)
      4. HMM 2-state (unsupervised)
      5. Random Forest - LOCO (supervised, leave-one-crisis-out)
      6. Oracle RF (supervised, trained on ALL crises - in-sample)
      7. Multi-Scale Chern (unsupervised, 5 scales: 10-100 days)
      8. Quantum Ensemble (unsupervised, 4 indicators combined)
      9. QFI Susceptibility (unsupervised, tr(quantum metric tensor))
     10. Scalar Curvature (unsupervised, Ricci scalar of quantum metric manifold)
     11. Geometric Consensus (unsupervised, persistence + voting across 4 geometric methods)

    Returns dict mapping crisis name -> list of per-method result dicts.
    """
    results: Dict[str, List[Dict]] = {}

    # Prepare Oracle RF: train once on ALL crises (fully in-sample)
    print("\n" + "=" * 60)
    print("PREPARING ORACLE RF (trained on ALL crises)")
    print("=" * 60)
    oracle_rf = None
    oracle_rf_n_features = None
    try:
        X_oracle, y_oracle, oracle_rf_n_features = prepare_oracle_rf_data(crises, config)
        oracle_rf = RandomForestRegimeDetector(
            n_estimators=200, max_depth=6, seed=seed, lookback=20,
        )
        oracle_rf.fit_with_labels(X_oracle, y_oracle)
        logger.info(f"Oracle RF trained on {len(crises)} crises ({len(X_oracle)} samples, {oracle_rf_n_features} features)")
    except Exception as e:
        logger.error(f"Oracle RF training failed: {e}")
        oracle_rf = None

    enriched_lookback = 20  # must match RF's lookback for fair comparison

    for crisis in crises:
        print(f"\n{'='*60}")
        print(f"  Crisis: {crisis.name} ({crisis.crisis_date})")
        print(f"{'='*60}")

        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )

        # Skip crisis if no data available
        if X is None:
            logger.warning(f"Skipping {crisis.name} - no data available")
            continue

        # Trimmed times/crisis_idx for enriched features (lost lookback-1 rows)
        trim = enriched_lookback - 1
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        crisis_results: List[Dict] = []

        # 1. QCML Chern (uses enriched features)
        print(f"  Running QCML Chern...")
        det_chern = QCMLChernDetector(
            hilbert_dim=config.hilbert_dim,
            window_size=config.window_size,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        )
        det_chern.fit(X_enriched)
        crisis_results.append(evaluate_method(
            det_chern, X_enriched, times_enriched, crisis_idx_enriched,
            crisis, config, n_bootstrap, n_permutations, seed,
        ))

        # 2. Rolling Volatility Z-score
        print(f"  Running Rolling Vol Z...")
        det_vol = RollingVolatilityDetector(vol_window=20, min_expanding=60)
        det_vol.fit(X)
        crisis_results.append(evaluate_method(
            det_vol, X, times, crisis_idx, crisis, config,
            n_bootstrap, n_permutations, seed,
        ))

        # 3. CUSUM
        print(f"  Running CUSUM...")
        det_cusum = CUSUMDetector(burn_in=60)
        det_cusum.fit(X)
        crisis_results.append(evaluate_method(
            det_cusum, X, times, crisis_idx, crisis, config,
            n_bootstrap, n_permutations, seed,
        ))

        # 4. HMM 2-state
        print(f"  Running HMM 2-state...")
        det_hmm = HMMRegimeDetector(n_iter=100, seed=seed)
        det_hmm.fit(X)
        crisis_results.append(evaluate_method(
            det_hmm, X, times, crisis_idx, crisis, config,
            n_bootstrap, n_permutations, seed,
        ))

        # 5. Random Forest (leave-one-crisis-out)
        print(f"  Running Random Forest (leave-one-crisis-out)...")
        det_rf = RandomForestRegimeDetector(
            n_estimators=200, max_depth=6, seed=seed, lookback=20,
        )
        try:
            X_train, y_train, rf_n_features = prepare_rf_training_data(crisis, crises, config)
            det_rf.fit_with_labels(X_train, y_train)
            # Truncate test data to match training feature count
            X_rf_test = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
            crisis_results.append(evaluate_method(
                det_rf, X_rf_test, times, crisis_idx, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  RF failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Random Forest"))

        # 6. Oracle RF (trained on ALL crises, fully in-sample)
        if oracle_rf is not None:
            print(f"  Running Oracle RF (in-sample)...")
            try:
                # Need to give oracle_rf a custom name for display
                class OracleRFWrapper:
                    """Wrapper to give oracle RF a distinguishable name."""
                    def __init__(self, rf):
                        self._rf = rf

                    @property
                    def name(self):
                        return "Oracle RF (in-sample)"

                    def compute_regime_scores(self, X):
                        return self._rf.compute_regime_scores(X)

                oracle_wrapper = OracleRFWrapper(oracle_rf)
                # Truncate test data to match oracle training feature count
                X_oracle_test = X[:, :oracle_rf_n_features] if X.shape[1] > oracle_rf_n_features else X
                crisis_results.append(evaluate_method(
                    oracle_wrapper, X_oracle_test, times, crisis_idx, crisis, config,
                    n_bootstrap, n_permutations, seed,
                ))
            except Exception as e:
                logger.error(f"  Oracle RF failed for {crisis.name}: {e}")
                crisis_results.append(_empty_result("Oracle RF (in-sample)"))
        else:
            crisis_results.append(_empty_result("Oracle RF (in-sample)"))

        # 7. Multi-Scale Chern Consensus (uses enriched features)
        print(f"  Running Multi-Scale Chern...")
        det_multiscale = MultiScaleChernDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            det_multiscale.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_multiscale, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Multi-Scale Chern failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Multi-Scale Chern"))

        # 8. Quantum Ensemble (uses enriched features)
        print(f"  Running Quantum Ensemble...")
        det_ensemble = QuantumEnsembleDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            window_size=config.window_size,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            det_ensemble.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_ensemble, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Quantum Ensemble failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Quantum Ensemble"))

        # 9. QFI Susceptibility (uses enriched features)
        print(f"  Running QFI Susceptibility...")
        det_qfi = QFISusceptibilityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            min_expanding=60,
            seed=seed,
        )
        try:
            det_qfi.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_qfi, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  QFI Susceptibility failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("QFI Susceptibility"))

        # 10. Scalar Curvature (uses enriched features)
        print(f"  Running Scalar Curvature...")
        det_curv = ScalarCurvatureDetector(
            hilbert_dim=config.hilbert_dim,
            n_curvature_components=8,
            operator_method=config.operator_method,
            min_expanding=60,
            seed=seed,
        )
        try:
            det_curv.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_curv, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Scalar Curvature failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Scalar Curvature"))

        # 11. Geometric Consensus (uses enriched features)
        print(f"  Running Geometric Consensus...")
        det_consensus = GeometricConsensusDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=8,
            n_curvature_components=8,
            operator_method=config.operator_method,
            min_persistence=3,
            min_agreement=2,
            seed=seed,
        )
        try:
            det_consensus.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_consensus, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Geometric Consensus failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Geometric Consensus"))

        # 12. Adaptive Ensemble (uses enriched features)
        print(f"  Running Adaptive Ensemble...")
        n_pca_adaptive = min(20, X_enriched.shape[1])
        det_adaptive = AdaptiveRegimeEnsemble(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=n_pca_adaptive,
            n_curvature_components=8,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            # Fetch prices for crisis classification
            try:
                dataset = fetch_real_crisis_data(crisis)
                prices = dataset.prices
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]
            except Exception as price_err:
                logger.warning(f"  Could not fetch prices for {crisis.name}: {price_err}")
                prices = None

            det_adaptive.fit(X_enriched, prices=prices)
            crisis_results.append(evaluate_method(
                det_adaptive, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            import traceback
            logger.error(f"  Adaptive Ensemble failed for {crisis.name}: {e}")
            traceback.print_exc()
            crisis_results.append(_empty_result("Adaptive Ensemble"))

        # 13. QFI Determinant (uses enriched features)
        print(f"  Running QFI Determinant...")
        det_qfi_det = QFIDeterminantDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            det_qfi_det.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_qfi_det, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  QFI Determinant failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("QFI Determinant"))

        # 14. Berry Phase Rate (uses enriched features)
        print(f"  Running Berry Phase Rate...")
        det_berry = BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            det_berry.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_berry, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Berry Phase Rate failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Berry Phase Rate"))

        # 15. Multi-Lag Fidelity (uses enriched features)
        print(f"  Running Multi-Lag Fidelity...")
        det_fidelity = MultiLagFidelityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            det_fidelity.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_fidelity, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Multi-Lag Fidelity failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Multi-Lag Fidelity"))

        # 16. Metric Condition Number (uses enriched features)
        print(f"  Running Metric Condition Number...")
        det_kappa = MetricConditionNumberDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        )
        try:
            det_kappa.fit(X_enriched)
            crisis_results.append(evaluate_method(
                det_kappa, X_enriched, times_enriched,
                crisis_idx_enriched, crisis, config,
                n_bootstrap, n_permutations, seed,
            ))
        except Exception as e:
            logger.error(f"  Metric Condition Number failed for {crisis.name}: {e}")
            crisis_results.append(_empty_result("Metric Condition Number"))

        results[crisis.name] = crisis_results

    return results


# ---------------------------------------------------------------------------
# Formatting & output
# ---------------------------------------------------------------------------

def format_comparison_table(results: Dict[str, List[Dict]]) -> str:
    """Format publication-quality comparison table."""
    lines = []
    lines.append("=" * 90)
    lines.append("REGIME DETECTION METHOD COMPARISON")
    lines.append("=" * 90)

    for crisis_name, methods in results.items():
        lines.append(f"\n--- {crisis_name} ---")
        header = (
            f"  {'Method':<20s} {'|d|':>6s}  {'|d|N':>6s}  {'t-stat':>7s}  {'p-val':>8s}  "
            f"{'BF10':>9s}  {'F1':>5s}  {'Lead':>5s}  {'FP':>3s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 88)

        for m in methods:
            lead_str = str(m['lead_time_days']) if m['lead_time_days'] is not None else "N/A"
            bf_str = f"{m['bayes_factor']:.1e}" if m['bayes_factor'] > 1e3 else f"{m['bayes_factor']:.2f}"
            d_norm = m.get('effect_size_d_normalized', 0.0)
            lines.append(
                f"  {m['method_name']:<20s} {m['effect_size_d']:>6.2f}  {d_norm:>6.2f}  "
                f"{m['t_stat']:>7.2f}  {m['p_value']:>8.4f}  "
                f"{bf_str:>9s}  {m['f1']:>5.2f}  {lead_str:>5s}  "
                f"{m['n_false_positives']:>3d}"
            )

    # Aggregate summary
    lines.append(f"\n{'='*90}")
    lines.append("AGGREGATE SUMMARY")
    lines.append(f"{'='*90}")

    method_names = [m['method_name'] for m in list(results.values())[0]]
    header = (
        f"  {'Method':<20s} {'Avg |d|':>8s}  {'Avg |d|N':>9s}  "
        f"{'Avg p-val':>10s}  {'Wins':>5s}  {'Verdict':<30s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 90)

    for method_name in method_names:
        ds = []
        ds_norm = []
        ps = []
        wins = 0
        for crisis_name, methods in results.items():
            for m in methods:
                if m['method_name'] == method_name:
                    ds.append(m['effect_size_d'])
                    ds_norm.append(m.get('effect_size_d_normalized', 0.0))
                    ps.append(m['p_value'])
                    if m['p_value'] < 0.05 and m['effect_size_d'] > 0.8:
                        wins += 1

        avg_d = np.mean(ds) if ds else 0.0
        avg_d_norm = np.mean(ds_norm) if ds_norm else 0.0
        avg_p = np.mean(ps) if ps else 1.0

        if avg_d >= 0.8 and avg_p < 0.05:
            verdict = "Strong regime signal"
        elif avg_d >= 0.5 and avg_p < 0.1:
            verdict = "Moderate regime signal"
        elif avg_d >= 0.2:
            verdict = "Weak regime signal"
        else:
            verdict = "No signal"

        lines.append(
            f"  {method_name:<20s} {avg_d:>8.2f}  {avg_d_norm:>9.2f}  "
            f"{avg_p:>10.4f}  {wins:>5d}/{len(results)}  {verdict:<30s}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def save_results(
    results: Dict[str, List[Dict]],
    config: ValidationConfig,
    output_dir: str,
    seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> str:
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        'timestamp': timestamp,
        'experiment': 'regime_detection_comparison',
        'config': config_to_dict(config),
        'parameters': {
            'seed': seed,
            'n_bootstrap': n_bootstrap,
            'n_permutations': n_permutations,
        },
        'crises': results,
    }

    filepath = output_path / f"comparison_{timestamp}.json"
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Head-to-Head Regime Detection Comparison"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/outputs/regime_detection/results",
    )
    parser.add_argument(
        "--crises", type=str, default="all",
        help=(
            "Crisis selection: 'all' (original 3), 'extended' (8 total), "
            "'full' (15 total for statistical power), "
            "'available' (12 with Polygon data, 2007+), "
            "or specific crisis name (e.g., '2008_crisis')"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_dotenv(project_root / '.env')
    seed_everything(args.seed)

    config = get_default_validation_config()

    # Select crises based on --crises argument
    if args.crises == "all":
        crises = ALL_CRISES
    elif args.crises == "extended":
        crises = EXTENDED_CRISES
    elif args.crises == "full":
        crises = FULL_CRISES
    elif args.crises == "available":
        crises = DATA_AVAILABLE_CRISES
    else:
        # Assume it's a specific crisis name
        # Try full first, then extended, then all
        try:
            crises = [get_crisis_by_name(args.crises, use_full=True)]
        except ValueError:
            try:
                crises = [get_crisis_by_name(args.crises, use_extended=True)]
            except ValueError:
                try:
                    crises = [get_crisis_by_name(args.crises, use_extended=False)]
                except ValueError:
                    parser.error(
                        f"Unknown crisis: {args.crises}. "
                        f"Use 'all', 'extended', 'full', or one of: "
                        f"{[c.name for c in FULL_CRISES]}"
                    )

    print("=" * 90)
    print("HEAD-TO-HEAD REGIME DETECTION COMPARISON")
    print("=" * 90)
    print(f"Crises: {[c.name for c in crises]}")
    print(f"Methods: QCML Chern, Vol Z, CUSUM, HMM, RF(LOCO), Oracle RF, MultiScale, Ensemble, QFI, Scalar Curvature, Geometric Consensus")
    print(f"Bootstrap: n={args.n_bootstrap}, Permutations: n={args.n_permutations}")
    print(f"Seed: {args.seed}")
    print("=" * 90)

    results = run_full_comparison(
        crises, config,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    table = format_comparison_table(results)
    print(table)

    filepath = save_results(
        results, config, args.output_dir,
        args.seed, args.n_bootstrap, args.n_permutations,
    )
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
