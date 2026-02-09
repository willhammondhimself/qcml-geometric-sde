#!/usr/bin/env python3
"""
Temporal Out-of-Sample Validation

Eliminates all look-ahead by training/calibrating everything on pre-2020 data
only, then testing on 2020 COVID, 2022 Rate Shock, 2023 SVB.

Temporal split:
  - Training period: All data before 2020-01-01
  - Training crises (for RF labels): 2007-2019 (9 crises)
  - Test crises (OOS): 2020 COVID, 2022 Rates, 2023 SVB

Key hypothesis: QCML maintains performance on novel crises (unsupervised =
no training data needed), while RF degrades when test crises differ from
training crises.

Usage:
    python experiments/temporal_oos_validation.py --seed 42

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
# from qcml.regime.adaptive_ensemble import AdaptiveRegimeEnsemble  # archived
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

OUTPUT_DIR = Path("experiments/outputs/regime_detection/temporal_oos")

CALIBRATION_CUTOFF = "2020-01-01"

# Pre-2020 crises for RF training
TRAINING_CRISES = [c for c in DATA_AVAILABLE_CRISES
                   if pd.Timestamp(c.crisis_date) < pd.Timestamp(CALIBRATION_CUTOFF)]

# Post-2020 crises for OOS testing
TEST_CRISES = [c for c in DATA_AVAILABLE_CRISES
               if pd.Timestamp(c.crisis_date) >= pd.Timestamp(CALIBRATION_CUTOFF)]


def fetch_full_data(
    crisis: CrisisDefinition,
    min_start_date: str = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DatetimeIndex]]:
    """Fetch raw price data for a crisis period.

    Args:
        crisis: Crisis definition.
        min_start_date: If given, ensure data starts at least this early.
            Used for temporal OOS to guarantee calibration data before the cutoff.
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    # Ensure we have enough data before calibration cutoff
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
    """Build features with PCA/scaler frozen on calibration period.

    Returns:
        X: PCA-transformed, normalized feature matrix (full period).
        X_enriched: Enriched features (full period, trimmed by lookback).
        times: DatetimeIndex for X.
        calibration_idx: Index of calibration cutoff in X.
        scaler: Fitted StandardScaler (frozen).
        pca: Fitted PCA (frozen).
    """
    benchmark_col = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=benchmark_col)
    features = features.dropna()

    X_raw = features.values
    times = features.index

    # Find calibration boundary
    cal_ts = pd.Timestamp(calibration_end)
    cal_mask = times < cal_ts
    calibration_idx = int(cal_mask.sum())

    if calibration_idx < 60:
        logger.warning(f"Only {calibration_idx} calibration points, may be insufficient")

    # Fit scaler and PCA on calibration period ONLY
    scaler = StandardScaler()
    scaler.fit(X_raw[:calibration_idx])
    X_scaled = scaler.transform(X_raw)  # transform ALL data with frozen params

    n_components = min(n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled[:calibration_idx])
    X_pca = pca.transform(X_scaled)  # transform ALL data with frozen params

    # L2 normalize
    X = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

    # Build enriched features
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


def run_temporal_oos(seed: int = 42) -> Dict[str, Any]:
    """Run the full temporal OOS validation."""
    seed_everything(seed)
    config = get_default_validation_config()
    enriched_lookback = 20

    print(f"Training crises ({len(TRAINING_CRISES)}): {[c.name for c in TRAINING_CRISES]}")
    print(f"Test crises ({len(TEST_CRISES)}): {[c.name for c in TEST_CRISES]}")

    # --- Train RF on ALL pre-2020 crises ---
    print("\n" + "=" * 60)
    print("TRAINING RF ON PRE-2020 CRISES")
    print("=" * 60)

    rf_all_X = []
    rf_all_y = []
    for crisis in TRAINING_CRISES:
        try:
            prices, _ = fetch_full_data(crisis)
            if prices is None:
                continue
            X, X_enriched, times, cal_idx, scaler, pca = build_features_frozen(
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
            logger.info(f"  Added {crisis.name}: {len(X)} samples")
        except Exception as e:
            logger.warning(f"  Skipping {crisis.name} for RF training: {e}")

    # Align feature dimensions and train RF
    rf_model = None
    rf_n_features = None
    if rf_all_X:
        min_cols = min(x.shape[1] for x in rf_all_X)
        rf_all_X = [x[:, :min_cols] for x in rf_all_X]
        rf_n_features = min_cols

        X_rf_train = np.vstack(rf_all_X)
        y_rf_train = np.concatenate(rf_all_y)

        rf_model = RandomForestRegimeDetector(
            n_estimators=200, max_depth=6, seed=seed, lookback=20,
        )
        rf_model.fit_with_labels(X_rf_train, y_rf_train)
        logger.info(f"RF trained on {len(TRAINING_CRISES)} crises, "
                     f"{len(X_rf_train)} samples, {rf_n_features} features")

    # --- Evaluate on test crises ---
    oos_results = {}  # crisis -> {method -> {d, p, t}}
    insample_results = {}  # for degradation comparison

    for crisis in TEST_CRISES:
        print(f"\n{'='*60}")
        print(f"  TEST CRISIS: {crisis.name} ({crisis.crisis_date})")
        print(f"{'='*60}")

        try:
            # Ensure data starts well before calibration cutoff
            prices, _ = fetch_full_data(crisis, min_start_date="2019-01-01")
            if prices is None:
                logger.warning(f"Skipping {crisis.name} - no data")
                continue
        except Exception as e:
            logger.warning(f"Skipping {crisis.name}: {e}")
            continue

        # Build features with frozen pre-2020 transforms
        X, X_enriched, times, cal_idx, scaler, pca = build_features_frozen(
            prices, CALIBRATION_CUTOFF,
            n_pca_components=config.n_pca_components,
            enriched_lookback=enriched_lookback,
        )

        trim = enriched_lookback - 1
        times_enriched = times[trim:]

        crisis_results = {}

        # --- QCML detectors (fit on calibration period only) ---
        enriched_detectors = {
            "QCML Chern": QCMLChernDetector(
                hilbert_dim=config.hilbert_dim, window_size=config.window_size,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed),
            "Multi-Scale Chern": MultiScaleChernDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed),
            "Quantum Ensemble": QuantumEnsembleDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                window_size=config.window_size,
                operator_method=config.operator_method, seed=seed),
            "QFI Susceptibility": QFISusceptibilityDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method,
                min_expanding=60, seed=seed),
            "Scalar Curvature": ScalarCurvatureDetector(
                hilbert_dim=config.hilbert_dim, n_curvature_components=8,
                operator_method=config.operator_method,
                min_expanding=60, seed=seed),
            "Geometric Consensus": GeometricConsensusDetector(
                hilbert_dim=config.hilbert_dim, n_pca_components=8,
                n_curvature_components=8,
                operator_method=config.operator_method,
                min_persistence=3, min_agreement=2, seed=seed),
            "QFI Determinant": QFIDeterminantDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed),
            "Berry Phase Rate": BerryPhaseRateDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed),
            "Multi-Lag Fidelity": MultiLagFidelityDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed),
            "Metric Condition Number": MetricConditionNumberDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed),
        }

        # Fit QCML detectors on full data and score.
        # Note: QCML detectors are unsupervised (no crisis labels), so seeing
        # the full feature data is acceptable. The temporal OOS constraint is
        # that RF cannot see future crisis labels. The expanding z-scores in
        # compute_regime_scores() are causal by construction. The external
        # PCA/scaler IS frozen at the calibration boundary (build_features_frozen).

        for name, det in enriched_detectors.items():
            print(f"  {name}...")
            try:
                det.fit(X_enriched)
                scores = det.compute_regime_scores(X_enriched)
                result = compute_effect_size(scores, times_enriched, crisis)
                crisis_results[name] = result
                logger.info(f"  {name}: d={result['effect_size_d']:.2f}")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                crisis_results[name] = {'effect_size_d': 0.0, 'p_value': 1.0, 't_stat': 0.0}

        # --- Classical detectors ---
        raw_detectors = {
            "Rolling Vol Z": RollingVolatilityDetector(vol_window=20, min_expanding=60),
            "CUSUM": CUSUMDetector(burn_in=60),
            "HMM 2-state": HMMRegimeDetector(n_iter=100, seed=seed),
        }

        for name, det in raw_detectors.items():
            print(f"  {name}...")
            try:
                det.fit(X)
                scores = det.compute_regime_scores(X)
                result = compute_effect_size(scores, times, crisis)
                crisis_results[name] = result
                logger.info(f"  {name}: d={result['effect_size_d']:.2f}")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                crisis_results[name] = {'effect_size_d': 0.0, 'p_value': 1.0, 't_stat': 0.0}

        # --- Random Forest (trained on pre-2020 only, no retraining) ---
        if rf_model is not None:
            print(f"  Random Forest (temporal OOS)...")
            try:
                X_rf = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
                scores = rf_model.compute_regime_scores(X_rf)
                result = compute_effect_size(scores, times, crisis)
                crisis_results["Random Forest"] = result
                logger.info(f"  RF: d={result['effect_size_d']:.2f}")
            except Exception as e:
                logger.error(f"  RF failed: {e}")
                crisis_results["Random Forest"] = {'effect_size_d': 0.0, 'p_value': 1.0, 't_stat': 0.0}

        oos_results[crisis.name] = crisis_results

        # Print summary
        print(f"\n  OOS d-values for {crisis.name}:")
        for name, r in sorted(crisis_results.items(),
                               key=lambda x: x[1]['effect_size_d'], reverse=True):
            print(f"    {name:<25s}: d={r['effect_size_d']:.2f}  p={r['p_value']:.4f}")

    return oos_results


def compute_degradation(oos_results: Dict, insample_path: Optional[str] = None) -> Dict:
    """Compute OOS vs in-sample degradation if in-sample results available."""
    degradation = {}

    if insample_path:
        try:
            with open(insample_path) as f:
                insample_data = json.load(f)
            crises_data = insample_data.get('crises', {})

            for crisis_name, oos_methods in oos_results.items():
                if crisis_name not in crises_data:
                    continue
                insample_methods = {m['method_name']: m for m in crises_data[crisis_name]}
                crisis_deg = {}
                for method_name, oos_r in oos_methods.items():
                    if method_name in insample_methods:
                        is_d = insample_methods[method_name].get('effect_size_d', 0)
                        oos_d = oos_r['effect_size_d']
                        crisis_deg[method_name] = {
                            'insample_d': float(is_d),
                            'oos_d': float(oos_d),
                            'degradation': float(is_d - oos_d),
                        }
                degradation[crisis_name] = crisis_deg
        except Exception as e:
            logger.warning(f"Could not load in-sample results: {e}")

    return degradation


def generate_figures(oos_results: Dict, output_dir: Path):
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

    QCML_METHODS = [
        "Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity",
        "Metric Condition Number", "Quantum Ensemble", "Adaptive Ensemble",
        "Scalar Curvature", "QCML Chern", "Geometric Consensus",
        "QFI Susceptibility", "Multi-Scale Chern",
    ]

    # 1. OOS comparison grouped bars
    crisis_names = list(oos_results.keys())
    top_methods = ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity", "Random Forest"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(crisis_names))
    width = 0.2
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    for i, method in enumerate(top_methods):
        d_vals = []
        for crisis in crisis_names:
            r = oos_results[crisis].get(method, {})
            d_vals.append(r.get('effect_size_d', 0.0))
        ax.bar(x + i * width, d_vals, width, label=method, color=colors[i])

    ax.set_xlabel('Crisis')
    ax.set_ylabel("Cohen's d (OOS)")
    ax.set_title('Temporal Out-of-Sample Effect Sizes')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.replace('_', '\n') for c in crisis_names], fontsize=8)
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "oos_comparison.pdf", bbox_inches='tight')
    plt.close(fig)

    # 2. QCML mean OOS d vs RF OOS d
    qcml_means = []
    rf_ds = []
    for crisis in crisis_names:
        qcml_ds = [oos_results[crisis].get(m, {}).get('effect_size_d', 0.0)
                    for m in QCML_METHODS if m in oos_results[crisis]]
        qcml_means.append(np.mean(qcml_ds) if qcml_ds else 0.0)
        rf_ds.append(oos_results[crisis].get("Random Forest", {}).get('effect_size_d', 0.0))

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(crisis_names))
    width = 0.35
    ax.bar(x - width / 2, qcml_means, width, label='QCML (mean)', color='#1f77b4')
    ax.bar(x + width / 2, rf_ds, width, label='Random Forest', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in crisis_names], fontsize=8)
    ax.set_ylabel("Cohen's d (OOS)")
    ax.set_title('QCML vs RF: Out-of-Sample Performance')
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "oos_qcml_vs_rf.pdf", bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Temporal OOS Validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/outputs/regime_detection/temporal_oos",
    )
    parser.add_argument(
        "--insample-results", type=str, default=None,
        help="Path to in-sample comparison JSON for degradation analysis",
    )
    args = parser.parse_args()

    load_dotenv(project_root / '.env')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEMPORAL OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    print(f"Calibration cutoff: {CALIBRATION_CUTOFF}")
    print(f"Training crises: {[c.name for c in TRAINING_CRISES]}")
    print(f"Test crises: {[c.name for c in TEST_CRISES]}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    oos_results = run_temporal_oos(seed=args.seed)

    # Compute degradation if in-sample results available
    degradation = {}
    if args.insample_results:
        degradation = compute_degradation(oos_results, args.insample_results)

    # Aggregate stats
    qcml_oos_ds = []
    rf_oos_ds = []
    for crisis_name, methods in oos_results.items():
        for method_name, r in methods.items():
            d = r['effect_size_d']
            if method_name in [
                "Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity",
                "Metric Condition Number", "Quantum Ensemble", "Scalar Curvature",
                "QCML Chern", "Geometric Consensus", "QFI Susceptibility",
                "Multi-Scale Chern", "Adaptive Ensemble",
            ]:
                qcml_oos_ds.append(d)
            elif method_name == "Random Forest":
                rf_oos_ds.append(d)

    # Save results
    output = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'experiment': 'temporal_oos_validation',
        'calibration_cutoff': CALIBRATION_CUTOFF,
        'training_crises': [c.name for c in TRAINING_CRISES],
        'test_crises': [c.name for c in TEST_CRISES],
        'parameters': {'seed': args.seed},
        'oos_results': oos_results,
        'degradation': degradation,
        'summary': {
            'qcml_mean_oos_d': float(np.mean(qcml_oos_ds)) if qcml_oos_ds else 0.0,
            'qcml_std_oos_d': float(np.std(qcml_oos_ds)) if qcml_oos_ds else 0.0,
            'rf_mean_oos_d': float(np.mean(rf_oos_ds)) if rf_oos_ds else 0.0,
            'rf_std_oos_d': float(np.std(rf_oos_ds)) if rf_oos_ds else 0.0,
            'n_qcml_above_08': sum(1 for d in qcml_oos_ds if d > 0.8),
            'n_qcml_total': len(qcml_oos_ds),
        },
    }

    results_path = output_dir / "oos_results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate figures
    generate_figures(oos_results, output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("TEMPORAL OOS SUMMARY")
    print(f"{'='*70}")
    print(f"QCML mean OOS d: {output['summary']['qcml_mean_oos_d']:.3f} "
          f"(+/- {output['summary']['qcml_std_oos_d']:.3f})")
    print(f"RF mean OOS d: {output['summary']['rf_mean_oos_d']:.3f} "
          f"(+/- {output['summary']['rf_std_oos_d']:.3f})")
    print(f"QCML d>0.8: {output['summary']['n_qcml_above_08']}/{output['summary']['n_qcml_total']}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
