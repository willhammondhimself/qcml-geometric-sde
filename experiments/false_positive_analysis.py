#!/usr/bin/env python3
"""
False Positive Rate / Specificity Analysis

Fills a critical gap: the paper only measures crisis sensitivity (Cohen's d)
but never tests how often methods fire falsely during calm periods.

Defines calm periods (VIX < 15, no drawdown > 5%), computes FPR, precision,
recall, F1 at multiple thresholds, and generates ROC / precision-recall curves.

Usage:
    python experiments/false_positive_analysis.py

Author: QCML Research
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score as sklearn_f1,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    DATA_AVAILABLE_CRISES,
    ValidationConfig,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    prepare_data,
    prepare_rf_training_data,
    seed_everything,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calm period definitions (VIX < 15, no drawdown > 5%)
# ---------------------------------------------------------------------------

CALM_PERIODS = [
    {
        "name": "2005-2006 Goldilocks",
        "start": "2005-01-03",
        "end": "2006-12-29",
        "description": "Low vol, steady growth pre-crisis",
    },
    {
        "name": "2012-2013 Recovery",
        "start": "2012-01-03",
        "end": "2013-12-31",
        "description": "Post-euro crisis, QE-fueled calm",
    },
    {
        "name": "2017 H1 Low Vol",
        "start": "2017-01-03",
        "end": "2017-06-30",
        "description": "Record low VIX period",
    },
    {
        "name": "2021 H1 Reopening",
        "start": "2021-01-04",
        "end": "2021-06-30",
        "description": "Post-vaccine reopening rally",
    },
]


def fetch_calm_period_data(
    period: dict, symbols: List[str] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Fetch SPY data for a calm period and build features.

    Returns:
        X: PCA-reduced feature matrix
        X_enriched: Enriched features (rolling mean/std/min/max)
        times: DatetimeIndex
    """
    if symbols is None:
        symbols = ["SPY"]

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found")

    source = PolygonDataSource(api_key=api_key)
    try:
        raw = source.fetch_equities(
            symbols, start_date=period["start"], end_date=period["end"]
        )
    except Exception as e:
        logger.warning(f"Failed to fetch {period['name']}: {e}")
        return None

    engine = MinimalFeatureEngine()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["close"].unstack("symbol") if "close" in raw.columns.get_level_values(0) else raw.xs("close", level=0, axis=1)
    else:
        prices = raw[["close"]] if "close" in raw.columns else raw

    try:
        feature_matrix = engine.create_feature_matrix(prices)
    except Exception:
        # Fallback: use close prices directly
        if hasattr(raw, "xs"):
            close = raw.xs("close", level=1, axis=1) if raw.columns.nlevels > 1 else raw["close"]
        else:
            close = raw["close"] if "close" in raw.columns else raw.iloc[:, 0]
        feature_matrix = pd.DataFrame(close)

    X_raw = feature_matrix.values
    times = feature_matrix.index

    # Standard pipeline: scale + PCA + normalize
    config = get_default_validation_config()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    n_components = min(config.n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)

    return X, X_enriched, times


def build_all_detectors(
    config: ValidationConfig, seed: int = 42
) -> List[BaseRegimeDetector]:
    """Instantiate all 16 detectors (excluding Oracle RF which needs special handling)."""
    detectors = [
        QCMLChernDetector(
            hilbert_dim=config.hilbert_dim,
            window_size=config.window_size,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        ),
        RollingVolatilityDetector(vol_window=20, min_expanding=60),
        CUSUMDetector(burn_in=60),
        HMMRegimeDetector(n_iter=100, seed=seed),
        MultiScaleChernDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        ),
        QuantumEnsembleDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            window_size=config.window_size,
            operator_method=config.operator_method,
            seed=seed,
        ),
        QFISusceptibilityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            min_expanding=60,
            seed=seed,
        ),
        ScalarCurvatureDetector(
            hilbert_dim=config.hilbert_dim,
            n_curvature_components=8,
            operator_method=config.operator_method,
            min_expanding=60,
            seed=seed,
        ),
        GeometricConsensusDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=8,
            n_curvature_components=8,
            operator_method=config.operator_method,
            min_persistence=3,
            min_agreement=2,
            seed=seed,
        ),
        QFIDeterminantDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        ),
        BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        ),
        MultiLagFidelityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        ),
        MetricConditionNumberDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            seed=seed,
        ),
    ]
    return detectors


def is_enriched_detector(det: BaseRegimeDetector) -> bool:
    """Check if detector needs enriched features."""
    return det.name not in ("Rolling Vol Z", "CUSUM", "HMM 2-state", "Random Forest")


def compute_fpr_at_thresholds(
    calm_scores: np.ndarray,
    crisis_scores: np.ndarray,
    n_thresholds: int = 200,
) -> Dict:
    """Compute FPR, precision, recall, F1 at multiple thresholds.

    Args:
        calm_scores: Scores during calm periods (should be low)
        crisis_scores: Scores during crisis windows (should be high)

    Returns:
        Dict with thresholds, fpr, precision, recall, f1, roc_auc, pr_auc
    """
    # Clean NaNs
    calm_valid = calm_scores[~np.isnan(calm_scores)]
    crisis_valid = crisis_scores[~np.isnan(crisis_scores)]

    if len(calm_valid) < 10 or len(crisis_valid) < 10:
        return {"error": "insufficient data"}

    # Build binary labels: 0 = calm, 1 = crisis
    y_true = np.concatenate([np.zeros(len(calm_valid)), np.ones(len(crisis_valid))])
    y_scores = np.concatenate([calm_valid, crisis_valid])

    # ROC curve
    fpr_roc, tpr_roc, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc_val = auc(fpr_roc, tpr_roc)

    # Precision-recall curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc_val = auc(pr_recall, pr_precision)

    # Compute F1 at multiple thresholds
    all_scores = np.sort(np.unique(y_scores))
    step = max(1, len(all_scores) // n_thresholds)
    sampled_thresholds = all_scores[::step]

    best_f1 = 0.0
    best_threshold = 0.0
    f1_at_thresholds = []

    for thresh in sampled_thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        fpr_val = fp / (fp + np.sum(y_true == 0)) if np.sum(y_true == 0) > 0 else 0.0

        f1_at_thresholds.append({
            "threshold": float(thresh),
            "fpr": float(fpr_val),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Overall FPR at 95th percentile threshold (standard)
    p95_threshold = np.percentile(calm_valid, 95)
    fpr_at_p95 = float(np.mean(calm_valid >= p95_threshold))
    recall_at_p95 = float(np.mean(crisis_valid >= p95_threshold))

    return {
        "roc_auc": float(roc_auc_val),
        "pr_auc": float(pr_auc_val),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "fpr_at_p95": fpr_at_p95,
        "recall_at_p95": recall_at_p95,
        "n_calm": int(len(calm_valid)),
        "n_crisis": int(len(crisis_valid)),
        "fpr_roc": [float(x) for x in fpr_roc.tolist()],
        "tpr_roc": [float(x) for x in tpr_roc.tolist()],
        "pr_precision": [float(x) for x in pr_precision.tolist()],
        "pr_recall": [float(x) for x in pr_recall.tolist()],
        "f1_curve": f1_at_thresholds,
    }


def generate_figures(results: Dict, output_dir: Path) -> None:
    """Generate PR curve and ROC figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        logger.warning("matplotlib not available, skipping figure generation")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- Figure 1: ROC curves for all methods ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (method_name, method_data) in enumerate(results.items()):
        if "error" in method_data:
            continue
        c = colors[i % 20]

        # ROC
        axes[0].plot(
            method_data["fpr_roc"],
            method_data["tpr_roc"],
            color=c,
            linewidth=1.2,
            label=f"{method_name} ({method_data['roc_auc']:.2f})",
        )

        # PR curve
        axes[1].plot(
            method_data["pr_recall"],
            method_data["pr_precision"],
            color=c,
            linewidth=1.2,
            label=f"{method_name} ({method_data['pr_auc']:.2f})",
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves: Crisis Detection vs. Calm Periods")
    axes[0].legend(fontsize=6, loc="lower right", ncol=2)

    baseline_precision = results.get(list(results.keys())[0], {}).get("n_crisis", 100) / (
        results.get(list(results.keys())[0], {}).get("n_calm", 1000) +
        results.get(list(results.keys())[0], {}).get("n_crisis", 100)
    )
    axes[1].axhline(baseline_precision, color="k", linestyle="--", alpha=0.3, linewidth=0.8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(fontsize=6, loc="upper right", ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / "roc_pr_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "roc_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: FPR comparison bar chart ---
    methods = []
    fprs = []
    roc_aucs = []
    pr_aucs = []
    best_f1s = []

    for method_name, method_data in results.items():
        if "error" in method_data:
            continue
        methods.append(method_name)
        fprs.append(method_data["fpr_at_p95"])
        roc_aucs.append(method_data["roc_auc"])
        pr_aucs.append(method_data["pr_auc"])
        best_f1s.append(method_data["best_f1"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(methods))
    width = 0.6

    axes[0].barh(x, fprs, height=width, color="coral", alpha=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(methods, fontsize=7)
    axes[0].set_xlabel("False Positive Rate (at 95th pctl)")
    axes[0].set_title("FPR During Calm Periods")
    axes[0].axvline(0.05, color="red", linestyle="--", alpha=0.5, label="5% target")
    axes[0].legend(fontsize=8)

    axes[1].barh(x, roc_aucs, height=width, color="steelblue", alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(methods, fontsize=7)
    axes[1].set_xlabel("ROC AUC")
    axes[1].set_title("ROC AUC")
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[1].legend(fontsize=8)

    axes[2].barh(x, best_f1s, height=width, color="seagreen", alpha=0.8)
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(methods, fontsize=7)
    axes[2].set_xlabel("Best F1 Score")
    axes[2].set_title("Optimal F1")

    plt.tight_layout()
    fig.savefig(output_dir / "fpr_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fpr_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Figures saved to {output_dir}")


def run_false_positive_analysis(seed: int = 42) -> Dict:
    """Run the full false positive rate analysis."""
    seed_everything(seed)
    config = get_default_validation_config()

    output_dir = Path("experiments/outputs/regime_detection/false_positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect scores during calm periods
    print("=" * 60)
    print("FALSE POSITIVE RATE ANALYSIS")
    print("=" * 60)

    calm_data = {}
    for period in CALM_PERIODS:
        print(f"\nFetching calm period: {period['name']}...")
        result = fetch_calm_period_data(period)
        if result is not None:
            calm_data[period["name"]] = {
                "X": result[0],
                "X_enriched": result[1],
                "times": result[2],
            }
            print(f"  Got {len(result[0])} observations")
        else:
            print(f"  SKIPPED (no data)")

    if not calm_data:
        raise ValueError("No calm period data available")

    # 2. Collect scores during crisis periods (reuse from regime comparison)
    print("\nFetching crisis period data...")
    from experiments.rigorous_crisis_validation import fetch_real_crisis_data
    crisis_data = {}
    crises = DATA_AVAILABLE_CRISES
    for crisis in crises:
        try:
            X, X_enriched, times, crisis_idx = prepare_data(crisis, config)
            if X is not None:
                window = config.analysis_window_days
                start = max(0, crisis_idx - window)
                end = min(len(X), crisis_idx + window)
                crisis_data[crisis.name] = {
                    "X": X,
                    "X_enriched": X_enriched,
                    "times": times,
                    "crisis_idx": crisis_idx,
                    "crisis_start": start,
                    "crisis_end": end,
                }
                print(f"  {crisis.name}: {len(X)} obs, crisis window [{start}:{end}]")
        except Exception as e:
            logger.warning(f"Skipping {crisis.name}: {e}")

    # 3. Train RF on all crises for supervised comparison
    print("\nTraining Random Forest...")
    rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=seed, lookback=20)
    try:
        X_train_all = []
        y_train_all = []
        for crisis in crises:
            X, X_enriched, times, crisis_idx = prepare_data(crisis, config)
            if X is None:
                continue
            y = np.zeros(len(X))
            window = config.analysis_window_days
            y[max(0, crisis_idx - window):min(len(X), crisis_idx + window)] = 1
            X_train_all.append(X)
            y_train_all.append(y)

        if X_train_all:
            min_cols = min(x.shape[1] for x in X_train_all)
            X_train_all = [x[:, :min_cols] for x in X_train_all]
            rf.fit_with_labels(np.vstack(X_train_all), np.concatenate(y_train_all))
            rf_n_features = min_cols
            print(f"  RF trained on {len(X_train_all)} crises, {rf_n_features} features")
    except Exception as e:
        logger.error(f"RF training failed: {e}")
        rf = None
        rf_n_features = None

    # 4. For each detector, collect calm and crisis scores
    print("\nComputing scores for all detectors...")
    detectors = build_all_detectors(config, seed)

    # Add RF if available
    if rf is not None:
        detectors.append(rf)

    all_results = {}

    for det in detectors:
        method_name = det.name
        print(f"\n  {method_name}...")

        calm_scores_all = []
        crisis_scores_all = []
        enriched = is_enriched_detector(det)

        # Compute calm scores
        for period_name, data in calm_data.items():
            try:
                X_use = data["X_enriched"] if enriched else data["X"]

                if method_name == "Random Forest" and rf_n_features is not None:
                    X_use = data["X"][:, :rf_n_features] if data["X"].shape[1] > rf_n_features else data["X"]

                # Fit on the data (needed for stateful detectors)
                det_fresh = _clone_detector(det, config, seed)

                if method_name == "Random Forest":
                    # RF is already trained
                    det_fresh = rf
                else:
                    det_fresh.fit(X_use)

                scores = det_fresh.compute_regime_scores(X_use)
                valid = scores[~np.isnan(scores)]
                calm_scores_all.extend(valid.tolist())
            except Exception as e:
                logger.warning(f"    {method_name} failed on {period_name}: {e}")

        # Compute crisis scores
        for crisis_name, data in crisis_data.items():
            try:
                X_use = data["X_enriched"] if enriched else data["X"]
                trim = 19 if enriched else 0

                if method_name == "Random Forest" and rf_n_features is not None:
                    X_use = data["X"][:, :rf_n_features] if data["X"].shape[1] > rf_n_features else data["X"]
                    trim = 0

                det_fresh = _clone_detector(det, config, seed)
                if method_name == "Random Forest":
                    det_fresh = rf
                else:
                    det_fresh.fit(X_use)

                scores = det_fresh.compute_regime_scores(X_use)
                ci = data["crisis_idx"] - trim if enriched else data["crisis_idx"]
                ci = max(0, ci)
                window = config.analysis_window_days
                start = max(0, ci - window)
                end = min(len(scores), ci + window)
                crisis_window_scores = scores[start:end]
                valid = crisis_window_scores[~np.isnan(crisis_window_scores)]
                crisis_scores_all.extend(valid.tolist())
            except Exception as e:
                logger.warning(f"    {method_name} failed on {crisis_name}: {e}")

        # Compute FPR metrics
        if calm_scores_all and crisis_scores_all:
            calm_arr = np.array(calm_scores_all)
            crisis_arr = np.array(crisis_scores_all)
            metrics = compute_fpr_at_thresholds(calm_arr, crisis_arr)
            all_results[method_name] = metrics
            print(
                f"    ROC AUC={metrics['roc_auc']:.3f}, "
                f"PR AUC={metrics['pr_auc']:.3f}, "
                f"Best F1={metrics['best_f1']:.3f}, "
                f"FPR@p95={metrics['fpr_at_p95']:.3f}"
            )
        else:
            all_results[method_name] = {"error": "no data"}
            print(f"    FAILED: no data")

    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Strip large arrays for JSON (keep ROC/PR curve data)
    json_results = {}
    for method, data in all_results.items():
        if "error" in data:
            json_results[method] = data
        else:
            json_results[method] = {
                k: v for k, v in data.items()
                if k not in ("f1_curve",)  # Keep ROC/PR for figure regeneration
            }

    with open(output_dir / f"false_positive_{timestamp}.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    # 6. Generate figures
    generate_figures(all_results, output_dir)

    # 7. Print summary table
    print("\n" + "=" * 80)
    print("FALSE POSITIVE RATE SUMMARY")
    print("=" * 80)
    header = f"  {'Method':<25s} {'ROC AUC':>8s} {'PR AUC':>8s} {'Best F1':>8s} {'FPR@p95':>8s} {'Rec@p95':>8s}"
    print(header)
    print("  " + "-" * 75)
    for method, data in all_results.items():
        if "error" in data:
            print(f"  {method:<25s} {'N/A':>8s}")
        else:
            print(
                f"  {method:<25s} {data['roc_auc']:>8.3f} {data['pr_auc']:>8.3f} "
                f"{data['best_f1']:>8.3f} {data['fpr_at_p95']:>8.3f} {data['recall_at_p95']:>8.3f}"
            )
    print("=" * 80)

    return all_results


def _clone_detector(det: BaseRegimeDetector, config: ValidationConfig, seed: int):
    """Create a fresh copy of a detector with the same parameters."""
    name = det.name
    if name == "QCML Chern":
        return QCMLChernDetector(
            hilbert_dim=config.hilbert_dim, window_size=config.window_size,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
    elif name == "Rolling Vol Z":
        return RollingVolatilityDetector(vol_window=20, min_expanding=60)
    elif name == "CUSUM":
        return CUSUMDetector(burn_in=60)
    elif name == "HMM 2-state":
        return HMMRegimeDetector(n_iter=100, seed=seed)
    elif name == "Multi-Scale Chern":
        return MultiScaleChernDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
    elif name == "Quantum Ensemble":
        return QuantumEnsembleDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            window_size=config.window_size, operator_method=config.operator_method, seed=seed,
        )
    elif name == "QFI Susceptibility":
        return QFISusceptibilityDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, min_expanding=60, seed=seed,
        )
    elif name == "Scalar Curvature":
        return ScalarCurvatureDetector(
            hilbert_dim=config.hilbert_dim, n_curvature_components=8,
            operator_method=config.operator_method, min_expanding=60, seed=seed,
        )
    elif name == "Geometric Consensus":
        return GeometricConsensusDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=8,
            n_curvature_components=8, operator_method=config.operator_method,
            min_persistence=3, min_agreement=2, seed=seed,
        )
    elif name == "QFI Determinant":
        return QFIDeterminantDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
    elif name == "Berry Phase Rate":
        return BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
    elif name == "Multi-Lag Fidelity":
        return MultiLagFidelityDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
    elif name == "Metric Condition Number":
        return MetricConditionNumberDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
    else:
        return det


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_false_positive_analysis()
