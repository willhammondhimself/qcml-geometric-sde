#!/usr/bin/env python3
"""
QCML Method Correlation Analysis

Analyzes redundancy and independence among the 11 QCML-based regime detection
methods. Produces:
  1. Pairwise Spearman correlation matrix (11x11)
  2. Hierarchical clustering dendrogram (Ward linkage)
  3. PCA of score series — how many independent signals?
  4. Mutual information between each QCML method and RF

Usage:
    python experiments/method_correlation.py

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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.regime.classical_baselines import (
    BaseRegimeDetector,
    QCMLChernDetector,
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
    MultiScaleChernDetector,
    QuantumEnsembleDetector,
    QFISusceptibilityDetector,
    ScalarCurvatureDetector,
    GeometricConsensusDetector,
    QFIDeterminantDetector,
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    MetricConditionNumberDetector,
)
from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    prepare_data,
    prepare_rf_training_data,
    seed_everything,
)

logger = logging.getLogger(__name__)

# The 11 QCML-based methods
QCML_METHODS = [
    "QCML Chern",
    "Multi-Scale Chern",
    "Quantum Ensemble",
    "QFI Susceptibility",
    "Scalar Curvature",
    "Geometric Consensus",
    "QFI Determinant",
    "Berry Phase Rate",
    "Multi-Lag Fidelity",
    "Metric Condition Number",
    "Adaptive Ensemble",
]


def collect_all_scores(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all detectors on all crises, collect score time series.

    Returns:
        qcml_scores: DataFrame (T_total, 11) of QCML method scores
        all_scores: DataFrame (T_total, 16) of all method scores including baselines
    """
    config = get_default_validation_config()
    crises = DATA_AVAILABLE_CRISES

    # Collect scores per method per crisis
    method_scores = {}
    enriched_lookback = 20

    for crisis in crises:
        print(f"  Processing {crisis.name}...")
        X, X_enriched, times, crisis_idx = prepare_data(crisis, config, enriched_lookback=enriched_lookback)
        if X is None:
            continue

        trim = enriched_lookback - 1
        times_enriched = times[trim:]

        # Build detectors
        detectors = [
            ("QCML Chern", QCMLChernDetector(hilbert_dim=config.hilbert_dim,
                window_size=config.window_size, n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed), True),
            ("Rolling Vol Z", RollingVolatilityDetector(vol_window=20, min_expanding=60), False),
            ("CUSUM", CUSUMDetector(burn_in=60), False),
            ("HMM 2-state", HMMRegimeDetector(n_iter=100, seed=seed), False),
            ("Multi-Scale Chern", MultiScaleChernDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, operator_method=config.operator_method,
                seed=seed), True),
            ("Quantum Ensemble", QuantumEnsembleDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, window_size=config.window_size,
                operator_method=config.operator_method, seed=seed), True),
            ("QFI Susceptibility", QFISusceptibilityDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, operator_method=config.operator_method,
                min_expanding=60, seed=seed), True),
            ("Scalar Curvature", ScalarCurvatureDetector(hilbert_dim=config.hilbert_dim,
                n_curvature_components=8, operator_method=config.operator_method,
                min_expanding=60, seed=seed), True),
            ("Geometric Consensus", GeometricConsensusDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=8, n_curvature_components=8,
                operator_method=config.operator_method, min_persistence=3,
                min_agreement=2, seed=seed), True),
            ("QFI Determinant", QFIDeterminantDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, operator_method=config.operator_method,
                seed=seed), True),
            ("Berry Phase Rate", BerryPhaseRateDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, operator_method=config.operator_method,
                seed=seed), True),
            ("Multi-Lag Fidelity", MultiLagFidelityDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, operator_method=config.operator_method,
                seed=seed), True),
            ("Metric Condition Number", MetricConditionNumberDetector(hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components, operator_method=config.operator_method,
                seed=seed), True),
        ]

        for name, det, use_enriched in detectors:
            try:
                X_use = X_enriched if use_enriched else X
                det.fit(X_use)
                scores = det.compute_regime_scores(X_use)

                if name not in method_scores:
                    method_scores[name] = []
                method_scores[name].extend(scores.tolist())
            except Exception as e:
                logger.warning(f"    {name} failed on {crisis.name}: {e}")

    # Align all methods to same length (use shortest)
    min_len = min(len(v) for v in method_scores.values())
    for k in method_scores:
        method_scores[k] = method_scores[k][:min_len]

    all_scores_df = pd.DataFrame(method_scores)

    # Extract QCML-only columns
    qcml_cols = [c for c in all_scores_df.columns if c in QCML_METHODS or c in [
        "QCML Chern", "Multi-Scale Chern", "Quantum Ensemble",
        "QFI Susceptibility", "Scalar Curvature", "Geometric Consensus",
        "QFI Determinant", "Berry Phase Rate", "Multi-Lag Fidelity",
        "Metric Condition Number",
    ]]
    qcml_scores_df = all_scores_df[qcml_cols]

    return qcml_scores_df, all_scores_df


def compute_spearman_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Spearman correlation and p-value matrices."""
    cols = df.columns
    n = len(cols)
    corr = np.zeros((n, n))
    pval = np.zeros((n, n))

    # Drop NaN rows for pairwise computation
    for i in range(n):
        for j in range(i, n):
            mask = ~(np.isnan(df.iloc[:, i]) | np.isnan(df.iloc[:, j]))
            if mask.sum() < 10:
                corr[i, j] = corr[j, i] = 0.0
                pval[i, j] = pval[j, i] = 1.0
            else:
                r, p = stats.spearmanr(df.iloc[:, i][mask], df.iloc[:, j][mask])
                corr[i, j] = corr[j, i] = r
                pval[i, j] = pval[j, i] = p

    return (
        pd.DataFrame(corr, index=cols, columns=cols),
        pd.DataFrame(pval, index=cols, columns=cols),
    )


def compute_pca_analysis(df: pd.DataFrame) -> Dict:
    """PCA of QCML score series to find independent signals."""
    # Drop NaN rows
    clean = df.dropna()
    if len(clean) < 20:
        return {"error": "insufficient clean data"}

    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)

    pca = PCA()
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "n_components_95pct": n_95,
        "n_components_90pct": n_90,
        "n_total": len(pca.explained_variance_ratio_),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=df.columns,
            columns=[f"PC{i+1}" for i in range(len(pca.components_))],
        ),
    }


def compute_mutual_information(df: pd.DataFrame, target_col: str = "Random Forest",
                               n_bins: int = 20) -> Dict[str, float]:
    """Compute mutual information between each QCML method and RF."""
    if target_col not in df.columns:
        return {}

    target = df[target_col].dropna()
    mi_results = {}

    for col in df.columns:
        if col == target_col:
            continue

        # Align indices
        both = pd.concat([df[col], target], axis=1).dropna()
        if len(both) < 20:
            mi_results[col] = 0.0
            continue

        # Discretize for MI computation
        x_binned = pd.cut(both.iloc[:, 0], bins=n_bins, labels=False).values
        y_binned = pd.cut(both.iloc[:, 1], bins=n_bins, labels=False).values

        # Remove any NaN from binning edge cases
        mask = ~(np.isnan(x_binned) | np.isnan(y_binned))
        if mask.sum() < 10:
            mi_results[col] = 0.0
            continue

        mi = mutual_info_score(x_binned[mask].astype(int), y_binned[mask].astype(int))
        mi_results[col] = float(mi)

    return mi_results


def generate_figures(
    corr_matrix: pd.DataFrame,
    pca_result: Dict,
    mi_results: Dict,
    output_dir: Path,
) -> None:
    """Generate correlation heatmap, dendrogram, and PCA scree plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0, 0],
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    axes[0, 0].set_title("Spearman Correlation (QCML Methods)", fontsize=11)
    axes[0, 0].tick_params(labelsize=7)

    # 2. Dendrogram
    # Convert correlation to distance: d = 1 - |rho|
    dist = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dist, 0)
    # Ensure symmetry
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    dendrogram(
        Z,
        labels=corr_matrix.columns.tolist(),
        ax=axes[0, 1],
        leaf_rotation=45,
        leaf_font_size=7,
    )
    axes[0, 1].set_title("Hierarchical Clustering (Ward Linkage)", fontsize=11)
    axes[0, 1].set_ylabel("Distance (1 - |rho|)")

    # 3. PCA scree plot
    if "error" not in pca_result:
        evr = pca_result["explained_variance_ratio"]
        cumvar = pca_result["cumulative_variance"]
        n_comp = len(evr)

        ax3 = axes[1, 0]
        ax3.bar(range(1, n_comp + 1), evr, color="steelblue", alpha=0.7, label="Individual")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(range(1, n_comp + 1), cumvar, "ro-", markersize=4, label="Cumulative")
        ax3_twin.axhline(0.95, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax3_twin.axhline(0.90, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
        ax3.set_xlabel("Principal Component")
        ax3.set_ylabel("Explained Variance Ratio")
        ax3_twin.set_ylabel("Cumulative Variance")
        ax3.set_title(
            f"PCA of QCML Scores ({pca_result['n_components_90pct']} PCs for 90%, "
            f"{pca_result['n_components_95pct']} for 95%)",
            fontsize=11,
        )
        ax3.legend(loc="upper left", fontsize=8)
        ax3_twin.legend(loc="center right", fontsize=8)

    # 4. Mutual information bar chart
    if mi_results:
        methods = list(mi_results.keys())
        mi_vals = [mi_results[m] for m in methods]
        sorted_idx = np.argsort(mi_vals)[::-1]
        methods_sorted = [methods[i] for i in sorted_idx]
        mi_sorted = [mi_vals[i] for i in sorted_idx]

        axes[1, 1].barh(range(len(methods_sorted)), mi_sorted, color="teal", alpha=0.7)
        axes[1, 1].set_yticks(range(len(methods_sorted)))
        axes[1, 1].set_yticklabels(methods_sorted, fontsize=7)
        axes[1, 1].set_xlabel("Mutual Information (bits)")
        axes[1, 1].set_title("MI with Random Forest", fontsize=11)
        axes[1, 1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "method_correlation.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "method_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Figures saved to {output_dir}")


def run_method_correlation(seed: int = 42) -> Dict:
    """Run the full method correlation analysis."""
    seed_everything(seed)
    output_dir = Path("experiments/outputs/regime_detection/method_correlation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QCML METHOD CORRELATION ANALYSIS")
    print("=" * 60)

    # 1. Collect all scores
    print("\nCollecting scores from all detectors across all crises...")
    qcml_scores, all_scores = collect_all_scores(seed)
    print(f"  QCML scores shape: {qcml_scores.shape}")
    print(f"  All scores shape: {all_scores.shape}")

    # 2. Spearman correlation
    print("\nComputing Spearman correlations...")
    corr_matrix, pval_matrix = compute_spearman_matrix(qcml_scores)
    print(corr_matrix.to_string())

    # 3. PCA analysis
    print("\nRunning PCA on QCML score series...")
    pca_result = compute_pca_analysis(qcml_scores)
    if "error" not in pca_result:
        print(f"  Components for 90% variance: {pca_result['n_components_90pct']}")
        print(f"  Components for 95% variance: {pca_result['n_components_95pct']}")
        print(f"  Variance explained by PC1: {pca_result['explained_variance_ratio'][0]:.3f}")

    # 4. Mutual information with RF
    print("\nComputing mutual information with RF...")
    mi_results = compute_mutual_information(all_scores)
    for method, mi_val in sorted(mi_results.items(), key=lambda x: -x[1]):
        print(f"  {method:<25s}: MI = {mi_val:.4f}")

    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "spearman_correlation": corr_matrix.to_dict(),
        "spearman_pvalues": pval_matrix.to_dict(),
        "pca": {k: v for k, v in pca_result.items() if k != "loadings"} if "error" not in pca_result else pca_result,
        "mutual_information": mi_results,
    }

    with open(output_dir / f"correlation_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 6. Generate figures
    generate_figures(corr_matrix, pca_result, mi_results, output_dir)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_method_correlation()
