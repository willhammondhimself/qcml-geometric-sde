#!/usr/bin/env python3
"""
Geometric SDE Empirical Validation

Validates the geometric SDE (Section 4 of paper) empirically:
1. Fit QCMLGeometry to SPY feature data.
2. Simulate 1000 paths from geometric SDE (metric-induced diffusion).
3. Simulate 1000 paths from standard SDE (isotropic diffusion, same drift).
4. Compare to actual returns: QQ plots, K-S test, Anderson-Darling test.
5. Test whether geometric SDE produces more realistic regime transitions.

Usage:
    python experiments/validate_geometric_sde.py

Author: QCML Research
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.data import PolygonDataSource, MinimalFeatureEngine
from qcml.qcml_geometry import QCMLGeometry
from qcml.geometric_sde import GeometricSDE
from experiments.regime_comparison import seed_everything

logger = logging.getLogger(__name__)


def fetch_spy_data(
    start_date: str = "2006-01-01",
    end_date: str = "2023-12-31",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fetch SPY data and compute features + returns.

    Returns:
        prices: Raw price DataFrame.
        X: Feature matrix (T, d) — PCA-reduced, normalized.
        returns: Feature-space first differences (T-1, d).
    """
    ds = PolygonDataSource()
    prices = ds.fetch_equities(["SPY"], start_date=start_date, end_date=end_date)

    if prices is None or prices.empty:
        raise ValueError("No price data returned from Polygon API")

    # Build features
    engine = MinimalFeatureEngine()
    close = prices["close"].unstack("symbol")
    feature_matrix = engine.create_feature_matrix(close)

    X_raw = feature_matrix.values
    times = feature_matrix.index

    # Standard pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    n_components = min(5, X_raw.shape[1])  # Use 5 PCA components for tractability
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Feature-space returns (first differences)
    returns = np.diff(X, axis=0)

    return prices, X, returns


def simulate_sde_paths(
    geometry: QCMLGeometry,
    X: np.ndarray,
    returns: np.ndarray,
    n_paths: int = 1000,
    n_steps: int = 252,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate paths from geometric and standard SDEs.

    Args:
        geometry: Fitted QCMLGeometry.
        X: Feature matrix for fitting drift.
        returns: Feature-space returns.
        n_paths: Number of paths per SDE type.
        n_steps: Steps per path (trading days).
        seed: Random seed.

    Returns:
        geo_paths: (n_paths, n_steps+1, d) from geometric SDE.
        std_paths: (n_paths, n_steps+1, d) from standard SDE.
    """
    d = X.shape[1]

    # Geometric SDE
    geo_sde = GeometricSDE(geometry)
    geo_sde.fit_to_data(returns, dt=1.0)

    # Standard SDE (same drift, isotropic diffusion)
    empirical_std = np.std(returns, axis=0)
    std_sde = GeometricSDE(
        geometry,
        drift_fn=lambda x, t: geo_sde._empirical_drift,
        diffusion_fn=lambda x, t: np.diag(empirical_std),
    )

    # Start from a representative point (median of data)
    x0 = np.median(X, axis=0)

    # Simulate
    print(f"  Simulating {n_paths} geometric SDE paths ({n_steps} steps)...")
    geo_paths, _ = geo_sde.simulate_euler_maruyama(
        x0, T=n_steps, dt=1.0, n_paths=n_paths, seed=seed,
        use_metric_diffusion=True, diffusion_scale=0.1,
    )

    print(f"  Simulating {n_paths} standard SDE paths ({n_steps} steps)...")
    std_paths, _ = std_sde.simulate_euler_maruyama(
        x0, T=n_steps, dt=1.0, n_paths=n_paths, seed=seed + 1,
        use_metric_diffusion=False, diffusion_scale=1.0,
    )

    return geo_paths, std_paths


def compute_distributional_tests(
    actual_returns: np.ndarray,
    geo_paths: np.ndarray,
    std_paths: np.ndarray,
) -> Dict[str, Any]:
    """Compare simulated path returns to actual returns.

    Tests:
      - K-S test (Kolmogorov-Smirnov) for each feature dimension.
      - Anderson-Darling test for normality of residuals.
      - Higher-moment comparison (skewness, kurtosis).

    Returns dict with test results.
    """
    d = actual_returns.shape[1]

    # Compute returns from simulated paths
    geo_returns = np.diff(geo_paths, axis=1)  # (n_paths, n_steps, d)
    std_returns = np.diff(std_paths, axis=1)

    # Pool across paths for distribution comparison
    geo_pooled = geo_returns.reshape(-1, d)
    std_pooled = std_returns.reshape(-1, d)

    results = {
        "ks_geo": [],
        "ks_std": [],
        "ad_actual": [],
        "moments": {},
    }

    for dim in range(d):
        actual_dim = actual_returns[:, dim]
        geo_dim = geo_pooled[:, dim]
        std_dim = std_pooled[:, dim]

        # Subsample for K-S test efficiency
        n_sample = min(5000, len(geo_dim))
        rng = np.random.RandomState(42)
        geo_sample = rng.choice(geo_dim, n_sample, replace=False)
        std_sample = rng.choice(std_dim, n_sample, replace=False)

        # K-S test: how well does simulated distribution match actual?
        ks_geo_stat, ks_geo_p = stats.ks_2samp(actual_dim, geo_sample)
        ks_std_stat, ks_std_p = stats.ks_2samp(actual_dim, std_sample)

        results["ks_geo"].append({"stat": ks_geo_stat, "p": ks_geo_p})
        results["ks_std"].append({"stat": ks_std_stat, "p": ks_std_p})

        # Anderson-Darling normality test on actual
        ad_result = stats.anderson(actual_dim, dist="norm")
        results["ad_actual"].append({
            "statistic": ad_result.statistic,
            "critical_values": ad_result.critical_values.tolist(),
        })

    # Higher moments comparison
    actual_skew = stats.skew(actual_returns, axis=0)
    actual_kurt = stats.kurtosis(actual_returns, axis=0)
    geo_skew = stats.skew(geo_pooled, axis=0)
    geo_kurt = stats.kurtosis(geo_pooled, axis=0)
    std_skew = stats.skew(std_pooled, axis=0)
    std_kurt = stats.kurtosis(std_pooled, axis=0)

    results["moments"] = {
        "actual": {"skew": actual_skew.tolist(), "kurtosis": actual_kurt.tolist()},
        "geometric": {"skew": geo_skew.tolist(), "kurtosis": geo_kurt.tolist()},
        "standard": {"skew": std_skew.tolist(), "kurtosis": std_kurt.tolist()},
    }

    # Summary: average K-S statistic across dimensions
    results["summary"] = {
        "geo_mean_ks": np.mean([r["stat"] for r in results["ks_geo"]]),
        "std_mean_ks": np.mean([r["stat"] for r in results["ks_std"]]),
        "geo_better": sum(1 for g, s in zip(results["ks_geo"], results["ks_std"])
                          if g["stat"] < s["stat"]),
        "n_dims": d,
    }

    return results


def detect_regime_transitions_in_paths(
    geometry: QCMLGeometry,
    paths: np.ndarray,
    n_sample_paths: int = 50,
) -> np.ndarray:
    """Count metric eigenvalue spikes in simulated paths.

    For each path, compute the quantum metric at each step and flag
    steps where the largest eigenvalue exceeds 2x its running median.

    Returns:
        transition_counts: (n_sample_paths,) — number of transitions per path.
    """
    n_paths = min(n_sample_paths, paths.shape[0])
    counts = np.zeros(n_paths)

    for p in range(n_paths):
        path = paths[p]
        n_steps = path.shape[0]
        max_eigs = np.empty(n_steps)

        for t in range(n_steps):
            try:
                g = geometry.quantum_metric(path[t])
                max_eigs[t] = np.max(np.linalg.eigvalsh(g))
            except Exception:
                max_eigs[t] = np.nan

        # Flag spikes: max eigenvalue > 2x running median (window=20)
        valid = ~np.isnan(max_eigs)
        if valid.sum() < 30:
            continue

        window = 20
        for t in range(window, len(max_eigs)):
            if valid[t] and valid[t - window:t].sum() >= 10:
                median_val = np.nanmedian(max_eigs[t - window:t])
                if median_val > 0 and max_eigs[t] > 2 * median_val:
                    counts[p] += 1

    return counts


def generate_figures(
    actual_returns: np.ndarray,
    geo_paths: np.ndarray,
    std_paths: np.ndarray,
    test_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate SDE validation figures.

    Creates:
      (a) QQ plots for each dimension (actual vs geometric, actual vs standard).
      (b) K-S statistic comparison bar chart.
      (c) Sample simulated paths (geometric vs standard).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    d = actual_returns.shape[1]
    geo_returns = np.diff(geo_paths, axis=1).reshape(-1, d)
    std_returns = np.diff(std_paths, axis=1).reshape(-1, d)

    # (a) QQ plots — first 3 dimensions
    n_qq = min(3, d)
    fig, axes = plt.subplots(n_qq, 2, figsize=(10, 3.5 * n_qq))
    if n_qq == 1:
        axes = axes.reshape(1, -1)

    for dim in range(n_qq):
        actual_sorted = np.sort(actual_returns[:, dim])
        n_actual = len(actual_sorted)

        # Geometric SDE
        rng = np.random.RandomState(42)
        geo_sample = np.sort(rng.choice(geo_returns[:, dim], n_actual, replace=True))
        axes[dim, 0].scatter(actual_sorted, geo_sample, s=2, alpha=0.3, color="steelblue")
        lim = max(abs(actual_sorted).max(), abs(geo_sample).max()) * 1.1
        axes[dim, 0].plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
        axes[dim, 0].set_xlabel(f"Actual (dim {dim+1})")
        axes[dim, 0].set_ylabel("Geometric SDE")
        axes[dim, 0].set_title(f"QQ: Geometric SDE vs Actual (dim {dim+1})")

        # Standard SDE
        std_sample = np.sort(rng.choice(std_returns[:, dim], n_actual, replace=True))
        axes[dim, 1].scatter(actual_sorted, std_sample, s=2, alpha=0.3, color="darkorange")
        axes[dim, 1].plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
        axes[dim, 1].set_xlabel(f"Actual (dim {dim+1})")
        axes[dim, 1].set_ylabel("Standard SDE")
        axes[dim, 1].set_title(f"QQ: Standard SDE vs Actual (dim {dim+1})")

    plt.tight_layout()
    fig.savefig(output_dir / "sde_qq_plots.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "sde_qq_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) K-S statistic comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    dims = np.arange(d)
    geo_ks = [r["stat"] for r in test_results["ks_geo"]]
    std_ks = [r["stat"] for r in test_results["ks_std"]]
    width = 0.35
    ax.bar(dims - width/2, geo_ks, width, label="Geometric SDE", color="steelblue")
    ax.bar(dims + width/2, std_ks, width, label="Standard SDE", color="darkorange")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("K-S Statistic (lower = better)")
    ax.set_title("Kolmogorov-Smirnov: Simulated vs Actual Returns")
    ax.set_xticks(dims)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "sde_ks_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "sde_ks_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (c) Sample paths (first 5 paths, dim 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for p in range(min(5, geo_paths.shape[0])):
        axes[0].plot(geo_paths[p, :, 0], alpha=0.5, linewidth=0.8)
        axes[1].plot(std_paths[p, :, 0], alpha=0.5, linewidth=0.8)

    axes[0].set_title("Geometric SDE Paths (dim 1)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Feature Value")
    axes[1].set_title("Standard SDE Paths (dim 1)")
    axes[1].set_xlabel("Time Step")

    plt.tight_layout()
    fig.savefig(output_dir / "sde_sample_paths.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "sde_sample_paths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {output_dir}")


def run_sde_validation(seed: int = 42) -> Dict[str, Any]:
    """Run the full geometric SDE validation experiment."""
    seed_everything(seed)

    output_dir = Path("experiments/outputs/regime_detection/sde_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEOMETRIC SDE EMPIRICAL VALIDATION")
    print("=" * 60)

    # 1. Fetch data
    print("\nFetching SPY data...")
    prices, X, returns = fetch_spy_data()
    print(f"  X shape: {X.shape}, returns shape: {returns.shape}")

    # 2. Fit geometry
    print("\nFitting QCMLGeometry...")
    d = X.shape[1]
    geometry = QCMLGeometry(n_features=d, hilbert_dim=8)
    geometry.fit_operators(X, method="pca_inspired")
    print(f"  Geometry fitted (hilbert_dim=8, d={d})")

    # 3. Simulate paths
    print("\nSimulating SDE paths...")
    n_paths = 500  # Reduced for speed
    n_steps = min(252, returns.shape[0])
    geo_paths, std_paths = simulate_sde_paths(
        geometry, X, returns, n_paths=n_paths, n_steps=n_steps, seed=seed
    )
    print(f"  Geometric paths: {geo_paths.shape}")
    print(f"  Standard paths: {std_paths.shape}")

    # 4. Distributional tests
    print("\nRunning distributional tests...")
    test_results = compute_distributional_tests(returns, geo_paths, std_paths)

    print(f"\n  K-S Results (lower = better match to actual):")
    print(f"    Geometric SDE mean K-S: {test_results['summary']['geo_mean_ks']:.4f}")
    print(f"    Standard SDE mean K-S:  {test_results['summary']['std_mean_ks']:.4f}")
    print(f"    Geometric better in {test_results['summary']['geo_better']}/{test_results['summary']['n_dims']} dims")

    # Moments comparison
    moments = test_results["moments"]
    print(f"\n  Kurtosis (actual vs simulated):")
    print(f"    Actual:    {np.mean(moments['actual']['kurtosis']):.3f}")
    print(f"    Geometric: {np.mean(moments['geometric']['kurtosis']):.3f}")
    print(f"    Standard:  {np.mean(moments['standard']['kurtosis']):.3f}")

    # 5. Regime transition detection in simulated paths
    print("\nDetecting regime transitions in simulated paths...")
    geo_transitions = detect_regime_transitions_in_paths(geometry, geo_paths, n_sample_paths=50)
    std_transitions = detect_regime_transitions_in_paths(geometry, std_paths, n_sample_paths=50)

    print(f"  Geometric SDE: {np.mean(geo_transitions):.1f} transitions/path (std={np.std(geo_transitions):.1f})")
    print(f"  Standard SDE:  {np.mean(std_transitions):.1f} transitions/path (std={np.std(std_transitions):.1f})")

    # Mann-Whitney test for transition count difference
    if len(geo_transitions) > 5 and len(std_transitions) > 5:
        mw_stat, mw_p = stats.mannwhitneyu(geo_transitions, std_transitions, alternative="two-sided")
        print(f"  Mann-Whitney p={mw_p:.4f}")
        test_results["transition_test"] = {
            "geo_mean": float(np.mean(geo_transitions)),
            "std_mean": float(np.mean(std_transitions)),
            "mw_statistic": float(mw_stat),
            "mw_p_value": float(mw_p),
        }

    # 6. Generate figures
    print("\nGenerating figures...")
    generate_figures(returns, geo_paths, std_paths, test_results, output_dir)

    # 7. Save results
    serializable = {
        "summary": {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in test_results["summary"].items()},
        "moments": test_results["moments"],
        "ks_geo": [{"stat": float(r["stat"]), "p": float(r["p"])} for r in test_results["ks_geo"]],
        "ks_std": [{"stat": float(r["stat"]), "p": float(r["p"])} for r in test_results["ks_std"]],
    }
    if "transition_test" in test_results:
        serializable["transition_test"] = test_results["transition_test"]

    with open(output_dir / "sde_validation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    return test_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_sde_validation()
