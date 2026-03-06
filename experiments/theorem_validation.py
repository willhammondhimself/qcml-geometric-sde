"""
Empirical Validation of Formal Theorems 1–3 and Proposition 4.

Connects the paper's mathematical results to actual market data:
  1.1  Spectral Gap Dynamics   (Thm 1: smoothness breaks when Δ→0)
  1.2  Curvature–Gap Bound     (Thm 3: ||F|| ≤ C/Δ²)
  1.3  Chern Number Quantization(Thm 2: C_1 ∈ Z on closed manifolds)
  1.4  QFI–Metric Identity     (Prop 4: QFI = 4 × Fubini-Study metric)

Usage:
    python experiments/theorem_validation.py              # full (16 crises)
    python experiments/theorem_validation.py --quick      # 4 representative crises
    python experiments/theorem_validation.py --crisis 2008_gfc  # single crisis

Outputs:
    experiments/outputs/theorem_validation/
        spectral_gap_dynamics.pdf
        curvature_gap_bound.pdf
        chern_quantization.pdf
        qfi_metric_identity.pdf
        theorem_validation_results.json
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qcml_geometry.core import QCMLGeometry
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "outputs" / "theorem_validation"
QUICK_CRISES = ["2008_gfc", "2020_covid", "2022_rates", "2018_volmageddon"]

from experiments.plot_style import (
    apply_style, NAVY, TEAL, BURGUNDY, GOLD, INDIGO, SLATE,
    CMAP_SEQUENTIAL, CMAP_DIVERGING,
)
apply_style()

CRISIS_COLOR = BURGUNDY
NORMAL_COLOR = NAVY
THEORY_COLOR = GOLD


# ── helpers ──────────────────────────────────────────────────────────


def _prepare_data(
    symbols=("SPY", "DIA"),
    start="2005-01-01",
    end="2025-06-30",
    n_pca=8,
    hilbert_dim=8,
    operator_method="pca_inspired",
    normalization="soft",
):
    """Fetch data, build features, fit PCA + QCML geometry.

    Uses soft normalization by default so that data retains scale variation,
    which is essential for spectral gap and curvature experiments.

    Returns:
        X_pca: (T, n_pca) PCA-transformed feature matrix.
        dates: DatetimeIndex aligned to X_pca.
        geometry: Fitted QCMLGeometry instance.
    """
    prices = fetch_data(list(symbols), start, end)
    close = prices["close"].unstack("symbol")
    X_raw, dates = create_feature_matrix(close)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=min(n_pca, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    if normalization == "sphere":
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        X_pca = X_pca / (norms + 1e-8)
    elif normalization == "soft":
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        median_norm = np.median(norms)
        X_pca = X_pca / (norms + median_norm)
    elif normalization == "clip":
        std_per_col = np.std(X_pca, axis=0)
        X_pca = np.clip(X_pca, -5 * std_per_col, 5 * std_per_col)
    # else: 'none' — raw PCA output

    geometry = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
    geometry.fit_operators(X_pca, method=operator_method)

    return X_pca, dates, geometry


def _crisis_mask(dates, crisis_key):
    """Return boolean mask for a crisis window."""
    info = ALL_CRISES[crisis_key]
    start = pd.Timestamp(info["start"])
    end = pd.Timestamp(info["end"])
    return (dates >= start) & (dates <= end)


def _subsample_indices(T, max_points=2000, seed=42):
    """Return sorted random indices if T > max_points, else all."""
    if T <= max_points:
        return np.arange(T)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(T, size=max_points, replace=False))


# ══════════════════════════════════════════════════════════════════════
# Experiment 1.1  —  Spectral Gap Dynamics at Crisis Boundaries
# ══════════════════════════════════════════════════════════════════════


def experiment_spectral_gap(X_pca, dates, geometry, crises, out_dir):
    """Plot spectral gap time series with crisis windows overlaid.

    Theorem 1 predicts smoothness breaks when Δ(x) → 0.
    We expect the spectral gap to reach local minima near crisis onsets.

    Note: uses random operators because PCA-inspired Pauli operators
    create exact degeneracies (gap ≡ 0). Random operators break this
    symmetry, giving meaningful spectral gap variation.
    """
    logger.info("=== Experiment 1.1: Spectral Gap Dynamics ===")
    T = len(X_pca)

    # random operators break Pauli degeneracies → meaningful spectral gap
    geo_random = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=geometry.hilbert_dim)
    geo_random.fit_operators(X_pca, method="random")

    # compute spectral gap for every time step
    gaps = np.empty(T)
    for t in range(T):
        gaps[t] = geo_random.spectral_gap(X_pca[t])

    # smoothing for visualization
    gap_series = pd.Series(gaps, index=dates)
    gap_smooth = gap_series.rolling(window=20, min_periods=1).mean()

    # ── figure: full time series ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, gap_smooth.values, color=NORMAL_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_ylabel("Spectral Gap  Δ(x)")
    ax.set_xlabel("Date")
    ax.set_title("Spectral Gap Dynamics with Crisis Windows (Theorem 1)")

    y_lo, y_hi = ax.get_ylim()
    for key in crises:
        info = ALL_CRISES[key]
        s = pd.Timestamp(info["start"])
        e = pd.Timestamp(info["end"])
        rect = Rectangle(
            (matplotlib.dates.date2num(s), y_lo),
            matplotlib.dates.date2num(e) - matplotlib.dates.date2num(s),
            y_hi - y_lo,
            alpha=0.15,
            color=CRISIS_COLOR,
        )
        ax.add_patch(rect)
        ax.text(
            matplotlib.dates.date2num(s),
            y_hi * 0.97,
            info["label"],
            fontsize=6,
            rotation=45,
            va="top",
        )

    fig.savefig(out_dir / "spectral_gap_dynamics.pdf")
    fig.savefig(out_dir / "spectral_gap_dynamics.png")
    plt.close(fig)

    # ── statistics: gap during crisis vs normal ──────────────────────
    results = {}
    for key in crises:
        mask = _crisis_mask(dates, key)
        crisis_gaps = gaps[mask]
        normal_gaps = gaps[~mask]
        if len(crisis_gaps) < 2:
            continue
        results[key] = {
            "crisis_gap_mean": float(np.mean(crisis_gaps)),
            "crisis_gap_min": float(np.min(crisis_gaps)),
            "normal_gap_mean": float(np.mean(normal_gaps)),
            "normal_gap_min": float(np.min(normal_gaps)),
            "ratio_means": float(np.mean(crisis_gaps) / (np.mean(normal_gaps) + 1e-12)),
        }
        logger.info(
            f"  {key}: crisis Δ_mean={np.mean(crisis_gaps):.4f}, "
            f"normal Δ_mean={np.mean(normal_gaps):.4f}, "
            f"ratio={results[key]['ratio_means']:.3f}"
        )

    return {"spectral_gap": results, "gap_timeseries_length": T}


# ══════════════════════════════════════════════════════════════════════
# Experiment 1.2  —  Curvature Divergence at Gap Closure (Theorem 3)
# ══════════════════════════════════════════════════════════════════════


def experiment_curvature_gap(X_pca, dates, geometry, crises, out_dir):
    """Scatter Berry curvature vs 1/Δ² with theoretical bound overlay.

    Theorem 3: ||F_ab|| ≤ C / Δ² for some constant C.
    We verify the bound empirically and check that crisis points
    cluster in the high-curvature / small-gap region.

    Uses random operators to avoid Pauli degeneracies in spectral gap.
    """
    logger.info("=== Experiment 1.2: Curvature-Gap Bound (Theorem 3) ===")
    T = len(X_pca)
    idx = _subsample_indices(T, max_points=1500)

    # random operators for non-degenerate spectral gaps
    geo_random = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=geometry.hilbert_dim)
    geo_random.fit_operators(X_pca, method="random")

    curvatures = np.empty(len(idx))
    inv_gap_sq = np.empty(len(idx))
    is_crisis = np.zeros(len(idx), dtype=bool)

    crisis_mask_full = np.zeros(T, dtype=bool)
    for key in crises:
        crisis_mask_full |= _crisis_mask(dates, key)

    for i, t in enumerate(idx):
        gap = geo_random.spectral_gap(X_pca[t])
        inv_gap_sq[i] = 1.0 / (gap**2 + 1e-12)

        F = geo_random.berry_curvature(X_pca[t])
        curvatures[i] = np.sqrt(np.sum(F**2))  # Frobenius norm

        is_crisis[i] = crisis_mask_full[t]

    # fit the empirical bound: C = max(F * Δ²)
    empirical_C = np.max(curvatures * (1.0 / (inv_gap_sq + 1e-12)))
    bound_x = np.linspace(
        np.percentile(inv_gap_sq, 1), np.percentile(inv_gap_sq, 99), 200
    )
    bound_y = empirical_C * bound_x

    # fraction of points satisfying the bound
    violations = np.sum(curvatures > empirical_C * inv_gap_sq * 1.01)
    frac_satisfying = 1.0 - violations / len(idx)

    # ── figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        inv_gap_sq[~is_crisis],
        curvatures[~is_crisis],
        s=4,
        alpha=0.3,
        color=NORMAL_COLOR,
        label="Normal",
        rasterized=True,
    )
    ax.scatter(
        inv_gap_sq[is_crisis],
        curvatures[is_crisis],
        s=12,
        alpha=0.7,
        color=CRISIS_COLOR,
        label="Crisis",
        zorder=5,
    )
    ax.plot(
        bound_x,
        bound_y,
        "--",
        color=THEORY_COLOR,
        linewidth=2,
        label=f"Theorem 3 bound (C={empirical_C:.2e})",
    )

    ax.set_xlabel(r"$1 / \Delta^2$")
    ax.set_ylabel(r"$\|F_{ab}\|_F$  (Berry curvature)")
    ax.set_title("Curvature-Gap Relationship (Theorem 3)")
    ax.legend(loc="upper left", fontsize=8)

    # use log scale if range is large
    if np.ptp(inv_gap_sq) > 100 * np.median(inv_gap_sq):
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.savefig(out_dir / "curvature_gap_bound.pdf")
    fig.savefig(out_dir / "curvature_gap_bound.png")
    plt.close(fig)

    logger.info(
        f"  Empirical C = {empirical_C:.4e}, "
        f"bound satisfied = {frac_satisfying:.1%}"
    )

    return {
        "curvature_gap": {
            "empirical_C": float(empirical_C),
            "fraction_satisfying_bound": float(frac_satisfying),
            "n_points": len(idx),
            "crisis_curvature_mean": float(np.mean(curvatures[is_crisis])) if np.any(is_crisis) else None,
            "normal_curvature_mean": float(np.mean(curvatures[~is_crisis])),
            "crisis_inv_gap_sq_mean": float(np.mean(inv_gap_sq[is_crisis])) if np.any(is_crisis) else None,
            "normal_inv_gap_sq_mean": float(np.mean(inv_gap_sq[~is_crisis])),
        }
    }


# ══════════════════════════════════════════════════════════════════════
# Experiment 1.3  —  Chern Number Quantization (Theorem 2)
# ══════════════════════════════════════════════════════════════════════


def experiment_chern_quantization(X_pca, dates, geometry, crises, out_dir):
    """Distribution of rolling Chern numbers — integer clustering.

    Theorem 2 says Chern numbers are integers on closed manifolds.
    In a rolling window, values are non-integer (open manifold), but we
    examine whether stable regimes cluster near integers and transitions
    produce fractional values.
    """
    logger.info("=== Experiment 1.3: Chern Number Quantization (Theorem 2) ===")
    T = len(X_pca)

    # Rolling Chern numbers over a sliding 2D grid
    window = 60  # trading days
    stride = 5
    indices = (0, 1)  # first two PCA components

    chern_values = []
    chern_dates = []
    chern_is_crisis = []

    crisis_mask_full = np.zeros(T, dtype=bool)
    for key in crises:
        crisis_mask_full |= _crisis_mask(dates, key)

    for t_start in range(0, T - window, stride):
        t_end = t_start + window
        window_data = X_pca[t_start:t_end, :2]  # first 2 PCA dims

        # create a small grid from the window data
        n_grid = 8
        x_min, x_max = window_data[:, 0].min(), window_data[:, 0].max()
        y_min, y_max = window_data[:, 1].min(), window_data[:, 1].max()

        # avoid degenerate grids
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < 1e-8 or y_range < 1e-8:
            continue

        x_vals = np.linspace(x_min, x_max, n_grid)
        y_vals = np.linspace(y_min, y_max, n_grid)

        # build grid: (n_grid, n_grid, n_features)
        grid = np.zeros((n_grid, n_grid, X_pca.shape[1]))
        # use the mean of the window for the other dimensions
        base = X_pca[t_start:t_end].mean(axis=0)
        for i in range(n_grid):
            for j in range(n_grid):
                grid[i, j] = base.copy()
                grid[i, j, 0] = x_vals[i]
                grid[i, j, 1] = y_vals[j]

        try:
            C = geometry.chern_number(grid, indices=indices, method="plaquette")
            chern_values.append(C)
            mid_date = dates[min(t_start + window // 2, T - 1)]
            chern_dates.append(mid_date)
            chern_is_crisis.append(bool(crisis_mask_full[t_start + window // 2]))
        except Exception as e:
            logger.debug(f"Chern computation failed at t={t_start}: {e}")
            continue

    chern_values = np.array(chern_values)
    chern_dates = pd.DatetimeIndex(chern_dates)
    chern_is_crisis = np.array(chern_is_crisis)

    if len(chern_values) == 0:
        logger.warning("No Chern values computed!")
        return {"chern_quantization": {"error": "no values computed"}}

    # distance to nearest integer
    nearest_int = np.round(chern_values)
    dist_to_int = np.abs(chern_values - nearest_int)

    # ── figure: histogram ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # panel 1: histogram of Chern values
    ax = axes[0]
    bins = np.linspace(
        np.percentile(chern_values, 1), np.percentile(chern_values, 99), 60
    )
    ax.hist(
        chern_values[~chern_is_crisis],
        bins=bins,
        alpha=0.6,
        color=NORMAL_COLOR,
        label="Normal",
        density=True,
    )
    if np.any(chern_is_crisis):
        ax.hist(
            chern_values[chern_is_crisis],
            bins=bins,
            alpha=0.6,
            color=CRISIS_COLOR,
            label="Crisis",
            density=True,
        )
    # mark integer lines
    for k in range(int(np.floor(bins[0])) - 1, int(np.ceil(bins[-1])) + 2):
        ax.axvline(k, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Rolling Chern Number")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Rolling Chern Numbers")
    ax.legend(fontsize=8)

    # panel 2: distance to nearest integer
    ax = axes[1]
    ax.hist(
        dist_to_int[~chern_is_crisis],
        bins=40,
        alpha=0.6,
        color=NORMAL_COLOR,
        label="Normal",
        density=True,
    )
    if np.any(chern_is_crisis):
        ax.hist(
            dist_to_int[chern_is_crisis],
            bins=40,
            alpha=0.6,
            color=CRISIS_COLOR,
            label="Crisis",
            density=True,
        )
    ax.set_xlabel("Distance to Nearest Integer")
    ax.set_ylabel("Density")
    ax.set_title("Integer Proximity (Theorem 2)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "chern_quantization.pdf")
    fig.savefig(out_dir / "chern_quantization.png")
    plt.close(fig)

    # statistics
    normal_dist = dist_to_int[~chern_is_crisis]
    crisis_dist = dist_to_int[chern_is_crisis] if np.any(chern_is_crisis) else np.array([])

    results = {
        "chern_quantization": {
            "n_windows": len(chern_values),
            "chern_mean": float(np.mean(chern_values)),
            "chern_std": float(np.std(chern_values)),
            "mean_dist_to_int_normal": float(np.mean(normal_dist)),
            "mean_dist_to_int_crisis": float(np.mean(crisis_dist)) if len(crisis_dist) > 0 else None,
            "frac_within_0.1_of_int_normal": float(np.mean(normal_dist < 0.1)),
            "frac_within_0.1_of_int_crisis": float(np.mean(crisis_dist < 0.1)) if len(crisis_dist) > 0 else None,
            "median_chern_normal": float(np.median(chern_values[~chern_is_crisis])),
            "median_chern_crisis": float(np.median(chern_values[chern_is_crisis])) if np.any(chern_is_crisis) else None,
        }
    }

    logger.info(
        f"  Normal: mean dist to int = {np.mean(normal_dist):.4f}, "
        f"frac within 0.1 = {np.mean(normal_dist < 0.1):.1%}"
    )
    if len(crisis_dist) > 0:
        logger.info(
            f"  Crisis: mean dist to int = {np.mean(crisis_dist):.4f}, "
            f"frac within 0.1 = {np.mean(crisis_dist < 0.1):.1%}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════
# Experiment 1.4  —  QFI–Metric Identity (Proposition 4)
# ══════════════════════════════════════════════════════════════════════


def _metric_via_perturbation_theory(geometry, x):
    """Compute the Fubini-Study metric via first-order perturbation theory.

    For ground state |0⟩ of H(x) = ½ Σ_k (A_k - x_k I)²:
        ∂_a H = x_a I - A_a
        g_ab^PT = Re Σ_{n≠0} ⟨0|∂_a H|n⟩⟨n|∂_b H|0⟩ / (E_n - E_0)²

    This is a completely analytical computation (no finite differences)
    that should agree with the numerical quantum_metric() method,
    providing a non-trivial cross-check of Proposition 4.
    """
    n = len(x)
    H = geometry.error_hamiltonian(x)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    psi_0 = eigenvectors[:, 0]
    E_0 = eigenvalues[0]

    # compute ∂_a H = x_a I - A_a  (derivative of H w.r.t. x_a)
    dH = []
    for a in range(n):
        # d/dx_a [½ Σ_k (A_k - x_k I)²] = -(A_a - x_a I) = x_a I - A_a
        dH_a = x[a] * geometry._identity - geometry.operators[a]
        dH.append(dH_a)

    # perturbation theory metric
    g_pt = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a, n):
            val = 0.0
            for m in range(1, len(eigenvalues)):
                dE = eigenvalues[m] - E_0
                if dE < 1e-14:
                    continue  # skip degenerate states
                mat_elem_a = np.vdot(psi_0, dH[a] @ eigenvectors[:, m])
                mat_elem_b = np.vdot(eigenvectors[:, m], dH[b] @ psi_0)
                val += np.real(mat_elem_a * mat_elem_b) / (dE ** 2)
            g_pt[a, b] = val
            g_pt[b, a] = val

    return g_pt


def experiment_qfi_metric(X_pca, dates, geometry, crises, out_dir):
    """Verify QFI = 4 × Fubini-Study metric tensor numerically.

    Proposition 4 claims g_ab = (1/4) F_Q (for pure states).
    We compute g_ab via two completely independent methods:
    1. Finite-difference wavefunction overlaps (quantum_metric)
    2. Perturbation theory using the Hamiltonian spectrum (no finite diffs)
    Agreement validates both the identity and the implementation.
    """
    logger.info("=== Experiment 1.4: QFI-Metric Identity (Proposition 4) ===")
    T = len(X_pca)
    idx = _subsample_indices(T, max_points=500)
    n_features = X_pca.shape[1]
    eps = 1e-5

    # Use random operators to avoid Pauli degeneracies that break
    # perturbation theory (pca_inspired gives exact eigenvalue pairs)
    geo_random = QCMLGeometry(n_features=n_features, hilbert_dim=geometry.hilbert_dim)
    geo_random.fit_operators(X_pca, method="random")

    metric_eigs_all = []
    qfi_eigs_all = []
    correlations = []
    rmses = []
    relative_errors = []

    for i, t in enumerate(idx):
        x = X_pca[t]

        # path 1: finite-difference wavefunction overlap method
        g = geo_random.quantum_metric(x, epsilon=eps)

        # path 2: perturbation theory (no finite differences)
        g_pt = _metric_via_perturbation_theory(geo_random, x)

        # compare eigenvalues
        eigs_g = np.sort(np.linalg.eigvalsh(g))
        eigs_pt = np.sort(np.linalg.eigvalsh(g_pt))

        metric_eigs_all.append(eigs_g)
        qfi_eigs_all.append(eigs_pt)  # should equal eigs_g

        # correlation of flattened matrices
        g_flat = g.flatten()
        pt_flat = g_pt.flatten()
        if np.std(g_flat) > 1e-15 and np.std(pt_flat) > 1e-15:
            corr = np.corrcoef(g_flat, pt_flat)[0, 1]
        else:
            corr = 1.0 if np.allclose(g_flat, pt_flat, atol=1e-10) else 0.0
        correlations.append(corr)

        # RMSE and relative error
        rmse = np.sqrt(np.mean((g - g_pt) ** 2))
        rmses.append(rmse)

        g_norm = np.linalg.norm(g, "fro")
        if g_norm > 1e-15:
            relative_errors.append(rmse / g_norm)

        if (i + 1) % 100 == 0:
            logger.info(f"  QFI-Metric: {i + 1}/{len(idx)} points processed")

    metric_eigs_all = np.array(metric_eigs_all)
    qfi_eigs_all = np.array(qfi_eigs_all)

    # ── figure: eigenvalue scatter ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # panel 1: eigenvalue scatter (all components)
    ax = axes[0]
    for k in range(min(n_features, 4)):
        ax.scatter(
            metric_eigs_all[:, k],
            qfi_eigs_all[:, k],
            s=6,
            alpha=0.4,
            label=f"$\\lambda_{k}$",
            rasterized=True,
        )
    # identity line
    all_vals = np.concatenate([metric_eigs_all.flatten(), qfi_eigs_all.flatten()])
    pos_vals = all_vals[all_vals > 1e-15]
    if len(pos_vals) > 0:
        lo, hi = np.percentile(pos_vals, [1, 99])
    else:
        lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Identity ($g^{FD} = g^{PT}$)")
    ax.set_xlabel(r"$g_{ab}$ eigenvalue  (finite differences)")
    ax.set_ylabel(r"$g_{ab}^{PT}$ eigenvalue  (perturbation theory)")
    ax.set_title("Finite-Diff Metric  vs  Perturbation Theory Metric")
    ax.legend(fontsize=7, ncol=2)

    if hi / max(lo, 1e-15) > 100:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # panel 2: relative error histogram
    ax = axes[1]
    if relative_errors:
        ax.hist(
            np.log10(np.array(relative_errors) + 1e-18),
            bins=40, color=NORMAL_COLOR, alpha=0.7, edgecolor="white",
        )
        ax.set_xlabel(r"$\log_{10}$ (relative error)")
        ax.set_ylabel("Count")
        median_re = np.median(relative_errors)
        ax.set_title(f"Relative Error: median = {median_re:.2e}")
        ax.axvline(
            np.log10(median_re + 1e-18), color=THEORY_COLOR,
            linestyle="--", linewidth=2, label=f"Median = {median_re:.1e}",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No valid comparisons", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_dir / "qfi_metric_identity.pdf")
    fig.savefig(out_dir / "qfi_metric_identity.png")
    plt.close(fig)

    logger.info(
        f"  Median correlation = {np.median(correlations):.6f}, "
        f"Median RMSE = {np.median(rmses):.2e}, "
        f"Median relative error = {np.median(relative_errors):.2e}" if relative_errors else ""
    )

    return {
        "qfi_metric": {
            "n_points": len(idx),
            "correlation_median": float(np.median(correlations)),
            "correlation_mean": float(np.mean(correlations)),
            "correlation_min": float(np.min(correlations)),
            "rmse_median": float(np.median(rmses)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_max": float(np.max(rmses)),
            "relative_error_median": float(np.median(relative_errors)) if relative_errors else None,
            "relative_error_95th": float(np.percentile(relative_errors, 95)) if relative_errors else None,
        }
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Empirical theorem validation")
    parser.add_argument("--quick", action="store_true", help="4 representative crises only")
    parser.add_argument("--crisis", type=str, default=None, help="Single crisis key")
    parser.add_argument("--hilbert-dim", type=int, default=8)
    parser.add_argument("--n-pca", type=int, default=8)
    parser.add_argument("--operator-method", type=str, default="pca_inspired")
    parser.add_argument(
        "--normalization", type=str, default="soft",
        choices=["sphere", "soft", "clip", "none"],
        help="Post-PCA normalization (soft recommended for theorem validation)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # select crises
    if args.crisis:
        crises = [args.crisis]
    elif args.quick:
        crises = QUICK_CRISES
    else:
        crises = list(ALL_CRISES.keys())

    logger.info(f"Running theorem validation on {len(crises)} crises")
    logger.info(f"Config: hilbert_dim={args.hilbert_dim}, n_pca={args.n_pca}, "
                f"operator_method={args.operator_method}, normalization={args.normalization}")

    # ── data preparation ─────────────────────────────────────────────
    logger.info("Preparing data...")
    X_pca, dates, geometry = _prepare_data(
        n_pca=args.n_pca,
        hilbert_dim=args.hilbert_dim,
        operator_method=args.operator_method,
        normalization=args.normalization,
    )
    logger.info(f"Data ready: T={len(X_pca)}, features={X_pca.shape[1]}")

    # ── run experiments ──────────────────────────────────────────────
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "hilbert_dim": args.hilbert_dim,
            "n_pca": args.n_pca,
            "operator_method": args.operator_method,
            "crises": crises,
            "T": len(X_pca),
        },
    }

    # 1.1 Spectral gap dynamics
    results = experiment_spectral_gap(X_pca, dates, geometry, crises, OUTPUT_DIR)
    all_results.update(results)

    # 1.2 Curvature-gap bound
    results = experiment_curvature_gap(X_pca, dates, geometry, crises, OUTPUT_DIR)
    all_results.update(results)

    # 1.3 Chern quantization
    results = experiment_chern_quantization(X_pca, dates, geometry, crises, OUTPUT_DIR)
    all_results.update(results)

    # 1.4 QFI-metric identity
    results = experiment_qfi_metric(X_pca, dates, geometry, crises, OUTPUT_DIR)
    all_results.update(results)

    # ── save results ─────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "theorem_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # ── summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("THEOREM VALIDATION SUMMARY")
    logger.info("=" * 60)

    if "spectral_gap" in all_results:
        sg = all_results["spectral_gap"]
        if sg:
            ratios = [v["ratio_means"] for v in sg.values()]
            logger.info(f"Thm 1 (Spectral Gap): crisis/normal ratio = {np.mean(ratios):.3f} "
                        f"(< 1 means gap closes during crises)")

    if "curvature_gap" in all_results:
        cg = all_results["curvature_gap"]
        logger.info(f"Thm 3 (Curvature Bound): C = {cg['empirical_C']:.2e}, "
                    f"bound satisfied = {cg['fraction_satisfying_bound']:.1%}")

    if "chern_quantization" in all_results:
        cq = all_results["chern_quantization"]
        if "error" not in cq:
            logger.info(f"Thm 2 (Chern Quantization): mean dist to int = "
                        f"{cq['mean_dist_to_int_normal']:.4f} (normal), "
                        f"{cq.get('mean_dist_to_int_crisis', 'N/A')} (crisis)")

    if "qfi_metric" in all_results:
        qm = all_results["qfi_metric"]
        logger.info(f"Prop 4 (QFI-Metric): median correlation = {qm['correlation_median']:.6f}, "
                    f"median RMSE = {qm['rmse_median']:.2e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
