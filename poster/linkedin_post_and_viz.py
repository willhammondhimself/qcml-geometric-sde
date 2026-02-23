"""
Generate LinkedIn-ready 3D Berry curvature and QFI surface visualizations.

Uses the QCMLGeometry class with real market data (SPY, QQQ, IWM via Polygon API)
to compute Berry curvature and quantum metric tensor over a 2D PCA grid, producing
visually striking surfaces that illustrate how Hilbert-space geometry deforms
during regime stress.

Outputs:
    poster/figures/linkedin_berry_surface.png
    poster/figures/linkedin_qfi_surface.png
    poster/figures/linkedin_hero_combined.png

Also prints the LinkedIn post text to stdout.

Usage:
    python poster/linkedin_post_and_viz.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.core import QCMLGeometry
from experiments.data_loader import fetch_polygon_data, create_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

BG = "#0D1117"
CARD_BG = "#161B22"
TEXT = "#F0F0F0"
GOLD = "#C8850F"


def _build_geometry_real(hilbert_dim: int = 8):
    """Fit QCMLGeometry on real market data from Polygon API.

    Fetches SPY, QQQ, IWM daily close prices (2005-2024) and builds a
    cross-sectional feature matrix for PCA-inspired operator construction.
    """
    logger.info("Fetching real market data from Polygon (SPY, QQQ, IWM)...")
    symbols = ['SPY', 'QQQ', 'IWM']
    prices_df = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')

    # Unstack to get close prices as columns
    close = prices_df['close'].unstack('symbol')
    logger.info(f"  Got {len(close)} trading days, {close.shape[1]} symbols")

    features, dates = create_feature_matrix(close)
    logger.info(f"  Feature matrix: {features.shape}")

    # Standardize for numerical stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # PCA to reduce to n_features dimensions for the geometry
    from sklearn.decomposition import PCA
    n_features = min(X.shape[1], 8)
    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X)
    logger.info(f"  PCA: {X.shape[1]} -> {n_features} features, "
                f"explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    geom = QCMLGeometry(n_features=n_features, hilbert_dim=hilbert_dim)
    geom.fit_operators(X_pca, method='pca_inspired')
    logger.info("  QCMLGeometry fitted with pca_inspired operators")

    return geom, X_pca, pca


def _compute_surfaces(geom, X, grid_res=60):
    """Compute Berry curvature and QFI over a 2D grid in PCA space.

    Scans over the first two principal components while holding other
    dimensions at their mean values.
    """
    mean = X.mean(axis=0)

    # Grid spans +/- 3 std along the first two PCA dimensions
    std0, std1 = X[:, 0].std(), X[:, 1].std()
    x_range = np.linspace(mean[0] - 3 * std0, mean[0] + 3 * std0, grid_res)
    y_range = np.linspace(mean[1] - 3 * std1, mean[1] + 3 * std1, grid_res)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    Z_berry = np.zeros_like(X_grid)
    Z_qfi = np.zeros_like(X_grid)

    for i in range(grid_res):
        for j in range(grid_res):
            point = mean.copy()
            point[0] = X_grid[i, j]
            point[1] = Y_grid[i, j]

            try:
                F_ab = geom.berry_curvature_2d(point, indices=(0, 1), epsilon=1e-4)
                Z_berry[i, j] = np.abs(F_ab)
            except Exception:
                Z_berry[i, j] = 0.0

            try:
                g = geom.quantum_metric(point, epsilon=1e-4)
                eigvals = np.linalg.eigvalsh(g)
                eigvals = np.maximum(eigvals, 0)
                Z_qfi[i, j] = (
                    np.prod(eigvals[eigvals > 1e-12])
                    if np.any(eigvals > 1e-12) else 0.0
                )
            except Exception:
                Z_qfi[i, j] = 0.0

    # Smooth to reduce numerical noise from finite-difference derivatives
    Z_berry = gaussian_filter(Z_berry, sigma=1.2)
    Z_qfi = gaussian_filter(Z_qfi, sigma=1.2)

    return X_grid, Y_grid, Z_berry, Z_qfi


def _style_3d_axes(ax, xlabel, ylabel, zlabel, title):
    """Apply dark-theme styling to a 3D axis."""
    ax.set_facecolor(BG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(CARD_BG)
    ax.yaxis.pane.set_edgecolor(CARD_BG)
    ax.zaxis.pane.set_edgecolor(CARD_BG)

    ax.set_xlabel(xlabel, color=TEXT, fontsize=13, labelpad=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=13, labelpad=10)
    ax.set_zlabel(zlabel, color=TEXT, fontsize=13, labelpad=10)
    ax.set_title(title, color=TEXT, fontsize=17, fontweight='bold', pad=20)

    ax.tick_params(axis='x', colors=TEXT, labelsize=9)
    ax.tick_params(axis='y', colors=TEXT, labelsize=9)
    ax.tick_params(axis='z', colors=TEXT, labelsize=9)
    ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0.08)
    ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0.08)
    ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0.08)


def fig_berry_surface(X_grid, Y_grid, Z_berry):
    """3D surface of Berry curvature magnitude with turbo colormap."""
    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=Z_berry.min(), vmax=Z_berry.max())
    surf = ax.plot_surface(
        X_grid, Y_grid, Z_berry,
        cmap=cm.turbo, norm=norm,
        rstride=1, cstride=1,
        antialiased=True, alpha=0.92,
        edgecolor='none',
    )

    _style_3d_axes(
        ax,
        xlabel="PC$_1$ (Market Direction)",
        ylabel="PC$_2$ (Volatility)",
        zlabel="$|F_{12}|$  (Berry Curvature)",
        title="Berry Curvature Landscape — Real Equity Data (2005–2024)",
    )
    ax.view_init(elev=28, azim=-55)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.1)
    cbar.set_label("Curvature Magnitude", color=TEXT, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=9)

    out = OUT_DIR / "linkedin_berry_surface.png"
    fig.savefig(out, dpi=300, facecolor=BG, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    logger.info(f"Saved {out}")


def fig_qfi_surface(X_grid, Y_grid, Z_qfi):
    """3D surface of QFI pseudo-determinant with plasma colormap."""
    Z_plot = np.log1p(Z_qfi)

    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=Z_plot.min(), vmax=Z_plot.max())
    surf = ax.plot_surface(
        X_grid, Y_grid, Z_plot,
        cmap=cm.plasma, norm=norm,
        rstride=1, cstride=1,
        antialiased=True, alpha=0.92,
        edgecolor='none',
    )

    _style_3d_axes(
        ax,
        xlabel="PC$_1$ (Market Direction)",
        ylabel="PC$_2$ (Volatility)",
        zlabel="$\\log(1 + \\det\\, g)$  (QFI)",
        title="Fisher Information Surface — Real Equity Data (2005–2024)",
    )
    ax.view_init(elev=30, azim=-60)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.1)
    cbar.set_label("log(1 + det g)", color=TEXT, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=9)

    out = OUT_DIR / "linkedin_qfi_surface.png"
    fig.savefig(out, dpi=300, facecolor=BG, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    logger.info(f"Saved {out}")


def fig_combined(X_grid, Y_grid, Z_berry, Z_qfi):
    """Side-by-side hero image with both surfaces."""
    Z_qfi_plot = np.log1p(Z_qfi)

    fig = plt.figure(figsize=(22, 9), facecolor=BG)

    ax1 = fig.add_subplot(121, projection='3d')
    norm1 = Normalize(vmin=Z_berry.min(), vmax=Z_berry.max())
    ax1.plot_surface(
        X_grid, Y_grid, Z_berry,
        cmap=cm.turbo, norm=norm1,
        rstride=1, cstride=1, antialiased=True, alpha=0.92, edgecolor='none',
    )
    _style_3d_axes(
        ax1,
        xlabel="PC$_1$", ylabel="PC$_2$",
        zlabel="$|F_{12}|$",
        title="Berry Curvature",
    )
    ax1.view_init(elev=28, azim=-55)

    ax2 = fig.add_subplot(122, projection='3d')
    norm2 = Normalize(vmin=Z_qfi_plot.min(), vmax=Z_qfi_plot.max())
    ax2.plot_surface(
        X_grid, Y_grid, Z_qfi_plot,
        cmap=cm.plasma, norm=norm2,
        rstride=1, cstride=1, antialiased=True, alpha=0.92, edgecolor='none',
    )
    _style_3d_axes(
        ax2,
        xlabel="PC$_1$", ylabel="PC$_2$",
        zlabel="$\\log(1 + \\det\\, g)$",
        title="Fisher Information",
    )
    ax2.view_init(elev=30, azim=-60)

    fig.suptitle(
        "Geometric Observables for Regime Detection — SPY / QQQ / IWM (2005–2024)",
        color=TEXT, fontsize=22, fontweight='bold', y=0.97,
    )

    out = OUT_DIR / "linkedin_hero_combined.png"
    fig.savefig(out, dpi=300, facecolor=BG, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    logger.info(f"Saved {out}")


# Latest results from comparison_20260222_212020.json (16 crises, 11 methods)
# Median Cohen's d across 12 evaluable crises (pre-2005 crises null due to data limits)
LINKEDIN_POST = """
Excited to present my solo-authored research at the 2026 APS Global Physics Summit in Denver — Job Seeker Poster Session, Tuesday March 17th.

My paper, "Geometric Observables for Financial Regime Detection," models equity time series as states in a projective Hilbert space. The Fubini-Study metric and Berry curvature yield three unsupervised observables — Berry curvature rate, QFI determinant, and multi-lag fidelity — for regime detection.

Key results (12 crises, 2007-2024, 10,000-sample bootstrap CIs):
  - QFI Determinant: median Cohen's d = 0.61
  - Berry Curvature Rate: median d = 0.55
  - Multi-Lag Fidelity: median d = 0.42
  - vs. Random Forest (supervised): median d = 0.22
  - Best geometric observable wins on 11 of 12 crises vs RF
  - Zero look-ahead bias: global-PCA pipeline with causal walk-forward

These are honest, fully unsupervised numbers with fixed default hyperparameters — no cherry-picking, no tuning per crisis.

Also looking forward to discussing my systematic equity strategy work at the career fair.

If you'll be at APS in Denver, let me know — happy to talk physics, geometry, and quantitative finance.

#APS2026 #Physics #QuantitativeFinance #MachineLearning #QuantResearch #DifferentialGeometry #SpectralGeometry
""".strip()


def main():
    geom, X, pca = _build_geometry_real(hilbert_dim=8)

    logger.info(f"Computing Berry curvature and QFI over 60x60 grid...")
    X_grid, Y_grid, Z_berry, Z_qfi = _compute_surfaces(geom, X, grid_res=60)

    logger.info("Rendering figures...")
    fig_berry_surface(X_grid, Y_grid, Z_berry)
    fig_qfi_surface(X_grid, Y_grid, Z_qfi)
    fig_combined(X_grid, Y_grid, Z_berry, Z_qfi)

    print("\n" + "=" * 70)
    print("LINKEDIN POST")
    print("=" * 70)
    print(LINKEDIN_POST)
    print("=" * 70)
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
