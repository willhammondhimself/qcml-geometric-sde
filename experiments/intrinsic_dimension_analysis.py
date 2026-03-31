"""
Intrinsic dimension analysis via QCML quantum metric eigenvalue spectrum.

Demonstrates that the quantum metric g(x) exhibits a spectral gap at the
intrinsic dimension of the data manifold (Candelori et al. 2025), and that
this dimension shifts during financial crises (dimension collapse).

Computes:
1. Eigenvalue spectrum of g(x) at each rolling window position
2. Spectral gap location → local dimension estimate d_eff(t)
3. Participation ratio of metric eigenvalues (alternative estimator)
4. Comparison: QCML d_eff vs PCA effective rank vs Absorption Ratio

Usage:
    python experiments/intrinsic_dimension_analysis.py
    python experiments/intrinsic_dimension_analysis.py --quick

Outputs:
    experiments/outputs/intrinsic_dimension/eigenvalue_spectra.pdf
    experiments/outputs/intrinsic_dimension/dimension_timeseries.pdf
    experiments/outputs/intrinsic_dimension/dimension_by_regime.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.core import QCMLGeometry
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.plot_style import (
    apply_style, save_figure, NAVY, TEAL, BURGUNDY, GOLD, INDIGO, SLATE,
    CMAP_SEQUENTIAL,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

np.random.seed(42)

OUTPUT_DIR = ROOT / 'experiments' / 'outputs' / 'intrinsic_dimension'


def compute_metric_eigenvalues(
    X_pca: np.ndarray,
    hilbert_dim: int = 8,
    operator_method: str = 'random',
    rolling_window: int = 20,
    epsilon: float = 1e-5,
    seed: int = 42,
) -> np.ndarray:
    """Compute eigenvalues of the quantum metric tensor at each time step.

    Args:
        X_pca: PCA-projected feature matrix, shape (T, p).
        hilbert_dim: Hilbert space dimension for QCML embedding.
        operator_method: 'random' or 'pca_inspired'.
        rolling_window: Rolling window for expanding normalization.
        epsilon: Finite-difference step for metric computation.
        seed: Random seed for operator construction.

    Returns:
        eigenvalues: Array of shape (T, p) with sorted eigenvalues at each step.
    """
    T, p = X_pca.shape
    qcml = QCMLGeometry(n_features=p, hilbert_dim=hilbert_dim)

    if operator_method == 'random':
        qcml.fit_operators(np.zeros((1, p)), method='random')
    else:
        qcml.fit_operators(np.zeros((1, p)), method='pca_inspired')

    eigenvalues = np.full((T, p), np.nan)

    for t in range(T):
        try:
            g = qcml.quantum_metric(X_pca[t], epsilon=epsilon)
            eigs = np.linalg.eigvalsh(g)
            eigenvalues[t] = np.sort(eigs)[::-1]  # descending
        except Exception as e:
            logger.warning(f"Metric computation failed at t={t}: {e}")

    return eigenvalues


def spectral_gap_dimension(eigenvalues: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Estimate intrinsic dimension from spectral gap in metric eigenvalues.

    The intrinsic dimension is estimated as the number of eigenvalues above
    a fraction of the largest eigenvalue, following the spectral gap criterion
    of Candelori et al. (2025).

    Args:
        eigenvalues: Array of shape (T, p), sorted descending.
        threshold: Fraction of lambda_max below which eigenvalues are noise.

    Returns:
        d_eff: Array of shape (T,) with estimated dimension at each step.
    """
    T, p = eigenvalues.shape
    d_eff = np.full(T, np.nan)

    for t in range(T):
        eigs = eigenvalues[t]
        if np.isnan(eigs[0]):
            continue
        lambda_max = eigs[0]
        if lambda_max <= 0:
            d_eff[t] = 0
            continue
        d_eff[t] = np.sum(eigs > threshold * lambda_max)

    return d_eff


def participation_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute participation ratio of metric eigenvalues.

    PR = (sum lambda_i)^2 / sum(lambda_i^2), a continuous dimension estimator.

    Args:
        eigenvalues: Array of shape (T, p), sorted descending.

    Returns:
        pr: Array of shape (T,) with participation ratio at each step.
    """
    T = eigenvalues.shape[0]
    pr = np.full(T, np.nan)

    for t in range(T):
        eigs = eigenvalues[t]
        if np.isnan(eigs[0]):
            continue
        pos_eigs = eigs[eigs > 0]
        if len(pos_eigs) == 0:
            pr[t] = 0
            continue
        pr[t] = np.sum(pos_eigs) ** 2 / np.sum(pos_eigs ** 2)

    return pr


def pca_effective_rank(X: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling PCA effective rank (exponential of Shannon entropy of eigenvalues).

    Args:
        X: Feature matrix, shape (T, d).
        window: Rolling window size.

    Returns:
        eff_rank: Array of shape (T,) with effective rank at each step.
    """
    from sklearn.decomposition import PCA

    T = X.shape[0]
    eff_rank = np.full(T, np.nan)

    for t in range(window, T):
        X_win = X[t - window:t]
        try:
            pca = PCA().fit(X_win)
            eigs = pca.explained_variance_ratio_
            eigs = eigs[eigs > 1e-10]
            entropy = -np.sum(eigs * np.log(eigs))
            eff_rank[t] = np.exp(entropy)
        except Exception:
            pass

    return eff_rank


def absorption_ratio(X: np.ndarray, window: int = 60, n_top: int = 1) -> np.ndarray:
    """Rolling Absorption Ratio (Kritzman et al. 2011).

    Fraction of total variance explained by top eigenvalue(s) of correlation matrix.

    Args:
        X: Feature matrix, shape (T, d).
        window: Rolling window size.
        n_top: Number of top eigenvalues to include.

    Returns:
        ar: Array of shape (T,) with absorption ratio at each step.
    """
    T = X.shape[0]
    ar = np.full(T, np.nan)

    for t in range(window, T):
        X_win = X[t - window:t]
        try:
            corr = np.corrcoef(X_win.T)
            eigs = np.linalg.eigvalsh(corr)
            eigs = np.sort(eigs)[::-1]
            ar[t] = np.sum(eigs[:n_top]) / np.sum(eigs)
        except Exception:
            pass

    return ar


def make_crisis_mask(dates: pd.DatetimeIndex, crises: dict) -> pd.Series:
    """Create boolean crisis mask from crisis definitions.

    Args:
        dates: DatetimeIndex aligned with feature matrix.
        crises: Dict of crisis definitions with 'start' and 'end' keys.

    Returns:
        Boolean Series indexed by dates.
    """
    mask = pd.Series(False, index=dates)
    for crisis_info in crises.values():
        start = pd.Timestamp(crisis_info['start'])
        end = pd.Timestamp(crisis_info['end'])
        mask |= (dates >= start) & (dates <= end)
    return mask


def plot_eigenvalue_spectra(
    eigenvalues: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_mask: pd.Series,
    output_dir: Path,
):
    """Plot eigenvalue spectra at calm vs crisis timepoints.

    Generates Figure for Section 2.4: shows spectral gap at intrinsic dimension,
    and how it shifts during crises.
    """
    apply_style()

    valid = ~np.isnan(eigenvalues[:, 0])
    calm_idx = np.where(valid & ~np.asarray(crisis_mask))[0]
    crisis_idx = np.where(valid & np.asarray(crisis_mask))[0]

    if len(calm_idx) == 0 or len(crisis_idx) == 0:
        logger.warning("Insufficient data for eigenvalue spectra plot")
        return

    # Sample representative spectra
    n_samples = min(200, len(calm_idx), len(crisis_idx))
    calm_sample = calm_idx[np.linspace(0, len(calm_idx) - 1, n_samples, dtype=int)]
    crisis_sample = crisis_idx[np.linspace(0, len(crisis_idx) - 1, n_samples, dtype=int)]

    calm_spectra = eigenvalues[calm_sample]
    crisis_spectra = eigenvalues[crisis_sample]

    p = eigenvalues.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Mean eigenvalue spectra (log scale)
    ax = axes[0]
    calm_mean = np.nanmean(calm_spectra, axis=0)
    calm_std = np.nanstd(calm_spectra, axis=0)
    crisis_mean = np.nanmean(crisis_spectra, axis=0)
    crisis_std = np.nanstd(crisis_spectra, axis=0)

    x = np.arange(1, p + 1)
    ax.semilogy(x, calm_mean, 'o-', color=NAVY, label='Normal', markersize=5)
    ax.fill_between(x, np.maximum(calm_mean - calm_std, 1e-15),
                    calm_mean + calm_std, alpha=0.2, color=NAVY)
    ax.semilogy(x, crisis_mean, 's-', color=BURGUNDY, label='Crisis', markersize=5)
    ax.fill_between(x, np.maximum(crisis_mean - crisis_std, 1e-15),
                    crisis_mean + crisis_std, alpha=0.2, color=BURGUNDY)

    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel(r'$\lambda_i(g)$')
    ax.set_title('(a) Quantum metric eigenvalue spectrum')
    ax.legend(frameon=False)

    # Mark approximate spectral gap
    ratio = calm_mean[:-1] / calm_mean[1:]
    gap_idx = np.argmax(ratio)
    ax.axvline(gap_idx + 1.5, color=GOLD, ls='--', alpha=0.7,
               label=f'Gap at $d \\approx {gap_idx + 1}$')
    ax.legend(frameon=False)

    # Panel (b): Eigenvalue ratio (consecutive)
    ax = axes[1]
    calm_ratios = calm_mean[:-1] / np.maximum(calm_mean[1:], 1e-15)
    crisis_ratios = crisis_mean[:-1] / np.maximum(crisis_mean[1:], 1e-15)

    ax.bar(x[:-1] - 0.15, calm_ratios, width=0.3, color=NAVY,
           alpha=0.8, label='Normal')
    ax.bar(x[:-1] + 0.15, crisis_ratios, width=0.3, color=BURGUNDY,
           alpha=0.8, label='Crisis')
    ax.set_xlabel('Eigenvalue index $i$')
    ax.set_ylabel(r'$\lambda_i / \lambda_{i+1}$')
    ax.set_title('(b) Consecutive eigenvalue ratio')
    ax.axhline(2.0, color=SLATE, ls=':', alpha=0.5, label='Gap threshold')
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, 'eigenvalue_spectra', output_dir=output_dir)
    logger.info(f"Saved eigenvalue_spectra to {output_dir}")


def plot_dimension_timeseries(
    dates: pd.DatetimeIndex,
    d_gap: np.ndarray,
    d_pr: np.ndarray,
    d_pca: np.ndarray,
    d_ar: np.ndarray,
    crisis_mask: pd.Series,
    output_dir: Path,
):
    """Plot d_eff(t) time series with crisis shading.

    4-panel figure comparing QCML spectral gap dimension, participation ratio,
    PCA effective rank, and absorption ratio.
    """
    apply_style()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Crisis shading
    for ax in axes:
        for crisis_key, crisis_info in ALL_CRISES.items():
            start = pd.Timestamp(crisis_info['start'])
            end = pd.Timestamp(crisis_info['end'])
            if start >= dates[0] and end <= dates[-1]:
                ax.axvspan(start, end, alpha=0.12, color=GOLD, zorder=0)

    # Panel 1: QCML spectral gap dimension
    axes[0].plot(dates, d_gap, color=NAVY, lw=0.8, alpha=0.9)
    axes[0].set_ylabel(r'$d_{\mathrm{gap}}$')
    axes[0].set_title('(a) QCML spectral gap dimension')

    # Panel 2: QCML participation ratio
    axes[1].plot(dates, d_pr, color=TEAL, lw=0.8, alpha=0.9)
    axes[1].set_ylabel(r'PR$(g)$')
    axes[1].set_title('(b) Metric participation ratio')

    # Panel 3: PCA effective rank
    axes[2].plot(dates, d_pca, color=INDIGO, lw=0.8, alpha=0.9)
    axes[2].set_ylabel(r'$d_{\mathrm{PCA}}$')
    axes[2].set_title('(c) PCA effective rank')

    # Panel 4: Absorption ratio
    axes[3].plot(dates, d_ar, color=BURGUNDY, lw=0.8, alpha=0.9)
    axes[3].set_ylabel('AR')
    axes[3].set_title('(d) Absorption Ratio (top-1)')
    axes[3].set_xlabel('Date')

    fig.tight_layout()
    save_figure(fig, 'dimension_timeseries', output_dir=output_dir)
    logger.info(f"Saved dimension_timeseries to {output_dir}")


def compute_regime_statistics(
    dates: pd.DatetimeIndex,
    d_gap: np.ndarray,
    d_pr: np.ndarray,
    d_pca: np.ndarray,
    d_ar: np.ndarray,
    crisis_mask: pd.Series,
) -> dict:
    """Compute dimension statistics by regime (normal vs crisis).

    Returns dict with per-regime means, medians, and Cohen's d for each estimator.
    """
    from scipy import stats as sp_stats

    mask = np.asarray(crisis_mask)
    results = {}

    for name, values in [
        ('qcml_spectral_gap', d_gap),
        ('participation_ratio', d_pr),
        ('pca_effective_rank', d_pca),
        ('absorption_ratio', d_ar),
    ]:
        valid = ~np.isnan(values)
        normal_vals = values[valid & ~mask]
        crisis_vals = values[valid & mask]

        if len(normal_vals) < 10 or len(crisis_vals) < 10:
            results[name] = {'error': 'insufficient data'}
            continue

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(normal_vals) - 1) * np.std(normal_vals, ddof=1) ** 2 +
             (len(crisis_vals) - 1) * np.std(crisis_vals, ddof=1) ** 2) /
            (len(normal_vals) + len(crisis_vals) - 2)
        )
        cohens_d = (np.mean(normal_vals) - np.mean(crisis_vals)) / pooled_std

        # Welch t-test
        t_stat, p_val = sp_stats.ttest_ind(normal_vals, crisis_vals, equal_var=False)

        results[name] = {
            'normal_mean': float(np.mean(normal_vals)),
            'normal_median': float(np.median(normal_vals)),
            'normal_std': float(np.std(normal_vals, ddof=1)),
            'crisis_mean': float(np.mean(crisis_vals)),
            'crisis_median': float(np.median(crisis_vals)),
            'crisis_std': float(np.std(crisis_vals, ddof=1)),
            'cohens_d': float(cohens_d),
            'welch_t': float(t_stat),
            'welch_p': float(p_val),
            'n_normal': int(len(normal_vals)),
            'n_crisis': int(len(crisis_vals)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Intrinsic dimension analysis')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer PCA components, smaller window')
    parser.add_argument('--hilbert-dim', type=int, default=8,
                        help='Hilbert space dimension (default: 8)')
    parser.add_argument('--n-pca', type=int, default=8,
                        help='Number of PCA components (default: 8)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory override')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    hilbert_dim = args.hilbert_dim
    n_pca = args.n_pca if not args.quick else 6
    logger.info(f"Intrinsic dimension analysis: hilbert_dim={hilbert_dim}, n_pca={n_pca}")

    # ---- Data loading ----
    logger.info("Fetching SPY/DIA data...")
    prices_df = fetch_data(['SPY', 'DIA'], '1998-01-01', '2024-12-31')
    # fetch_data returns MultiIndex (symbol, timestamp) with columns [open, high, low, close, volume]
    close_prices = prices_df['close'].unstack(level='symbol')

    features, dates = create_feature_matrix(close_prices)
    logger.info(f"Feature matrix: {features.shape}, dates: {dates[0]} to {dates[-1]}")

    # ---- Feature preprocessing ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f} "
                f"({n_pca} components)")

    # L2-normalize (sphere projection)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_pca_norm = X_pca / (norms + 1e-8)

    # ---- Compute QCML metric eigenvalues ----
    logger.info("Computing quantum metric eigenvalues (this may take several minutes)...")
    eigenvalues = compute_metric_eigenvalues(
        X_pca_norm,
        hilbert_dim=hilbert_dim,
        operator_method='random',
        epsilon=1e-5,
        seed=42,
    )

    valid_count = np.sum(~np.isnan(eigenvalues[:, 0]))
    logger.info(f"Computed metric at {valid_count}/{len(dates)} time steps")

    # ---- Dimension estimators ----
    logger.info("Computing dimension estimators...")
    d_gap = spectral_gap_dimension(eigenvalues, threshold=0.1)
    d_pr = participation_ratio(eigenvalues)
    d_pca = pca_effective_rank(features, window=60)
    d_ar = absorption_ratio(features, window=60, n_top=1)

    # ---- Crisis mask ----
    crisis_mask = make_crisis_mask(dates, ALL_CRISES)
    logger.info(f"Crisis days: {crisis_mask.sum()}, Normal days: {(~crisis_mask).sum()}")

    # ---- Plots ----
    logger.info("Generating plots...")
    plot_eigenvalue_spectra(eigenvalues, dates, crisis_mask, output_dir)
    plot_dimension_timeseries(dates, d_gap, d_pr, d_pca, d_ar, crisis_mask, output_dir)

    # ---- Regime statistics ----
    logger.info("Computing regime statistics...")
    regime_stats = compute_regime_statistics(
        dates, d_gap, d_pr, d_pca, d_ar, crisis_mask,
    )

    # Summary
    for name, stats in regime_stats.items():
        if 'error' in stats:
            logger.warning(f"  {name}: {stats['error']}")
        else:
            logger.info(
                f"  {name}: normal={stats['normal_mean']:.3f}±{stats['normal_std']:.3f}, "
                f"crisis={stats['crisis_mean']:.3f}±{stats['crisis_std']:.3f}, "
                f"Cohen's d={stats['cohens_d']:.3f}, p={stats['welch_p']:.2e}"
            )

    # Save results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'hilbert_dim': hilbert_dim,
            'n_pca': n_pca,
            'operator_method': 'random',
            'n_timesteps': int(valid_count),
            'n_crises': int(crisis_mask.sum()),
            'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
        },
        'regime_statistics': regime_stats,
        'eigenvalue_summary': {
            'mean_spectrum_normal': [
                float(v) for v in np.nanmean(
                    eigenvalues[~np.asarray(crisis_mask) & ~np.isnan(eigenvalues[:, 0])],
                    axis=0
                )
            ] if np.any(~np.asarray(crisis_mask) & ~np.isnan(eigenvalues[:, 0])) else [],
            'mean_spectrum_crisis': [
                float(v) for v in np.nanmean(
                    eigenvalues[np.asarray(crisis_mask) & ~np.isnan(eigenvalues[:, 0])],
                    axis=0
                )
            ] if np.any(np.asarray(crisis_mask) & ~np.isnan(eigenvalues[:, 0])) else [],
        },
    }

    results_path = output_dir / 'dimension_by_regime.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
