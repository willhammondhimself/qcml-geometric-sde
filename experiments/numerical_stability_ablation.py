"""
Numerical stability ablation study (Review Item 7).

Tests sensitivity of Berry curvature, QFI determinant, and Multi-Lag Fidelity to:
  1. Finite-difference epsilon: {1e-3, 1e-4, 1e-5, 1e-6} (Berry + QFI only)
  2. PCA dimension: {5, 10, 15, 30} (Berry + QFI + MLF)

Run on 4 representative crises with FIXED hyperparameters (no per-crisis tuning).
The point is stability, not optimization.

Usage:
    python experiments/numerical_stability_ablation.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import (
    fetch_polygon_data, create_feature_matrix, ALL_CRISES,
)
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)

EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6]
PCA_DIMS = [5, 10, 15, 30]
REPRESENTATIVE_CRISES = ['2008_gfc', '2010_flash', '2020_covid', '2023_svb']
HILBERT_DIM = 8
MLF_LAGS = [1, 3, 5, 10]


def compute_observable_timeseries(X_pca, geometry, observable, epsilon=1e-5):
    """Compute a raw observable time series.

    Args:
        X_pca: Transformed data (T, p).
        geometry: Fitted QCMLGeometry.
        observable: 'berry', 'qfi_det', or 'mlf'.
        epsilon: Finite-difference step size (used by berry/qfi_det).

    Returns:
        1-D array of length T.
    """
    T = len(X_pca)
    values = np.empty(T)

    for t in range(T):
        x = X_pca[t]
        if observable == 'berry':
            values[t] = geometry.berry_curvature_2d(x, indices=(0, 1), epsilon=epsilon)
        elif observable == 'qfi_det':
            g = geometry.quantum_metric(x, epsilon=epsilon)
            eigs = np.linalg.eigvalsh(g)
            pos_eigs = eigs[eigs > 1e-10]
            values[t] = np.sum(np.log(pos_eigs)) if len(pos_eigs) > 0 else -50.0
        elif observable == 'mlf':
            values[t] = _compute_mlf_at_t(X_pca, t, geometry)
        else:
            raise ValueError(f"Unknown observable: {observable}")

    return values


def _compute_mlf_at_t(X_pca, t, geometry):
    """Compute multi-lag fidelity at time t.

    Fidelity = |<psi(t)|psi(t-lag)>|^2, averaged across MLF_LAGS.
    """
    fidelities = []
    for lag in MLF_LAGS:
        if t - lag < 0:
            continue
        x_now = X_pca[t]
        x_prev = X_pca[t - lag]
        # Quasi-coherent state overlap via inner product in Hilbert space
        psi_now = geometry.quasi_coherent_state(x_now)
        psi_prev = geometry.quasi_coherent_state(x_prev)
        fid = abs(np.vdot(psi_now, psi_prev)) ** 2
        fidelities.append(fid)

    if len(fidelities) == 0:
        return 1.0  # No lags available, perfect fidelity
    # Return 1 - mean fidelity (so higher = more regime change)
    return 1.0 - np.mean(fidelities)


def z_score_series(raw, rolling_window=20, min_expanding=60):
    """Apply rolling smoothing + expanding z-score."""
    smoothed = pd.Series(raw).rolling(rolling_window, min_periods=1).mean().values
    T = len(smoothed)
    z = np.full(T, np.nan)
    for t in range(min_expanding, T):
        mu = np.mean(smoothed[:t])
        sigma = np.std(smoothed[:t], ddof=1)
        if sigma > 1e-12:
            z[t] = abs((smoothed[t] - mu) / sigma)
        else:
            z[t] = 0.0
    return z


def compute_d_for_config(X_enriched, dates_enriched, n_pca, epsilon, observable,
                         crisis_keys, window_ext=10):
    """Compute Cohen's d for a specific (n_pca, epsilon, observable) config.

    Returns:
        dict {crisis_key: d_value}
    """
    n_components = min(n_pca, X_enriched.shape[1])

    scaler = StandardScaler()
    scaler.fit(X_enriched)

    pca = PCA(n_components=n_components)
    X_scaled = scaler.transform(X_enriched)
    pca.fit(X_scaled)

    X_pca = pca.transform(X_scaled)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_pca = X_pca / (norms + 1e-8)

    geometry = QCMLGeometry(n_features=n_components, hilbert_dim=HILBERT_DIM)
    geometry.fit_operators(X_pca, method='pca_inspired')

    raw_values = compute_observable_timeseries(X_pca, geometry, observable, epsilon)

    if observable == 'berry':
        raw_values = np.abs(np.diff(raw_values))
        raw_values = np.concatenate([[0], raw_values])

    z_scores = z_score_series(raw_values)

    results = {}
    for ck in crisis_keys:
        ci = ALL_CRISES[ck]
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_ext)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_ext)

        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        d, _, _ = compute_cohens_d_with_ci(
            z_scores[crisis_mask], z_scores[normal_mask], n_bootstrap=1000
        )
        results[ck] = float(d) if not np.isnan(d) else None

    return results


def run_ablation():
    """Run full epsilon × PCA dimension ablation."""
    logger.info("=" * 70)
    logger.info("Numerical Stability Ablation Study")
    logger.info("=" * 70)

    # Fetch data
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Enriched features: {X_enriched.shape}")

    results = {}

    # Epsilon sweep: Berry and QFI only (MLF doesn't use epsilon)
    for observable in ['berry', 'qfi_det']:
        logger.info(f"\n[2] Observable: {observable}")
        results[observable] = {}

        # Epsilon sweep (fixed PCA = 15)
        logger.info("  Epsilon sweep (PCA=15):")
        for eps in EPSILONS:
            logger.info(f"    epsilon={eps}...")
            ds = compute_d_for_config(
                X_enriched, dates_enriched, n_pca=15, epsilon=eps,
                observable=observable, crisis_keys=REPRESENTATIVE_CRISES,
            )
            results[observable][f'eps_{eps}'] = ds
            median_d = np.nanmedian([v for v in ds.values() if v is not None])
            logger.info(f"      median d = {median_d:.3f}")

    # PCA dimension sweep: Berry, QFI, AND MLF (fixed epsilon = 1e-5)
    for observable in ['berry', 'qfi_det', 'mlf']:
        if observable not in results:
            results[observable] = {}
        logger.info(f"\n[3] PCA dimension sweep for {observable} (eps=1e-5):")
        for n_pca in PCA_DIMS:
            logger.info(f"    n_pca={n_pca}...")
            ds = compute_d_for_config(
                X_enriched, dates_enriched, n_pca=n_pca, epsilon=1e-5,
                observable=observable, crisis_keys=REPRESENTATIVE_CRISES,
            )
            results[observable][f'pca_{n_pca}'] = ds
            median_d = np.nanmedian([v for v in ds.values() if v is not None])
            logger.info(f"      median d = {median_d:.3f}")

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)

    for observable in ['berry', 'qfi_det', 'mlf']:
        logger.info(f"\n{observable.upper()}:")
        logger.info(f"  {'Config':20s}" + "".join(f"  {ck:15s}" for ck in REPRESENTATIVE_CRISES) + "  Median")

        for config_key, ds in results.get(observable, {}).items():
            vals = [ds.get(ck) for ck in REPRESENTATIVE_CRISES]
            median = np.nanmedian([v for v in vals if v is not None])
            row = f"  {config_key:20s}"
            for v in vals:
                row += f"  {v:15.3f}" if v is not None else f"  {'N/A':>15s}"
            row += f"  {median:.3f}"
            logger.info(row)

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'epsilons': EPSILONS,
        'pca_dims': PCA_DIMS,
        'crises': REPRESENTATIVE_CRISES,
        'hilbert_dim': HILBERT_DIM,
        'results': results,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'numerical_stability'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'stability_ablation_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def main():
    run_ablation()


if __name__ == '__main__':
    main()
