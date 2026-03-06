"""
Empirical adiabatic verification: check r(t) = v(t)/Delta(t) distribution.

The paper invokes the adiabatic theorem (Born-Fock 1928, Kato 1966) but
never verifies whether r(t) << 1 actually holds. This script computes r(t)
using the SpeedLimitRatioDetector and reports the distribution in normal
vs. crisis periods.

If r(t) >> 1 during crises (expected), the Berry phase should be reframed
as a "discrete Wilson loop" rather than an adiabatic holonomy.

Usage:
    python experiments/adiabatic_check.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import _apply_normalization, _transform_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES as CRISES


def compute_raw_speed_limit_ratio(X, hilbert_dim=8, n_pca=8, seed=42):
    """Compute raw r(t) = v(t) / Delta(t) without z-scoring.

    Args:
        X: Feature matrix, shape (T, d).
        hilbert_dim: Hilbert space dimension.
        n_pca: Number of PCA components.
        seed: Random seed.

    Returns:
        velocity: Fubini-Study velocity per timestep, shape (T,).
        gaps: Spectral gap per timestep, shape (T,).
        ratio: v(t)/Delta(t) per timestep, shape (T,).
    """
    np.random.seed(seed)
    T, d = X.shape
    n_components = min(n_pca, d)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    train_norms = np.linalg.norm(X_pca, axis=1)
    train_std = np.std(X_pca, axis=0)
    X_norm = _apply_normalization(X_pca, 'soft', train_norms, train_std)

    geo = QCMLGeometry(n_features=n_components, hilbert_dim=hilbert_dim)
    geo.fit_operators(X_norm, method='random', scale_exponent=None)

    # Compute states and gaps
    states = []
    gaps = np.empty(T)
    for t in range(T):
        states.append(geo.quasi_coherent_state(X_norm[t]))
        gaps[t] = geo.spectral_gap(X_norm[t])

    # FS velocity
    velocity = np.full(T, np.nan)
    for t in range(1, T):
        overlap = np.abs(np.vdot(states[t], states[t - 1]))
        overlap = np.clip(overlap, 0.0, 1.0)
        velocity[t] = np.arccos(overlap)

    # Speed limit ratio
    ratio = np.full(T, np.nan)
    valid = (~np.isnan(velocity)) & (gaps > 1e-6)
    ratio[valid] = velocity[valid] / gaps[valid]

    return velocity, gaps, ratio


def main():
    print("=" * 60)
    print("EMPIRICAL ADIABATIC VERIFICATION")
    print("=" * 60)

    # Fetch data (same pipeline as regime_comparison.py)
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    print(f"Data: {len(X)} observations, {X.shape[1]} features")

    # Compute raw r(t)
    print("\nComputing speed limit ratio r(t) = v(t)/Delta(t)...")
    velocity, gaps, ratio = compute_raw_speed_limit_ratio(X)

    valid = ~np.isnan(ratio)
    print(f"Valid r(t) values: {np.sum(valid)}/{len(ratio)}")

    # Overall statistics
    r_valid = ratio[valid]
    print(f"\nOverall r(t) distribution:")
    print(f"  Mean:   {np.mean(r_valid):.4f}")
    print(f"  Median: {np.median(r_valid):.4f}")
    print(f"  Std:    {np.std(r_valid):.4f}")
    print(f"  P25:    {np.percentile(r_valid, 25):.4f}")
    print(f"  P75:    {np.percentile(r_valid, 75):.4f}")
    print(f"  P95:    {np.percentile(r_valid, 95):.4f}")
    print(f"  Max:    {np.max(r_valid):.4f}")
    print(f"  Frac > 1: {np.mean(r_valid > 1):.1%}")
    print(f"  Frac > 0.1: {np.mean(r_valid > 0.1):.1%}")

    # Per-crisis analysis
    results = {}
    print(f"\n{'Crisis':<25} {'Normal r(t)':<20} {'Crisis r(t)':<20} {'Ratio':<10}")
    print("-" * 75)

    for crisis_key, ci in CRISES.items():
        start = pd.Timestamp(ci['start'])
        end = pd.Timestamp(ci['end'])

        crisis_mask = (dates >= start) & (dates <= end)
        normal_mask = ~crisis_mask & valid

        if np.sum(crisis_mask & valid) < 5:
            continue

        r_crisis = ratio[crisis_mask & valid]
        r_normal = ratio[normal_mask]

        med_crisis = float(np.median(r_crisis))
        med_normal = float(np.median(r_normal))
        ratio_val = med_crisis / med_normal if med_normal > 0 else float('inf')

        results[crisis_key] = {
            'crisis_median': med_crisis,
            'crisis_mean': float(np.mean(r_crisis)),
            'crisis_p95': float(np.percentile(r_crisis, 95)),
            'crisis_frac_gt_1': float(np.mean(r_crisis > 1)),
            'normal_median': med_normal,
            'normal_mean': float(np.mean(r_normal)),
            'crisis_to_normal_ratio': ratio_val,
        }

        print(f"{crisis_key:<25} {med_normal:.4f}              {med_crisis:.4f}              {ratio_val:.2f}x")

    # Adiabatic assessment
    overall_frac_gt_1 = float(np.mean(r_valid > 1))
    crisis_all = []
    normal_all = []
    for crisis_key, ci in CRISES.items():
        start = pd.Timestamp(ci['start'])
        end = pd.Timestamp(ci['end'])
        crisis_mask = (dates >= start) & (dates <= end) & valid
        normal_mask = ~((dates >= start) & (dates <= end)) & valid
        crisis_all.extend(ratio[crisis_mask].tolist())
        normal_all.extend(ratio[normal_mask].tolist())

    crisis_all = np.array(crisis_all)
    normal_all = np.array(normal_all)

    print(f"\n{'=' * 60}")
    print("ADIABATIC ASSESSMENT")
    print(f"{'=' * 60}")
    print(f"Normal periods:  median r = {np.median(normal_all):.4f}, "
          f"frac(r>1) = {np.mean(normal_all > 1):.1%}")
    print(f"Crisis periods:  median r = {np.median(crisis_all):.4f}, "
          f"frac(r>1) = {np.mean(crisis_all > 1):.1%}")

    if np.median(normal_all) < 0.1:
        assessment = "ADIABATIC in normal periods (r << 1)"
    elif np.median(normal_all) < 1.0:
        assessment = "QUASI-ADIABATIC in normal periods (r < 1 but not << 1)"
    else:
        assessment = "NON-ADIABATIC even in normal periods (r >= 1)"

    if np.median(crisis_all) > 1.0:
        assessment += "; DIABATIC during crises (r > 1)"
    elif np.median(crisis_all) > 0.1:
        assessment += "; BORDERLINE during crises (0.1 < r < 1)"

    print(f"\nAssessment: {assessment}")

    if np.median(normal_all) > 0.1 or np.median(crisis_all) > 1.0:
        print("\nRECOMMENDATION: Reframe Berry phase as 'discrete Wilson loop'")
        print("rather than adiabatic holonomy. The geometric phase is well-defined")
        print("regardless of adiabaticity, but the adiabatic interpretation does")
        print("not hold during crisis periods.")
    else:
        print("\nAdiabatic approximation appears valid.")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'mean_r': float(np.mean(r_valid)),
            'median_r': float(np.median(r_valid)),
            'std_r': float(np.std(r_valid)),
            'frac_gt_1': overall_frac_gt_1,
            'frac_gt_01': float(np.mean(r_valid > 0.1)),
        },
        'normal_aggregate': {
            'median_r': float(np.median(normal_all)),
            'mean_r': float(np.mean(normal_all)),
            'frac_gt_1': float(np.mean(normal_all > 1)),
        },
        'crisis_aggregate': {
            'median_r': float(np.median(crisis_all)),
            'mean_r': float(np.mean(crisis_all)),
            'frac_gt_1': float(np.mean(crisis_all > 1)),
        },
        'per_crisis': results,
        'assessment': assessment,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'adiabatic'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'adiabatic_check_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return output


if __name__ == '__main__':
    main()
