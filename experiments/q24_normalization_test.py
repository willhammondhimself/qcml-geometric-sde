"""
Q24 Quick Test: Normalization interaction with BerryPhaseRateDetector.

Tests sphere / soft / clip / none normalization on 4 key crises with
BerryPhaseRateDetector (HPO-optimal params from normalization_ablation.py).

Reports Cohen's d per normalization per crisis and mean across crises.
Note: A full ablation already exists in normalization_ablation.py (12 crises,
3 detectors). This quick test isolates the BerryPhaseRate finding and also
checks SpectralEntropy (the current #1 method) which was added after the
original ablation.

Usage:
    python experiments/q24_normalization_test.py
"""

import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import (
    BerryPhaseRateDetector,
    SpectralEntropyDetector,
    BaseRegimeDetector,
)

NORMS = ['sphere', 'soft', 'clip', 'none']
CRISIS_KEYS = ['2008_gfc', '2020_covid', '2011_euro', '2018_volmageddon']
N_BOOTSTRAP = 1000
SEED = 42

# HPO-optimal parameters from normalization_ablation.py / regime_comparison.py
DETECTOR_CONFIGS = {
    'BerryPhaseRate': {
        'cls': BerryPhaseRateDetector,
        'base_params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=SEED,
        ),
    },
    'SpectralEntropy': {
        'cls': SpectralEntropyDetector,
        'base_params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=SEED,
        ),
    },
}


def _crisis_masks(dates_enriched, crisis_info):
    cs = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    ce = pd.Timestamp(crisis_info['end']) + pd.Timedelta(days=10)
    crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
    return crisis_mask, ~crisis_mask


def _causal_fit_end(dates_enriched, crisis_info):
    cutoff = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    return max(100, int(np.searchsorted(dates_enriched, cutoff)))


def main():
    print("=" * 70)
    print("Q24: Normalization x Observable Interaction Test")
    print("=" * 70)

    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31', use_cache=True)
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    print(f"Data: {X_enriched.shape[0]} days, {X_enriched.shape[1]} features\n")

    crises = {k: ALL_CRISES[k] for k in CRISIS_KEYS if k in ALL_CRISES}

    # results[det_name][norm][crisis_key] = d
    all_results = {}

    for det_name, cfg in DETECTOR_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Detector: {det_name}")
        print(f"{'='*50}")
        all_results[det_name] = {nm: {} for nm in NORMS}

        for norm in NORMS:
            print(f"\n  normalization = {norm}")
            for ck, ci in crises.items():
                fit_end = _causal_fit_end(dates_enriched, ci)
                crisis_mask, normal_mask = _crisis_masks(dates_enriched, ci)

                params = {
                    **cfg['base_params'],
                    'causal_fit_length': fit_end,
                    'normalization': norm,
                    # adaptive_epsilon makes sense when data scale varies (not sphere)
                    'adaptive_epsilon': (norm != 'sphere'),
                }

                det = cfg['cls'](**params)
                det.fit(X_enriched)
                scores = det.compute_regime_scores(X_enriched)

                d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                    scores[crisis_mask], scores[normal_mask],
                    n_bootstrap=N_BOOTSTRAP, seed=SEED,
                )
                all_results[det_name][norm][ck] = float(d) if np.isfinite(d) else None
                d_str = f"{d:.3f}" if np.isfinite(d) else " N/A"
                print(f"    {ci['label']:<30}  d = {d_str}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ---- Summary table per detector ----
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Cohen's d across crises")
    print(f"  {'Detector':<20} {'sphere':>8} {'soft':>8} {'clip':>8} {'none':>8}  {'Best norm':>10}")
    print("-" * 70)

    for det_name in DETECTOR_CONFIGS:
        row_str = f"  {det_name:<20}"
        means = {}
        for norm in NORMS:
            ds = [v for v in all_results[det_name][norm].values() if v is not None]
            mean_d = np.mean(ds) if ds else np.nan
            means[norm] = mean_d
            row_str += f" {mean_d:>8.3f}" if np.isfinite(mean_d) else f" {'N/A':>8}"
        best_norm = max(means, key=lambda n: means[n] if np.isfinite(means[n]) else -999)
        best_d = means[best_norm]
        row_str += f"  {best_norm:>10} ({best_d:.3f})"
        print(row_str)
    print("=" * 70)

    # ---- Key interpretation ----
    print("\nKey findings:")
    for det_name in DETECTOR_CONFIGS:
        sphere_d = np.nanmean(
            [v for v in all_results[det_name]['sphere'].values() if v is not None]
        )
        best_norm = max(NORMS, key=lambda n: np.nanmean(
            [v for v in all_results[det_name][n].values() if v is not None] or [0]
        ))
        best_d = np.nanmean(
            [v for v in all_results[det_name][best_norm].values() if v is not None]
        )
        gain = best_d - sphere_d
        if gain > 0.05:
            print(f"  {det_name}: {best_norm} > sphere by {gain:+.3f} d -> normalization matters significantly")
        elif gain > 0.01:
            print(f"  {det_name}: {best_norm} > sphere by {gain:+.3f} d -> small consistent gain from {best_norm}")
        else:
            print(f"  {det_name}: normalization has negligible impact (max gain {gain:+.3f})")

    print("\nNote: 'sphere' = project to unit sphere (original default)")
    print("      'soft'   = divide by norm + median_norm (preserves magnitude)")
    print("      'clip'   = clip at ±5σ per component (outlier robust)")
    print("      'none'   = raw PCA output (full magnitude information)")


if __name__ == '__main__':
    main()
