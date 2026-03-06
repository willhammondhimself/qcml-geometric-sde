"""
Q22 Quick Test: Multi-scale SpectralEntropy fusion vs single-scale.

Runs SpectralEntropyDetector at 4 window sizes [10, 30, 60, 120] on 4 key crises
(2008_gfc, 2020_covid, 2011_euro, 2018_volmageddon) and compares:
  - Single best scale Cohen's d
  - Mean of z-scores (simple fusion)
  - RMS of z-scores (energy-based fusion)
  - Rank-average fusion

All fits are causal (causal_fit_length set to 10 days before crisis start).
n_bootstrap=1000 for speed (quick test, not paper-quality).

Usage:
    python experiments/q22_multiscale_test.py
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
from qcml_geometry.observables import SpectralEntropyDetector, BaseRegimeDetector


WINDOWS = [10, 30, 60, 120]
# Use strong signals across crises with varied character
CRISIS_KEYS = ['2008_gfc', '2020_covid', '2011_euro', '2018_volmageddon']
N_BOOTSTRAP = 1000  # quick-test quality
SEED = 42


def _crisis_masks(dates_enriched, crisis_info):
    """Return (crisis_mask, normal_mask) boolean arrays for a crisis."""
    cs = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    ce = pd.Timestamp(crisis_info['end']) + pd.Timedelta(days=10)
    crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
    return crisis_mask, ~crisis_mask


def _causal_fit_end(dates_enriched, crisis_info):
    """Index of last date >=10 days before crisis start."""
    cutoff = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    return max(100, int(np.searchsorted(dates_enriched, cutoff)))


def rank_normalize(arr):
    """Map finite values to [0,1] ranks; NaN stays NaN."""
    out = np.full_like(arr, np.nan)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return out
    ranks = arr[valid].argsort().argsort().astype(float)
    ranks /= ranks.max()
    out[valid] = ranks
    return out


def main():
    print("=" * 70)
    print("Q22: Multi-scale SpectralEntropy Fusion Test")
    print("=" * 70)

    # ---- Load data ----
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31', use_cache=True)
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    print(f"Data: {X_enriched.shape[0]} days, {X_enriched.shape[1]} features\n")

    crises = {k: ALL_CRISES[k] for k in CRISIS_KEYS if k in ALL_CRISES}
    results = {}  # crisis_key -> {'w10': d, ..., 'mean_fuse': d, 'rms_fuse': d, 'rank_fuse': d}

    for ck, ci in crises.items():
        print(f"\n--- {ci['label']} ---")
        fit_end = _causal_fit_end(dates_enriched, ci)
        crisis_mask, normal_mask = _crisis_masks(dates_enriched, ci)
        crisis_dates_enriched = dates_enriched

        # --- Compute scores at each window ---
        scores_per_window = {}
        for w in WINDOWS:
            det = SpectralEntropyDetector(
                hilbert_dim=8,
                n_pca_components=8,
                operator_method='random',
                rolling_window=w,
                min_expanding=60,
                seed=SEED,
                causal_fit_length=fit_end,
                normalization='soft',
                adaptive_epsilon=True,
            )
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)
            scores_per_window[w] = scores

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=N_BOOTSTRAP, seed=SEED
            )
            print(f"  window={w:3d}d:  d = {d:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

        # Best single-window d
        best_w = max(scores_per_window, key=lambda w: np.nanmean(
            scores_per_window[w][crisis_mask]) - np.nanmean(scores_per_window[w][normal_mask])
        )

        # --- Fusion: mean of z-scores ---
        T = X_enriched.shape[0]
        stack = np.stack([scores_per_window[w] for w in WINDOWS], axis=1)  # (T, 4)
        # Replace NaN with column median for fusion (only for fusion; individual d computed raw)
        stack_filled = stack.copy()
        for col in range(stack_filled.shape[1]):
            med = np.nanmedian(stack_filled[:, col])
            stack_filled[np.isnan(stack_filled[:, col]), col] = med

        mean_fused = np.nanmean(stack_filled, axis=1)
        rms_fused = np.sqrt(np.nanmean(stack_filled ** 2, axis=1))

        # Rank fusion: rank-normalize each scale then average
        rank_cols = np.stack([rank_normalize(stack[:, j]) for j in range(4)], axis=1)
        rank_fused = np.nanmean(rank_cols, axis=1)

        for name, fused in [('mean_fuse', mean_fused), ('rms_fuse', rms_fused), ('rank_fuse', rank_fused)]:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                fused[crisis_mask], fused[normal_mask], n_bootstrap=N_BOOTSTRAP, seed=SEED
            )
            print(f"  {name:12s}:  d = {d:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

        # Store best individual vs best fusion
        best_w_scores = scores_per_window[best_w]
        d_best_single, _, _ = compute_cohens_d_with_ci(
            best_w_scores[crisis_mask], best_w_scores[normal_mask],
            n_bootstrap=N_BOOTSTRAP, seed=SEED,
        )
        d_rms, _, _ = compute_cohens_d_with_ci(
            rms_fused[crisis_mask], rms_fused[normal_mask],
            n_bootstrap=N_BOOTSTRAP, seed=SEED,
        )
        results[ck] = {
            'label': ci['label'],
            'best_single_w': best_w,
            'd_best_single': float(d_best_single),
            'd_rms_fuse': float(d_rms),
            'delta': float(d_rms - d_best_single),
        }

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY: Best single-window vs RMS-fused multi-scale")
    print(f"{'Crisis':<35} {'Best-single d':>14} {'RMS-fuse d':>12} {'Delta':>8}")
    print("-" * 70)
    deltas = []
    for ck, r in results.items():
        delta_str = f"{r['delta']:+.3f}"
        print(f"  {r['label']:<33} {r['d_best_single']:>14.3f} {r['d_rms_fuse']:>12.3f} {delta_str:>8}")
        deltas.append(r['delta'])
    print("-" * 70)
    print(f"  {'Mean improvement':<33} {'':>14} {'':>12} {np.mean(deltas):>+8.3f}")
    print("=" * 70)

    # ---- Interpretation ----
    mean_delta = np.mean(deltas)
    print(f"\nInterpretation:")
    if mean_delta > 0.05:
        print(f"  Multi-scale fusion adds +{mean_delta:.3f} d on average -> MODERATE gain")
        print(f"  Recommendation: quick_test -> worth a full HPO run")
    elif mean_delta > 0:
        print(f"  Multi-scale fusion adds +{mean_delta:.3f} d on average -> MARGINAL gain")
        print(f"  Recommendation: future_work (cost vs. benefit unclear)")
    else:
        print(f"  Multi-scale fusion does NOT improve ({mean_delta:.3f}) -> window averaging hurts")
        print(f"  Recommendation: abandon (pick best single window per HPO)")


if __name__ == '__main__':
    main()
