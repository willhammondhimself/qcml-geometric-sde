"""
Full 17-crisis benchmark for Level Spacing Ratio Detector.

Runs LevelSpacingRatioDetector across ALL crises in the benchmark suite
with bootstrap confidence intervals (n=10,000). Computes:

1. Cohen's d per crisis x variant (mean_ratio, std_ratio, poisson_fraction)
2. Bootstrap 95% CIs for each d-value
3. Median d across crises per variant
4. Cross-correlation with existing top observables (if available)
5. Aggregate statistics and RMT comparison

Output: research/ideation/level_spacing/full_benchmark_results.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.level_spacing.detector import (
    LevelSpacingRatioDetector,
    _expanding_zscore,
    RMT_POISSON,
    RMT_GOE,
    RMT_GUE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Exclude pre-2005 crises if DIA data is unavailable
# DIA launched 1998-01-14, so 1997_asia might not have data
# We'll try all crises and skip those with insufficient data
SKIP_CRISES = set()  # Will be populated dynamically


def run_full_benchmark(
    n_bootstrap: int = 10_000,
    hilbert_dim: int = 8,
    n_pca_components: int = 8,
    operator_method: str = 'random',
    rolling_window: int = 20,
    min_expanding: int = 60,
    normalization: str = 'soft',
    seed: int = 42,
):
    """Run Level Spacing Ratio on all crises with bootstrap CIs."""
    logger.info("=" * 70)
    logger.info("LEVEL SPACING RATIO — FULL 17-CRISIS BENCHMARK")
    logger.info("=" * 70)

    # --- Fetch data (extended range for pre-2005 crises) ---
    logger.info("Fetching SPY + DIA data (1996-2025)...")
    t0 = time.time()
    try:
        raw = fetch_data(['SPY', 'DIA'], '1996-01-01', '2025-06-01')
        prices_df = raw['close'].unstack('symbol').dropna()
    except Exception as e:
        logger.warning(f"Extended range fetch failed ({e}), trying 2005-2025...")
        raw = fetch_data(['SPY', 'DIA'], '2005-01-01', '2025-06-01')
        prices_df = raw['close'].unstack('symbol').dropna()

    X, dates = create_feature_matrix(prices_df)
    dates_idx = pd.DatetimeIndex(dates)
    data_start = dates_idx.min()
    data_end = dates_idx.max()
    logger.info(
        f"Data: {X.shape[0]} obs x {X.shape[1]} features, "
        f"{data_start.date()} to {data_end.date()}, "
        f"fetched in {time.time() - t0:.1f}s"
    )

    # --- Fit detector ---
    logger.info(
        f"Fitting LevelSpacingRatioDetector (d={hilbert_dim}, "
        f"{operator_method}, {normalization})..."
    )
    t0 = time.time()
    detector = LevelSpacingRatioDetector(
        hilbert_dim=hilbert_dim,
        n_pca_components=n_pca_components,
        operator_method=operator_method,
        rolling_window=rolling_window,
        min_expanding=min_expanding,
        seed=seed,
        normalization=normalization,
    )
    detector.fit(X)
    fit_time = time.time() - t0
    logger.info(f"Fit completed in {fit_time:.1f}s")

    # --- Compute all variants ---
    logger.info("Computing level spacing observables...")
    t0 = time.time()
    raw_variants = detector.compute_all_variants(X)
    compute_time = time.time() - t0
    logger.info(f"All variants computed in {compute_time:.1f}s")

    # --- Z-score each variant ---
    T = len(X)
    variant_names = ['mean_ratio', 'std_ratio', 'poisson_fraction']
    z_scored = {}
    for vname in variant_names:
        z_scored[vname] = _expanding_zscore(
            raw_variants[vname], rolling_window, min_expanding, T,
        )

    # --- Build global normal mask (union of all crisis periods) ---
    all_crisis_mask = np.zeros(T, dtype=bool)
    for ck, cv in ALL_CRISES.items():
        cs = pd.Timestamp(cv['start'])
        ce = pd.Timestamp(cv['end'])
        all_crisis_mask |= (dates_idx >= cs) & (dates_idx <= ce)
    normal_mask = ~all_crisis_mask

    # --- Evaluate each crisis x variant ---
    logger.info(f"\nEvaluating {len(ALL_CRISES)} crises x {len(variant_names)} variants "
                f"(n_bootstrap={n_bootstrap})...")
    logger.info("-" * 100)

    results_per_variant = {vn: {} for vn in variant_names}
    d_values_per_variant = {vn: [] for vn in variant_names}
    crises_evaluated = []

    for crisis_key, crisis_info in ALL_CRISES.items():
        crisis_start = pd.Timestamp(crisis_info['start'])
        crisis_end = pd.Timestamp(crisis_info['end'])
        label = crisis_info['label']

        # Check if crisis falls within data range
        if crisis_start < data_start or crisis_end > data_end:
            # Check if we have at least SOME overlap
            overlap_mask = (dates_idx >= crisis_start) & (dates_idx <= crisis_end)
            if overlap_mask.sum() < 10:
                logger.info(f"  SKIP {crisis_key}: insufficient data overlap ({overlap_mask.sum()} days)")
                SKIP_CRISES.add(crisis_key)
                continue

        crisis_mask = (dates_idx >= crisis_start) & (dates_idx <= crisis_end)
        n_crisis_days = crisis_mask.sum()

        if n_crisis_days < 5:
            logger.info(f"  SKIP {crisis_key}: too few crisis days ({n_crisis_days})")
            SKIP_CRISES.add(crisis_key)
            continue

        crises_evaluated.append(crisis_key)

        for vname in variant_names:
            scores = z_scored[vname]
            valid = ~np.isnan(scores)

            crisis_scores = np.abs(scores[crisis_mask & valid])
            normal_scores = np.abs(scores[normal_mask & valid])

            if len(crisis_scores) < 5:
                d_val, ci_lo, ci_hi = 0.0, 0.0, 0.0
                p_val = 1.0
            else:
                d_val, ci_lo, ci_hi = compute_cohens_d_with_ci(
                    crisis_scores, normal_scores,
                    n_bootstrap=n_bootstrap, seed=seed,
                )
                _, p_val = stats.ttest_ind(
                    crisis_scores, normal_scores, equal_var=False,
                )

            results_per_variant[vname][crisis_key] = {
                'd': round(float(d_val), 4) if not np.isnan(d_val) else 0.0,
                'ci_lo': round(float(ci_lo), 4) if not np.isnan(ci_lo) else 0.0,
                'ci_hi': round(float(ci_hi), 4) if not np.isnan(ci_hi) else 0.0,
                'p': round(float(p_val), 6) if not np.isnan(p_val) else 1.0,
                'n_crisis': int(len(crisis_scores)),
                'label': label,
            }
            d_values_per_variant[vname].append(
                float(d_val) if not np.isnan(d_val) else 0.0
            )

        # Log progress
        mean_d = results_per_variant['mean_ratio'][crisis_key]['d']
        poisson_d = results_per_variant['poisson_fraction'][crisis_key]['d']
        logger.info(
            f"  {crisis_key:<22} mean_ratio d={mean_d:>6.3f}  "
            f"poisson_frac d={poisson_d:>6.3f}  "
            f"(n={n_crisis_days})"
        )

    # --- Summary table ---
    logger.info("\n" + "=" * 100)
    logger.info("FULL RESULTS TABLE")
    logger.info("=" * 100)

    header = f"{'Crisis':<22}"
    for vn in variant_names:
        header += f" {vn:>18}"
    logger.info(header)
    logger.info("-" * 100)

    for ck in crises_evaluated:
        row = f"{ck:<22}"
        for vn in variant_names:
            r = results_per_variant[vn][ck]
            row += f" {r['d']:>6.3f} [{r['ci_lo']:>5.2f},{r['ci_hi']:>5.2f}]"
        logger.info(row)

    logger.info("-" * 100)

    # Compute medians and means
    summary = {}
    for vn in variant_names:
        d_vals = d_values_per_variant[vn]
        summary[vn] = {
            'median_d': round(float(np.median(d_vals)), 4),
            'mean_d': round(float(np.mean(d_vals)), 4),
            'max_d': round(float(np.max(d_vals)), 4),
            'min_d': round(float(np.min(d_vals)), 4),
            'n_above_0.5': int(sum(1 for d in d_vals if d > 0.5)),
            'n_above_0.8': int(sum(1 for d in d_vals if d > 0.8)),
            'n_negative': int(sum(1 for d in d_vals if d < 0)),
            'per_crisis': results_per_variant[vn],
        }
        logger.info(
            f"  {vn:<22} median={summary[vn]['median_d']:>6.3f}  "
            f"mean={summary[vn]['mean_d']:>6.3f}  "
            f">0.8: {summary[vn]['n_above_0.8']}/{len(d_vals)}  "
            f"neg: {summary[vn]['n_negative']}/{len(d_vals)}"
        )

    # --- Cross-correlations between variants ---
    logger.info("\nCross-correlations between variants:")
    variant_correlations = {}
    for i, v1 in enumerate(variant_names):
        for v2 in variant_names[i + 1:]:
            s1, s2 = z_scored[v1], z_scored[v2]
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if np.sum(valid) > 10:
                corr = float(np.corrcoef(s1[valid], s2[valid])[0, 1])
                spearman = float(stats.spearmanr(s1[valid], s2[valid]).statistic)
                key = f"{v1}_vs_{v2}"
                variant_correlations[key] = {
                    'pearson': round(corr, 4),
                    'spearman': round(spearman, 4),
                }
                logger.info(f"  {key}: Pearson={corr:.4f}, Spearman={spearman:.4f}")

    # --- RMT comparison ---
    mean_r = raw_variants['mean_ratio']
    overall_mean = float(np.nanmean(mean_r))
    rmt_comparison = {
        'overall_mean_ratio': round(overall_mean, 4),
        'distance_to_Poisson': round(abs(overall_mean - RMT_POISSON), 4),
        'distance_to_GOE': round(abs(overall_mean - RMT_GOE), 4),
        'distance_to_GUE': round(abs(overall_mean - RMT_GUE), 4),
    }
    # Classify
    dists = {
        'Poisson': rmt_comparison['distance_to_Poisson'],
        'GOE': rmt_comparison['distance_to_GOE'],
        'GUE': rmt_comparison['distance_to_GUE'],
    }
    if overall_mean > RMT_GUE:
        rmt_comparison['closest_ensemble'] = 'above GUE (super-repulsive)'
    else:
        rmt_comparison['closest_ensemble'] = min(dists, key=dists.get)

    logger.info(f"\nRMT: overall <r> = {overall_mean:.4f}, "
                f"closest = {rmt_comparison['closest_ensemble']}")

    # --- Raw statistics ---
    raw_stats = {}
    for vname in variant_names:
        vals = raw_variants[vname]
        valid = vals[~np.isnan(vals)]
        raw_stats[vname] = {
            'mean': round(float(np.mean(valid)), 6),
            'std': round(float(np.std(valid)), 6),
            'min': round(float(np.min(valid)), 6),
            'max': round(float(np.max(valid)), 6),
            'q25': round(float(np.percentile(valid, 25)), 6),
            'q75': round(float(np.percentile(valid, 75)), 6),
        }

    # --- Best variant ---
    best_variant = max(
        variant_names, key=lambda vn: summary[vn]['median_d'],
    )
    best_median_d = summary[best_variant]['median_d']

    logger.info("\n" + "=" * 70)
    logger.info(f"BEST VARIANT: {best_variant} (median d = {best_median_d:.4f})")
    logger.info(f"CRISES EVALUATED: {len(crises_evaluated)}/{len(ALL_CRISES)}")
    if SKIP_CRISES:
        logger.info(f"SKIPPED: {sorted(SKIP_CRISES)}")
    logger.info("=" * 70)

    # --- Assemble output ---
    output = {
        'detector': 'LevelSpacingRatioDetector',
        'config': {
            'hilbert_dim': hilbert_dim,
            'n_pca_components': n_pca_components,
            'operator_method': operator_method,
            'normalization': normalization,
            'rolling_window': rolling_window,
            'min_expanding': min_expanding,
            'seed': seed,
            'n_bootstrap': n_bootstrap,
        },
        'data': {
            'n_observations': int(T),
            'n_features': int(X.shape[1]),
            'date_range': f"{data_start.date()} to {data_end.date()}",
        },
        'timing': {
            'fit_seconds': round(fit_time, 2),
            'compute_seconds': round(compute_time, 2),
        },
        'crises_evaluated': crises_evaluated,
        'crises_skipped': sorted(SKIP_CRISES),
        'best_variant': best_variant,
        'best_median_d': best_median_d,
        'per_variant_summary': summary,
        'variant_correlations': variant_correlations,
        'rmt_comparison': rmt_comparison,
        'raw_stats': raw_stats,
    }

    # Save
    output_path = Path(__file__).parent / 'full_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return output


if __name__ == '__main__':
    run_full_benchmark()
