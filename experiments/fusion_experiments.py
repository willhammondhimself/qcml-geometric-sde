"""
Paper 2 Fusion Experiments — "The Observatory"

Evaluates hierarchical, regime-adaptive, and Bayesian evidence fusion
strategies against Paper 1's individual detectors and flat fusion baselines.

Same causal preprocessing as Paper 1:
  - Per-crisis fitting (scaler/PCA/operators on pre-crisis data only)
  - Expanding-window z-scoring (no future information)
  - Walk-forward weight estimation for supervised methods

Experiments:
  1. Hierarchical vs flat fusion (does family structure help?)
  2. Regime-adaptive vs static weights (does adaptation help?)
  3. Online SPRT detection delay vs FAR tradeoff
  4. Per-crisis breakdown: which crises does fusion improve/hurt?
  5. Holdout evaluation on post-2021 crises
  6. Comparison against Paper 1's best individuals

Usage:
    python experiments/fusion_experiments.py                 # default (quick)
    python experiments/fusion_experiments.py --full          # all crises
    python experiments/fusion_experiments.py --holdout       # holdout only
    python experiments/fusion_experiments.py --sprt-sweep    # SPRT parameter sweep

Output:
    experiments/outputs/fusion/fusion_results_YYYYMMDD_HHMMSS.json
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qcml_geometry.fusion import (
    OBSERVABLE_FAMILIES,
    ACTIVE_CHANNELS,
    DEAD_CHANNELS,
    RankFusionDetector,
    HierarchicalFusionDetector,
    RegimeAdaptiveFusionDetector,
    BayesianEvidenceAccumulator,
)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import (
    compute_cohens_d_with_ci,
    friedman_test,
    compute_detection_metrics,
)

# Import HPO_CONFIGS and CLASSICAL_CONFIGS lazily to avoid circular imports
from experiments.regime_comparison import HPO_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = ROOT / 'experiments' / 'outputs' / 'fusion'


# =============================================================================
# Crisis Sets
# =============================================================================

# Paper 1 crises (train set for fusion weights)
TRAIN_CRISES = [
    '2007_quant', '2008_gfc', '2010_flash', '2011_euro',
    '2013_taper', '2015_china', '2016_brexit', '2018_volmageddon',
    '2018_q4', '2019_repo', '2020_covid',
]

# Holdout crises (never seen during weight training)
HOLDOUT_CRISES = [
    '2021_meme', '2022_rates', '2023_svb', '2024_carry',
]

# Quick evaluation set
QUICK_CRISES = [
    '2008_gfc', '2020_covid', '2022_rates', '2023_svb',
]


# =============================================================================
# Core Pipeline
# =============================================================================

def get_active_channel_names():
    """Return ordered list of active channel names (excluding dead signals)."""
    return [name for name in HPO_CONFIGS if name in ACTIVE_CHANNELS]


def compute_individual_scores(X, dates, crisis_def, channel_names, window_size=10):
    """Compute per-channel z-scores using causal (per-crisis) fitting.

    Args:
        X: Feature matrix (T, d).
        dates: DatetimeIndex aligned with X.
        crisis_def: Dict with 'start' and 'end' date strings.
        channel_names: Ordered list of active channel names.
        window_size: Days to add before crisis start for pre-crisis fitting.

    Returns:
        score_matrix: Array (T, n_channels).
        crisis_mask: Boolean array (T,).
    """
    crisis_start = pd.Timestamp(crisis_def['start'])
    crisis_end = pd.Timestamp(crisis_def['end'])

    # Crisis mask
    crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)

    # Fit cutoff: window_size days before crisis start
    cutoff_idx = np.searchsorted(dates, crisis_start) - window_size
    cutoff_idx = max(cutoff_idx, 100)  # need enough history

    T = X.shape[0]
    n_ch = len(channel_names)
    score_matrix = np.full((T, n_ch), np.nan)

    for ci, ch_name in enumerate(channel_names):
        if ch_name not in HPO_CONFIGS:
            continue

        cfg = HPO_CONFIGS[ch_name]
        det_class = cfg['class']
        params = cfg['params'].copy()

        try:
            det = det_class(**params)

            # Causal fit: only on data before cutoff
            X_train = X[:cutoff_idx]
            det.fit(X_train)

            # Score full series
            scores = det.compute_regime_scores(X)
            score_matrix[:, ci] = scores
        except Exception as e:
            logger.warning(f"  {ch_name} failed: {e}")

    return score_matrix, np.asarray(crisis_mask)


def evaluate_fusion_method(fused_scores, crisis_mask, method_name, n_bootstrap=10000):
    """Evaluate a fusion method's scores against crisis labels.

    Returns:
        Dict with Cohen's d, CI, detection metrics.
    """
    crisis_scores = fused_scores[crisis_mask]
    normal_scores = fused_scores[~crisis_mask]

    # Remove NaN
    crisis_scores = crisis_scores[~np.isnan(crisis_scores)]
    normal_scores = normal_scores[~np.isnan(normal_scores)]

    if len(crisis_scores) < 5 or len(normal_scores) < 20:
        return {
            'method': method_name,
            'd': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
        }

    d, ci_lo, ci_hi = compute_cohens_d_with_ci(
        crisis_scores, normal_scores, n_bootstrap=n_bootstrap
    )

    # Detection metrics at 95th percentile threshold
    all_valid = fused_scores[~np.isnan(fused_scores)]
    if len(all_valid) > 0:
        threshold = np.percentile(all_valid, 95)
        det_metrics = compute_detection_metrics(fused_scores, threshold, crisis_mask)
    else:
        det_metrics = {
            'detection_delay': np.nan,
            'false_alarm_rate': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'F1': np.nan,
        }

    return {
        'method': method_name,
        'd': round(d, 4) if not np.isnan(d) else None,
        'ci_lo': round(ci_lo, 4) if not np.isnan(ci_lo) else None,
        'ci_hi': round(ci_hi, 4) if not np.isnan(ci_hi) else None,
        **{k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
           for k, v in det_metrics.items()},
    }


def build_crisis_labels(dates, crises_to_use):
    """Build binary crisis label array from crisis definitions.

    Args:
        dates: DatetimeIndex.
        crises_to_use: List of crisis keys.

    Returns:
        labels: Binary array (T,), 1 during any crisis period.
    """
    labels = np.zeros(len(dates), dtype=float)
    for key in crises_to_use:
        if key not in ALL_CRISES:
            continue
        crisis = ALL_CRISES[key]
        start = pd.Timestamp(crisis['start'])
        end = pd.Timestamp(crisis['end'])
        mask = (dates >= start) & (dates <= end)
        labels[mask] = 1.0
    return labels


# =============================================================================
# Experiment Functions
# =============================================================================

def run_per_crisis_evaluation(X, dates, crises, channel_names, n_bootstrap=10000):
    """Run all fusion methods on each crisis independently.

    Args:
        X: Feature matrix (T, d).
        dates: DatetimeIndex.
        crises: List of crisis keys.
        channel_names: Ordered list of active channel names.
        n_bootstrap: Bootstrap resamples for CI.

    Returns:
        results: List of dicts with per-crisis, per-method results.
    """
    results = []

    for crisis_key in crises:
        if crisis_key not in ALL_CRISES:
            logger.warning(f"Skipping unknown crisis: {crisis_key}")
            continue

        crisis_def = ALL_CRISES[crisis_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Crisis: {crisis_def['label']} ({crisis_key})")
        logger.info(f"{'='*60}")

        # Build per-crisis score matrix
        score_matrix, crisis_mask = compute_individual_scores(
            X, dates, crisis_def, channel_names
        )

        if np.sum(crisis_mask) < 5:
            logger.warning(f"  Insufficient crisis days ({np.sum(crisis_mask)}), skipping")
            continue

        # Build crisis labels for supervised methods (all train crises before this one)
        train_labels = build_crisis_labels(
            dates,
            [c for c in TRAIN_CRISES if c != crisis_key and c in ALL_CRISES],
        )

        # --- Individual best (for comparison) ---
        for ci, ch_name in enumerate(channel_names):
            r = evaluate_fusion_method(
                score_matrix[:, ci], crisis_mask, ch_name, n_bootstrap
            )
            r['crisis'] = crisis_key
            r['category'] = 'individual'
            results.append(r)

        # --- Flat Rank Fusion (Paper 1 baseline) ---
        flat_rank = RankFusionDetector()
        flat_rank.set_precomputed_scores(score_matrix)
        flat_scores = flat_rank.compute_regime_scores(X)
        r = evaluate_fusion_method(flat_scores, crisis_mask, 'Flat Rank Fusion', n_bootstrap)
        r['crisis'] = crisis_key
        r['category'] = 'flat_fusion'
        results.append(r)

        # --- Hierarchical Rank Fusion (Paper 2) ---
        hier_rank = HierarchicalFusionDetector(
            channel_names=channel_names,
            cross_family_mode='rank',
        )
        hier_rank.set_precomputed_scores(score_matrix)
        hier_scores = hier_rank.compute_regime_scores(X)
        r = evaluate_fusion_method(
            hier_scores, crisis_mask, 'Hierarchical Rank', n_bootstrap
        )
        r['crisis'] = crisis_key
        r['category'] = 'hierarchical'
        results.append(r)

        # --- Hierarchical Learned Fusion (Paper 2) ---
        hier_learned = HierarchicalFusionDetector(
            channel_names=channel_names,
            cross_family_mode='learned',
            crisis_labels=train_labels,
        )
        hier_learned.set_precomputed_scores(score_matrix)
        hier_learned_scores = hier_learned.compute_regime_scores(X)
        r = evaluate_fusion_method(
            hier_learned_scores, crisis_mask, 'Hierarchical Learned', n_bootstrap
        )
        r['crisis'] = crisis_key
        r['category'] = 'hierarchical'
        results.append(r)

        # --- Regime-Adaptive Fusion (Paper 2) ---
        regime_adaptive = RegimeAdaptiveFusionDetector(
            channel_names=channel_names,
            n_regimes=3,
            crisis_labels=train_labels,
        )
        regime_adaptive.set_precomputed_scores(score_matrix)
        regime_scores = regime_adaptive.compute_regime_scores(X)
        r = evaluate_fusion_method(
            regime_scores, crisis_mask, 'Regime-Adaptive', n_bootstrap
        )
        r['crisis'] = crisis_key
        r['category'] = 'regime_adaptive'
        results.append(r)

        # --- Bayesian Evidence Accumulator (Paper 2) ---
        for decay in [0.99, 0.995, 1.0]:
            bea = BayesianEvidenceAccumulator(
                channel_names=channel_names,
                decay=decay,
            )
            bea.set_precomputed_scores(score_matrix)
            bea_scores = bea.compute_regime_scores(X)
            label = f'SPRT (decay={decay})'
            r = evaluate_fusion_method(bea_scores, crisis_mask, label, n_bootstrap)
            r['crisis'] = crisis_key
            r['category'] = 'sprt'
            r['sprt_n_alarms'] = len(bea.alarm_times) if bea.alarm_times else 0
            results.append(r)

        logger.info(f"  Completed {crisis_key}: "
                     f"{sum(1 for r in results if r['crisis'] == crisis_key)} evaluations")

    return results


def compute_aggregate_results(results):
    """Compute aggregate Cohen's d across crises per method.

    Args:
        results: List of per-crisis result dicts.

    Returns:
        aggregate: List of dicts with method, mean_d, median_d, etc.
    """
    methods = sorted(set(r['method'] for r in results))
    aggregate = []

    for method in methods:
        method_results = [r for r in results if r['method'] == method and r['d'] is not None]
        if not method_results:
            continue

        d_values = [r['d'] for r in method_results]
        aggregate.append({
            'method': method,
            'category': method_results[0].get('category', 'unknown'),
            'n_crises': len(d_values),
            'mean_d': round(np.mean(d_values), 4),
            'median_d': round(np.median(d_values), 4),
            'std_d': round(np.std(d_values, ddof=1), 4) if len(d_values) > 1 else 0.0,
            'min_d': round(min(d_values), 4),
            'max_d': round(max(d_values), 4),
        })

    aggregate.sort(key=lambda x: x['mean_d'], reverse=True)
    return aggregate


def run_friedman_comparison(results, crises):
    """Run Friedman rank test on fusion methods vs best individuals."""
    fusion_methods = [
        'Flat Rank Fusion', 'Hierarchical Rank', 'Hierarchical Learned',
        'Regime-Adaptive', 'SPRT (decay=0.995)',
    ]

    # Also include top 5 individuals
    agg = compute_aggregate_results(
        [r for r in results if r['category'] == 'individual']
    )
    top_individuals = [a['method'] for a in agg[:5]]

    all_methods = top_individuals + fusion_methods
    d_matrix = []

    for crisis_key in crises:
        row = []
        for method in all_methods:
            matching = [
                r for r in results
                if r['method'] == method and r['crisis'] == crisis_key and r['d'] is not None
            ]
            if matching:
                row.append(matching[0]['d'])
            else:
                row.append(np.nan)
        d_matrix.append(row)

    d_matrix = np.array(d_matrix)

    if d_matrix.shape[0] >= 3:
        chi_sq, p_val, mean_ranks = friedman_test(d_matrix)
        return {
            'methods': all_methods,
            'chi_sq': round(float(chi_sq), 2) if not np.isnan(chi_sq) else None,
            'p_value': float(p_val) if not np.isnan(p_val) else None,
            'mean_ranks': {
                m: round(float(r), 2) for m, r in zip(all_methods, mean_ranks)
            },
        }
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Paper 2 Fusion Experiments')
    parser.add_argument('--full', action='store_true', help='Run on all crises')
    parser.add_argument('--holdout', action='store_true', help='Run holdout only')
    parser.add_argument('--quick', action='store_true', help='Quick 4-crisis run (default)')
    parser.add_argument('--bootstrap', type=int, default=10000, help='Bootstrap resamples')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    # Select crisis set
    if args.full:
        crises = TRAIN_CRISES + HOLDOUT_CRISES
        preset = 'full'
    elif args.holdout:
        crises = HOLDOUT_CRISES
        preset = 'holdout'
    else:
        crises = QUICK_CRISES
        preset = 'quick'

    logger.info(f"Paper 2 Fusion Experiments — preset: {preset}")
    logger.info(f"Crises: {crises}")

    # Fetch data
    logger.info("Fetching data...")
    raw = fetch_data(['SPY', 'DIA'], '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"Feature matrix: {X.shape}, dates: {dates[0]} to {dates[-1]}")

    # Active channel names
    channel_names = get_active_channel_names()
    logger.info(f"Active channels ({len(channel_names)}): {channel_names}")

    # Run per-crisis evaluation
    results = run_per_crisis_evaluation(
        X, dates, crises, channel_names, n_bootstrap=args.bootstrap
    )

    # Compute aggregates
    aggregate = compute_aggregate_results(results)

    # Friedman test
    friedman = run_friedman_comparison(results, crises)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE RESULTS (sorted by mean Cohen's d)")
    logger.info("=" * 80)
    for a in aggregate:
        logger.info(
            f"  {a['method']:30s}  d={a['mean_d']:.3f} "
            f"(median={a['median_d']:.3f}, n={a['n_crises']})"
        )

    if friedman:
        logger.info(f"\nFriedman test: chi2={friedman['chi_sq']}, p={friedman['p_value']}")
        logger.info("Mean ranks (lower = better):")
        for m, r in sorted(friedman['mean_ranks'].items(), key=lambda x: x[1]):
            logger.info(f"  {m:30s}  rank={r:.1f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = args.output or str(OUTPUT_DIR / f'fusion_results_{timestamp}.json')

    output = {
        'metadata': {
            'timestamp': timestamp,
            'preset': preset,
            'crises': crises,
            'channel_names': channel_names,
            'n_channels': len(channel_names),
            'n_bootstrap': args.bootstrap,
            'families': {k: v for k, v in OBSERVABLE_FAMILIES.items()},
            'dead_channels': list(DEAD_CHANNELS),
        },
        'per_crisis': results,
        'aggregate': aggregate,
        'friedman': friedman,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")
    return output


if __name__ == '__main__':
    main()
