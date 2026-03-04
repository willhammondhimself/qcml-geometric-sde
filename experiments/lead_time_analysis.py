"""
Lead time analysis for regime detection methods.

Computes how many trading days before each crisis each detector first triggers
an alarm, using expanding-window thresholds (no lookahead). Provides
statistical tests comparing geometric vs classical lead times.

The script re-runs all detectors to obtain raw score time series,
then computes lead times from expanding-window thresholds.

Usage:
    python experiments/lead_time_analysis.py
    python experiments/lead_time_analysis.py --lookback 180

Outputs:
    experiments/outputs/lead_time/lead_time_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import friedman_test
from experiments.regime_comparison import HPO_CONFIGS, CLASSICAL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

np.random.seed(42)


def compute_lead_time(
    scores: np.ndarray,
    dates: np.ndarray,
    crisis_start: pd.Timestamp,
    lookback_days: int = 252,
    min_persistence: int = 3,
    threshold_quantile: float = 0.95,
    min_expanding: int = 60,
) -> dict:
    """Compute lead time for a single method on a single crisis.

    Lead time = number of trading days between the first persistent alarm
    and the crisis start date. Only alarms within `lookback_days` before
    the crisis are considered.

    Args:
        scores: 1-D array of regime scores (same length as dates).
        dates: Array of pd.Timestamp dates.
        crisis_start: Crisis start date.
        lookback_days: Maximum days before crisis to search for alarms.
        min_persistence: Minimum consecutive alarm days.
        threshold_quantile: Quantile for expanding threshold.
        min_expanding: Minimum history before threshold is computed.

    Returns:
        dict with:
            lead_time_days: int or NaN (trading days before crisis)
            first_alarm_date: str or None
            n_early_alarm_episodes: int (number of distinct alarm runs)
    """
    scores = np.asarray(scores, dtype=float)
    dates = pd.DatetimeIndex(dates)
    T = len(scores)

    # Find crisis start index
    crisis_idx = dates.searchsorted(crisis_start)
    if crisis_idx >= T:
        return {'lead_time_days': np.nan, 'first_alarm_date': None,
                'n_early_alarm_episodes': 0}

    # Lookback window
    lookback_start = max(0, crisis_idx - lookback_days)

    # Compute expanding threshold (causal: only uses data up to each point)
    thresholds = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = scores[:t]
        valid = past[np.isfinite(past)]
        if len(valid) > 0:
            thresholds[t] = np.nanpercentile(valid, threshold_quantile * 100)

    # Alarm: score exceeds expanding threshold
    alarm = np.zeros(T, dtype=bool)
    for t in range(min_expanding, T):
        if np.isfinite(scores[t]) and np.isfinite(thresholds[t]):
            alarm[t] = scores[t] > thresholds[t]

    # Apply persistence filter
    alarm = _persistence_filter(alarm, min_persistence)

    # Search for alarms in [lookback_start, crisis_idx)
    pre_crisis_alarm = alarm[lookback_start:crisis_idx]
    pre_crisis_dates = dates[lookback_start:crisis_idx]

    if not np.any(pre_crisis_alarm):
        return {'lead_time_days': np.nan, 'first_alarm_date': None,
                'n_early_alarm_episodes': 0}

    # Find first alarm
    first_alarm_idx = np.argmax(pre_crisis_alarm)
    first_alarm_date = pre_crisis_dates[first_alarm_idx]
    lead_time = crisis_idx - (lookback_start + first_alarm_idx)

    # Count distinct alarm episodes
    n_episodes = _count_episodes(pre_crisis_alarm)

    return {
        'lead_time_days': int(lead_time),
        'first_alarm_date': str(pd.Timestamp(first_alarm_date).date()),
        'n_early_alarm_episodes': n_episodes,
    }


def _persistence_filter(alarm: np.ndarray, min_run: int) -> np.ndarray:
    """Remove alarm runs shorter than min_run consecutive days."""
    result = np.zeros_like(alarm, dtype=bool)
    T = len(alarm)
    i = 0
    while i < T:
        if alarm[i]:
            j = i
            while j < T and alarm[j]:
                j += 1
            if (j - i) >= min_run:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def _count_episodes(alarm: np.ndarray) -> int:
    """Count number of distinct True runs."""
    count = 0
    in_run = False
    for v in alarm:
        if v and not in_run:
            count += 1
            in_run = True
        elif not v:
            in_run = False
    return count


def run_lead_time_analysis(
    lookback_days: int = 252,
):
    """Run lead time analysis by computing raw scores for all methods.

    Args:
        lookback_days: Primary lookback window (trading days).

    Returns:
        dict with lead time matrix, statistical tests, and metadata.
    """
    logger.info("=" * 70)
    logger.info("LEAD TIME ANALYSIS")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols + ['^VIX'], '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()

    if '^VIX' in prices_df.columns:
        vix_series = prices_df['^VIX'].copy()
        prices_df = prices_df.drop(columns=['^VIX'])
    else:
        vix_series = None

    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Feature matrix: {X_enriched.shape}, "
                f"dates: {dates_enriched[0].date()} to {dates_enriched[-1].date()}")

    # VIX alignment
    if vix_series is not None:
        vix_aligned = vix_series.reindex(dates).values
        vix_enriched = vix_aligned[19:]
    else:
        vix_enriched = None

    # ---- Determine valid crises ----
    crisis_keys = []
    for key, info in ALL_CRISES.items():
        crisis_start = pd.Timestamp(info['start'])
        crisis_idx = np.searchsorted(dates_enriched, crisis_start)
        if crisis_idx > 252:  # Need enough history
            crisis_keys.append(key)
    logger.info(f"  Valid crises for lead time: {len(crisis_keys)}")

    # ---- Compute scores for all methods ----
    logger.info(f"\n[2] Computing scores for {len(HPO_CONFIGS) + len(CLASSICAL_CONFIGS)} methods...")
    all_scores = {}

    # QCML detectors
    for method_name, config in HPO_CONFIGS.items():
        logger.info(f"  {method_name}...")
        params = config['params'].copy()
        det = config['class'](**params)

        try:
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)
            all_scores[method_name] = scores
        except Exception as e:
            logger.warning(f"  {method_name} failed: {e}")

    # Classical baselines
    for method_name, config in CLASSICAL_CONFIGS.items():
        logger.info(f"  {method_name}...")
        det = config['class'](**config['params'])

        try:
            det.fit(X_enriched)
            if hasattr(det, 'set_vix') and vix_enriched is not None:
                det.set_vix(vix_enriched)
            scores = det.compute_regime_scores(X_enriched)
            all_scores[method_name] = scores
        except Exception as e:
            logger.warning(f"  {method_name} failed: {e}")

    logger.info(f"  Successfully scored: {len(all_scores)} methods")

    # ---- Compute lead times ----
    logger.info(f"\n[3] Computing lead times (lookback={lookback_days} days)...")
    method_names = sorted(all_scores.keys())
    lead_time_matrix = {}

    for method_name in method_names:
        scores = all_scores[method_name]
        lead_time_matrix[method_name] = {}

        for crisis_key in crisis_keys:
            crisis_start = pd.Timestamp(ALL_CRISES[crisis_key]['start'])
            lt = compute_lead_time(
                scores, dates_enriched, crisis_start,
                lookback_days=lookback_days,
            )
            lead_time_matrix[method_name][crisis_key] = lt
            if lt['lead_time_days'] is not None and np.isfinite(lt.get('lead_time_days', np.nan)):
                logger.info(f"    {method_name} -> {crisis_key}: "
                            f"{lt['lead_time_days']} days (alarm: {lt['first_alarm_date']})")

    # Build numpy matrix for statistical tests
    lt_np = np.array([
        [lead_time_matrix[m][c].get('lead_time_days', np.nan) for c in crisis_keys]
        for m in method_names
    ])  # shape: (n_methods, n_crises)

    # ---- Statistical tests ----
    logger.info("\n[4] Statistical tests...")
    test_results = _run_statistical_tests(lt_np, method_names, crisis_keys)

    # ---- Summary statistics ----
    summary = {}
    for i, method in enumerate(method_names):
        valid = lt_np[i][np.isfinite(lt_np[i])]
        summary[method] = {
            'mean_lead_time': float(np.mean(valid)) if len(valid) > 0 else None,
            'median_lead_time': float(np.median(valid)) if len(valid) > 0 else None,
            'max_lead_time': float(np.max(valid)) if len(valid) > 0 else None,
            'n_detected': int(len(valid)),
            'n_total': len(crisis_keys),
        }

    output = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'lookback_days': lookback_days,
            'n_methods': len(method_names),
            'n_crises': len(crisis_keys),
            'crisis_keys': crisis_keys,
        },
        'lead_time_matrix': {
            m: {c: lead_time_matrix[m][c] for c in crisis_keys}
            for m in method_names
        },
        'summary': summary,
        'statistical_tests': test_results,
    }

    # Save output
    out_dir = ROOT / 'experiments' / 'outputs' / 'lead_time'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'lead_time_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary
    _print_summary(summary, method_names, test_results)

    return output


def _run_statistical_tests(lt_matrix, method_names, crisis_keys):
    """Run statistical tests on lead time matrix.

    Args:
        lt_matrix: (n_methods, n_crises) array of lead times.
        method_names: List of method names.
        crisis_keys: List of crisis keys.

    Returns:
        dict with test results.
    """
    results = {}

    # Identify geometric vs classical methods
    geometric_indices = []
    classical_indices = []
    for i, m in enumerate(method_names):
        is_classical = any(kw in m.lower() for kw in [
            'rolling vol', 'cusum', 'hmm', 'bocpd', 'isolation',
            'random forest', 'vix', 'garch', 'hamilton', 'ewma',
            'mahalanobis', 'structural', 'transfer entropy',
            'rolling rf',
        ])
        if is_classical:
            classical_indices.append(i)
        else:
            geometric_indices.append(i)

    # 1. Friedman rank test on full lead time matrix
    # Need at least 3 methods with detections across crises
    lt_for_friedman = lt_matrix.T  # (n_crises, n_methods)
    # Replace NaN with 0 for methods that didn't detect
    lt_friedman_filled = np.where(np.isfinite(lt_for_friedman), lt_for_friedman, 0)
    try:
        chi_sq, p_val, mean_ranks = friedman_test(lt_friedman_filled)
        results['friedman'] = {
            'chi_sq': float(chi_sq) if np.isfinite(chi_sq) else None,
            'p_value': float(p_val) if np.isfinite(p_val) else None,
            'mean_ranks': {m: float(r) for m, r in zip(method_names, mean_ranks)},
        }
    except Exception as e:
        results['friedman'] = {'error': str(e)}

    # 2. Paired Wilcoxon signed-rank (geometric vs baseline lead times per crisis)
    if geometric_indices and classical_indices:
        geo_leads = lt_matrix[geometric_indices]  # (n_geo, n_crises)
        cls_leads = lt_matrix[classical_indices]  # (n_cls, n_crises)

        # Median lead time per crisis for each category
        geo_medians = np.nanmedian(geo_leads, axis=0)
        cls_medians = np.nanmedian(cls_leads, axis=0)

        valid = np.isfinite(geo_medians) & np.isfinite(cls_medians)
        if np.sum(valid) >= 5:
            try:
                w_stat, w_pval = stats.wilcoxon(
                    geo_medians[valid], cls_medians[valid],
                    alternative='greater',
                )
                results['wilcoxon_geo_vs_classical'] = {
                    'W_statistic': float(w_stat),
                    'p_value': float(w_pval),
                    'n_pairs': int(np.sum(valid)),
                    'geo_median_lead': float(np.nanmedian(geo_medians[valid])),
                    'cls_median_lead': float(np.nanmedian(cls_medians[valid])),
                }
            except Exception as e:
                results['wilcoxon_geo_vs_classical'] = {'error': str(e)}

    # 3. Bootstrap CIs for mean lead time per method
    rng = np.random.default_rng(42)
    bootstrap_cis = {}
    for i, method in enumerate(method_names):
        valid = lt_matrix[i][np.isfinite(lt_matrix[i])]
        if len(valid) < 3:
            bootstrap_cis[method] = {'mean': None, 'ci_lo': None, 'ci_hi': None}
            continue
        boot_means = np.array([
            np.mean(rng.choice(valid, size=len(valid), replace=True))
            for _ in range(10000)
        ])
        bootstrap_cis[method] = {
            'mean': float(np.mean(valid)),
            'ci_lo': float(np.percentile(boot_means, 2.5)),
            'ci_hi': float(np.percentile(boot_means, 97.5)),
        }
    results['bootstrap_cis'] = bootstrap_cis

    # 4. Permutation null (scrambled lead times)
    if geometric_indices:
        geo_leads_flat = lt_matrix[geometric_indices].flatten()
        geo_leads_flat = geo_leads_flat[np.isfinite(geo_leads_flat)]
        if len(geo_leads_flat) >= 5:
            observed_mean = np.mean(geo_leads_flat)
            all_leads = lt_matrix.flatten()
            all_leads = all_leads[np.isfinite(all_leads)]
            null_means = []
            rng_perm = np.random.default_rng(42)
            for _ in range(5000):
                perm_sample = rng_perm.choice(all_leads, size=len(geo_leads_flat), replace=False)
                null_means.append(np.mean(perm_sample))
            null_means = np.array(null_means)
            p_perm = np.mean(null_means >= observed_mean)
            results['permutation_test'] = {
                'observed_mean_geo': float(observed_mean),
                'null_mean': float(np.mean(null_means)),
                'p_value': float(p_perm),
                'n_permutations': 5000,
            }

    return results


def _print_summary(summary, method_names, test_results):
    """Print a formatted summary table."""
    logger.info("\n" + "=" * 70)
    logger.info("LEAD TIME ANALYSIS SUMMARY")
    logger.info("=" * 70)

    # Sort by mean lead time (descending = earliest alarm)
    sorted_methods = sorted(
        method_names,
        key=lambda m: summary[m]['mean_lead_time'] or 0,
        reverse=True,
    )

    logger.info(f"\n{'Method':<30} {'Mean':>8} {'Median':>8} {'Max':>8} {'Det':>6}")
    logger.info("-" * 64)
    for m in sorted_methods:
        s = summary[m]
        mean_str = f"{s['mean_lead_time']:.0f}" if s['mean_lead_time'] else "N/A"
        med_str = f"{s['median_lead_time']:.0f}" if s['median_lead_time'] else "N/A"
        max_str = f"{s['max_lead_time']:.0f}" if s['max_lead_time'] else "N/A"
        logger.info(f"{m:<30} {mean_str:>8} {med_str:>8} {max_str:>8} "
                     f"{s['n_detected']:>3}/{s['n_total']}")

    friedman = test_results.get('friedman', {})
    if friedman.get('chi_sq'):
        logger.info(f"\nFriedman chi² = {friedman['chi_sq']:.1f}, "
                     f"p = {friedman['p_value']:.2e}")

    wilcoxon = test_results.get('wilcoxon_geo_vs_classical', {})
    if wilcoxon.get('p_value') is not None:
        logger.info(f"Wilcoxon (geo > classical): W = {wilcoxon['W_statistic']:.0f}, "
                     f"p = {wilcoxon['p_value']:.4f}")
        logger.info(f"  Geometric median lead: {wilcoxon['geo_median_lead']:.0f} days")
        logger.info(f"  Classical median lead: {wilcoxon['cls_median_lead']:.0f} days")

    perm = test_results.get('permutation_test', {})
    if perm.get('p_value') is not None:
        logger.info(f"Permutation test: p = {perm['p_value']:.4f} "
                     f"(geo mean = {perm['observed_mean_geo']:.0f}, "
                     f"null mean = {perm['null_mean']:.0f})")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lead time analysis')
    parser.add_argument('--lookback', type=int, default=252,
                        help='Lookback window in trading days')
    args = parser.parse_args()

    run_lead_time_analysis(lookback_days=args.lookback)
