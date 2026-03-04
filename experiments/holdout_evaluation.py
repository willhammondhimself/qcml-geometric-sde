"""
Blind holdout stress test for regime detection methods.

Protocol:
    1. Cutoff: 2021-12-31
    2. Freeze: Fit scaler, PCA, operators, HPO, thresholds, RF labels
       all on pre-2022 data only.
    3. Deploy: Apply frozen pipeline to 2022-01-01 through 2024-12-31.
    4. Evaluate: Cohen's d + bootstrap CI, detection delay, FAR, lead time
       on each holdout crisis.

Holdout crises:
    - 2022 Rate Hikes (slow burn, 10 months)
    - 2023 SVB (sharp, 2 months)
    - 2024 Carry Unwind (flash, 6 weeks)

Key difference from walk-forward: walk-forward re-fits annually;
holdout freezes everything at cutoff.  Holdout is stricter — tests
whether geometry learned from 2021 data still detects crises 3 years later.

Usage:
    python experiments/holdout_evaluation.py
    python experiments/holdout_evaluation.py --cutoff 2021-12-31

Outputs:
    experiments/outputs/holdout/holdout_YYYYMMDD_HHMMSS.json
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

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import (
    compute_cohens_d_with_ci,
    compute_detection_metrics,
)
from experiments.regime_comparison import (
    HPO_CONFIGS,
    CLASSICAL_CONFIGS,
    compute_adaptive_threshold,
    apply_persistence_filter,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)

# Holdout crises (must also exist in ALL_CRISES)
HOLDOUT_CRISES = {
    '2022_rates': ALL_CRISES['2022_rates'],
    '2023_svb': ALL_CRISES['2023_svb'],
    '2024_carry': ALL_CRISES['2024_carry'],
}


def run_holdout_evaluation(
    cutoff_date: str = '2021-12-31',
    window_size: int = 10,
    n_bootstrap: int = 10000,
):
    """Run blind holdout stress test.

    Args:
        cutoff_date: Last date for training data (inclusive).
        window_size: Crisis window extension in trading days.
        n_bootstrap: Bootstrap resamples for CIs.

    Returns:
        dict with holdout results for all methods and crises.
    """
    cutoff = pd.Timestamp(cutoff_date)
    logger.info("=" * 70)
    logger.info(f"BLIND HOLDOUT STRESS TEST (cutoff: {cutoff.date()})")
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

    # ---- Split at cutoff ----
    cutoff_idx = int(np.searchsorted(dates_enriched, cutoff))
    logger.info(f"  Cutoff index: {cutoff_idx} / {len(dates_enriched)} "
                f"({dates_enriched[cutoff_idx - 1].date()})")

    # ---- Evaluate each method ----
    results = {}

    logger.info(f"\n[2] Evaluating {len(HPO_CONFIGS) + len(CLASSICAL_CONFIGS)} methods "
                f"on {len(HOLDOUT_CRISES)} holdout crises...")

    # --- QCML detectors ---
    for method_name, config in HPO_CONFIGS.items():
        logger.info(f"\n  {method_name}...")
        params = {**config['params'], 'causal_fit_length': cutoff_idx}
        det = config['class'](**params)

        try:
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)
        except Exception as e:
            logger.warning(f"  {method_name} failed: {e}")
            results[method_name] = {'error': str(e)}
            continue

        results[method_name] = _evaluate_holdout_crises(
            scores, dates_enriched, cutoff_idx, window_size, n_bootstrap,
        )

    # --- Classical baselines ---
    for method_name, config in CLASSICAL_CONFIGS.items():
        logger.info(f"\n  {method_name}...")
        det = config['class'](**config['params'])

        try:
            # Fit on pre-cutoff data
            det.fit(X_enriched[:cutoff_idx])

            # Special handling for VIX detector
            if hasattr(det, 'set_vix') and vix_enriched is not None:
                det.set_vix(vix_enriched)

            scores = det.compute_regime_scores(X_enriched)
        except Exception as e:
            logger.warning(f"  {method_name} failed: {e}")
            results[method_name] = {'error': str(e)}
            continue

        results[method_name] = _evaluate_holdout_crises(
            scores, dates_enriched, cutoff_idx, window_size, n_bootstrap,
        )

    # ---- Aggregate statistics ----
    aggregate = _compute_aggregate(results)

    output = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'cutoff_date': cutoff_date,
            'window_size': window_size,
            'n_bootstrap': n_bootstrap,
            'n_methods': len(results),
            'holdout_crises': list(HOLDOUT_CRISES.keys()),
        },
        'per_method': results,
        'aggregate': aggregate,
    }

    # Save
    out_dir = ROOT / 'experiments' / 'outputs' / 'holdout'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'holdout_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary
    _print_summary(results, aggregate)

    return output


def _evaluate_holdout_crises(
    scores, dates, cutoff_idx, window_size, n_bootstrap,
):
    """Evaluate a single method's scores on all holdout crises.

    Returns:
        dict keyed by crisis_key with d, CI, detection metrics.
    """
    crisis_results = {}

    for crisis_key, crisis_info in HOLDOUT_CRISES.items():
        crisis_start = pd.Timestamp(crisis_info['start'])
        crisis_end = pd.Timestamp(crisis_info['end'])

        cs = crisis_start - pd.Timedelta(days=window_size)
        ce = crisis_end + pd.Timedelta(days=window_size)

        crisis_mask = (dates >= cs) & (dates <= ce)
        # Normal = post-cutoff but not crisis
        post_cutoff = np.arange(len(dates)) >= cutoff_idx
        normal_mask = post_cutoff & ~crisis_mask

        crisis_scores = scores[crisis_mask]
        normal_scores = scores[normal_mask]

        # Cohen's d
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_scores, normal_scores, n_bootstrap=n_bootstrap,
        )

        # Threshold from pre-cutoff data only
        pre_cutoff_scores = scores[:cutoff_idx]
        valid_pre = pre_cutoff_scores[np.isfinite(pre_cutoff_scores)]
        if len(valid_pre) > 0:
            threshold = np.percentile(valid_pre, 95)
        else:
            threshold = np.nan

        # Detection metrics
        if np.isfinite(threshold):
            metrics = compute_detection_metrics(
                scores, threshold, crisis_mask,
            )
        else:
            metrics = {
                'detection_delay': np.nan,
                'false_alarm_rate': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'F1': np.nan,
            }

        crisis_results[crisis_key] = {
            'd': float(d) if np.isfinite(d) else None,
            'ci_lo': float(ci_lo) if np.isfinite(ci_lo) else None,
            'ci_hi': float(ci_hi) if np.isfinite(ci_hi) else None,
            'detection_delay': _safe_float(metrics['detection_delay']),
            'false_alarm_rate': _safe_float(metrics['false_alarm_rate']),
            'precision': _safe_float(metrics['precision']),
            'recall': _safe_float(metrics['recall']),
            'F1': _safe_float(metrics['F1']),
            'label': crisis_info['label'],
        }

    return crisis_results


def _compute_aggregate(results):
    """Compute aggregate statistics across methods."""
    aggregate = {}
    for method, crises in results.items():
        if 'error' in crises:
            continue
        ds = [crises[c].get('d') for c in HOLDOUT_CRISES if c in crises]
        ds = [d for d in ds if d is not None]
        aggregate[method] = {
            'mean_d': float(np.mean(ds)) if ds else None,
            'median_d': float(np.median(ds)) if ds else None,
            'n_detected': sum(1 for d in ds if d > 0.5),
            'n_total': len(HOLDOUT_CRISES),
        }
    return aggregate


def _print_summary(results, aggregate):
    """Print formatted summary."""
    logger.info("\n" + "=" * 70)
    logger.info("HOLDOUT EVALUATION SUMMARY")
    logger.info("=" * 70)

    # Sort by mean d
    sorted_methods = sorted(
        aggregate.keys(),
        key=lambda m: aggregate[m].get('mean_d') or 0,
        reverse=True,
    )

    logger.info(f"\n{'Method':<30} {'Mean d':>8} {'2022':>6} {'SVB':>6} {'Carry':>6}")
    logger.info("-" * 60)
    for m in sorted_methods[:20]:  # top 20
        agg = aggregate[m]
        method_data = results[m]
        d_2022 = method_data.get('2022_rates', {}).get('d')
        d_svb = method_data.get('2023_svb', {}).get('d')
        d_carry = method_data.get('2024_carry', {}).get('d')

        mean_str = f"{agg['mean_d']:.3f}" if agg['mean_d'] else "N/A"
        d22 = f"{d_2022:.2f}" if d_2022 else "N/A"
        dsvb = f"{d_svb:.2f}" if d_svb else "N/A"
        dcar = f"{d_carry:.2f}" if d_carry else "N/A"
        logger.info(f"{m:<30} {mean_str:>8} {d22:>6} {dsvb:>6} {dcar:>6}")


def _safe_float(x):
    """Convert to float, handling NaN/inf."""
    if x is None:
        return None
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return None
    return x


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
    parser = argparse.ArgumentParser(description='Blind holdout stress test')
    parser.add_argument('--cutoff', type=str, default='2021-12-31',
                        help='Cutoff date for training data')
    parser.add_argument('--bootstrap', type=int, default=10000,
                        help='Bootstrap resamples')
    args = parser.parse_args()

    run_holdout_evaluation(
        cutoff_date=args.cutoff,
        n_bootstrap=args.bootstrap,
    )
