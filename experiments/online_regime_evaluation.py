"""
Online Regime Detection Evaluation

Evaluates online detection models that produce P(crisis) at each timestep
using only causally available data. Proper online detection metrics:

- AUC-ROC / AUC-PR on continuous P(crisis) output
- Detection rate at fixed false alarm rates (0.5, 1.0, 2.0 alarms/year)
- Detection delay at fixed FAR levels
- DET curve (detection rate vs FAR)
- Comparison against online RF, CUSUM, z>2.0 baseline

Usage:
    python experiments/online_regime_evaluation.py
    python experiments/online_regime_evaluation.py --quick

Outputs:
    experiments/outputs/regime_detection/online_detection_YYYYMMDD_HHMMSS.json
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
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.observables import BaseRegimeDetector
from qcml_geometry.online_detection import (
    OnlineGeometricFeatureComputer,
    ExpandingPercentileDetector,
    OnlineBayesianDetector,
    OnlineHMMDetector,
    OnlineLogisticDetector,
    OnlineEnsembleDetector,
)

from experiments.data_loader import (
    fetch_polygon_data, create_feature_matrix, ALL_CRISES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Online Detection Runner
# =============================================================================

def run_online_detection(X_enriched, dates, crisis_labels):
    """Run all online detection models point-by-point.

    Args:
        X_enriched: Enriched feature matrix (T, d).
        dates: DatetimeIndex aligned with X_enriched.
        crisis_labels: Binary array (T,) where 1 = crisis.

    Returns:
        Dict mapping model name to P(crisis) time series (T,).
    """
    T = len(X_enriched)

    # Initialize feature computer
    feat_computer = OnlineGeometricFeatureComputer(
        hilbert_dim=8, n_pca_components=15,
        operator_method='pca_inspired',
        seed=42,
    )

    # Shared detectors: ensemble wraps these, individual outputs extracted
    percentile = ExpandingPercentileDetector(min_history=60)
    bayesian = OnlineBayesianDetector(
        transition_prob=0.02, persistence=0.95,
    )
    hmm = OnlineHMMDetector(seed=42)
    logistic = OnlineLogisticDetector()

    # Ensemble wraps bayesian + hmm + percentile (shared instances)
    ensemble = OnlineEnsembleDetector(
        detectors=[bayesian, hmm, percentile],
        weights=[0.4, 0.4, 0.2],
    )

    # Names for all outputs
    all_names = [bayesian.name, hmm.name, percentile.name, logistic.name, ensemble.name]
    p_crisis = {name: np.full(T, np.nan) for name in all_names}

    logger.info(f"  Running online detection for {T} timesteps...")
    log_interval = max(T // 10, 1)

    for t in range(T):
        if (t + 1) % log_interval == 0:
            logger.info(f"    Step {t+1}/{T}")

        # Compute geometric features for this timestep
        features = feat_computer.update(X_enriched[t])

        # Feed label to supervised detector (causal: label at t is known at t+1)
        if t > 0:
            logistic.add_label(crisis_labels[t - 1])

        # Update logistic separately (not in ensemble)
        p_crisis[logistic.name][t] = logistic.update(features)

        # Update ensemble (internally updates bayesian, hmm, percentile)
        p = ensemble.update(features)
        p_crisis[ensemble.name][t] = p

        # Extract individual outputs from shared detectors
        for det_name, det_p in ensemble.last_individual.items():
            p_crisis[det_name][t] = det_p

    return p_crisis


# =============================================================================
# Online Baselines (for comparison)
# =============================================================================

def run_online_cusum_baseline(X_enriched, burn_in=60):
    """Online CUSUM on first feature as P(crisis) baseline.

    Returns P(crisis) as sigmoid of CUSUM statistic.
    """
    T = X_enriched.shape[0]
    x = X_enriched[:, 0]

    # Expanding mean/std for first feature
    p_crisis = np.full(T, np.nan)
    s_high = 0.0
    s_low = 0.0
    k = 0.5

    for t in range(T):
        if t < burn_in:
            continue

        mu = np.mean(x[:t])
        sigma = np.std(x[:t], ddof=1)
        if sigma < 1e-12:
            continue

        z = (x[t] - mu) / sigma
        s_high = max(0.0, s_high + z - k)
        s_low = max(0.0, s_low - z - k)
        cusum = max(s_high, s_low)

        # Convert to probability via sigmoid
        p_crisis[t] = 1.0 / (1.0 + np.exp(-0.5 * (cusum - 4.0)))

    return p_crisis


def run_online_volz_baseline(X_enriched, vol_window=20, min_expanding=60):
    """Online rolling volatility z-score as P(crisis) baseline.

    Returns P(crisis) as CDF of rolling vol z-score using the mean
    absolute value across all enriched features as a proxy for market stress.
    """
    T = X_enriched.shape[0]
    # Use mean absolute value across all features as stress proxy
    x = np.mean(np.abs(X_enriched), axis=1)

    vol = pd.Series(x).rolling(vol_window, min_periods=1).std().values
    p_crisis = np.full(T, np.nan)

    for t in range(min_expanding, T):
        past_vol = vol[1:t]  # skip first (might be 0)
        past_vol = past_vol[~np.isnan(past_vol)]
        if len(past_vol) < 10:
            continue
        mu = np.mean(past_vol)
        sigma = np.std(past_vol, ddof=1)
        if sigma > 1e-12:
            z = abs((vol[t] - mu) / sigma)
            # Sigmoid mapping to [0, 1]
            p_crisis[t] = 1.0 / (1.0 + np.exp(-1.0 * (z - 2.0)))

    return p_crisis


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_online_metrics(p_crisis, crisis_labels, dates, far_levels=None):
    """Compute online detection metrics.

    Args:
        p_crisis: P(crisis) time series (T,).
        crisis_labels: Binary labels (T,).
        dates: DatetimeIndex (T,).
        far_levels: FAR levels to evaluate (alarms/year).

    Returns:
        Dict of metrics.
    """
    if far_levels is None:
        far_levels = [0.5, 1.0, 2.0, 5.0]

    # Remove NaN entries
    valid = ~np.isnan(p_crisis) & ~np.isnan(crisis_labels)
    p_valid = p_crisis[valid]
    y_valid = crisis_labels[valid]
    dates_valid = dates[valid]

    if len(p_valid) < 10 or len(np.unique(y_valid)) < 2:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'far_analysis': {}}

    # AUC-ROC and AUC-PR
    auc_roc = float(roc_auc_score(y_valid, p_valid))
    auc_pr = float(average_precision_score(y_valid, p_valid))

    # Detection at various FAR levels
    total_years = (dates_valid[-1] - dates_valid[0]).days / 365.25
    normal_days = int(np.sum(y_valid == 0))
    normal_years = normal_days / 252

    far_analysis = {}
    for target_far in far_levels:
        result = _evaluate_at_far(
            p_valid, y_valid, dates_valid, target_far, normal_years
        )
        far_analysis[f'far_{target_far}'] = result

    # Per-crisis detection analysis
    per_crisis = _per_crisis_analysis(p_crisis, crisis_labels, dates)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'n_valid': int(np.sum(valid)),
        'crisis_rate': float(np.mean(y_valid)),
        'total_years': float(total_years),
        'far_analysis': far_analysis,
        'per_crisis': per_crisis,
    }


def _evaluate_at_far(p_crisis, y, dates, target_far, normal_years):
    """Find threshold that achieves target FAR, then compute detection metrics.

    Args:
        p_crisis: P(crisis) array (valid entries only).
        y: Binary labels.
        dates: Dates.
        target_far: Target false alarm rate (episodes/year).
        normal_years: Years of normal-period data.

    Returns:
        Dict with threshold, detection_rate, mean_delay.
    """
    normal_mask = y == 0

    # Binary search for threshold achieving target FAR
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        alarm = p_crisis > mid
        fa_episodes = _count_episodes(alarm & normal_mask)
        achieved_far = fa_episodes / max(normal_years, 0.01)

        if achieved_far > target_far:
            lo = mid
        else:
            hi = mid

    threshold = hi
    alarm = p_crisis > threshold

    # Detection rate: fraction of crises that have at least one alarm
    crisis_indices = np.where(y == 1)[0]
    if len(crisis_indices) == 0:
        return {
            'threshold': float(threshold),
            'detection_rate': 0.0,
            'achieved_far': 0.0,
            'mean_delay': np.nan,
        }

    # Find crisis episodes
    crisis_episodes = _find_crisis_episodes(y)
    n_detected = 0
    delays = []

    for start, end in crisis_episodes:
        alarm_in_crisis = np.where(alarm[start:end + 1])[0]
        if len(alarm_in_crisis) > 0:
            n_detected += 1
            delays.append(int(alarm_in_crisis[0]))

    detection_rate = n_detected / len(crisis_episodes) if crisis_episodes else 0.0
    mean_delay = float(np.mean(delays)) if delays else np.nan

    # Achieved FAR
    fa_episodes = _count_episodes(alarm & normal_mask)
    achieved_far = fa_episodes / max(normal_years, 0.01)

    return {
        'threshold': float(threshold),
        'detection_rate': float(detection_rate),
        'n_detected': n_detected,
        'n_crises': len(crisis_episodes),
        'achieved_far': float(achieved_far),
        'mean_delay_days': float(mean_delay) if not np.isnan(mean_delay) else None,
    }


def _per_crisis_analysis(p_crisis, crisis_labels, dates):
    """Analyze detection for each individual crisis window."""
    results = {}
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])

        crisis_mask = (dates >= cs) & (dates <= ce)
        if not np.any(crisis_mask):
            continue

        crisis_p = p_crisis[crisis_mask]
        valid_crisis_p = crisis_p[~np.isnan(crisis_p)]

        if len(valid_crisis_p) == 0:
            continue

        results[ck] = {
            'mean_p_crisis': float(np.mean(valid_crisis_p)),
            'max_p_crisis': float(np.max(valid_crisis_p)),
            'pct_above_50': float(np.mean(valid_crisis_p > 0.5)),
            'n_days': int(np.sum(crisis_mask)),
        }

    return results


def _count_episodes(alarm_mask, gap=5):
    """Count distinct alarm episodes."""
    indices = np.where(alarm_mask)[0]
    if len(indices) == 0:
        return 0
    episodes = 1
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > gap:
            episodes += 1
    return episodes


def _find_crisis_episodes(y):
    """Find contiguous crisis episodes in binary labels."""
    episodes = []
    in_crisis = False
    start = 0

    for i in range(len(y)):
        if y[i] == 1 and not in_crisis:
            start = i
            in_crisis = True
        elif y[i] == 0 and in_crisis:
            episodes.append((start, i - 1))
            in_crisis = False

    if in_crisis:
        episodes.append((start, len(y) - 1))

    return episodes


# =============================================================================
# DET Curve
# =============================================================================

def compute_det_curve(p_crisis, crisis_labels, dates, n_thresholds=200):
    """Compute DET curve (detection rate vs false alarm rate).

    Args:
        p_crisis: P(crisis) time series.
        dates: DatetimeIndex.
        crisis_labels: Binary labels.
        n_thresholds: Number of threshold levels to evaluate.

    Returns:
        Dict with 'far' and 'detection_rate' arrays.
    """
    valid = ~np.isnan(p_crisis) & ~np.isnan(crisis_labels)
    p_valid = p_crisis[valid]
    y_valid = crisis_labels[valid]
    dates_valid = dates[valid]

    if len(p_valid) < 10:
        return {'far': [], 'detection_rate': []}

    normal_mask = y_valid == 0
    normal_years = np.sum(normal_mask) / 252

    crisis_episodes = _find_crisis_episodes(y_valid)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    fars = []
    det_rates = []

    for thresh in thresholds:
        alarm = p_valid > thresh

        # FAR
        fa_episodes = _count_episodes(alarm & normal_mask)
        far = fa_episodes / max(normal_years, 0.01)

        # Detection rate
        n_detected = 0
        for start, end in crisis_episodes:
            if np.any(alarm[start:end + 1]):
                n_detected += 1

        det_rate = n_detected / len(crisis_episodes) if crisis_episodes else 0.0

        fars.append(float(far))
        det_rates.append(float(det_rate))

    return {
        'thresholds': [float(t) for t in thresholds],
        'far': fars,
        'detection_rate': det_rates,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_evaluation(quick=False):
    """Run full online regime detection evaluation.

    Args:
        quick: Use shorter date range and fewer crises.

    Returns:
        Full results dict.
    """
    logger.info("=" * 70)
    logger.info("ONLINE REGIME DETECTION EVALUATION")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1/5] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(prices_df)

    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Enriched features: {X_enriched.shape}")

    # Build crisis labels
    crisis_labels = np.zeros(len(dates_enriched))
    window_ext = 10
    for ci in ALL_CRISES.values():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_ext)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_ext)
        mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        crisis_labels[mask] = 1.0

    logger.info(
        f"  Crisis days: {int(np.sum(crisis_labels))} / {len(crisis_labels)} "
        f"({100 * np.mean(crisis_labels):.1f}%)"
    )

    # ---- Run online detectors ----
    logger.info("\n[2/5] Running online geometric detectors...")
    geo_p_crisis = run_online_detection(X_enriched, dates_enriched, crisis_labels)

    # ---- Run baselines ----
    logger.info("\n[3/5] Running online baselines...")
    cusum_p = run_online_cusum_baseline(X_enriched)
    volz_p = run_online_volz_baseline(X_enriched)

    all_models = {}
    all_models.update(geo_p_crisis)
    all_models['Online CUSUM'] = cusum_p
    all_models['Online Vol-Z'] = volz_p

    # ---- Evaluate ----
    logger.info("\n[4/5] Computing evaluation metrics...")
    results = {}
    for model_name, p_crisis in all_models.items():
        logger.info(f"  {model_name}...")

        metrics = compute_online_metrics(
            p_crisis, crisis_labels, dates_enriched,
        )
        det_curve = compute_det_curve(
            p_crisis, crisis_labels, dates_enriched,
        )

        results[model_name] = {
            'metrics': metrics,
            'det_curve': det_curve,
        }

        if not np.isnan(metrics.get('auc_roc', np.nan)):
            logger.info(
                f"    AUC-ROC={metrics['auc_roc']:.3f}, "
                f"AUC-PR={metrics['auc_pr']:.3f}"
            )
            for far_key, far_res in metrics.get('far_analysis', {}).items():
                det_rate = far_res.get('detection_rate', 0)
                delay = far_res.get('mean_delay_days')
                delay_str = f"{delay:.0f}d" if delay is not None else "N/A"
                logger.info(
                    f"    {far_key}: det_rate={det_rate:.0%}, "
                    f"delay={delay_str}, "
                    f"threshold={far_res['threshold']:.3f}"
                )

    # ---- Summary ----
    logger.info("\n[5/5] Summary")
    logger.info("=" * 70)
    logger.info(f"{'Model':25s} {'AUC-ROC':>8s} {'AUC-PR':>8s} "
                f"{'Det@1/yr':>9s} {'Delay':>7s}")
    logger.info("-" * 70)

    for model_name, r in sorted(
        results.items(),
        key=lambda x: x[1]['metrics'].get('auc_roc', 0) or 0,
        reverse=True,
    ):
        m = r['metrics']
        auc_roc = m.get('auc_roc', np.nan)
        auc_pr = m.get('auc_pr', np.nan)

        far1 = m.get('far_analysis', {}).get('far_1.0', {})
        det_rate = far1.get('detection_rate', np.nan)
        delay = far1.get('mean_delay_days')

        delay_str = f"{delay:.0f}d" if delay is not None else "N/A"
        logger.info(
            f"  {model_name:25s} {auc_roc:8.3f} {auc_pr:8.3f} "
            f"{det_rate:8.0%} {delay_str:>7s}"
        )

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick': quick,
            'symbols': symbols,
            'date_range': ['2005-01-01', '2024-12-31'],
            'n_timesteps': len(X_enriched),
            'n_crises': len(ALL_CRISES),
            'crisis_fraction': float(np.mean(crisis_labels)),
            'geo_config': {
                'hilbert_dim': 8,
                'n_pca_components': 15,
                'operator_method': 'pca_inspired',
                'refit_interval': 21,
                'min_history': 126,
            },
        },
        'results': _make_serializable(results),
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'online_detection_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def _make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return [_make_serializable(v) for v in obj.tolist()]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Online regime detection evaluation'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (still runs all models)')
    args = parser.parse_args()

    run_evaluation(quick=args.quick)


if __name__ == '__main__':
    main()
