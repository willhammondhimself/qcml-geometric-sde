"""
Walk-forward detection benchmark (Review Item 2).

Expanding-window fit (always starting 2005), 1-year evaluation, stepped annually.
For each eval year:
  - Fit QCML detectors on ONLY the training window (2005 to eval_year-1)
  - Compute scores on the eval year
  - If crisis falls in eval year: compute detection delay and false alarm rate

Usage:
    python experiments/walk_forward_evaluation.py
    python experiments/walk_forward_evaluation.py --quick
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

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import (
    fetch_polygon_data, create_feature_matrix, ALL_CRISES,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    BOCPDDetector,
    RandomForestRegimeDetector,
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


def build_unsupervised_detectors():
    """Build one instance of each unsupervised method for walk-forward fitting."""
    common_qcml = dict(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
        rolling_window=20, seed=42,
    )
    return [
        ('Berry Phase Rate', lambda: BerryPhaseRateDetector(**common_qcml)),
        ('QFI Determinant', lambda: QFIDeterminantDetector(**common_qcml)),
        ('Multi-Lag Fidelity', lambda: MultiLagFidelityDetector(**common_qcml)),
        ('Rolling Vol Z', lambda: RollingVolatilityDetector(vol_window=20, min_expanding=60)),
        ('BOCPD', lambda: BOCPDDetector(hazard_rate=250.0)),
    ]


def count_alarm_episodes(alarm_mask, gap_days=5):
    """Group consecutive or near-consecutive alarms into episodes.

    Args:
        alarm_mask: Boolean array where True = alarm.
        gap_days: Max gap between alarms to count as same episode.

    Returns:
        Number of distinct alarm episodes.
    """
    alarm_indices = np.where(alarm_mask)[0]
    if len(alarm_indices) == 0:
        return 0
    episodes = 1
    for i in range(1, len(alarm_indices)):
        if alarm_indices[i] - alarm_indices[i - 1] > gap_days:
            episodes += 1
    return episodes


def find_crises_in_period(start_date, end_date):
    """Return crisis keys whose windows overlap [start_date, end_date]."""
    matching = []
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        if cs <= end_date and ce >= start_date:
            matching.append(ck)
    return matching


def find_training_crises(train_end):
    """Return crisis keys whose windows END before train_end (for RF labels)."""
    matching = []
    for ck, ci in ALL_CRISES.items():
        ce = pd.Timestamp(ci['end'])
        if ce <= train_end:
            matching.append(ck)
    return matching


def run_walk_forward(quick=False):
    """Run walk-forward evaluation with expanding window.

    Expanding windows:
        Train 2005-2009, Eval 2010
        Train 2005-2010, Eval 2011
        ...
        Train 2005-2022, Eval 2023

    Args:
        quick: If True, only use 3 windows.
    """
    logger.info("=" * 70)
    logger.info("Walk-Forward Detection Benchmark (Expanding Window)")
    logger.info("=" * 70)

    # Fetch full data range
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_full, dates_full = create_feature_matrix(prices_df)
    logger.info(f"  Full feature matrix: {X_full.shape}")

    # Define walk-forward windows (expanding from 2005)
    if quick:
        window_starts = [2010, 2015, 2020]
    else:
        window_starts = list(range(2010, 2024))

    unsup_factories = build_unsupervised_detectors()
    results = {}

    for eval_start_year in window_starts:
        # Expanding window: always start from 2005
        train_start = pd.Timestamp('2005-01-01')
        train_end = pd.Timestamp(f'{eval_start_year - 1}-12-31')
        eval_start = pd.Timestamp(f'{eval_start_year}-01-01')
        eval_end = pd.Timestamp(f'{eval_start_year}-12-31')

        logger.info(f"\n[Window] Train: {train_start.date()} to {train_end.date()}, "
                     f"Eval: {eval_start.date()} to {eval_end.date()}")

        # Slice data
        train_mask = (dates_full >= train_start) & (dates_full <= train_end)
        eval_mask = (dates_full >= eval_start) & (dates_full <= eval_end)
        full_mask = (dates_full >= train_start) & (dates_full <= eval_end)

        X_train = X_full[train_mask]
        X_eval = X_full[eval_mask]
        X_window = X_full[full_mask]
        dates_eval = dates_full[eval_mask]
        dates_window = dates_full[full_mask]
        dates_train = dates_full[train_mask]

        if len(X_train) < 100 or len(X_eval) < 50:
            logger.info(f"  Skipping (insufficient data: train={len(X_train)}, eval={len(X_eval)})")
            continue

        # Build enriched features
        X_train_enriched = BaseRegimeDetector.build_enriched_features(X_train, lookback=20)
        X_window_enriched = BaseRegimeDetector.build_enriched_features(X_window, lookback=20)
        dates_window_enriched = dates_window[19:]

        # Identify where eval period starts in the window-enriched array
        eval_start_idx = np.searchsorted(dates_window_enriched, eval_start)

        # Find crises in eval period
        eval_crises = find_crises_in_period(eval_start, eval_end)
        logger.info(f"  Crises in eval: {eval_crises or 'none'}")

        # --- Unsupervised methods ---
        for method_name, factory in unsup_factories:
            det = factory()
            det.fit(X_train_enriched)
            scores = det.compute_regime_scores(X_window_enriched)
            eval_scores = scores[eval_start_idx:]
            eval_dates = dates_window_enriched[eval_start_idx:]

            for ck in eval_crises:
                _record_crisis_result(results, method_name, ck, eval_start_year,
                                      eval_scores, eval_dates)

        # --- RF (supervised, expanding-window training) ---
        # Find crises that ended in the training period for labels
        train_crises = find_training_crises(train_end)
        if len(train_crises) >= 2:
            # Build labels on pre-enrichment data (RF enriches internally)
            y_train = np.zeros(len(X_train))
            for ck in train_crises:
                ci = ALL_CRISES[ck]
                cs = pd.Timestamp(ci['start'])
                ce = pd.Timestamp(ci['end'])
                mask = (dates_train >= cs) & (dates_train <= ce)
                y_train[mask] = 1.0

            rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
            rf.fit_with_labels(X_train, y_train)

            rf_scores_full = rf.compute_regime_scores(X_window)
            # RF scores have lookback-1 NaN pad at front, align to enriched timeline
            rf_window_scores = rf_scores_full[19:] if len(rf_scores_full) > len(dates_window_enriched) else rf_scores_full
            if len(rf_window_scores) >= len(dates_window_enriched):
                rf_window_scores = rf_window_scores[:len(dates_window_enriched)]
            rf_eval = rf_window_scores[eval_start_idx:]
            eval_dates_rf = dates_window_enriched[eval_start_idx:]

            for ck in eval_crises:
                _record_crisis_result(results, 'Random Forest', ck, eval_start_year,
                                      rf_eval, eval_dates_rf)
        else:
            logger.info(f"  RF skipped (only {len(train_crises)} training crises)")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD SUMMARY")
    logger.info("=" * 70)

    method_summary = _compute_summary(results)

    for mname in sorted(method_summary.keys()):
        s = method_summary[mname]
        logger.info(f"  {mname:25s}: median_d={s['median_d']:.2f}, "
                    f"median_delay={s['median_delay']:.1f}d, FAR={s['median_far']:.1f}/yr, "
                    f"detected={s['n_detected']}/{s['n_total']} ({s['detection_rate']:.0%})")

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'window_type': 'expanding',
            'train_start': '2005-01-01',
            'threshold': 2.0,
            'alarm_episode_gap': 5,
            'quick': quick,
        },
        'results': results,
        'method_summary': {
            mname: s for mname, s in method_summary.items()
        },
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'walk_forward'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'walk_forward_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def _record_crisis_result(results, method_name, ck, eval_start_year,
                          eval_scores, eval_dates):
    """Record detection metrics for one method × crisis × window."""
    ci = ALL_CRISES[ck]
    cs = pd.Timestamp(ci['start'])
    ce = pd.Timestamp(ci['end'])

    crisis_mask = (eval_dates >= cs) & (eval_dates <= ce)
    normal_mask = ~crisis_mask

    crisis_scores = eval_scores[crisis_mask]
    normal_scores = eval_scores[normal_mask]

    # Cohen's d
    d, ci_lo, ci_hi = compute_cohens_d_with_ci(
        crisis_scores, normal_scores, n_bootstrap=5000
    )

    # Detection delay (days from crisis start to first z > 2.0)
    threshold = 2.0
    alarm_mask = eval_scores > threshold
    crisis_start_idx = np.searchsorted(eval_dates, cs)
    crisis_end_idx = np.searchsorted(eval_dates, ce)

    alarms_in_crisis = np.where(
        alarm_mask[crisis_start_idx:crisis_end_idx]
    )[0]
    detection_delay = int(alarms_in_crisis[0]) if len(alarms_in_crisis) > 0 else None

    # False alarm rate: alarm episodes per year outside crisis
    normal_alarm_mask = alarm_mask & normal_mask
    normal_days = np.sum(normal_mask)
    episodes = count_alarm_episodes(normal_alarm_mask, gap_days=5)
    years = max(normal_days / 252, 0.01)
    far = float(episodes / years)

    key = f"{method_name}|{ck}|{eval_start_year}"
    results[key] = {
        'method': method_name,
        'crisis': ck,
        'eval_year': eval_start_year,
        'd': float(d) if not np.isnan(d) else None,
        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        'detection_delay_days': detection_delay,
        'false_alarm_rate_per_year': round(far, 3),
        'crisis_detected': detection_delay is not None,
    }

    logger.info(f"    {method_name:25s} on {ck:20s}: "
               f"d={d:.2f}, delay={detection_delay}, FAR={far:.1f}/yr")


def _compute_summary(results):
    """Aggregate per-method: median d, median delay, median FAR, detection rate."""
    method_summary = {}
    for key, r in results.items():
        mname = r['method']
        if mname not in method_summary:
            method_summary[mname] = {'ds': [], 'delays': [], 'fars': [],
                                      'detected': 0, 'total': 0}

        method_summary[mname]['total'] += 1
        if r['d'] is not None:
            method_summary[mname]['ds'].append(r['d'])
        if r['detection_delay_days'] is not None:
            method_summary[mname]['delays'].append(r['detection_delay_days'])
            method_summary[mname]['detected'] += 1
        method_summary[mname]['fars'].append(r['false_alarm_rate_per_year'])

    summary = {}
    for mname, s in method_summary.items():
        median_d = float(np.median(s['ds'])) if s['ds'] else float('nan')
        median_delay = float(np.median(s['delays'])) if s['delays'] else float('nan')
        median_far = float(np.median(s['fars'])) if s['fars'] else float('nan')
        detect_rate = s['detected'] / max(s['total'], 1)
        summary[mname] = {
            'median_d': median_d,
            'median_delay': median_delay,
            'median_far': median_far,
            'detection_rate': detect_rate,
            'n_detected': s['detected'],
            'n_total': s['total'],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Walk-forward detection benchmark')
    parser.add_argument('--quick', action='store_true', help='Only 3 windows')
    args = parser.parse_args()
    run_walk_forward(quick=args.quick)


if __name__ == '__main__':
    main()
