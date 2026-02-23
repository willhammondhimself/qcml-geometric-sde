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
from qcml_geometry.adaptive_threshold import (
    RollingQuantileThreshold,
    ScoreVelocityThreshold,
    CombinedAdaptiveThreshold,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)


def build_unsupervised_detectors():
    """Build one instance of each unsupervised method for walk-forward fitting.

    QCML detector configuration addresses three root causes of walk-forward
    underperformance identified in the root-cause analysis:

    Root Cause 1 + 4 (threshold calibration / baseline absorption):
        Fixed by the detectors themselves — fit() now stores training baseline
        (mu, sigma); compute_regime_scores() uses that fixed baseline instead of
        an expanding mean that absorbs crisis signals.

    Root Cause 2 (frozen geometry during eval):
        expanding_refit_interval=21 builds monthly snapshots during training so
        the geometry is as fresh as possible at the start of each eval year.

    Root Cause 3 (degenerate PCA):
        n_pca_components=8 with 13 available features gives genuine
        dimensionality reduction (captures ~85-90% variance) and produces
        more stable operator scales across walk-forward windows.
    """
    common_qcml = dict(
        hilbert_dim=8,
        n_pca_components=10,         # C4: bumped from 8 to absorb QQQ's additional variance
        operator_method='pca_inspired',
        rolling_window=10,           # C2: faster window; multi-scale scoring (A) preserves 20/40
        seed=42,
        expanding_refit_interval=21,  # Root Cause 2 fix: monthly refit snapshots
    )
    return [
        ('Berry Phase Rate', lambda: BerryPhaseRateDetector(**common_qcml)),
        ('QFI Determinant', lambda: QFIDeterminantDetector(**common_qcml)),
        ('Multi-Lag Fidelity', lambda: MultiLagFidelityDetector(**common_qcml)),
        ('Rolling Vol Z', lambda: RollingVolatilityDetector(vol_window=20, min_expanding=60)),
        ('BOCPD', lambda: BOCPDDetector(hazard_rate=100.0)),  # C3: h=1/100 → crisis_peak off 0
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


def calibrate_threshold(scores, dates, crisis_keys, target_far=1.0, gap_days=5,
                        method_name=None):
    """Calibrate threshold tau on training data to achieve target FAR.

    Given training-period scores, find threshold tau such that the false alarm
    rate (alarm episodes per year outside known crisis windows) <= target_far.

    The search range adapts to the method's score scale:
    - Z-score methods (QCML, Vol Z): search [0.5, 5.0]
    - Probability methods (BOCPD, HMM, RF): search [0.05, 0.95]
    - Generic fallback: uses score quantiles [50th, 99th]

    Args:
        scores: 1-D array of scores from training period.
        dates: Corresponding dates.
        crisis_keys: List of crisis keys active in training period.
        target_far: Target false alarm rate (episodes/year). Default 1.0.
        gap_days: Gap for episode counting.
        method_name: Name of method (for scale-aware calibration).

    Returns:
        float: Calibrated threshold tau.
    """
    # Build normal mask (exclude known crisis windows)
    normal_mask = np.ones(len(scores), dtype=bool)
    for ck in crisis_keys:
        if ck not in ALL_CRISES:
            continue
        ci = ALL_CRISES[ck]
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        crisis_m = (dates >= cs) & (dates <= ce)
        normal_mask &= ~crisis_m

    normal_days = np.sum(normal_mask)
    years = max(normal_days / 252, 0.1)

    # Determine search range based on method type
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        return 2.0

    prob_methods = {'BOCPD', 'HMM', 'Isolation Forest', 'Random Forest'}
    if method_name and method_name in prob_methods:
        lo = max(0.05, np.percentile(valid_scores, 50))
        hi = min(0.95, np.percentile(valid_scores, 99))
    else:
        lo, hi = 0.5, 5.0

    if lo >= hi:
        lo, hi = np.percentile(valid_scores, 50), np.percentile(valid_scores, 99)
    if lo >= hi:
        return float(np.percentile(valid_scores, 95))

    # Binary search for threshold
    for _ in range(50):
        mid = (lo + hi) / 2
        alarm_mask = (scores > mid) & normal_mask
        episodes = count_alarm_episodes(alarm_mask, gap_days=gap_days)
        far = episodes / years
        if far > target_far:
            lo = mid
        else:
            hi = mid

    return round(hi, 4)


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


def run_walk_forward(quick=False, far_target=None, adaptive=False):
    """Run walk-forward evaluation with expanding window.

    Expanding windows:
        Train 2005-2009, Eval 2010
        Train 2005-2010, Eval 2011
        ...
        Train 2005-2022, Eval 2023

    Args:
        quick: If True, only use 3 windows.
        far_target: If not None, calibrate a per-method threshold on training
            data to achieve this false alarm rate (episodes/year). Default None
            (z>2 only, backward-compatible).
        adaptive: If True, use adaptive rolling quantile + velocity thresholds
            calibrated on training data.
    """
    logger.info("=" * 70)
    logger.info("Walk-Forward Detection Benchmark (Expanding Window)")
    logger.info("=" * 70)

    # Fetch full data range
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA', 'QQQ']  # C4: QQQ adds tech-crisis sensitivity (Flash Crash, 2022)
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
        method_scores_map = {}  # store for ensemble
        for method_name, factory in unsup_factories:
            det = factory()
            det.fit(X_train_enriched)
            scores = det.compute_regime_scores(X_window_enriched)
            method_scores_map[method_name] = scores
            eval_scores = scores[eval_start_idx:]
            eval_dates = dates_window_enriched[eval_start_idx:]

            # FAR-calibrated threshold on training data
            tau = None
            if far_target is not None:
                train_scores = scores[:eval_start_idx]
                train_dates_enriched = dates_window_enriched[:eval_start_idx]
                train_crisis_keys = find_training_crises(train_end)
                tau = calibrate_threshold(
                    train_scores, train_dates_enriched,
                    train_crisis_keys, target_far=far_target,
                    method_name=method_name,
                )
                logger.info(f"    {method_name:25s} FAR-calibrated tau={tau:.4f} "
                           f"(target={far_target}/yr)")

            # Adaptive threshold (rolling quantile + velocity on training data)
            adaptive_tau = None
            if adaptive:
                combined = CombinedAdaptiveThreshold(
                    quantile_params=dict(lookback=252, quantile=0.95, persistence=3, gap=5),
                    velocity_params=dict(smoothing_window=5, velocity_lookback=252, z_threshold=2.0, persistence=2),
                )
                # Run detection on the full window (training + eval) — thresholds
                # are computed causally from past data only
                adaptive_alarm, adaptive_details = combined.detect(scores)
                # Extract eval-period alarms
                adaptive_eval_alarm = adaptive_alarm[eval_start_idx:]

            train_scores_for_diag = scores[:eval_start_idx]
            for ck in eval_crises:
                _record_crisis_result(results, method_name, ck, eval_start_year,
                                      eval_scores, eval_dates,
                                      threshold_far=tau,
                                      train_scores=train_scores_for_diag,
                                      adaptive_alarm=adaptive_eval_alarm if adaptive else None)

        # --- Ensemble: fire if (any QCML z > 1.0) AND (Vol Z > 1.5) ---
        qcml_names = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity']
        available_qcml = [method_scores_map[n] for n in qcml_names if n in method_scores_map]
        vol_scores = method_scores_map.get('Rolling Vol Z')
        if available_qcml and vol_scores is not None:
            qcml_stack = np.array(available_qcml)
            qcml_max = np.nanmax(qcml_stack, axis=0)
            # Ensemble score: product of QCML activity and vol confirmation
            ensemble_scores = np.where(
                (qcml_max > 1.0) & (vol_scores > 1.5),
                (qcml_max + vol_scores) / 2,
                0.0,
            )
            ens_eval = ensemble_scores[eval_start_idx:]
            eval_dates = dates_window_enriched[eval_start_idx:]

            ens_tau = None
            if far_target is not None:
                ens_train = ensemble_scores[:eval_start_idx]
                train_dates_enriched = dates_window_enriched[:eval_start_idx]
                train_crisis_keys = find_training_crises(train_end)
                ens_tau = calibrate_threshold(
                    ens_train, train_dates_enriched,
                    train_crisis_keys, target_far=far_target,
                    method_name='QCML+Vol Ensemble',
                )

            ens_train_scores = ensemble_scores[:eval_start_idx]
            for ck in eval_crises:
                _record_crisis_result(results, 'QCML+Vol Ensemble', ck, eval_start_year,
                                      ens_eval, eval_dates, threshold_far=ens_tau,
                                      train_scores=ens_train_scores)

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

            # FAR-calibrated threshold for RF
            rf_tau = None
            if far_target is not None:
                rf_train_scores = rf_window_scores[:eval_start_idx]
                train_dates_enriched = dates_window_enriched[:eval_start_idx]
                rf_tau = calibrate_threshold(
                    rf_train_scores, train_dates_enriched,
                    train_crises, target_far=far_target,
                    method_name='Random Forest',
                )
                logger.info(f"    {'Random Forest':25s} FAR-calibrated tau={rf_tau:.4f} "
                           f"(target={far_target}/yr)")

            rf_train_scores = rf_window_scores[:eval_start_idx]
            for ck in eval_crises:
                _record_crisis_result(results, 'Random Forest', ck, eval_start_year,
                                      rf_eval, eval_dates_rf,
                                      threshold_far=rf_tau,
                                      train_scores=rf_train_scores)
        else:
            logger.info(f"  RF skipped (only {len(train_crises)} training crises)")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD SUMMARY")
    logger.info("=" * 70)

    method_summary = _compute_summary(results)

    for mname in sorted(method_summary.keys()):
        s = method_summary[mname]
        line = (f"  {mname:25s}: median_d={s['median_d']:.2f}, "
                f"z2: delay={s['median_delay_z2']:.1f}d, FAR={s['median_far_z2']:.1f}/yr, "
                f"det={s['n_detected_z2']}/{s['n_total']}")
        if not np.isnan(s.get('median_far_calibrated', float('nan'))):
            line += (f"  |  FAR-cal: delay={s['median_delay_far']:.1f}d, "
                     f"FAR={s['median_far_calibrated']:.1f}/yr, "
                     f"det={s['n_detected_far']}/{s['n_total']}, "
                     f"tau={s['median_threshold_far']:.2f}")
        if not np.isnan(s.get('median_delay_adaptive', float('nan'))):
            line += (f"  |  Adaptive: delay={s['median_delay_adaptive']:.1f}d, "
                     f"FAR={s['median_far_adaptive']:.1f}/yr, "
                     f"det={s['n_detected_adaptive']}/{s['n_total']}")
        logger.info(line)

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'window_type': 'expanding',
            'train_start': '2005-01-01',
            'threshold_z2': 2.0,
            'far_target': far_target,
            'adaptive': adaptive,
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
                          eval_scores, eval_dates, threshold_far=None,
                          train_scores=None, adaptive_alarm=None):
    """Record detection metrics for one method x crisis x window.

    Computes metrics for both a fixed z>2 threshold and, if provided,
    a FAR-calibrated threshold. Optionally records the crisis peak percentile
    rank within the training-period score distribution (diagnostic field that
    distinguishes "signal absent" from "signal present but threshold too high").

    Args:
        results: Dict to store results (mutated in place).
        method_name: Name of the detection method.
        ck: Crisis key.
        eval_start_year: Year of the evaluation window.
        eval_scores: Z-scores for the evaluation period.
        eval_dates: Dates for the evaluation period.
        threshold_far: FAR-calibrated threshold (if None, only z>2 recorded).
        train_scores: Training-period scores used to compute crisis_peak_pctile.
        adaptive_alarm: Pre-computed boolean alarm array from adaptive threshold
            (same length as eval_scores). If None, adaptive metrics not recorded.
    """
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

    normal_days = np.sum(normal_mask)
    years = max(normal_days / 252, 0.01)
    crisis_start_idx = np.searchsorted(eval_dates, cs)
    crisis_end_idx = np.searchsorted(eval_dates, ce)

    # --- Fixed z > 2.0 threshold ---
    threshold_z2 = 2.0
    alarm_z2 = eval_scores > threshold_z2
    alarms_in_crisis_z2 = np.where(alarm_z2[crisis_start_idx:crisis_end_idx])[0]
    delay_z2 = int(alarms_in_crisis_z2[0]) if len(alarms_in_crisis_z2) > 0 else None
    normal_alarm_z2 = alarm_z2 & normal_mask
    episodes_z2 = count_alarm_episodes(normal_alarm_z2, gap_days=5)
    far_z2 = float(episodes_z2 / years)

    # --- FAR-calibrated threshold ---
    delay_far = None
    far_calibrated = None
    threshold_far_value = None
    if threshold_far is not None:
        threshold_far_value = float(threshold_far)
        alarm_far = eval_scores > threshold_far
        alarms_in_crisis_far = np.where(alarm_far[crisis_start_idx:crisis_end_idx])[0]
        delay_far = int(alarms_in_crisis_far[0]) if len(alarms_in_crisis_far) > 0 else None
        normal_alarm_far = alarm_far & normal_mask
        episodes_far = count_alarm_episodes(normal_alarm_far, gap_days=5)
        far_calibrated = float(episodes_far / years)

    # Diagnostic: peak crisis signal and its percentile rank within training distribution.
    # crisis_peak_pctile immediately shows how "close" a miss was:
    #   90% = signal strongly present, threshold too tight.
    #   40% = signal genuinely absent.
    crisis_peak = float(np.nanmax(crisis_scores)) if len(crisis_scores) > 0 and np.any(~np.isnan(crisis_scores)) else float('nan')
    crisis_peak_pctile = None
    if train_scores is not None and not np.isnan(crisis_peak):
        tv = train_scores[~np.isnan(train_scores)]
        if len(tv) >= 10:
            crisis_peak_pctile = float(np.mean(tv <= crisis_peak))

    # --- Adaptive threshold results ---
    delay_adaptive = None
    far_adaptive = None
    if adaptive_alarm is not None and len(adaptive_alarm) == len(eval_scores):
        alarms_in_crisis_adp = np.where(adaptive_alarm[crisis_start_idx:crisis_end_idx])[0]
        delay_adaptive = int(alarms_in_crisis_adp[0]) if len(alarms_in_crisis_adp) > 0 else None
        normal_alarm_adp = adaptive_alarm & normal_mask
        episodes_adp = count_alarm_episodes(normal_alarm_adp, gap_days=5)
        far_adaptive = float(episodes_adp / years)

    key = f"{method_name}|{ck}|{eval_start_year}"
    results[key] = {
        'method': method_name,
        'crisis': ck,
        'eval_year': eval_start_year,
        'd': float(d) if not np.isnan(d) else None,
        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        # Fixed z > 2 results
        'detection_delay_z2': delay_z2,
        'far_z2': round(far_z2, 3),
        'crisis_detected_z2': delay_z2 is not None,
        # FAR-calibrated results (if available)
        'detection_delay_far': delay_far,
        'far_calibrated': round(far_calibrated, 3) if far_calibrated is not None else None,
        'crisis_detected_far': delay_far is not None if threshold_far is not None else None,
        'threshold_far_value': threshold_far_value,
        # Adaptive threshold results (if available)
        'detection_delay_adaptive': delay_adaptive,
        'far_adaptive': round(far_adaptive, 3) if far_adaptive is not None else None,
        'crisis_detected_adaptive': delay_adaptive is not None if adaptive_alarm is not None else None,
        # Diagnostic (additive, no impact on detection logic)
        'crisis_peak': round(crisis_peak, 4) if not np.isnan(crisis_peak) else None,
        'crisis_peak_pctile': crisis_peak_pctile,
        # Backward compat aliases
        'detection_delay_days': delay_z2,
        'false_alarm_rate_per_year': round(far_z2, 3),
        'crisis_detected': delay_z2 is not None,
    }

    peak_vs_tau = ""
    if threshold_far is not None and not np.isnan(crisis_peak):
        relation = "EXCEEDS" if crisis_peak > threshold_far else "below"
        peak_vs_tau = f" | crisis_peak={crisis_peak:.2f} {relation} tau={threshold_far_value:.2f}"
    if crisis_peak_pctile is not None:
        peak_vs_tau += f" | peak_pctile={crisis_peak_pctile * 100:.1f}%"

    far_str = f"FAR_cal={far_calibrated:.1f}/yr (tau={threshold_far_value:.2f})" if threshold_far is not None else ""
    logger.info(f"    {method_name:25s} on {ck:20s}: "
               f"d={d:.2f}, delay_z2={delay_z2}, FAR_z2={far_z2:.1f}/yr"
               f"  {far_str}{peak_vs_tau}")


def _compute_summary(results):
    """Aggregate per-method: median d, delay, FAR for both z>2 and FAR-calibrated."""
    method_summary = {}
    for key, r in results.items():
        mname = r['method']
        if mname not in method_summary:
            method_summary[mname] = {
                'ds': [],
                # z > 2 tracking
                'delays_z2': [], 'fars_z2': [], 'detected_z2': 0,
                # FAR-calibrated tracking
                'delays_far': [], 'fars_far': [], 'detected_far': 0,
                'thresholds_far': [],
                # Adaptive tracking
                'delays_adaptive': [], 'fars_adaptive': [], 'detected_adaptive': 0,
                'total': 0,
            }

        method_summary[mname]['total'] += 1
        if r['d'] is not None:
            method_summary[mname]['ds'].append(r['d'])

        # z > 2 metrics
        if r.get('detection_delay_z2') is not None:
            method_summary[mname]['delays_z2'].append(r['detection_delay_z2'])
            method_summary[mname]['detected_z2'] += 1
        method_summary[mname]['fars_z2'].append(r.get('far_z2', r.get('false_alarm_rate_per_year', 0)))

        # FAR-calibrated metrics
        if r.get('detection_delay_far') is not None:
            method_summary[mname]['delays_far'].append(r['detection_delay_far'])
            method_summary[mname]['detected_far'] += 1
        if r.get('far_calibrated') is not None:
            method_summary[mname]['fars_far'].append(r['far_calibrated'])
        if r.get('threshold_far_value') is not None:
            method_summary[mname]['thresholds_far'].append(r['threshold_far_value'])

        # Adaptive metrics
        if r.get('detection_delay_adaptive') is not None:
            method_summary[mname]['delays_adaptive'].append(r['detection_delay_adaptive'])
            method_summary[mname]['detected_adaptive'] += 1
        if r.get('far_adaptive') is not None:
            method_summary[mname]['fars_adaptive'].append(r['far_adaptive'])

    summary = {}
    for mname, s in method_summary.items():
        median_d = float(np.median(s['ds'])) if s['ds'] else float('nan')

        # z > 2
        median_delay_z2 = float(np.median(s['delays_z2'])) if s['delays_z2'] else float('nan')
        median_far_z2 = float(np.median(s['fars_z2'])) if s['fars_z2'] else float('nan')
        detect_rate_z2 = s['detected_z2'] / max(s['total'], 1)

        # FAR-calibrated
        median_delay_far = float(np.median(s['delays_far'])) if s['delays_far'] else float('nan')
        median_far_cal = float(np.median(s['fars_far'])) if s['fars_far'] else float('nan')
        detect_rate_far = s['detected_far'] / max(s['total'], 1)
        median_tau = float(np.median(s['thresholds_far'])) if s['thresholds_far'] else float('nan')

        # Adaptive
        median_delay_adp = float(np.median(s['delays_adaptive'])) if s['delays_adaptive'] else float('nan')
        median_far_adp = float(np.median(s['fars_adaptive'])) if s['fars_adaptive'] else float('nan')
        detect_rate_adp = s['detected_adaptive'] / max(s['total'], 1)

        summary[mname] = {
            'median_d': median_d,
            # z > 2 summary
            'median_delay_z2': median_delay_z2,
            'median_far_z2': median_far_z2,
            'detection_rate_z2': detect_rate_z2,
            'n_detected_z2': s['detected_z2'],
            # FAR-calibrated summary
            'median_delay_far': median_delay_far,
            'median_far_calibrated': median_far_cal,
            'detection_rate_far': detect_rate_far,
            'n_detected_far': s['detected_far'],
            'median_threshold_far': median_tau,
            # Adaptive summary
            'median_delay_adaptive': median_delay_adp,
            'median_far_adaptive': median_far_adp,
            'detection_rate_adaptive': detect_rate_adp,
            'n_detected_adaptive': s['detected_adaptive'],
            # Backward compat
            'median_delay': median_delay_z2,
            'median_far': median_far_z2,
            'detection_rate': detect_rate_z2,
            'n_detected': s['detected_z2'],
            'n_total': s['total'],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Walk-forward detection benchmark')
    parser.add_argument('--quick', action='store_true', help='Only 3 windows')
    parser.add_argument('--far-target', type=float, default=2.0,
                        help='Target false alarm rate (episodes/yr) for calibration. '
                             'Default: 2.0. Use 1.0 for the canonical paper number.')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive rolling quantile + velocity thresholds')
    args = parser.parse_args()
    run_walk_forward(quick=args.quick, far_target=args.far_target, adaptive=args.adaptive)


if __name__ == '__main__':
    main()
