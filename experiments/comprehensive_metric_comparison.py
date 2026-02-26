"""
Comprehensive multi-metric comparison of regime detection methods.

Workstream B: Find alternative metrics where QCML geometric methods beat CUSUM.
CUSUM currently wins on Cohen's d; this script evaluates 10+ metrics per
method-crisis pair to identify dimensions where geometry excels.

Reads enhanced_comparison JSON (scores, thresholds, crisis masks) and computes:
  - Cohen's d (from saved results)
  - F1 with tolerance windows (5d, 10d, 30d) via TCPDBench protocol
  - AUC-PR (area under precision-recall curve)
  - Detection delay (days from crisis start to first alarm)
  - False alarm rate (annualized)
  - Precision and recall at adaptive threshold
  - Early warning score (alarms before crisis start)
  - Signal persistence (fraction of crisis days with score > threshold)

Produces a unified comparison table ranked by each metric, identifies where
QCML methods rank highest, and saves results as JSON.

Usage:
    python experiments/comprehensive_metric_comparison.py <enhanced_comparison.json>
    python experiments/comprehensive_metric_comparison.py --latest
    python experiments/comprehensive_metric_comparison.py <file> --tolerance 5 10 30
    python experiments/comprehensive_metric_comparison.py <file> --early-window 30

Output:
    experiments/outputs/regime_detection/comprehensive_metrics_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.detection_metrics import (
    compute_auc_pr,
    compute_detection_delay,
    compute_f1_with_tolerance,
    compute_false_alarm_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Methods that use QCML geometric framework (for ranking analysis).
QCML_METHODS = {
    "Berry Phase Rate",
    "QFI Determinant",
    "Multi-Lag Fidelity",
    "QCML Chern",
    "Geometric Consensus",
}


# =============================================================================
# Novel metric functions
# =============================================================================


def compute_early_warning(
    scores,
    threshold,
    crisis_start_idx,
    lookback_days=60,
):
    """Count trading days with score > threshold BEFORE crisis start.

    Measures advance warning capability. Reactive methods like CUSUM may
    score poorly here because they require sustained deviation to fire,
    whereas geometric methods may detect structural changes earlier.

    The lookback window prevents counting ancient alarms as "early warning."
    Only alarms in [crisis_start_idx - lookback_days, crisis_start_idx) count.

    Args:
        scores: 1-D array of regime scores.
        threshold: 1-D array (per-timestep) or scalar threshold.
        crisis_start_idx: Integer index of crisis start in the time series.
        lookback_days: How far back before crisis_start to search (trading days).

    Returns:
        dict with keys:
            n_early_alarms: Number of alarm days before crisis start.
            earliest_alarm_days: Days before crisis start of the first alarm,
                or None if no early alarm.
            early_alarm_rate: Fraction of lookback window with alarms.
    """
    scores = np.asarray(scores, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(len(scores), float(threshold))

    # Define lookback window: [start_of_window, crisis_start_idx)
    window_start = max(0, crisis_start_idx - lookback_days)
    window_end = crisis_start_idx  # exclusive

    if window_end <= window_start:
        return {
            "n_early_alarms": 0,
            "earliest_alarm_days": None,
            "early_alarm_rate": 0.0,
        }

    window_scores = scores[window_start:window_end]
    window_thresh = threshold[window_start:window_end]

    with np.errstate(invalid="ignore"):
        alarms = (window_scores > window_thresh) & np.isfinite(window_scores)

    n_early = int(np.sum(alarms))
    window_len = window_end - window_start

    # Find the earliest alarm (most days before crisis start)
    if n_early > 0:
        alarm_indices = np.where(alarms)[0]
        # alarm_indices are relative to window_start
        # earliest alarm is the smallest index -> most days before crisis
        earliest_relative = alarm_indices[0]
        earliest_alarm_days = window_len - earliest_relative
    else:
        earliest_alarm_days = None

    return {
        "n_early_alarms": n_early,
        "earliest_alarm_days": earliest_alarm_days,
        "early_alarm_rate": n_early / window_len if window_len > 0 else 0.0,
    }


def compute_signal_persistence(scores, threshold, crisis_start_idx, crisis_end_idx):
    """Fraction of crisis days where score > threshold.

    Measures sustained detection vs. one-off spikes. A good detector should
    stay elevated throughout the crisis, not just fire once and go quiet.

    Args:
        scores: 1-D array of regime scores.
        threshold: 1-D array (per-timestep) or scalar threshold.
        crisis_start_idx: Integer index of crisis start.
        crisis_end_idx: Integer index of crisis end.

    Returns:
        dict with keys:
            persistence: Fraction of crisis days with alarm (0.0 to 1.0).
            n_alarm_days: Number of alarm days during crisis.
            crisis_length: Total crisis days in window.
    """
    scores = np.asarray(scores, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(len(scores), float(threshold))

    T = len(scores)
    start = max(0, crisis_start_idx)
    end = min(T, crisis_end_idx + 1)  # inclusive end

    if end <= start:
        return {
            "persistence": 0.0,
            "n_alarm_days": 0,
            "crisis_length": 0,
        }

    crisis_scores = scores[start:end]
    crisis_thresh = threshold[start:end]

    with np.errstate(invalid="ignore"):
        alarms = (crisis_scores > crisis_thresh) & np.isfinite(crisis_scores)

    n_alarm = int(np.sum(alarms))
    crisis_len = end - start

    return {
        "persistence": n_alarm / crisis_len if crisis_len > 0 else 0.0,
        "n_alarm_days": n_alarm,
        "crisis_length": crisis_len,
    }


def compute_precision_recall_at_threshold(scores, threshold, crisis_mask):
    """Compute pointwise precision and recall at the adaptive threshold.

    Uses the binary alarm mask (score > threshold) against the crisis mask.
    Different from F1-with-tolerance: this is strict pointwise classification.

    Args:
        scores: 1-D array of regime scores.
        threshold: 1-D array (per-timestep) or scalar threshold.
        crisis_mask: Boolean array (True = crisis period).

    Returns:
        dict with keys: precision, recall, f1, n_tp, n_fp, n_fn.
    """
    scores = np.asarray(scores, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(len(scores), float(threshold))
    crisis_mask = np.asarray(crisis_mask, dtype=bool)

    with np.errstate(invalid="ignore"):
        alarm = (scores > threshold) & np.isfinite(scores)

    tp = int(np.sum(alarm & crisis_mask))
    fp = int(np.sum(alarm & ~crisis_mask))
    fn = int(np.sum(~alarm & crisis_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_tp": tp,
        "n_fp": fp,
        "n_fn": fn,
    }


# =============================================================================
# Pipeline
# =============================================================================


def _find_date_index(dates_list, target_date_str):
    """Find index of the closest date >= target in a sorted date list.

    Args:
        dates_list: List of date strings (YYYY-MM-DD format).
        target_date_str: Target date string.

    Returns:
        Integer index, or None if target is after all dates.
    """
    for i, d in enumerate(dates_list):
        if d >= target_date_str:
            return i
    return None


def _find_latest_enhanced_json(out_dir):
    """Find the most recent enhanced_comparison JSON in output directory.

    Args:
        out_dir: Path to output directory.

    Returns:
        Path to the most recent file, or None.
    """
    candidates = sorted(out_dir.glob("enhanced_comparison_*.json"), reverse=True)
    if not candidates:
        return None
    return candidates[0]


def process_comprehensive_metrics(
    input_path,
    tolerance_windows=(5, 10, 30),
    early_window=60,
):
    """Process enhanced comparison JSON and compute all detection metrics.

    For each method and crisis, computes 10+ metrics covering discrimination,
    timeliness, precision, and robustness dimensions.

    Args:
        input_path: Path to enhanced_comparison JSON file.
        tolerance_windows: Tuple of tolerance windows in trading days for F1.
        early_window: Lookback window (trading days) for early warning score.

    Returns:
        dict with per-method-crisis metrics, aggregated summary, and rankings.
    """
    input_path = Path(input_path)
    logger.info(f"Reading enhanced comparison from {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    results = data["results"]
    crisis_masks = data["crisis_masks"]

    # ---- Compute all metrics per method per crisis ----
    all_metrics = {}

    for method, crises_data in results.items():
        all_metrics[method] = {}

        for crisis_key, method_crisis_data in crises_data.items():
            if crisis_key not in crisis_masks:
                logger.warning(f"No crisis mask for {crisis_key}, skipping")
                continue

            cm = crisis_masks[crisis_key]
            dates = cm["dates"]
            mask = np.array(cm["mask"], dtype=bool)
            crisis_start_str = cm["crisis_start"]
            crisis_end_str = cm["crisis_end"]

            # Find boundary indices
            crisis_start_idx = _find_date_index(dates, crisis_start_str)
            crisis_end_idx = _find_date_index(dates, crisis_end_str)

            if crisis_start_idx is None or crisis_end_idx is None:
                logger.warning(
                    f"Cannot find boundary indices for {crisis_key}, skipping"
                )
                continue

            scores = np.array(method_crisis_data["scores"], dtype=float)
            threshold = np.array(method_crisis_data["threshold"], dtype=float)

            crisis_result = {}

            # 1. Cohen's d (from saved results)
            crisis_result["cohens_d"] = method_crisis_data.get("d")

            # 2. F1 at each tolerance window (TCPDBench protocol)
            for tol in tolerance_windows:
                f1_out = compute_f1_with_tolerance(
                    scores,
                    threshold,
                    crisis_start_idx,
                    crisis_end_idx,
                    tolerance_days=tol,
                )
                crisis_result[f"f1@{tol}d"] = _round_safe(f1_out["f1"])
                crisis_result[f"precision@{tol}d"] = _round_safe(f1_out["precision"])
                crisis_result[f"recall@{tol}d"] = _round_safe(f1_out["recall"])

            # 3. AUC-PR
            crisis_result["auc_pr"] = _round_safe(compute_auc_pr(scores, mask))

            # 4. Detection delay
            delay = compute_detection_delay(
                scores, threshold, crisis_start_idx, crisis_end_idx
            )
            crisis_result["detection_delay"] = (
                int(delay) if np.isfinite(delay) else None
            )

            # 5. False alarm rate (annualized)
            crisis_result["false_alarm_rate"] = _round_safe(
                compute_false_alarm_rate(scores, threshold, mask)
            )

            # 6. Precision and recall at adaptive threshold (pointwise)
            pr_out = compute_precision_recall_at_threshold(
                scores, threshold, mask
            )
            crisis_result["pointwise_precision"] = _round_safe(pr_out["precision"])
            crisis_result["pointwise_recall"] = _round_safe(pr_out["recall"])
            crisis_result["pointwise_f1"] = _round_safe(pr_out["f1"])

            # 7. Early warning score
            ew_out = compute_early_warning(
                scores, threshold, crisis_start_idx, lookback_days=early_window
            )
            crisis_result["early_warning_days"] = ew_out["earliest_alarm_days"]
            crisis_result["n_early_alarms"] = ew_out["n_early_alarms"]
            crisis_result["early_alarm_rate"] = _round_safe(ew_out["early_alarm_rate"])

            # 8. Signal persistence
            sp_out = compute_signal_persistence(
                scores, threshold, crisis_start_idx, crisis_end_idx
            )
            crisis_result["signal_persistence"] = _round_safe(sp_out["persistence"])
            crisis_result["n_alarm_days_in_crisis"] = sp_out["n_alarm_days"]
            crisis_result["crisis_length"] = sp_out["crisis_length"]

            all_metrics[method][crisis_key] = crisis_result

    # ---- Aggregate across crises per method ----
    summary = _aggregate_summary(all_metrics, tolerance_windows)

    # ---- Rank methods by each metric ----
    rankings = _compute_rankings(summary)

    # ---- Identify where QCML methods rank best ----
    qcml_advantage = _identify_qcml_advantages(rankings, summary)

    # ---- Save ----
    output = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "config": {
            "tolerance_windows": list(tolerance_windows),
            "early_warning_lookback_days": early_window,
            "qcml_methods": sorted(QCML_METHODS),
        },
        "per_method_crisis": _sanitize_for_json(all_metrics),
        "summary": _sanitize_for_json(summary),
        "rankings": _sanitize_for_json(rankings),
        "qcml_advantage": _sanitize_for_json(qcml_advantage),
    }

    out_dir = ROOT / "experiments" / "outputs" / "regime_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"comprehensive_metrics_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Saved comprehensive metrics to {out_path}")

    # ---- Print summary tables ----
    _print_unified_table(summary, tolerance_windows)
    _print_rankings_table(rankings)
    _print_qcml_advantage_table(qcml_advantage)

    return output


# =============================================================================
# Aggregation
# =============================================================================

# Metrics where higher is better (for ranking direction).
HIGHER_IS_BETTER = {
    "cohens_d",
    "f1@5d",
    "f1@10d",
    "f1@30d",
    "auc_pr",
    "pointwise_precision",
    "pointwise_recall",
    "pointwise_f1",
    "early_warning_days",
    "n_early_alarms",
    "early_alarm_rate",
    "signal_persistence",
}

# Metrics where lower is better.
LOWER_IS_BETTER = {
    "detection_delay",
    "false_alarm_rate",
}

# All metrics to aggregate (the canonical set).
ALL_METRIC_KEYS = sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)


def _aggregate_summary(all_metrics, tolerance_windows):
    """Aggregate per-crisis metrics into per-method summary statistics.

    Computes mean and median for each metric across all crises.

    Args:
        all_metrics: dict of method -> crisis_key -> metric_dict.
        tolerance_windows: Tuple of tolerance windows used.

    Returns:
        dict of method -> aggregated summary.
    """
    summary = {}

    for method, crises_data in all_metrics.items():
        if not crises_data:
            continue

        method_summary = {"n_crises": len(crises_data)}

        for key in ALL_METRIC_KEYS:
            values = []
            for crisis_metrics in crises_data.values():
                v = crisis_metrics.get(key)
                if v is not None and np.isfinite(v):
                    values.append(v)

            if values:
                method_summary[f"mean_{key}"] = round(float(np.mean(values)), 4)
                method_summary[f"median_{key}"] = round(float(np.median(values)), 4)
                method_summary[f"std_{key}"] = round(float(np.std(values)), 4)
            else:
                method_summary[f"mean_{key}"] = None
                method_summary[f"median_{key}"] = None
                method_summary[f"std_{key}"] = None

        summary[method] = method_summary

    return summary


def _compute_rankings(summary):
    """Rank methods by each metric (1 = best).

    For HIGHER_IS_BETTER metrics, rank descending (highest value = rank 1).
    For LOWER_IS_BETTER metrics, rank ascending (lowest value = rank 1).
    Methods with missing values get the worst rank.

    Args:
        summary: dict of method -> aggregated summary.

    Returns:
        dict with:
            by_metric: {metric_name: [(rank, method, value), ...]}
            by_method: {method: {metric_name: rank}}
    """
    methods = sorted(summary.keys())
    n_methods = len(methods)

    by_metric = {}
    by_method = {m: {} for m in methods}

    for key in ALL_METRIC_KEYS:
        mean_key = f"mean_{key}"

        # Collect (value, method) pairs
        entries = []
        for m in methods:
            val = summary[m].get(mean_key)
            entries.append((val, m))

        # Sort: higher is better -> descending; lower is better -> ascending
        if key in HIGHER_IS_BETTER:
            # None values go last (worst)
            entries.sort(
                key=lambda x: x[0] if x[0] is not None else -np.inf,
                reverse=True,
            )
        else:
            # Lower is better -> ascending; None values go last (worst)
            entries.sort(
                key=lambda x: x[0] if x[0] is not None else np.inf,
            )

        ranked = []
        for rank, (val, m) in enumerate(entries, 1):
            ranked.append({"rank": rank, "method": m, "value": val})
            by_method[m][key] = rank

        by_metric[key] = ranked

    return {"by_metric": by_metric, "by_method": by_method}


def _identify_qcml_advantages(rankings, summary):
    """Find metrics where QCML methods collectively outperform baselines.

    For each metric, computes the best QCML rank and the best baseline rank.
    Reports metrics where QCML has advantage (best QCML rank < best baseline rank)
    and metrics where QCML ties or loses.

    Args:
        rankings: Output of _compute_rankings.
        summary: Aggregated summary dict.

    Returns:
        dict with:
            advantages: Metrics where best QCML outranks best baseline.
            ties: Metrics where best QCML ties best baseline.
            disadvantages: Metrics where best baseline outranks best QCML.
            method_rank_summary: Mean rank per method across all metrics.
    """
    methods = sorted(summary.keys())
    by_method = rankings["by_method"]

    advantages = []
    ties = []
    disadvantages = []

    for key in ALL_METRIC_KEYS:
        qcml_ranks = []
        baseline_ranks = []

        for m in methods:
            rank = by_method[m].get(key)
            if rank is None:
                continue
            if m in QCML_METHODS:
                qcml_ranks.append((rank, m))
            else:
                baseline_ranks.append((rank, m))

        if not qcml_ranks or not baseline_ranks:
            continue

        best_qcml = min(qcml_ranks, key=lambda x: x[0])
        best_baseline = min(baseline_ranks, key=lambda x: x[0])

        entry = {
            "metric": key,
            "best_qcml_rank": best_qcml[0],
            "best_qcml_method": best_qcml[1],
            "best_qcml_value": summary[best_qcml[1]].get(f"mean_{key}"),
            "best_baseline_rank": best_baseline[0],
            "best_baseline_method": best_baseline[1],
            "best_baseline_value": summary[best_baseline[1]].get(f"mean_{key}"),
        }

        if best_qcml[0] < best_baseline[0]:
            advantages.append(entry)
        elif best_qcml[0] == best_baseline[0]:
            ties.append(entry)
        else:
            disadvantages.append(entry)

    # Mean rank per method across all metrics (lower = better overall)
    method_rank_summary = {}
    for m in methods:
        ranks = [
            by_method[m][key]
            for key in ALL_METRIC_KEYS
            if key in by_method[m]
        ]
        method_rank_summary[m] = {
            "mean_rank": round(float(np.mean(ranks)), 2) if ranks else None,
            "best_rank": min(ranks) if ranks else None,
            "worst_rank": max(ranks) if ranks else None,
            "is_qcml": m in QCML_METHODS,
        }

    return {
        "advantages": advantages,
        "ties": ties,
        "disadvantages": disadvantages,
        "method_rank_summary": method_rank_summary,
    }


# =============================================================================
# Printing
# =============================================================================


def _print_unified_table(summary, tolerance_windows):
    """Print unified comparison table sorted by mean Cohen's d."""
    sorted_methods = sorted(
        summary.keys(),
        key=lambda m: summary[m].get("mean_cohens_d") or -np.inf,
        reverse=True,
    )

    print("\n" + "=" * 120)
    print("COMPREHENSIVE METRIC COMPARISON (sorted by mean Cohen's d)")
    print("=" * 120)

    # Header row
    header = (
        f"{'Method':<25} "
        f"{'d':>6} "
        f"{'F1@10':>6} "
        f"{'AUC-PR':>7} "
        f"{'Delay':>6} "
        f"{'FAR/yr':>7} "
        f"{'Prec':>6} "
        f"{'Rec':>6} "
        f"{'Early':>6} "
        f"{'Persist':>8} "
        f"{'Type':>8}"
    )
    print(header)
    print("-" * 120)

    for method in sorted_methods:
        s = summary[method]
        is_qcml = method in QCML_METHODS
        method_type = "QCML" if is_qcml else "Baseline"

        d_val = s.get("mean_cohens_d")
        f1_val = s.get("mean_f1@10d")
        auc_val = s.get("mean_auc_pr")
        delay_val = s.get("mean_detection_delay")
        far_val = s.get("mean_false_alarm_rate")
        prec_val = s.get("mean_pointwise_precision")
        rec_val = s.get("mean_pointwise_recall")
        early_val = s.get("mean_early_warning_days")
        persist_val = s.get("mean_signal_persistence")

        row = f"{method:<25} "
        row += _fmt_val(d_val, width=6, decimals=3)
        row += _fmt_val(f1_val, width=6, decimals=3)
        row += _fmt_val(auc_val, width=7, decimals=3)
        row += _fmt_val(delay_val, width=6, decimals=1)
        row += _fmt_val(far_val, width=7, decimals=1)
        row += _fmt_val(prec_val, width=6, decimals=3)
        row += _fmt_val(rec_val, width=6, decimals=3)
        row += _fmt_val(early_val, width=6, decimals=1)
        row += _fmt_val(persist_val, width=8, decimals=3)
        row += f" {method_type:>8}"
        print(row)

    print("=" * 120)
    print("  d = Cohen's d | F1@10 = F1 with 10d tolerance | Delay = detection delay (days)")
    print("  FAR/yr = false alarms per year | Prec/Rec = pointwise precision/recall")
    print("  Early = earliest alarm (days before crisis) | Persist = signal persistence")


def _print_rankings_table(rankings):
    """Print rankings by each metric."""
    by_metric = rankings["by_metric"]

    print("\n" + "=" * 100)
    print("METHOD RANKINGS BY METRIC (1 = best)")
    print("=" * 100)

    for key in ALL_METRIC_KEYS:
        if key not in by_metric:
            continue

        direction = "higher better" if key in HIGHER_IS_BETTER else "lower better"
        print(f"\n  {key} ({direction}):")

        entries = by_metric[key]
        for entry in entries:
            rank = entry["rank"]
            method = entry["method"]
            val = entry["value"]
            marker = " *" if method in QCML_METHODS else "  "
            val_str = f"{val:.4f}" if val is not None else "N/A"
            print(f"    {rank:2d}. {method:<25s} {val_str:>10}{marker}")


def _print_qcml_advantage_table(qcml_advantage):
    """Print summary of where QCML methods have advantages."""
    advantages = qcml_advantage["advantages"]
    disadvantages = qcml_advantage["disadvantages"]
    method_ranks = qcml_advantage["method_rank_summary"]

    print("\n" + "=" * 100)
    print("QCML ADVANTAGE ANALYSIS")
    print("=" * 100)

    if advantages:
        print(f"\n  QCML WINS ({len(advantages)} metrics):")
        for entry in advantages:
            metric = entry["metric"]
            qr = entry["best_qcml_rank"]
            qm = entry["best_qcml_method"]
            qv = entry["best_qcml_value"]
            br = entry["best_baseline_rank"]
            bm = entry["best_baseline_method"]
            bv = entry["best_baseline_value"]
            qv_str = f"{qv:.4f}" if qv is not None else "N/A"
            bv_str = f"{bv:.4f}" if bv is not None else "N/A"
            print(
                f"    {metric:<25s}  "
                f"QCML #{qr} ({qm}: {qv_str}) vs "
                f"Baseline #{br} ({bm}: {bv_str})"
            )
    else:
        print("\n  QCML WINS: None")

    if disadvantages:
        print(f"\n  BASELINE WINS ({len(disadvantages)} metrics):")
        for entry in disadvantages:
            metric = entry["metric"]
            qr = entry["best_qcml_rank"]
            br = entry["best_baseline_rank"]
            bm = entry["best_baseline_method"]
            bv = entry["best_baseline_value"]
            bv_str = f"{bv:.4f}" if bv is not None else "N/A"
            print(
                f"    {metric:<25s}  "
                f"Best QCML rank #{qr} vs "
                f"Baseline #{br} ({bm}: {bv_str})"
            )
    else:
        print("\n  BASELINE WINS: None")

    # Overall method ranking
    print("\n  OVERALL METHOD RANKING (mean rank across all metrics):")
    sorted_overall = sorted(
        method_ranks.items(),
        key=lambda x: x[1]["mean_rank"] if x[1]["mean_rank"] is not None else np.inf,
    )
    for method, info in sorted_overall:
        mr = info["mean_rank"]
        best = info["best_rank"]
        worst = info["worst_rank"]
        tag = "QCML" if info["is_qcml"] else "Base"
        if mr is not None:
            print(
                f"    {method:<25s}  "
                f"mean={mr:5.2f}  best={best}  worst={worst}  [{tag}]"
            )

    print("=" * 100)


# =============================================================================
# Helpers
# =============================================================================


def _round_safe(val, decimals=4):
    """Round a value safely, handling NaN and None.

    Args:
        val: Numeric value or NaN or None.
        decimals: Number of decimal places.

    Returns:
        Rounded float, or None if input is invalid.
    """
    if val is None:
        return None
    if np.isnan(val):
        return None
    return round(float(val), decimals)


def _fmt_val(val, width=8, decimals=3):
    """Format a value for table printing.

    Args:
        val: Numeric value or None.
        width: Column width.
        decimals: Decimal places.

    Returns:
        Formatted string with trailing space.
    """
    if val is None:
        return f"{'N/A':>{width}} "
    return f"{val:>{width}.{decimals}f} "


def _sanitize_for_json(obj):
    """Recursively convert numpy types and NaN to JSON-safe types.

    Args:
        obj: Any nested structure (dict, list, numpy scalar).

    Returns:
        JSON-serializable version of obj.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive multi-metric comparison of regime detection methods. "
            "Workstream B: find metrics where QCML geometric methods beat CUSUM."
        ),
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=str,
        default=None,
        help="Path to enhanced_comparison JSON file.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent enhanced_comparison JSON in output directory.",
    )
    parser.add_argument(
        "--tolerance",
        nargs="+",
        type=int,
        default=[5, 10, 30],
        help="Tolerance windows in trading days (default: 5 10 30).",
    )
    parser.add_argument(
        "--early-window",
        type=int,
        default=60,
        help="Lookback window for early warning metric (default: 60 trading days).",
    )
    args = parser.parse_args()

    # Resolve input file
    if args.latest or args.input_file is None:
        out_dir = ROOT / "experiments" / "outputs" / "regime_detection"
        input_path = _find_latest_enhanced_json(out_dir)
        if input_path is None:
            logger.error(
                "No enhanced_comparison JSON found. Run enhanced_comparison.py first."
            )
            sys.exit(1)
        logger.info(f"Using latest enhanced comparison: {input_path.name}")
    else:
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

    process_comprehensive_metrics(
        input_path,
        tolerance_windows=tuple(args.tolerance),
        early_window=args.early_window,
    )


if __name__ == "__main__":
    main()
