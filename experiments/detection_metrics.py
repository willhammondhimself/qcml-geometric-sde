"""
Detection metrics for regime detection evaluation.

Computes F1 with tolerance windows (TCPDBench protocol), AUC-PR,
detection delay, and false alarm rate from saved score time series.

Reads the JSON output of enhanced_comparison.py, which contains raw scores,
adaptive thresholds, and crisis masks per method per crisis.

Usage:
    python experiments/detection_metrics.py <enhanced_comparison.json>
    python experiments/detection_metrics.py --tolerance 5 10 30 <file>

Output:
    experiments/outputs/regime_detection/detection_metrics_YYYYMMDD_HHMMSS.json
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

logger = logging.getLogger(__name__)


# =============================================================================
# Core metric functions
# =============================================================================


def compute_f1_with_tolerance(
    scores,
    threshold,
    crisis_start_idx,
    crisis_end_idx,
    tolerance_days=10,
):
    """Compute F1 with tolerance windows following the TCPDBench protocol.

    An alarm at time t is a true positive if |t - boundary| <= tolerance_days
    for some unmatched boundary. Each boundary can be matched at most once
    (by the closest alarm). Boundaries are crisis_start_idx and crisis_end_idx.

    Reference: Van den Burg & Williams (2020), "An Evaluation of Change Point
    Detection Algorithms".

    Args:
        scores: 1-D array of regime scores.
        threshold: 1-D array (per-timestep) or scalar threshold.
            An alarm fires where score > threshold.
        crisis_start_idx: Integer index of crisis start in the time series.
        crisis_end_idx: Integer index of crisis end in the time series.
        tolerance_days: Maximum distance (in trading days) for an alarm to
            count as a match to a boundary.

    Returns:
        dict with keys: precision, recall, f1, n_tp, n_fp, n_boundaries_detected.
    """
    scores = np.asarray(scores, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(len(scores), float(threshold))

    T = len(scores)

    # Identify alarm indices (score > threshold, ignoring NaN)
    with np.errstate(invalid="ignore"):
        alarms = np.where((scores > threshold) & np.isfinite(scores))[0]

    boundaries = [crisis_start_idx, crisis_end_idx]
    n_boundaries = len(boundaries)

    # Match alarms to boundaries: each boundary matched by closest alarm
    matched_alarms = set()
    matched_boundaries = set()

    for b_idx, boundary in enumerate(boundaries):
        # Find alarms within tolerance of this boundary
        if len(alarms) == 0:
            continue
        distances = np.abs(alarms - boundary)
        within_tol = np.where(distances <= tolerance_days)[0]

        if len(within_tol) == 0:
            continue

        # Pick closest alarm that hasn't been matched yet
        sorted_by_dist = within_tol[np.argsort(distances[within_tol])]
        for candidate_idx in sorted_by_dist:
            alarm_pos = alarms[candidate_idx]
            if alarm_pos not in matched_alarms:
                matched_alarms.add(alarm_pos)
                matched_boundaries.add(b_idx)
                break

    n_tp = len(matched_alarms)
    n_fp = len(alarms) - n_tp
    n_boundaries_detected = len(matched_boundaries)

    precision = n_tp / len(alarms) if len(alarms) > 0 else 0.0
    recall = n_boundaries_detected / n_boundaries if n_boundaries > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_boundaries_detected": n_boundaries_detected,
    }


def compute_auc_pr(scores, crisis_mask):
    """Compute area under the precision-recall curve.

    Args:
        scores: 1-D array of regime scores (higher = more anomalous).
        crisis_mask: Boolean array (True = crisis period).

    Returns:
        Float AUC-PR value, or NaN if no positive samples or all scores invalid.
    """
    scores = np.asarray(scores, dtype=float)
    crisis_mask = np.asarray(crisis_mask, dtype=bool)

    # Handle NaN scores: replace with median of finite values
    valid = np.isfinite(scores)
    if not np.any(valid):
        return np.nan

    median_val = np.nanmedian(scores)
    scores_clean = scores.copy()
    scores_clean[~valid] = median_val

    # Check for degenerate cases
    if not np.any(crisis_mask) or np.all(crisis_mask):
        return np.nan

    precision_arr, recall_arr, _ = precision_recall_curve(crisis_mask, scores_clean)
    return float(auc(recall_arr, precision_arr))


def compute_detection_delay(scores, threshold, crisis_start_idx, crisis_end_idx=None):
    """Compute detection delay: days from crisis start to first alarm.

    Searches for the first alarm (score > threshold) within the crisis window
    [crisis_start_idx, crisis_end_idx]. An alarm at crisis_start_idx gives
    delay = 0.

    Args:
        scores: 1-D array of regime scores.
        threshold: 1-D array (per-timestep) or scalar threshold.
        crisis_start_idx: Integer index of crisis start.
        crisis_end_idx: Integer index of crisis end. If None, uses end of array.

    Returns:
        Integer delay (0 = detected on day 1), or NaN if no alarm in window.
    """
    scores = np.asarray(scores, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(len(scores), float(threshold))

    T = len(scores)
    if crisis_end_idx is None:
        crisis_end_idx = T - 1

    # Search within crisis window [start, end] inclusive
    for t in range(crisis_start_idx, min(crisis_end_idx + 1, T)):
        if np.isfinite(scores[t]) and scores[t] > threshold[t]:
            return t - crisis_start_idx

    return np.nan


def compute_false_alarm_rate(
    scores, threshold, crisis_mask, trading_days_per_year=252
):
    """Compute annualized false alarm rate: alarms per year outside crisis.

    Args:
        scores: 1-D array of regime scores.
        threshold: 1-D array (per-timestep) or scalar threshold.
        crisis_mask: Boolean array (True = crisis period).
        trading_days_per_year: Annualization factor (default 252).

    Returns:
        Float: annualized false alarm rate.
    """
    scores = np.asarray(scores, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(len(scores), float(threshold))
    crisis_mask = np.asarray(crisis_mask, dtype=bool)

    # NaN scores are not alarms
    with np.errstate(invalid="ignore"):
        alarm = (scores > threshold) & np.isfinite(scores)

    false_alarms = np.sum(alarm & ~crisis_mask)
    normal_days = np.sum(~crisis_mask)

    if normal_days == 0:
        return 0.0

    return float(false_alarms / normal_days) * trading_days_per_year


# =============================================================================
# Pipeline: process enhanced comparison output
# =============================================================================


def _find_date_index(dates_list, target_date_str):
    """Find the index of the closest date >= target in a sorted date list.

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


def process_enhanced_results(input_path, tolerance_windows=(5, 10, 30)):
    """Process enhanced comparison JSON and compute detection metrics.

    For each method and crisis, computes:
    - F1 at each tolerance window (5d, 10d, 30d)
    - AUC-PR
    - Detection delay (days)
    - False alarm rate (per year)

    Args:
        input_path: Path to enhanced_comparison JSON file.
        tolerance_windows: Tuple of tolerance windows in trading days.

    Returns:
        dict with per-method-crisis metrics and aggregated summary.
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        data = json.load(f)

    results = data["results"]
    crisis_masks = data["crisis_masks"]

    # Compute metrics per method per crisis
    metrics = {}

    for method, crises_data in results.items():
        metrics[method] = {}

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

            crisis_metrics = {
                "d": method_crisis_data.get("d"),
            }

            # F1 at each tolerance window
            for tol in tolerance_windows:
                f1_result = compute_f1_with_tolerance(
                    scores, threshold, crisis_start_idx, crisis_end_idx,
                    tolerance_days=tol,
                )
                crisis_metrics[f"f1@{tol}d"] = round(f1_result["f1"], 4)
                crisis_metrics[f"precision@{tol}d"] = round(f1_result["precision"], 4)
                crisis_metrics[f"recall@{tol}d"] = round(f1_result["recall"], 4)

            # AUC-PR
            crisis_metrics["auc_pr"] = round(
                compute_auc_pr(scores, mask), 4
            )

            # Detection delay
            delay = compute_detection_delay(
                scores, threshold, crisis_start_idx, crisis_end_idx,
            )
            crisis_metrics["detection_delay"] = (
                int(delay) if np.isfinite(delay) else None
            )

            # False alarm rate
            crisis_metrics["false_alarm_rate"] = round(
                compute_false_alarm_rate(scores, threshold, mask), 2
            )

            metrics[method][crisis_key] = crisis_metrics

    # Aggregate across crises per method
    summary = {}
    for method, crises_data in metrics.items():
        if not crises_data:
            continue

        method_summary = {"n_crises": len(crises_data)}

        # Collect values for each metric across crises
        for key in ["f1@10d", "auc_pr", "detection_delay", "false_alarm_rate"]:
            values = []
            for ck, cm in crises_data.items():
                v = cm.get(key)
                if v is not None and np.isfinite(v):
                    values.append(v)
            if values:
                method_summary[f"mean_{key}"] = round(np.mean(values), 4)
                method_summary[f"median_{key}"] = round(np.median(values), 4)

        # Also aggregate F1 at other tolerance windows
        for tol in tolerance_windows:
            key = f"f1@{tol}d"
            values = [
                cm.get(key)
                for cm in crises_data.values()
                if cm.get(key) is not None
            ]
            if values:
                method_summary[f"mean_{key}"] = round(np.mean(values), 4)
                method_summary[f"median_{key}"] = round(np.median(values), 4)

        summary[method] = method_summary

    # Sort summary by mean F1@10d descending
    sorted_methods = sorted(
        summary.keys(),
        key=lambda m: summary[m].get("mean_f1@10d", 0),
        reverse=True,
    )

    output = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "tolerance_windows": list(tolerance_windows),
        "per_method_crisis": metrics,
        "summary": {m: summary[m] for m in sorted_methods},
    }

    # Save output
    out_dir = ROOT / "experiments" / "outputs" / "regime_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"detection_metrics_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Saved detection metrics to {out_path}")

    # Print summary table
    _print_summary_table(summary, sorted_methods, tolerance_windows)

    return output


def _print_summary_table(summary, sorted_methods, tolerance_windows):
    """Print a formatted summary table to console.

    Args:
        summary: dict of method -> aggregated metrics.
        sorted_methods: list of method names sorted by F1@10d.
        tolerance_windows: tuple of tolerance windows used.
    """
    print("\n" + "=" * 90)
    print("DETECTION METRICS SUMMARY (sorted by mean F1@10d)")
    print("=" * 90)

    header = f"{'Method':<30} "
    for tol in tolerance_windows:
        header += f"{'F1@' + str(tol) + 'd':>8} "
    header += f"{'AUC-PR':>8} {'Delay':>8} {'FAR/yr':>8}"
    print(header)
    print("-" * 90)

    for method in sorted_methods:
        s = summary[method]
        row = f"{method:<30} "
        for tol in tolerance_windows:
            val = s.get(f"mean_f1@{tol}d", np.nan)
            row += f"{val:>8.3f} " if np.isfinite(val) else f"{'N/A':>8} "
        auc_val = s.get("mean_auc_pr", np.nan)
        row += f"{auc_val:>8.3f} " if np.isfinite(auc_val) else f"{'N/A':>8} "
        delay_val = s.get("mean_detection_delay", np.nan)
        row += f"{delay_val:>8.1f} " if np.isfinite(delay_val) else f"{'N/A':>8} "
        far_val = s.get("mean_false_alarm_rate", np.nan)
        row += f"{far_val:>8.1f}" if np.isfinite(far_val) else f"{'N/A':>8}"
        print(row)

    print("=" * 90)


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    parser = argparse.ArgumentParser(
        description="Compute detection metrics from enhanced comparison output."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to enhanced_comparison JSON file.",
    )
    parser.add_argument(
        "--tolerance",
        nargs="+",
        type=int,
        default=[5, 10, 30],
        help="Tolerance windows in trading days (default: 5 10 30).",
    )
    args = parser.parse_args()

    process_enhanced_results(args.input_file, tuple(args.tolerance))


if __name__ == "__main__":
    main()
