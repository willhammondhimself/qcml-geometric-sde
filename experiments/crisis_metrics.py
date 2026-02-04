"""
Crisis Metrics Module

Statistical metrics computation for crisis validation experiments.
Provides functions for significance testing, detection metrics, and lead time analysis.

Author: QCML Research
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from scipy import stats
import logging

from .crisis_config import (
    CrisisValidationResult,
    AggregateMetrics,
    CrisisDefinition,
    ValidationConfig
)

logger = logging.getLogger(__name__)


def compute_statistical_significance(
    chern_before: np.ndarray,
    chern_after: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute statistical significance of Chern number change.

    Performs Welch's t-test (unequal variance t-test) to determine
    if the difference between pre and post-crisis Chern numbers is significant.

    Args:
        chern_before: Chern values before crisis
        chern_after: Chern values after crisis
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with:
            - delta_chern: Difference in means
            - t_statistic: t-test statistic
            - p_value: Two-tailed p-value
            - effect_size: Cohen's d effect size
            - is_significant: Whether result is significant at alpha level
            - confidence_interval: 95% CI for the difference

    Example:
        >>> before = np.array([0.1, 0.15, 0.12, 0.11, 0.13])
        >>> after = np.array([0.35, 0.40, 0.38, 0.42, 0.36])
        >>> result = compute_statistical_significance(before, after)
        >>> print(f"t={result['t_statistic']:.2f}, p={result['p_value']:.4f}")
    """
    chern_before = np.asarray(chern_before).flatten()
    chern_after = np.asarray(chern_after).flatten()

    # Handle edge cases
    if len(chern_before) < 2 or len(chern_after) < 2:
        logger.warning("Insufficient samples for statistical test")
        return {
            'delta_chern': np.mean(chern_after) - np.mean(chern_before),
            't_statistic': 0.0,
            'p_value': 1.0,
            'effect_size': 0.0,
            'is_significant': False,
            'confidence_interval': (0.0, 0.0)
        }

    # Compute means and standard deviations
    mean_before = np.mean(chern_before)
    mean_after = np.mean(chern_after)
    std_before = np.std(chern_before, ddof=1)
    std_after = np.std(chern_after, ddof=1)

    delta_chern = mean_after - mean_before

    # Welch's t-test (handles unequal variances)
    t_stat, p_value = stats.ttest_ind(chern_after, chern_before, equal_var=False)

    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(chern_before) - 1) * std_before**2 + (len(chern_after) - 1) * std_after**2)
        / (len(chern_before) + len(chern_after) - 2)
    )
    effect_size = delta_chern / (pooled_std + 1e-10)

    # Confidence interval for the difference (Welch-Satterthwaite approximation)
    se_diff = np.sqrt(std_before**2 / len(chern_before) + std_after**2 / len(chern_after))

    # Degrees of freedom (Welch-Satterthwaite)
    df_num = (std_before**2 / len(chern_before) + std_after**2 / len(chern_after))**2
    df_den = (
        (std_before**2 / len(chern_before))**2 / (len(chern_before) - 1) +
        (std_after**2 / len(chern_after))**2 / (len(chern_after) - 1)
    )
    df = df_num / (df_den + 1e-10)

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = delta_chern - t_crit * se_diff
    ci_upper = delta_chern + t_crit * se_diff

    return {
        'delta_chern': delta_chern,
        't_statistic': abs(t_stat),
        'p_value': p_value,
        'effect_size': abs(effect_size),
        'is_significant': p_value < alpha,
        'confidence_interval': (ci_lower, ci_upper),
        'mean_before': mean_before,
        'mean_after': mean_after,
        'std_before': std_before,
        'std_after': std_after
    }


def compute_precision_recall(
    detected_transitions: List[int],
    true_crisis_idx: int,
    tolerance_days: int = 20,
    total_days: int = 252
) -> Dict[str, float]:
    """
    Compute precision and recall for crisis detection.

    Precision = TP / (TP + FP): What fraction of detected transitions are real?
    Recall = TP / (TP + FN): Did we detect the crisis?

    Since we have only one true crisis event, recall is binary (0 or 1).
    Precision depends on how many false positives we generated.

    Args:
        detected_transitions: Indices where transitions were detected
        true_crisis_idx: Index of the actual crisis
        tolerance_days: Days around crisis to consider a true positive
        total_days: Total number of days in the analysis period

    Returns:
        Dictionary with:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN) = 1.0 if crisis detected, else 0.0
            - f1_score: 2 * (precision * recall) / (precision + recall)
            - n_true_positives: Number of transitions near crisis
            - n_false_positives: Number of transitions far from crisis

    Example:
        >>> transitions = [95, 100, 150, 200]  # Detected at days 95, 100, 150, 200
        >>> true_idx = 100  # Crisis was at day 100
        >>> result = compute_precision_recall(transitions, true_idx, tolerance_days=10)
        >>> print(f"P={result['precision']:.2f}, R={result['recall']:.2f}")
    """
    detected_transitions = list(detected_transitions)

    if not detected_transitions:
        # No detections: recall = 0, precision undefined (set to 0)
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'n_true_positives': 0,
            'n_false_positives': 0
        }

    # True positives: transitions within tolerance of crisis
    true_positives = [
        t for t in detected_transitions
        if abs(t - true_crisis_idx) <= tolerance_days
    ]
    n_tp = len(true_positives)

    # False positives: transitions far from crisis
    false_positives = [
        t for t in detected_transitions
        if abs(t - true_crisis_idx) > tolerance_days
    ]
    n_fp = len(false_positives)

    # Recall: 1 if we detected the crisis (at least one TP), else 0
    recall = 1.0 if n_tp > 0 else 0.0

    # Precision: TP / (TP + FP)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0

    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_true_positives': n_tp,
        'n_false_positives': n_fp
    }


def compute_lead_time(
    chern_series: np.ndarray,
    times: np.ndarray,
    crisis_date: str,
    threshold: float = 0.1,
    min_consecutive: int = 2
) -> Optional[int]:
    """
    Compute lead time: days before crisis that Chern change was first detected.

    Looks for the earliest point where the Chern number shows a sustained
    deviation from its pre-crisis baseline.

    Args:
        chern_series: Rolling Chern number values
        times: Timestamps corresponding to Chern series
        crisis_date: Crisis date (YYYY-MM-DD)
        threshold: Minimum change from baseline to consider a detection
        min_consecutive: Minimum consecutive days above threshold

    Returns:
        Lead time in trading days, or None if no lead time detected

    Example:
        >>> chern = np.array([0.1, 0.12, 0.15, 0.3, 0.35, 0.4, 0.38])
        >>> times = pd.date_range("2008-09-01", periods=7, freq="B")
        >>> lead = compute_lead_time(chern, times.values, "2008-09-08")
        >>> print(f"Lead time: {lead} days")
    """
    chern_series = np.asarray(chern_series)
    times = pd.DatetimeIndex(times)
    crisis_ts = pd.Timestamp(crisis_date)

    # Find crisis index
    crisis_mask = times >= crisis_ts
    if not crisis_mask.any():
        logger.warning(f"Crisis date {crisis_date} is after data range")
        return None

    crisis_idx = crisis_mask.argmax()

    if crisis_idx < 10:
        logger.warning("Insufficient data before crisis for lead time analysis")
        return None

    # Compute baseline (mean of first half of pre-crisis period)
    baseline_end = crisis_idx // 2
    baseline = np.mean(chern_series[:max(baseline_end, 5)])

    # Look for sustained deviation from baseline before crisis
    pre_crisis_chern = chern_series[:crisis_idx]

    # Find first point where change from baseline exceeds threshold
    # and is sustained for min_consecutive days
    deviations = np.abs(pre_crisis_chern - baseline)

    for i in range(len(deviations) - min_consecutive + 1):
        # Check if consecutive points are above threshold
        if all(deviations[i:i + min_consecutive] > threshold):
            # Lead time is days from first detection to crisis
            detection_idx = i
            lead_time = crisis_idx - detection_idx
            return lead_time

    # Alternative: look for rolling change exceeding threshold
    if len(pre_crisis_chern) > 5:
        rolling_change = np.abs(np.diff(pre_crisis_chern))
        for i in range(len(rolling_change) - min_consecutive + 1):
            if all(rolling_change[i:i + min_consecutive] > threshold / 2):
                detection_idx = i
                lead_time = crisis_idx - detection_idx
                return lead_time

    return None


def compute_rolling_statistics(
    chern_series: np.ndarray,
    window: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling statistics for Chern number series.

    Args:
        chern_series: Chern number time series
        window: Rolling window size

    Returns:
        Dictionary with:
            - rolling_mean: Rolling mean
            - rolling_std: Rolling standard deviation
            - rolling_zscore: Rolling z-score (deviation from rolling mean)
            - rolling_change: Rolling change (diff)

    Example:
        >>> chern = np.random.randn(100)
        >>> stats = compute_rolling_statistics(chern, window=20)
        >>> print(f"Max z-score: {np.max(np.abs(stats['rolling_zscore'])):.2f}")
    """
    chern_series = np.asarray(chern_series)
    n = len(chern_series)

    if n < window:
        window = n

    # Compute rolling statistics using pandas for efficiency
    series = pd.Series(chern_series)

    rolling_mean = series.rolling(window=window, min_periods=1).mean().values
    rolling_std = series.rolling(window=window, min_periods=1).std().values

    # Z-score relative to rolling statistics
    rolling_zscore = (chern_series - rolling_mean) / (rolling_std + 1e-10)

    # Rolling change
    rolling_change = np.diff(chern_series, prepend=chern_series[0])

    return {
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'rolling_zscore': rolling_zscore,
        'rolling_change': rolling_change
    }


def aggregate_cross_crisis_metrics(
    results: List[CrisisValidationResult]
) -> AggregateMetrics:
    """
    Aggregate metrics across all crisis validations.

    Args:
        results: List of CrisisValidationResult objects

    Returns:
        AggregateMetrics with averaged/aggregated statistics

    Example:
        >>> results = [result_2008, result_2020, result_2022]
        >>> aggregate = aggregate_cross_crisis_metrics(results)
        >>> print(f"Success rate: {aggregate.success_rate:.1%}")
    """
    if not results:
        return AggregateMetrics(
            n_crises_total=0,
            n_crises_validated=0,
            success_rate=0.0,
            avg_delta_chern=0.0,
            avg_t_statistic=0.0,
            avg_p_value=1.0,
            avg_lead_time_days=None,
            avg_precision=0.0,
            avg_recall=0.0,
            avg_f1_score=0.0,
            median_effect_size=0.0
        )

    n_total = len(results)
    n_validated = sum(1 for r in results if r.hypothesis_supported)

    # Collect metrics
    delta_cherns = [abs(r.delta_chern) for r in results]
    t_stats = [r.t_statistic for r in results]
    p_values = [r.p_value for r in results]
    effect_sizes = [r.effect_size for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    f1_scores = [r.f1_score for r in results]

    # Lead times (filter out None values)
    lead_times = [r.lead_time_days for r in results if r.lead_time_days is not None]
    avg_lead_time = np.mean(lead_times) if lead_times else None

    return AggregateMetrics(
        n_crises_total=n_total,
        n_crises_validated=n_validated,
        success_rate=n_validated / n_total,
        avg_delta_chern=np.mean(delta_cherns),
        avg_t_statistic=np.mean(t_stats),
        avg_p_value=np.mean(p_values),
        avg_lead_time_days=avg_lead_time,
        avg_precision=np.mean(precisions),
        avg_recall=np.mean(recalls),
        avg_f1_score=np.mean(f1_scores),
        median_effect_size=np.median(effect_sizes)
    )


def evaluate_hypothesis(
    result: CrisisValidationResult,
    targets: Dict[str, float] = None
) -> Dict[str, bool]:
    """
    Evaluate whether hypothesis targets are met.

    Default targets from the plan:
        - delta_chern: |Δ| > 0.1
        - t_statistic: > 2.0
        - p_value: < 0.05
        - lead_time_days: > 5 days
        - precision: > 0.5
        - recall: > 0.8

    Args:
        result: CrisisValidationResult to evaluate
        targets: Optional custom targets (uses defaults if None)

    Returns:
        Dictionary mapping metric name to whether target was met

    Example:
        >>> evaluation = evaluate_hypothesis(result_2008)
        >>> print(f"All targets met: {all(evaluation.values())}")
    """
    if targets is None:
        targets = {
            'delta_chern_min': 0.1,
            't_statistic_min': 2.0,
            'p_value_max': 0.05,
            'lead_time_min': 5,
            'precision_min': 0.5,
            'recall_min': 0.8
        }

    evaluation = {
        'delta_chern_met': abs(result.delta_chern) > targets['delta_chern_min'],
        't_statistic_met': result.t_statistic > targets['t_statistic_min'],
        'p_value_met': result.p_value < targets['p_value_max'],
        'precision_met': result.precision > targets['precision_min'],
        'recall_met': result.recall > targets['recall_min']
    }

    # Lead time is optional
    if result.lead_time_days is not None:
        evaluation['lead_time_met'] = result.lead_time_days > targets['lead_time_min']
    else:
        evaluation['lead_time_met'] = False

    evaluation['all_primary_met'] = (
        evaluation['delta_chern_met'] and
        evaluation['t_statistic_met'] and
        evaluation['p_value_met']
    )

    evaluation['all_targets_met'] = all(evaluation.values())

    return evaluation


def compute_effect_size_interpretation(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        effect_size: Cohen's d value

    Returns:
        String interpretation of effect size

    Standard interpretation:
        - < 0.2: Negligible
        - 0.2 - 0.5: Small
        - 0.5 - 0.8: Medium
        - > 0.8: Large
    """
    d = abs(effect_size)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def format_results_summary(
    results: List[CrisisValidationResult],
    aggregate: AggregateMetrics
) -> str:
    """
    Format results as a human-readable summary.

    Args:
        results: List of individual crisis results
        aggregate: Aggregate metrics

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 70)
    lines.append("QCML CRISIS VALIDATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Individual results
    for r in results:
        status = "✓ SUPPORTED" if r.hypothesis_supported else "✗ NOT SUPPORTED"
        lines.append(f"Crisis: {r.crisis_name} ({r.crisis_date})")
        lines.append(f"  Status: {status}")
        lines.append(f"  ΔChern: {r.delta_chern:.4f} (t={r.t_statistic:.2f}, p={r.p_value:.4f})")
        lines.append(f"  Effect Size: {r.effect_size:.2f} ({compute_effect_size_interpretation(r.effect_size)})")
        if r.lead_time_days:
            lines.append(f"  Lead Time: {r.lead_time_days} days")
        lines.append(f"  Detection: P={r.precision:.2f}, R={r.recall:.2f}, F1={r.f1_score:.2f}")
        lines.append("")

    # Aggregate
    lines.append("-" * 70)
    lines.append("AGGREGATE METRICS")
    lines.append("-" * 70)
    lines.append(f"Crises Validated: {aggregate.n_crises_validated}/{aggregate.n_crises_total}")
    lines.append(f"Success Rate: {aggregate.success_rate:.1%}")
    lines.append(f"Avg t-statistic: {aggregate.avg_t_statistic:.2f}")
    lines.append(f"Avg p-value: {aggregate.avg_p_value:.4f}")
    lines.append(f"Avg |ΔChern|: {aggregate.avg_delta_chern:.4f}")
    if aggregate.avg_lead_time_days:
        lines.append(f"Avg Lead Time: {aggregate.avg_lead_time_days:.1f} days")
    lines.append(f"Avg Precision: {aggregate.avg_precision:.2f}")
    lines.append(f"Avg Recall: {aggregate.avg_recall:.2f}")
    lines.append(f"Avg F1: {aggregate.avg_f1_score:.2f}")
    lines.append("")

    # Verdict
    lines.append("=" * 70)
    if aggregate.success_rate >= 0.67:  # 2/3 crises
        lines.append("OVERALL VERDICT: HYPOTHESIS SUPPORTED")
    elif aggregate.success_rate >= 0.33:  # 1/3 crises
        lines.append("OVERALL VERDICT: HYPOTHESIS PARTIALLY SUPPORTED")
    else:
        lines.append("OVERALL VERDICT: HYPOTHESIS NOT SUPPORTED")
    lines.append("=" * 70)

    return "\n".join(lines)
