"""
Statistical evaluation utilities for regime detection experiments.

Provides:
    compute_cohens_d_with_ci  — Cohen's d with bootstrap CI
    welch_t_test              — Welch's unequal-variance t-test
    holm_bonferroni_correction — Multiple comparison correction
    permutation_test          — Permutation test for mean difference
    bayes_factor              — Bayesian hypothesis test (BF10)
    friedman_test             — Friedman rank test for multi-method comparison
    compute_detection_metrics — Delay, FAR, precision, recall, F1
"""

import numpy as np
from scipy import stats


def compute_cohens_d_with_ci(crisis_scores, normal_scores, n_bootstrap=10000, seed=42):
    """Compute Cohen's d with bootstrap confidence interval.

    Args:
        crisis_scores: 1-D array of scores during crisis.
        normal_scores: 1-D array of scores during normal periods.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        (d, ci_lo, ci_hi): Point estimate and 95% CI bounds.
    """
    crisis_scores = np.asarray(crisis_scores, dtype=float)
    normal_scores = np.asarray(normal_scores, dtype=float)

    crisis_scores = crisis_scores[~np.isnan(crisis_scores)]
    normal_scores = normal_scores[~np.isnan(normal_scores)]

    n_c, n_n = len(crisis_scores), len(normal_scores)
    if n_c < 2 or n_n < 2:
        return np.nan, np.nan, np.nan

    d = _cohens_d(crisis_scores, normal_scores)

    rng = np.random.default_rng(seed)
    boot_ds = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        c_boot = rng.choice(crisis_scores, size=n_c, replace=True)
        n_boot = rng.choice(normal_scores, size=n_n, replace=True)
        boot_ds[i] = _cohens_d(c_boot, n_boot)

    ci_lo, ci_hi = np.percentile(boot_ds, [2.5, 97.5])
    return d, ci_lo, ci_hi


def _cohens_d(group1, group2):
    """Compute Cohen's d (absolute value) between two groups."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return abs(np.mean(group1) - np.mean(group2)) / pooled_std


def welch_t_test(group1, group2):
    """Welch's t-test (unequal variance).

    Returns:
        (t_stat, p_value)
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_val


def holm_bonferroni_correction(p_values):
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Args:
        p_values: List or array of p-values.

    Returns:
        adjusted_p: Array of adjusted p-values.
        rejected: Boolean array indicating which hypotheses are rejected at alpha=0.05.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)

    for rank, idx in enumerate(order):
        adjusted[idx] = min(p_values[idx] * (m - rank), 1.0)

    # Enforce monotonicity
    for i in range(1, m):
        idx = order[i]
        idx_prev = order[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[idx_prev])

    rejected = adjusted < 0.05
    return adjusted, rejected


def permutation_test(group1, group2, n_permutations=5000, seed=42):
    """Permutation test for difference in means.

    Args:
        group1, group2: Arrays to compare.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        (observed_diff, p_value)
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    observed_diff = abs(np.mean(group1) - np.mean(group2))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        if perm_diff >= observed_diff:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed_diff, p_value


def bayes_factor(group1, group2, r=0.707):
    """Compute Bayes factor (BF10) for two-sample comparison.

    Uses Rouder et al. (2009) default prior (Cauchy, r=0.707).
    Approximation via BIC.

    Args:
        group1, group2: Arrays to compare.
        r: Prior width (Cauchy scale).

    Returns:
        bf10: Bayes factor in favor of H1 (means differ).
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    n1, n2 = len(group1), len(group2)
    n = n1 + n2

    # Effect size
    d = _cohens_d(group1, group2)

    # BIC approximation: BF10 ~ sqrt(n) * exp(t^2 / 2)
    # More precise: use Savage-Dickey density ratio
    t_stat, _ = stats.ttest_ind(group1, group2, equal_var=False)
    df = n - 2

    # JZS Bayes factor approximation
    t2 = t_stat ** 2
    bf10 = np.sqrt(1 + n / (r ** 2)) * ((1 + t2 / df) ** (-(df + 1) / 2)) / \
           ((1 + t2 / (df * (1 + n / (r ** 2)))) ** (-(df + 1) / 2))

    return max(bf10, 1e-300)


def friedman_test(d_matrix):
    """Friedman rank test for comparing multiple methods.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.

    Returns:
        (chi_sq, p_value, mean_ranks): Test statistic, p-value, and
        mean rank per method (lower rank = higher d).
    """
    d_matrix = np.asarray(d_matrix, dtype=float)

    # Remove rows with any NaN
    valid_rows = ~np.any(np.isnan(d_matrix), axis=1)
    d_clean = d_matrix[valid_rows]

    if d_clean.shape[0] < 3:
        return np.nan, np.nan, np.full(d_matrix.shape[1], np.nan)

    chi_sq, p_value = stats.friedmanchisquare(*[d_clean[:, j] for j in range(d_clean.shape[1])])

    # Compute mean ranks (rank 1 = highest d)
    ranks = np.zeros_like(d_clean)
    for i in range(d_clean.shape[0]):
        ranks[i] = stats.rankdata(-d_clean[i])  # negative: higher d gets rank 1

    mean_ranks = ranks.mean(axis=0)
    return chi_sq, p_value, mean_ranks


def compute_detection_metrics(scores, threshold, crisis_mask, lead_time_days=None):
    """Compute detection performance metrics.

    Args:
        scores: 1-D array of regime scores.
        threshold: Scalar threshold (score > threshold = alarm).
        crisis_mask: Boolean array (True = crisis period).
        lead_time_days: If provided, count alarm within this many days before
            crisis start as true positive (not false alarm).

    Returns:
        dict with keys: detection_delay, false_alarm_rate, precision, recall, F1.
    """
    scores = np.asarray(scores, dtype=float)
    crisis_mask = np.asarray(crisis_mask, dtype=bool)
    T = len(scores)

    alarm = scores > threshold

    # True positives: alarm during crisis
    tp = np.sum(alarm & crisis_mask)
    # False positives: alarm outside crisis
    fp = np.sum(alarm & ~crisis_mask)
    # False negatives: crisis without alarm
    fn = np.sum(~alarm & crisis_mask)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Detection delay: days from crisis start to first alarm within crisis
    crisis_indices = np.where(crisis_mask)[0]
    if len(crisis_indices) > 0:
        crisis_start = crisis_indices[0]
        alarm_in_crisis = np.where(alarm & crisis_mask)[0]
        if len(alarm_in_crisis) > 0:
            detection_delay = alarm_in_crisis[0] - crisis_start
        else:
            detection_delay = np.nan
    else:
        detection_delay = np.nan

    # False alarm rate: fraction of non-crisis days with alarm
    n_normal = np.sum(~crisis_mask)
    false_alarm_rate = fp / n_normal if n_normal > 0 else 0.0

    return {
        'detection_delay': detection_delay,
        'false_alarm_rate': false_alarm_rate,
        'precision': precision,
        'recall': recall,
        'F1': f1,
    }
