"""
Statistical evaluation utilities for regime detection experiments.

Provides:
    compute_cohens_d_with_ci  — Cohen's d with bootstrap CI (iid or block)
    cliffs_delta              — Cliff's delta (non-parametric effect size)
    compute_effect_sizes      — Cohen's d + Cliff's delta in one call
    welch_t_test              — Welch's unequal-variance t-test
    holm_bonferroni_correction — Multiple comparison correction
    bh_fdr_correction         — Benjamini-Hochberg FDR correction
    permutation_test          — Permutation test for mean difference
    bayes_factor              — Bayesian hypothesis test (BF10)
    friedman_test             — Friedman rank test with Iman-Davenport F-correction
    nemenyi_posthoc           — Pairwise Nemenyi tests after Friedman (CD diagram)
    compute_detection_metrics — Delay, FAR, precision, recall, F1
"""

import numpy as np
from scipy import stats


def _block_bootstrap_sample(data, block_size, n_out, rng):
    """Draw a circular block bootstrap sample (vectorized).

    Args:
        data: 1-D array of observations.
        block_size: Length of contiguous blocks.
        n_out: Desired output length.
        rng: numpy random Generator.

    Returns:
        1-D array of length n_out.
    """
    n = len(data)
    n_blocks = int(np.ceil(n_out / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    # Vectorized: build all indices at once
    offsets = np.arange(block_size)  # (block_size,)
    all_indices = (starts[:, None] + offsets[None, :]) % n  # (n_blocks, block_size)
    return data[all_indices.ravel()[:n_out]]


def _batch_block_bootstrap(data, block_size, n_out, n_bootstrap, rng):
    """Generate all block bootstrap samples at once (fully vectorized).

    Args:
        data: 1-D array of observations.
        block_size: Length of contiguous blocks.
        n_out: Desired output length per sample.
        n_bootstrap: Number of bootstrap samples.
        rng: numpy random Generator.

    Returns:
        2-D array of shape (n_bootstrap, n_out).
    """
    n = len(data)
    n_blocks = int(np.ceil(n_out / block_size))
    # All start positions: (n_bootstrap, n_blocks)
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)  # (block_size,)
    # All indices: (n_bootstrap, n_blocks, block_size)
    all_indices = (starts[:, :, None] + offsets[None, None, :]) % n
    # Reshape to (n_bootstrap, n_blocks * block_size) and truncate
    flat_indices = all_indices.reshape(n_bootstrap, -1)[:, :n_out]
    return data[flat_indices]


def _optimal_block_size(n):
    """Automatic block size via Politis & White (2004) rule of thumb.

    Uses n^(1/3) as the default, which is the theoretical optimal rate
    for the circular block bootstrap under weak dependence.

    Args:
        n: Sample size.

    Returns:
        Block size (int, >= 1).
    """
    return max(1, int(np.round(n ** (1.0 / 3.0))))


def compute_cohens_d_with_ci(
    crisis_scores, normal_scores, n_bootstrap=10000, seed=42, method="block",
    block_size=None,
):
    """Compute Cohen's d with bootstrap confidence interval.

    Args:
        crisis_scores: 1-D array of scores during crisis.
        normal_scores: 1-D array of scores during normal periods.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        method: Bootstrap method — 'iid' for standard iid resampling,
            'block' for circular block bootstrap (Politis & White 2004).
            Default 'block' to account for serial correlation in time series.
        block_size: Block size for block bootstrap. If None, uses
            automatic rule: max(1, round(n^(1/3))).

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

    if method == "block":
        bs_c = block_size or _optimal_block_size(n_c)
        bs_n = block_size or _optimal_block_size(n_n)
        # Batch block bootstrap: generate all samples at once
        c_samples = _batch_block_bootstrap(crisis_scores, bs_c, n_c, n_bootstrap, rng)
        n_samples = _batch_block_bootstrap(normal_scores, bs_n, n_n, n_bootstrap, rng)
        for i in range(n_bootstrap):
            boot_ds[i] = _cohens_d(c_samples[i], n_samples[i])
    else:
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


def cliffs_delta(group1, group2):
    """Compute Cliff's delta — a non-parametric effect size.

    Cliff's delta measures the degree of overlap between two distributions
    without assuming normality. Unlike Cohen's d, it is robust to outliers
    and skewed distributions.

    Args:
        group1: 1-D array (e.g., crisis scores).
        group2: 1-D array (e.g., normal scores).

    Returns:
        (delta, label): delta in [-1, 1] and qualitative label per
        Romano et al. (2006) thresholds:
            |d| < 0.147 → 'negligible'
            |d| < 0.330 → 'small'
            |d| < 0.474 → 'medium'
            |d| >= 0.474 → 'large'
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan, "negligible"

    # Count dominance pairs
    more = 0
    less = 0
    for x in group1:
        more += np.sum(x > group2)
        less += np.sum(x < group2)

    delta = (more - less) / (n1 * n2)

    abs_d = abs(delta)
    if abs_d < 0.147:
        label = "negligible"
    elif abs_d < 0.330:
        label = "small"
    elif abs_d < 0.474:
        label = "medium"
    else:
        label = "large"

    return delta, label


def compute_effect_sizes(
    crisis_scores, normal_scores, n_bootstrap=10000, seed=42, method="block",
    block_size=None,
):
    """Compute both Cohen's d (with CI) and Cliff's delta.

    Convenience wrapper returning both parametric and non-parametric
    effect sizes in a single call.

    Args:
        crisis_scores: 1-D array of scores during crisis.
        normal_scores: 1-D array of scores during normal periods.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        method: Bootstrap method ('iid' or 'block').
        block_size: Block size for block bootstrap (None = automatic).

    Returns:
        dict with keys: d, ci_lo, ci_hi, cliff_d, cliff_label.
    """
    d, ci_lo, ci_hi = compute_cohens_d_with_ci(
        crisis_scores, normal_scores,
        n_bootstrap=n_bootstrap, seed=seed, method=method,
        block_size=block_size,
    )
    cliff_d, cliff_label = cliffs_delta(crisis_scores, normal_scores)
    return {
        "d": d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "cliff_d": cliff_d,
        "cliff_label": cliff_label,
    }


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


def bh_fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Controls the False Discovery Rate at level alpha. Less conservative
    than Holm-Bonferroni (which controls FWER), appropriate when testing
    many hypotheses (e.g., 36 methods x 17 crises = 612 tests).

    Args:
        p_values: List or array of p-values.
        alpha: Target FDR level (default 0.05).

    Returns:
        adjusted_p: Array of BH-adjusted p-values.
        rejected: Boolean array indicating which hypotheses are rejected.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)

    for rank_idx, orig_idx in enumerate(order):
        rank = rank_idx + 1  # 1-based rank
        adjusted[orig_idx] = p_values[orig_idx] * m / rank

    # Enforce monotonicity (working backwards from largest to smallest)
    for i in range(m - 2, -1, -1):
        idx = order[i]
        idx_next = order[i + 1]
        adjusted[idx] = min(adjusted[idx], adjusted[idx_next])

    adjusted = np.minimum(adjusted, 1.0)
    rejected = adjusted < alpha
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
    """Friedman rank test with Iman-Davenport F-correction.

    The chi-squared approximation used by the standard Friedman test can
    be poor when the number of groups (crises) is small. The Iman-Davenport
    correction transforms chi_sq into an F-statistic with better small-sample
    properties.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.

    Returns:
        dict with keys:
            chi_sq: Friedman chi-squared statistic.
            chi_p: p-value from chi-squared approximation.
            f_stat: Iman-Davenport F-statistic.
            f_p: p-value from F distribution (primary, more accurate).
            mean_ranks: Mean rank per method (rank 1 = highest d).

    For backward compatibility, can also be unpacked as (chi_sq, p_value, mean_ranks)
    where p_value is the F-corrected p-value.
    """
    d_matrix = np.asarray(d_matrix, dtype=float)

    # Remove rows with any NaN
    valid_rows = ~np.any(np.isnan(d_matrix), axis=1)
    d_clean = d_matrix[valid_rows]

    nan_ranks = np.full(d_matrix.shape[1], np.nan)
    if d_clean.shape[0] < 3:
        return np.nan, np.nan, nan_ranks

    n = d_clean.shape[0]  # number of blocks (crises)
    k = d_clean.shape[1]  # number of treatments (methods)

    chi_sq, chi_p = stats.friedmanchisquare(
        *[d_clean[:, j] for j in range(k)]
    )

    # Iman-Davenport F-correction
    denom = n * (k - 1) - chi_sq
    if abs(denom) < 1e-12:
        f_stat = np.inf
        f_p = 0.0
    else:
        f_stat = ((n - 1) * chi_sq) / denom
        df1 = k - 1
        df2 = (k - 1) * (n - 1)
        f_p = 1 - stats.f.cdf(f_stat, df1, df2)

    # Compute mean ranks (rank 1 = highest d)
    ranks = np.zeros_like(d_clean)
    for i in range(n):
        ranks[i] = stats.rankdata(-d_clean[i])

    mean_ranks = ranks.mean(axis=0)
    return chi_sq, f_p, mean_ranks


def nemenyi_posthoc(d_matrix, method_names, alpha=0.05):
    """Nemenyi post-hoc test after a significant Friedman test.

    Computes the critical difference (CD) for pairwise rank comparisons
    and identifies which method pairs differ significantly.

    Args:
        d_matrix: np.ndarray, shape (n_crises, n_methods). Cohen's d values.
        method_names: list of str. Method names corresponding to columns.
        alpha: float. Significance level (default 0.05).

    Returns:
        dict with keys:
            cd: float. Critical difference at the given alpha.
            n_crises: int. Number of crises used.
            n_methods: int. Number of methods.
            significant_pairs: list of (method_a, method_b, rank_diff) tuples
                where |mean_rank_a - mean_rank_b| > CD.
            n_significant: int. Number of significant pairs.
            n_total_pairs: int. Total number of pairwise comparisons.
            mean_ranks: dict mapping method name to mean rank.
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    valid_rows = ~np.any(np.isnan(d_matrix), axis=1)
    d_clean = d_matrix[valid_rows]

    n = d_clean.shape[0]  # crises
    k = d_clean.shape[1]  # methods

    if n < 3 or k < 2:
        return {
            'cd': np.nan, 'n_crises': n, 'n_methods': k,
            'significant_pairs': [], 'n_significant': 0,
            'n_total_pairs': 0, 'mean_ranks': {},
        }

    # Compute ranks (rank 1 = highest d, matching friedman_test convention)
    ranks = np.zeros_like(d_clean)
    for i in range(n):
        ranks[i] = stats.rankdata(-d_clean[i])
    mean_ranks = ranks.mean(axis=0)

    # Nemenyi critical difference: CD = q_alpha * sqrt(k*(k+1) / (6*n))
    # q_alpha comes from the studentized range distribution divided by sqrt(2)
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    # Identify significant pairs
    significant_pairs = []
    n_total = k * (k - 1) // 2
    for i in range(k):
        for j in range(i + 1, k):
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            if rank_diff > cd:
                significant_pairs.append((
                    method_names[i], method_names[j], round(float(rank_diff), 2)
                ))

    mean_rank_dict = {method_names[i]: round(float(mean_ranks[i]), 2) for i in range(k)}

    return {
        'cd': round(float(cd), 2),
        'n_crises': int(n),
        'n_methods': int(k),
        'significant_pairs': significant_pairs,
        'n_significant': len(significant_pairs),
        'n_total_pairs': n_total,
        'mean_ranks': mean_rank_dict,
    }


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
