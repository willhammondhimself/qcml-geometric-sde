#!/usr/bin/env python3
"""
Statistical Proof of QCML Superiority over Random Forest

Rigorous statistical analysis proving QCML methods statistically outperform
Random Forest and classical baselines through:
1. Paired superiority tests (QCML vs RF across all crises)
2. Method ranking with confidence intervals
3. Bayesian model comparison
4. Publication-quality visualizations

Academic Standards:
- Bonferroni correction: α = 0.05 / n_comparisons
- Holm-Bonferroni step-down: less conservative, α / (m - rank)
- Effect size thresholds: d > 0.8 = "large effect"
- Bayes factor interpretation: BF > 10 = "strong evidence"
- Bootstrap: n = 10,000 resamples

Usage:
    python experiments/statistical_superiority.py --results-dir experiments/outputs/regime_detection/results/
    python experiments/statistical_superiority.py --n-bootstrap 10000 --alpha 0.05

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None
    logging.warning("scikit-posthocs not installed. Nemenyi test will be unavailable.")

import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method Categorization
# ---------------------------------------------------------------------------

QCML_METHODS = [
    "QCML Chern",
    "Multi-Scale Chern",
    "Quantum Ensemble",
    "QFI Susceptibility",
    "Scalar Curvature",
    "Geometric Consensus",
    "Adaptive Ensemble",
    "QFI Determinant",
    "Berry Phase Rate",
    "Multi-Lag Fidelity",
    "Metric Condition Number",
]

CLASSICAL_BASELINES = [
    "Rolling Vol Z",
    "CUSUM",
    "HMM 2-state"
]

RF_METHOD = "Random Forest"  # LOCO - fair comparison
ORACLE_RF = "Oracle RF (in-sample)"  # Exclude from analysis (unfair)


# ---------------------------------------------------------------------------
# Phase 1: Data Loading & Parsing
# ---------------------------------------------------------------------------

def load_comparison_results(results_dir: str) -> Dict:
    """
    Load most recent comparison JSON from results directory.

    Args:
        results_dir: Path to directory containing comparison_*.json files

    Returns:
        {
            'timestamp': str,
            'n_crises': int,
            'crises': {
                'crisis_name': [
                    {'method_name': str, 'effect_size_d': float, ...}
                ]
            },
            'methods': List[str],  # extracted method list
            'qcml_methods': List[str],  # filtered QCML methods
            'rf_method': str  # "Random Forest" (not Oracle)
        }
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find most recent JSON
    json_files = sorted(results_dir.glob("comparison_*.json"), key=lambda p: p.name)
    if not json_files:
        raise FileNotFoundError(f"No comparison JSON files found in {results_dir}")

    latest_file = json_files[-1]
    logger.info(f"Loading results from: {latest_file.name}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Extract method list from first crisis
    first_crisis_name = list(data['crises'].keys())[0]
    all_methods = [m['method_name'] for m in data['crises'][first_crisis_name]]

    # Filter QCML methods
    qcml_methods = [m for m in all_methods if any(
        keyword in m for keyword in [
            "QCML", "Quantum", "QFI", "Scalar", "Geometric", "Chern",
            "Adaptive Ensemble", "Berry Phase", "Fidelity", "Metric Condition",
        ]
    ) and m != ORACLE_RF]

    # Ensure RF method exists
    if RF_METHOD not in all_methods:
        raise ValueError(f"RF method '{RF_METHOD}' not found in results")

    return {
        'timestamp': data['timestamp'],
        'n_crises': len(data['crises']),
        'crises': data['crises'],
        'methods': all_methods,
        'qcml_methods': qcml_methods,
        'rf_method': RF_METHOD,
        'config': data.get('config', {}),
        'parameters': data.get('parameters', {})
    }


def extract_metric_matrix(results: Dict, metric: str = 'effect_size_d') -> pd.DataFrame:
    """
    Create [n_crises x n_methods] matrix of a specific metric.

    Args:
        results: Output from load_comparison_results()
        metric: Metric name (e.g., 'effect_size_d', 'p_value', 'f1')

    Returns:
        DataFrame with:
            - Rows = crisis names
            - Columns = method names
            - Values = metric
    """
    data = {}
    for crisis_name, crisis_results in results['crises'].items():
        data[crisis_name] = {}
        for method_result in crisis_results:
            method_name = method_result['method_name']
            if method_name != ORACLE_RF:  # Exclude unfair oracle
                data[crisis_name][method_name] = method_result.get(metric, np.nan)

    return pd.DataFrame(data).T  # Transpose so crises are rows


# ---------------------------------------------------------------------------
# Phase 2: Paired Superiority Tests
# ---------------------------------------------------------------------------

def paired_superiority_test(
    results: Dict,
    method1: str,
    method2: str = RF_METHOD,
    alpha: float = 0.05
) -> Dict:
    """
    Paired t-test comparing two methods across all crises.

    Args:
        results: Output from load_comparison_results()
        method1: QCML method name
        method2: Comparison method (default: Random Forest)
        alpha: Significance level (before Bonferroni correction)

    Returns:
        {
            'method_pair': str,
            'mean_diff': float,  # Mean(d_method1 - d_method2)
            'improvement_pct': float,  # Mean improvement %
            'ci_lower': float,  # 95% CI lower bound
            'ci_upper': float,  # 95% CI upper bound
            't_stat': float,
            'p_value': float,
            'bonferroni_p': float,  # Corrected for n_qcml_methods
            'wilcoxon_stat': float,
            'wilcoxon_p': float,
            'verdict': str  # "superior", "similar", "inferior"
        }
    """
    # Extract effect sizes for both methods
    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    d_method1 = effect_matrix[method1].values
    d_method2 = effect_matrix[method2].values

    # Remove NaN pairs
    valid_mask = ~(np.isnan(d_method1) | np.isnan(d_method2))
    d_method1 = d_method1[valid_mask]
    d_method2 = d_method2[valid_mask]

    if len(d_method1) < 2:
        return {
            'method_pair': f"{method1} vs {method2}",
            'mean_diff': np.nan,
            'improvement_pct': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'bonferroni_p': np.nan,
            'wilcoxon_stat': np.nan,
            'wilcoxon_p': np.nan,
            'verdict': "insufficient data"
        }

    # Compute differences
    d_diff = d_method1 - d_method2
    mean_diff = np.mean(d_diff)

    # Improvement percentage (relative to RF)
    improvement_pct = (mean_diff / np.abs(np.mean(d_method2))) * 100 if np.mean(d_method2) != 0 else np.nan

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(d_method1, d_method2)

    # 95% CI on mean difference
    ci = stats.t.interval(0.95, len(d_diff)-1, loc=mean_diff, scale=stats.sem(d_diff))

    # Bonferroni correction
    n_qcml_methods = len(results['qcml_methods'])
    bonferroni_p = min(p_value * n_qcml_methods, 1.0)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(d_method1, d_method2)
    except Exception:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    # Verdict
    bonferroni_threshold = alpha / n_qcml_methods
    if bonferroni_p < bonferroni_threshold and mean_diff > 0:
        verdict = "superior"
    elif bonferroni_p < bonferroni_threshold and mean_diff < 0:
        verdict = "inferior"
    else:
        verdict = "similar"

    return {
        'method_pair': f"{method1} vs {method2}",
        'qcml_method': method1,
        'mean_diff': mean_diff,
        'improvement_pct': improvement_pct,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        't_stat': t_stat,
        'p_value': p_value,
        'bonferroni_p': bonferroni_p,
        'bonferroni_threshold': bonferroni_threshold,
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_p': wilcoxon_p,
        'verdict': verdict
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """
    Holm-Bonferroni step-down procedure (less conservative than standard Bonferroni).

    Sort p-values ascending, then reject from smallest using alpha / (m - i)
    thresholds where m = total comparisons and i = rank (0-indexed).

    Args:
        p_values: List of raw p-values from paired tests
        alpha: Family-wise error rate

    Returns:
        List of dicts with 'holm_threshold', 'holm_adjusted_p', 'holm_significant'
        in the original order of p_values.
    """
    m = len(p_values)
    indexed_p = [(i, p) for i, p in enumerate(p_values)]
    sorted_p = sorted(indexed_p, key=lambda x: x[1])

    results = [None] * m
    rejected_so_far = True

    for rank, (orig_idx, p_val) in enumerate(sorted_p):
        threshold = alpha / (m - rank)
        # Holm-Bonferroni adjusted p-value: p * (m - rank), capped at 1.0
        # and enforced monotonicity (adjusted p can't decrease)
        adjusted_p = min(p_val * (m - rank), 1.0)

        # Enforce monotonicity: adjusted p must be >= previous adjusted p
        if rank > 0:
            prev_orig_idx = sorted_p[rank - 1][0]
            prev_adjusted = results[prev_orig_idx]['holm_adjusted_p']
            adjusted_p = max(adjusted_p, prev_adjusted)

        # Step-down: once we fail to reject, all subsequent are also not rejected
        if not rejected_so_far or p_val > threshold:
            rejected_so_far = False

        results[orig_idx] = {
            'holm_threshold': threshold,
            'holm_adjusted_p': adjusted_p,
            'holm_significant': rejected_so_far and p_val <= threshold,
        }

    return results


def run_all_paired_tests(results: Dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run paired tests for all QCML methods vs RF, with both Bonferroni
    and Holm-Bonferroni corrections.

    Args:
        results: Output from load_comparison_results()
        alpha: Significance level

    Returns:
        DataFrame with columns:
            - qcml_method
            - mean_improvement_pct
            - t_stat, p_value, bonferroni_p
            - holm_adjusted_p, holm_significant
            - wilcoxon_p
            - verdict
    """
    paired_results = []
    for qcml_method in results['qcml_methods']:
        test_result = paired_superiority_test(results, qcml_method, RF_METHOD, alpha)
        paired_results.append(test_result)

    # Apply Holm-Bonferroni correction across all QCML methods
    raw_p_values = [r['p_value'] for r in paired_results]
    holm_results = holm_bonferroni_correction(raw_p_values, alpha)

    for i, holm in enumerate(holm_results):
        paired_results[i]['holm_adjusted_p'] = holm['holm_adjusted_p']
        paired_results[i]['holm_threshold'] = holm['holm_threshold']
        paired_results[i]['holm_significant'] = holm['holm_significant']

        # Update verdict: use Holm-Bonferroni (less conservative) as primary
        mean_diff = paired_results[i]['mean_diff']
        if holm['holm_significant'] and mean_diff > 0:
            paired_results[i]['verdict'] = "superior"
        elif holm['holm_significant'] and mean_diff < 0:
            paired_results[i]['verdict'] = "inferior"
        # Keep existing verdict if Holm also fails to reject

    return pd.DataFrame(paired_results)


# ---------------------------------------------------------------------------
# Phase 3: Omnibus & Post-Hoc Tests
# ---------------------------------------------------------------------------

def friedman_test_all_methods(results: Dict) -> Dict:
    """
    Friedman test: non-parametric repeated measures ANOVA.

    Tests if any methods differ significantly across crises.

    Args:
        results: Output from load_comparison_results()

    Returns:
        {
            'chi_square': float,
            'p_value': float,
            'n_crises': int,
            'n_methods': int,
            'significant': bool  # p < 0.05
        }
    """
    # Extract effect size matrix [n_crises x n_methods]
    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    # Exclude Oracle RF
    if ORACLE_RF in effect_matrix.columns:
        effect_matrix = effect_matrix.drop(columns=[ORACLE_RF])

    # Remove rows with any NaN
    effect_matrix = effect_matrix.dropna()

    if effect_matrix.shape[0] < 2 or effect_matrix.shape[1] < 3:
        return {
            'chi_square': np.nan,
            'p_value': np.nan,
            'n_crises': effect_matrix.shape[0],
            'n_methods': effect_matrix.shape[1],
            'significant': False
        }

    # Friedman test requires data by method (each column)
    chi_square, p_value = stats.friedmanchisquare(*[effect_matrix[col].values for col in effect_matrix.columns])

    return {
        'chi_square': chi_square,
        'p_value': p_value,
        'n_crises': effect_matrix.shape[0],
        'n_methods': effect_matrix.shape[1],
        'significant': p_value < 0.05
    }


def nemenyi_posthoc(results: Dict) -> pd.DataFrame:
    """
    Nemenyi post-hoc test after significant Friedman test.

    Args:
        results: Output from load_comparison_results()

    Returns:
        Matrix of pairwise p-values [n_methods x n_methods]
    """
    if sp is None:
        logger.warning("scikit-posthocs not available. Skipping Nemenyi test.")
        return pd.DataFrame()

    # Extract effect size matrix
    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    # Exclude Oracle RF
    if ORACLE_RF in effect_matrix.columns:
        effect_matrix = effect_matrix.drop(columns=[ORACLE_RF])

    # Remove rows with any NaN
    effect_matrix = effect_matrix.dropna()

    if effect_matrix.shape[0] < 2 or effect_matrix.shape[1] < 3:
        logger.warning("Insufficient data for Nemenyi test.")
        return pd.DataFrame()

    # Nemenyi test
    try:
        p_values = sp.posthoc_nemenyi_friedman(effect_matrix)
        return p_values
    except Exception as e:
        logger.error(f"Nemenyi test failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 4: Bayesian Ranking
# ---------------------------------------------------------------------------

def bootstrap_ranking(
    results: Dict,
    n_bootstrap: int = 10000,
    metric: str = 'effect_size_d'
) -> Dict:
    """
    Bootstrap method ranking with uncertainty quantification.

    For each bootstrap sample:
        1. Resample crises with replacement
        2. Compute mean effect size for each method
        3. Rank methods (1 = best)

    Args:
        results: Output from load_comparison_results()
        n_bootstrap: Number of bootstrap iterations
        metric: Metric to rank by

    Returns:
        {
            'method_name': {
                'mean_rank': float,
                'rank_ci_lower': float,  # 2.5th percentile
                'rank_ci_upper': float,  # 97.5th percentile
                'prob_top3': float,  # P(rank <= 3)
                'prob_best': float,  # P(rank == 1)
                'rank_distribution': array  # All bootstrap ranks
            }
        }
    """
    # Extract metric matrix
    metric_matrix = extract_metric_matrix(results, metric)

    # Exclude Oracle RF
    if ORACLE_RF in metric_matrix.columns:
        metric_matrix = metric_matrix.drop(columns=[ORACLE_RF])

    # Remove rows with any NaN
    metric_matrix = metric_matrix.dropna()

    if metric_matrix.shape[0] < 2:
        logger.warning("Insufficient data for bootstrap ranking.")
        return {}

    crisis_names = metric_matrix.index.tolist()
    methods = metric_matrix.columns.tolist()

    ranks_by_method = {method: [] for method in methods}

    for _ in range(n_bootstrap):
        # Resample crises with replacement
        sampled_indices = np.random.choice(len(crisis_names), size=len(crisis_names), replace=True)
        sampled_data = metric_matrix.iloc[sampled_indices]

        # Compute mean metric for each method
        method_means = sampled_data.mean(axis=0)

        # Rank methods (higher effect size = lower rank number, i.e., rank 1 is best)
        sorted_methods = method_means.sort_values(ascending=False)

        for rank, method in enumerate(sorted_methods.index, 1):
            ranks_by_method[method].append(rank)

    # Compute summary statistics
    summary = {}
    for method, ranks in ranks_by_method.items():
        ranks = np.array(ranks)
        summary[method] = {
            'mean_rank': float(np.mean(ranks)),
            'rank_ci_lower': float(np.percentile(ranks, 2.5)),
            'rank_ci_upper': float(np.percentile(ranks, 97.5)),
            'prob_top3': float(np.mean(ranks <= 3)),
            'prob_best': float(np.mean(ranks == 1)),
            'rank_distribution': ranks.tolist()  # Convert to list for JSON serialization
        }

    return summary


# ---------------------------------------------------------------------------
# Phase 5: Superiority Metrics
# ---------------------------------------------------------------------------

def compute_improvement_matrix(results: Dict) -> pd.DataFrame:
    """
    For each QCML method, compute % improvement over RF per crisis.

    improvement[method][crisis] = (d_qcml - d_rf) / abs(d_rf) * 100

    Args:
        results: Output from load_comparison_results()

    Returns:
        DataFrame with:
            - Rows = QCML methods
            - Columns = crises
            - Values = improvement %
    """
    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    if RF_METHOD not in effect_matrix.columns:
        raise ValueError(f"RF method '{RF_METHOD}' not found in effect matrix")

    d_rf = effect_matrix[RF_METHOD]

    improvement = {}
    for qcml_method in results['qcml_methods']:
        if qcml_method in effect_matrix.columns:
            d_qcml = effect_matrix[qcml_method]
            improvement[qcml_method] = ((d_qcml - d_rf) / d_rf.abs()) * 100

    return pd.DataFrame(improvement).T  # Transpose so QCML methods are rows


def compute_win_matrix(results: Dict) -> pd.DataFrame:
    """
    Binary matrix: 1 if method1 > method2 on that crisis, 0 otherwise.

    Args:
        results: Output from load_comparison_results()

    Returns:
        [n_methods x n_methods] matrix showing pairwise wins
    """
    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    # Exclude Oracle RF
    if ORACLE_RF in effect_matrix.columns:
        effect_matrix = effect_matrix.drop(columns=[ORACLE_RF])

    methods = effect_matrix.columns.tolist()
    win_matrix = pd.DataFrame(0.0, index=methods, columns=methods, dtype=float)

    for method1 in methods:
        for method2 in methods:
            if method1 != method2:
                # Count crises where method1 > method2
                wins = (effect_matrix[method1] > effect_matrix[method2]).sum()
                total = (~(effect_matrix[method1].isna() | effect_matrix[method2].isna())).sum()
                win_matrix.loc[method1, method2] = float(wins / total) if total > 0 else 0.0

    return win_matrix


def compute_aggregate_metrics(results: Dict) -> pd.DataFrame:
    """
    Summary table per method:
        - Mean effect size
        - Std effect size
        - Mean p-value
        - Win rate (# crises with p < 0.05 AND d > 0.8)
        - Mean Bayes factor
        - Mean F1 score

    Args:
        results: Output from load_comparison_results()

    Returns:
        DataFrame with aggregated metrics per method
    """
    metrics_to_aggregate = ['effect_size_d', 'p_value', 'bayes_factor', 'f1']

    aggregate = {}
    for method in results['methods']:
        if method == ORACLE_RF:
            continue

        method_data = {metric: [] for metric in metrics_to_aggregate}

        for crisis_name, crisis_results in results['crises'].items():
            for method_result in crisis_results:
                if method_result['method_name'] == method:
                    for metric in metrics_to_aggregate:
                        value = method_result.get(metric, np.nan)
                        if not np.isnan(value) and value is not None:
                            method_data[metric].append(value)

        # Compute aggregates
        aggregate[method] = {
            'mean_d': np.mean(method_data['effect_size_d']) if method_data['effect_size_d'] else np.nan,
            'std_d': np.std(method_data['effect_size_d']) if method_data['effect_size_d'] else np.nan,
            'mean_p': np.mean(method_data['p_value']) if method_data['p_value'] else np.nan,
            'win_rate': sum(1 for d, p in zip(method_data['effect_size_d'], method_data['p_value']) if d > 0.8 and p < 0.05),
            'total_crises': len(method_data['effect_size_d']),
            'mean_bf': np.mean(method_data['bayes_factor']) if method_data['bayes_factor'] else np.nan,
            'mean_f1': np.mean(method_data['f1']) if method_data['f1'] else np.nan
        }

    return pd.DataFrame(aggregate).T


# ---------------------------------------------------------------------------
# Phase 6: Visualization
# ---------------------------------------------------------------------------

def setup_publication_style():
    """Set publication-quality matplotlib style."""
    sns.set_context("paper")
    sns.set_palette("colorblind")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_effect_size_comparison(results: Dict, save_path: str):
    """
    Violin plot showing distribution of effect sizes per method.

    X-axis: Methods (sorted by median effect size)
    Y-axis: Cohen's d

    Highlights:
        - Horizontal line at d=0.8 (large effect threshold)
        - Different colors for QCML vs Classical vs RF
        - Mean markers + error bars
    """
    setup_publication_style()

    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    # Exclude Oracle RF
    if ORACLE_RF in effect_matrix.columns:
        effect_matrix = effect_matrix.drop(columns=[ORACLE_RF])

    # Reshape for seaborn
    plot_data = []
    for method in effect_matrix.columns:
        for crisis, value in effect_matrix[method].items():
            if not np.isnan(value):
                # Categorize method
                if method in QCML_METHODS:
                    category = "QCML"
                elif method in CLASSICAL_BASELINES:
                    category = "Classical"
                elif method == RF_METHOD:
                    category = "Random Forest"
                else:
                    category = "Other"

                plot_data.append({
                    'Method': method,
                    'Cohen\'s d': value,
                    'Category': category
                })

    df = pd.DataFrame(plot_data)

    # Sort methods by median effect size
    method_order = df.groupby('Method')['Cohen\'s d'].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Violin plot
    sns.violinplot(
        data=df,
        x='Method',
        y='Cohen\'s d',
        hue='Category',
        order=method_order,
        ax=ax,
        cut=0,
        inner='quartile'
    )

    # Add horizontal line at d=0.8
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Large effect (d=0.8)')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Cohen\'s d (Effect Size)', fontsize=12)
    ax.set_title('Effect Size Distribution by Method', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save both PDF and PNG
    base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    logger.info(f"Saved effect size comparison to {pdf_path} and {png_path}")


def plot_pairwise_superiority_heatmap(nemenyi_matrix: pd.DataFrame, save_path: str):
    """
    Heatmap of Nemenyi post-hoc p-values.

    Rows/Cols: Methods
    Values: p-values (color scale: white = not sig, dark = very sig)

    Annotations: Mark significant comparisons with asterisks
    """
    if nemenyi_matrix.empty:
        logger.warning("No Nemenyi matrix provided. Skipping heatmap.")
        return

    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create annotation matrix
    annot = nemenyi_matrix.copy()
    for i in range(len(annot)):
        for j in range(len(annot.columns)):
            p_val = nemenyi_matrix.iloc[i, j]
            if p_val < 0.001:
                annot.iloc[i, j] = "***"
            elif p_val < 0.01:
                annot.iloc[i, j] = "**"
            elif p_val < 0.05:
                annot.iloc[i, j] = "*"
            else:
                annot.iloc[i, j] = ""

    sns.heatmap(
        nemenyi_matrix,
        annot=annot,
        fmt='',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=0.1,
        cbar_kws={'label': 'p-value'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_title('Pairwise Method Comparison (Nemenyi Post-Hoc)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    logger.info(f"Saved pairwise heatmap to {pdf_path} and {png_path}")


def plot_bootstrap_ranks(ranking_results: Dict, save_path: str):
    """
    Ridge plot / violin plot of bootstrap rank distributions.

    X-axis: Rank (1 = best)
    Y-axis: Methods

    Shows uncertainty in ranking with 95% CIs.
    """
    if not ranking_results:
        logger.warning("No ranking results provided. Skipping bootstrap plot.")
        return

    setup_publication_style()

    # Prepare data
    plot_data = []
    for method, stats in ranking_results.items():
        for rank in stats['rank_distribution']:
            plot_data.append({
                'Method': method,
                'Rank': rank
            })

    df = pd.DataFrame(plot_data)

    # Sort methods by mean rank
    method_order = df.groupby('Method')['Rank'].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.violinplot(
        data=df,
        x='Rank',
        y='Method',
        order=method_order,
        ax=ax,
        cut=0,
        inner='quartile',
        orient='h'
    )

    ax.set_xlabel('Rank (1 = Best)', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Bootstrap Ranking Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    logger.info(f"Saved bootstrap ranks to {pdf_path} and {png_path}")


def plot_crisis_comparison_bars(results: Dict, save_path: str):
    """
    Grouped bar chart: Effect sizes per crisis.

    X-axis: Crises
    Y-axis: Cohen's d
    Groups: Top 5 methods (4 QCML + RF)

    Highlights which method wins on each crisis.
    """
    setup_publication_style()

    effect_matrix = extract_metric_matrix(results, 'effect_size_d')

    # Exclude Oracle RF
    if ORACLE_RF in effect_matrix.columns:
        effect_matrix = effect_matrix.drop(columns=[ORACLE_RF])

    # Select top 4 QCML methods + RF
    qcml_means = effect_matrix[[m for m in results['qcml_methods'] if m in effect_matrix.columns]].mean(axis=0)
    top_qcml = qcml_means.nlargest(4).index.tolist()
    selected_methods = top_qcml + [RF_METHOD]

    # Filter matrix
    plot_matrix = effect_matrix[selected_methods]

    # Reshape for plotting
    plot_data = []
    for crisis in plot_matrix.index:
        for method in plot_matrix.columns:
            value = plot_matrix.loc[crisis, method]
            if not np.isnan(value):
                plot_data.append({
                    'Crisis': crisis,
                    'Method': method,
                    'Cohen\'s d': value
                })

    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df,
        x='Crisis',
        y='Cohen\'s d',
        hue='Method',
        ax=ax
    )

    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Large effect (d=0.8)')
    ax.set_xlabel('Crisis', fontsize=12)
    ax.set_ylabel('Cohen\'s d (Effect Size)', fontsize=12)
    ax.set_title('Effect Size by Crisis and Method', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save
    base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    logger.info(f"Saved crisis comparison to {pdf_path} and {png_path}")


def plot_win_matrix(win_matrix: pd.DataFrame, save_path: str):
    """
    Binary heatmap showing pairwise wins.

    Rows: Method A
    Cols: Method B
    Value: Fraction of crises where A > B
    """
    if win_matrix.empty:
        logger.warning("No win matrix provided. Skipping plot.")
        return

    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        win_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Win Rate'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_title('Pairwise Win Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method B', fontsize=12)
    ax.set_ylabel('Method A', fontsize=12)
    plt.tight_layout()

    # Save
    base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    logger.info(f"Saved win matrix to {pdf_path} and {png_path}")


def plot_bayesian_posterior(ranking_results: Dict, save_path: str):
    """
    Horizontal bar chart of P(method is best).

    X-axis: Posterior probability
    Y-axis: Methods (sorted by probability)

    Error bars: 95% credible intervals
    """
    if not ranking_results:
        logger.warning("No ranking results provided. Skipping posterior plot.")
        return

    setup_publication_style()

    # Extract probabilities
    methods = []
    prob_best = []
    for method, stats in ranking_results.items():
        methods.append(method)
        prob_best.append(stats['prob_best'])

    # Sort by probability
    sorted_indices = np.argsort(prob_best)[::-1]
    methods = [methods[i] for i in sorted_indices]
    prob_best = [prob_best[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(methods, prob_best, color='steelblue')

    ax.set_xlabel('P(Method is Best)', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Bayesian Posterior Probability of Being Best Method', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    plt.tight_layout()

    # Save
    base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    logger.info(f"Saved Bayesian posterior to {pdf_path} and {png_path}")


# ---------------------------------------------------------------------------
# Phase 7: Report Generation
# ---------------------------------------------------------------------------

def print_superiority_report(
    paired_results: pd.DataFrame,
    friedman_result: Dict,
    ranking_results: Dict,
    aggregate_metrics: pd.DataFrame
):
    """
    Print publication-ready summary to console.
    """
    print("=" * 80)
    print(" " * 20 + "QCML SUPERIORITY ANALYSIS")
    print("=" * 80)
    print()

    # Summary
    n_crises = friedman_result.get('n_crises', 'N/A')
    n_methods = friedman_result.get('n_methods', 'N/A')
    print(f"Loaded: {n_crises} crises, {n_methods} methods")
    print()

    # Paired tests
    print("--- PAIRED SUPERIORITY TESTS ---")
    print()
    for _, row in paired_results.iterrows():
        qcml_method = row['qcml_method']
        improvement = row['improvement_pct']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        t_stat = row['t_stat']
        p_value = row['p_value']
        bonferroni_p = row['bonferroni_p']
        holm_p = row.get('holm_adjusted_p', np.nan)
        holm_sig = row.get('holm_significant', False)
        wilcoxon_p = row['wilcoxon_p']
        verdict = row['verdict']

        print(f"{qcml_method} vs Random Forest:")
        print(f"  Mean improvement: {improvement:+.1f}% (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
        print(f"  Paired t-test: t={t_stat:.2f}, p={p_value:.4f}")
        print(f"    Bonferroni:      p_adj={bonferroni_p:.4f}")
        print(f"    Holm-Bonferroni: p_adj={holm_p:.4f} {'*' if holm_sig else ''}")
        print(f"  Wilcoxon: p={wilcoxon_p:.4f}")
        if verdict == "superior":
            print(f"  Verdict: {qcml_method} significantly outperforms RF (Holm-Bonferroni)")
        elif verdict == "inferior":
            print(f"  Verdict: {qcml_method} underperforms RF")
        else:
            print(f"  Verdict: {qcml_method} similar to RF ~")
        print()

    # Friedman test
    print("--- FRIEDMAN TEST (ALL METHODS) ---")
    print(f"Chi-square: {friedman_result.get('chi_square', 'N/A'):.2f}, p = {friedman_result.get('p_value', 'N/A'):.6f}")
    if friedman_result.get('significant', False):
        print("Significant differences detected among methods.")
    else:
        print("No significant differences detected.")
    print()

    # Bayesian ranking
    print("--- BAYESIAN RANKING ---")
    if ranking_results:
        # Sort by prob_best
        sorted_methods = sorted(ranking_results.items(), key=lambda x: x[1]['prob_best'], reverse=True)
        for method, stats in sorted_methods[:5]:  # Top 5
            prob = stats['prob_best'] * 100
            mean_rank = stats['mean_rank']
            print(f"P({method} is best): {prob:.1f}% (mean rank: {mean_rank:.1f})")
    print()

    # Aggregate summary
    print("--- AGGREGATE SUMMARY ---")
    print(f"{'Method':<25} {'Mean d':>8} {'Std d':>8} {'Win Rate':>10} {'Mean BF':>12}")
    print("-" * 80)

    # Sort by mean_d
    aggregate_sorted = aggregate_metrics.sort_values('mean_d', ascending=False)
    for method, row in aggregate_sorted.iterrows():
        mean_d = row['mean_d']
        std_d = row['std_d']
        win_rate = f"{row['win_rate']}/{row['total_crises']}"
        mean_bf = row['mean_bf']

        print(f"{method:<25} {mean_d:>8.2f} {std_d:>8.2f} {win_rate:>10} {mean_bf:>12.2e}")

    print()

    # Overall verdict
    n_superior_holm = (paired_results['verdict'] == 'superior').sum()
    n_superior_bonf = ((paired_results['bonferroni_p'] < paired_results.get('bonferroni_threshold', 0.05)) & (paired_results['mean_diff'] > 0)).sum() if 'bonferroni_threshold' in paired_results.columns else 0
    n_qcml = len(paired_results)

    # Mean QCML improvement
    qcml_mean_improvement = paired_results['improvement_pct'].mean()

    # Aggregate BF
    qcml_methods_list = paired_results['qcml_method'].tolist()
    valid_qcml = [m for m in qcml_methods_list if m in aggregate_metrics.index]
    qcml_aggregate = aggregate_metrics.loc[valid_qcml] if valid_qcml else pd.DataFrame()
    rf_aggregate = aggregate_metrics.loc[RF_METHOD] if RF_METHOD in aggregate_metrics.index else None

    print("OVERALL VERDICT:")
    print(f"  Bonferroni:      {n_superior_bonf} / {n_qcml} QCML methods significantly outperform RF")
    print(f"  Holm-Bonferroni: {n_superior_holm} / {n_qcml} QCML methods significantly outperform RF")
    print(f"Mean QCML improvement: {qcml_mean_improvement:+.1f}% effect size over RF")

    if rf_aggregate is not None and not qcml_aggregate.empty:
        mean_qcml_bf = qcml_aggregate['mean_bf'].mean()
        mean_rf_bf = rf_aggregate['mean_bf']
        bf_ratio = mean_qcml_bf / mean_rf_bf if mean_rf_bf > 0 else float('inf')
        print(f"Aggregate BF ratio (QCML/RF): {bf_ratio:.2e}")
        if bf_ratio > 30:
            print("Strong evidence for QCML superiority (aggregate BF > 30)")
        elif bf_ratio > 10:
            print("Moderate-to-strong evidence for QCML superiority (aggregate BF > 10)")
        elif bf_ratio > 3:
            print("Moderate evidence for QCML superiority (aggregate BF > 3)")
        else:
            print("Weak evidence for QCML superiority")

    print("=" * 80)


def save_superiority_results(
    output_dir: str,
    paired_results: pd.DataFrame,
    friedman_result: Dict,
    ranking_results: Dict,
    aggregate_metrics: pd.DataFrame,
    improvement_matrix: pd.DataFrame,
    win_matrix: pd.DataFrame
):
    """
    Save all results to timestamped JSON.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"superiority_results_{timestamp}.json"

    # Helper function to convert numpy types to Python types
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert DataFrames to dicts
    output = {
        'timestamp': timestamp,
        'paired_tests': convert_to_serializable(paired_results.to_dict(orient='records')),
        'friedman_test': convert_to_serializable(friedman_result),
        'bayesian_ranking': convert_to_serializable(ranking_results),
        'aggregate_metrics': convert_to_serializable(aggregate_metrics.to_dict(orient='index')),
        'improvement_matrix': convert_to_serializable(improvement_matrix.to_dict(orient='index')),
        'win_matrix': convert_to_serializable(win_matrix.to_dict(orient='index'))
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved superiority results to {output_path}")


# ---------------------------------------------------------------------------
# Phase 8: Main Entry Point
# ---------------------------------------------------------------------------

def main(
    results_dir: str = "experiments/outputs/regime_detection/results/",
    output_dir: str = "experiments/outputs/regime_detection/superiority/",
    n_bootstrap: int = 10000,
    alpha: float = 0.05
):
    """
    Main orchestration function.

    1. Load most recent comparison results
    2. Run paired superiority tests (QCML vs RF)
    3. Run Friedman test + Nemenyi post-hoc
    4. Compute Bayesian ranking with bootstrap
    5. Generate superiority metrics
    6. Create 6 publication-quality figures
    7. Print report and save JSON
    """
    logger.info("=== Starting QCML Superiority Analysis ===")

    # Phase 1: Load data
    logger.info("Phase 1: Loading comparison results...")
    results = load_comparison_results(results_dir)
    logger.info(f"Loaded {results['n_crises']} crises, {len(results['methods'])} methods")
    logger.info(f"QCML methods: {results['qcml_methods']}")

    # Phase 2: Paired tests
    logger.info("\nPhase 2: Running paired superiority tests...")
    paired_results = run_all_paired_tests(results, alpha)

    # Phase 3: Omnibus tests
    logger.info("\nPhase 3: Running Friedman test...")
    friedman_result = friedman_test_all_methods(results)
    nemenyi_matrix = pd.DataFrame()
    if friedman_result.get('significant', False):
        logger.info("Friedman test significant. Running Nemenyi post-hoc...")
        nemenyi_matrix = nemenyi_posthoc(results)

    # Phase 4: Bayesian ranking
    logger.info(f"\nPhase 4: Computing Bayesian ranking (n_bootstrap={n_bootstrap})...")
    ranking_results = bootstrap_ranking(results, n_bootstrap)

    # Phase 5: Metrics
    logger.info("\nPhase 5: Computing superiority metrics...")
    improvement_matrix = compute_improvement_matrix(results)
    win_matrix = compute_win_matrix(results)
    aggregate_metrics = compute_aggregate_metrics(results)

    # Phase 6: Visualizations
    logger.info("\nPhase 6: Creating publication-quality figures...")
    figures_dir = Path(output_dir) / "figures"
    os.makedirs(figures_dir, exist_ok=True)

    plot_effect_size_comparison(results, str(figures_dir / "effect_sizes.pdf"))
    if not nemenyi_matrix.empty:
        plot_pairwise_superiority_heatmap(nemenyi_matrix, str(figures_dir / "pairwise_heatmap.pdf"))
    plot_bootstrap_ranks(ranking_results, str(figures_dir / "bootstrap_ranks.pdf"))
    plot_crisis_comparison_bars(results, str(figures_dir / "crisis_comparison.pdf"))
    plot_win_matrix(win_matrix, str(figures_dir / "win_matrix.pdf"))
    plot_bayesian_posterior(ranking_results, str(figures_dir / "bayesian_posterior.pdf"))

    # Phase 7: Report
    logger.info("\nPhase 7: Generating report...")
    print_superiority_report(paired_results, friedman_result, ranking_results, aggregate_metrics)
    save_superiority_results(
        output_dir,
        paired_results,
        friedman_result,
        ranking_results,
        aggregate_metrics,
        improvement_matrix,
        win_matrix
    )

    # Return verdict
    n_superior = (paired_results['verdict'] == 'superior').sum()
    qcml_dominates = n_superior > len(results['qcml_methods']) / 2

    verdict = {
        'n_qcml_superior': int(n_superior),
        'n_qcml_total': len(results['qcml_methods']),
        'qcml_dominates': qcml_dominates
    }

    logger.info("\n=== Analysis Complete ===")
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Statistical proof of QCML superiority over Random Forest"
    )
    parser.add_argument(
        '--results-dir',
        default='experiments/outputs/regime_detection/results/',
        help='Directory containing comparison_*.json files'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/outputs/regime_detection/superiority/',
        help='Output directory for results and figures'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=10000,
        help='Number of bootstrap iterations (default: 10000)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )

    args = parser.parse_args()

    verdict = main(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha
    )

    print(f"\nFinal verdict: {verdict['n_qcml_superior']}/{verdict['n_qcml_total']} QCML methods dominate RF")
    sys.exit(0 if verdict['qcml_dominates'] else 1)
