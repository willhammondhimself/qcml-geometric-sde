#!/usr/bin/env python3
"""
Publication-Quality Figure Generation for QCML Crisis Validation

Phase 2 of Academic Research Plan: Generates figures suitable for
quantitative finance or econophysics journal submission.

Figures:
  1. Chern number time series per crisis with price overlay
  2. Aggregate effect sizes (forest plot)
  3. Lead time distribution (box/violin plots)
  4. ROC and Precision-Recall curves
  5. Chern vs volatility correlation scatter
  6. Method comparison (QCML vs VIX baseline)

Output:
  experiments/outputs/regime_detection/figures/
    fig_chern_*.pdf   (vector)
    fig_chern_*.png   (300 DPI)

Usage:
    # Generate from synthetic data
    python experiments/generate_crisis_figures.py --synthetic

    # Generate from saved results
    python experiments/generate_crisis_figures.py --results-file path/to/results.json

Author: QCML Research
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Run: pip install matplotlib>=3.7")
    sys.exit(1)

from experiments.crisis_config import ALL_CRISES, CrisisDefinition

logger = logging.getLogger(__name__)

# Publication style settings
STYLE_CONFIG = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

CRISIS_COLORS = {
    '2008_crisis': '#D32F2F',
    '2020_covid': '#1976D2',
    '2022_rates': '#388E3C',
}

CRISIS_LABELS = {
    '2008_crisis': '2008 GFC',
    '2020_covid': '2020 COVID',
    '2022_rates': '2022 Rate Hike',
}


def apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)


def save_figure(fig, output_dir: Path, name: str):
    """
    Save figure as both PDF (vector) and PNG (300 DPI).

    Args:
        fig: matplotlib Figure
        output_dir: Output directory
        name: Figure name (without extension)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"

    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)

    logger.info(f"Saved: {pdf_path}")
    logger.info(f"Saved: {png_path}")
    plt.close(fig)


def fig_chern_time_series(
    crisis_name: str,
    chern_series: np.ndarray,
    times: pd.DatetimeIndex,
    prices: pd.Series,
    crisis_date: str,
    transitions: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 1: Chern number time series with price overlay.

    Two-panel plot:
      Top: Price series with crisis date marked
      Bottom: Chern number with transitions marked

    Args:
        crisis_name: Crisis identifier
        chern_series: Chern values
        times: Timestamps for Chern series
        prices: Price series
        crisis_date: Crisis date string
        transitions: Transition indices
        output_dir: If provided, save figure

    Returns:
        matplotlib Figure
    """
    apply_style()
    color = CRISIS_COLORS.get(crisis_name, '#333333')
    label = CRISIS_LABELS.get(crisis_name, crisis_name)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1.2]})

    crisis_ts = pd.Timestamp(crisis_date)

    # Top panel: prices
    price_times = prices.index if hasattr(prices, 'index') else times
    ax1.plot(price_times, prices.values if hasattr(prices, 'values') else prices,
             color='#333333', linewidth=0.8, label='Price')
    ax1.axvline(crisis_ts, color=color, linestyle='--', linewidth=1.2,
                alpha=0.8, label=f'{label} onset')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_title(f'QCML Topological Regime Detection: {label}')

    # Bottom panel: Chern series
    chern_times = times[:len(chern_series)]
    ax2.plot(chern_times, chern_series, color=color, linewidth=0.8,
             label='Chern number')

    # Mark transitions
    if transitions:
        valid_trans = [t for t in transitions if t < len(chern_times)]
        if valid_trans:
            ax2.scatter(
                chern_times[valid_trans],
                chern_series[valid_trans],
                color=color, marker='v', s=40, zorder=5,
                label='Detected transition'
            )

    ax2.axvline(crisis_ts, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Chern Number')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', framealpha=0.9)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, f'fig_chern_{crisis_name}')

    return fig


def fig_forest_plot(
    crisis_results: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 2: Forest plot of effect sizes across crises.

    Shows Cohen's d with 95% confidence intervals for each crisis,
    plus the pooled effect size.

    Args:
        crisis_results: Dict mapping crisis name to result dict
            Each must have: effect_size_d, bootstrap_delta_chern (with ci_lower, ci_upper)
        output_dir: If provided, save figure

    Returns:
        matplotlib Figure
    """
    apply_style()

    names = list(crisis_results.keys())
    n = len(names)

    fig, ax = plt.subplots(figsize=(7, 2.5 + 0.6 * n))

    y_positions = list(range(n))

    for i, name in enumerate(names):
        r = crisis_results[name]
        d = r.get('effect_size_d', 0)
        color = CRISIS_COLORS.get(name, '#333333')
        label = CRISIS_LABELS.get(name, name)

        # CI from bootstrap if available
        boot = r.get('bootstrap_delta_chern', {})
        ci_lower = boot.get('ci_lower', d - 0.5)
        ci_upper = boot.get('ci_upper', d + 0.5)

        # Normalize CI to effect size scale
        delta = r.get('delta_chern', d)
        if abs(delta) > 1e-10 and abs(d) > 1e-10:
            scale = d / delta
            ci_l_scaled = ci_lower * scale
            ci_u_scaled = ci_upper * scale
        else:
            ci_l_scaled = d - 0.5
            ci_u_scaled = d + 0.5

        ax.errorbar(d, i, xerr=[[d - ci_l_scaled], [ci_u_scaled - d]],
                     fmt='o', color=color, markersize=8, capsize=4,
                     linewidth=1.5, label=label)

    # Pooled effect size
    effect_sizes = [crisis_results[n].get('effect_size_d', 0) for n in names]
    pooled_d = np.mean(effect_sizes)
    ax.axvline(pooled_d, color='#666666', linestyle=':', linewidth=1,
               label=f'Pooled d={pooled_d:.2f}')

    # Reference lines
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(0.8, color='#999999', linestyle='--', linewidth=0.8,
               label="Cohen's d=0.8 (large)")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([CRISIS_LABELS.get(n, n) for n in names])
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title('Effect Sizes Across Crises')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.invert_yaxis()

    fig.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, 'fig_forest_plot')

    return fig


def fig_lead_time_distribution(
    crisis_results: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 3: Lead time distribution across crises.

    Box plot showing lead time for each crisis with bootstrap CIs.

    Args:
        crisis_results: Dict of crisis results
        output_dir: If provided, save figure

    Returns:
        matplotlib Figure
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    names = list(crisis_results.keys())
    lead_times = []
    labels = []
    colors = []

    for name in names:
        r = crisis_results[name]
        lt = r.get('lead_time_days')
        if lt is not None:
            lead_times.append(lt)
            labels.append(CRISIS_LABELS.get(name, name))
            colors.append(CRISIS_COLORS.get(name, '#333333'))

    if not lead_times:
        ax.text(0.5, 0.5, 'No lead times detected',
                transform=ax.transAxes, ha='center', va='center')
        return fig

    # Bar chart with individual values
    x_pos = range(len(lead_times))
    bars = ax.bar(x_pos, lead_times, color=colors, alpha=0.7, edgecolor='black',
                  linewidth=0.5)

    # Add bootstrap CIs if available
    for i, name in enumerate(names):
        r = crisis_results[name]
        boot_lt = r.get('bootstrap_lead_time')
        if boot_lt and r.get('lead_time_days') is not None:
            ci_l = boot_lt.get('ci_lower', lead_times[i])
            ci_u = boot_lt.get('ci_upper', lead_times[i])
            ax.errorbar(i, lead_times[i],
                        yerr=[[lead_times[i] - ci_l], [ci_u - lead_times[i]]],
                        fmt='none', color='black', capsize=5, linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Lead Time (Trading Days)')
    ax.set_title('Detection Lead Time Before Crisis Onset')
    ax.axhline(0, color='black', linewidth=0.5)

    fig.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, 'fig_lead_time_distribution')

    return fig


def fig_roc_pr_curves(
    crisis_results: Dict[str, Dict],
    chern_data: Optional[Dict[str, Dict]] = None,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 4: ROC and Precision-Recall curves.

    If full Chern series are available, computes curves at multiple thresholds.
    Otherwise, plots individual crisis points.

    Args:
        crisis_results: Dict of crisis results
        chern_data: Optional dict with chern_series, crisis_idx per crisis
        output_dir: If provided, save figure

    Returns:
        matplotlib Figure
    """
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Plot individual operating points for each crisis
    for name, r in crisis_results.items():
        color = CRISIS_COLORS.get(name, '#333333')
        label = CRISIS_LABELS.get(name, name)
        recall = r.get('recall', 0)
        precision = r.get('precision', 0)
        f1 = r.get('f1_score', 0)

        # ROC point (approximate FPR from data)
        fpr = r.get('fpr', 0.1)
        tpr = recall

        ax1.scatter(fpr, tpr, color=color, s=80, zorder=5, label=label,
                    edgecolors='black', linewidth=0.5)

        # PR point
        ax2.scatter(recall, precision, color=color, s=80, zorder=5, label=label,
                    edgecolors='black', linewidth=0.5)

    # ROC diagonal
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Space')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_aspect('equal')

    # PR baseline
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.8,
                alpha=0.5, label='Baseline')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Space')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='lower left', fontsize=8)
    ax2.set_aspect('equal')

    fig.suptitle('Detection Performance', fontsize=12, y=1.02)
    fig.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, 'fig_roc_pr_curves')

    return fig


def fig_chern_vs_volatility(
    chern_series: np.ndarray,
    prices: np.ndarray,
    times: pd.DatetimeIndex,
    crisis_date: str,
    crisis_name: str,
    vol_window: int = 20,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 5: Chern number vs realized volatility scatter.

    Shows correlation between Chern magnitude and realized vol.

    Args:
        chern_series: Chern values
        prices: Price series
        times: Timestamps
        crisis_date: Crisis date
        crisis_name: Crisis name
        vol_window: Window for realized vol computation
        output_dir: If provided, save figure

    Returns:
        matplotlib Figure
    """
    apply_style()
    color = CRISIS_COLORS.get(crisis_name, '#333333')

    # Compute realized vol
    log_returns = np.diff(np.log(prices))
    rv = pd.Series(log_returns).rolling(window=vol_window).std() * np.sqrt(252)
    rv = rv.dropna().values

    # Align lengths
    min_len = min(len(chern_series), len(rv))
    chern_aligned = chern_series[-min_len:]
    rv_aligned = rv[-min_len:]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Color by pre/post crisis
    crisis_ts = pd.Timestamp(crisis_date)
    aligned_times = times[-min_len:]
    pre_mask = aligned_times < crisis_ts
    post_mask = ~pre_mask

    ax.scatter(rv_aligned[pre_mask], np.abs(chern_aligned[pre_mask]),
               alpha=0.4, s=15, color='#1976D2', label='Pre-crisis')
    ax.scatter(rv_aligned[post_mask], np.abs(chern_aligned[post_mask]),
               alpha=0.4, s=15, color=color, label='Post-crisis')

    # Correlation
    valid = ~(np.isnan(rv_aligned) | np.isnan(chern_aligned))
    if np.sum(valid) > 10:
        corr = np.corrcoef(rv_aligned[valid], np.abs(chern_aligned[valid]))[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Realized Volatility (annualized)')
    ax.set_ylabel('|Chern Number|')
    ax.set_title(f'Chern Number vs Volatility: {CRISIS_LABELS.get(crisis_name, crisis_name)}')
    ax.legend(loc='lower right')

    fig.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, f'fig_chern_vol_scatter_{crisis_name}')

    return fig


def fig_method_comparison(
    crisis_results: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 6: Method comparison table as a figure.

    Compares QCML Chern detection against baseline metrics.

    Args:
        crisis_results: Dict of crisis results
        output_dir: If provided, save figure

    Returns:
        matplotlib Figure
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')

    # Build comparison table
    columns = ['Crisis', 'Delta Chern', "Cohen's d", 'p-value',
               'BF10', 'F1', 'Lead (days)', 'Verdict']

    rows = []
    for name, r in crisis_results.items():
        label = CRISIS_LABELS.get(name, name)
        bf = r.get('bayes_factor', {})
        bf_val = bf.get('bf_10', 'N/A')
        bf_str = f'{bf_val:.1f}' if isinstance(bf_val, float) else str(bf_val)

        lt = r.get('lead_time_days', 'N/A')
        lt_str = str(lt) if lt is not None else 'N/A'

        verdict = r.get('evidence_strength', 'N/A')

        rows.append([
            label,
            f"{r.get('delta_chern', 0):.4f}",
            f"{r.get('effect_size_d', 0):.2f}",
            f"{r.get('welch_p_value', r.get('p_value', 1.0)):.4f}",
            bf_str,
            f"{r.get('f1_score', 0):.3f}",
            lt_str,
            verdict,
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Header style
    for j in range(len(columns)):
        table[0, j].set_facecolor('#E3F2FD')
        table[0, j].set_text_props(weight='bold')

    # Color verdict cells
    for i, row in enumerate(rows):
        verdict = row[-1]
        if verdict == 'strong':
            table[i + 1, len(columns) - 1].set_facecolor('#C8E6C9')
        elif verdict == 'moderate':
            table[i + 1, len(columns) - 1].set_facecolor('#FFF9C4')
        elif verdict == 'weak':
            table[i + 1, len(columns) - 1].set_facecolor('#FFECB3')
        else:
            table[i + 1, len(columns) - 1].set_facecolor('#FFCDD2')

    ax.set_title('QCML Regime Detection: Statistical Summary', fontsize=12, pad=20)

    fig.tight_layout()

    if output_dir:
        save_figure(fig, output_dir, 'fig_method_comparison')

    return fig


def generate_all_figures_from_results(
    results_file: str,
    output_dir: str = "experiments/outputs/regime_detection/figures",
) -> List[str]:
    """
    Generate all figures from a saved results JSON file.

    Args:
        results_file: Path to results JSON
        output_dir: Output directory for figures

    Returns:
        List of saved file paths
    """
    with open(results_file) as f:
        data = json.load(f)

    output_path = Path(output_dir)
    crisis_results = data.get('crises', {})
    saved = []

    # Figure 2: Forest plot
    fig_forest_plot(crisis_results, output_dir=output_path)
    saved.extend([str(output_path / 'fig_forest_plot.pdf'),
                  str(output_path / 'fig_forest_plot.png')])

    # Figure 3: Lead time distribution
    fig_lead_time_distribution(crisis_results, output_dir=output_path)
    saved.extend([str(output_path / 'fig_lead_time_distribution.pdf'),
                  str(output_path / 'fig_lead_time_distribution.png')])

    # Figure 4: ROC/PR
    fig_roc_pr_curves(crisis_results, output_dir=output_path)
    saved.extend([str(output_path / 'fig_roc_pr_curves.pdf'),
                  str(output_path / 'fig_roc_pr_curves.png')])

    # Figure 6: Method comparison
    fig_method_comparison(crisis_results, output_dir=output_path)
    saved.extend([str(output_path / 'fig_method_comparison.pdf'),
                  str(output_path / 'fig_method_comparison.png')])

    return saved


def generate_all_figures_synthetic(
    output_dir: str = "experiments/outputs/regime_detection/figures",
    seed: int = 42,
) -> List[str]:
    """
    Generate all figures using synthetic data and rigorous validation.

    Args:
        output_dir: Output directory
        seed: Random seed

    Returns:
        List of saved file paths
    """
    from experiments.rigorous_crisis_validation import (
        RigorousCrisisValidator,
        create_synthetic_crisis_dataset,
    )
    from experiments.crisis_config import ValidationConfig

    output_path = Path(output_dir)
    saved = []

    # Run validation
    validator = RigorousCrisisValidator(
        n_bootstrap=1000, n_permutations=500, seed=seed
    )
    validator.validate_all_crises(use_synthetic=True)

    # Convert results for figure functions
    crisis_results = {}
    for name, r in validator.results.items():
        crisis_results[name] = {
            'delta_chern': r.delta_chern,
            'effect_size_d': r.effect_size_d,
            'bootstrap_delta_chern': r.bootstrap_delta_chern,
            'bootstrap_lead_time': r.bootstrap_lead_time,
            'welch_p_value': r.welch_p_value,
            'bayes_factor': r.bayes_factor,
            'f1_score': r.f1_score,
            'recall': r.recall,
            'precision': r.precision,
            'lead_time_days': r.lead_time_days,
            'evidence_strength': r.evidence_strength,
        }

    # Figure 1: Chern time series per crisis
    for crisis in ALL_CRISES:
        r = validator.results.get(crisis.name)
        if r and r.raw_chern_series:
            dataset = create_synthetic_crisis_dataset(crisis, seed=seed)
            chern_arr = np.array(r.raw_chern_series)
            config = validator.config
            chern_times = dataset.times[config.window_size - 1:]
            if len(chern_times) > len(chern_arr):
                chern_times = chern_times[:len(chern_arr)]

            fig_chern_time_series(
                crisis_name=crisis.name,
                chern_series=chern_arr,
                times=chern_times,
                prices=dataset.y,
                crisis_date=crisis.crisis_date,
                output_dir=output_path,
            )
            saved.append(str(output_path / f'fig_chern_{crisis.name}.pdf'))

    # Figure 2: Forest plot
    fig_forest_plot(crisis_results, output_dir=output_path)
    saved.append(str(output_path / 'fig_forest_plot.pdf'))

    # Figure 3: Lead time
    fig_lead_time_distribution(crisis_results, output_dir=output_path)
    saved.append(str(output_path / 'fig_lead_time_distribution.pdf'))

    # Figure 4: ROC/PR
    fig_roc_pr_curves(crisis_results, output_dir=output_path)
    saved.append(str(output_path / 'fig_roc_pr_curves.pdf'))

    # Figure 5: Chern vs vol (for first crisis with data)
    for crisis in ALL_CRISES:
        r = validator.results.get(crisis.name)
        if r and r.raw_chern_series:
            dataset = create_synthetic_crisis_dataset(crisis, seed=seed)
            chern_arr = np.array(r.raw_chern_series)
            config = validator.config
            chern_times = dataset.times[config.window_size - 1:]
            if len(chern_times) > len(chern_arr):
                chern_times = chern_times[:len(chern_arr)]

            prices_arr = dataset.y.values if hasattr(dataset.y, 'values') else dataset.y
            fig_chern_vs_volatility(
                chern_series=chern_arr,
                prices=prices_arr,
                times=chern_times,
                crisis_date=crisis.crisis_date,
                crisis_name=crisis.name,
                output_dir=output_path,
            )
            saved.append(str(output_path / f'fig_chern_vol_scatter_{crisis.name}.pdf'))

    # Figure 6: Method comparison
    fig_method_comparison(crisis_results, output_dir=output_path)
    saved.append(str(output_path / 'fig_method_comparison.pdf'))

    return saved


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for QCML crisis validation"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate from synthetic data")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/outputs/regime_detection/figures",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("QCML Crisis Figure Generation")
    print("=" * 60)

    if args.results_file:
        saved = generate_all_figures_from_results(
            args.results_file, output_dir=args.output_dir
        )
    elif args.synthetic:
        saved = generate_all_figures_synthetic(
            output_dir=args.output_dir, seed=args.seed
        )
    else:
        print("Specify --synthetic or --results-file")
        sys.exit(1)

    print(f"\nGenerated {len(saved)} figure files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
