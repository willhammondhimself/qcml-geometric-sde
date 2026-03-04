#!/usr/bin/env python3
"""
Regenerate all paper figures from canonical JSON experiment outputs.

Reads the following pre-computed results (no new experiments):
  - regime_detection/causal_comparison_20260303_231013.json  (36 methods x 17 crises)
  - fusion/fusion_results_20260304_101523.json               (fusion train, 15 crises)
  - fusion/fusion_results_20260304_101842.json               (fusion holdout, 4 crises)
  - lead_time/lead_time_20260304_024009.json                 (lead times)
  - holdout/holdout_20260304_023737.json                     (blind holdout)

Outputs all figures to paper/figures/ in PDF + PNG.

Usage:
    python experiments/regenerate_all_figures.py
    python experiments/regenerate_all_figures.py --skip-narratives  # skip yfinance fetch
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.plot_style import (
    apply_style, save_figure, comparison_barchart, heatmap_figure, crisis_figure,
    format_date_axis,
    NAVY, TEAL, BURGUNDY, GOLD, INDIGO, SLATE, LIGHT,
    METHOD_COLORS, COLOR_CYCLE, CMAP_SEQUENTIAL, FIGURE_DIR,
)
from experiments.data_loader import (
    fetch_data, create_feature_matrix_single_asset,
    ALL_CRISES, NARRATIVE_CRISES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# =============================================================================
# Data file paths
# =============================================================================

OUTPUTS = ROOT / 'experiments' / 'outputs'

MAIN_JSON = OUTPUTS / 'regime_detection' / 'causal_comparison_20260303_231013.json'
FUSION_TRAIN_JSON = OUTPUTS / 'fusion' / 'fusion_results_20260304_101523.json'
FUSION_HOLDOUT_JSON = OUTPUTS / 'fusion' / 'fusion_results_20260304_101842.json'
LEAD_TIME_JSON = OUTPUTS / 'lead_time' / 'lead_time_20260304_024009.json'
HOLDOUT_JSON = OUTPUTS / 'holdout' / 'holdout_20260304_023737.json'

# Baseline method names (for coloring)
BASELINE_NAMES = {
    'Random Forest', 'Rolling Vol Z', 'CUSUM', 'GARCH(1,1)', 'Hamilton MS',
    'EWMA Vol', 'Mahalanobis', 'Structural Break', 'Transfer Entropy',
    'Isolation Forest', 'VIX Level', 'Rolling RF (VIX)', 'HMM', 'BOCPD',
}


def _color_for(method):
    """Return color for a method: check METHOD_COLORS, then baseline vs QCML."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    if method in BASELINE_NAMES:
        return SLATE
    # Cycle through QCML colors for unknown QCML methods
    return INDIGO


def load_main():
    """Load 36x17 main comparison results."""
    with open(MAIN_JSON) as f:
        return json.load(f)


def load_fusion_train():
    with open(FUSION_TRAIN_JSON) as f:
        return json.load(f)


def load_fusion_holdout():
    with open(FUSION_HOLDOUT_JSON) as f:
        return json.load(f)


def load_lead_time():
    with open(LEAD_TIME_JSON) as f:
        return json.load(f)


def load_holdout():
    with open(HOLDOUT_JSON) as f:
        return json.load(f)


# =============================================================================
# Figure 1: Ranked Bar Chart (36 methods)
# =============================================================================

def figure_1_ranked_barchart():
    """All 36 methods sorted by median Cohen's d, horizontal bars."""
    logger.info("Figure 1: Ranked bar chart (36 methods)")
    data = load_main()
    median_d = data['summary']['median_d']

    # Sort descending
    sorted_items = sorted(median_d.items(), key=lambda x: x[1], reverse=True)
    methods = [m for m, _ in sorted_items]
    values = [d for _, d in sorted_items]

    # Height scales with method count
    figsize = (7, max(8, len(methods) * 0.32))
    fig, ax = comparison_barchart(
        methods, values,
        title='Regime Detection: All Methods Ranked by Median Cohen\'s $d$',
        xlabel='Median Cohen\'s $d$ (across 17 crises)',
        figsize=figsize,
    )

    # Add reference lines
    ax.axvline(0.8, color=BURGUNDY, ls='--', lw=0.8, alpha=0.5, label='$d=0.8$ (large)')
    ax.axvline(0.5, color=TEAL, ls='--', lw=0.8, alpha=0.5, label='$d=0.5$ (medium)')
    ax.legend(fontsize=7, loc='lower right')

    # Override colors using our logic
    for bar, method in zip(ax.patches, methods):
        bar.set_color(_color_for(method))

    fig.tight_layout()
    save_figure(fig, 'ranked_methods_barchart')
    logger.info("  -> ranked_methods_barchart.pdf/png")


# =============================================================================
# Figure 2: Crisis Specialization Heatmap (36 x 17)
# =============================================================================

def figure_2_crisis_heatmap():
    """36 methods (rows) x 17 crises (cols), cells = Cohen's d."""
    logger.info("Figure 2: Crisis specialization heatmap (36x17)")
    data = load_main()
    results = data['results']
    median_d = data['summary']['median_d']

    # Sort methods by median d descending
    sorted_methods = sorted(median_d.items(), key=lambda x: x[1], reverse=True)
    method_names = [m for m, _ in sorted_methods]

    # Get crisis keys from first method
    first_method = list(results.keys())[0]
    crisis_keys = list(results[first_method].keys())

    # Build matrix
    matrix = np.full((len(method_names), len(crisis_keys)), np.nan)
    for i, method in enumerate(method_names):
        for j, crisis in enumerate(crisis_keys):
            if crisis in results.get(method, {}):
                matrix[i, j] = results[method][crisis]['d']

    # Crisis labels from ALL_CRISES
    crisis_labels = []
    for ck in crisis_keys:
        if ck in ALL_CRISES:
            crisis_labels.append(ALL_CRISES[ck]['label'])
        else:
            crisis_labels.append(ck.replace('_', ' ').title())

    figsize = (max(8, len(crisis_keys) * 0.85), max(10, len(method_names) * 0.38))
    fig, ax = heatmap_figure(
        matrix, method_names, crisis_labels,
        title='Cohen\'s $d$ by Method and Crisis (36 methods $\\times$ 17 crises)',
        cmap=CMAP_SEQUENTIAL,
        vmin=0, vmax=1.5,
        figsize=figsize,
        annotate=True,
        cbar_label='Cohen\'s $d$',
    )

    # Bold cells with d > 0.8
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val) and val > 0.8:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=GOLD, linewidth=1.5,
                ))

    fig.tight_layout()
    save_figure(fig, 'crisis_heatmap_36x17')
    logger.info("  -> crisis_heatmap_36x17.pdf/png")


# =============================================================================
# Figure 3: Top-10 Effect Sizes Violin
# =============================================================================

def figure_3_violin_top10():
    """Violin plot of Cohen's d distributions for top 10 methods."""
    logger.info("Figure 3: Top-10 effect sizes violin")
    data = load_main()
    results = data['results']
    median_d = data['summary']['median_d']

    # Top 10 by median d
    sorted_methods = sorted(median_d.items(), key=lambda x: x[1], reverse=True)[:10]
    method_names = [m for m, _ in sorted_methods]

    # Collect d-values across crises for each method
    distributions = []
    for method in method_names:
        d_vals = [v['d'] for v in results[method].values() if isinstance(v, dict) and 'd' in v]
        distributions.append(d_vals)

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    parts = ax.violinplot(distributions, positions=range(len(method_names)),
                          showmeans=True, showmedians=True, showextrema=False)

    # Color the violins
    for i, (pc, method) in enumerate(zip(parts['bodies'], method_names)):
        pc.set_facecolor(_color_for(method))
        pc.set_alpha(0.7)

    parts['cmeans'].set_color(NAVY)
    parts['cmedians'].set_color(BURGUNDY)

    # Reference lines
    ax.axhline(0.8, color=BURGUNDY, ls='--', lw=0.8, alpha=0.5, label='$d=0.8$ (large)')
    ax.axhline(0.5, color=TEAL, ls='--', lw=0.8, alpha=0.5, label='$d=0.5$ (medium)')

    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel("Cohen's $d$")
    ax.set_title('Effect Size Distributions: Top 10 Methods (17 crises)')
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    save_figure(fig, 'effect_sizes_top10')
    logger.info("  -> effect_sizes_top10.pdf/png")


# =============================================================================
# Figure 4: Fusion Comparison Bar Chart
# =============================================================================

def figure_4_fusion_comparison():
    """5 fusion strategies + top 3 individuals, grouped bars with error bars."""
    logger.info("Figure 4: Fusion comparison bar chart")
    fusion = load_fusion_train()
    agg = fusion['aggregate']

    # Separate fusion methods from individuals
    fusion_methods = [r for r in agg if r['category'] not in ('individual',)]
    individual_methods = [r for r in agg if r['category'] == 'individual']

    # Top 3 individuals by mean_d
    top_individuals = sorted(individual_methods, key=lambda x: x['mean_d'], reverse=True)[:3]

    # Combine: fusion first, then top individuals
    combined = fusion_methods + top_individuals

    # Sort combined by mean_d descending
    combined = sorted(combined, key=lambda x: x['mean_d'], reverse=True)

    names = [r['method'] for r in combined]
    mean_d = [r['mean_d'] for r in combined]
    std_d = [r['std_d'] for r in combined]

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = []
    for r in combined:
        if r['category'] == 'individual':
            colors.append(SLATE)
        elif r['category'] == 'regime_adaptive':
            colors.append(BURGUNDY)
        elif r['category'] == 'hierarchical':
            colors.append(TEAL)
        elif r['category'] == 'flat_fusion':
            colors.append(NAVY)
        elif r['category'] == 'sprt':
            colors.append(GOLD)
        else:
            colors.append(INDIGO)

    bars = ax.barh(range(len(names)), mean_d, xerr=std_d,
                   color=colors, edgecolor='white', linewidth=0.5,
                   capsize=3, alpha=0.85)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Cohen\'s $d$ ($\\pm$ std)')
    ax.set_title('Fusion Strategies vs Top Individual Methods (15 crises)')

    # Reference lines
    ax.axvline(0.8, color=BURGUNDY, ls='--', lw=0.8, alpha=0.4)
    ax.axvline(0.5, color=TEAL, ls='--', lw=0.8, alpha=0.4)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BURGUNDY, label='Regime-Adaptive'),
        Patch(facecolor=TEAL, label='Hierarchical'),
        Patch(facecolor=NAVY, label='Flat Fusion'),
        Patch(facecolor=GOLD, label='SPRT'),
        Patch(facecolor=SLATE, label='Individual (top 3)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    fig.tight_layout()
    save_figure(fig, 'fusion_comparison')
    logger.info("  -> fusion_comparison.pdf/png")


# =============================================================================
# Figure 5: Holdout Generalization Plot
# =============================================================================

def figure_5_holdout_generalization():
    """Paired slope plot: train d vs holdout d for each method."""
    logger.info("Figure 5: Holdout generalization (train vs holdout)")
    train = load_fusion_train()
    holdout = load_fusion_holdout()

    # Build lookup: method -> train mean_d
    train_lookup = {r['method']: r for r in train['aggregate']}
    holdout_lookup = {r['method']: r for r in holdout['aggregate']}

    # Find common methods
    common = sorted(set(train_lookup) & set(holdout_lookup))

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in common:
        t_d = train_lookup[method]['mean_d']
        h_d = holdout_lookup[method]['mean_d']
        cat = train_lookup[method]['category']

        if cat == 'regime_adaptive':
            color, marker, zorder = BURGUNDY, 'D', 10
        elif cat == 'individual':
            color, marker, zorder = SLATE, 'o', 5
        elif 'hierarchical' in cat:
            color, marker, zorder = TEAL, 's', 7
        elif cat == 'flat_fusion':
            color, marker, zorder = NAVY, '^', 7
        elif 'sprt' in cat:
            color, marker, zorder = GOLD, 'v', 6
        else:
            color, marker, zorder = INDIGO, 'o', 5

        ax.plot([0, 1], [t_d, h_d], color=color, alpha=0.4, lw=1.2, zorder=zorder - 1)
        ax.scatter([0], [t_d], color=color, marker=marker, s=50, zorder=zorder, alpha=0.8)
        ax.scatter([1], [h_d], color=color, marker=marker, s=50, zorder=zorder, alpha=0.8)

        # Label key methods
        if method in ('Regime-Adaptive', 'Reduced Purity', 'Spectral Entropy',
                       'Hierarchical Learned', 'Flat Rank Fusion'):
            offset_x = 0.03 if h_d >= t_d else -0.03
            ha = 'left' if h_d >= t_d else 'right'
            ax.annotate(method, (1, h_d), xytext=(1 + 0.05, h_d),
                        fontsize=7, color=color, va='center')

    # Diagonal reference
    ax.plot([0, 1], [0, 0], color=SLATE, ls=':', lw=0.5, alpha=0.3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Train (15 crises)', 'Holdout (4 post-2020)'], fontsize=10)
    ax.set_ylabel("Mean Cohen's $d$")
    ax.set_title('Generalization: Train vs Holdout Performance')
    ax.set_xlim(-0.15, 1.6)

    # Add "perfect generalization" diagonal hint
    ax.axhline(0.8, color=BURGUNDY, ls='--', lw=0.6, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'holdout_generalization')
    logger.info("  -> holdout_generalization.pdf/png")


# =============================================================================
# Figure 6: Lead Time Visualization
# =============================================================================

def figure_6_lead_time():
    """Horizontal bar chart: mean lead time per method, colored by detection rate."""
    logger.info("Figure 6: Lead time analysis")
    lt_data = load_lead_time()
    summary = lt_data['summary']

    # Filter to methods that detect >= 3 crises
    filtered = {m: s for m, s in summary.items() if s['n_detected'] >= 3}

    if not filtered:
        logger.warning("  No methods detected >= 3 crises, skipping figure 6")
        return

    # Sort by mean lead time descending
    sorted_methods = sorted(filtered.items(), key=lambda x: x[1]['mean_lead_time'], reverse=True)
    names = [m for m, _ in sorted_methods]
    lead_times = [s['mean_lead_time'] for _, s in sorted_methods]
    detection_rates = [s['n_detected'] / s['n_total'] for _, s in sorted_methods]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, max(5, len(names) * 0.35)))

    # Color by detection rate: higher rate = darker
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=CMAP_SEQUENTIAL, norm=norm)

    colors = [sm.to_rgba(r) for r in detection_rates]

    bars = ax.barh(range(len(names)), lead_times, color=colors,
                   edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Lead Time (trading days)')
    ax.set_title('Early Warning Lead Times (methods detecting $\\geq 3$ crises)')

    # Add detection rate text at end of each bar
    for i, (lt, dr) in enumerate(zip(lead_times, detection_rates)):
        ax.text(lt + 0.5, i, f'{dr:.0%}', va='center', fontsize=7, color=NAVY)

    # Colorbar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Detection Rate', fontsize=8)

    fig.tight_layout()
    save_figure(fig, 'lead_time_analysis')
    logger.info("  -> lead_time_analysis.pdf/png")


# =============================================================================
# Figure 7: Crisis Timeline (17 crises)
# =============================================================================

def figure_7_crisis_timeline():
    """Horizontal timeline bars for all 17 crises (1997-2024)."""
    logger.info("Figure 7: Crisis timeline (17 crises)")

    # Category colors
    category_colors = {
        'equity': NAVY,
        'credit': BURGUNDY,
        'flash': GOLD,
        'rates': TEAL,
        'geopolitical': INDIGO,
        'liquidity': SLATE,
    }

    # Manual category assignment for each crisis
    crisis_categories = {
        '1997_asia': 'equity',
        '1998_ltcm': 'liquidity',
        '2000_dotcom': 'equity',
        '2001_911': 'geopolitical',
        '2007_quant': 'liquidity',
        '2008_gfc': 'credit',
        '2010_flash': 'flash',
        '2011_euro': 'credit',
        '2013_taper': 'rates',
        '2015_china': 'equity',
        '2016_brexit': 'geopolitical',
        '2018_volmageddon': 'flash',
        '2018_q4': 'equity',
        '2019_repo': 'liquidity',
        '2020_covid': 'geopolitical',
        '2021_meme': 'equity',
        '2022_rates': 'rates',
        '2023_svb': 'credit',
        '2024_carry': 'liquidity',
    }

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort crises chronologically
    sorted_crises = sorted(ALL_CRISES.items(), key=lambda x: x[1]['start'])

    for i, (key, info) in enumerate(sorted_crises):
        start = pd.Timestamp(info['start'])
        end = pd.Timestamp(info['end'])
        duration = (end - start).days

        cat = crisis_categories.get(key, 'equity')
        color = category_colors.get(cat, SLATE)

        ax.barh(i, duration, left=mdates.date2num(start), height=0.6,
                color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.text(mdates.date2num(end) + 5, i, info['label'],
                va='center', fontsize=8, color=NAVY)

    ax.set_yticks(range(len(sorted_crises)))
    ax.set_yticklabels([])
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Year')
    ax.set_title('Financial Crises Timeline (1997\u20132024)')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=v, label=k.title()) for k, v in category_colors.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right', ncol=3)

    fig.tight_layout()
    save_figure(fig, 'crisis_timeline_17')
    logger.info("  -> crisis_timeline_17.pdf/png")


# =============================================================================
# Figure 8: Crisis Narrative Panels (3 crises x 8 panels each)
# =============================================================================

NARRATIVE_DETECTORS = {
    'Reduced Purity': {
        'class': 'ReducedPurityDetector',
        'color': INDIGO,
    },
    'Berry Phase': {
        'class': 'BerryPhaseRateDetector',
        'color': BURGUNDY,
    },
    'QFI': {
        'class': 'QFIDeterminantDetector',
        'color': NAVY,
    },
    'Spectral Entropy': {
        'class': 'SpectralEntropyDetector',
        'color': TEAL,
    },
    'Multi-Lag Fidelity': {
        'class': 'MultiLagFidelityDetector',
        'color': GOLD,
    },
    'Hamiltonian Sens.': {
        'class': 'HamiltonianSensitivityDetector',
        'color': '#5D8AA8',
    },
    'Dim. Collapse': {
        'class': 'DimensionalityCollapseDetector',
        'color': '#9B59B6',
    },
    'Speed Limit': {
        'class': 'SpeedLimitRatioDetector',
        'color': '#E67E22',
    },
}


def _run_detector(detector_class_name, features, dates):
    """Instantiate, fit, and run a detector, returning its regime score series."""
    import qcml_geometry.observables as obs
    cls = getattr(obs, detector_class_name)
    detector = cls()
    detector.fit(features)
    scores = detector.compute_regime_scores(features)
    return pd.Series(scores, index=dates[:len(scores)])


def figure_8_narrative(crisis_key, crisis_info):
    """8-panel crisis anatomy figure for a single crisis."""
    label = crisis_info['short']
    logger.info(f"Figure 8: Narrative for {label}")

    context_start = crisis_info['context_start']
    context_end = crisis_info['context_end']
    crisis_start = pd.Timestamp(crisis_info['start'])
    crisis_end = pd.Timestamp(crisis_info['end'])

    # Fetch data
    raw = fetch_data(['SPY'], context_start, context_end)
    prices = raw['close'].unstack('symbol')['SPY']
    features, feat_dates = create_feature_matrix_single_asset(prices)

    n_panels = len(NARRATIVE_DETECTORS)
    fig, axes = crisis_figure(
        n_panels, 1,
        crisis_start=crisis_start, crisis_end=crisis_end,
        figsize=(8, 2.2 * n_panels),
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, NARRATIVE_DETECTORS.items()):
        try:
            signal = _run_detector(info['class'], features, feat_dates)
            ax.plot(signal.index, signal.values, color=info['color'], lw=1.0)
            ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=70, va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'{name}: error\n{e}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='red')
            ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=70, va='center')

        ax.set_xlabel('')
        if ax != axes[-1]:
            ax.set_xticklabels([])

    format_date_axis(axes[-1], interval_months=3)
    fig.suptitle(f'Crisis Anatomy: {label}', fontsize=12, y=1.01)
    fig.tight_layout()

    filename = f"narrative_{crisis_key}"
    save_figure(fig, filename)
    logger.info(f"  -> {filename}.pdf/png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Regenerate all paper figures from JSON data')
    parser.add_argument('--skip-narratives', action='store_true',
                        help='Skip narrative figures (avoids yfinance fetch)')
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {FIGURE_DIR}")

    # Verify all JSON files exist
    for path in [MAIN_JSON, FUSION_TRAIN_JSON, FUSION_HOLDOUT_JSON, LEAD_TIME_JSON, HOLDOUT_JSON]:
        if not path.exists():
            logger.error(f"Missing: {path}")
            sys.exit(1)
    logger.info("All JSON data files found")

    # Generate figures 1-7
    figure_1_ranked_barchart()
    figure_2_crisis_heatmap()
    figure_3_violin_top10()
    figure_4_fusion_comparison()
    figure_5_holdout_generalization()
    figure_6_lead_time()
    figure_7_crisis_timeline()

    # Figure 8: Narrative panels (requires yfinance)
    if not args.skip_narratives:
        for crisis_key, crisis_info in NARRATIVE_CRISES.items():
            figure_8_narrative(crisis_key, crisis_info)
    else:
        logger.info("Skipping narrative figures (--skip-narratives)")

    logger.info("All figures regenerated successfully!")


if __name__ == '__main__':
    main()
