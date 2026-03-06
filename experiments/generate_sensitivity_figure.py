"""
Generate Berry Phase Rate hyperparameter sensitivity figure.

Reads the JSON output from berry_sensitivity_sweep.py and produces a 2x3
panel figure:

    (a) hilbert_dim       (b) n_pca_components   (c) rolling_window
    (d) normalization     (e) berry_aggregation   (f) 2D heatmap (hd x rw)

Usage:
    python experiments/generate_sensitivity_figure.py <json_path>
    python experiments/generate_sensitivity_figure.py experiments/outputs/regime_detection/berry_sensitivity_*.json

Outputs:
    berry_sensitivity.pdf  (vector, for paper)
    berry_sensitivity.png  (300 DPI, for review)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent

from experiments.plot_style import (
    apply_style, NAVY, TEAL, BURGUNDY, GOLD, INDIGO, SLATE,
    CMAP_SEQUENTIAL,
)

COLORS = {
    'berry': BURGUNDY,
    'default_marker': BURGUNDY,
    'line': NAVY,
    'fill': TEAL,
    'threshold': SLATE,
}

PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def _line_panel(ax, param_name, oat_data, defaults, panel_label):
    """Line plot with IQR shading for a continuous OAT parameter."""
    values = sorted(oat_data.keys(), key=lambda x: float(x))
    xs = [float(v) for v in values]
    medians = [oat_data[v]['summary']['median_d'] or 0.0 for v in values]
    q25s = [oat_data[v]['summary']['q25'] or 0.0 for v in values]
    q75s = [oat_data[v]['summary']['q75'] or 0.0 for v in values]

    ax.fill_between(xs, q25s, q75s, alpha=0.2, color=COLORS['fill'])
    ax.plot(xs, medians, '-o', color=COLORS['line'], markersize=5, linewidth=1.5)

    # Default marker
    default_val = float(defaults[param_name])
    ax.axvline(default_val, color=COLORS['default_marker'], linestyle='--',
               alpha=0.6, linewidth=1.0, label=f'default={int(default_val)}')

    # d=0.5 threshold
    ax.axhline(0.5, color=COLORS['threshold'], linestyle=':', alpha=0.5, linewidth=0.8)

    ax.set_xlabel(param_name.replace('_', ' '))
    ax.set_ylabel("Median Cohen's $d$")
    ax.set_title(f"{panel_label} {param_name.replace('_', ' ')}")
    ax.legend(fontsize=7, loc='best')


def _bar_panel(ax, param_name, oat_data, defaults, panel_label):
    """Bar chart with IQR error bars for a categorical OAT parameter."""
    labels = list(oat_data.keys())
    medians = [oat_data[v]['summary']['median_d'] or 0.0 for v in labels]
    q25s = [oat_data[v]['summary']['q25'] or 0.0 for v in labels]
    q75s = [oat_data[v]['summary']['q75'] or 0.0 for v in labels]
    yerr_lo = [m - q for m, q in zip(medians, q25s)]
    yerr_hi = [q - m for m, q in zip(medians, q75s)]

    default_val = str(defaults[param_name])
    bar_colors = [COLORS['default_marker'] if l == default_val else COLORS['fill']
                  for l in labels]

    bars = ax.bar(labels, medians, color=bar_colors, alpha=0.7, edgecolor='white',
                  linewidth=0.5)
    ax.errorbar(labels, medians, yerr=[yerr_lo, yerr_hi], fmt='none',
                ecolor=NAVY, capsize=3, linewidth=1.0)

    # d=0.5 threshold
    ax.axhline(0.5, color=COLORS['threshold'], linestyle=':', alpha=0.5, linewidth=0.8)

    ax.set_xlabel(param_name.replace('_', ' '))
    ax.set_ylabel("Median Cohen's $d$")
    ax.set_title(f"{panel_label} {param_name.replace('_', ' ')}")

    # Mark default bar
    for i, l in enumerate(labels):
        if l == default_val:
            ax.annotate('default', xy=(i, medians[i]), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=7,
                        color=COLORS['default_marker'])


def _heatmap_panel(ax, grid_data, defaults, panel_label):
    """2D heatmap of hilbert_dim x rolling_window."""
    # Collect unique dims/windows
    hd_set = sorted(set(e['hilbert_dim'] for e in grid_data.values()))
    rw_set = sorted(set(e['rolling_window'] for e in grid_data.values()))

    matrix = np.full((len(hd_set), len(rw_set)), np.nan)
    for entry in grid_data.values():
        i = hd_set.index(entry['hilbert_dim'])
        j = rw_set.index(entry['rolling_window'])
        med = entry['summary']['median_d']
        if med is not None:
            matrix[i, j] = med

    im = ax.imshow(matrix, cmap=CMAP_SEQUENTIAL, aspect='auto', origin='lower',
                   norm=Normalize(vmin=max(0, np.nanmin(matrix) - 0.05),
                                  vmax=np.nanmax(matrix) + 0.05))

    # Annotate cells
    for i in range(len(hd_set)):
        for j in range(len(rw_set)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < (np.nanmin(matrix) + np.nanmax(matrix)) / 2 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=text_color)

    # Star at default
    default_hd = defaults['hilbert_dim']
    default_rw = defaults['rolling_window']
    if default_hd in hd_set and default_rw in rw_set:
        di = hd_set.index(default_hd)
        dj = rw_set.index(default_rw)
        ax.plot(dj, di, '*', color=COLORS['default_marker'], markersize=14,
                markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xticks(range(len(rw_set)))
    ax.set_xticklabels(rw_set)
    ax.set_yticks(range(len(hd_set)))
    ax.set_yticklabels(hd_set)
    ax.set_xlabel('rolling window')
    ax.set_ylabel('hilbert dim')
    ax.set_title(f"{panel_label} hilbert dim $\\times$ rolling window")

    plt.colorbar(im, ax=ax, label="Median Cohen's $d$", shrink=0.85)


def generate_figure(data, output_dir=None):
    """Generate the 2x3 sensitivity panel figure."""
    apply_style()

    if output_dir is None:
        output_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = data['defaults']
    oat = data['oat']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: continuous OAT params (line plots)
    _line_panel(axes[0, 0], 'hilbert_dim', oat['hilbert_dim'], defaults, PANEL_LABELS[0])
    _line_panel(axes[0, 1], 'n_pca_components', oat['n_pca_components'], defaults, PANEL_LABELS[1])
    _line_panel(axes[0, 2], 'rolling_window', oat['rolling_window'], defaults, PANEL_LABELS[2])

    # Row 2: categorical OAT params (bar charts) + 2D heatmap
    _bar_panel(axes[1, 0], 'normalization', oat['normalization'], defaults, PANEL_LABELS[3])
    _bar_panel(axes[1, 1], 'berry_aggregation', oat['berry_aggregation'], defaults, PANEL_LABELS[4])
    _heatmap_panel(axes[1, 2], data['grid_2d'], defaults, PANEL_LABELS[5])

    fig.suptitle('Berry Phase Rate: Hyperparameter Sensitivity Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    pdf_path = output_dir / 'berry_sensitivity.pdf'
    png_path = output_dir / 'berry_sensitivity.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")

    # Print aggregate stats for paper
    agg = data.get('aggregate', {})
    if agg:
        print(f"\nOAT configs with d > 0.5: {agg.get('oat_n_above_0.5')}/{agg.get('oat_n_configs')} "
              f"({agg.get('oat_pct_above_0.5')}%)")
        print(f"OAT median-of-medians: {agg.get('oat_median_of_medians')}")
        print(f"OAT range: [{agg.get('oat_min_median')}, {agg.get('oat_max_median')}]")
        print(f"Grid median-of-medians: {agg.get('grid_median_of_medians')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Berry sensitivity figure from sweep JSON',
    )
    parser.add_argument('json_path', type=str, help='Path to berry_sensitivity_*.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as JSON)')
    args = parser.parse_args()

    json_path = Path(args.json_path)
    with open(json_path) as f:
        data = json.load(f)

    out_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    generate_figure(data, output_dir=out_dir)
