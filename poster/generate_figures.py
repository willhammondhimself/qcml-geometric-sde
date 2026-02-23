"""Generate poster-optimized figures for APS Global Physics Summit 2026 (v3 — real data).

Produces 2 figures (white-background, modern design):
1. crisis_timeline.png — Real SPY prices + real geometric P(crisis) signal
2. method_comparison.png — Horizontal bar chart from POSTER_RESULTS.json

Data source: POSTER_FIGURE_DATA.json and POSTER_RESULTS.json from poster_evaluation.py.
Falls back to hardcoded results if JSON files not found (for standalone use).

Usage:
    python poster/generate_figures.py                    # uses real data if available
    python poster/generate_figures.py --from-results     # requires POSTER_RESULTS.json
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

# Directories
ROOT = Path(__file__).resolve().parent.parent
OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "experiments" / "outputs" / "regime_detection"

# Color palette (poster accent colors on white)
GOLD = "#C8850F"
GREEN = "#4CAF50"
BLUE = "#42A5F5"
DARK = "#1A1A2E"
MEDIUM_GRAY = "#555555"
LIGHT_GRAY = "#CCCCCC"
RED_ACCENT = "#E53935"
TEAL = "#00897B"
BG = "#FFFFFF"

# Crisis period definitions (for shading)
CRISIS_SHADING = [
    ('2008-09-01', '2009-03-31', '2008 GFC'),
    ('2018-01-26', '2018-04-30', 'Volmageddon'),
    ('2020-02-20', '2020-04-30', 'COVID'),
    ('2022-01-01', '2022-10-31', 'Rate Shock'),
]


def _load_figure_data():
    """Load real figure data from POSTER_FIGURE_DATA.json."""
    path = RESULTS_DIR / 'POSTER_FIGURE_DATA.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_results():
    """Load results from POSTER_RESULTS.json."""
    path = RESULTS_DIR / 'POSTER_RESULTS.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fig1_method_comparison(results=None):
    """Horizontal bar chart: Cohen's d for key detection methods (white bg).

    If results dict is provided, uses actual numbers from POSTER_RESULTS.json.
    Otherwise falls back to latest known results.
    """
    # Try to extract from results
    if results and 'offline_comparison' in results:
        offline = results['offline_comparison']
        method_d = []
        for method_name, per_crisis in offline.items():
            g = per_crisis.get('global')
            if g is not None:
                method_d.append((method_name, g))
        method_d.sort(key=lambda x: x[1], reverse=True)
        methods = [m[0] for m in method_d]
        cohens_d = [m[1] for m in method_d]
    else:
        # Fallback: hardcoded latest results including new detectors
        methods = [
            "Multi-Plane Berry",
            "Geodesic Distance",
            "Berry Phase Rate",
            "QFI Determinant",
            "Multi-Lag Fidelity",
            "Ricci Scalar",
        ]
        cohens_d = [1.05, 0.98, 0.93, 0.93, 0.84, 0.75]

    colors = []
    for m in methods:
        if "Random Forest" in m:
            colors.append(RED_ACCENT)
        elif any(k in m for k in [
            "Berry", "QFI", "Multi-Lag", "Fused", "Geometric",
            "Geodesic", "Ricci", "Multi-Plane",
        ]):
            colors.append(GOLD)
        else:
            colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.55)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, cohens_d, color=colors, edgecolor='none', height=0.7, zorder=3)

    for bar, val in zip(bars, cohens_d):
        x_pos = bar.get_width() + 0.03
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"d = {val:.2f}",
                va='center', ha='left', color=DARK, fontsize=13, fontweight='bold')

    ax.axvline(x=0.8, color=MEDIUM_GRAY, linestyle='--', linewidth=1.5, alpha=0.4, zorder=2)
    ax.text(0.82, len(methods) - 0.3, "Large effect\n(d = 0.8)", color=MEDIUM_GRAY,
            fontsize=10, alpha=0.6, va='top')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, color=DARK, fontsize=12)
    ax.set_xlabel("Cohen's d (effect size vs. calm periods)", color=DARK, fontsize=14)
    ax.set_title("Geometric Regime Detection: Method Comparison (12 Crises, 5 ETFs)",
                 color=DARK, fontsize=16, fontweight='bold', pad=15)

    ax.tick_params(axis='x', colors=DARK, labelsize=11)
    ax.tick_params(axis='y', colors=DARK)
    max_d = max(cohens_d) if cohens_d else 2.0
    ax.set_xlim(0, max_d + 0.35)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color=LIGHT_GRAY, alpha=0.5, zorder=1)

    legend_elements = [
        mpatches.Patch(facecolor=GOLD, label='Geometric Methods (ours)'),
        mpatches.Patch(facecolor=RED_ACCENT, label='Supervised Baseline'),
        mpatches.Patch(facecolor="#9E9E9E", label='Classical Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
              facecolor=BG, edgecolor=LIGHT_GRAY, labelcolor=DARK)

    fig.tight_layout(pad=1.5)
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTDIR / f"method_comparison.{fmt}", dpi=300, facecolor=BG,
                    bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print("Saved method_comparison.png + .pdf")


def fig2_crisis_timeline(figure_data=None):
    """Crisis detection timeline with real SPY prices and geometric signal.

    Uses real data from POSTER_FIGURE_DATA.json if available.
    """
    if figure_data is not None:
        dates = pd.to_datetime(figure_data['spy_dates'])
        prices = np.array(figure_data['spy_prices'])
        signal = np.array(figure_data['p_crisis'])
    else:
        # Fallback: synthetic data (for standalone testing)
        print("  WARNING: Using synthetic data — run poster_evaluation.py first for real data")
        np.random.seed(42)
        n = 252 * 17
        dates = pd.bdate_range('2007-01-03', periods=n)
        prices = 150 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        signal = np.random.rand(n) * 0.5

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), height_ratios=[2, 1.2],
        sharex=True, gridspec_kw={'hspace': 0.06},
    )
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    # Price panel
    ax1.plot(dates, prices, color=DARK, linewidth=1.2, alpha=0.85, zorder=3)

    # Crisis shading + labels
    y_max = np.nanmax(prices) * 1.15
    ax1.set_ylim(0, y_max * 1.18)
    y_fracs = [1.10, 1.02, 1.10, 1.02]

    for (cs, ce, label), y_frac in zip(CRISIS_SHADING, y_fracs):
        cs_dt = pd.Timestamp(cs)
        ce_dt = pd.Timestamp(ce)

        # Shade
        ax1.axvspan(cs_dt, ce_dt, color=RED_ACCENT, alpha=0.12, zorder=1)
        ax2.axvspan(cs_dt, ce_dt, color=RED_ACCENT, alpha=0.08, zorder=1)

        # Label with arrow
        mid_dt = cs_dt + (ce_dt - cs_dt) / 2
        mask = (dates >= cs_dt) & (dates <= ce_dt)
        if np.any(mask):
            y_at_start = prices[np.argmax(mask)]
            label_y = y_max * y_frac
            ax1.annotate(
                label, xy=(mid_dt, y_at_start),
                xytext=(mid_dt, label_y),
                color=RED_ACCENT, fontsize=13, ha='center', va='bottom',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED_ACCENT, lw=1.5, alpha=0.6),
            )

    ax1.set_ylabel("SPY Price ($)", color=DARK, fontsize=14, fontweight='bold')
    ax1.set_title(
        "Geometric Signal Detection Across Major Financial Crises (2007\u20132024)",
        color=DARK, fontsize=17, fontweight='bold', pad=12,
    )
    ax1.tick_params(colors=DARK, labelsize=11)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color(LIGHT_GRAY)

    # Signal panel — P(crisis)
    threshold = 0.5
    ax2.fill_between(
        dates, 0, signal, where=signal > threshold,
        color=GOLD, alpha=0.7, zorder=3, label=f'P(crisis) > {threshold}',
    )
    ax2.fill_between(
        dates, 0, signal, where=signal <= threshold,
        color=TEAL, alpha=0.2, zorder=2,
    )
    ax2.plot(dates, signal, color=TEAL, linewidth=0.8, alpha=0.5, zorder=2)
    ax2.axhline(y=threshold, color=GOLD, linestyle='--', linewidth=1.8, alpha=0.8, zorder=4)

    # Place threshold label at a visible location
    mid_idx = len(dates) // 3
    ax2.text(dates[mid_idx], threshold + 0.05, f"Alert threshold ({threshold})",
             color=GOLD, fontsize=12, fontweight='bold', alpha=0.9)

    ax2.set_ylabel("P(crisis)", color=DARK, fontsize=13, fontweight='bold')
    ax2.set_xlabel("Date", color=DARK, fontsize=14, fontweight='bold')
    ax2.tick_params(colors=DARK, labelsize=11)
    ax2.set_ylim(-0.05, 1.05)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color(LIGHT_GRAY)

    ax2.legend(loc='upper right', fontsize=12, facecolor=BG,
               edgecolor=LIGHT_GRAY, labelcolor=DARK)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout(pad=1.0)
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTDIR / f"crisis_timeline.{fmt}", dpi=300, facecolor=BG,
                    bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print("Saved crisis_timeline.png + .pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-results', action='store_true',
                        help='Require POSTER_RESULTS.json (fail if missing)')
    args = parser.parse_args()

    print("Generating poster figures (v3 — real data)...")

    figure_data = _load_figure_data()
    results = _load_results()

    if args.from_results and (figure_data is None or results is None):
        print("ERROR: POSTER_RESULTS.json or POSTER_FIGURE_DATA.json not found.")
        print("Run: python experiments/poster_evaluation.py first.")
        exit(1)

    fig1_method_comparison(results)
    fig2_crisis_timeline(figure_data)
    print(f"All figures saved to {OUTDIR}/")
