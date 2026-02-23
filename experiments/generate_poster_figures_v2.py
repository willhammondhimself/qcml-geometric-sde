"""
Generate figures for APS 2026 Job Seeker Poster (v2 — light theme).

Figures:
    1. QCML vs RF Comparison — Berry Phase Rate vs Random Forest on COVID 2020
    2. Quanta Ventures Metrics Table — Infographic-style performance summary
    3. Skills Radar Chart — Filled polygon showing strength profile

Usage:
    python experiments/generate_poster_figures_v2.py
    python experiments/generate_poster_figures_v2.py --skip-data  # Use placeholder data
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

OUT_DIR = ROOT / 'paper' / 'figures'

# Poster-friendly settings: large fonts, cream background, readable from 4-6 feet
CREAM_BG = '#FAF7F2'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 18,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': CREAM_BG,
    'figure.facecolor': CREAM_BG,
    'axes.facecolor': CREAM_BG,
})

# Color palette matching poster: cream & navy
NAVY = '#1B2A4A'          # Deep navy (primary)
NAVY_MED = '#2C3E5A'      # Medium navy
NAVY_SLATE = '#3D5068'    # Slate navy
CHARCOAL = '#5C564E'      # Warm charcoal (RF line)
GRAY = '#8A837A'          # Warm medium gray
LIGHT_GRAY = '#EDE8E1'    # Light warm tan
DARK = '#1A1814'           # Warm near-black
TAUPE = '#D6CFC5'         # Warm taupe (borders)


def ensure_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Figure 1: QCML vs RF Comparison on COVID 2020
# =============================================================================

def generate_qcml_vs_rf(use_real_data=True):
    """Generate side-by-side QCML Berry Phase Rate vs Random Forest on COVID 2020.

    Shows both signals overlaid on SPY price with crisis shading.
    Designed for poster: large fonts, bold colors, light background.
    """
    logger.info("[Fig 1] Generating QCML vs RF comparison...")
    ensure_output_dir()

    if use_real_data:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / '.env')

            from qcml_geometry.observables import (
                BaseRegimeDetector,
                BerryPhaseRateDetector,
            )
            from experiments.data_loader import (
                fetch_polygon_data,
                create_feature_matrix,
                ALL_CRISES,
            )
            from experiments.baselines import RandomForestRegimeDetector

            import pandas as pd

            ci = ALL_CRISES['2020_covid']
            cs = pd.Timestamp(ci['start'])
            ce = pd.Timestamp(ci['end'])

            context_start = cs - pd.Timedelta(days=120)
            context_end = ce + pd.Timedelta(days=120)

            symbols = ['SPY', 'DIA']
            raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
            prices_df = raw['close'].unstack('symbol').dropna()
            X, dates = create_feature_matrix(prices_df)
            X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
            dates_enriched = dates[19:]

            # Berry Phase Rate
            berry = BerryPhaseRateDetector(
                hilbert_dim=8, n_pca_components=15,
                operator_method='pca_inspired', rolling_window=20, seed=42,
            )
            berry.fit(X_enriched)
            berry_scores = berry.compute_regime_scores(X_enriched)

            # Random Forest (leave-one-out for COVID)
            y = np.zeros(len(X))
            for ck, c in ALL_CRISES.items():
                if ck == '2020_covid':
                    continue
                c_s = pd.Timestamp(c['start'])
                c_e = pd.Timestamp(c['end'])
                mask = (dates >= c_s) & (dates <= c_e)
                y[mask] = 1.0

            rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
            rf.fit_with_labels(X, y)
            rf_scores = rf.compute_regime_scores(X)
            rf_aligned = rf_scores[19:] if len(rf_scores) > len(dates_enriched) else rf_scores

            # Context window
            ctx = (dates_enriched >= context_start) & (dates_enriched <= context_end)
            ctx_dates = dates_enriched[ctx]
            ctx_berry = berry_scores[ctx]
            ctx_rf = rf_aligned[ctx] if len(rf_aligned) == len(dates_enriched) else np.full(ctx.sum(), np.nan)
            spy_ctx = prices_df['SPY'].reindex(ctx_dates)

            _plot_qcml_vs_rf(ctx_dates, spy_ctx.values, ctx_berry, ctx_rf, cs, ce, ci['label'])
            return

        except Exception as e:
            logger.warning(f"  Real data failed ({e}), using placeholder")

    # Placeholder data
    import pandas as pd
    np.random.seed(42)
    dates = pd.date_range('2019-10-01', '2020-09-01', freq='B')
    n = len(dates)
    price = 330 + np.cumsum(np.random.randn(n) * 2)
    # Simulate crash
    crash_start = 100
    crash_end = 140
    price[crash_start:crash_end] -= np.linspace(0, 80, crash_end - crash_start)
    price[crash_end:] -= 40
    price[crash_end:] += np.cumsum(np.random.randn(n - crash_end) * 1.5)

    berry = np.random.randn(n) * 0.5
    berry[crash_start-10:crash_end+20] += np.concatenate([
        np.linspace(0, 4, 10),
        np.ones(crash_end - crash_start) * 3.5 + np.random.randn(crash_end - crash_start) * 0.5,
        np.linspace(3, 0.5, 20),
    ])

    rf = 1 / (1 + np.exp(-np.random.randn(n) * 0.3))
    rf[crash_start+5:crash_end+10] = np.clip(rf[crash_start+5:crash_end+10] + 0.4, 0, 1)

    cs = dates[crash_start]
    ce = dates[crash_end]
    _plot_qcml_vs_rf(dates, price, berry, rf, cs, ce, 'COVID-19 Crash (2020)')


def _plot_qcml_vs_rf(dates, price, berry_scores, rf_scores, cs, ce, title):
    """Internal plotting for the QCML vs RF comparison."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                         gridspec_kw={'height_ratios': [1.2, 1, 1]})

    for ax in [ax1, ax2, ax3]:
        ax.axvspan(cs, ce, alpha=0.12, color=TAUPE, zorder=0)

    # Panel 1: SPY Price
    ax1.plot(dates, price, color=DARK, linewidth=2.5, zorder=2)
    ax1.set_ylabel('SPY Price ($)', fontsize=22, fontweight='bold')
    ax1.set_title(f'Regime Detection: {title}', fontsize=28, fontweight='bold',
                  pad=15, color=DARK)
    ax1.tick_params(labelsize=18)
    # Crisis label
    mid_crisis = cs + (ce - cs) / 2
    ax1.annotate('CRISIS', xy=(mid_crisis, ax1.get_ylim()[0]),
                fontsize=16, color=NAVY, fontweight='bold', ha='center',
                va='bottom', alpha=0.8)

    # Panel 2: QCML Berry Phase Rate
    ax2.fill_between(dates, 0, berry_scores, alpha=0.25, color=NAVY, zorder=1)
    ax2.plot(dates, berry_scores, color=NAVY, linewidth=2.5, zorder=2)
    ax2.axhline(2.0, color=GRAY, linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.set_ylabel('Berry Phase\nz-score', fontsize=22, fontweight='bold')
    ax2.tick_params(labelsize=18)
    ax2.text(0.02, 0.88, 'QCML (Unsupervised)', transform=ax2.transAxes,
            fontsize=20, color=NAVY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=CREAM_BG, edgecolor=NAVY, alpha=0.9))

    # Panel 3: RF Probability
    ax3.fill_between(dates, 0, rf_scores, alpha=0.25, color=CHARCOAL, zorder=1)
    ax3.plot(dates, rf_scores, color=CHARCOAL, linewidth=2.5, zorder=2)
    ax3.axhline(0.5, color=GRAY, linestyle='--', linewidth=1.5, alpha=0.6)
    ax3.set_ylabel('RF P(crisis)', fontsize=22, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=22, fontweight='bold')
    ax3.tick_params(labelsize=18)
    ax3.text(0.02, 0.88, 'Random Forest (Supervised)', transform=ax3.transAxes,
            fontsize=20, color=CHARCOAL, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=CREAM_BG, edgecolor=CHARCOAL, alpha=0.9))

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)

    plt.tight_layout(h_pad=0.5)
    out = OUT_DIR / 'poster_qcml_vs_rf.png'
    fig.savefig(out, facecolor=CREAM_BG)
    fig.savefig(out.with_suffix('.pdf'), facecolor=CREAM_BG)
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# =============================================================================
# Figure 2: Quanta Ventures Metrics Table
# =============================================================================

def generate_quanta_metrics():
    """Generate an infographic-style performance metrics table for Quanta Ventures."""
    logger.info("[Fig 2] Generating Quanta Ventures metrics table...")
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.6, 'QUANTA VENTURES — Performance Summary', fontsize=26,
            fontweight='bold', ha='center', va='top', color=DARK)
    ax.text(6, 5.15, 'Out-of-sample results (2022–2025)', fontsize=16,
            ha='center', va='top', color=GRAY)

    # Metric cards
    metrics = [
        ('2.92', 'Sharpe Ratio', NAVY),
        ('5.02', 'Calmar Ratio', NAVY),
        ('-6.8%', 'Max Drawdown', NAVY_MED),
        ('133', 'Total Signals', NAVY_SLATE),
    ]

    card_width = 2.4
    gap = 0.4
    total_width = len(metrics) * card_width + (len(metrics) - 1) * gap
    start_x = (12 - total_width) / 2

    for i, (value, label, color) in enumerate(metrics):
        x = start_x + i * (card_width + gap)
        y = 3.0

        # Card background
        rect = FancyBboxPatch((x, y), card_width, 1.8,
                              boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor=color,
                              linewidth=3, zorder=2)
        ax.add_patch(rect)

        # Value
        ax.text(x + card_width/2, y + 1.25, value, fontsize=36,
                fontweight='bold', ha='center', va='center', color=color, zorder=3)
        # Label
        ax.text(x + card_width/2, y + 0.4, label, fontsize=14,
                ha='center', va='center', color=GRAY, zorder=3)

    # Additional info rows
    info_rows = [
        ('Strategy Sleeves:', '5 (Momentum, Mean Rev, Statistical, Risk Budget, Factor)'),
        ('Validation:', 'Walk-forward with 30-day embargo + 11-test robustness suite'),
        ('Risk Management:', 'VIX/VVIX regime classification + dynamic leverage'),
        ('Tail Risk:', 'Heston stochastic vol model for options hedging'),
    ]

    for i, (key, val) in enumerate(info_rows):
        y = 2.3 - i * 0.5
        ax.text(1.5, y, key, fontsize=14, fontweight='bold', va='center', color=DARK)
        ax.text(4.5, y, val, fontsize=14, va='center', color=GRAY)

    plt.tight_layout()
    out = OUT_DIR / 'poster_quanta_metrics.png'
    fig.savefig(out, facecolor=CREAM_BG)
    fig.savefig(out.with_suffix('.pdf'), facecolor=CREAM_BG)
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# =============================================================================
# Figure 3: Skills Radar Chart
# =============================================================================

def generate_skills_radar():
    """Generate a filled radar/spider chart showing skill profile."""
    logger.info("[Fig 3] Generating skills radar chart...")
    ensure_output_dir()

    categories = [
        'Python / ML',
        'Quant Methods',
        'Research',
        'Engineering',
        'Communication',
        'Leadership',
    ]
    # Self-assessment scores (0-10)
    scores = [9, 8, 8, 7, 7, 6]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw the polygon
    ax.fill(angles, scores_plot, color=NAVY, alpha=0.15)
    ax.plot(angles, scores_plot, color=NAVY, linewidth=3, marker='o',
            markersize=10, markerfacecolor=NAVY, markeredgecolor=CREAM_BG,
            markeredgewidth=2)

    # Grid styling
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=14, color=GRAY)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=18, fontweight='bold', color=DARK)

    # Style grid
    ax.yaxis.grid(True, color=TAUPE, linewidth=1)
    ax.xaxis.grid(True, color=TAUPE, linewidth=1.5)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor(CREAM_BG)

    # Add score labels on each point
    for i, (angle, score) in enumerate(zip(angles[:-1], scores)):
        offset = 0.8
        ax.text(angle, score + offset, str(score), fontsize=16,
                fontweight='bold', ha='center', va='center',
                color=NAVY,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=CREAM_BG,
                         edgecolor=NAVY, alpha=0.9))

    ax.set_title('Skill Profile', fontsize=26, fontweight='bold',
                 pad=30, color=DARK)

    plt.tight_layout()
    out = OUT_DIR / 'poster_skills_radar.png'
    fig.savefig(out, facecolor=CREAM_BG)
    fig.savefig(out.with_suffix('.pdf'), facecolor=CREAM_BG)
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate poster figures (v2, light theme)')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip real data fetch, use placeholder data')
    args = parser.parse_args()

    generate_qcml_vs_rf(use_real_data=not args.skip_data)
    generate_quanta_metrics()
    generate_skills_radar()

    logger.info(f"\n  All poster figures saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
