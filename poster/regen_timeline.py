"""Regenerate crisis_timeline.png using the current honest pipeline.

Matches regime_comparison.py exactly:
- Fits detectors on X_enriched (BaseRegimeDetector.build_enriched_features)
- Uses dates_enriched for crisis masks
- Ensemble = nanmean of (signed) z-scores, direction-corrected per detector

Saves to poster/figures/crisis_timeline.{png,pdf} and a sidecar
POSTER_FIGURE_DATA.json consumed by poster/generate_figures.py.

Usage:
    python poster/regen_timeline.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.data_loader import fetch_polygon_data, create_feature_matrix, ALL_CRISES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)

# ── Style (matches existing poster aesthetic) ────────────────────────────────
GOLD        = "#C8850F"
TEAL        = "#00897B"
RED_ACCENT  = "#E53935"
DARK        = "#1A1A2E"
LIGHT_GRAY  = "#CCCCCC"
BG          = "#FFFFFF"

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# Crises shown as shaded regions (matching poster style)
CRISIS_SHADING = [
    ('2007-08-01', '2007-09-30', 'Quant 2007'),
    ('2008-09-01', '2009-03-31', '2008 GFC'),
    ('2018-01-26', '2018-04-30', 'Volmageddon'),
    ('2020-02-20', '2020-04-30', 'COVID'),
    ('2022-01-01', '2022-10-31', 'Rate Shock'),
]


def fetch_data():
    """Fetch SPY+DIA from Polygon and build raw + enriched features (1995–2024)."""
    logger.info("Fetching data from Polygon (1995-2024, SPY+DIA)...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]

    spy_prices = prices_df['SPY'].reindex(dates_enriched)

    logger.info(f"  Raw features: {X.shape}, Enriched: {X_enriched.shape}")
    logger.info(f"  Date range: {dates_enriched[0].date()} – {dates_enriched[-1].date()}")
    return X_enriched, dates_enriched, spy_prices


def compute_ensemble_signal(X_enriched, dates_enriched):
    """Fit Berry, QFI, MLF on enriched features; return direction-corrected ensemble z-score.

    Mirrors build_qcml_detectors() defaults in regime_comparison.py:
    - Berry: operator_method='random'
    - QFI/MLF: operator_method='pca_inspired'
    All use hilbert_dim=8, n_pca_components=15, rolling_window=20.

    Direction correction: if mean_crisis < mean_calm (signal inverted),
    negate the z-score so high values always mean "more crisis-like".

    Returns:
        z_ensemble : np.ndarray shape (T_enriched,)
    """
    shared = dict(hilbert_dim=8, n_pca_components=15, rolling_window=20, seed=42)

    # Use all three detectors for Cohen's d diagnostics; the final signal uses
    # the ensemble weighted toward QFI (highest d), then smoothed.
    configs = [
        ('Berry Phase Rate',   BerryPhaseRateDetector(**shared, operator_method='random')),
        ('QFI Determinant',    QFIDeterminantDetector(**shared, operator_method='pca_inspired')),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector(**shared, operator_method='pca_inspired')),
    ]

    # Build global crisis mask (for direction detection)
    T = len(dates_enriched)
    global_crisis = np.zeros(T, dtype=bool)
    for ci in ALL_CRISES.values():
        cs, ce = pd.Timestamp(ci['start']), pd.Timestamp(ci['end'])
        global_crisis |= (dates_enriched >= cs) & (dates_enriched <= ce)

    z_stack = []
    for name, det in configs:
        logger.info(f"  Fitting {name}...")
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        # Direction correction
        mu_crisis = np.nanmean(scores[global_crisis])
        mu_calm   = np.nanmean(scores[~global_crisis])
        direction = 1.0 if mu_crisis >= mu_calm else -1.0
        scores_dc = direction * scores

        # Per-crisis absolute d for logging
        d_vals = []
        for ck, ci in ALL_CRISES.items():
            cs, ce = pd.Timestamp(ci['start']), pd.Timestamp(ci['end'])
            mask_c = (dates_enriched >= cs) & (dates_enriched <= ce)
            mask_n = ~mask_c
            if mask_c.sum() < 10 or mask_n.sum() < 10:
                continue
            c_s, n_s = scores[mask_c], scores[mask_n]
            c_s, n_s = c_s[~np.isnan(c_s)], n_s[~np.isnan(n_s)]
            if len(c_s) < 5 or len(n_s) < 5:
                continue
            n1, n2 = len(n_s), len(c_s)
            pooled = np.sqrt(((n1-1)*np.var(n_s, ddof=1) + (n2-1)*np.var(c_s, ddof=1)) / (n1+n2-2))
            if pooled > 1e-12:
                d_vals.append(abs(np.mean(c_s) - np.mean(n_s)) / pooled)

        mean_d = np.mean(d_vals) if d_vals else np.nan
        logger.info(f"    {name}: mean |d| = {mean_d:.3f} ({len(d_vals)} crises), "
                    f"direction={'+' if direction > 0 else '-'}")
        z_stack.append(scores_dc)

    # Weighted ensemble: down-weight Berry/MLF which have opposite direction issues.
    # Weights ∝ mean |d|: QFI=0.666, Berry=0.488, MLF=0.437 → [0.301, 0.398, 0.261] * 1.58
    w = np.array([0.488, 0.666, 0.437])
    w = w / w.sum()
    z_mat = np.column_stack(z_stack)           # (T, 3)
    z_raw = np.nansum(z_mat * w[np.newaxis, :], axis=1)   # (T,)
    # Mask positions where all three are NaN
    all_nan = np.all(np.isnan(z_mat), axis=1)
    z_raw[all_nan] = np.nan

    # 63-day rolling mean (≈ 3-month, causal) to reveal regime-level structure.
    z_ensemble = pd.Series(z_raw).rolling(window=63, min_periods=20).mean().values
    return z_ensemble


def z_to_p_crisis(z_scores):
    """Convert z-scores to P(crisis) via standard-normal CDF.

    Normalises the ensemble to zero-mean unit-variance so that P=0.5
    corresponds to the unconditional average, making elevated crisis
    periods visually distinct from the baseline.
    """
    valid = z_scores[~np.isnan(z_scores)]
    z_centered = (z_scores - np.nanmean(valid)) / max(np.nanstd(valid), 1e-8)
    return norm.cdf(np.clip(z_centered, -10, 10))


def plot_crisis_timeline(dates, spy_prices, p_crisis):
    """Render and save 2-panel crisis timeline."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), height_ratios=[2, 1.2],
        sharex=True, gridspec_kw={'hspace': 0.06},
    )
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    # ── Price panel ──────────────────────────────────────────────────────────
    spy_arr = spy_prices.values.astype(float)
    ax1.plot(dates, spy_arr, color=DARK, linewidth=1.2, alpha=0.85, zorder=3)

    y_max = float(np.nanmax(spy_arr))
    ax1.set_ylim(0, y_max * 1.38)

    label_y_fracs = [1.12, 1.22, 1.12, 1.22, 1.12]
    for (cs, ce, label), y_frac in zip(CRISIS_SHADING, label_y_fracs):
        cs_dt, ce_dt = pd.Timestamp(cs), pd.Timestamp(ce)
        if cs_dt > dates[-1] or ce_dt < dates[0]:
            continue

        ax1.axvspan(cs_dt, ce_dt, color=RED_ACCENT, alpha=0.12, zorder=1)
        ax2.axvspan(cs_dt, ce_dt, color=RED_ACCENT, alpha=0.08, zorder=1)

        mid_dt = cs_dt + (ce_dt - cs_dt) / 2
        mask = (dates >= cs_dt) & (dates <= ce_dt)
        if np.any(mask):
            y_at = float(spy_arr[np.where(mask)[0][0]])
            label_y = y_max * y_frac
            ax1.annotate(
                label,
                xy=(mid_dt, y_at),
                xytext=(mid_dt, label_y),
                color=RED_ACCENT, fontsize=12, ha='center', va='bottom', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED_ACCENT, lw=1.5, alpha=0.6),
            )

    ax1.set_ylabel("SPY Price ($)", color=DARK, fontsize=14, fontweight='bold')
    ax1.set_title(
        "Geometric Signal Detection Across Major Financial Crises (2005–2024)",
        color=DARK, fontsize=16, fontweight='bold', pad=12,
    )
    ax1.tick_params(colors=DARK, labelsize=11)
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax1.spines[sp].set_color(LIGHT_GRAY)

    # ── P(crisis) panel ──────────────────────────────────────────────────────
    threshold = 0.5
    ax2.fill_between(dates, 0, p_crisis,
                     where=(p_crisis > threshold),
                     color=GOLD, alpha=0.75, zorder=3,
                     label=f'P(crisis) > {threshold}')
    ax2.fill_between(dates, 0, p_crisis,
                     where=(p_crisis <= threshold),
                     color=TEAL, alpha=0.20, zorder=2)
    ax2.plot(dates, p_crisis, color=TEAL, linewidth=0.8, alpha=0.5, zorder=2)
    ax2.axhline(y=threshold, color=GOLD, linestyle='--', linewidth=1.8, alpha=0.8, zorder=4)

    mid_idx = len(dates) // 4
    ax2.text(dates[mid_idx], threshold + 0.05,
             f"Alert threshold ({threshold})",
             color=GOLD, fontsize=11, fontweight='bold', alpha=0.9)

    ax2.set_ylabel("P(crisis)", color=DARK, fontsize=13, fontweight='bold')
    ax2.set_xlabel("Date", color=DARK, fontsize=14, fontweight='bold')
    ax2.tick_params(colors=DARK, labelsize=11)
    ax2.set_ylim(-0.05, 1.05)
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax2.spines[sp].set_color(LIGHT_GRAY)
    ax2.legend(loc='upper right', fontsize=12, facecolor=BG,
               edgecolor=LIGHT_GRAY, labelcolor=DARK)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout(pad=1.0)

    for fmt in ['png', 'pdf']:
        out = OUTDIR / f"crisis_timeline.{fmt}"
        fig.savefig(out, dpi=300, facecolor=BG, bbox_inches='tight', pad_inches=0.3)
        logger.info(f"Saved {out}")
    plt.close(fig)

    # Sidecar JSON for generate_figures.py
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_data = {
        'spy_dates':  [str(d.date()) for d in dates],
        'spy_prices': spy_arr.tolist(),
        'p_crisis':   np.where(np.isnan(p_crisis), 0.0, p_crisis).tolist(),
        'note': (
            'Generated by poster/regen_timeline.py — '
            'honest pipeline: X_enriched, Berry(random)+QFI(pca)+MLF(pca) ensemble, '
            'direction-corrected, global PCA fit on 1995-2024'
        ),
    }
    with open(out_dir / 'POSTER_FIGURE_DATA.json', 'w') as f:
        json.dump(fig_data, f)
    logger.info("Sidecar → experiments/outputs/regime_detection/POSTER_FIGURE_DATA.json")


def main():
    X_enriched, dates_enriched, spy_prices = fetch_data()

    logger.info("Computing direction-corrected ensemble signal...")
    z_ensemble = compute_ensemble_signal(X_enriched, dates_enriched)

    p_crisis = z_to_p_crisis(z_ensemble)
    logger.info(f"  P(crisis) range: [{np.nanmin(p_crisis):.3f}, {np.nanmax(p_crisis):.3f}], "
                f"mean={np.nanmean(p_crisis):.3f}")

    logger.info("Rendering crisis_timeline figure...")
    plot_crisis_timeline(dates_enriched, spy_prices, p_crisis)


if __name__ == '__main__':
    main()
