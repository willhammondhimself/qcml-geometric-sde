"""
Generate Paper 1 figures: narrative panels + summary statistics.

Figures produced (Paper 1 scope: 4 geometric observables):
1. narrative_2008_gfc.pdf   — 8-panel crisis anatomy for 2008 GFC
2. narrative_2020_covid.pdf — 8-panel crisis anatomy for 2020 COVID
3. narrative_2022_rates.pdf — 8-panel crisis anatomy for 2022 rate hikes
4. effect_sizes.pdf         — Violin plot of Cohen's d across 12 crises
5. bootstrap_ranks.pdf      — Bootstrap ranking distribution (n=10,000)

Panels: Price, Returns, Berry Phase Rate, Spectral Entropy,
        Reduced Purity, Hamiltonian Sensitivity, Spectral Gap, Combined.

Usage:
    python experiments/generate_paper_figures.py
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.dates import DateFormatter  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / '.env')

from qcml_geometry import (  # noqa: E402
    BerryPhaseRateDetector,
    SpectralEntropyDetector,
    ReducedPurityDetector,
    HamiltonianSensitivityDetector,
)
from qcml_geometry.core import QCMLGeometry  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Seed for reproducibility
np.random.seed(42)

# ============================================================================
# Constants
# ============================================================================

CRISES = {
    '2008_gfc': {
        'label': '2008 Global Financial Crisis',
        'short': 'GFC 2008',
        'start': '2008-09-01', 'end': '2009-03-31',
        'context_start': '2007-06-01', 'context_end': '2009-12-31',
    },
    '2020_covid': {
        'label': '2020 COVID-19 Pandemic Crash',
        'short': 'COVID 2020',
        'start': '2020-02-20', 'end': '2020-04-30',
        'context_start': '2019-06-01', 'context_end': '2020-12-31',
    },
    '2022_rates': {
        'label': '2022 Federal Reserve Rate Hikes',
        'short': 'Rate Hikes 2022',
        'start': '2022-01-01', 'end': '2022-10-31',
        'context_start': '2021-01-01', 'context_end': '2023-06-30',
    },
}

ALL_CRISES = {
    '2007_quant': {'start': '2007-08-01', 'end': '2007-09-30', 'label': 'Quant Crisis 2007'},
    '2008_gfc': {'start': '2008-09-01', 'end': '2009-03-31', 'label': 'GFC 2008'},
    '2010_flash': {'start': '2010-05-01', 'end': '2010-06-30', 'label': 'Flash Crash 2010'},
    '2011_euro': {'start': '2011-07-01', 'end': '2011-10-31', 'label': 'Euro Crisis 2011'},
    '2015_china': {'start': '2015-07-01', 'end': '2015-09-30', 'label': 'China Crash 2015'},
    '2018_volmageddon': {'start': '2018-01-26', 'end': '2018-04-30', 'label': 'Volmageddon 2018'},
    '2018_q4': {'start': '2018-10-01', 'end': '2018-12-31', 'label': 'Q4 Selloff 2018'},
    '2019_repo': {'start': '2019-09-01', 'end': '2019-10-31', 'label': 'Repo Crisis 2019'},
    '2020_covid': {'start': '2020-02-20', 'end': '2020-04-30', 'label': 'COVID 2020'},
    '2022_rates': {'start': '2022-01-01', 'end': '2022-10-31', 'label': 'Rate Hikes 2022'},
    '2023_svb': {'start': '2023-03-01', 'end': '2023-04-30', 'label': 'SVB 2023'},
    '2024_carry': {'start': '2024-07-15', 'end': '2024-08-31', 'label': 'Carry Unwind 2024'},
}

from experiments.plot_style import (  # noqa: E402
    apply_style, NAVY, TEAL, BURGUNDY, GOLD, INDIGO, SLATE,
)
apply_style()

COLORS = {
    'berry': BURGUNDY,
    'spectral_entropy': NAVY,
    'reduced_purity': TEAL,
    'hamiltonian_sensitivity': GOLD,
    'spectral_gap': INDIGO,
    'energy': SLATE,
    'price': NAVY,
    'returns': SLATE,
    'crisis': BURGUNDY,
}


# ============================================================================
# Data Fetching (via unified dispatcher)
# ============================================================================

from experiments.data_loader import fetch_data  # noqa: E402


def create_feature_matrix(prices_df):
    """Create a minimal feature matrix from a close-price DataFrame.

    Args:
        prices_df: DataFrame with columns = symbols, index = dates, values = close prices.

    Returns:
        features: np.ndarray (T', d) after warmup.
        dates: DatetimeIndex aligned with features.
    """
    # Use log returns and rolling statistics
    log_ret = np.log(prices_df / prices_df.shift(1))

    features_dict = {}
    for col in prices_df.columns:
        features_dict[f'{col}_ret'] = log_ret[col]
        features_dict[f'{col}_vol5'] = log_ret[col].rolling(5).std()
        features_dict[f'{col}_vol20'] = log_ret[col].rolling(20).std()
        features_dict[f'{col}_mom5'] = prices_df[col].pct_change(5)
        features_dict[f'{col}_mom20'] = prices_df[col].pct_change(20)

    # Cross-sectional features if multiple symbols
    if len(prices_df.columns) > 1:
        features_dict['cross_corr5'] = log_ret.rolling(5).corr().groupby(level=0).mean().mean(axis=1)
        features_dict['cross_vol_disp'] = log_ret.rolling(20).std().std(axis=1)
        features_dict['avg_ret'] = log_ret.mean(axis=1)

    feat_df = pd.DataFrame(features_dict)
    feat_df = feat_df.dropna()

    return feat_df.values, feat_df.index


# ============================================================================
# Regime Score Computation
# ============================================================================

def compute_all_scores(X, dates):
    """Compute regime scores for Paper 1's 4 QCML detectors + spectral gap.

    Returns dict of {name: (scores_array, aligned_dates)}.
    """
    results = {}

    # Build enriched features for the detectors
    from qcml_geometry.observables import BaseRegimeDetector
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]  # trim by lookback-1

    detectors = [
        ('Berry Phase Rate', BerryPhaseRateDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42)),
        ('Spectral Entropy', SpectralEntropyDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42)),
        ('Reduced Purity', ReducedPurityDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42)),
        ('Hamiltonian Sensitivity', HamiltonianSensitivityDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42)),
    ]

    for name, det in detectors:
        logger.info(f"  Computing {name}...")
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        results[name] = (scores, dates_enriched)

    # Also compute spectral gap for narrative panels (supports Theorem 1)
    logger.info("  Computing spectral gap...")
    n_pca = min(15, X_enriched.shape[1])
    scaler = StandardScaler().fit(X_enriched)
    pca = PCA(n_components=n_pca).fit(scaler.transform(X_enriched))
    X_pca = pca.transform(scaler.transform(X_enriched))
    X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

    geo = QCMLGeometry(n_features=n_pca, hilbert_dim=8)
    geo.fit_operators(X_pca, method='pca_inspired')

    spectral_gaps = np.empty(len(X_pca))
    for t in range(len(X_pca)):
        H = geo.error_hamiltonian(X_pca[t])
        evals = np.linalg.eigvalsh(H)
        spectral_gaps[t] = evals[1] - evals[0]

    results['Spectral Gap'] = (spectral_gaps, dates_enriched)

    return results


# ============================================================================
# Figure 1-3: Crisis Narrative Figures
# ============================================================================

def generate_narrative_figure(crisis_key, crisis_info, scores_dict, prices_df, output_dir):
    """Generate an 8-panel crisis narrative figure.

    Panels: Price, Returns, Berry Phase Rate, Spectral Entropy,
    Reduced Purity, Hamiltonian Sensitivity, Spectral Gap, Combined overlay.
    """
    apply_style()

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f'QCML Geometric Crisis Anatomy: {crisis_info["label"]}',
                 fontsize=13, fontweight='bold', y=0.98)

    crisis_start = pd.Timestamp(crisis_info['start'])
    crisis_end = pd.Timestamp(crisis_info['end'])
    ctx_start = pd.Timestamp(crisis_info['context_start'])
    ctx_end = pd.Timestamp(crisis_info['context_end'])

    def add_crisis_shading(ax):
        ylim = ax.get_ylim()
        ax.axvspan(crisis_start, crisis_end, alpha=0.15, color=COLORS['crisis'],
                   label='Crisis period')
        ax.set_ylim(ylim)

    def trim_to_context(scores, dates):
        mask = (dates >= ctx_start) & (dates <= ctx_end)
        return scores[mask], dates[mask]

    # Use first symbol for price reference
    spy_col = 'SPY' if 'SPY' in prices_df.columns else prices_df.columns[0]
    price_mask = (prices_df.index >= ctx_start) & (prices_df.index <= ctx_end)
    ctx_prices = prices_df.loc[price_mask, spy_col]
    ctx_returns = np.log(ctx_prices / ctx_prices.shift(1)).dropna()

    # Panel 1: Price
    ax = axes[0, 0]
    ax.plot(ctx_prices.index, ctx_prices.values, color=COLORS['price'], linewidth=1.2)
    ax.set_ylabel('SPY Close ($)')
    ax.set_title('Price')
    add_crisis_shading(ax)

    # Panel 2: Returns
    ax = axes[0, 1]
    ax.bar(ctx_returns.index, ctx_returns.values, color=COLORS['returns'], alpha=0.7, width=1)
    ax.set_ylabel('Log Returns')
    ax.set_title('Daily Returns')
    add_crisis_shading(ax)

    # Panel 3: Berry Phase Rate
    ax = axes[1, 0]
    s, d = scores_dict['Berry Phase Rate']
    s_ctx, d_ctx = trim_to_context(s, d)
    ax.plot(d_ctx, s_ctx, color=COLORS['berry'], linewidth=1.0)
    ax.set_ylabel('Z-Score')
    ax.set_title('Berry Phase Rate')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    add_crisis_shading(ax)

    # Panel 4: Spectral Entropy
    ax = axes[1, 1]
    s, d = scores_dict['Spectral Entropy']
    s_ctx, d_ctx = trim_to_context(s, d)
    ax.plot(d_ctx, s_ctx, color=COLORS['spectral_entropy'], linewidth=1.0)
    ax.set_ylabel('Z-Score')
    ax.set_title('Spectral Entropy')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    add_crisis_shading(ax)

    # Panel 5: Reduced Purity
    ax = axes[2, 0]
    s, d = scores_dict['Reduced Purity']
    s_ctx, d_ctx = trim_to_context(s, d)
    ax.plot(d_ctx, s_ctx, color=COLORS['reduced_purity'], linewidth=1.0)
    ax.set_ylabel('Z-Score')
    ax.set_title('Reduced State Purity')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    add_crisis_shading(ax)

    # Panel 6: Hamiltonian Sensitivity
    ax = axes[2, 1]
    s, d = scores_dict['Hamiltonian Sensitivity']
    s_ctx, d_ctx = trim_to_context(s, d)
    ax.plot(d_ctx, s_ctx, color=COLORS['hamiltonian_sensitivity'], linewidth=1.0)
    ax.set_ylabel('Z-Score')
    ax.set_title('Hamiltonian Sensitivity')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    add_crisis_shading(ax)

    # Panel 7: Spectral Gap (supports Theorem 1)
    ax = axes[3, 0]
    s, d = scores_dict['Spectral Gap']
    s_ctx, d_ctx = trim_to_context(s, d)
    ax.plot(d_ctx, s_ctx, color=COLORS['spectral_gap'], linewidth=1.0)
    ax.set_ylabel('Gap ($\\Delta$)')
    ax.set_title('Spectral Gap')
    add_crisis_shading(ax)

    # Panel 8: Combined signal overlay (4 geometric observables)
    ax = axes[3, 1]
    for name, color_key in [('Berry Phase Rate', 'berry'),
                             ('Spectral Entropy', 'spectral_entropy'),
                             ('Reduced Purity', 'reduced_purity'),
                             ('Hamiltonian Sensitivity', 'hamiltonian_sensitivity')]:
        s, d = scores_dict[name]
        s_ctx, d_ctx = trim_to_context(s, d)
        s_valid = s_ctx[~np.isnan(s_ctx)]
        if len(s_valid) > 0:
            s_norm = (s_ctx - np.nanmin(s_ctx)) / (np.nanmax(s_ctx) - np.nanmin(s_ctx) + 1e-12)
            ax.plot(d_ctx, s_norm, color=COLORS[color_key], linewidth=1.0,
                    label=name, alpha=0.8)
    ax.set_ylabel('Normalized Score')
    ax.set_title('Combined QCML Observables')
    ax.legend(loc='upper left', fontsize=7)
    add_crisis_shading(ax)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save as PDF and PNG
    pdf_path = output_dir / f'narrative_{crisis_key}.pdf'
    png_path = output_dir / f'narrative_{crisis_key}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {pdf_path}")


# ============================================================================
# Figure 4: Effect Sizes Violin Plot
# ============================================================================

def compute_crisis_effect_sizes(X, dates, crisis_dict):
    """Compute Cohen's d for each method on each crisis.

    Returns DataFrame: rows=methods, columns=crises, values=Cohen's d.
    """
    from qcml_geometry.observables import BaseRegimeDetector
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]

    methods = {
        'Berry Phase Rate': BerryPhaseRateDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42),
        'Spectral Entropy': SpectralEntropyDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42),
        'Reduced Purity': ReducedPurityDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42),
        'Hamiltonian Sensitivity': HamiltonianSensitivityDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=20, seed=42),
    }

    # Compute scores for each method
    all_scores = {}
    for name, det in methods.items():
        logger.info(f"  Fitting {name}...")
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        all_scores[name] = scores

    # Compute Cohen's d for each method × crisis
    results = {}
    for name, scores in all_scores.items():
        crisis_ds = {}
        for ckey, cinfo in crisis_dict.items():
            cs = pd.Timestamp(cinfo['start'])
            ce = pd.Timestamp(cinfo['end'])

            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            crisis_scores = scores[crisis_mask]
            normal_scores = scores[normal_mask]

            # Remove NaN
            crisis_scores = crisis_scores[~np.isnan(crisis_scores)]
            normal_scores = normal_scores[~np.isnan(normal_scores)]

            if len(crisis_scores) > 2 and len(normal_scores) > 2:
                pooled_std = np.sqrt(
                    ((len(crisis_scores) - 1) * np.var(crisis_scores, ddof=1) +
                     (len(normal_scores) - 1) * np.var(normal_scores, ddof=1)) /
                    (len(crisis_scores) + len(normal_scores) - 2)
                )
                if pooled_std > 1e-12:
                    d = (np.mean(crisis_scores) - np.mean(normal_scores)) / pooled_std
                    crisis_ds[cinfo['label']] = abs(d)
                else:
                    crisis_ds[cinfo['label']] = 0.0
            else:
                crisis_ds[cinfo['label']] = np.nan

        results[name] = crisis_ds

    return pd.DataFrame(results)


def generate_effect_sizes_figure(d_values_df, output_dir):
    """Generate violin plot of Cohen's d distributions."""
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = d_values_df.columns.tolist()
    data = [d_values_df[m].dropna().values for m in methods]
    positions = range(len(methods))

    colors_list = [COLORS['berry'], COLORS['spectral_entropy'],
                   COLORS['reduced_purity'], COLORS['hamiltonian_sensitivity']]

    parts = ax.violinplot(data, positions=positions, showmeans=True,
                          showmedians=True, widths=0.7)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)

    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('darkred')

    # Overlay individual points
    for i, (d, m) in enumerate(zip(data, methods)):
        jitter = np.random.uniform(-0.1, 0.1, len(d))
        ax.scatter(np.full_like(d, i) + jitter, d, color=colors_list[i],
                   alpha=0.5, s=20, zorder=5)

    # Reference lines
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label="Large effect (d=0.8)")
    ax.axhline(1.13, color='darkred', linestyle=':', alpha=0.5, label="RF baseline (d=1.13)")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Cohen's $d$")
    ax.set_title("Distribution of Effect Sizes Across 12 Crises")
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    pdf_path = output_dir / 'effect_sizes.pdf'
    png_path = output_dir / 'effect_sizes.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {pdf_path}")


# ============================================================================
# Figure 5: Bootstrap Ranking Distribution
# ============================================================================

def generate_bootstrap_ranks_figure(d_values_df, output_dir, n_bootstrap=10000):
    """Generate bootstrap ranking distribution figure."""
    apply_style()

    methods = d_values_df.columns.tolist()
    n_crises = len(d_values_df)
    n_methods = len(methods)

    # Bootstrap: resample crises, rank methods each time
    rank_counts = {m: np.zeros(n_methods) for m in methods}

    rng = np.random.default_rng(42)
    d_matrix = d_values_df.values  # (n_crises, n_methods)

    for _ in range(n_bootstrap):
        idx = rng.choice(n_crises, size=n_crises, replace=True)
        mean_d = np.nanmean(d_matrix[idx], axis=0)
        ranks = stats.rankdata(-mean_d, method='min')  # Higher d = rank 1
        for j, m in enumerate(methods):
            rank_counts[m][int(ranks[j]) - 1] += 1

    # Normalize to probabilities
    for m in methods:
        rank_counts[m] = rank_counts[m] / n_bootstrap

    fig, ax = plt.subplots(figsize=(10, 5))

    colors_list = [COLORS['berry'], COLORS['spectral_entropy'],
                   COLORS['reduced_purity'], COLORS['hamiltonian_sensitivity']]
    x = np.arange(1, n_methods + 1)
    width = 0.2

    for i, (m, c) in enumerate(zip(methods, colors_list)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, rank_counts[m], width=width, color=c,
               alpha=0.7, label=m, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Rank')
    ax.set_ylabel('Bootstrap Probability')
    ax.set_title(f'Bootstrap Ranking Distribution ($n = {n_bootstrap:,}$)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank {r}' for r in x])
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    pdf_path = output_dir / 'bootstrap_ranks.pdf'
    png_path = output_dir / 'bootstrap_ranks.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {pdf_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("QCML Paper Figure Generation")
    logger.info("=" * 70)

    # Create output directories
    narrative_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'narratives'
    summary_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'superiority_publication' / 'figures'
    narrative_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Fetch data ----
    logger.info("\n[Step 1] Fetching SPY + DIA data (2005-2024)...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')

    # Build close price DataFrame
    prices_df = raw['close'].unstack('symbol')
    prices_df = prices_df.dropna()
    logger.info(f"  Price data: {prices_df.shape[0]} rows, {prices_df.shape[1]} symbols")

    # ---- Step 2: Create feature matrix ----
    logger.info("\n[Step 2] Creating feature matrix...")
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"  Feature matrix: {X.shape}")

    # ---- Step 3: Compute scores for narrative figures ----
    logger.info("\n[Step 3] Computing QCML regime scores (all observables)...")
    scores_dict = compute_all_scores(X, dates)

    # ---- Step 4: Generate narrative figures ----
    logger.info("\n[Step 4] Generating crisis narrative figures...")
    for crisis_key, crisis_info in CRISES.items():
        logger.info(f"  Generating {crisis_key}...")
        generate_narrative_figure(crisis_key, crisis_info, scores_dict, prices_df, narrative_dir)

    # ---- Step 5: Compute effect sizes for all 12 crises ----
    logger.info("\n[Step 5] Computing Cohen's d across 12 crises...")
    d_values_df = compute_crisis_effect_sizes(X, dates, ALL_CRISES)
    logger.info(f"  Effect sizes computed:\n{d_values_df.describe()}")

    # ---- Step 6: Generate summary figures ----
    logger.info("\n[Step 6] Generating effect sizes violin plot...")
    generate_effect_sizes_figure(d_values_df, summary_dir)

    logger.info("\n[Step 7] Generating bootstrap ranking distribution...")
    generate_bootstrap_ranks_figure(d_values_df, summary_dir, n_bootstrap=10000)

    logger.info("\n" + "=" * 70)
    logger.info("All 5 figures generated successfully!")
    logger.info(f"  Narratives: {narrative_dir}")
    logger.info(f"  Summaries:  {summary_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
