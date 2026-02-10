"""
Interaction test for method-type × crisis-type complementarity (Review Item 8).

Pre-defines crisis categories (novel vs conventional) BEFORE seeing results.
Runs 2-way ANOVA: method_type × crisis_type on Cohen's d.
Reports F-statistic for interaction, p-value, partial eta-squared.

Design: 5 geometric + 5 classical UNSUPERVISED methods (no RF).
All 12 crises × 10 methods = 120 observations.

Usage:
    python experiments/interaction_test.py
    python experiments/interaction_test.py --quick
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import (
    fetch_polygon_data, create_feature_matrix, ALL_CRISES, CRISIS_CATEGORIES,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    BOCPDDetector,
    IsolationForestDetector,
)
from experiments.additional_detectors import (
    QCMLChernDetector,
    GeometricConsensusDetector,
)
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)

# Method-type definitions (5 geometric + 5 classical unsupervised, no RF)
GEOMETRIC_METHODS = [
    'Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity',
    'QCML Chern', 'Geometric Consensus',
]
CLASSICAL_METHODS = [
    'Rolling Vol Z', 'CUSUM', 'HMM', 'BOCPD', 'Isolation Forest',
]


def run_interaction_test(quick=False):
    """Run 2-way ANOVA: method_type × crisis_type.

    Args:
        quick: Only use 4 crises (2 novel, 2 conventional).
    """
    logger.info("=" * 70)
    logger.info("Interaction Test: Method-Type × Crisis-Type ANOVA")
    logger.info("5 geometric + 5 classical unsupervised (no RF)")
    logger.info("=" * 70)

    # Fetch data
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Enriched features: {X_enriched.shape}")

    if quick:
        novel_crises = ['2018_volmageddon', '2023_svb']
        conventional_crises = ['2008_gfc', '2020_covid']
    else:
        novel_crises = CRISIS_CATEGORIES['novel']
        conventional_crises = CRISIS_CATEGORIES['conventional']

    all_crises = novel_crises + conventional_crises

    # Build and fit geometric methods (5)
    logger.info("\n[2] Fitting geometric methods...")
    common_qcml = dict(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
        rolling_window=20, seed=42,
    )
    geometric_detectors = {
        'Berry Phase Rate': BerryPhaseRateDetector(**common_qcml),
        'QFI Determinant': QFIDeterminantDetector(**common_qcml),
        'Multi-Lag Fidelity': MultiLagFidelityDetector(**common_qcml),
        'QCML Chern': QCMLChernDetector(
            hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
            window_size=20, seed=42,
        ),
        'Geometric Consensus': GeometricConsensusDetector(
            hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
            rolling_window=20, seed=42,
        ),
    }

    geo_scores = {}
    for name, det in geometric_detectors.items():
        logger.info(f"  {name}...")
        det.fit(X_enriched)
        geo_scores[name] = det.compute_regime_scores(X_enriched)

    # Build and fit classical unsupervised methods (5)
    logger.info("\n[3] Fitting classical unsupervised methods...")
    classical_detectors = {
        'Rolling Vol Z': RollingVolatilityDetector(vol_window=20, min_expanding=60),
        'CUSUM': CUSUMDetector(burn_in=60),
        'HMM': HMMRegimeDetector(n_iter=100, seed=42),
        'BOCPD': BOCPDDetector(hazard_rate=250.0),
        'Isolation Forest': IsolationForestDetector(n_estimators=100, seed=42),
    }

    classical_scores = {}
    for name, det in classical_detectors.items():
        logger.info(f"  {name}...")
        det.fit(X_enriched)
        classical_scores[name] = det.compute_regime_scores(X_enriched)

    # Compute Cohen's d for each method × crisis
    logger.info("\n[4] Computing Cohen's d...")
    d_values = []  # List of (method_type, crisis_type, method_name, crisis_key, d)

    window_ext = 10

    def compute_d(scores, ck):
        ci = ALL_CRISES[ck]
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_ext)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_ext)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask
        d, _, _ = compute_cohens_d_with_ci(
            scores[crisis_mask], scores[normal_mask], n_bootstrap=1000
        )
        return float(d) if not np.isnan(d) else 0.0

    for mname, scores in geo_scores.items():
        for ck in all_crises:
            crisis_type = 'novel' if ck in novel_crises else 'conventional'
            d = compute_d(scores, ck)
            d_values.append(('geometric', crisis_type, mname, ck, d))

    for mname, scores in classical_scores.items():
        for ck in all_crises:
            crisis_type = 'novel' if ck in novel_crises else 'conventional'
            d = compute_d(scores, ck)
            d_values.append(('classical', crisis_type, mname, ck, d))

    # Build DataFrame
    df = pd.DataFrame(d_values, columns=['method_type', 'crisis_type', 'method', 'crisis', 'd'])

    logger.info(f"\n  Total observations: {len(df)}")
    logger.info(f"  Method types: {df['method_type'].value_counts().to_dict()}")
    logger.info(f"  Crisis types: {df['crisis_type'].value_counts().to_dict()}")

    # ---- 2-way ANOVA ----
    logger.info("\n[5] Running 2-way ANOVA...")

    # Manual sum-of-squares calculation
    grand_mean = df['d'].mean()

    # Main effects
    method_means = df.groupby('method_type')['d'].mean()
    crisis_means = df.groupby('crisis_type')['d'].mean()

    # Interaction means
    cell_means = df.groupby(['method_type', 'crisis_type'])['d'].mean()
    cell_counts = df.groupby(['method_type', 'crisis_type'])['d'].count()

    # SS Total
    ss_total = np.sum((df['d'] - grand_mean) ** 2)

    # SS Method Type
    ss_method = 0
    for mt, mm in method_means.items():
        n_mt = len(df[df['method_type'] == mt])
        ss_method += n_mt * (mm - grand_mean) ** 2

    # SS Crisis Type
    ss_crisis = 0
    for ct, cm in crisis_means.items():
        n_ct = len(df[df['crisis_type'] == ct])
        ss_crisis += n_ct * (cm - grand_mean) ** 2

    # SS Interaction
    ss_interaction = 0
    for (mt, ct), cell_mean in cell_means.items():
        n_cell = cell_counts[(mt, ct)]
        expected = method_means[mt] + crisis_means[ct] - grand_mean
        ss_interaction += n_cell * (cell_mean - expected) ** 2

    # SS Error
    ss_error = ss_total - ss_method - ss_crisis - ss_interaction

    # Degrees of freedom
    a = len(method_means)  # 2 method types
    b = len(crisis_means)  # 2 crisis types
    N = len(df)
    df_method = a - 1
    df_crisis = b - 1
    df_interaction = df_method * df_crisis
    df_error = N - a * b

    # Mean squares
    ms_method = ss_method / max(df_method, 1)
    ms_crisis = ss_crisis / max(df_crisis, 1)
    ms_interaction = ss_interaction / max(df_interaction, 1)
    ms_error = ss_error / max(df_error, 1)

    # F statistics
    f_method = ms_method / max(ms_error, 1e-12)
    f_crisis = ms_crisis / max(ms_error, 1e-12)
    f_interaction = ms_interaction / max(ms_error, 1e-12)

    # P-values
    p_method = 1 - spstats.f.cdf(f_method, df_method, df_error)
    p_crisis = 1 - spstats.f.cdf(f_crisis, df_crisis, df_error)
    p_interaction = 1 - spstats.f.cdf(f_interaction, df_interaction, df_error)

    # Partial eta-squared
    eta2_method = ss_method / (ss_method + ss_error)
    eta2_crisis = ss_crisis / (ss_crisis + ss_error)
    eta2_interaction = ss_interaction / (ss_interaction + ss_error)

    # Print results
    logger.info("\n  ANOVA Table:")
    logger.info(f"  {'Source':20s}  {'SS':>8s}  {'df':>4s}  {'MS':>8s}  {'F':>8s}  {'p':>8s}  {'eta2':>6s}")
    logger.info(f"  {'Method Type':20s}  {ss_method:8.3f}  {df_method:4d}  {ms_method:8.3f}  {f_method:8.3f}  {p_method:8.4f}  {eta2_method:6.3f}")
    logger.info(f"  {'Crisis Type':20s}  {ss_crisis:8.3f}  {df_crisis:4d}  {ms_crisis:8.3f}  {f_crisis:8.3f}  {p_crisis:8.4f}  {eta2_crisis:6.3f}")
    logger.info(f"  {'Interaction':20s}  {ss_interaction:8.3f}  {df_interaction:4d}  {ms_interaction:8.3f}  {f_interaction:8.3f}  {p_interaction:8.4f}  {eta2_interaction:6.3f}")
    logger.info(f"  {'Error':20s}  {ss_error:8.3f}  {df_error:4d}  {ms_error:8.3f}")
    logger.info(f"  {'Total':20s}  {ss_total:8.3f}  {N-1:4d}")

    # Cell means
    logger.info("\n  Cell means (d):")
    for (mt, ct), mean_d in cell_means.items():
        logger.info(f"    {mt:12s} × {ct:14s}: d = {mean_d:.3f}")

    # Key finding
    logger.info(f"\n  INTERACTION: F({df_interaction},{df_error}) = {f_interaction:.3f}, "
               f"p = {p_interaction:.4f}, eta2 = {eta2_interaction:.3f}")

    if p_interaction < 0.05:
        logger.info("  => Significant interaction: method effectiveness depends on crisis type")
    else:
        logger.info("  => No significant interaction at alpha=0.05")

    # ---- Grouped bar chart ----
    logger.info("\n[6] Generating interaction bar chart...")
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'interaction_test'
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))

        method_types = ['geometric', 'classical']
        crisis_types = ['novel', 'conventional']
        x = np.arange(len(method_types))
        width = 0.35

        # Compute means and SEMs for each cell
        means = []
        sems = []
        for ct in crisis_types:
            ct_means = []
            ct_sems = []
            for mt in method_types:
                cell = df[(df['method_type'] == mt) & (df['crisis_type'] == ct)]['d']
                ct_means.append(cell.mean())
                ct_sems.append(cell.std() / np.sqrt(len(cell)))
            means.append(ct_means)
            sems.append(ct_sems)

        bars1 = ax.bar(x - width / 2, means[0], width, yerr=sems[0],
                        label='Novel', capsize=4, color='#4C72B0', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, means[1], width, yerr=sems[1],
                        label='Conventional', capsize=4, color='#DD8452', edgecolor='black', linewidth=0.5)

        ax.set_ylabel("Mean Cohen's $d$", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['Geometric', 'Classical'], fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'Interaction: $F$ = {f_interaction:.2f}, $p$ = {p_interaction:.3f}',
                     fontsize=11)

        plt.tight_layout()
        fig.savefig(out_dir / 'interaction_barplot.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / 'interaction_barplot.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Bar chart saved to {out_dir / 'interaction_barplot.pdf'}")
    except Exception as e:
        logger.warning(f"  Could not generate bar chart: {e}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'novel_crises': novel_crises,
            'conventional_crises': conventional_crises,
            'geometric_methods': GEOMETRIC_METHODS,
            'classical_methods': CLASSICAL_METHODS,
            'quick': quick,
        },
        'anova': {
            'method_type': {'F': f_method, 'p': p_method, 'eta2': eta2_method, 'ss': ss_method, 'df': df_method},
            'crisis_type': {'F': f_crisis, 'p': p_crisis, 'eta2': eta2_crisis, 'ss': ss_crisis, 'df': df_crisis},
            'interaction': {'F': f_interaction, 'p': p_interaction, 'eta2': eta2_interaction, 'ss': ss_interaction, 'df': df_interaction},
            'error': {'ss': ss_error, 'df': df_error},
        },
        'cell_means': {f'{mt}_{ct}': float(m) for (mt, ct), m in cell_means.items()},
        'd_values': d_values,
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'interaction_test_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Interaction test: method_type × crisis_type')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_interaction_test(quick=args.quick)


if __name__ == '__main__':
    main()
