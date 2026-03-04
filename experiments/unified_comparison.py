"""Unified comparison: single source of truth for all Paper 1 numbers.

Runs all 7 geometric observatory detectors + all baselines across 14 crises
with HPO-optimized configs. Produces:
  - Per-method aggregate Cohen's d + CI
  - Per-crisis d matrix (14 crises x N methods)
  - Per-crisis winner table
  - Crisis taxonomy analysis (5 mechanism categories)
  - Lead time verification
  - LaTeX tables for paper

Usage:
    python experiments/unified_comparison.py              # Full run (~1-2h)
    python experiments/unified_comparison.py --quick      # 4 crises (~10min)
    python experiments/unified_comparison.py --no-rf      # Skip RF (faster)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
    GeodesicVelocityDetector,
    SpeedLimitRatioDetector,
    DimensionalityCollapseDetector,
    SectionalCurvatureDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from qcml_geometry.fusion import RankFusionDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    BOCPDDetector,
    RandomForestRegimeDetector,
)
from experiments.evaluation import (
    compute_cohens_d_with_ci,
    welch_t_test,
    friedman_test,
    holm_bonferroni_correction,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

np.random.seed(42)


# ============================================================================
# Detector Configurations (HPO-optimized from observatory_analysis.py)
# ============================================================================

GEOMETRIC_DETECTORS = {
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
            normalization='sphere', berry_aggregation='f01',
        ),
    },
    'QFI Determinant': {
        'class': QFIDeterminantDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            normalization='soft', qfi_mode='logdet', adaptive_epsilon=True,
        ),
    },
    'Multi-Lag Fidelity': {
        'class': MultiLagFidelityDetector,
        'params': dict(
            hilbert_dim=4, n_pca_components=8, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            normalization='sphere',
        ),
    },
    'Geodesic Velocity': {
        'class': GeodesicVelocityDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=15,
            operator_method='pca_inspired', seed=42,
            normalization='sphere',
        ),
    },
    'Speed Limit Ratio': {
        'class': SpeedLimitRatioDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=42,
            normalization='soft', adaptive_epsilon=True,
        ),
    },
    'Dim. Collapse': {
        'class': DimensionalityCollapseDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=42,
            normalization='soft', adaptive_epsilon=True, subsample=5,
        ),
    },
    'Sect. Curvature': {
        'class': SectionalCurvatureDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=3,
            operator_method='pca_inspired', seed=42,
            normalization='soft', adaptive_epsilon=True,
            score_mode='neg_fraction', neg_fraction_window=20, subsample=10,
        ),
    },
}

BASELINE_DETECTORS = {
    'Rolling Vol Z': {
        'class': RollingVolatilityDetector,
        'params': dict(vol_window=20, min_expanding=60),
    },
    'CUSUM': {
        'class': CUSUMDetector,
        'params': dict(burn_in=60),
    },
    'BOCPD': {
        'class': BOCPDDetector,
        'params': dict(hazard_rate=250.0),
    },
}

# Crisis taxonomy: 5 mechanism-based categories
CRISIS_TAXONOMY = {
    'Volatility Shocks': {
        'crises': ['2018_volmageddon', '2010_flash', '2018_q4'],
        'mechanism': 'Sudden volatility regime shift',
    },
    'Systemic/Credit': {
        'crises': ['2008_gfc', '2011_euro', '2023_svb'],
        'mechanism': 'Credit contagion and bank runs',
    },
    'Exogenous Shocks': {
        'crises': ['2001_911', '2020_covid'],
        'mechanism': 'External trigger, not endogenous',
    },
    'Slow Burns': {
        'crises': ['2000_dotcom', '2022_rates', '2024_carry'],
        'mechanism': 'Gradual regime shift over months',
    },
    'Liquidity/Microstructure': {
        'crises': ['2007_quant', '2019_repo', '2015_china'],
        'mechanism': 'Market structure and liquidity breakdown',
    },
}

# 14 crises for Paper 1
PAPER_CRISES = [
    '2000_dotcom', '2001_911', '2007_quant', '2008_gfc',
    '2010_flash', '2011_euro', '2015_china',
    '2018_volmageddon', '2018_q4', '2019_repo',
    '2020_covid', '2022_rates', '2023_svb', '2024_carry',
]


# ============================================================================
# Core Functions
# ============================================================================

def build_crisis_masks(dates, crises, pad_days=10):
    """Build per-crisis and aggregate masks."""
    crisis_masks = {}
    any_crisis = np.zeros(len(dates), dtype=bool)
    for ck, ci in crises.items():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=pad_days)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=pad_days)
        mask = (dates >= cs) & (dates <= ce)
        crisis_masks[ck] = mask
        any_crisis |= mask
    return crisis_masks, any_crisis


def run_unsupervised_detectors(X, detector_configs):
    """Run all unsupervised detectors. Returns {name: scores}."""
    results = {}
    for name, cfg in detector_configs.items():
        logger.info(f"  Running {name}...")
        det = cfg['class'](**cfg['params'])
        det.fit(X)
        scores = det.compute_regime_scores(X)
        n_valid = np.sum(~np.isnan(scores))
        results[name] = scores
        logger.info(f"    {name}: {n_valid} valid scores")
    return results


def run_rf_leave_one_out(X, dates, crises, n_bootstrap=1000):
    """Run Random Forest with leave-one-crisis-out cross-validation.

    Returns {crisis_key: scores_array} for per-crisis evaluation,
    where each crisis's scores come from an RF trained without that crisis.
    """
    rf_per_crisis_scores = {}

    for held_out_key, ci in crises.items():
        crisis_start = pd.Timestamp(ci['start'])

        # Build labels excluding held-out crisis
        y = np.zeros(len(dates))
        for train_ck, train_ci in crises.items():
            if train_ck == held_out_key:
                continue
            tc_start = pd.Timestamp(train_ci['start'])
            tc_end = pd.Timestamp(train_ci['end'])
            mask = (dates >= tc_start) & (dates <= tc_end)
            y[mask] = 1.0

        # Causal: only train on data before held-out crisis
        cutoff_date = crisis_start - pd.Timedelta(days=30)
        fit_end_idx = int(np.searchsorted(dates, cutoff_date))
        if fit_end_idx < 100:
            logger.warning(f"  RF skipping {held_out_key}: insufficient data")
            continue

        rf = RandomForestRegimeDetector(
            n_estimators=200, max_depth=6, seed=42, lookback=20,
        )
        y_train = y[:fit_end_idx]
        if np.sum(y_train) == 0:
            logger.warning(f"  RF {held_out_key}: no crisis labels in training window")

        rf.fit_with_labels(X[:fit_end_idx], y_train)
        scores = rf.compute_regime_scores(X)
        rf_per_crisis_scores[held_out_key] = scores
        logger.info(f"    RF fitted for {held_out_key}")

    return rf_per_crisis_scores


def compute_per_crisis_d_matrix(score_dict, crisis_masks, any_crisis, n_bootstrap=1000,
                                 rf_scores=None):
    """Compute Cohen's d for each method x crisis combination.

    Args:
        score_dict: {method_name: 1-D scores} for unsupervised methods.
        crisis_masks: {crisis_key: boolean mask}.
        any_crisis: Aggregate boolean mask.
        n_bootstrap: Bootstrap resamples for CI.
        rf_scores: {crisis_key: 1-D scores} from leave-one-out RF.

    Returns:
        d_matrix: dict of {method: {crisis: {d, ci_lo, ci_hi, p_value}}}.
    """
    normal_mask = ~any_crisis
    d_matrix = {}

    # Unsupervised methods: same scores for all crises
    for method_name, scores in score_dict.items():
        d_matrix[method_name] = {}
        valid = ~np.isnan(scores)
        normal_scores = scores[valid & normal_mask]

        for ck, cmask in crisis_masks.items():
            crisis_scores = scores[valid & cmask]
            if len(crisis_scores) < 3 or len(normal_scores) < 10:
                d_matrix[method_name][ck] = {
                    'd': None, 'ci_lo': None, 'ci_hi': None, 'p_value': None,
                }
                continue

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_scores, normal_scores, n_bootstrap=n_bootstrap,
            )
            _, p_val = welch_t_test(crisis_scores, normal_scores)

            d_matrix[method_name][ck] = {
                'd': round(float(d), 3) if not np.isnan(d) else None,
                'ci_lo': round(float(ci_lo), 3) if not np.isnan(ci_lo) else None,
                'ci_hi': round(float(ci_hi), 3) if not np.isnan(ci_hi) else None,
                'p_value': float(p_val) if not np.isnan(p_val) else None,
            }

    # RF: per-crisis scores from leave-one-out
    if rf_scores:
        d_matrix['Random Forest'] = {}
        for ck, cmask in crisis_masks.items():
            if ck not in rf_scores:
                d_matrix['Random Forest'][ck] = {
                    'd': None, 'ci_lo': None, 'ci_hi': None, 'p_value': None,
                }
                continue

            scores = rf_scores[ck]
            valid = ~np.isnan(scores)
            crisis_scores = scores[valid & cmask]
            normal_scores = scores[valid & normal_mask]

            if len(crisis_scores) < 3 or len(normal_scores) < 10:
                d_matrix['Random Forest'][ck] = {
                    'd': None, 'ci_lo': None, 'ci_hi': None, 'p_value': None,
                }
                continue

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_scores, normal_scores, n_bootstrap=n_bootstrap,
            )
            _, p_val = welch_t_test(crisis_scores, normal_scores)

            d_matrix['Random Forest'][ck] = {
                'd': round(float(d), 3) if not np.isnan(d) else None,
                'ci_lo': round(float(ci_lo), 3) if not np.isnan(ci_lo) else None,
                'ci_hi': round(float(ci_hi), 3) if not np.isnan(ci_hi) else None,
                'p_value': float(p_val) if not np.isnan(p_val) else None,
            }

    return d_matrix


def compute_aggregate_stats(d_matrix, crisis_keys):
    """Compute aggregate d, CI, and p-value per method across crises.

    Returns {method: {mean_d, median_d, ci, n_significant, n_crises}}.
    """
    agg = {}
    for method, per_crisis in d_matrix.items():
        d_vals = []
        n_sig = 0
        for ck in crisis_keys:
            entry = per_crisis.get(ck, {})
            d = entry.get('d')
            p = entry.get('p_value')
            if d is not None:
                d_vals.append(d)
            if p is not None and p < 0.05:
                n_sig += 1

        if d_vals:
            agg[method] = {
                'mean_d': round(float(np.mean(d_vals)), 3),
                'median_d': round(float(np.median(d_vals)), 3),
                'std_d': round(float(np.std(d_vals)), 3),
                'min_d': round(float(np.min(d_vals)), 3),
                'max_d': round(float(np.max(d_vals)), 3),
                'n_significant': n_sig,
                'n_crises': len(d_vals),
            }
        else:
            agg[method] = {
                'mean_d': None, 'median_d': None, 'std_d': None,
                'min_d': None, 'max_d': None,
                'n_significant': 0, 'n_crises': 0,
            }
    return agg


def compute_per_crisis_winners(d_matrix, crisis_keys):
    """Find the winning method for each crisis (highest d).

    Returns {crisis_key: {winner, d, all_methods: [{name, d}]}}.
    """
    winners = {}
    for ck in crisis_keys:
        best_d = -np.inf
        best_name = None
        all_methods = []

        for method, per_crisis in d_matrix.items():
            entry = per_crisis.get(ck, {})
            d = entry.get('d')
            if d is not None:
                all_methods.append({'name': method, 'd': d})
                if d > best_d:
                    best_d = d
                    best_name = method

        all_methods.sort(key=lambda x: -x['d'])
        winners[ck] = {
            'winner': best_name,
            'd': round(float(best_d), 3) if best_name else None,
            'all_methods': all_methods,
        }
    return winners


def compute_taxonomy_analysis(d_matrix, crisis_keys):
    """Per-category analysis: mean d per method per crisis category.

    Returns {category: {methods: {method: mean_d}, winner: str}}.
    """
    taxonomy_results = {}
    for cat_name, cat_info in CRISIS_TAXONOMY.items():
        cat_crises = [c for c in cat_info['crises'] if c in crisis_keys]
        if not cat_crises:
            continue

        method_means = {}
        for method, per_crisis in d_matrix.items():
            d_vals = []
            for ck in cat_crises:
                entry = per_crisis.get(ck, {})
                d = entry.get('d')
                if d is not None:
                    d_vals.append(d)
            if d_vals:
                method_means[method] = round(float(np.mean(d_vals)), 3)

        winner = max(method_means, key=method_means.get) if method_means else None

        taxonomy_results[cat_name] = {
            'mechanism': cat_info['mechanism'],
            'crises': cat_crises,
            'methods': method_means,
            'winner': winner,
            'winner_d': method_means.get(winner) if winner else None,
        }
    return taxonomy_results


def compute_lead_times(score_dict, dates, crisis_masks, crises,
                       threshold_z=2.0, lookback_days=120):
    """Compute lead time for each method x crisis.

    Lead time = days before crisis start that the signal first exceeds z=2
    within a lookback window before the crisis.

    Returns {method: {crisis: lead_days or None}}.
    """
    lead_times = {}

    for method_name, scores in score_dict.items():
        lead_times[method_name] = {}
        valid = ~np.isnan(scores)

        # Z-score the scores using expanding window
        z_scores = np.full_like(scores, np.nan)
        for t in range(60, len(scores)):
            past = scores[:t]
            past_valid = past[~np.isnan(past)]
            if len(past_valid) > 10:
                mu = np.mean(past_valid)
                sigma = np.std(past_valid)
                if sigma > 1e-10:
                    z_scores[t] = (scores[t] - mu) / sigma

        for ck in crises:
            crisis_start = pd.Timestamp(crises[ck]['start'])
            lookback_start = crisis_start - pd.Timedelta(days=lookback_days)

            pre_window = (dates >= lookback_start) & (dates < crisis_start) & valid
            alarm_indices = np.where(pre_window & (z_scores > threshold_z))[0]

            if len(alarm_indices) > 0:
                first_alarm_date = dates[alarm_indices[0]]
                lead_days = (crisis_start - first_alarm_date).days
                lead_times[method_name][ck] = int(lead_days)
            else:
                lead_times[method_name][ck] = None

    return lead_times


def compute_lead_time_summary(lead_times):
    """Compute median lead time per method."""
    summary = {}
    for method, per_crisis in lead_times.items():
        vals = [v for v in per_crisis.values() if v is not None]
        summary[method] = {
            'median_days': float(np.median(vals)) if vals else None,
            'mean_days': round(float(np.mean(vals)), 1) if vals else None,
            'n_detected': len(vals),
            'n_crises': len(per_crisis),
        }
    return summary


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_per_crisis_winners_tex(d_matrix, crisis_keys, crises, output_path):
    """Generate LaTeX table: 14 crises x methods with d-values, winners bolded."""
    methods = list(d_matrix.keys())
    geometric_methods = list(GEOMETRIC_DETECTORS.keys())

    # Short method names for table headers
    short_names = {
        'Berry Phase Rate': 'Berry',
        'QFI Determinant': 'QFI',
        'Multi-Lag Fidelity': 'MLF',
        'Geodesic Velocity': 'Geo.Vel',
        'Speed Limit Ratio': 'SLR',
        'Dim. Collapse': 'DimC',
        'Sect. Curvature': 'SeCu',
        'Rolling Vol Z': 'RVol',
        'CUSUM': 'CUSUM',
        'BOCPD': 'BOCPD',
        'Random Forest': 'RF',
    }

    n_cols = len(methods)
    col_spec = 'l' + 'r' * n_cols
    header_row = ' & '.join([short_names.get(m, m) for m in methods])

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r"\caption{Per-crisis Cohen's $d$ for all detection methods. "
        r"Bold indicates the winner for each crisis. "
        r"Geometric channels (left) vs.\ baselines (right).}",
        r'\label{tab:per_crisis_winners}',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{3pt}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        f'Crisis & {header_row} \\\\',
        r'\midrule',
    ]

    for ck in crisis_keys:
        label = crises[ck]['label']
        # Shorten label for table
        label = label.replace(' 20', " '")

        vals = []
        d_vals = []
        for m in methods:
            entry = d_matrix.get(m, {}).get(ck, {})
            d = entry.get('d')
            d_vals.append(d)

        # Find max d for bolding
        valid_d = [d for d in d_vals if d is not None]
        max_d = max(valid_d) if valid_d else None

        for d in d_vals:
            if d is None:
                vals.append('--')
            elif max_d is not None and abs(d - max_d) < 0.001:
                vals.append(f'\\textbf{{{d:.2f}}}')
            else:
                vals.append(f'{d:.2f}')

        row = f'{label} & ' + ' & '.join(vals) + ' \\\\'
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    output_path.write_text('\n'.join(lines))
    logger.info(f"Saved {output_path}")


def generate_crisis_taxonomy_tex(taxonomy_results, output_path):
    """Generate LaTeX table: crisis taxonomy with winning channels."""
    geometric_names = set(GEOMETRIC_DETECTORS.keys())

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r"\caption{Crisis taxonomy and geometric channel specialization. "
        r"Each crisis category has a distinct mechanism.  ``Best Overall'' "
        r"includes baselines; ``Best Geometric'' is restricted to the "
        r"seven observatory channels.}",
        r'\label{tab:crisis_taxonomy}',
        r'\small',
        r'\begin{tabular}{lllll}',
        r'\toprule',
        r'Category & Mechanism & Crises & Best Overall & Best Geometric \\',
        r'\midrule',
    ]

    for cat_name, cat_info in taxonomy_results.items():
        mechanism = cat_info['mechanism']
        winner = cat_info.get('winner', '--')
        winner_d = cat_info.get('winner_d')

        # Best geometric channel for this category
        geo_methods = {m: d for m, d in cat_info['methods'].items() if m in geometric_names}
        if geo_methods:
            best_geo = max(geo_methods, key=geo_methods.get)
            best_geo_d = geo_methods[best_geo]
        else:
            best_geo, best_geo_d = '--', None

        # Short crisis list
        crisis_labels = []
        for ck in cat_info['crises']:
            if ck in ALL_CRISES:
                label = ALL_CRISES[ck]['label']
                parts = label.split()
                crisis_labels.append(parts[-1] if len(parts) > 1 else label)

        crises_str = ', '.join(crisis_labels)
        if len(crises_str) > 25:
            crises_str = crises_str[:22] + '...'

        winner_str = f'{winner} ({winner_d:.2f})' if winner_d is not None else winner
        geo_str = f'{best_geo} ({best_geo_d:.2f})' if best_geo_d is not None else best_geo

        row = f'{cat_name} & {mechanism} & {crises_str} & {winner_str} & {geo_str} \\\\'
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    output_path.write_text('\n'.join(lines))
    logger.info(f"Saved {output_path}")


def generate_aggregate_comparison_tex(agg_stats, output_path):
    """Generate LaTeX table: aggregate d comparison across all methods."""
    # Sort by mean_d descending
    sorted_methods = sorted(
        agg_stats.items(),
        key=lambda x: x[1]['mean_d'] if x[1]['mean_d'] is not None else -1,
        reverse=True,
    )

    geometric_names = set(GEOMETRIC_DETECTORS.keys())

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r"\caption{Aggregate effect sizes across 14 crises. Mean and median "
        r"Cohen's $d$, with number of crises where $p < 0.05$.}",
        r'\label{tab:aggregate_comparison}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r"Method & Mean $d$ & Median $d$ & Max $d$ & $n_{\text{sig}}$ \\",
        r'\midrule',
    ]

    for method, st in sorted_methods:
        if st['mean_d'] is None:
            continue
        # Mark geometric methods
        prefix = r'\textit{' if method not in geometric_names else ''
        suffix = '}' if method not in geometric_names else ''

        row = (
            f'{prefix}{method}{suffix} & '
            f'{st["mean_d"]:.3f} & {st["median_d"]:.3f} & '
            f'{st["max_d"]:.3f} & {st["n_significant"]}/{st["n_crises"]} \\\\'
        )
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    output_path.write_text('\n'.join(lines))
    logger.info(f"Saved {output_path}")


def generate_taxonomy_heatmap(taxonomy_results, output_path):
    """Generate crisis specialization heatmap: channels x categories."""
    categories = list(taxonomy_results.keys())
    # Collect all methods that appear
    all_methods = set()
    for cat_info in taxonomy_results.values():
        all_methods.update(cat_info['methods'].keys())
    methods = sorted(all_methods)

    # Build matrix
    matrix = np.full((len(methods), len(categories)), np.nan)
    for j, cat in enumerate(categories):
        for i, method in enumerate(methods):
            d = taxonomy_results[cat]['methods'].get(method)
            if d is not None:
                matrix[i, j] = d

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(methods, fontsize=9)

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    # Mark winners per category
    for j, cat in enumerate(categories):
        winner = taxonomy_results[cat].get('winner')
        if winner and winner in methods:
            i = methods.index(winner)
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=2,
            ))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Cohen's d", fontsize=10)
    ax.set_title('Crisis Category Specialization by Detection Channel', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified comparison for Paper 1')
    parser.add_argument('--quick', action='store_true', help='4-crisis subset')
    parser.add_argument('--no-rf', action='store_true', help='Skip Random Forest')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / 'experiments' / 'outputs' / 'unified_comparison'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = ROOT / 'paper' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Crisis selection
    if args.quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = PAPER_CRISES

    crises = {k: ALL_CRISES[k] for k in crisis_keys if k in ALL_CRISES}
    logger.info(f"Running unified comparison with {len(crises)} crises")

    # ---- Data ----
    logger.info("[1/7] Fetching data...")
    raw = fetch_data(['SPY', 'DIA'], '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"  Feature matrix: {X.shape}, dates: {dates[0]} to {dates[-1]}")

    # Build masks
    crisis_masks, any_crisis = build_crisis_masks(dates, crises)

    # ---- Run geometric detectors ----
    logger.info("[2/7] Running geometric observatory detectors...")
    geometric_scores = run_unsupervised_detectors(X, GEOMETRIC_DETECTORS)

    # ---- Run baselines ----
    logger.info("[3/7] Running baseline detectors...")
    baseline_scores = run_unsupervised_detectors(X, BASELINE_DETECTORS)

    # ---- Run RF ----
    rf_scores = None
    if not args.no_rf:
        logger.info("[4/7] Running Random Forest (leave-one-out)...")
        rf_scores = run_rf_leave_one_out(X, dates, crises, args.n_bootstrap)
    else:
        logger.info("[4/7] Skipping Random Forest")

    # ---- Compute per-crisis d matrix ----
    logger.info("[5/7] Computing per-crisis Cohen's d matrix...")
    all_scores = {**geometric_scores, **baseline_scores}
    d_matrix = compute_per_crisis_d_matrix(
        all_scores, crisis_masks, any_crisis,
        n_bootstrap=args.n_bootstrap, rf_scores=rf_scores,
    )

    # ---- Aggregate statistics ----
    agg_stats = compute_aggregate_stats(d_matrix, crisis_keys)
    winners = compute_per_crisis_winners(d_matrix, crisis_keys)
    taxonomy = compute_taxonomy_analysis(d_matrix, crisis_keys)

    # ---- Lead time analysis ----
    logger.info("[6/7] Computing lead times...")
    lead_times = compute_lead_times(all_scores, dates, crisis_masks, crises)
    lead_summary = compute_lead_time_summary(lead_times)

    # ---- Friedman test ----
    methods_list = list(d_matrix.keys())
    d_array = np.full((len(crisis_keys), len(methods_list)), np.nan)
    for j, method in enumerate(methods_list):
        for i, ck in enumerate(crisis_keys):
            entry = d_matrix[method].get(ck, {})
            d = entry.get('d')
            if d is not None:
                d_array[i, j] = d

    chi_sq, friedman_p, mean_ranks = friedman_test(d_array)

    # ---- Generate outputs ----
    logger.info("[7/7] Generating outputs...")

    # LaTeX tables
    generate_per_crisis_winners_tex(d_matrix, crisis_keys, crises, tables_dir / 'per_crisis_winners.tex')
    generate_crisis_taxonomy_tex(taxonomy, tables_dir / 'crisis_taxonomy.tex')
    generate_aggregate_comparison_tex(agg_stats, tables_dir / 'aggregate_comparison.tex')

    # Figures
    generate_taxonomy_heatmap(taxonomy, output_dir / 'crisis_specialization_heatmap.png')

    # ---- Save full results JSON ----
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_crises': len(crises),
        'crisis_keys': crisis_keys,
        'n_bootstrap': args.n_bootstrap,
        'detector_configs': {
            'geometric': {k: str(v['params']) for k, v in GEOMETRIC_DETECTORS.items()},
            'baseline': {k: str(v['params']) for k, v in BASELINE_DETECTORS.items()},
        },
        'd_matrix': d_matrix,
        'aggregate_stats': agg_stats,
        'per_crisis_winners': winners,
        'taxonomy_analysis': taxonomy,
        'lead_times': lead_times,
        'lead_time_summary': lead_summary,
        'friedman_test': {
            'chi_sq': round(float(chi_sq), 3) if not np.isnan(chi_sq) else None,
            'p_value': float(friedman_p) if not np.isnan(friedman_p) else None,
            'mean_ranks': {m: round(float(r), 2) for m, r in zip(methods_list, mean_ranks)},
        },
    }

    json_path = output_dir / f'unified_comparison_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {json_path}")

    # Also save a 'latest' symlink-style copy
    latest_path = output_dir / 'latest_results.json'
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Print summary ----
    logger.info("\n" + "=" * 70)
    logger.info("UNIFIED COMPARISON RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nFriedman test: chi²={chi_sq:.1f}, p={friedman_p:.2e}")

    logger.info("\n--- Aggregate Effect Sizes (sorted by mean d) ---")
    sorted_agg = sorted(
        agg_stats.items(),
        key=lambda x: x[1]['mean_d'] if x[1]['mean_d'] is not None else -1,
        reverse=True,
    )
    for method, st in sorted_agg:
        if st['mean_d'] is None:
            continue
        logger.info(
            f"  {method:25s}  mean_d={st['mean_d']:.3f}  "
            f"median_d={st['median_d']:.3f}  "
            f"sig={st['n_significant']}/{st['n_crises']}"
        )

    logger.info("\n--- Per-Crisis Winners ---")
    for ck in crisis_keys:
        w = winners[ck]
        label = crises[ck]['label']
        logger.info(f"  {label:25s}  {w['winner']:25s}  d={w['d']:.3f}")

    logger.info("\n--- Crisis Taxonomy Specialization ---")
    for cat, info in taxonomy.items():
        logger.info(f"  {cat:25s}  winner={info['winner']}  d={info['winner_d']:.3f}")

    logger.info("\n--- Lead Time Summary (median days before crisis) ---")
    sorted_lead = sorted(
        lead_summary.items(),
        key=lambda x: x[1]['median_days'] if x[1]['median_days'] is not None else -1,
        reverse=True,
    )
    for method, lt in sorted_lead:
        if lt['median_days'] is not None:
            logger.info(
                f"  {method:25s}  median={lt['median_days']:.0f}d  "
                f"detected={lt['n_detected']}/{lt['n_crises']}"
            )

    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"LaTeX tables saved to: {tables_dir}")


if __name__ == '__main__':
    main()
