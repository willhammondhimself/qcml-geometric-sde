"""
Observatory analysis: orthogonality matrix, oracle fusion, and complementarity.

Computes pairwise Spearman correlations between all geometric observables and
baselines, generates a heatmap, and evaluates an oracle (best-of-N) fusion
strategy that picks the highest-scoring geometric channel per crisis.

Usage:
    python experiments/observatory_analysis.py
    python experiments/observatory_analysis.py --quick   # 4 crises
    python experiments/observatory_analysis.py --output-dir paper/figures

Outputs:
    - Orthogonality heatmap (PNG + PDF)
    - Oracle fusion JSON
    - Complementarity statistics
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
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.baselines import RollingVolatilityDetector, RandomForestRegimeDetector
from experiments.evaluation import compute_cohens_d_with_ci, welch_t_test

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

np.random.seed(42)

# The 7 geometric observatory channels
OBSERVATORY_DETECTORS = {
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
    'Sect. Curv. Sign': {
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
}


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


def run_detectors(X, detector_configs, causal_fit_length=None):
    """Run all detectors and return {name: scores} dict."""
    results = {}
    for name, cfg in detector_configs.items():
        logger.info(f"  Running {name}...")
        det = cfg['class'](**cfg['params'])
        if causal_fit_length is not None and hasattr(det, 'causal_fit_length'):
            det.causal_fit_length = causal_fit_length
        det.fit(X)
        scores = det.compute_regime_scores(X)
        results[name] = scores
        logger.info(f"    {name}: {np.sum(~np.isnan(scores))} valid scores")
    return results


def compute_orthogonality_matrix(score_dict):
    """Compute pairwise Spearman correlation matrix.

    Args:
        score_dict: {name: 1-D score array}.

    Returns:
        (names, corr_matrix): List of names and (N, N) correlation matrix.
    """
    names = list(score_dict.keys())
    N = len(names)
    corr = np.eye(N)

    for i in range(N):
        for j in range(i + 1, N):
            s1 = score_dict[names[i]]
            s2 = score_dict[names[j]]
            valid = ~np.isnan(s1) & ~np.isnan(s2)
            if np.sum(valid) < 30:
                corr[i, j] = corr[j, i] = np.nan
                continue
            rho, _ = stats.spearmanr(s1[valid], s2[valid])
            corr[i, j] = corr[j, i] = rho

    return names, corr


def oracle_fusion(score_dict, dates, crisis_masks, any_crisis, geometric_names):
    """Oracle fusion: for each crisis, pick the best geometric channel.

    Returns per-crisis best detector name and d, plus overall oracle d.
    """
    results = {}
    all_oracle_crisis = []
    all_oracle_normal = []

    valid_normal_mask = ~any_crisis

    for ck, mask in crisis_masks.items():
        best_d = -np.inf
        best_name = None
        for name in geometric_names:
            scores = score_dict[name]
            valid = ~np.isnan(scores)
            c = scores[valid & mask]
            n = scores[valid & valid_normal_mask]
            if len(c) < 3 or len(n) < 10:
                continue
            d, _, _ = compute_cohens_d_with_ci(c, n, n_bootstrap=500)
            if not np.isnan(d) and d > best_d:
                best_d = d
                best_name = name

        results[ck] = {
            'best_detector': best_name,
            'cohens_d': round(float(best_d), 3) if best_name else None,
        }

        if best_name is not None:
            scores = score_dict[best_name]
            valid = ~np.isnan(scores)
            all_oracle_crisis.extend(scores[valid & mask].tolist())
            all_oracle_normal.extend(scores[valid & valid_normal_mask].tolist())

    # Overall oracle d
    if len(all_oracle_crisis) > 5 and len(all_oracle_normal) > 10:
        oracle_d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            np.array(all_oracle_crisis), np.array(all_oracle_normal)
        )
    else:
        oracle_d, ci_lo, ci_hi = np.nan, np.nan, np.nan

    return {
        'per_crisis': results,
        'oracle_d': round(float(oracle_d), 3),
        'oracle_ci': [round(float(ci_lo), 3), round(float(ci_hi), 3)],
    }


def plot_orthogonality_heatmap(names, corr, output_path):
    """Plot and save the orthogonality heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            val = corr[i, j]
            if np.isnan(val):
                continue
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman correlation', fontsize=10)
    ax.set_title('Geometric Observatory: Score Orthogonality Matrix', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved heatmap: {output_path} and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Observatory orthogonality analysis')
    parser.add_argument('--quick', action='store_true', help='Use 4-crisis quick subset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: experiments/outputs/observatory)')
    args = parser.parse_args()

    # Output setup
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / 'experiments' / 'outputs' / 'observatory'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crisis subset
    if args.quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = [
            '2000_dotcom', '2001_911', '2007_quant', '2008_gfc',
            '2010_flash', '2011_euro', '2015_china',
            '2018_volmageddon', '2018_q4', '2019_repo',
            '2020_covid', '2022_rates', '2023_svb', '2024_carry',
        ]
    crises = {k: ALL_CRISES[k] for k in crisis_keys if k in ALL_CRISES}

    # Fetch data
    logger.info("Fetching data...")
    raw = fetch_data(['SPY', 'DIA'], '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"Feature matrix: {X.shape}, dates: {dates[0]} to {dates[-1]}")

    # Build masks
    crisis_masks, any_crisis = build_crisis_masks(dates, crises)

    # Run all detectors
    logger.info("Running observatory detectors...")
    all_configs = {**OBSERVATORY_DETECTORS, **BASELINE_DETECTORS}
    score_dict = run_detectors(X, all_configs)

    # Compute orthogonality matrix
    logger.info("Computing orthogonality matrix...")
    names, corr = compute_orthogonality_matrix(score_dict)

    # Compute per-detector Cohen's d
    logger.info("Computing per-detector effect sizes...")
    detector_stats = {}
    for name, scores in score_dict.items():
        valid = ~np.isnan(scores)
        c = scores[valid & any_crisis]
        n = scores[valid & ~any_crisis]
        if len(c) >= 5 and len(n) >= 10:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(c, n)
            _, p_val = welch_t_test(c, n)
        else:
            d, ci_lo, ci_hi, p_val = np.nan, np.nan, np.nan, np.nan
        detector_stats[name] = {
            'cohens_d': round(float(d), 3),
            'ci': [round(float(ci_lo), 3), round(float(ci_hi), 3)],
            'p_value': float(p_val) if not np.isnan(p_val) else None,
        }

    # Oracle fusion
    logger.info("Computing oracle fusion...")
    geometric_names = list(OBSERVATORY_DETECTORS.keys())
    oracle = oracle_fusion(score_dict, dates, crisis_masks, any_crisis, geometric_names)

    # Complementarity score: mean |rho| among geometric channels
    geo_indices = [names.index(n) for n in geometric_names if n in names]
    geo_corr = corr[np.ix_(geo_indices, geo_indices)]
    mask = ~np.eye(len(geo_indices), dtype=bool)
    mean_abs_rho = float(np.nanmean(np.abs(geo_corr[mask])))

    # Geometric vs baseline correlations
    baseline_names = list(BASELINE_DETECTORS.keys())
    geo_vs_baseline_rhos = []
    for gn in geometric_names:
        if gn not in names:
            continue
        gi = names.index(gn)
        for bn in baseline_names:
            if bn not in names:
                continue
            bi = names.index(bn)
            if not np.isnan(corr[gi, bi]):
                geo_vs_baseline_rhos.append(abs(corr[gi, bi]))
    mean_geo_baseline_rho = float(np.mean(geo_vs_baseline_rhos)) if geo_vs_baseline_rhos else np.nan

    # Plot heatmap
    plot_orthogonality_heatmap(
        names, corr, output_dir / 'orthogonality_heatmap.png'
    )

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_crises': len(crises),
        'crisis_keys': list(crises.keys()),
        'detector_stats': detector_stats,
        'orthogonality': {
            'names': names,
            'matrix': corr.tolist(),
            'mean_abs_rho_geometric': round(mean_abs_rho, 3),
            'mean_abs_rho_geo_vs_baseline': round(mean_geo_baseline_rho, 3),
        },
        'oracle_fusion': oracle,
        'complementarity': {
            'n_geometric_channels': len(geometric_names),
            'mean_intra_geometric_abs_rho': round(mean_abs_rho, 3),
            'mean_geo_vs_baseline_abs_rho': round(mean_geo_baseline_rho, 3),
        },
    }

    json_path = output_dir / f'observatory_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {json_path}")

    # Summary
    logger.info("\n=== OBSERVATORY SUMMARY ===")
    logger.info(f"Geometric channels: {len(geometric_names)}")
    logger.info(f"Mean intra-geometric |rho|: {mean_abs_rho:.3f}")
    logger.info(f"Mean geo vs baseline |rho|: {mean_geo_baseline_rho:.3f}")
    logger.info(f"Oracle fusion d: {oracle['oracle_d']} "
                f"CI: [{oracle['oracle_ci'][0]}, {oracle['oracle_ci'][1]}]")
    logger.info("\nPer-detector effect sizes:")
    for name, st in sorted(detector_stats.items(), key=lambda x: -x[1]['cohens_d']):
        logger.info(f"  {name:25s}  d={st['cohens_d']:.3f}  "
                    f"CI=[{st['ci'][0]:.3f}, {st['ci'][1]:.3f}]")


if __name__ == '__main__':
    main()
