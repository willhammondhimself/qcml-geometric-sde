"""
Normalization ablation study for QCML geometric detectors.

Tests 4 normalization modes (sphere, none, soft, clip) × 3 detectors
across 12 crises with causal preprocessing.

Usage:
    python experiments/normalization_ablation.py
    python experiments/normalization_ablation.py --quick  # 4 crises only

Outputs:
    experiments/outputs/regime_detection/normalization_ablation_YYYYMMDD_HHMMSS.json
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

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci, friedman_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

NORMALIZATION_MODES = ['sphere', 'none', 'soft', 'clip']

# HPO-optimal params (from regime_comparison.py), normalization overridden per config
DETECTOR_CONFIGS = {
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
        ),
    },
    'QFI Determinant': {
        'class': QFIDeterminantDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
        ),
    },
    'Multi-Lag Fidelity': {
        'class': MultiLagFidelityDetector,
        'params': dict(
            hilbert_dim=4, n_pca_components=8, rolling_window=20,
            operator_method='pca_inspired', seed=42,
        ),
    },
}


def run_ablation(quick=False, n_bootstrap=10000):
    logger.info("=" * 70)
    logger.info("Normalization Ablation Study")
    logger.info("=" * 70)

    # ---- Data ----
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"Feature matrix: {X_enriched.shape}")

    # ---- Crisis selection ----
    if quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = [k for k in ALL_CRISES if int(k[:4]) >= 2005]
    crises = {k: ALL_CRISES[k] for k in crisis_keys if k in ALL_CRISES}
    logger.info(f"Evaluating {len(crises)} crises")

    # ---- Ablation ----
    # results[detector][norm_mode][crisis] = {'d': ..., 'ci_lo': ..., 'ci_hi': ...}
    results = {}

    for det_name, config in DETECTOR_CONFIGS.items():
        results[det_name] = {}
        for norm_mode in NORMALIZATION_MODES:
            results[det_name][norm_mode] = {}
            logger.info(f"\n  {det_name} / normalization={norm_mode}")

            for ck, ci in crises.items():
                crisis_start = pd.Timestamp(ci['start'])
                crisis_end = pd.Timestamp(ci['end'])
                cutoff_date = crisis_start - pd.Timedelta(days=10)
                fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

                if fit_end_idx < 100:
                    logger.warning(f"    Skipping {ck}: insufficient data")
                    continue

                params = {
                    **config['params'],
                    'causal_fit_length': fit_end_idx,
                    'normalization': norm_mode,
                    'adaptive_epsilon': (norm_mode != 'sphere'),
                }
                det = config['class'](**params)
                det.fit(X_enriched)
                scores = det.compute_regime_scores(X_enriched)

                cs = crisis_start - pd.Timedelta(days=10)
                ce = crisis_end + pd.Timedelta(days=10)
                crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
                normal_mask = ~crisis_mask

                d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                    scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
                )

                results[det_name][norm_mode][ck] = {
                    'd': float(d) if not np.isnan(d) else None,
                    'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                    'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
                }
                d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
                logger.info(f"    {ck:20s}  d = {d_str}")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("Summary: Mean Cohen's d by detector × normalization")
    logger.info("=" * 70)

    summary = {}
    for det_name in DETECTOR_CONFIGS:
        summary[det_name] = {}
        for norm_mode in NORMALIZATION_MODES:
            ds = [
                v['d'] for v in results[det_name][norm_mode].values()
                if v['d'] is not None
            ]
            mean_d = np.mean(ds) if ds else np.nan
            std_d = np.std(ds) if ds else np.nan
            summary[det_name][norm_mode] = {
                'mean_d': float(mean_d),
                'std_d': float(std_d),
                'n_crises': len(ds),
            }
            logger.info(f"  {det_name:25s}  {norm_mode:8s}  mean d = {mean_d:.3f} +/- {std_d:.3f}")

    # ---- Friedman test per detector ----
    logger.info("\nFriedman test per detector:")
    for det_name in DETECTOR_CONFIGS:
        crisis_list = sorted(set().union(
            *[results[det_name][nm].keys() for nm in NORMALIZATION_MODES]
        ))
        matrix = []
        for nm in NORMALIZATION_MODES:
            row = []
            for ck in crisis_list:
                v = results[det_name][nm].get(ck, {}).get('d')
                row.append(v if v is not None else np.nan)
            matrix.append(row)
        matrix = np.array(matrix)

        valid_cols = ~np.any(np.isnan(matrix), axis=0)
        if np.sum(valid_cols) >= 3:
            matrix_valid = matrix[:, valid_cols]
            chi_sq, p_val, _ = friedman_test(matrix_valid)
            logger.info(f"  {det_name}: chi-sq={chi_sq:.2f}, p={p_val:.4f}")
        else:
            logger.info(f"  {det_name}: insufficient data for Friedman test")

    # ---- Save ----
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'normalization_ablation_{ts}.json'

    output = {
        'timestamp': ts,
        'n_crises': len(crises),
        'normalization_modes': NORMALIZATION_MODES,
        'results': results,
        'summary': summary,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    args = parser.parse_args()
    run_ablation(quick=args.quick, n_bootstrap=args.n_bootstrap)
