"""
Feature aggregation sweep for Berry and QFI detectors.

Tests Berry aggregation modes (f01, frobenius, max) and QFI modes
(logdet, trace, max_eig, condition, entropy) using the best normalization
from the normalization ablation study.

Usage:
    python experiments/feature_aggregation_sweep.py --norm none
    python experiments/feature_aggregation_sweep.py --norm soft --quick

Outputs:
    experiments/outputs/regime_detection/aggregation_sweep_YYYYMMDD_HHMMSS.json
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
)
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci, friedman_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

BERRY_AGGREGATION_MODES = ['f01', 'frobenius', 'max']
QFI_MODES = ['logdet', 'trace', 'max_eig', 'condition', 'entropy']


def run_sweep(norm_mode='none', quick=False, n_bootstrap=10000):
    logger.info("=" * 70)
    logger.info(f"Feature Aggregation Sweep (normalization={norm_mode})")
    logger.info("=" * 70)

    # ---- Data ----
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"Feature matrix: {X_enriched.shape}")

    if quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = [k for k in ALL_CRISES if int(k[:4]) >= 2005]
    crises = {k: ALL_CRISES[k] for k in crisis_keys if k in ALL_CRISES}
    logger.info(f"Evaluating {len(crises)} crises")

    use_adaptive_eps = (norm_mode != 'sphere')
    results = {'Berry': {}, 'QFI': {}}

    # ---- Berry aggregation sweep ----
    logger.info("\n--- Berry Phase Rate aggregation sweep ---")
    for agg_mode in BERRY_AGGREGATION_MODES:
        results['Berry'][agg_mode] = {}
        logger.info(f"\n  berry_aggregation={agg_mode}")

        for ck, ci in crises.items():
            crisis_start = pd.Timestamp(ci['start'])
            crisis_end = pd.Timestamp(ci['end'])
            cutoff_date = crisis_start - pd.Timedelta(days=10)
            fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

            if fit_end_idx < 100:
                continue

            det = BerryPhaseRateDetector(
                hilbert_dim=6, n_pca_components=8, rolling_window=15,
                operator_method='random', seed=42,
                causal_fit_length=fit_end_idx,
                normalization=norm_mode,
                berry_aggregation=agg_mode,
                adaptive_epsilon=use_adaptive_eps,
            )
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)

            cs = crisis_start - pd.Timedelta(days=10)
            ce = crisis_end + pd.Timedelta(days=10)
            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
            )

            results['Berry'][agg_mode][ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }
            d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
            logger.info(f"    {ck:20s}  d = {d_str}")

    # ---- QFI mode sweep ----
    logger.info("\n--- QFI Determinant mode sweep ---")
    for qfi_mode in QFI_MODES:
        results['QFI'][qfi_mode] = {}
        logger.info(f"\n  qfi_mode={qfi_mode}")

        for ck, ci in crises.items():
            crisis_start = pd.Timestamp(ci['start'])
            crisis_end = pd.Timestamp(ci['end'])
            cutoff_date = crisis_start - pd.Timedelta(days=10)
            fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

            if fit_end_idx < 100:
                continue

            det = QFIDeterminantDetector(
                hilbert_dim=8, n_pca_components=15, rolling_window=20,
                operator_method='pca_inspired', seed=42,
                causal_fit_length=fit_end_idx,
                normalization=norm_mode,
                qfi_mode=qfi_mode,
                adaptive_epsilon=use_adaptive_eps,
            )
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)

            cs = crisis_start - pd.Timedelta(days=10)
            ce = crisis_end + pd.Timedelta(days=10)
            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
            )

            results['QFI'][qfi_mode][ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }
            d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
            logger.info(f"    {ck:20s}  d = {d_str}")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)

    summary = {}
    for det_name, modes in [('Berry', BERRY_AGGREGATION_MODES), ('QFI', QFI_MODES)]:
        summary[det_name] = {}
        for mode in modes:
            ds = [v['d'] for v in results[det_name][mode].values() if v['d'] is not None]
            mean_d = np.mean(ds) if ds else np.nan
            std_d = np.std(ds) if ds else np.nan
            summary[det_name][mode] = {
                'mean_d': float(mean_d),
                'std_d': float(std_d),
                'n_crises': len(ds),
            }
            logger.info(f"  {det_name:6s} {mode:12s}  mean d = {mean_d:.3f} +/- {std_d:.3f}")

    # ---- Save ----
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'aggregation_sweep_{ts}.json'

    output = {
        'timestamp': ts,
        'normalization': norm_mode,
        'n_crises': len(crises),
        'results': results,
        'summary': summary,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default='none', choices=['sphere', 'none', 'soft', 'clip'])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    args = parser.parse_args()
    run_sweep(norm_mode=args.norm, quick=args.quick, n_bootstrap=args.n_bootstrap)
