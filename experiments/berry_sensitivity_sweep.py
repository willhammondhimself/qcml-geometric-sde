"""
Berry Phase Rate hyperparameter sensitivity analysis.

One-at-a-time (OAT) sweep over Berry Phase Rate hyperparameters plus a 2D grid
over (hilbert_dim x rolling_window) to assess interaction effects.  For each
configuration the script computes Cohen's d across all 14 paper crises and
saves the full results as JSON.

Usage:
    python experiments/berry_sensitivity_sweep.py               # full (~35 min)
    python experiments/berry_sensitivity_sweep.py --quick        # 4 crises (~5 min)
    python experiments/berry_sensitivity_sweep.py --n-bootstrap 500  # faster CIs

Outputs:
    experiments/outputs/regime_detection/berry_sensitivity_YYYYMMDD_HHMMSS.json
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

from qcml_geometry import BerryPhaseRateDetector
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES

from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

# ---- Defaults from config.yaml:206-213 ----
DEFAULTS = {
    'hilbert_dim': 6,
    'n_pca_components': 8,
    'rolling_window': 15,
    'operator_method': 'random',
    'normalization': 'sphere',
    'berry_aggregation': 'f01',
    'seed': 42,
}

# ---- OAT ranges (vary one, hold others at default) ----
OAT_RANGES = {
    'hilbert_dim': [4, 6, 8, 10, 12],
    'n_pca_components': [4, 6, 8, 10, 15],
    'rolling_window': [5, 10, 15, 20, 30, 50],
    'normalization': ['sphere', 'none', 'soft', 'clip'],
    'berry_aggregation': ['f01', 'frobenius', 'max'],
}

# ---- 2D grid: hilbert_dim x rolling_window ----
GRID_HILBERT_DIMS = [4, 6, 8, 10, 12]
GRID_ROLLING_WINDOWS = [5, 10, 15, 20, 30, 50]


def _fit_and_score(params, X_enriched, dates_enriched, crises, n_bootstrap):
    """Fit a BerryPhaseRateDetector and compute Cohen's d for each crisis."""
    results = {}
    use_adaptive_eps = (params['normalization'] != 'sphere')

    for ck, ci in crises.items():
        crisis_start = pd.Timestamp(ci['start'])
        crisis_end = pd.Timestamp(ci['end'])
        cutoff_date = crisis_start - pd.Timedelta(days=10)
        fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

        if fit_end_idx < 100:
            results[ck] = {'d': None, 'ci_lo': None, 'ci_hi': None}
            continue

        det = BerryPhaseRateDetector(
            hilbert_dim=params['hilbert_dim'],
            n_pca_components=params['n_pca_components'],
            rolling_window=params['rolling_window'],
            operator_method=params['operator_method'],
            seed=params['seed'],
            causal_fit_length=fit_end_idx,
            normalization=params['normalization'],
            berry_aggregation=params['berry_aggregation'],
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

        results[ck] = {
            'd': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        }

    return results


def _summarize(crisis_results):
    """Compute median and IQR of Cohen's d across crises."""
    ds = [v['d'] for v in crisis_results.values() if v['d'] is not None]
    if not ds:
        return {'median_d': None, 'q25': None, 'q75': None, 'n_crises': 0}
    arr = np.array(ds)
    return {
        'median_d': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'n_crises': len(ds),
    }


def run_sweep(quick=False, n_bootstrap=10000):
    logger.info("=" * 70)
    logger.info("Berry Phase Rate Hyperparameter Sensitivity Sweep")
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
        crisis_keys = list(ALL_CRISES.keys())
    crises = {k: ALL_CRISES[k] for k in crisis_keys if k in ALL_CRISES}
    logger.info(f"Evaluating {len(crises)} crises (bootstrap n={n_bootstrap})")

    output = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'defaults': DEFAULTS,
        'n_crises': len(crises),
        'crisis_keys': list(crises.keys()),
        'n_bootstrap': n_bootstrap,
        'oat': {},
        'grid_2d': {},
    }

    # ================================================================
    # OAT sweep
    # ================================================================
    total_oat = sum(len(v) for v in OAT_RANGES.values())
    done = 0
    for param_name, values in OAT_RANGES.items():
        output['oat'][param_name] = {}
        logger.info(f"\n--- OAT: {param_name} ---")

        for val in values:
            done += 1
            params = dict(DEFAULTS)
            params[param_name] = val
            label = str(val)
            logger.info(f"  [{done}/{total_oat}] {param_name}={val}")

            crisis_results = _fit_and_score(
                params, X_enriched, dates_enriched, crises, n_bootstrap,
            )
            summary = _summarize(crisis_results)
            output['oat'][param_name][label] = {
                'params': params,
                'crises': crisis_results,
                'summary': summary,
            }
            med = summary['median_d']
            med_str = f"{med:.3f}" if med is not None else "N/A"
            logger.info(f"    median d = {med_str}")

    # ================================================================
    # 2D grid: hilbert_dim x rolling_window
    # ================================================================
    logger.info("\n--- 2D Grid: hilbert_dim x rolling_window ---")
    total_grid = len(GRID_HILBERT_DIMS) * len(GRID_ROLLING_WINDOWS)
    done = 0
    for hd in GRID_HILBERT_DIMS:
        for rw in GRID_ROLLING_WINDOWS:
            done += 1
            params = dict(DEFAULTS)
            params['hilbert_dim'] = hd
            params['rolling_window'] = rw
            key = f"{hd}_{rw}"
            logger.info(f"  [{done}/{total_grid}] hd={hd}, rw={rw}")

            crisis_results = _fit_and_score(
                params, X_enriched, dates_enriched, crises, n_bootstrap,
            )
            summary = _summarize(crisis_results)
            output['grid_2d'][key] = {
                'hilbert_dim': hd,
                'rolling_window': rw,
                'crises': crisis_results,
                'summary': summary,
            }
            med = summary['median_d']
            med_str = f"{med:.3f}" if med is not None else "N/A"
            logger.info(f"    median d = {med_str}")

    # ================================================================
    # Aggregate statistics
    # ================================================================
    oat_medians = []
    for param_name in OAT_RANGES:
        for label, entry in output['oat'][param_name].items():
            med = entry['summary']['median_d']
            if med is not None:
                oat_medians.append(med)

    n_above_05 = sum(1 for m in oat_medians if m > 0.5)
    pct_above_05 = 100.0 * n_above_05 / len(oat_medians) if oat_medians else 0.0

    grid_medians = [e['summary']['median_d'] for e in output['grid_2d'].values()
                    if e['summary']['median_d'] is not None]

    output['aggregate'] = {
        'oat_n_configs': len(oat_medians),
        'oat_n_above_0.5': n_above_05,
        'oat_pct_above_0.5': round(pct_above_05, 1),
        'oat_median_of_medians': float(np.median(oat_medians)) if oat_medians else None,
        'oat_min_median': float(np.min(oat_medians)) if oat_medians else None,
        'oat_max_median': float(np.max(oat_medians)) if oat_medians else None,
        'grid_n_cells': len(grid_medians),
        'grid_median_of_medians': float(np.median(grid_medians)) if grid_medians else None,
        'grid_min_median': float(np.min(grid_medians)) if grid_medians else None,
        'grid_max_median': float(np.max(grid_medians)) if grid_medians else None,
    }

    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"  OAT configs: {len(oat_medians)}")
    logger.info(f"  OAT configs with median d > 0.5: {n_above_05}/{len(oat_medians)} ({pct_above_05:.0f}%)")
    logger.info(f"  OAT median-of-medians: {np.median(oat_medians):.3f}" if oat_medians else "  N/A")
    logger.info(f"  2D grid cells: {len(grid_medians)}")
    if grid_medians:
        logger.info(f"  Grid median-of-medians: {np.median(grid_medians):.3f}")

    # ---- Save ----
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = output['timestamp']
    out_path = out_dir / f'berry_sensitivity_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Berry Phase Rate hyperparameter sensitivity sweep',
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 4 crises instead of all 16')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap resamples for CIs')
    args = parser.parse_args()
    run_sweep(quick=args.quick, n_bootstrap=args.n_bootstrap)
