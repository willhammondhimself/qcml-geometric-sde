"""
Window-size sensitivity analysis (Review Item 6).

Re-compute Cohen's d for top methods at crisis window sizes
±5, ±10 (default), ±20, ±60 trading days.
Report rank correlation (Kendall tau) across window sizes.

Usage:
    python experiments/window_sensitivity.py
    python experiments/window_sensitivity.py --quick
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
    fetch_polygon_data, create_feature_matrix, ALL_CRISES,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
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

WINDOW_SIZES = [5, 10, 20, 60]


def run_window_sensitivity(quick=False):
    """Evaluate sensitivity of results to crisis window size.

    Args:
        quick: Only run on 4 crises.
    """
    logger.info("=" * 70)
    logger.info("Window-Size Sensitivity Analysis")
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
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = list(ALL_CRISES.keys())

    # Build and fit detectors (fit once, score once per method)
    common_qcml = dict(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
        rolling_window=20, seed=42,
    )
    methods = {
        'Berry Phase Rate': BerryPhaseRateDetector(**common_qcml),
        'QFI Determinant': QFIDeterminantDetector(**common_qcml),
        'Multi-Lag Fidelity': MultiLagFidelityDetector(**common_qcml),
        'Rolling Vol Z': RollingVolatilityDetector(vol_window=20, min_expanding=60),
        'HMM': HMMRegimeDetector(n_iter=100, seed=42),
    }

    logger.info("\n[2] Fitting detectors...")
    all_scores = {}
    for name, det in methods.items():
        logger.info(f"  {name}...")
        det.fit(X_enriched)
        all_scores[name] = det.compute_regime_scores(X_enriched)

    # RF with leave-one-crisis-out (fit per crisis)
    rf_scores = {}
    logger.info("  Random Forest (LOCO)...")
    for ck in crisis_keys:
        # Build labels on full (pre-enrichment) date index
        y = np.zeros(len(X))
        for ok, oi in ALL_CRISES.items():
            if ok == ck:
                continue
            cs = pd.Timestamp(oi['start'])
            ce = pd.Timestamp(oi['end'])
            mask = (dates >= cs) & (dates <= ce)
            y[mask] = 1.0

        rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
        rf.fit_with_labels(X, y)
        scores = rf.compute_regime_scores(X)
        rf_scores[ck] = scores[19:] if len(scores) > len(dates_enriched) else scores

    # Evaluate at each window size
    logger.info("\n[3] Computing d at each window size...")
    results = {}  # {window_size: {method: {crisis: d}}}

    for ws in WINDOW_SIZES:
        logger.info(f"\n  Window ±{ws} days:")
        results[ws] = {}

        for method_name, scores in all_scores.items():
            method_ds = {}
            for ck in crisis_keys:
                ci = ALL_CRISES[ck]
                cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=ws)
                ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=ws)

                crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
                normal_mask = ~crisis_mask

                d, _, _ = compute_cohens_d_with_ci(
                    scores[crisis_mask], scores[normal_mask], n_bootstrap=1000
                )
                method_ds[ck] = float(d) if not np.isnan(d) else None

            results[ws][method_name] = method_ds

        # RF
        rf_ds = {}
        for ck in crisis_keys:
            ci = ALL_CRISES[ck]
            cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=ws)
            ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=ws)

            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            sc = rf_scores.get(ck)
            if sc is not None and len(sc) == len(dates_enriched):
                d, _, _ = compute_cohens_d_with_ci(
                    sc[crisis_mask], sc[normal_mask], n_bootstrap=1000
                )
                rf_ds[ck] = float(d) if not np.isnan(d) else None
            else:
                rf_ds[ck] = None

        results[ws]['Random Forest'] = rf_ds

    # Compute rank correlations across window sizes
    logger.info("\n[4] Computing rank correlations (Kendall tau)...")
    all_method_names = list(all_scores.keys()) + ['Random Forest']

    rank_corrs = {}
    for ws1_idx in range(len(WINDOW_SIZES)):
        for ws2_idx in range(ws1_idx + 1, len(WINDOW_SIZES)):
            ws1, ws2 = WINDOW_SIZES[ws1_idx], WINDOW_SIZES[ws2_idx]

            # Compute median d per method for each window size
            medians1 = []
            medians2 = []
            for mname in all_method_names:
                ds1 = [v for v in results[ws1].get(mname, {}).values() if v is not None]
                ds2 = [v for v in results[ws2].get(mname, {}).values() if v is not None]
                medians1.append(np.median(ds1) if ds1 else 0.0)
                medians2.append(np.median(ds2) if ds2 else 0.0)

            tau, p = spstats.kendalltau(medians1, medians2)
            rank_corrs[f'±{ws1} vs ±{ws2}'] = {'tau': float(tau), 'p': float(p)}
            logger.info(f"  ±{ws1} vs ±{ws2}: Kendall tau = {tau:.3f} (p = {p:.4f})")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("MEDIAN d BY WINDOW SIZE")
    logger.info("=" * 70)
    header = f"{'Method':25s}" + "".join(f"  ±{ws:3d}d" for ws in WINDOW_SIZES)
    logger.info(header)

    for mname in all_method_names:
        row = f"{mname:25s}"
        for ws in WINDOW_SIZES:
            ds = [v for v in results[ws].get(mname, {}).values() if v is not None]
            med = np.median(ds) if ds else float('nan')
            row += f"  {med:6.3f}"
        logger.info(row)

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'window_sizes': WINDOW_SIZES,
        'results': {str(k): v for k, v in results.items()},
        'rank_correlations': rank_corrs,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'window_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'window_sensitivity_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Window-size sensitivity analysis')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_window_sensitivity(quick=args.quick)


if __name__ == '__main__':
    main()
