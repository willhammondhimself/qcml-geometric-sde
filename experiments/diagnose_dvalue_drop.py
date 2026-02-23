"""
Diagnostic 2x2 experiment: isolate cause of d-value drop between runs.

Tests 4 configs:
  A: start=2005, Berry=pca_inspired  (old committed version)
  B: start=2005, Berry=random        (operator change only)
  C: start=1995, Berry=pca_inspired  (date change only)
  D: start=1995, Berry=random        (current working version)

QFI and MLF always use pca_inspired (unchanged between versions).
Runs Berry, QFI, MLF, RF, CUSUM on 4 crises: GFC, COVID, Rates, SVB.

Usage:
    python experiments/diagnose_dvalue_drop.py
"""

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
from experiments.data_loader import fetch_polygon_data, create_feature_matrix, ALL_CRISES
from experiments.baselines import (
    CUSUMDetector,
    RandomForestRegimeDetector,
)
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

QUICK_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']

CONFIGS = {
    'A (2005, pca_inspired)': {'start_date': '2005-01-01', 'berry_operator': 'pca_inspired'},
    'B (2005, random)':       {'start_date': '2005-01-01', 'berry_operator': 'random'},
    'C (1995, pca_inspired)': {'start_date': '1995-01-01', 'berry_operator': 'pca_inspired'},
    'D (1995, random)':       {'start_date': '1995-01-01', 'berry_operator': 'random'},
}


def fetch_and_prepare(start_date):
    """Fetch data and build enriched feature matrix."""
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, start_date, '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    return X, dates, X_enriched, dates_enriched


def run_one_config(config_name, cfg, crises, n_bootstrap=2000):
    """Run Berry, QFI, MLF, RF, CUSUM for one config on given crises.

    Returns:
        dict: {method_name: {crisis_key: d_value}}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Config {config_name}: start={cfg['start_date']}, berry_op={cfg['berry_operator']}")
    logger.info(f"{'='*60}")

    X, dates, X_enriched, dates_enriched = fetch_and_prepare(cfg['start_date'])
    logger.info(f"  Data: X={X.shape}, enriched={X_enriched.shape}, dates {dates[0]} to {dates[-1]}")

    shared = dict(hilbert_dim=8, n_pca_components=15, rolling_window=20, seed=42)

    detectors = [
        ('Berry Phase Rate', BerryPhaseRateDetector(**shared, operator_method=cfg['berry_operator'])),
        ('QFI Determinant', QFIDeterminantDetector(**shared, operator_method='pca_inspired')),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector(**shared, operator_method='pca_inspired')),
        ('CUSUM', CUSUMDetector(burn_in=60)),
    ]

    all_scores = {}
    for name, det in detectors:
        logger.info(f"  Fitting {name}...")
        det.fit(X_enriched)
        all_scores[name] = det.compute_regime_scores(X_enriched)

    # RF with leave-one-crisis-out
    logger.info("  Fitting RF (LOCO)...")
    rf_scores_per_crisis = {}
    for held_out_key in crises:
        y = np.zeros(len(X))
        for ck, ci in crises.items():
            if ck == held_out_key:
                continue
            cs = pd.Timestamp(ci['start'])
            ce = pd.Timestamp(ci['end'])
            mask = (dates >= cs) & (dates <= ce)
            y[mask] = 1.0

        rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
        rf.fit_with_labels(X, y)
        scores = rf.compute_regime_scores(X)
        rf_scores_per_crisis[held_out_key] = scores[19:] if len(scores) > len(dates_enriched) else scores

    # Compute Cohen's d
    results = {}
    window_size = 10

    for method_name, scores in all_scores.items():
        method_results = {}
        for ck, ci in crises.items():
            cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
            ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)
            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask
            d, _, _ = compute_cohens_d_with_ci(scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap)
            method_results[ck] = round(float(d), 3) if not np.isnan(d) else None
        results[method_name] = method_results

    # RF
    rf_results = {}
    for ck, ci in crises.items():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask
        rf_sc = rf_scores_per_crisis.get(ck)
        if rf_sc is not None and len(rf_sc) == len(dates_enriched):
            d, _, _ = compute_cohens_d_with_ci(rf_sc[crisis_mask], rf_sc[normal_mask], n_bootstrap=n_bootstrap)
        else:
            d = np.nan
        rf_results[ck] = round(float(d), 3) if not np.isnan(d) else None
    results['Random Forest'] = rf_results

    return results


def compute_medians(results, crisis_keys):
    """Compute median d across crises for each method."""
    medians = {}
    for method, crisis_d in results.items():
        vals = [crisis_d.get(ck) for ck in crisis_keys]
        vals = [v for v in vals if v is not None]
        medians[method] = round(float(np.median(vals)), 3) if vals else None
    return medians


def main():
    crises = {k: ALL_CRISES[k] for k in QUICK_CRISES}
    all_results = {}
    all_medians = {}

    for config_name, cfg in CONFIGS.items():
        results = run_one_config(config_name, cfg, crises)
        all_results[config_name] = results
        all_medians[config_name] = compute_medians(results, QUICK_CRISES)

    # Print 2x2 results table
    methods = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity', 'Random Forest', 'CUSUM']
    config_names = list(CONFIGS.keys())

    print("\n" + "=" * 90)
    print("2x2 DIAGNOSTIC RESULTS — Median Cohen's d across 4 crises")
    print("=" * 90)

    header = f"{'Method':25s}"
    for cn in config_names:
        header += f"  {cn:>16s}"
    print(header)
    print("-" * 90)

    for m in methods:
        row = f"{m:25s}"
        for cn in config_names:
            val = all_medians[cn].get(m)
            row += f"  {val:16.3f}" if val is not None else f"  {'N/A':>16s}"
        print(row)

    # Effect decomposition
    print("\n" + "=" * 90)
    print("EFFECT DECOMPOSITION (median d change)")
    print("=" * 90)

    for m in methods:
        a = all_medians['A (2005, pca_inspired)'].get(m, 0) or 0
        b = all_medians['B (2005, random)'].get(m, 0) or 0
        c = all_medians['C (1995, pca_inspired)'].get(m, 0) or 0
        d = all_medians['D (1995, random)'].get(m, 0) or 0

        # Operator effect: average of (B-A) and (D-C)
        op_effect = ((b - a) + (d - c)) / 2
        # Date range effect: average of (C-A) and (D-B)
        date_effect = ((c - a) + (d - b)) / 2
        total = d - a

        print(f"  {m:25s}  operator: {op_effect:+.3f}  date_range: {date_effect:+.3f}  total: {total:+.3f}")

    # Per-crisis detail for Berry (most affected method)
    print("\n" + "=" * 90)
    print("PER-CRISIS DETAIL — Berry Phase Rate")
    print("=" * 90)
    header = f"{'Crisis':20s}"
    for cn in config_names:
        header += f"  {cn:>16s}"
    print(header)
    print("-" * 90)
    for ck in QUICK_CRISES:
        row = f"{ck:20s}"
        for cn in config_names:
            val = all_results[cn].get('Berry Phase Rate', {}).get(ck)
            row += f"  {val:16.3f}" if val is not None else f"  {'N/A':>16s}"
        print(row)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'configs': {cn: cfg for cn, cfg in CONFIGS.items()},
        'per_config_results': all_results,
        'medians': all_medians,
    }
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'diagnostic_dvalue_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
