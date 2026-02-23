"""
Systematic operator benchmark: 6 methods × 4 scale_exponents × 3 detectors × 4 crises.

Tests whether structured operators (n-qubit Pauli, Gell-Mann) outperform
the current random Hermitian fallback for hilbert_dim=8.

Usage:
    python experiments/operator_benchmark.py
    python experiments/operator_benchmark.py --full  # all 16 crises

Outputs:
    experiments/outputs/regime_detection/operator_benchmark_YYYYMMDD_HHMMSS.json
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
from experiments.data_loader import fetch_polygon_data, create_feature_matrix, ALL_CRISES
from experiments.baselines import CUSUMDetector, RandomForestRegimeDetector
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

QUICK_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']

# Operator methods to test
OPERATOR_METHODS = [
    'random',        # Current baseline (seeded random Hermitian)
    'pca_inspired',  # Random Hermitian × √eigenvalue (current default for QFI/MLF)
    'pauli',         # N-qubit Pauli tensor products (NEW: proper structured basis)
    'gell_mann',     # Generalized Gell-Mann SU(N) generators (NEW)
    'pca_pauli',     # Pauli basis × eigenvalue^exp (NEW)
    'pca_gell_mann', # Gell-Mann basis × eigenvalue^exp (NEW)
]

# Scale exponents (only relevant for pca_* methods; None for unscaled methods)
SCALE_EXPONENTS = [0.0, 0.25, 0.5, 1.0]

DETECTOR_CLASSES = {
    'Berry': BerryPhaseRateDetector,
    'QFI': QFIDeterminantDetector,
    'MLF': MultiLagFidelityDetector,
}


def fetch_and_prepare():
    """Fetch data and build enriched feature matrix."""
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    return X, dates, X_enriched, dates_enriched


def compute_d_for_detector(det, X_enriched, dates_enriched, crises, n_bootstrap=2000):
    """Fit detector and compute Cohen's d for each crisis.

    Returns:
        dict: {crisis_key: d_value}
    """
    det.fit(X_enriched)
    scores = det.compute_regime_scores(X_enriched)

    window_size = 10
    crisis_ds = {}
    for ck, ci in crises.items():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask
        d, _, _ = compute_cohens_d_with_ci(
            scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap
        )
        crisis_ds[ck] = round(float(d), 3) if not np.isnan(d) else None

    return crisis_ds


def run_benchmark(crisis_keys, n_bootstrap=2000):
    """Run the full operator benchmark.

    Returns:
        list of dicts: [{method, scale_exp, detector, crisis_key, d_value}, ...]
    """
    logger.info("Fetching data...")
    X, dates, X_enriched, dates_enriched = fetch_and_prepare()
    crises = {k: ALL_CRISES[k] for k in crisis_keys}
    logger.info(f"Data: X_enriched={X_enriched.shape}, {len(crises)} crises")

    shared = dict(hilbert_dim=8, n_pca_components=15, rolling_window=20, seed=42)
    results = []
    total_configs = 0

    # Count total configs
    for method in OPERATOR_METHODS:
        if method.startswith('pca_'):
            total_configs += len(SCALE_EXPONENTS) * len(DETECTOR_CLASSES)
        else:
            total_configs += len(DETECTOR_CLASSES)

    done = 0
    for method in OPERATOR_METHODS:
        exponents = SCALE_EXPONENTS if method.startswith('pca_') else [None]

        for exp in exponents:
            for det_name, det_cls in DETECTOR_CLASSES.items():
                done += 1
                exp_str = f"exp={exp}" if exp is not None else "unscaled"
                logger.info(f"[{done}/{total_configs}] {det_name} / {method} / {exp_str}")

                kwargs = {**shared, 'operator_method': method}
                if exp is not None:
                    kwargs['scale_exponent'] = exp

                det = det_cls(**kwargs)

                try:
                    crisis_ds = compute_d_for_detector(
                        det, X_enriched, dates_enriched, crises, n_bootstrap
                    )
                except Exception as e:
                    logger.warning(f"  FAILED: {e}")
                    crisis_ds = {ck: None for ck in crisis_keys}

                vals = [v for v in crisis_ds.values() if v is not None]
                median_d = round(float(np.median(vals)), 3) if vals else None

                result_row = {
                    'method': method,
                    'scale_exponent': exp,
                    'detector': det_name,
                    'median_d': median_d,
                    'per_crisis': crisis_ds,
                }
                results.append(result_row)

                logger.info(f"    median d = {median_d}")

    # Also run CUSUM and RF baselines for reference
    logger.info("\nRunning baselines (CUSUM, RF)...")

    cusum = CUSUMDetector(burn_in=60)
    cusum_ds = compute_d_for_detector(cusum, X_enriched, dates_enriched, crises, n_bootstrap)
    cusum_vals = [v for v in cusum_ds.values() if v is not None]
    results.append({
        'method': 'CUSUM_baseline',
        'scale_exponent': None,
        'detector': 'CUSUM',
        'median_d': round(float(np.median(cusum_vals)), 3) if cusum_vals else None,
        'per_crisis': cusum_ds,
    })
    logger.info(f"  CUSUM baseline: median d = {np.median(cusum_vals):.3f}")

    # RF with leave-one-crisis-out
    rf_ds = {}
    for held_out_key in crises:
        y = np.zeros(len(X))
        for ck, ci in crises.items():
            if ck == held_out_key:
                continue
            cs, ce = pd.Timestamp(ci['start']), pd.Timestamp(ci['end'])
            mask = (dates >= cs) & (dates <= ce)
            y[mask] = 1.0

        rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
        rf.fit_with_labels(X, y)
        scores = rf.compute_regime_scores(X)
        scores = scores[19:] if len(scores) > len(dates_enriched) else scores

        window_size = 10
        ci_info = crises[held_out_key]
        cs = pd.Timestamp(ci_info['start']) - pd.Timedelta(days=window_size)
        ce = pd.Timestamp(ci_info['end']) + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask
        d, _, _ = compute_cohens_d_with_ci(scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap)
        rf_ds[held_out_key] = round(float(d), 3) if not np.isnan(d) else None

    rf_vals = [v for v in rf_ds.values() if v is not None]
    results.append({
        'method': 'RF_baseline',
        'scale_exponent': None,
        'detector': 'RF',
        'median_d': round(float(np.median(rf_vals)), 3) if rf_vals else None,
        'per_crisis': rf_ds,
    })
    logger.info(f"  RF baseline: median d = {np.median(rf_vals):.3f}")

    return results


def print_results_table(results):
    """Print ranked results table."""
    # Sort by median d descending
    sorted_results = sorted(results, key=lambda r: r['median_d'] or 0, reverse=True)

    print("\n" + "=" * 90)
    print("OPERATOR BENCHMARK RESULTS — Ranked by median Cohen's d")
    print("=" * 90)
    print(f"{'Rank':>4}  {'Detector':>8}  {'Method':>15}  {'Exp':>5}  {'Median d':>9}  Per-crisis d-values")
    print("-" * 90)

    for i, r in enumerate(sorted_results):
        exp_str = f"{r['scale_exponent']:.1f}" if r['scale_exponent'] is not None else "  -"
        crisis_str = "  ".join(
            f"{ck.split('_')[1][:5]}={v:.2f}" if v is not None else f"{ck.split('_')[1][:5]}=N/A"
            for ck, v in r['per_crisis'].items()
        )
        med = f"{r['median_d']:.3f}" if r['median_d'] is not None else "  N/A"
        print(f"{i+1:>4}  {r['detector']:>8}  {r['method']:>15}  {exp_str:>5}  {med:>9}  {crisis_str}")

    # Summary by detector
    print("\n" + "=" * 90)
    print("BEST CONFIG PER DETECTOR")
    print("=" * 90)

    for det_name in list(DETECTOR_CLASSES.keys()) + ['CUSUM', 'RF']:
        det_results = [r for r in sorted_results if r['detector'] == det_name]
        if det_results:
            best = det_results[0]
            exp_str = f"exp={best['scale_exponent']}" if best['scale_exponent'] is not None else "unscaled"
            print(f"  {det_name:>8}: {best['method']} ({exp_str}) → median d = {best['median_d']:.3f}")

    # Summary by method family
    print("\n" + "=" * 90)
    print("AVERAGE MEDIAN d BY OPERATOR METHOD (across detectors)")
    print("=" * 90)

    method_families = {}
    for r in results:
        if r['method'] in ('CUSUM_baseline', 'RF_baseline'):
            continue
        key = r['method']
        if r['method'].startswith('pca_') and r['scale_exponent'] is not None:
            key = f"{r['method']}(exp={r['scale_exponent']})"
        if key not in method_families:
            method_families[key] = []
        if r['median_d'] is not None:
            method_families[key].append(r['median_d'])

    for key in sorted(method_families.keys(), key=lambda k: np.mean(method_families[k]) if method_families[k] else 0, reverse=True):
        vals = method_families[key]
        if vals:
            print(f"  {key:>30}: mean={np.mean(vals):.3f}  (Berry={vals[0] if len(vals)>0 else 'N/A':.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run all 16 crises (slow)')
    args = parser.parse_args()

    crisis_keys = list(ALL_CRISES.keys()) if args.full else QUICK_CRISES

    results = run_benchmark(crisis_keys)
    print_results_table(results)

    # Save
    outdir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = outdir / f'operator_benchmark_{ts}.json'
    with open(outpath, 'w') as f:
        json.dump({
            'timestamp': ts,
            'crisis_keys': crisis_keys,
            'results': results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
