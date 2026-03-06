"""
Smoke test for Modular Hamiltonian Detector (Q13).

Evaluates Tr(K^2) of K = -log(rho_A) as a regime detector over 4 crises:
2008_gfc, 2020_covid, 2022_rates, 2023_svb

Symbols: SPY, DIA (2005-2025)
Metric: Cohen's d with n_bootstrap=1000
Output: smoke_results.json
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.modular_hamiltonian.detector import ModularHamiltonianDetector

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)

SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60


def evaluate_crisis(scores, dates, crisis_key):
    crisis = ALL_CRISES[crisis_key]
    crisis_start = pd.Timestamp(crisis['start'])
    crisis_end = pd.Timestamp(crisis['end'])
    normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

    crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
    normal_mask = (dates >= normal_start) & (dates < crisis_start)

    crisis_scores = scores[np.asarray(crisis_mask)]
    normal_scores = scores[np.asarray(normal_mask)]
    crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
    normal_valid = normal_scores[~np.isnan(normal_scores)]

    logger.info(f"  {crisis_key}: crisis_n={len(crisis_valid)}, normal_n={len(normal_valid)}")

    if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_valid, normal_valid, n_bootstrap=N_BOOTSTRAP, seed=42)
    else:
        d, ci_lo, ci_hi = np.nan, np.nan, np.nan

    return {
        'cohens_d': float(d) if not np.isnan(d) else None,
        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        'crisis_n': int(len(crisis_valid)),
        'normal_n': int(len(normal_valid)),
        'crisis_mean': float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
        'normal_mean': float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
    }


def main():
    t0 = time.time()

    logger.info(f"Fetching {SYMBOLS} {START_DATE} to {END_DATE}")
    df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close = df['close'].unstack('symbol').dropna()
    logger.info(f"Data: {close.shape}, {close.index[0]} to {close.index[-1]}")

    features, dates = create_feature_matrix(close)
    dates = pd.DatetimeIndex(dates)
    logger.info(f"Features: {features.shape}")

    detector = ModularHamiltonianDetector(
        hilbert_dim=4, dim_A=2, dim_B=2,
        n_pca_components=8, operator_method='random',
        normalization='soft', rolling_window=20, min_expanding=60,
        seed=42, negate_score=True,
    )

    t_fit = time.time()
    detector.fit(features)
    dt_fit = time.time() - t_fit
    logger.info(f"Fit in {dt_fit:.1f}s")

    t_scores = time.time()
    scores = detector.compute_regime_scores(features)
    dt_scores = time.time() - t_scores
    n_valid = int(np.sum(~np.isnan(scores)))
    logger.info(f"Scores: {n_valid}/{len(scores)} valid, in {dt_scores:.1f}s")

    per_crisis = {}
    for ck in TEST_CRISES:
        per_crisis[ck] = evaluate_crisis(scores, dates, ck)
        r = per_crisis[ck]
        if r['cohens_d'] is not None:
            logger.info(f"  {ck}: d={r['cohens_d']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

    d_values = [r['cohens_d'] for r in per_crisis.values() if r['cohens_d'] is not None]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None
    total_time = time.time() - t0

    summary = {
        'detector': 'Modular Hamiltonian',
        'question': 'Q13',
        'config': {
            'hilbert_dim': 4, 'dim_A': 2, 'dim_B': 2,
            'operator_method': 'random', 'normalization': 'soft',
            'rolling_window': 20, 'min_expanding': 60,
            'negate_score': True, 'n_bootstrap': N_BOOTSTRAP,
        },
        'data': {
            'symbols': SYMBOLS, 'start_date': START_DATE, 'end_date': END_DATE,
            'feature_shape': list(features.shape), 'n_valid_scores': n_valid,
        },
        'cohens_d_per_crisis': {k: v['cohens_d'] for k, v in per_crisis.items()},
        'per_crisis_detail': per_crisis,
        'median_d': median_d,
        'max_d': max_d,
        'passes_threshold': median_d is not None and median_d > 0.3,
        'timing': {
            'total_seconds': round(total_time, 1),
            'fit_seconds': round(dt_fit, 1),
            'score_seconds': round(dt_scores, 1),
        },
    }

    out = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved to {out}")

    print("\n" + "=" * 65)
    print("MODULAR HAMILTONIAN (Q13) — SMOKE TEST RESULTS")
    print("=" * 65)
    for ck in TEST_CRISES:
        r = per_crisis[ck]
        d = r['cohens_d']
        if d is not None:
            print(f"  {ck:22s}  d={d:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                  f"  (n_c={r['crisis_n']}, n_n={r['normal_n']})")
        else:
            print(f"  {ck:22s}  d=NaN")
    print(f"\n  Median d: {median_d:.3f}" if median_d else "  Median d: NaN")
    print(f"  Max d:    {max_d:.3f}" if max_d else "  Max d: NaN")
    print(f"  Passes threshold: {summary['passes_threshold']}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 65)

    return summary


if __name__ == '__main__':
    main()
