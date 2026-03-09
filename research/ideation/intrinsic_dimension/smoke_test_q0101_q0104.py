"""
Smoke test for Intrinsic Dimension Detectors (Q0101-Q0104).

Evaluates four dimension-based regime detectors on real market data:
  Q0101: IntrinsicDimensionDetector — spectral gap dimension estimate
  Q0102: SpectralGapRatioDetector  — gap sharpness as severity measure
  Q0103: DimensionRateDetector     — |dd/dt| for early warning
  Q0104: EffectiveDimensionDetector — participation ratio PR vs IPR

Symbols: SPY, DIA (2005-2024)
Metric: Cohen's d with n_bootstrap=1000 on 4 smoke crises
Output: smoke_results_q0101_q0104.json
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
from qcml_geometry.observables import BaseRegimeDetector
from research.ideation.intrinsic_dimension.detectors import (
    IntrinsicDimensionDetector,
    SpectralGapRatioDetector,
    DimensionRateDetector,
    EffectiveDimensionDetector,
)

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)

SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2024-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60


def evaluate_crisis(scores, dates, crisis_key):
    """Compute Cohen's d for a single crisis vs the preceding normal window."""
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


def run_detector(det, features, dates, label, question):
    """Fit a detector, score, evaluate across crises. Return summary dict."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {label} ({question})")
    logger.info(f"{'='*60}")

    t_fit = time.time()
    det.fit(features)
    dt_fit = time.time() - t_fit
    logger.info(f"  Fit: {dt_fit:.1f}s")

    t_scores = time.time()
    scores = det.compute_regime_scores(features)
    dt_scores = time.time() - t_scores
    n_valid = int(np.sum(~np.isnan(scores)))
    logger.info(f"  Scores: {n_valid}/{len(scores)} valid, in {dt_scores:.1f}s")

    per_crisis = {}
    for ck in TEST_CRISES:
        per_crisis[ck] = evaluate_crisis(scores, dates, ck)
        r = per_crisis[ck]
        if r['cohens_d'] is not None:
            logger.info(f"  {ck}: d={r['cohens_d']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

    d_values = [r['cohens_d'] for r in per_crisis.values() if r['cohens_d'] is not None]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None

    return {
        'detector': label,
        'question': question,
        'cohens_d_per_crisis': {k: v['cohens_d'] for k, v in per_crisis.items()},
        'per_crisis_detail': per_crisis,
        'median_d': median_d,
        'max_d': max_d,
        'passes_threshold': median_d is not None and median_d > 0.3,
        'timing': {
            'fit_seconds': round(dt_fit, 1),
            'score_seconds': round(dt_scores, 1),
        },
    }


def main():
    t0 = time.time()

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info(f"Fetching {SYMBOLS} {START_DATE} to {END_DATE}")
    df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close = df['close'].unstack('symbol').dropna()
    logger.info(f"Data: {close.shape}, {close.index[0]} to {close.index[-1]}")

    features, dates = create_feature_matrix(close)
    dates = pd.DatetimeIndex(dates)
    logger.info(f"Features: {features.shape}")

    enriched = BaseRegimeDetector.build_enriched_features(features, lookback=20)
    enriched_dates = dates[19:]  # lookback - 1
    logger.info(f"Enriched features: {enriched.shape}")

    # ── Shared config ─────────────────────────────────────────────────────
    cfg = dict(
        hilbert_dim=4,
        n_pca_components=8,
        operator_method='random',
        rolling_window=10,
        min_expanding=60,
        seed=42,
        normalization='soft',
        adaptive_epsilon=True,
        subsample=1,
    )

    # ── Detectors ─────────────────────────────────────────────────────────
    detectors = [
        (IntrinsicDimensionDetector(**cfg), "Intrinsic Dimension", "Q0101"),
        (SpectralGapRatioDetector(**cfg), "Spectral Gap Ratio", "Q0102"),
        (DimensionRateDetector(rate_window=5, **cfg), "Dimension Rate (dd/dt)", "Q0103"),
        (EffectiveDimensionDetector(**cfg), "Effective Dimension (PR)", "Q0104"),
    ]

    results = []
    for det, label, question in detectors:
        r = run_detector(det, enriched, enriched_dates, label, question)
        r['config'] = {**cfg}
        if question == 'Q0103':
            r['config']['rate_window'] = 5
        results.append(r)

    total_time = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'experiment': 'Intrinsic Dimension Smoke Test (Q0101-Q0104)',
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
            'enriched_shape': list(enriched.shape),
        },
        'crises_tested': TEST_CRISES,
        'n_bootstrap': N_BOOTSTRAP,
        'detectors': results,
        'timing': {'total_seconds': round(total_time, 1)},
    }

    out = os.path.join(os.path.dirname(__file__), 'smoke_results_q0101_q0104.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved to {out}")

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("INTRINSIC DIMENSION DETECTORS (Q0101-Q0104) -- SMOKE TEST RESULTS")
    print("=" * 78)
    header = f"{'Detector':<28s}"
    for ck in TEST_CRISES:
        header += f"  {ck:>12s}"
    header += f"  {'Median':>8s}  {'Pass':>5s}"
    print(header)
    print("-" * 78)

    for r in results:
        row = f"{r['detector']:<28s}"
        for ck in TEST_CRISES:
            d = r['cohens_d_per_crisis'].get(ck)
            row += f"  {d:>12.3f}" if d is not None else f"  {'NaN':>12s}"
        md = r['median_d']
        row += f"  {md:>8.3f}" if md is not None else f"  {'NaN':>8s}"
        row += f"  {'YES' if r['passes_threshold'] else 'no':>5s}"
        print(row)

    print("-" * 78)
    print(f"Total time: {total_time:.1f}s")
    print(f"Threshold: median d > 0.3")
    print(f"Benchmark: IPR (DimensionalityCollapseDetector) d=0.793")
    print("=" * 78)

    return summary


if __name__ == '__main__':
    main()
