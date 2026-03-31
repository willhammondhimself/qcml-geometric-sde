"""
Smoke test for Renyi Entropy Detector on 4 financial crises.

Q0124: "Can Renyi entropy (alpha=2, alpha=0.5) outperform von Neumann entropy
        for regime detection due to different tail sensitivity?"

Tests alpha = [0.5, 1.0 (von Neumann), 2.0, 3.0] on the REDUCED density matrix.
For 3-qubit system (hilbert_dim=8), partition = (4, 2): trace out 1 qubit, keep 2.

Key analysis:
  - Cohen's d per crisis per alpha
  - Correlation between Renyi-2 z-scores and Reduced Purity z-scores
  - Correlation between Renyi-0.5 z-scores and Reduced Purity z-scores

Test crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Metric: Cohen's d with n_bootstrap=1000 (block bootstrap)
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

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.renyi_entropy.detector import RenyiEntropyDetector

# Also import existing ReducedPurityDetector for correlation comparison
from qcml_geometry.observables import ReducedPurityDetector

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOLS = ['SPY', 'DIA']
START_DATE = '2006-01-01'
END_DATE = '2024-12-31'

TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
ALPHA_VALUES = [0.5, 1.0, 2.0, 3.0]
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60  # "normal" baseline: 60 trading days before crisis start

# Hilbert space: 3-qubit (dim=8), partition (4,2): keep 2 qubits, trace 1
HILBERT_DIM = 8
DIM_A = 4
DIM_B = 2


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_crisis(
    scores: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_key: str,
) -> dict:
    """Evaluate Cohen's d for a single crisis.

    Args:
        scores: Regime score time series of length T.
        dates: Corresponding timestamps, length T.
        crisis_key: Key into ALL_CRISES dict.

    Returns:
        dict with cohens_d, ci_lo, ci_hi, crisis_n, normal_n, etc.
    """
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

    logger.info(
        f"  {crisis_key}: crisis_n={len(crisis_valid)}, normal_n={len(normal_valid)}"
    )

    if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_valid, normal_valid,
            n_bootstrap=N_BOOTSTRAP, seed=42,
        )
    else:
        d, ci_lo, ci_hi = np.nan, np.nan, np.nan
        logger.warning(f"  {crisis_key}: insufficient data for Cohen's d")

    return {
        'cohens_d': float(d) if not np.isnan(d) else None,
        'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
        'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        'crisis_n': int(len(crisis_valid)),
        'normal_n': int(len(normal_valid)),
        'crisis_mean': float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
        'normal_mean': float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    implementation_issues = []

    # -------------------------------------------------------------------------
    # Step 1: Fetch data
    # -------------------------------------------------------------------------
    logger.info(f"Fetching data: {SYMBOLS}, {START_DATE} to {END_DATE}")
    df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices: {close_prices.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Create feature matrix
    # -------------------------------------------------------------------------
    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    dates = pd.DatetimeIndex(dates)
    logger.info(f"Feature matrix: {features.shape}, {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 3: Run Renyi detectors for each alpha
    # -------------------------------------------------------------------------
    all_scores = {}  # alpha -> scores array
    all_results = {}  # alpha -> {crisis_key: result_dict}
    timing = {}

    for alpha in ALPHA_VALUES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Renyi-{alpha} detector...")
        logger.info(f"  hilbert_dim={HILBERT_DIM}, partition=({DIM_A},{DIM_B})")

        detector = RenyiEntropyDetector(
            hilbert_dim=HILBERT_DIM,
            dim_A=DIM_A,
            dim_B=DIM_B,
            alpha=alpha,
            n_pca_components=8,
            operator_method='random',
            normalization='soft',
            rolling_window=20,
            min_expanding=60,
            seed=42,
        )

        # Fit
        t_fit = time.time()
        detector.fit(features)
        dt_fit = time.time() - t_fit

        # Compute scores
        t_scores = time.time()
        scores = detector.compute_regime_scores(features)
        dt_scores = time.time() - t_scores

        n_valid = int(np.sum(~np.isnan(scores)))
        logger.info(f"Renyi-{alpha}: scores computed in {dt_scores:.1f}s, valid={n_valid}/{len(scores)}")

        all_scores[alpha] = scores
        timing[f'alpha_{alpha}'] = {
            'fit_seconds': round(dt_fit, 1),
            'score_seconds': round(dt_scores, 1),
        }

        if n_valid == 0:
            implementation_issues.append(f"Renyi-{alpha}: all scores NaN")

        # Evaluate per-crisis
        per_crisis = {}
        for crisis_key in TEST_CRISES:
            per_crisis[crisis_key] = evaluate_crisis(scores, dates, crisis_key)
            r = per_crisis[crisis_key]
            if r['cohens_d'] is not None:
                logger.info(
                    f"  {crisis_key}: d={r['cohens_d']:.3f} "
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                )

        all_results[alpha] = per_crisis

    # -------------------------------------------------------------------------
    # Step 4: Reduced Purity baseline for correlation analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Running Reduced Purity baseline for correlation analysis...")

    purity_detector = ReducedPurityDetector(
        hilbert_dim=HILBERT_DIM,
        n_pca_components=8,
        operator_method='random',
        normalization='soft',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        partition=(DIM_A, DIM_B),
    )
    purity_detector.fit(features)
    purity_scores = purity_detector.compute_regime_scores(features)
    n_valid_purity = int(np.sum(~np.isnan(purity_scores)))
    logger.info(f"Reduced Purity: valid={n_valid_purity}/{len(purity_scores)}")

    # Evaluate purity per-crisis for reference
    purity_per_crisis = {}
    for crisis_key in TEST_CRISES:
        purity_per_crisis[crisis_key] = evaluate_crisis(purity_scores, dates, crisis_key)

    # -------------------------------------------------------------------------
    # Step 5: Correlation analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Correlation analysis: Renyi vs Reduced Purity z-scores")

    correlations = {}
    for alpha in ALPHA_VALUES:
        renyi_s = all_scores[alpha]
        # Only compare where both are valid
        both_valid = (~np.isnan(renyi_s)) & (~np.isnan(purity_scores))
        if np.sum(both_valid) > 10:
            r = np.corrcoef(renyi_s[both_valid], purity_scores[both_valid])[0, 1]
            correlations[f'renyi_{alpha}_vs_purity'] = round(float(r), 4)
            logger.info(f"  corr(Renyi-{alpha}, Purity) = {r:.4f}  (n={np.sum(both_valid)})")
        else:
            correlations[f'renyi_{alpha}_vs_purity'] = None
            logger.warning(f"  corr(Renyi-{alpha}, Purity): insufficient overlap")

    # Cross-correlations between Renyi alphas
    for i, a1 in enumerate(ALPHA_VALUES):
        for a2 in ALPHA_VALUES[i + 1:]:
            s1 = all_scores[a1]
            s2 = all_scores[a2]
            both_valid = (~np.isnan(s1)) & (~np.isnan(s2))
            if np.sum(both_valid) > 10:
                r = np.corrcoef(s1[both_valid], s2[both_valid])[0, 1]
                correlations[f'renyi_{a1}_vs_renyi_{a2}'] = round(float(r), 4)
                logger.info(f"  corr(Renyi-{a1}, Renyi-{a2}) = {r:.4f}")

    # -------------------------------------------------------------------------
    # Step 6: Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - t0

    # Build per-alpha summary
    alpha_summaries = {}
    for alpha in ALPHA_VALUES:
        d_values = [
            r['cohens_d'] for r in all_results[alpha].values()
            if r['cohens_d'] is not None
        ]
        median_d = float(np.median(d_values)) if d_values else None
        max_d = float(np.max(d_values)) if d_values else None
        mean_d = float(np.mean(d_values)) if d_values else None

        alpha_summaries[f'alpha_{alpha}'] = {
            'cohens_d_per_crisis': {
                k: v['cohens_d'] for k, v in all_results[alpha].items()
            },
            'per_crisis_detail': all_results[alpha],
            'median_d': median_d,
            'max_d': max_d,
            'mean_d': mean_d,
        }

    # Purity baseline summary
    purity_d_values = [
        r['cohens_d'] for r in purity_per_crisis.values()
        if r['cohens_d'] is not None
    ]

    summary = {
        'knight': 2,
        'role': 'empirical_test',
        'question': 'Q0124',
        'detector': 'Renyi Entropy',
        'config': {
            'hilbert_dim': HILBERT_DIM,
            'dim_A': DIM_A,
            'dim_B': DIM_B,
            'alpha_values': ALPHA_VALUES,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'n_bootstrap': N_BOOTSTRAP,
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
        },
        'data': {
            'feature_shape': list(features.shape),
            'n_valid_scores': {
                f'alpha_{a}': int(np.sum(~np.isnan(all_scores[a])))
                for a in ALPHA_VALUES
            },
        },
        'alpha_results': alpha_summaries,
        'purity_baseline': {
            'cohens_d_per_crisis': {
                k: v['cohens_d'] for k, v in purity_per_crisis.items()
            },
            'median_d': float(np.median(purity_d_values)) if purity_d_values else None,
        },
        'correlations': correlations,
        'implementation_issues': implementation_issues,
        'timing': {
            'total_seconds': round(total_time, 1),
            **timing,
        },
    }

    # -------------------------------------------------------------------------
    # Step 7: Save results
    # -------------------------------------------------------------------------
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RENYI ENTROPY DETECTOR -- SMOKE TEST RESULTS (Q0124)")
    print("=" * 80)
    print(f"  Hilbert dim: {HILBERT_DIM} ({DIM_A}x{DIM_B} partition)  |  method=random")
    print(f"  Symbols: {SYMBOLS}  |  Period: {START_DATE} to {END_DATE}")
    print()

    # Table header
    header = f"{'Crisis':22s}"
    for alpha in ALPHA_VALUES:
        header += f"  {'Renyi-'+str(alpha):>10s}"
    header += f"  {'Purity':>10s}"
    print(header)
    print("-" * len(header))

    for crisis_key in TEST_CRISES:
        row = f"  {crisis_key:20s}"
        for alpha in ALPHA_VALUES:
            d = all_results[alpha][crisis_key]['cohens_d']
            row += f"  {d:10.3f}" if d is not None else f"  {'NaN':>10s}"
        d_p = purity_per_crisis[crisis_key]['cohens_d']
        row += f"  {d_p:10.3f}" if d_p is not None else f"  {'NaN':>10s}"
        print(row)

    print("-" * len(header))

    # Median row
    row = f"  {'MEDIAN':20s}"
    for alpha in ALPHA_VALUES:
        md = alpha_summaries[f'alpha_{alpha}']['median_d']
        row += f"  {md:10.3f}" if md is not None else f"  {'NaN':>10s}"
    md_p = float(np.median(purity_d_values)) if purity_d_values else None
    row += f"  {md_p:10.3f}" if md_p is not None else f"  {'NaN':>10s}"
    print(row)

    print()
    print("Correlations with Reduced Purity z-scores:")
    for alpha in ALPHA_VALUES:
        r = correlations.get(f'renyi_{alpha}_vs_purity')
        if r is not None:
            redundant = "|r| > 0.7" if abs(r) > 0.7 else "|r| <= 0.7"
            print(f"  Renyi-{alpha} vs Purity: r={r:.4f}  ({redundant})")

    print()
    print("Cross-correlations between Renyi alphas:")
    for i, a1 in enumerate(ALPHA_VALUES):
        for a2 in ALPHA_VALUES[i + 1:]:
            key = f'renyi_{a1}_vs_renyi_{a2}'
            r = correlations.get(key)
            if r is not None:
                print(f"  Renyi-{a1} vs Renyi-{a2}: r={r:.4f}")

    print()
    print(f"  Total time: {total_time:.1f}s")
    if implementation_issues:
        print(f"\n  Issues: {implementation_issues}")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    main()
