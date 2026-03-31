"""
Smoke test for Channel Capacity (Holevo Information) Detector on 4 financial crises.

Q0125: "Does the quantum channel capacity between consecutive market states
        (Holevo bound) detect information bottlenecks at crisis onset?"

Tests state_window = [20, 40, 60] to assess sensitivity to window size.

Key analysis:
  - Cohen's d per crisis per window size
  - Correlation with Effective State Dimension z-scores
    (D_eff = exp(S_2(rho_avg)) vs chi = S_1(rho_avg); monotonically related
     but S_1 and S_2 differ when eigenvalue spectrum is non-uniform)
  - Direction check: does chi drop or spike during crises?

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
from research.ideation.channel_capacity.detector import ChannelCapacityDetector

# Import Effective State Dimension for correlation comparison
from qcml_geometry.observables import EffectiveStateDimensionDetector

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
WINDOW_SIZES = [20, 40, 60]
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60  # "normal" baseline: 60 trading days before crisis start

HILBERT_DIM = 8


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
    # Step 3: Run Channel Capacity detector for each window size
    # -------------------------------------------------------------------------
    all_scores = {}  # window -> scores array
    all_results = {}  # window -> {crisis_key: result_dict}
    timing = {}

    for W in WINDOW_SIZES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Channel Capacity detector (W={W})...")
        logger.info(f"  hilbert_dim={HILBERT_DIM}, operator_method=random")

        detector = ChannelCapacityDetector(
            hilbert_dim=HILBERT_DIM,
            state_window=W,
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
        logger.info(f"W={W}: scores computed in {dt_scores:.1f}s, valid={n_valid}/{len(scores)}")

        all_scores[W] = scores
        timing[f'W_{W}'] = {
            'fit_seconds': round(dt_fit, 1),
            'score_seconds': round(dt_scores, 1),
        }

        if n_valid == 0:
            implementation_issues.append(f"W={W}: all scores NaN")

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

        all_results[W] = per_crisis

    # -------------------------------------------------------------------------
    # Step 4: Effective State Dimension baseline for correlation analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Running Effective State Dimension baseline for correlation analysis...")

    esd_detector = EffectiveStateDimensionDetector(
        hilbert_dim=HILBERT_DIM,
        n_pca_components=8,
        operator_method='random',
        normalization='soft',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        state_window=20,
    )
    esd_detector.fit(features)
    esd_scores = esd_detector.compute_regime_scores(features)
    n_valid_esd = int(np.sum(~np.isnan(esd_scores)))
    logger.info(f"Effective State Dim: valid={n_valid_esd}/{len(esd_scores)}")

    # Evaluate ESD per-crisis for reference
    esd_per_crisis = {}
    for crisis_key in TEST_CRISES:
        esd_per_crisis[crisis_key] = evaluate_crisis(esd_scores, dates, crisis_key)

    # -------------------------------------------------------------------------
    # Step 5: Correlation analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Correlation analysis: Channel Capacity vs Effective State Dimension")

    correlations = {}
    for W in WINDOW_SIZES:
        cc_s = all_scores[W]
        # Only compare where both are valid
        both_valid = (~np.isnan(cc_s)) & (~np.isnan(esd_scores))
        if np.sum(both_valid) > 10:
            r = np.corrcoef(cc_s[both_valid], esd_scores[both_valid])[0, 1]
            correlations[f'W_{W}_vs_esd'] = round(float(r), 4)
            logger.info(f"  corr(CC_W{W}, ESD) = {r:.4f}  (n={np.sum(both_valid)})")
        else:
            correlations[f'W_{W}_vs_esd'] = None
            logger.warning(f"  corr(CC_W{W}, ESD): insufficient overlap")

    # Cross-correlations between window sizes
    for i, w1 in enumerate(WINDOW_SIZES):
        for w2 in WINDOW_SIZES[i + 1:]:
            s1 = all_scores[w1]
            s2 = all_scores[w2]
            both_valid = (~np.isnan(s1)) & (~np.isnan(s2))
            if np.sum(both_valid) > 10:
                r = np.corrcoef(s1[both_valid], s2[both_valid])[0, 1]
                correlations[f'W_{w1}_vs_W_{w2}'] = round(float(r), 4)
                logger.info(f"  corr(W={w1}, W={w2}) = {r:.4f}")

    # -------------------------------------------------------------------------
    # Step 6: Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - t0

    # Build per-window summary
    window_summaries = {}
    for W in WINDOW_SIZES:
        d_values = [
            r['cohens_d'] for r in all_results[W].values()
            if r['cohens_d'] is not None
        ]
        median_d = float(np.median(d_values)) if d_values else None
        max_d = float(np.max(d_values)) if d_values else None
        mean_d = float(np.mean(d_values)) if d_values else None

        window_summaries[f'W_{W}'] = {
            'cohens_d_per_crisis': {
                k: v['cohens_d'] for k, v in all_results[W].items()
            },
            'per_crisis_detail': all_results[W],
            'median_d': median_d,
            'max_d': max_d,
            'mean_d': mean_d,
        }

    # ESD baseline summary
    esd_d_values = [
        r['cohens_d'] for r in esd_per_crisis.values()
        if r['cohens_d'] is not None
    ]

    summary = {
        'knight': 2,
        'role': 'empirical_test',
        'question': 'Q0125',
        'detector': 'Channel Capacity (Holevo Information)',
        'config': {
            'hilbert_dim': HILBERT_DIM,
            'state_windows': WINDOW_SIZES,
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
                f'W_{W}': int(np.sum(~np.isnan(all_scores[W])))
                for W in WINDOW_SIZES
            },
        },
        'window_results': window_summaries,
        'esd_baseline': {
            'cohens_d_per_crisis': {
                k: v['cohens_d'] for k, v in esd_per_crisis.items()
            },
            'median_d': float(np.median(esd_d_values)) if esd_d_values else None,
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
    print("CHANNEL CAPACITY (HOLEVO INFORMATION) -- SMOKE TEST RESULTS (Q0125)")
    print("=" * 80)
    print(f"  Hilbert dim: {HILBERT_DIM}  |  method=random")
    print(f"  Symbols: {SYMBOLS}  |  Period: {START_DATE} to {END_DATE}")
    print()

    # Table header
    header = f"{'Crisis':22s}"
    for W in WINDOW_SIZES:
        header += f"  {'W='+str(W):>10s}"
    header += f"  {'ESD':>10s}"
    print(header)
    print("-" * len(header))

    for crisis_key in TEST_CRISES:
        row = f"  {crisis_key:20s}"
        for W in WINDOW_SIZES:
            d = all_results[W][crisis_key]['cohens_d']
            row += f"  {d:10.3f}" if d is not None else f"  {'NaN':>10s}"
        d_esd = esd_per_crisis[crisis_key]['cohens_d']
        row += f"  {d_esd:10.3f}" if d_esd is not None else f"  {'NaN':>10s}"
        print(row)

    print("-" * len(header))

    # Median row
    row = f"  {'MEDIAN':20s}"
    for W in WINDOW_SIZES:
        md = window_summaries[f'W_{W}']['median_d']
        row += f"  {md:10.3f}" if md is not None else f"  {'NaN':>10s}"
    md_esd = float(np.median(esd_d_values)) if esd_d_values else None
    row += f"  {md_esd:10.3f}" if md_esd is not None else f"  {'NaN':>10s}"
    print(row)

    print()
    print("Correlations with Effective State Dimension z-scores:")
    for W in WINDOW_SIZES:
        r = correlations.get(f'W_{W}_vs_esd')
        if r is not None:
            redundant = "HIGH (likely redundant)" if abs(r) > 0.7 else "moderate/low"
            print(f"  CC(W={W}) vs ESD: r={r:.4f}  ({redundant})")

    print()
    print("Cross-correlations between window sizes:")
    for i, w1 in enumerate(WINDOW_SIZES):
        for w2 in WINDOW_SIZES[i + 1:]:
            key = f'W_{w1}_vs_W_{w2}'
            r = correlations.get(key)
            if r is not None:
                print(f"  W={w1} vs W={w2}: r={r:.4f}")

    print()
    print(f"  Total time: {total_time:.1f}s")
    if implementation_issues:
        print(f"\n  Issues: {implementation_issues}")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    main()
