"""
Smoke test for Quantum Relative Entropy Detector on 4 financial crises.

Tests 4 variants:
  1. D(rho_t || rho_ref) with W=20 rolling window
  2. D(rho_t || rho_ref) with W=60 rolling window
  3. Simple infidelity 1 - <psi_t|rho_ref|psi_t> with W=20
  4. D(rho_t || rho_ref) with expanding window

Also computes correlation with Multi-Lag Fidelity for redundancy check.

Data: SPY, DIA from 2005-01-01 to 2025-12-31
Metric: Cohen's d (crisis vs normal) with bootstrap CI (n=2000)
Crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
"""

import json
import sys
import time
import os
import warnings
import logging

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.quantum_relative_entropy.detector import QuantumRelativeEntropyDetector

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# Configuration
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 2000  # Smoke test precision
CONTEXT_DAYS = 60  # Days before crisis for "normal" baseline

# Variant configurations
VARIANTS = {
    'D_W20': {'state_window': 20, 'variant': 'D_W20'},
    'D_W60': {'state_window': 60, 'variant': 'D_W60'},
    'infidelity_W20': {'state_window': 20, 'variant': 'infidelity'},
    'D_expanding': {'state_window': 20, 'variant': 'D_expanding'},  # window ignored for expanding
}


def evaluate_crisis(
    scores: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_key: str,
) -> dict:
    """Compute Cohen's d for a single crisis.

    Args:
        scores: 1-D regime score array.
        dates: DatetimeIndex aligned with scores.
        crisis_key: Key into ALL_CRISES.

    Returns:
        dict with cohens_d, ci_lo, ci_hi, crisis_n, normal_n, means.
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

    if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_valid, normal_valid,
            n_bootstrap=N_BOOTSTRAP, seed=42,
        )
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


def compute_correlation_with_fidelity(
    qre_scores: np.ndarray,
    features: np.ndarray,
) -> float:
    """Compute Spearman correlation between QRE scores and Multi-Lag Fidelity.

    Attempts to import and run MultiLagFidelityDetector. Returns NaN on failure.

    Args:
        qre_scores: QRE regime scores.
        features: Feature matrix for fitting fidelity detector.

    Returns:
        Spearman rho (correlation coefficient).
    """
    try:
        from qcml_geometry.observables import MultiLagFidelityDetector
        fid_det = MultiLagFidelityDetector(
            hilbert_dim=8,
            n_pca_components=8,
            operator_method='random',
            normalization='soft',
            rolling_window=20,
            min_expanding=60,
            seed=42,
        )
        fid_det.fit(features)
        fid_scores = fid_det.compute_regime_scores(features)

        # Correlate on valid overlap
        mask = ~np.isnan(qre_scores) & ~np.isnan(fid_scores)
        if np.sum(mask) < 30:
            return float('nan')

        from scipy.stats import spearmanr
        rho, _ = spearmanr(qre_scores[mask], fid_scores[mask])
        return float(rho)

    except Exception as e:
        logger.warning(f"Could not compute fidelity correlation: {e}")
        return float('nan')


def main():
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Fetch data and build features
    # -------------------------------------------------------------------------
    logger.info(f"Fetching data for {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices shape: {close_prices.shape}")

    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    dates = pd.DatetimeIndex(dates)
    logger.info(f"Feature matrix shape: {features.shape}, dates: {dates[0]} to {dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 2: Run all variants
    # -------------------------------------------------------------------------
    all_variant_results = {}
    all_scores = {}

    for variant_name, variant_cfg in VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_name}")
        logger.info(f"{'='*60}")

        detector = QuantumRelativeEntropyDetector(
            hilbert_dim=8,
            n_pca_components=8,
            operator_method='random',
            normalization='soft',
            rolling_window=20,
            min_expanding=60,
            seed=42,
            **variant_cfg,
        )

        logger.info(f"Fitting detector...")
        detector.fit(features)

        logger.info(f"Computing regime scores...")
        t_scores = time.time()
        scores = detector.compute_regime_scores(features)
        dt_scores = time.time() - t_scores
        logger.info(f"Scores computed in {dt_scores:.1f}s")

        n_valid = int(np.sum(~np.isnan(scores)))
        logger.info(f"Valid scores: {n_valid}/{len(scores)}")

        all_scores[variant_name] = scores

        # Evaluate per-crisis
        per_crisis = {}
        for crisis_key in TEST_CRISES:
            result = evaluate_crisis(scores, dates, crisis_key)
            per_crisis[crisis_key] = result
            d = result['cohens_d']
            ci = f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]" if result['ci_lo'] is not None else "N/A"
            logger.info(f"  {crisis_key:20s}  d={d:.3f}  CI={ci}")

        d_values = [r['cohens_d'] for r in per_crisis.values() if r['cohens_d'] is not None]
        median_d = float(np.median(d_values)) if d_values else None
        max_d = float(np.max(d_values)) if d_values else None

        all_variant_results[variant_name] = {
            'per_crisis': per_crisis,
            'summary': {
                'median_d': median_d,
                'max_d': max_d,
                'd_values': d_values,
                'passes_threshold_0.3': median_d is not None and median_d > 0.3,
            },
            'timing': {
                'score_computation_seconds': round(dt_scores, 1),
            },
            'n_valid_scores': n_valid,
        }

    # -------------------------------------------------------------------------
    # Step 3: Redundancy check -- correlation with Multi-Lag Fidelity
    # -------------------------------------------------------------------------
    logger.info("\nComputing correlation with Multi-Lag Fidelity...")
    correlations = {}
    for variant_name, scores in all_scores.items():
        rho = compute_correlation_with_fidelity(scores, features)
        correlations[variant_name] = rho
        logger.info(f"  {variant_name}: rho = {rho:.3f}")

    # -------------------------------------------------------------------------
    # Step 4: Cross-variant correlation matrix
    # -------------------------------------------------------------------------
    logger.info("\nCross-variant correlation matrix:")
    variant_names = list(all_scores.keys())
    n_variants = len(variant_names)
    cross_corr = np.full((n_variants, n_variants), np.nan)

    from scipy.stats import spearmanr
    for i in range(n_variants):
        for j in range(n_variants):
            s_i = all_scores[variant_names[i]]
            s_j = all_scores[variant_names[j]]
            mask = ~np.isnan(s_i) & ~np.isnan(s_j)
            if np.sum(mask) >= 30:
                cross_corr[i, j], _ = spearmanr(s_i[mask], s_j[mask])

    cross_corr_dict = {}
    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            key = f"{variant_names[i]}_vs_{variant_names[j]}"
            cross_corr_dict[key] = float(cross_corr[i, j]) if not np.isnan(cross_corr[i, j]) else None
            logger.info(f"  {key}: {cross_corr[i, j]:.3f}")

    # -------------------------------------------------------------------------
    # Step 5: Assemble and save results
    # -------------------------------------------------------------------------
    total_time = time.time() - t0

    summary = {
        'detector': 'Quantum Relative Entropy',
        'question': 'Q0127: Does D(rho_t || rho_ref) outperform fidelity-based measures?',
        'config': {
            'hilbert_dim': 8,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'n_bootstrap': N_BOOTSTRAP,
        },
        'data': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'feature_shape': list(features.shape),
        },
        'variants': all_variant_results,
        'correlations_with_fidelity': correlations,
        'cross_variant_correlations': cross_corr_dict,
        'overall_summary': {
            'best_variant': None,
            'best_median_d': None,
            'infidelity_vs_D_comparison': None,
        },
        'timing': {
            'total_seconds': round(total_time, 1),
        },
    }

    # Determine best variant
    best_name = None
    best_median = -1.0
    for vname, vdata in all_variant_results.items():
        md = vdata['summary']['median_d']
        if md is not None and md > best_median:
            best_median = md
            best_name = vname

    summary['overall_summary']['best_variant'] = best_name
    summary['overall_summary']['best_median_d'] = best_median

    # Compare D vs infidelity
    d_w20_median = all_variant_results.get('D_W20', {}).get('summary', {}).get('median_d')
    infid_median = all_variant_results.get('infidelity_W20', {}).get('summary', {}).get('median_d')
    if d_w20_median is not None and infid_median is not None:
        if d_w20_median > infid_median:
            summary['overall_summary']['infidelity_vs_D_comparison'] = (
                f"D outperforms infidelity: D_W20 median_d={d_w20_median:.3f} vs "
                f"infidelity median_d={infid_median:.3f}"
            )
        else:
            summary['overall_summary']['infidelity_vs_D_comparison'] = (
                f"Infidelity matches or outperforms D: infidelity median_d={infid_median:.3f} vs "
                f"D_W20 median_d={d_w20_median:.3f}"
            )

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("QUANTUM RELATIVE ENTROPY DETECTOR -- SMOKE TEST RESULTS")
    print("=" * 70)

    for variant_name, vdata in all_variant_results.items():
        print(f"\n--- {variant_name} ---")
        for crisis_key, r in vdata['per_crisis'].items():
            d = r['cohens_d']
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r['ci_lo'] is not None else "N/A"
            print(f"  {crisis_key:20s}  d={d:.3f}  CI={ci}  (n_crisis={r['crisis_n']}, n_normal={r['normal_n']})")
        md = vdata['summary']['median_d']
        mx = vdata['summary']['max_d']
        print(f"  Median d: {md:.3f}  |  Max d: {mx:.3f}  |  Pass (>0.3): {vdata['summary']['passes_threshold_0.3']}")

    print(f"\n--- Correlation with Multi-Lag Fidelity ---")
    for vname, rho in correlations.items():
        print(f"  {vname}: rho = {rho:.3f}")

    print(f"\n--- Cross-variant correlations ---")
    for pair, rho in cross_corr_dict.items():
        print(f"  {pair}: {rho:.3f}" if rho is not None else f"  {pair}: N/A")

    print(f"\n--- Overall ---")
    print(f"  Best variant: {best_name} (median d = {best_median:.3f})")
    print(f"  D vs infidelity: {summary['overall_summary']['infidelity_vs_D_comparison']}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    main()
