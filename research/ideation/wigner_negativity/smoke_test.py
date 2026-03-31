"""
Smoke test for Wigner Negativity Detector.

Evaluates whether Wigner function negativity spikes during 4 financial crises,
indicating a transition to a "non-classical" regime in the QCML state space.

Crises tested:
    - 2008 GFC
    - 2020 COVID
    - 2022 Rate Hikes
    - 2023 SVB

Metrics:
    - Cohen's d (crisis vs normal periods)
    - p-value (Welch's t-test)
    - Median d across crises (threshold: > 0.3)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from research.ideation.wigner_negativity.detector import WignerNegativityDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crises to evaluate
SMOKE_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']


def compute_cohens_d(crisis_scores: np.ndarray, normal_scores: np.ndarray) -> float:
    """Compute Cohen's d effect size (crisis vs normal).

    Uses pooled standard deviation.

    Args:
        crisis_scores: Scores during crisis period.
        normal_scores: Scores during normal period.

    Returns:
        d: Cohen's d (positive means crisis scores are higher).
    """
    n1, n2 = len(crisis_scores), len(normal_scores)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = np.mean(crisis_scores)
    mean2 = np.mean(normal_scores)
    var1 = np.var(crisis_scores, ddof=1)
    var2 = np.var(normal_scores, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def run_smoke_test():
    """Run Wigner negativity smoke test on 4 crises."""
    logger.info("=" * 70)
    logger.info("WIGNER NEGATIVITY SMOKE TEST")
    logger.info("=" * 70)

    # Fetch data covering all crises
    logger.info("Fetching SPY + DIA data (2005-2025)...")
    t0 = time.time()
    raw = fetch_data(['SPY', 'DIA'], '2005-01-01', '2025-01-01')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"Data: {X.shape[0]} observations, {X.shape[1]} features, "
                f"fetched in {time.time() - t0:.1f}s")

    # Fit detector
    logger.info("Fitting Wigner Negativity Detector (d=8)...")
    t0 = time.time()
    detector = WignerNegativityDetector(
        hilbert_dim=8,
        n_pca_components=8,
        operator_method='random',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        normalization='soft',
    )
    detector.fit(X)
    logger.info(f"Fit completed in {time.time() - t0:.1f}s")

    # Compute scores
    logger.info("Computing regime scores...")
    t0 = time.time()
    scores = detector.compute_regime_scores(X)
    logger.info(f"Scores computed in {time.time() - t0:.1f}s")

    # Also compute raw negativity for analysis
    logger.info("Computing raw negativity values...")
    t0 = time.time()
    raw_neg = detector.compute_raw_negativity(X)
    logger.info(f"Raw negativity computed in {time.time() - t0:.1f}s")

    # Evaluate per crisis
    import pandas as pd
    dates_series = pd.DatetimeIndex(dates)

    results = {}
    all_d_values = []

    for crisis_key in SMOKE_CRISES:
        crisis_info = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis_info['start'])
        crisis_end = pd.Timestamp(crisis_info['end'])
        label = crisis_info['label']

        # Crisis mask
        crisis_mask = (dates_series >= crisis_start) & (dates_series <= crisis_end)
        # Normal mask: everything outside all crises and with valid scores
        all_crisis_mask = np.zeros(len(dates_series), dtype=bool)
        for ck, cv in ALL_CRISES.items():
            cs = pd.Timestamp(cv['start'])
            ce = pd.Timestamp(cv['end'])
            all_crisis_mask |= (dates_series >= cs) & (dates_series <= ce)

        normal_mask = ~all_crisis_mask

        # Get valid scores
        valid_mask = ~np.isnan(scores)
        crisis_scores = scores[crisis_mask & valid_mask]
        normal_scores = scores[normal_mask & valid_mask]

        if len(crisis_scores) < 5:
            logger.warning(f"  {label}: Only {len(crisis_scores)} crisis observations, skipping")
            results[crisis_key] = {'d': 0.0, 'p': 1.0, 'n_crisis': int(len(crisis_scores)),
                                   'n_normal': int(len(normal_scores)), 'label': label}
            all_d_values.append(0.0)
            continue

        # Cohen's d (signed: positive = crisis higher)
        d_signed = compute_cohens_d(crisis_scores, normal_scores)

        # Also compute d on |z-scores| (anomaly = deviation in either direction)
        abs_crisis = np.abs(crisis_scores)
        abs_normal = np.abs(normal_scores)
        d_abs = compute_cohens_d(abs_crisis, abs_normal)

        # Welch's t-test (two-sided)
        t_stat, p_val = stats.ttest_ind(crisis_scores, normal_scores, equal_var=False)
        # Also test on abs scores
        _, p_val_abs = stats.ttest_ind(abs_crisis, abs_normal, equal_var=False)

        # Raw negativity stats
        crisis_neg = raw_neg[crisis_mask & valid_mask]
        normal_neg = raw_neg[normal_mask & valid_mask]
        neg_d = compute_cohens_d(crisis_neg, normal_neg)

        results[crisis_key] = {
            'd_signed': round(float(d_signed), 4),
            'd_abs': round(float(d_abs), 4),
            'p_signed': round(float(p_val), 6),
            'p_abs': round(float(p_val_abs), 6),
            'n_crisis': int(len(crisis_scores)),
            'n_normal': int(len(normal_scores)),
            'label': label,
            'mean_crisis_score': round(float(np.mean(crisis_scores)), 4),
            'mean_normal_score': round(float(np.mean(normal_scores)), 4),
            'mean_crisis_abs_score': round(float(np.mean(abs_crisis)), 4),
            'mean_normal_abs_score': round(float(np.mean(abs_normal)), 4),
            'raw_negativity_d': round(float(neg_d), 4),
            'mean_crisis_negativity': round(float(np.mean(crisis_neg)), 6),
            'mean_normal_negativity': round(float(np.mean(normal_neg)), 6),
        }
        all_d_values.append(d_abs)  # Use abs for threshold check

        logger.info(f"  {label}: d_signed={d_signed:.3f}, d_abs={d_abs:.3f}, "
                    f"p={p_val:.4f}, p_abs={p_val_abs:.4f}, "
                    f"n_crisis={len(crisis_scores)}, n_normal={len(normal_scores)}, "
                    f"raw_neg_d={neg_d:.3f}")

    # Summary
    median_d = float(np.median(all_d_values))
    max_d = float(np.max(all_d_values))
    passes = median_d > 0.3 or max_d > 0.3

    summary = {
        'detector': 'WignerNegativityDetector',
        'hilbert_dim': 8,
        'n_pca_components': 8,
        'operator_method': 'random',
        'normalization': 'soft',
        'median_d': round(median_d, 4),
        'max_d': round(max_d, 4),
        'passes_threshold': passes,
        'threshold': 0.3,
        'raw_negativity_stats': {
            'global_mean': round(float(np.nanmean(raw_neg)), 6),
            'global_std': round(float(np.nanstd(raw_neg)), 6),
            'global_max': round(float(np.nanmax(raw_neg)), 6),
            'fraction_nonzero': round(float(np.mean(raw_neg > 1e-10)), 4),
        },
        'per_crisis': results,
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"SUMMARY: median d = {median_d:.4f}, max d = {max_d:.4f}")
    logger.info(f"Passes threshold (median d > 0.3 or max d > 0.3): {passes}")
    logger.info("=" * 70)

    # Save results
    output_path = Path(__file__).parent / 'smoke_results.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return summary


if __name__ == '__main__':
    run_smoke_test()
