"""
Smoke test for Quantum Coherence Detector.

Evaluates whether quantum coherence (l1-norm) in three different bases
detects regime shifts differently from purity / IPR.

Three variants tested:
    1. Computational basis coherence (related to IPR)
    2. Reference basis coherence (drift from initial equilibrium)
    3. Temporal coherence (state novelty between consecutive steps)

Crises tested:
    - 2008 GFC
    - 2020 COVID
    - 2022 Rate Hikes
    - 2023 SVB

Metrics:
    - Cohen's d (crisis vs normal periods), using |z-score| for anomaly detection
    - Correlation between computational coherence and IPR (to verify redundancy)
    - Median d across crises (threshold: > 0.3)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from research.ideation.quantum_coherence.detector import (
    QuantumCoherenceDetector,
    _expanding_zscore,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SMOKE_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']


def compute_cohens_d(crisis_scores: np.ndarray, normal_scores: np.ndarray) -> float:
    """Compute Cohen's d effect size (crisis vs normal).

    Uses pooled standard deviation. Positive d means crisis scores are higher.

    Args:
        crisis_scores: Scores during crisis period.
        normal_scores: Scores during normal period.

    Returns:
        d: Cohen's d.
    """
    n1, n2 = len(crisis_scores), len(normal_scores)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = np.mean(crisis_scores)
    mean2 = np.mean(normal_scores)
    var1 = np.var(crisis_scores, ddof=1)
    var2 = np.var(normal_scores, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def run_smoke_test():
    """Run quantum coherence smoke test on 4 crises x 3 variants."""
    logger.info("=" * 70)
    logger.info("QUANTUM COHERENCE SMOKE TEST")
    logger.info("=" * 70)

    # Fetch data
    logger.info("Fetching SPY + DIA data (2005-2025)...")
    t0 = time.time()
    raw = fetch_data(['SPY', 'DIA'], '2005-01-01', '2025-01-01')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    logger.info(f"Data: {X.shape[0]} observations, {X.shape[1]} features, "
                f"fetched in {time.time() - t0:.1f}s")

    # Fit detector (variant doesn't matter for compute_all_variants)
    logger.info("Fitting Quantum Coherence Detector (d=8)...")
    t0 = time.time()
    detector = QuantumCoherenceDetector(
        hilbert_dim=8,
        n_pca_components=8,
        operator_method='random',
        rolling_window=20,
        min_expanding=60,
        seed=42,
        normalization='soft',
        reference_window=252,
    )
    detector.fit(X)
    logger.info(f"Fit completed in {time.time() - t0:.1f}s")

    # Compute all variants (raw values)
    logger.info("Computing all coherence variants...")
    t0 = time.time()
    raw_variants = detector.compute_all_variants(X)
    logger.info(f"All variants computed in {time.time() - t0:.1f}s")

    # Check computational vs IPR correlation
    comp = raw_variants['computational']
    ipr = raw_variants['ipr']
    # C_l1_comp = (sum|psi_i|)^2 - 1 and IPR = sum|psi_i|^4
    # For normalized states: C_l1_comp = 2*(1 - IPR) when d=2,
    # but for general d the relationship is: C_l1 = (sum|psi_i|)^2 - 1
    # while 2*(1-IPR) = 2 - 2*sum|psi_i|^4. These are NOT identical for d>2.
    # Let's compute correlation anyway.
    comp_ipr_corr = float(np.corrcoef(comp, ipr)[0, 1])
    logger.info(f"Correlation(computational_coherence, IPR) = {comp_ipr_corr:.4f}")
    logger.info(f"  (expected ~-1 if they are redundant)")

    # Z-score each variant
    T = len(X)
    variant_names = ['computational', 'reference', 'temporal']
    z_scored = {}
    for vname in variant_names:
        raw_vals = raw_variants[vname]
        skip = 1 if vname == 'temporal' else 0
        z_scored[vname] = _expanding_zscore(
            raw_vals, rolling_window=20, min_expanding=60, T=T, skip_nan_start=skip
        )

    # Also z-score IPR for comparison
    z_scored['ipr'] = _expanding_zscore(
        ipr, rolling_window=20, min_expanding=60, T=T
    )

    # Build crisis/normal masks
    import pandas as pd
    dates_series = pd.DatetimeIndex(dates)

    all_crisis_mask = np.zeros(len(dates_series), dtype=bool)
    for ck, cv in ALL_CRISES.items():
        cs = pd.Timestamp(cv['start'])
        ce = pd.Timestamp(cv['end'])
        all_crisis_mask |= (dates_series >= cs) & (dates_series <= ce)
    normal_mask = ~all_crisis_mask

    # Evaluate each variant x each crisis
    results = {vn: {} for vn in variant_names + ['ipr']}
    all_d_values = {vn: [] for vn in variant_names + ['ipr']}

    for crisis_key in SMOKE_CRISES:
        crisis_info = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis_info['start'])
        crisis_end = pd.Timestamp(crisis_info['end'])
        label = crisis_info['label']

        crisis_mask = (dates_series >= crisis_start) & (dates_series <= crisis_end)

        for vname in variant_names + ['ipr']:
            scores = z_scored[vname]
            valid_mask = ~np.isnan(scores)
            crisis_scores = np.abs(scores[crisis_mask & valid_mask])
            normal_scores = np.abs(scores[normal_mask & valid_mask])

            if len(crisis_scores) < 5:
                d_val = 0.0
                p_val = 1.0
            else:
                d_val = compute_cohens_d(crisis_scores, normal_scores)
                _, p_val = stats.ttest_ind(crisis_scores, normal_scores, equal_var=False)

            results[vname][crisis_key] = {
                'd': round(float(d_val), 4),
                'p': round(float(p_val), 6),
                'n_crisis': int(len(crisis_scores)),
                'label': label,
            }
            all_d_values[vname].append(d_val)

    # Print results table
    logger.info("")
    logger.info(f"{'Variant':<25} {'2008_gfc':>10} {'2020_covid':>12} "
                f"{'2022_rates':>12} {'2023_svb':>10} {'Median d':>10}")
    logger.info("-" * 80)

    summary = {}
    for vname in variant_names + ['ipr']:
        d_values = all_d_values[vname]
        median_d = float(np.median(d_values))
        row = f"{vname:<25}"
        for ck in SMOKE_CRISES:
            row += f" {results[vname][ck]['d']:>10.3f}"
        row += f" {median_d:>10.3f}"
        logger.info(row)

        summary[vname] = {
            'per_crisis': {ck: results[vname][ck] for ck in SMOKE_CRISES},
            'median_d': round(median_d, 4),
            'max_d': round(float(np.max(d_values)), 4),
        }

    # Cross-correlation between variants
    logger.info("")
    logger.info("Cross-correlations between variant z-scores:")
    for i, v1 in enumerate(variant_names):
        for v2 in variant_names[i + 1:]:
            s1, s2 = z_scored[v1], z_scored[v2]
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if np.sum(valid) > 10:
                corr = float(np.corrcoef(s1[valid], s2[valid])[0, 1])
                logger.info(f"  corr({v1}, {v2}) = {corr:.4f}")

    # Computational vs IPR correlation in z-scores
    s_comp, s_ipr = z_scored['computational'], z_scored['ipr']
    valid_both = ~(np.isnan(s_comp) | np.isnan(s_ipr))
    if np.sum(valid_both) > 10:
        zscore_corr = float(np.corrcoef(s_comp[valid_both], s_ipr[valid_both])[0, 1])
        logger.info(f"  corr(computational_zscore, ipr_zscore) = {zscore_corr:.4f}")

    # Determine best variant
    median_ds = {vn: summary[vn]['median_d'] for vn in variant_names}
    best_variant = max(median_ds, key=median_ds.get)
    best_median_d = median_ds[best_variant]
    passes = best_median_d > 0.3

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"BEST VARIANT: {best_variant} (median d = {best_median_d:.4f})")
    logger.info(f"PASSES THRESHOLD (median d > 0.3): {passes}")
    logger.info(f"Computational-IPR raw correlation: {comp_ipr_corr:.4f}")
    logger.info("=" * 70)

    # Raw stats for diagnostics
    raw_stats = {}
    for vname in variant_names:
        vals = raw_variants[vname]
        valid = vals[~np.isnan(vals)]
        raw_stats[vname] = {
            'mean': round(float(np.mean(valid)), 6),
            'std': round(float(np.std(valid)), 6),
            'min': round(float(np.min(valid)), 6),
            'max': round(float(np.max(valid)), 6),
        }

    # Assemble output
    output = {
        'detector': 'QuantumCoherenceDetector',
        'hilbert_dim': 8,
        'n_pca_components': 8,
        'operator_method': 'random',
        'normalization': 'soft',
        'reference_window': 252,
        'best_variant': best_variant,
        'passes_threshold': passes,
        'threshold': 0.3,
        'computational_ipr_raw_correlation': round(comp_ipr_corr, 4),
        'computational_ipr_zscore_correlation': round(
            zscore_corr if np.sum(valid_both) > 10 else 0.0, 4
        ),
        'per_variant': summary,
        'raw_stats': raw_stats,
    }

    # Save
    output_path = Path(__file__).parent / 'smoke_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return output


if __name__ == '__main__':
    run_smoke_test()
