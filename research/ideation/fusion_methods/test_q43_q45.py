"""
Smoke test for fusion method questions Q43-Q45.

Q43: Does rank aggregation (Borda count, RRF) outperform arithmetic mean fusion?
Q44: Can information-theoretic fusion (MI-weighted) improve on heuristic weighting?
Q45: Does hierarchical clustering before fusion reduce redundancy and improve d?

Tests: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
Data: SPY via yfinance
Base detectors: BerryPhaseRateDetector, SpectralGapDetector, ReducedPurityDetector,
                SpectralEntropyDetector, DimensionalityCollapseDetector
Metric: Cohen's d (crisis vs 60-day pre-crisis baseline)
Pass criterion: median_d > 0.3
"""

import json
import sys
import time
import os
import warnings
import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import (
    BerryPhaseRateDetector,
    SpectralGapDetector,
    ReducedPurityDetector,
    SpectralEntropyDetector,
    DimensionalityCollapseDetector,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)

# Configuration
SYMBOLS = ['SPY']
START_DATE = '2005-01-01'
END_DATE = '2025-12-31'
TEST_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
CONTEXT_DAYS = 60
PASS_THRESHOLD = 0.3


# =============================================================================
# Detector configuration
# =============================================================================

DETECTOR_CONFIGS = [
    {
        'name': 'BerryPhaseRate',
        'cls': BerryPhaseRateDetector,
        'kwargs': {
            'hilbert_dim': 8,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'seed': 42,
        },
    },
    {
        'name': 'SpectralGap',
        'cls': SpectralGapDetector,
        'kwargs': {
            'hilbert_dim': 8,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'seed': 42,
        },
    },
    {
        'name': 'ReducedPurity',
        'cls': ReducedPurityDetector,
        'kwargs': {
            'hilbert_dim': 8,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'seed': 42,
        },
    },
    {
        'name': 'SpectralEntropy',
        'cls': SpectralEntropyDetector,
        'kwargs': {
            'hilbert_dim': 8,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'seed': 42,
        },
    },
    {
        'name': 'DimensionalityCollapse',
        'cls': DimensionalityCollapseDetector,
        'kwargs': {
            'hilbert_dim': 8,
            'n_pca_components': 8,
            'operator_method': 'random',
            'normalization': 'soft',
            'rolling_window': 20,
            'min_expanding': 60,
            'seed': 42,
        },
    },
]


# =============================================================================
# Fusion methods
# =============================================================================

def zscore_series(scores: np.ndarray, min_periods: int = 60) -> np.ndarray:
    """Compute expanding-window z-score of a score series.

    Uses only past data at each time step (causal). Returns NaN for the
    first min_periods steps.

    Args:
        scores: 1-D array of raw scores.
        min_periods: Minimum history required before computing z-score.

    Returns:
        1-D array of z-scored values, same length as scores.
    """
    T = len(scores)
    z = np.full(T, np.nan)
    for t in range(min_periods, T):
        hist = scores[:t]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 2:
            continue
        mu = np.mean(hist)
        sigma = np.std(hist, ddof=1)
        if sigma > 1e-12 and not np.isnan(scores[t]):
            z[t] = (scores[t] - mu) / sigma
    return z


def arithmetic_mean_fusion(score_matrix: np.ndarray) -> np.ndarray:
    """Fuse score channels via arithmetic mean of z-scores.

    Args:
        score_matrix: (T, n_channels) array of per-channel scores.
            Each channel is already z-scored.

    Returns:
        1-D array of length T: mean across valid channels at each step.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(score_matrix, axis=1)


def borda_count_fusion(score_matrix: np.ndarray) -> np.ndarray:
    """Fuse score channels via binary Borda count.

    The classic Borda sum of ranks is always K*(K+1)/2 = constant when
    all K channels are valid — producing a constant series with d=0.
    To avoid this degeneracy, we use a binary Borda formulation:

        score(t) = (1/K) * sum_i I[z_i(t) > 0]

    This counts the fraction of channels in the "high regime" (z-score > 0,
    meaning above their own historical mean). When a crisis activates some
    but not all channels, this fraction rises above the ~0.5 baseline.

    This is the correct regime-detection interpretation: a channel "votes"
    for a regime change when its own standardized score is elevated.

    Args:
        score_matrix: (T, n_channels) z-scored scores.

    Returns:
        1-D array of binary Borda scores in [0, 1].
    """
    T, K = score_matrix.shape
    borda = np.full(T, np.nan)

    for t in range(T):
        row = score_matrix[t]
        valid_mask = ~np.isnan(row)
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            continue
        valid_vals = row[valid_mask]
        # Fraction of channels above zero (elevated regime signal)
        borda[t] = np.sum(valid_vals > 0) / n_valid

    return borda


def reciprocal_rank_fusion(score_matrix: np.ndarray, k: int = 60) -> np.ndarray:
    """Fuse score channels via Reciprocal Rank Fusion (RRF).

    RRF score = sum_i 1 / (k + rank_i) where rank_i is the rank of
    channel i at time t (rank 1 = highest score).

    When all K channels are always valid, sum_i 1/(k+rank_i) is constant
    since the set {rank_1,...,rank_K} = {1,...,K} always. To produce
    meaningful variation, we weight each channel's RRF contribution by
    its z-score magnitude (positive scores boost signal). Alternatively,
    we use the top-1 reciprocal rank: score = 1 / (k + rank_of_max).

    Implementation: we return the mean rank percentile of the max-scoring
    channel at each time step, which captures 'how extreme is the leading
    signal'. This avoids the constant-sum degeneracy.

    Args:
        score_matrix: (T, n_channels) z-scored scores.
        k: RRF damping constant (default 60, standard in IR literature).

    Returns:
        1-D array of RRF scores (lead-channel RRF).
    """
    T, K = score_matrix.shape
    rrf = np.full(T, np.nan)

    for t in range(T):
        row = score_matrix[t]
        valid_mask = ~np.isnan(row)
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            continue
        valid_vals = row[valid_mask]
        from scipy.stats import rankdata
        # Rank so rank 1 = highest score (best signal)
        ranks = rankdata(-valid_vals)
        # Classic RRF: weight each channel score by 1/(k+rank)
        rrf_weights = 1.0 / (k + ranks)  # shape (n_valid,)
        rrf_weights /= rrf_weights.sum()  # normalize weights
        # Weighted combination of raw z-scores (not ranks)
        rrf[t] = np.dot(rrf_weights, valid_vals)

    return rrf


def mi_weighted_fusion(
    score_matrix: np.ndarray,
    reference_signal: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """Fuse score channels weighted by mutual information with a reference signal.

    Computes MI(channel_i, reference) using histogram-based MI estimation
    on the full time series. Weights = softmax(MI_i / max(MI)).

    Args:
        score_matrix: (T, n_channels) z-scored scores.
        reference_signal: (T,) reference signal (e.g., realized vol, VIX proxy).
        n_bins: Number of histogram bins for MI estimation.

    Returns:
        1-D array of MI-weighted fused scores.
    """
    T, K = score_matrix.shape

    mi_values = np.zeros(K)
    for k in range(K):
        channel = score_matrix[:, k]
        valid = ~np.isnan(channel) & ~np.isnan(reference_signal)
        if valid.sum() < 50:
            mi_values[k] = 0.0
            continue
        x = channel[valid]
        y = reference_signal[valid]
        # Clip to ±5 sigma to avoid histogram edge effects
        x = np.clip(x, -5, 5)
        y = np.clip(y, np.nanpercentile(y, 1), np.nanpercentile(y, 99))
        # Joint histogram
        hist2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        hist2d = hist2d + 1e-10  # Laplace smoothing
        p_xy = hist2d / hist2d.sum()
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        mi = np.sum(p_xy * np.log(p_xy / (p_x * p_y)))
        mi_values[k] = max(mi, 0.0)

    logger.info(f"  MI values per channel: {mi_values.round(4)}")

    # Weights = softmax of MI values
    mi_max = mi_values.max()
    if mi_max < 1e-12:
        weights = np.ones(K) / K
    else:
        weights = np.exp(mi_values / (mi_max + 1e-12))
        weights /= weights.sum()

    logger.info(f"  MI weights: {weights.round(4)}")

    # Weighted mean across channels
    result = np.full(T, np.nan)
    for t in range(T):
        row = score_matrix[t]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        w = weights[valid]
        w /= w.sum()
        result[t] = np.dot(w, row[valid])

    return result


def hierarchical_cluster_fusion(
    score_matrix: np.ndarray,
    correlation_threshold: float = 0.7,
) -> np.ndarray:
    """Fuse score channels via hierarchical clustering + within-cluster averaging.

    1. Compute correlation matrix of channels using their full score time series.
    2. Convert to distance: d = 1 - |corr|.
    3. Hierarchical clustering (Ward linkage). Cut at correlation_threshold to
       form clusters.
    4. Within each cluster: average z-scores (reduces redundancy).
    5. Across clusters: average cluster scores (equal weight).

    Args:
        score_matrix: (T, n_channels) z-scored scores.
        correlation_threshold: Channels with |corr| > threshold are grouped.
            (Uses distance threshold = 1 - correlation_threshold.)

    Returns:
        1-D array of hierarchically-fused scores.
    """
    T, K = score_matrix.shape

    # Compute correlation on rows where all channels have valid values
    valid_rows = ~np.any(np.isnan(score_matrix), axis=1)
    if valid_rows.sum() < 50:
        logger.warning("  Not enough valid rows for correlation; falling back to mean fusion.")
        return arithmetic_mean_fusion(score_matrix)

    X_valid = score_matrix[valid_rows]
    # Correlation matrix (K x K)
    corr = np.corrcoef(X_valid.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)

    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method='ward')

    # Cut at (1 - correlation_threshold) so channels with |corr| > threshold cluster together
    cut_distance = 1.0 - correlation_threshold
    labels = fcluster(Z, t=cut_distance, criterion='distance')

    n_clusters = labels.max()
    logger.info(f"  Hierarchical clustering: {K} channels -> {n_clusters} clusters (labels={labels})")

    # Within-cluster mean, then cross-cluster mean
    result = np.full(T, np.nan)
    cluster_scores = np.full((T, n_clusters), np.nan)

    for c in range(1, n_clusters + 1):
        member_mask = labels == c
        cluster_mat = score_matrix[:, member_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cluster_scores[:, c - 1] = np.nanmean(cluster_mat, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(cluster_scores, axis=1)

    return result


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_fusion_method(
    fused_scores: np.ndarray,
    dates: pd.DatetimeIndex,
    method_name: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Compute per-crisis Cohen's d for a fused score series.

    Args:
        fused_scores: 1-D array of fused scores, aligned to dates.
        dates: DatetimeIndex aligned with fused_scores.
        method_name: Human-readable name for logging.
        n_bootstrap: Bootstrap resamples for CI.

    Returns:
        dict: per_crisis results + summary statistics.
    """
    results = {}
    for crisis_key in TEST_CRISES:
        crisis = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis['start'])
        crisis_end = pd.Timestamp(crisis['end'])
        normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

        crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
        normal_mask = (dates >= normal_start) & (dates < crisis_start)

        crisis_scores = fused_scores[np.asarray(crisis_mask)]
        normal_scores = fused_scores[np.asarray(normal_mask)]

        crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
        normal_valid = normal_scores[~np.isnan(normal_scores)]

        if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_valid, normal_valid,
                n_bootstrap=n_bootstrap, seed=42,
            )
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan

        results[crisis_key] = {
            'cohens_d': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'crisis_n': int(len(crisis_valid)),
            'normal_n': int(len(normal_valid)),
        }

        d_str = f"{d:.3f}" if not np.isnan(d) else "NaN"
        logger.info(f"    {crisis_key}: d={d_str}")

    d_values = [r['cohens_d'] for r in results.values() if r['cohens_d'] is not None]
    median_d = float(np.median(d_values)) if d_values else None
    max_d = float(np.max(d_values)) if d_values else None

    return {
        'per_crisis': results,
        'median_d': median_d,
        'max_d': max_d,
        'd_values': d_values,
        'passes_threshold': median_d is not None and median_d > PASS_THRESHOLD,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Fetch data (SPY only)
    # -------------------------------------------------------------------------
    logger.info(f"Fetching {SYMBOLS} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = raw_df['close'].unstack('symbol').dropna()
    logger.info(f"Close prices: {close_prices.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Build feature matrix (single-asset)
    # -------------------------------------------------------------------------
    logger.info("Building feature matrix...")
    from experiments.data_loader import create_feature_matrix_single_asset
    features, dates = create_feature_matrix_single_asset(close_prices['SPY'])
    dates = pd.DatetimeIndex(dates)
    T = len(features)
    logger.info(f"Feature matrix: {features.shape}, {dates[0]} to {dates[-1]}")

    # Reference signal for Q44: realized volatility (20-day rolling std of log-returns)
    log_ret = np.log(close_prices['SPY'] / close_prices['SPY'].shift(1)).dropna()
    realized_vol = log_ret.rolling(20).std().reindex(dates).values
    realized_vol = realized_vol / (np.nanstd(realized_vol) + 1e-12)  # normalize

    # -------------------------------------------------------------------------
    # Step 3: Fit all base detectors and collect raw scores
    # -------------------------------------------------------------------------
    logger.info("Fitting base detectors...")
    raw_scores = {}

    for cfg in DETECTOR_CONFIGS:
        name = cfg['name']
        logger.info(f"  Fitting {name}...")
        t_det = time.time()
        det = cfg['cls'](**cfg['kwargs'])
        det.fit(features)
        scores = det.compute_regime_scores(features)
        raw_scores[name] = scores
        dt = time.time() - t_det
        n_valid = int(np.sum(~np.isnan(scores)))
        logger.info(f"    Done in {dt:.1f}s, {n_valid}/{T} valid")

    # Stack into score matrix (T, K)
    names = list(raw_scores.keys())
    K = len(names)
    score_matrix = np.column_stack([raw_scores[n] for n in names])  # (T, K)
    logger.info(f"Score matrix shape: {score_matrix.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Evaluate individual detectors (baseline)
    # -------------------------------------------------------------------------
    logger.info("\n=== Individual Detector Baselines ===")
    individual_results = {}
    for name in names:
        logger.info(f"  Evaluating {name}...")
        res = evaluate_fusion_method(raw_scores[name], dates, name)
        individual_results[name] = res
        logger.info(f"    median_d={res['median_d']:.3f}")

    # -------------------------------------------------------------------------
    # Q43: Rank Aggregation vs Arithmetic Mean
    # -------------------------------------------------------------------------
    logger.info("\n=== Q43: Rank Aggregation vs Arithmetic Mean ===")

    logger.info("  Computing arithmetic mean fusion...")
    arith_scores = arithmetic_mean_fusion(score_matrix)
    q43_arith = evaluate_fusion_method(arith_scores, dates, 'ArithmeticMean')
    logger.info(f"  Arithmetic Mean median_d={q43_arith['median_d']:.3f}")

    logger.info("  Computing Borda count fusion...")
    borda_scores = borda_count_fusion(score_matrix)
    q43_borda = evaluate_fusion_method(borda_scores, dates, 'BordaCount')
    logger.info(f"  Borda Count median_d={q43_borda['median_d']:.3f}")

    logger.info("  Computing Reciprocal Rank Fusion (k=60)...")
    rrf_scores = reciprocal_rank_fusion(score_matrix, k=60)
    q43_rrf = evaluate_fusion_method(rrf_scores, dates, 'RRF_k60')
    logger.info(f"  RRF (k=60) median_d={q43_rrf['median_d']:.3f}")

    logger.info("  Computing Reciprocal Rank Fusion (k=10)...")
    rrf10_scores = reciprocal_rank_fusion(score_matrix, k=10)
    q43_rrf10 = evaluate_fusion_method(rrf10_scores, dates, 'RRF_k10')
    logger.info(f"  RRF (k=10) median_d={q43_rrf10['median_d']:.3f}")

    # -------------------------------------------------------------------------
    # Q44: Information-Theoretic (MI-Weighted) Fusion
    # -------------------------------------------------------------------------
    logger.info("\n=== Q44: MI-Weighted Fusion ===")

    logger.info("  Computing MI with realized volatility...")
    mi_scores = mi_weighted_fusion(score_matrix, realized_vol, n_bins=20)
    q44_mi = evaluate_fusion_method(mi_scores, dates, 'MI_Weighted')
    logger.info(f"  MI-Weighted median_d={q44_mi['median_d']:.3f}")

    # -------------------------------------------------------------------------
    # Q45: Hierarchical Clustering Fusion
    # -------------------------------------------------------------------------
    logger.info("\n=== Q45: Hierarchical Clustering Fusion ===")

    for threshold in [0.5, 0.7, 0.9]:
        logger.info(f"  Hierarchical clustering (corr_threshold={threshold})...")
        hclust_scores = hierarchical_cluster_fusion(score_matrix, threshold)
        q45 = evaluate_fusion_method(hclust_scores, dates, f'HClust_{threshold}')
        logger.info(f"  HClust (threshold={threshold}) median_d={q45['median_d']:.3f}")

    # Re-run for storage with all three thresholds
    hclust_50 = evaluate_fusion_method(
        hierarchical_cluster_fusion(score_matrix, 0.5), dates, 'HClust_0.5'
    )
    hclust_70 = evaluate_fusion_method(
        hierarchical_cluster_fusion(score_matrix, 0.7), dates, 'HClust_0.7'
    )
    hclust_90 = evaluate_fusion_method(
        hierarchical_cluster_fusion(score_matrix, 0.9), dates, 'HClust_0.9'
    )

    total_time = time.time() - t0

    # -------------------------------------------------------------------------
    # Compile and save results
    # -------------------------------------------------------------------------
    results = {
        'meta': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'test_crises': TEST_CRISES,
            'base_detectors': names,
            'n_bootstrap': N_BOOTSTRAP,
            'context_days': CONTEXT_DAYS,
            'pass_threshold': PASS_THRESHOLD,
            'total_seconds': round(total_time, 1),
        },
        'individual_baselines': {
            n: {
                'median_d': r['median_d'],
                'max_d': r['max_d'],
                'd_values': r['d_values'],
                'passes_threshold': r['passes_threshold'],
                'per_crisis': r['per_crisis'],
            }
            for n, r in individual_results.items()
        },
        'Q43_rank_aggregation': {
            'ArithmeticMean': {
                'median_d': q43_arith['median_d'],
                'max_d': q43_arith['max_d'],
                'd_values': q43_arith['d_values'],
                'passes': q43_arith['passes_threshold'],
                'per_crisis': q43_arith['per_crisis'],
            },
            'BordaCount': {
                'median_d': q43_borda['median_d'],
                'max_d': q43_borda['max_d'],
                'd_values': q43_borda['d_values'],
                'passes': q43_borda['passes_threshold'],
                'per_crisis': q43_borda['per_crisis'],
            },
            'RRF_k60': {
                'median_d': q43_rrf['median_d'],
                'max_d': q43_rrf['max_d'],
                'd_values': q43_rrf['d_values'],
                'passes': q43_rrf['passes_threshold'],
                'per_crisis': q43_rrf['per_crisis'],
            },
            'RRF_k10': {
                'median_d': q43_rrf10['median_d'],
                'max_d': q43_rrf10['max_d'],
                'd_values': q43_rrf10['d_values'],
                'passes': q43_rrf10['passes_threshold'],
                'per_crisis': q43_rrf10['per_crisis'],
            },
        },
        'Q44_mi_weighted': {
            'MI_Weighted': {
                'median_d': q44_mi['median_d'],
                'max_d': q44_mi['max_d'],
                'd_values': q44_mi['d_values'],
                'passes': q44_mi['passes_threshold'],
                'per_crisis': q44_mi['per_crisis'],
            },
        },
        'Q45_hierarchical_clustering': {
            'HClust_0.5': {
                'median_d': hclust_50['median_d'],
                'max_d': hclust_50['max_d'],
                'd_values': hclust_50['d_values'],
                'passes': hclust_50['passes_threshold'],
                'per_crisis': hclust_50['per_crisis'],
            },
            'HClust_0.7': {
                'median_d': hclust_70['median_d'],
                'max_d': hclust_70['max_d'],
                'd_values': hclust_70['d_values'],
                'passes': hclust_70['passes_threshold'],
                'per_crisis': hclust_70['per_crisis'],
            },
            'HClust_0.9': {
                'median_d': hclust_90['median_d'],
                'max_d': hclust_90['max_d'],
                'd_values': hclust_90['d_values'],
                'passes': hclust_90['passes_threshold'],
                'per_crisis': hclust_90['per_crisis'],
            },
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), 'smoke_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q43-Q45 FUSION METHOD SMOKE TEST RESULTS")
    print("=" * 70)

    print("\n--- Individual Baselines ---")
    for n, r in individual_results.items():
        print(f"  {n:30s}  median_d={r['median_d']:.3f}  max_d={r['max_d']:.3f}")

    print("\n--- Q43: Rank Aggregation vs Arithmetic Mean ---")
    for label, res in [
        ('ArithmeticMean', q43_arith),
        ('BordaCount', q43_borda),
        ('RRF (k=60)', q43_rrf),
        ('RRF (k=10)', q43_rrf10),
    ]:
        passes = "PASS" if res['passes_threshold'] else "FAIL"
        print(f"  {label:20s}  median_d={res['median_d']:.3f}  "
              f"max_d={res['max_d']:.3f}  [{passes}]")
        for crisis_key, r in res['per_crisis'].items():
            d_str = f"{r['cohens_d']:.3f}" if r['cohens_d'] is not None else "NaN"
            print(f"    {crisis_key:20s} d={d_str}")

    print("\n--- Q44: MI-Weighted Fusion ---")
    for label, res in [('MI_Weighted', q44_mi)]:
        passes = "PASS" if res['passes_threshold'] else "FAIL"
        print(f"  {label:20s}  median_d={res['median_d']:.3f}  "
              f"max_d={res['max_d']:.3f}  [{passes}]")
        for crisis_key, r in res['per_crisis'].items():
            d_str = f"{r['cohens_d']:.3f}" if r['cohens_d'] is not None else "NaN"
            print(f"    {crisis_key:20s} d={d_str}")

    print("\n--- Q45: Hierarchical Clustering Fusion ---")
    for label, res in [
        ('HClust (thr=0.5)', hclust_50),
        ('HClust (thr=0.7)', hclust_70),
        ('HClust (thr=0.9)', hclust_90),
    ]:
        passes = "PASS" if res['passes_threshold'] else "FAIL"
        print(f"  {label:22s}  median_d={res['median_d']:.3f}  "
              f"max_d={res['max_d']:.3f}  [{passes}]")
        for crisis_key, r in res['per_crisis'].items():
            d_str = f"{r['cohens_d']:.3f}" if r['cohens_d'] is not None else "NaN"
            print(f"    {crisis_key:20s} d={d_str}")

    print(f"\n  Total runtime: {total_time:.1f}s")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
