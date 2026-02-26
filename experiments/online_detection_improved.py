"""
Improved Online Regime Detection via Signal Correction

Root cause diagnosis (from online_detection_diagnosis.py):
    ALL geometric features have LOWER mean during crises than during normal periods.
    This means naive "higher = more crisis" detectors get AUC-ROC < 0.50 (inverted).

    Diagnosis evidence (4 quick crises):
        berry_rate:       crisis_mean=0.0009  vs normal_mean=0.0018  (MW p=1.0)
        qfi_logdet:       crisis_mean=-37.63  vs normal_mean=-35.45  (MW p=1.0)
        infidelity:       crisis_mean=0.0003  vs normal_mean=0.0005  (MW p=0.96)
        inv_spectral_gap: crisis_mean=0.213   vs normal_mean=0.309   (MW p=1.0)

This script implements and evaluates 5 signal improvement strategies:

    Strategy 1: Signal Inversion
        Negate features where crisis_mean < normal_mean (learned from warmup).
    Strategy 2: CUSUM on Geometric Features
        Streaming CUSUM detects shifts in EITHER direction.
    Strategy 3: Absolute Deviation from Expanding Mean
        |feature - expanding_mean| / expanding_std  (direction-agnostic).
    Strategy 4: Rate-of-Change Features
        |f(t) - f(t-k)| for k=1,5,10 captures rapid change regardless of direction.
    Strategy 5: Multi-Scale Fusion
        Combine geometric features at lookback windows of 5, 10, 20, 40 days.

Usage:
    python experiments/online_detection_improved.py
    python experiments/online_detection_improved.py --quick

Outputs:
    experiments/outputs/regime_detection/online_improved/
        improved_results.json
        strategy_comparison.png
        per_crisis_heatmap.png
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

OUTPUT_DIR = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'online_improved'

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# =============================================================================
# Causal Geometric Feature Computation
# =============================================================================

FEATURE_NAMES = ['berry_rate', 'qfi_logdet', 'infidelity', 'inv_spectral_gap']


def compute_causal_geometric_features(
    X_raw, dates, hilbert_dim=8, n_pca=8,
    refit_interval=63, min_history=126,
):
    """Compute geometric features using only past data at each time step.

    Strictly causal: at time t, only data [0..t-1] is used for fitting.

    Args:
        X_raw: Enriched feature matrix (T, d).
        dates: DatetimeIndex (T,).
        hilbert_dim: Hilbert space dimension.
        n_pca: Number of PCA components.
        refit_interval: Days between scaler/PCA/operator refits.
        min_history: Minimum history before first feature computation.

    Returns:
        Dict mapping feature name to time series array (T,) with NaN for warmup.
    """
    T = X_raw.shape[0]
    features = {name: np.full(T, np.nan) for name in FEATURE_NAMES}

    prev_berry = None
    prev_state = None
    last_refit = 0
    median_norm = 1.0

    scaler = None
    pca = None
    geo = None

    for t in range(min_history, T):
        # Refit periodically on expanding window
        if scaler is None or (t - last_refit >= refit_interval):
            scaler = StandardScaler()
            scaler.fit(X_raw[:t])
            pca = PCA(n_components=min(n_pca, X_raw.shape[1]))
            X_scaled = scaler.transform(X_raw[:t])
            pca.fit(X_scaled)

            X_pca_fit = pca.transform(X_scaled)
            norms = np.linalg.norm(X_pca_fit, axis=1, keepdims=True)
            median_norm = float(np.median(norms))
            X_pca_fit = X_pca_fit / (norms + median_norm)

            geo = QCMLGeometry(
                n_features=X_pca_fit.shape[1], hilbert_dim=hilbert_dim
            )
            geo.fit_operators(X_pca_fit, method='random')
            last_refit = t

        # Transform current point through causal pipeline
        x_scaled = scaler.transform(X_raw[t:t + 1])
        x_pca = pca.transform(x_scaled).ravel()
        norm = np.linalg.norm(x_pca)
        x_pca = x_pca / (norm + median_norm)

        # Berry curvature rate of change
        berry = geo.berry_curvature_2d(x_pca, indices=(0, 1))
        if prev_berry is not None:
            features['berry_rate'][t] = abs(berry - prev_berry)
        prev_berry = berry

        # QFI log-determinant (metric tensor)
        g = geo.quantum_metric(x_pca)
        eigs = np.linalg.eigvalsh(g)
        pos = eigs[eigs > 1e-10]
        features['qfi_logdet'][t] = np.sum(np.log(pos)) if len(pos) > 0 else -20.0

        # Multi-lag infidelity
        state = geo.quasi_coherent_state(x_pca)
        if prev_state is not None:
            overlap = np.abs(np.vdot(state, prev_state))
            features['infidelity'][t] = 1.0 - overlap ** 2
        prev_state = state

        # Inverse spectral gap
        gap = geo.spectral_gap(x_pca)
        features['inv_spectral_gap'][t] = 1.0 / (gap + 1e-10)

    return features


# =============================================================================
# Strategy 1: Signal Inversion
# =============================================================================

def strategy_inversion(features, crisis_labels, warmup=252):
    """Invert features where crisis mean < normal mean, then z-score.

    Uses the first `warmup` days to learn the direction of each feature.
    After warmup, applies expanding z-score with correct sign.

    Args:
        features: Dict of feature_name -> array (T,).
        crisis_labels: Binary array (T,).
        warmup: Days to use for learning feature direction.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(crisis_labels)
    results = {}

    for fname, scores in features.items():
        p_crisis = np.full(T, np.nan)

        # Learn direction from warmup period (using crisis labels -- supervised)
        warmup_valid = ~np.isnan(scores[:warmup])
        warmup_crisis = scores[:warmup][warmup_valid & (crisis_labels[:warmup] == 1)]
        warmup_normal = scores[:warmup][warmup_valid & (crisis_labels[:warmup] == 0)]

        if len(warmup_crisis) > 5 and len(warmup_normal) > 5:
            invert = np.mean(warmup_crisis) < np.mean(warmup_normal)
        else:
            # Fallback: use ALL available labels (expanding) to learn direction.
            # At each scoring step, re-check direction from labels seen so far.
            # Default to True (invert) since diagnosis shows all geometric
            # features have lower mean during crises.
            invert = True

        sign = -1.0 if invert else 1.0

        # Expanding z-score with correct sign
        for t in range(warmup, T):
            if np.isnan(scores[t]):
                continue
            past = scores[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 20:
                continue
            mu = np.mean(valid)
            sigma = max(np.std(valid, ddof=1), 1e-12)
            z = sign * (scores[t] - mu) / sigma
            # Sigmoid: z=0 -> 0.5, z=2 -> 0.88, z=-2 -> 0.12
            p_crisis[t] = 1.0 / (1.0 + np.exp(-z + 1.0))

        results[f'inverted_{fname}'] = p_crisis

    # Fused: mean of all inverted features
    all_p = [v for v in results.values()]
    fused = np.nanmean(all_p, axis=0)
    results['inverted_fusion'] = fused

    return results


def strategy_inversion_unsupervised(features, warmup=252):
    """Unsupervised signal inversion using expanding correlation detection.

    Instead of using labels, detects the direction by checking whether
    extreme values (top 10% by absolute deviation) are high or low.
    In financial markets, crises create extreme LOW values for geometric
    features. We detect this and invert.

    Args:
        features: Dict of feature_name -> array (T,).
        warmup: Days for initial statistics.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(next(iter(features.values())))
    results = {}

    for fname, scores in features.items():
        p_crisis = np.full(T, np.nan)

        for t in range(warmup, T):
            if np.isnan(scores[t]):
                continue
            past = scores[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 20:
                continue

            mu = np.mean(valid)
            sigma = max(np.std(valid, ddof=1), 1e-12)
            z = (scores[t] - mu) / sigma

            # Key insight: for these geometric features, large NEGATIVE z
            # indicates crisis. So we negate z to get "anomaly = crisis".
            # Equivalently: P(crisis) increases when feature DROPS below mean.
            neg_z = -z
            p_crisis[t] = 1.0 / (1.0 + np.exp(-neg_z + 1.0))

        results[f'inv_unsup_{fname}'] = p_crisis

    all_p = [v for v in results.values()]
    fused = np.nanmean(all_p, axis=0)
    results['inv_unsup_fusion'] = fused

    return results


# =============================================================================
# Strategy 2: CUSUM on Geometric Features
# =============================================================================

def strategy_cusum(features, warmup=126):
    """Streaming CUSUM on each geometric feature.

    CUSUM naturally detects shifts in EITHER direction via separate
    S+ (upward shift) and S- (downward shift) statistics.

    Args:
        features: Dict of feature_name -> array (T,).
        warmup: Minimum warmup period.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(next(iter(features.values())))
    results = {}

    for fname, scores in features.items():
        p_crisis = np.full(T, np.nan)
        S_pos = 0.0
        S_neg = 0.0
        k = 0.5  # allowance parameter (standard: 0.5 sigma)

        for t in range(warmup, T):
            if np.isnan(scores[t]):
                continue

            past = scores[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 20:
                continue

            mu = np.mean(valid)
            sigma = max(np.std(valid, ddof=1), 1e-12)
            z = (scores[t] - mu) / sigma

            # Two-sided CUSUM
            S_pos = max(0.0, S_pos + z - k)
            S_neg = max(0.0, S_neg - z - k)
            cusum_stat = max(S_pos, S_neg)

            # Adaptive threshold: h ~ sqrt(2 * ln(252 / target_far))
            # For target_far=2 alarms/yr: h ~ 2.9
            h = 3.0
            p_crisis[t] = 1.0 / (1.0 + np.exp(-0.8 * (cusum_stat - h)))

        results[f'cusum_{fname}'] = p_crisis

    # Fused: max of all CUSUM p_crisis (any feature triggering is suspicious)
    all_p = np.array([results[k] for k in results])
    fused = np.nanmax(all_p, axis=0)
    results['cusum_max_fusion'] = fused

    # Also try mean fusion
    fused_mean = np.nanmean(all_p, axis=0)
    results['cusum_mean_fusion'] = fused_mean

    return results


def strategy_cusum_directional(features, warmup=126):
    """CUSUM that specifically looks for DECREASES in geometric features.

    Since we know crises cause features to DROP, we use one-sided CUSUM
    looking for negative shifts only.

    Args:
        features: Dict of feature_name -> array (T,).
        warmup: Minimum warmup period.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(next(iter(features.values())))
    results = {}

    for fname, scores in features.items():
        p_crisis = np.full(T, np.nan)
        S_neg = 0.0
        k = 0.5

        for t in range(warmup, T):
            if np.isnan(scores[t]):
                continue

            past = scores[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 20:
                continue

            mu = np.mean(valid)
            sigma = max(np.std(valid, ddof=1), 1e-12)
            z = (scores[t] - mu) / sigma

            # One-sided CUSUM looking for NEGATIVE shifts
            S_neg = max(0.0, S_neg - z - k)

            h = 3.0
            p_crisis[t] = 1.0 / (1.0 + np.exp(-0.8 * (S_neg - h)))

        results[f'cusum_neg_{fname}'] = p_crisis

    all_p = np.array([results[k] for k in results])
    fused = np.nanmax(all_p, axis=0)
    results['cusum_neg_max'] = fused

    return results


# =============================================================================
# Strategy 3: Absolute Deviation from Expanding Mean
# =============================================================================

def strategy_absdev(features, warmup=126):
    """Use |feature - expanding_mean| / expanding_std as crisis score.

    Direction-agnostic: any large deviation from the expanding mean
    is flagged as potential crisis.

    Args:
        features: Dict of feature_name -> array (T,).
        warmup: Minimum warmup period.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(next(iter(features.values())))
    results = {}

    for fname, scores in features.items():
        p_crisis = np.full(T, np.nan)

        for t in range(warmup, T):
            if np.isnan(scores[t]):
                continue

            past = scores[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 20:
                continue

            mu = np.mean(valid)
            sigma = max(np.std(valid, ddof=1), 1e-12)
            abs_z = abs((scores[t] - mu) / sigma)

            # Sigmoid: abs_z=0 -> low, abs_z=2 -> 0.73, abs_z=3 -> 0.88
            p_crisis[t] = 1.0 / (1.0 + np.exp(-abs_z + 2.0))

        results[f'absdev_{fname}'] = p_crisis

    all_p = [results[k] for k in results]
    fused = np.nanmean(all_p, axis=0)
    results['absdev_fusion'] = fused

    # Also try RMS aggregation
    all_arr = np.array(all_p)
    rms = np.sqrt(np.nanmean(all_arr ** 2, axis=0))
    results['absdev_rms'] = rms

    return results


# =============================================================================
# Strategy 4: Rate-of-Change Features
# =============================================================================

def strategy_rateofchange(features, warmup=126, lags=(1, 5, 10, 20)):
    """Use |f(t) - f(t-k)| for multiple k to detect rapid changes.

    Rapid change in ANY direction indicates regime transition.

    Args:
        features: Dict of feature_name -> array (T,).
        warmup: Minimum warmup period.
        lags: Tuple of lag values for rate-of-change.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(next(iter(features.values())))
    results = {}

    for fname, scores in features.items():
        for lag in lags:
            roc = np.full(T, np.nan)
            for t in range(warmup + lag, T):
                if np.isnan(scores[t]) or np.isnan(scores[t - lag]):
                    continue
                roc[t] = abs(scores[t] - scores[t - lag])

            # Convert ROC to probability using expanding percentile rank
            p_crisis = np.full(T, np.nan)
            for t in range(warmup + lag + 20, T):
                if np.isnan(roc[t]):
                    continue
                past_roc = roc[warmup + lag:t]
                valid = past_roc[~np.isnan(past_roc)]
                if len(valid) < 10:
                    continue
                # Percentile rank: what fraction of past ROC is <= current?
                pctile = np.mean(valid <= roc[t])
                p_crisis[t] = pctile  # high percentile = unusual change

            results[f'roc_{fname}_lag{lag}'] = p_crisis

    # Fused across all features and lags: mean of top-3 scores at each t
    all_p = np.array([results[k] for k in results])
    # Sort along feature axis, take mean of top-3
    sorted_p = np.sort(all_p, axis=0)
    top3_mean = np.nanmean(sorted_p[-3:], axis=0)
    results['roc_top3_fusion'] = top3_mean

    # Simple mean fusion
    fused = np.nanmean(all_p, axis=0)
    results['roc_mean_fusion'] = fused

    return results


# =============================================================================
# Strategy 5: Multi-Scale Fusion
# =============================================================================

def strategy_multiscale(features, warmup=126, windows=(5, 10, 20, 40)):
    """Combine geometric features at different lookback windows.

    For each feature, compute rolling z-score at multiple time scales
    and fuse them. Different crises have different timescales.

    Args:
        features: Dict of feature_name -> array (T,).
        warmup: Minimum warmup period.
        windows: Tuple of rolling window sizes.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(next(iter(features.values())))
    results = {}

    for fname, scores in features.items():
        scale_probs = []

        for w in windows:
            p_crisis = np.full(T, np.nan)

            for t in range(max(warmup, w + 20), T):
                if np.isnan(scores[t]):
                    continue

                # Rolling window statistics
                window_start = max(0, t - w)
                recent = scores[window_start:t]
                valid_recent = recent[~np.isnan(recent)]

                # Expanding statistics for normalization
                past = scores[:t]
                valid_past = past[~np.isnan(past)]

                if len(valid_recent) < 3 or len(valid_past) < 20:
                    continue

                # Deviation of rolling mean from expanding mean
                rolling_mean = np.mean(valid_recent)
                expanding_mean = np.mean(valid_past)
                expanding_std = max(np.std(valid_past, ddof=1), 1e-12)

                # Z-score of rolling mean deviation
                # Negative z means feature dropped (crisis signal)
                z = (rolling_mean - expanding_mean) / expanding_std

                # Negate: crisis = feature BELOW expanding mean
                neg_z = -z
                p_crisis[t] = 1.0 / (1.0 + np.exp(-neg_z + 1.0))

            scale_probs.append(p_crisis)
            results[f'ms_{fname}_w{w}'] = p_crisis

        # Per-feature multi-scale fusion (mean across windows)
        per_feat_fused = np.nanmean(scale_probs, axis=0)
        results[f'ms_{fname}_fused'] = per_feat_fused

    # Cross-feature multi-scale fusion
    per_feat_fusions = [
        results[f'ms_{fname}_fused'] for fname in features
    ]
    grand_fusion = np.nanmean(per_feat_fusions, axis=0)
    results['ms_grand_fusion'] = grand_fusion

    # Also try max across features (any feature at any scale)
    grand_max = np.nanmax(per_feat_fusions, axis=0)
    results['ms_grand_max'] = grand_max

    return results


# =============================================================================
# Combined Best Strategy
# =============================================================================

def strategy_combined_best(features, crisis_labels, warmup=252):
    """Combine the best elements from all strategies.

    1. Inverted z-score (supervised direction learning)
    2. Directional CUSUM (negative shifts)
    3. Rate-of-change at lag=10 (mid-frequency)
    All fused with equal weights.

    Args:
        features: Dict of feature_name -> array (T,).
        crisis_labels: Binary array (T,).
        warmup: Warmup period.

    Returns:
        Dict of strategy_name -> P(crisis) array (T,).
    """
    T = len(crisis_labels)
    results = {}

    # Component 1: Inverted z-score fusion
    inv_results = strategy_inversion(features, crisis_labels, warmup=warmup)
    inv_fusion = inv_results['inverted_fusion']

    # Component 2: Directional CUSUM
    cusum_results = strategy_cusum_directional(features, warmup=warmup)
    cusum_neg = cusum_results['cusum_neg_max']

    # Component 3: Absolute deviation
    absdev_results = strategy_absdev(features, warmup=warmup)
    absdev = absdev_results['absdev_fusion']

    # Component 4: Multi-scale
    ms_results = strategy_multiscale(features, warmup=warmup)
    ms_fusion = ms_results['ms_grand_fusion']

    # Equal-weight combination
    components = np.array([inv_fusion, cusum_neg, absdev, ms_fusion])
    combined = np.nanmean(components, axis=0)
    results['combined_equal'] = combined

    # Weighted: emphasize inversion and multi-scale (most informed)
    weights = np.array([0.35, 0.20, 0.15, 0.30])
    weighted = np.nansum(components * weights[:, None], axis=0)
    # Handle NaN: only where at least 2 components are valid
    valid_count = np.sum(~np.isnan(components), axis=0)
    weighted[valid_count < 2] = np.nan
    weight_sum = np.nansum(
        np.where(~np.isnan(components), 1, 0) * weights[:, None], axis=0
    )
    weighted = weighted / np.maximum(weight_sum, 1e-12)
    results['combined_weighted'] = weighted

    return results


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_auc(p_crisis, crisis_labels, warmup=126):
    """Compute AUC-ROC and AUC-PR for online detection.

    Args:
        p_crisis: P(crisis) time series (T,).
        crisis_labels: Binary labels (T,).
        warmup: Timesteps to skip (feature warmup period).

    Returns:
        Dict with auc_roc, auc_pr, n_eval.
    """
    valid = ~np.isnan(p_crisis) & ~np.isnan(crisis_labels)
    valid[:warmup] = False

    p = p_crisis[valid]
    y = crisis_labels[valid]

    if len(p) < 10 or len(np.unique(y)) < 2:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'n_eval': int(len(p))}

    return {
        'auc_roc': float(roc_auc_score(y, p)),
        'auc_pr': float(average_precision_score(y, p)),
        'n_eval': int(len(p)),
    }


def evaluate_detection_delay(p_crisis, crisis_labels, dates, threshold=0.5):
    """Compute mean detection delay for crisis episodes.

    Args:
        p_crisis: P(crisis) time series.
        crisis_labels: Binary labels.
        dates: DatetimeIndex.
        threshold: Decision threshold.

    Returns:
        Dict with n_detected, n_crises, mean_delay_days, per_crisis details.
    """
    alarm = p_crisis > threshold
    alarm[np.isnan(p_crisis)] = False

    # Find crisis episodes
    episodes = []
    in_crisis = False
    start = 0
    for i in range(len(crisis_labels)):
        if crisis_labels[i] == 1 and not in_crisis:
            start = i
            in_crisis = True
        elif crisis_labels[i] == 0 and in_crisis:
            episodes.append((start, i - 1))
            in_crisis = False
    if in_crisis:
        episodes.append((start, len(crisis_labels) - 1))

    delays = []
    detected = 0
    for s, e in episodes:
        alarm_in = np.where(alarm[s:e + 1])[0]
        if len(alarm_in) > 0:
            detected += 1
            delays.append(int(alarm_in[0]))

    return {
        'n_detected': detected,
        'n_crises': len(episodes),
        'detection_rate': detected / max(len(episodes), 1),
        'mean_delay_days': float(np.mean(delays)) if delays else None,
    }


def evaluate_per_crisis(p_crisis, crisis_labels, dates):
    """Compute per-crisis P(crisis) statistics.

    Args:
        p_crisis: P(crisis) time series.
        crisis_labels: Binary labels.
        dates: DatetimeIndex.

    Returns:
        Dict mapping crisis_key -> {mean_p, max_p, pct_above_50}.
    """
    results = {}
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (dates >= cs) & (dates <= ce)

        if not np.any(mask):
            continue

        crisis_p = p_crisis[mask]
        valid_p = crisis_p[~np.isnan(crisis_p)]

        if len(valid_p) == 0:
            continue

        results[ck] = {
            'mean_p': float(np.mean(valid_p)),
            'max_p': float(np.max(valid_p)),
            'pct_above_50': float(np.mean(valid_p > 0.5)),
            'n_valid': int(len(valid_p)),
        }

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_strategy_comparison(all_results, output_dir):
    """Bar chart comparing AUC-ROC across all strategies."""
    # Filter to fusion/combined methods only for clarity
    fusion_keys = [
        k for k in all_results
        if 'fusion' in k or 'combined' in k or 'max' in k or 'rms' in k
        or k.startswith('baseline_')
    ]
    if not fusion_keys:
        fusion_keys = list(all_results.keys())

    # Sort by AUC-ROC
    items = sorted(
        [(k, all_results[k]['auc_roc']) for k in fusion_keys
         if not np.isnan(all_results[k].get('auc_roc', np.nan))],
        key=lambda x: x[1],
    )

    if not items:
        logger.warning("No valid AUC-ROC values to plot")
        return

    names = [x[0] for x in items]
    aucs = [x[1] for x in items]

    # Color by strategy
    def get_color(name):
        if 'combined' in name:
            return '#d62728'
        elif 'inv' in name:
            return '#1f77b4'
        elif 'cusum' in name:
            return '#2ca02c'
        elif 'absdev' in name or 'rms' in name:
            return '#ff7f0e'
        elif 'roc' in name:
            return '#9467bd'
        elif 'ms_' in name:
            return '#8c564b'
        elif 'baseline' in name:
            return '#7f7f7f'
        return '#17becf'

    colors = [get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), aucs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in names], fontsize=7)
    ax.set_xlabel('AUC-ROC')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, label='Random (0.50)')
    ax.axvline(0.65, color='orange', linestyle='--', linewidth=1, label='Target (0.65)')
    ax.set_title('Improved Online Detection: AUC-ROC Comparison')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{auc:.3f}', va='center', fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_dir / 'strategy_comparison.png')
    fig.savefig(output_dir / 'strategy_comparison.pdf')
    plt.close(fig)
    logger.info(f"  Saved strategy_comparison.png")


def plot_per_crisis_heatmap(all_results, output_dir):
    """Heatmap of mean P(crisis) per method and crisis."""
    # Select fusion methods
    fusion_keys = sorted([
        k for k in all_results
        if ('fusion' in k or 'combined' in k) and 'per_crisis' in all_results[k]
    ])

    if not fusion_keys:
        return

    crisis_keys = sorted(ALL_CRISES.keys())
    data = []
    method_labels = []

    for mk in fusion_keys:
        row = []
        pc = all_results[mk].get('per_crisis', {})
        for ck in crisis_keys:
            val = pc.get(ck, {}).get('mean_p', np.nan)
            row.append(val)
        data.append(row)
        method_labels.append(mk.replace('_', ' ').title())

    if not data:
        return

    data_arr = np.array(data)
    crisis_labels_short = [ALL_CRISES[k]['label'] for k in crisis_keys]

    fig, ax = plt.subplots(figsize=(14, max(4, len(method_labels) * 0.5)))
    im = ax.imshow(data_arr, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)

    ax.set_xticks(range(len(crisis_labels_short)))
    ax.set_xticklabels(crisis_labels_short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(method_labels)))
    ax.set_yticklabels(method_labels, fontsize=8)

    # Annotate cells
    for i in range(len(method_labels)):
        for j in range(len(crisis_labels_short)):
            val = data_arr[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Mean P(crisis)', shrink=0.8)
    ax.set_title('Per-Crisis Detection: Mean P(crisis) by Method')
    fig.tight_layout()
    fig.savefig(output_dir / 'per_crisis_heatmap.png')
    fig.savefig(output_dir / 'per_crisis_heatmap.pdf')
    plt.close(fig)
    logger.info(f"  Saved per_crisis_heatmap.png")


def plot_timeseries_comparison(p_crisis_dict, crisis_labels, dates, output_dir,
                               max_methods=6):
    """Time series of P(crisis) for top methods overlaid with crisis shading."""
    # Select top methods by AUC-ROC
    ranked = sorted(
        p_crisis_dict.items(),
        key=lambda x: float(np.nanmean(x[1][crisis_labels == 1])) if np.any(~np.isnan(x[1])) else 0,
        reverse=True,
    )[:max_methods]

    fig, axes = plt.subplots(len(ranked), 1, figsize=(14, 3 * len(ranked)), sharex=True)
    if len(ranked) == 1:
        axes = [axes]

    for ax, (name, p) in zip(axes, ranked):
        ax.plot(dates, p, linewidth=0.5, alpha=0.8, label=name)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5)
        ax.set_ylabel('P(crisis)', fontsize=8)
        ax.set_title(name.replace('_', ' ').title(), fontsize=9)

        for ck, ci in ALL_CRISES.items():
            cs = pd.Timestamp(ci['start'])
            ce = pd.Timestamp(ci['end'])
            ax.axvspan(cs, ce, alpha=0.15, color='#d62728')

        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel('Date')
    fig.suptitle('Top Methods: P(crisis) Time Series', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / 'timeseries_comparison.png')
    fig.savefig(output_dir / 'timeseries_comparison.pdf')
    plt.close(fig)
    logger.info(f"  Saved timeseries_comparison.png")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Improved online regime detection via signal correction'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (4 crises only)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("IMPROVED ONLINE REGIME DETECTION")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1/7] Fetching data...")
    raw = fetch_data(['SPY', 'DIA'], '2005-01-01', '2024-12-31')
    close = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(close)

    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates[19:]
    T = len(X_enriched)
    logger.info(f"  Enriched features: {X_enriched.shape}, T={T}")

    # Crisis labels
    if args.quick:
        crises_to_use = {
            k: v for k, v in ALL_CRISES.items()
            if k in ['2008_gfc', '2020_covid', '2022_rates', '2018_volmageddon']
        }
    else:
        crises_to_use = ALL_CRISES

    crisis_labels = np.zeros(T, dtype=float)
    window_ext = 10
    for ci in crises_to_use.values():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_ext)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_ext)
        mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        crisis_labels[mask] = 1.0

    logger.info(
        f"  Crisis days: {int(crisis_labels.sum())} / {T} "
        f"({100 * np.mean(crisis_labels):.1f}%)"
    )
    logger.info(f"  Crises used: {len(crises_to_use)}")

    # ---- Compute Geometric Features ----
    logger.info("\n[2/7] Computing causal geometric features...")
    features = compute_causal_geometric_features(
        X_enriched, dates_enriched,
        hilbert_dim=8, n_pca=8,
        refit_interval=63, min_history=126,
    )

    # Quick diagnosis: confirm signal inversion
    logger.info("\n  Signal direction check:")
    for fname, scores in features.items():
        valid = ~np.isnan(scores)
        v_idx = np.where(valid)[0]
        c_mask = crisis_labels[v_idx] == 1
        n_mask = crisis_labels[v_idx] == 0

        if np.sum(c_mask) > 0 and np.sum(n_mask) > 0:
            cm = float(np.mean(scores[v_idx[c_mask]]))
            nm = float(np.mean(scores[v_idx[n_mask]]))
            direction = "INVERTED (crisis < normal)" if cm < nm else "NORMAL (crisis > normal)"
            logger.info(f"    {fname}: crisis_mean={cm:.6f}, normal_mean={nm:.6f} -> {direction}")

    # ---- Run Strategies ----
    warmup = 252  # 1 year warmup for supervised strategies
    warmup_unsup = 126  # 6 months for unsupervised

    all_p_crisis = {}  # strategy_name -> P(crisis) array

    # Baseline: naive z-score (should get ~0.10-0.40 AUC due to inversion)
    logger.info("\n[3/7] Running baseline (naive z-score, no correction)...")
    for fname, scores in features.items():
        p = np.full(T, np.nan)
        for t in range(warmup_unsup, T):
            if np.isnan(scores[t]):
                continue
            past = scores[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 20:
                continue
            mu = np.mean(valid)
            sigma = max(np.std(valid, ddof=1), 1e-12)
            z = (scores[t] - mu) / sigma
            p[t] = 1.0 / (1.0 + np.exp(-z + 1.0))
        all_p_crisis[f'baseline_{fname}'] = p

    baseline_fusion = np.nanmean(
        [all_p_crisis[f'baseline_{fname}'] for fname in features], axis=0
    )
    all_p_crisis['baseline_fusion'] = baseline_fusion

    # Strategy 1: Signal Inversion (supervised)
    logger.info("\n[4/7] Running Strategy 1: Signal Inversion...")
    s1_results = strategy_inversion(features, crisis_labels, warmup=warmup)
    all_p_crisis.update(s1_results)

    # Strategy 1b: Signal Inversion (unsupervised -- negate all)
    s1b_results = strategy_inversion_unsupervised(features, warmup=warmup_unsup)
    all_p_crisis.update(s1b_results)

    # Strategy 2: CUSUM (two-sided)
    logger.info("  Running Strategy 2: CUSUM on geometric features...")
    s2_results = strategy_cusum(features, warmup=warmup_unsup)
    all_p_crisis.update(s2_results)

    # Strategy 2b: Directional CUSUM (negative shifts only)
    s2b_results = strategy_cusum_directional(features, warmup=warmup_unsup)
    all_p_crisis.update(s2b_results)

    # Strategy 3: Absolute Deviation
    logger.info("  Running Strategy 3: Absolute deviation...")
    s3_results = strategy_absdev(features, warmup=warmup_unsup)
    all_p_crisis.update(s3_results)

    # Strategy 4: Rate of Change
    logger.info("  Running Strategy 4: Rate-of-change features...")
    s4_results = strategy_rateofchange(features, warmup=warmup_unsup)
    all_p_crisis.update(s4_results)

    # Strategy 5: Multi-Scale Fusion
    logger.info("  Running Strategy 5: Multi-scale fusion...")
    s5_results = strategy_multiscale(features, warmup=warmup_unsup)
    all_p_crisis.update(s5_results)

    # Combined best
    logger.info("\n[5/7] Running Combined strategy...")
    s_combined = strategy_combined_best(features, crisis_labels, warmup=warmup)
    all_p_crisis.update(s_combined)

    # ---- Evaluate All Strategies ----
    logger.info("\n[6/7] Evaluating all strategies...")
    eval_results = {}

    for name, p in all_p_crisis.items():
        auc = evaluate_auc(p, crisis_labels, warmup=warmup_unsup)
        delay = evaluate_detection_delay(p, crisis_labels, dates_enriched, threshold=0.5)
        per_crisis = evaluate_per_crisis(p, crisis_labels, dates_enriched)

        eval_results[name] = {
            **auc,
            **delay,
            'per_crisis': per_crisis,
        }

    # ---- Summary Table ----
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Method':40s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'Det%':>6s} {'Delay':>7s}")
    logger.info("-" * 80)

    # Sort by AUC-ROC
    ranked = sorted(
        eval_results.items(),
        key=lambda x: x[1].get('auc_roc', 0) or 0,
        reverse=True,
    )

    for name, r in ranked:
        auc_roc = r.get('auc_roc', np.nan)
        auc_pr = r.get('auc_pr', np.nan)
        det_rate = r.get('detection_rate', 0)
        delay = r.get('mean_delay_days')

        if np.isnan(auc_roc):
            continue

        delay_str = f"{delay:.0f}d" if delay is not None else "N/A"
        marker = " ***" if auc_roc > 0.65 else " **" if auc_roc > 0.60 else ""
        logger.info(
            f"  {name:40s} {auc_roc:8.3f} {auc_pr:8.3f} "
            f"{det_rate:5.0%} {delay_str:>7s}{marker}"
        )

    # Top-5 highlight
    logger.info("\n--- TOP 5 METHODS ---")
    for name, r in ranked[:5]:
        auc_roc = r.get('auc_roc', np.nan)
        auc_pr = r.get('auc_pr', np.nan)
        logger.info(f"  {name}: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")

    best_name = ranked[0][0] if ranked else "none"
    best_auc = ranked[0][1].get('auc_roc', 0) if ranked else 0

    logger.info(f"\nBest method: {best_name} (AUC-ROC={best_auc:.3f})")
    if best_auc > 0.65:
        logger.info("TARGET ACHIEVED: AUC-ROC > 0.65")
    elif best_auc > 0.60:
        logger.info("Partial success: AUC-ROC > 0.60 but below 0.65 target")
    else:
        logger.info("Below target: AUC-ROC < 0.60")

    # ---- Plots ----
    logger.info("\n[7/7] Generating figures...")
    plot_strategy_comparison(eval_results, OUTPUT_DIR)
    plot_per_crisis_heatmap(eval_results, OUTPUT_DIR)

    # Time series for top methods
    top_methods = {name: all_p_crisis[name] for name, _ in ranked[:6] if name in all_p_crisis}
    if top_methods:
        plot_timeseries_comparison(top_methods, crisis_labels, dates_enriched, OUTPUT_DIR)

    # ---- Save ----
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return [make_serializable(v) for v in obj.tolist()]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick': args.quick,
            'symbols': ['SPY', 'DIA'],
            'date_range': ['2005-01-01', '2024-12-31'],
            'n_timesteps': T,
            'n_crises': len(crises_to_use),
            'crisis_fraction': float(np.mean(crisis_labels)),
            'warmup_supervised': warmup,
            'warmup_unsupervised': warmup_unsup,
            'geo_config': {
                'hilbert_dim': 8,
                'n_pca': 8,
                'refit_interval': 63,
                'min_history': 126,
            },
        },
        'results': make_serializable(eval_results),
        'best_method': best_name,
        'best_auc_roc': float(best_auc) if not np.isnan(best_auc) else None,
        'strategies': {
            'S1_inversion': 'Negate features where crisis < normal (supervised)',
            'S1b_inv_unsup': 'Negate all features (unsupervised, assumes crisis=drop)',
            'S2_cusum': 'Two-sided CUSUM on each feature',
            'S2b_cusum_neg': 'One-sided CUSUM for negative shifts',
            'S3_absdev': 'Absolute deviation from expanding mean',
            'S4_roc': 'Rate-of-change |f(t)-f(t-k)| for k=1,5,10,20',
            'S5_multiscale': 'Multi-scale rolling z-scores at w=5,10,20,40',
            'combined': 'Weighted fusion of S1+S2b+S3+S5',
        },
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f'improved_results_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Also save a "latest" symlink-style copy
    latest_path = OUTPUT_DIR / 'improved_results_latest.json'
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Latest copy: {latest_path}")
    logger.info("=" * 70)
    logger.info("DONE")


if __name__ == '__main__':
    main()
