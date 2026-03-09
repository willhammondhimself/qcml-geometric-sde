"""
Fusion/Combination Questions Q36-Q39: Empirical Smoke Tests
=============================================================

Q36: Attention-weighted fusion (self-attention via softmax over z-scores)
Q37: Majority vote (count of top-5 observables exceeding z-score threshold)
Q38: Gaussian copula joint exceedance probability
Q39: Online exponential weights (Hedge/MW algorithm)

Protocol
--------
- 4 crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
- 5 base detectors: BerryPhaseRate, SpectralGap, ReducedPurity,
  SpectralEntropy, DimensionalityCollapse
- SPY only, yfinance
- Cohen's d per crisis; verdict: keep if median d > 0.3

Run:
    cd /Users/willhammond/Will\ x\ Average\ Research/qcml-geometric-sde
    python research/ideation/fusion_methods/test_q36_q39.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

from qcml_geometry.observables import (
    BerryPhaseRateDetector,
    SpectralGapDetector,
    ReducedPurityDetector,
    SpectralEntropyDetector,
    DimensionalityCollapseDetector,
)
from experiments.data_loader import fetch_data, create_feature_matrix_single_asset, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)

SMOKE_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
SYMBOL = 'SPY'
DATA_START = '2000-01-01'
DATA_END = '2024-12-31'
KEEP_THRESHOLD = 0.3  # median Cohen's d to keep a method


# ---------------------------------------------------------------------------
# Step 1: Fetch data and build feature matrix
# ---------------------------------------------------------------------------

def load_data():
    print("Fetching SPY data ...")
    df = fetch_data([SYMBOL], DATA_START, DATA_END, source='yfinance')
    prices = df.xs(SYMBOL, level='symbol')['close']
    features, dates = create_feature_matrix_single_asset(prices, extra_lags=True)
    print(f"  Features: {features.shape}, dates: {dates[0].date()} to {dates[-1].date()}")
    return features, dates


def build_crisis_mask(dates, crisis_key: str) -> np.ndarray:
    """Binary array: 1 during crisis, 0 otherwise.

    Args:
        dates: DatetimeIndex or numpy array of Timestamps.
        crisis_key: Key into ALL_CRISES dict.

    Returns:
        mask: int array of shape (T,), 1 inside crisis window.
    """
    info = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(info['start'])
    ce = pd.Timestamp(info['end'])
    # Ensure we work with a pandas Series for consistent comparison
    dates_series = pd.Series(dates)
    mask = (dates_series >= cs) & (dates_series <= ce)
    return mask.values.astype(int)


# ---------------------------------------------------------------------------
# Step 2: Run base detectors and collect z-score time series
# ---------------------------------------------------------------------------

def run_base_detectors(features: np.ndarray) -> dict[str, np.ndarray]:
    """Fit and score the 5 base detectors. Returns {name: scores array}."""
    detectors = [
        BerryPhaseRateDetector(hilbert_dim=8, n_pca_components=15, seed=42),
        SpectralGapDetector(hilbert_dim=8, n_pca_components=15, seed=42),
        ReducedPurityDetector(hilbert_dim=8, n_pca_components=15, seed=42),
        SpectralEntropyDetector(hilbert_dim=8, n_pca_components=15, seed=42),
        DimensionalityCollapseDetector(hilbert_dim=8, n_pca_components=15, seed=42),
    ]

    scores = {}
    for det in detectors:
        print(f"  Running {det.name} ...")
        det.fit(features)
        s = det.compute_regime_scores(features)
        scores[det.name] = s
        print(f"    Valid: {np.sum(~np.isnan(s))}/{len(s)} steps, "
              f"mean={np.nanmean(s):.3f}, std={np.nanstd(s):.3f}")

    return scores


def score_matrix(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Stack dict of score arrays into (T, n_methods) matrix."""
    keys = list(scores.keys())
    return np.column_stack([scores[k] for k in keys]), keys


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def compute_all_d(fused_scores: np.ndarray, dates: pd.DatetimeIndex,
                  label: str = "") -> dict[str, float]:
    """Compute Cohen's d for each of the 4 smoke crises."""
    results = {}
    for crisis_key in SMOKE_CRISES:
        mask = build_crisis_mask(dates, crisis_key)
        crisis_idx = np.where(mask == 1)[0]
        normal_idx = np.where(mask == 0)[0]

        crisis_sc = fused_scores[crisis_idx]
        normal_sc = fused_scores[normal_idx]

        # Filter NaN
        crisis_sc = crisis_sc[~np.isnan(crisis_sc)]
        normal_sc = normal_sc[~np.isnan(normal_sc)]

        if len(crisis_sc) < 5 or len(normal_sc) < 20:
            results[crisis_key] = np.nan
            continue

        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_sc, normal_sc, n_bootstrap=1000, seed=42
        )
        results[crisis_key] = d

    ds = [v for v in results.values() if not np.isnan(v)]
    median_d = float(np.median(ds)) if ds else np.nan
    verdict = "KEEP" if median_d > KEEP_THRESHOLD else "SKIP"

    print(f"\n  {label}")
    for ck, d in results.items():
        name = ALL_CRISES[ck]['label']
        print(f"    {name:30s}: d={d:.3f}" if not np.isnan(d) else
              f"    {name:30s}: d=NaN")
    print(f"    Median d = {median_d:.3f}  →  {verdict}")

    return {'per_crisis': results, 'median_d': median_d, 'verdict': verdict}


# ---------------------------------------------------------------------------
# Q36: Attention-weighted fusion (self-attention)
# ---------------------------------------------------------------------------

def q36_attention_fusion(smat: np.ndarray) -> np.ndarray:
    """Self-attention fusion: softmax over current z-scores as weights.

    At each timestep t, weight = softmax(|z_i(t)|), then fused = sum(w_i * z_i).
    Negative z-scores are used in abs() so high-magnitude signals (in either
    direction) attract attention. The fused score is then expanding z-scored.

    Args:
        smat: (T, n_methods) z-score matrix.

    Returns:
        fused: (T,) expanding z-scored fused score.
    """
    T, n_methods = smat.shape
    fused_raw = np.full(T, np.nan)

    for t in range(T):
        row = smat[t]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        vals = row[valid]
        # Softmax attention on absolute z-scores — high-magnitude channels
        # get more weight regardless of sign
        abs_vals = np.abs(vals)
        abs_vals_shifted = abs_vals - abs_vals.max()  # numerical stability
        exp_vals = np.exp(abs_vals_shifted)
        weights = exp_vals / exp_vals.sum()
        # Weighted mean preserving sign
        fused_raw[t] = np.dot(weights, vals)

    # Expanding z-score of the fused signal
    from qcml_geometry.fusion import _expanding_zscore_1d
    return _expanding_zscore_1d(fused_raw, min_obs=60)


def q36_uniform_fusion(smat: np.ndarray) -> np.ndarray:
    """Uniform average as baseline for Q36.

    Args:
        smat: (T, n_methods) z-score matrix.

    Returns:
        fused: (T,) expanding z-scored fused score.
    """
    from qcml_geometry.fusion import _expanding_zscore_1d
    raw = np.nanmean(smat, axis=1)
    return _expanding_zscore_1d(raw, min_obs=60)


# ---------------------------------------------------------------------------
# Q37: Majority vote
# ---------------------------------------------------------------------------

def q37_majority_vote(smat: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Majority vote: fraction of observables exceeding z-score threshold.

    Score at time t = count(z_i(t) > threshold) / n_observables.
    Then expanding z-scored for comparability.

    Args:
        smat: (T, n_methods) z-score matrix.
        threshold: z-score threshold for "firing".

    Returns:
        fused: (T,) expanding z-scored vote fraction.
    """
    from qcml_geometry.fusion import _expanding_zscore_1d
    T, n_methods = smat.shape
    raw = np.full(T, np.nan)

    for t in range(T):
        row = smat[t]
        valid = ~np.isnan(row)
        n_valid = valid.sum()
        if n_valid < 1:
            continue
        n_firing = np.sum(row[valid] > threshold)
        raw[t] = n_firing / n_valid

    return _expanding_zscore_1d(raw, min_obs=60)


# ---------------------------------------------------------------------------
# Q38: Gaussian copula joint exceedance
# ---------------------------------------------------------------------------

def _empirical_cdf_expanding(vals: np.ndarray) -> np.ndarray:
    """Compute expanding-window empirical CDF rank for each timepoint.

    F_t(x_t) = P(X <= x_t | data[:t]) — strictly causal.

    Args:
        vals: (T,) raw signal.

    Returns:
        u: (T,) uniform marginals in [0, 1], NaN for first min_obs steps.
    """
    T = len(vals)
    u = np.full(T, np.nan)
    min_obs = 60
    for t in range(min_obs, T):
        past = vals[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10 or np.isnan(vals[t]):
            continue
        u[t] = np.sum(past_valid <= vals[t]) / len(past_valid)
        # Avoid 0/1 boundary (probit undefined)
        u[t] = np.clip(u[t], 1e-6, 1 - 1e-6)
    return u


def q38_gaussian_copula(smat: np.ndarray) -> np.ndarray:
    """Gaussian copula joint exceedance probability.

    1. Convert each marginal to uniform via expanding empirical CDF.
    2. Apply probit (Phi^{-1}) to get Gaussian-marginal scores.
    3. Estimate expanding correlation matrix R from past Gaussian scores.
    4. Joint exceedance under Gaussian copula ≈ multivariate normal CDF.
    5. Score = log(1 - C(u_1, ..., u_n)) — log tail probability.

    This captures tail dependence (crisis = simultaneous exceedance).

    Args:
        smat: (T, n_methods) z-score matrix.

    Returns:
        fused: (T,) expanding z-scored copula score.
    """
    from scipy.stats import norm as scipy_norm
    from qcml_geometry.fusion import _expanding_zscore_1d

    T, n_methods = smat.shape

    # Step 1: Expanding empirical CDFs -> uniform marginals
    u_mat = np.full((T, n_methods), np.nan)
    for m in range(n_methods):
        u_mat[:, m] = _empirical_cdf_expanding(smat[:, m])

    # Step 2: Probit transform to Gaussian marginals
    g_mat = scipy_norm.ppf(u_mat)  # NaN propagated where u is NaN

    # Step 3-4: Compute joint Gaussian score using expanding correlation
    # For speed: use the mean of the Gaussian marginals weighted by
    # the expanding correlation structure (simplified copula score).
    # Full multivariate CDF is expensive; instead use Mahalanobis-inspired
    # score: x^T R^{-1} x, where R is the running correlation matrix.

    raw = np.full(T, np.nan)
    min_corr_obs = 120  # need enough history to estimate correlations

    for t in range(min_corr_obs, T):
        g_past = g_mat[:t]
        valid_rows = ~np.any(np.isnan(g_past), axis=1)
        g_valid = g_past[valid_rows]

        if g_valid.shape[0] < 30:
            continue

        g_curr = g_mat[t]
        if np.any(np.isnan(g_curr)):
            # Fall back to mean of available channels
            raw[t] = np.nanmean(g_curr)
            continue

        # Expanding sample correlation matrix with regularization
        R = np.corrcoef(g_valid.T)
        # Regularize: shrink toward identity (Ledoit-Wolf simplified)
        alpha = 0.1
        R_reg = (1 - alpha) * R + alpha * np.eye(n_methods)

        try:
            R_inv = np.linalg.inv(R_reg)
            # Mahalanobis score: captures joint tail behavior
            raw[t] = float(g_curr @ R_inv @ g_curr)
        except np.linalg.LinAlgError:
            raw[t] = np.nanmean(g_curr)

    return _expanding_zscore_1d(raw, min_obs=60)


# ---------------------------------------------------------------------------
# Q39: Online exponential weights (Hedge algorithm)
# ---------------------------------------------------------------------------

def q39_exponential_weights(
    smat: np.ndarray, dates: pd.DatetimeIndex,
    crisis_keys: list[str], eta: float = 0.1,
) -> np.ndarray:
    """Exponential weights (Hedge/MW) with expanding-window hit-rate updates.

    At each step t, the weight of observable i is proportional to
    exp(eta * cumulative_hit_rate_i(t)), where hit rate = fraction of past
    steps where z_i > 1.0 AND market was anomalous (top-quartile realized vol).

    This is a causal, online learning approach — no future information used.

    Args:
        smat: (T, n_methods) z-score matrix.
        dates: DatetimeIndex of length T.
        crisis_keys: List of crisis identifiers for building reference labels.
        eta: Learning rate for exponential weighting.

    Returns:
        fused: (T,) expanding z-scored hedge-weighted score.
    """
    from qcml_geometry.fusion import _expanding_zscore_1d

    T, n_methods = smat.shape

    # Proxy "anomaly" labels: realized volatility in top 25% expanding quantile.
    # This is unsupervised and strictly causal — no crisis labels leaked.
    # We use a rolling 20-day window to smooth out noise.
    rolling_vol = pd.Series(np.nanstd(smat, axis=1)).rolling(20, min_periods=5).mean().values

    raw = np.full(T, np.nan)

    # Expanding hit-count tracker per method
    hit_counts = np.zeros(n_methods)
    total_counts = np.zeros(n_methods)

    for t in range(T):
        row = smat[t]
        valid = ~np.isnan(row)

        if t >= 60 and valid.sum() >= 2:
            # Compute weights from running hit rates
            hit_rates = np.where(
                total_counts > 0,
                hit_counts / np.maximum(total_counts, 1),
                1.0 / n_methods,
            )
            # Exponential weighting
            log_w = eta * hit_rates * valid.astype(float)
            log_w -= log_w.max()  # numerical stability
            weights = np.exp(log_w)
            weights = weights * valid  # zero out NaN channels
            w_sum = weights.sum()
            if w_sum > 1e-12:
                weights = weights / w_sum
            else:
                weights = valid.astype(float) / max(valid.sum(), 1)

            raw[t] = np.dot(weights, np.where(np.isnan(row), 0.0, row))

        # Update hit counts — expanding window, causal
        # "Hit" = method fired (z > 1.0) AND current vol is high
        # (top-quartile of past distribution)
        if t >= 20 and not np.isnan(rolling_vol[t]):
            past_vol = rolling_vol[:t]
            past_valid_vol = past_vol[~np.isnan(past_vol)]
            if len(past_valid_vol) >= 10:
                vol_thresh = np.percentile(past_valid_vol, 75)
                is_anomalous = rolling_vol[t] > vol_thresh
            else:
                is_anomalous = False

            for m in range(n_methods):
                if not np.isnan(row[m]):
                    total_counts[m] += 1
                    if row[m] > 1.0 and is_anomalous:
                        hit_counts[m] += 1

    return _expanding_zscore_1d(raw, min_obs=60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Fusion Questions Q36-Q39: Empirical Smoke Test")
    print("=" * 70)

    # --- Load data ---
    features, dates = load_data()

    # --- Run base detectors ---
    print("\nRunning 5 base detectors ...")
    base_scores = run_base_detectors(features)
    smat, method_names = score_matrix(base_scores)
    print(f"\n  Score matrix shape: {smat.shape}")
    print(f"  Methods: {method_names}")

    # Also compute individual detector performance for reference
    print("\n--- Individual Detector Performance (reference) ---")
    ind_results = {}
    for name, scores in base_scores.items():
        ind_results[name] = compute_all_d(scores, dates, label=f"Individual: {name}")

    # -----------------------------------------------------------------------
    # Q36: Attention-weighted vs uniform
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q36: Attention-weighted fusion (self-attention via softmax)")
    print("=" * 70)

    attn_scores = q36_attention_fusion(smat)
    unif_scores = q36_uniform_fusion(smat)

    q36_attn = compute_all_d(attn_scores, dates, label="Q36: Attention fusion")
    q36_unif = compute_all_d(unif_scores, dates, label="Q36: Uniform fusion (baseline)")

    attn_better = q36_attn['median_d'] > q36_unif['median_d']
    print(f"\n  Attention vs Uniform: {q36_attn['median_d']:.3f} vs {q36_unif['median_d']:.3f}")
    print(f"  Attention BETTER than uniform: {attn_better}")
    print(f"  VERDICT: {'KEEP' if q36_attn['median_d'] > KEEP_THRESHOLD else 'SKIP'}")

    # -----------------------------------------------------------------------
    # Q37: Majority vote at 3 thresholds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q37: Majority vote (count of observables above threshold)")
    print("=" * 70)

    q37_results = {}
    for threshold in [1.0, 1.5, 2.0]:
        vote_scores = q37_majority_vote(smat, threshold=threshold)
        label = f"Q37: Majority vote (threshold={threshold})"
        q37_results[threshold] = compute_all_d(vote_scores, dates, label=label)

    best_thresh = max(q37_results, key=lambda t: q37_results[t]['median_d'])
    best_d = q37_results[best_thresh]['median_d']
    ra_ref = 0.774  # Paper 2 Regime-Adaptive reference

    print(f"\n  Best threshold: {best_thresh}, median d={best_d:.3f}")
    print(f"  vs Regime-Adaptive (Paper 2): d={ra_ref:.3f}")
    print(f"  Majority vote beats Regime-Adaptive: {best_d > ra_ref}")
    print(f"  VERDICT: {'KEEP' if best_d > KEEP_THRESHOLD else 'SKIP'}")

    # -----------------------------------------------------------------------
    # Q38: Gaussian copula vs simple mean
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q38: Gaussian copula joint exceedance (Mahalanobis-copula)")
    print("=" * 70)

    copula_scores = q38_gaussian_copula(smat)
    mean_baseline = np.nanmean(smat, axis=1)  # simple marginal mean

    q38_copula = compute_all_d(copula_scores, dates, label="Q38: Gaussian copula")
    q38_mean = compute_all_d(mean_baseline, dates, label="Q38: Simple mean (baseline)")

    copula_better = q38_copula['median_d'] > q38_mean['median_d']
    print(f"\n  Copula vs Simple Mean: {q38_copula['median_d']:.3f} vs {q38_mean['median_d']:.3f}")
    print(f"  Copula BETTER than marginal mean: {copula_better}")
    print(f"  VERDICT: {'KEEP' if q38_copula['median_d'] > KEEP_THRESHOLD else 'SKIP'}")

    # -----------------------------------------------------------------------
    # Q39: Online exponential weights
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q39: Online exponential weights (Hedge/MW algorithm)")
    print("=" * 70)

    hedge_scores = q39_exponential_weights(smat, dates, SMOKE_CRISES, eta=0.1)
    q39_hedge = compute_all_d(hedge_scores, dates, label="Q39: Exponential weights (eta=0.1)")

    # Also try eta=0.5
    hedge_05 = q39_exponential_weights(smat, dates, SMOKE_CRISES, eta=0.5)
    q39_hedge_05 = compute_all_d(hedge_05, dates, label="Q39: Exponential weights (eta=0.5)")

    best_hedge_d = max(q39_hedge['median_d'], q39_hedge_05['median_d'])
    print(f"\n  Best Hedge median d={best_hedge_d:.3f}")
    print(f"  vs Regime-Adaptive (Paper 2): d={ra_ref:.3f}")
    print(f"  VERDICT: {'KEEP' if best_hedge_d > KEEP_THRESHOLD else 'SKIP'}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<40} {'Median d':>10} {'Verdict':>10}")
    print("-" * 62)

    # Individual baselines
    for name, res in ind_results.items():
        d = res['median_d']
        v = res['verdict']
        print(f"  {name:<38} {d:>10.3f} {v:>10}")

    print("-" * 62)
    print(f"  {'Q36 Attention Fusion':<38} {q36_attn['median_d']:>10.3f} {q36_attn['verdict']:>10}")
    print(f"  {'Q36 Uniform Fusion':<38} {q36_unif['median_d']:>10.3f} {q36_unif['verdict']:>10}")
    print("-" * 62)
    for thr, res in q37_results.items():
        label = f"Q37 Vote (thr={thr})"
        print(f"  {label:<38} {res['median_d']:>10.3f} {res['verdict']:>10}")
    print("-" * 62)
    print(f"  {'Q38 Gaussian Copula':<38} {q38_copula['median_d']:>10.3f} {q38_copula['verdict']:>10}")
    print(f"  {'Q38 Simple Mean':<38} {q38_mean['median_d']:>10.3f} {q38_mean['verdict']:>10}")
    print("-" * 62)
    print(f"  {'Q39 Hedge eta=0.1':<38} {q39_hedge['median_d']:>10.3f} {q39_hedge['verdict']:>10}")
    print(f"  {'Q39 Hedge eta=0.5':<38} {q39_hedge_05['median_d']:>10.3f} {q39_hedge_05['verdict']:>10}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Per-crisis breakdown for main fusion candidates
    # -----------------------------------------------------------------------
    print("\nPer-crisis breakdown for new fusion methods:")
    for qname, res, scores in [
        ("Q36 Attention", q36_attn, attn_scores),
        ("Q37 Vote thr=1.0", q37_results[1.0], q37_majority_vote(smat, 1.0)),
        ("Q38 Copula", q38_copula, copula_scores),
        ("Q39 Hedge eta=0.1", q39_hedge, hedge_scores),
    ]:
        print(f"\n  {qname}:")
        for ck in SMOKE_CRISES:
            d = res['per_crisis'].get(ck, np.nan)
            label = ALL_CRISES[ck]['label']
            bar = "#" * int(min(d, 2.0) * 10) if not np.isnan(d) else ""
            print(f"    {label:30s}: d={d:.3f}  {bar}")

    print("\nDone.")


if __name__ == "__main__":
    main()
