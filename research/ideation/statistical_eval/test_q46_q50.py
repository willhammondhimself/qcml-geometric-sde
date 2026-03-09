"""
Q46-Q50: Statistical/Evaluation Methodology Investigation

Tests alternative evaluation approaches for QCML regime detectors.

Q46: Detection delay (lead time) as primary metric vs Cohen's d
Q47: Conformal prediction for distribution-free confidence intervals
Q48: Survival analysis (Cox PH) framing for lead time
Q49: Granger causality from observables to realized volatility
Q50: Continuous crisis intensity (realized vol) vs binary labels

Usage:
    python research/ideation/statistical_eval/test_q46_q50.py

Data: SPY via yfinance. 4 standard crises (GFC, COVID, Flash Crash, Euro Crisis).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.lead_time_analysis import compute_lead_time
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import BaseRegimeDetector
from qcml_geometry import (
    SpectralEntropyDetector,
    BerryPhaseRateDetector,
    ReducedPurityDetector,
    DimensionalityCollapseDetector,
    HamiltonianSensitivityDetector,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# 4 standard crises for focused tests
# ---------------------------------------------------------------------------
FOUR_CRISES = {
    '2008_gfc': ALL_CRISES['2008_gfc'],
    '2010_flash': ALL_CRISES['2010_flash'],
    '2011_euro': ALL_CRISES['2011_euro'],
    '2020_covid': ALL_CRISES['2020_covid'],
}

# Top-5 detectors by Cohen's d (from canonical JSON)
TOP5_CONFIGS = {
    'Spectral Entropy': {
        'class': SpectralEntropyDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=42,
            normalization='soft', adaptive_epsilon=True,
        ),
        'cohens_d_rank': 1,
        'cohens_d': 0.830,
    },
    'Reduced Purity': {
        'class': ReducedPurityDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=42,
            normalization='soft', adaptive_epsilon=True,
            partition=(2, 4),
        ),
        'cohens_d_rank': 2,
        'cohens_d': 0.643,
    },
    'Dimensionality Collapse': {
        'class': DimensionalityCollapseDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=42,
            normalization='soft', adaptive_epsilon=True,
            subsample=5,
        ),
        'cohens_d_rank': 3,
        'cohens_d': 0.793,
    },
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
            normalization='sphere', berry_aggregation='f01',
        ),
        'cohens_d_rank': 4,
        'cohens_d': 0.608,
    },
    'Hamiltonian Sensitivity': {
        'class': HamiltonianSensitivityDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=8, rolling_window=20,
            operator_method='random', seed=42,
            normalization='soft', adaptive_epsilon=True,
        ),
        'cohens_d_rank': 5,
        'cohens_d': 0.534,
    },
}


# =============================================================================
# Shared data loading
# =============================================================================

def load_shared_data():
    """Fetch SPY+DIA, build enriched feature matrix, return scores for top-5."""
    print("\n[DATA] Fetching SPY+DIA 2004-2024...")
    raw = fetch_data(['SPY', 'DIA'], '2004-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()

    X, dates = create_feature_matrix(prices_df)
    X_enr = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enr = dates[19:]

    # SPY close prices aligned with enriched dates
    spy_close = prices_df['SPY'].reindex(dates_enr)

    print(f"  Feature matrix: {X_enr.shape}, "
          f"dates: {dates_enr[0].date()} to {dates_enr[-1].date()}")

    print("\n[DATA] Computing scores for top-5 detectors...")
    all_scores = {}
    for name, cfg in TOP5_CONFIGS.items():
        det = cfg['class'](**cfg['params'])
        try:
            det.fit(X_enr)
            scores = det.compute_regime_scores(X_enr)
            all_scores[name] = scores
            print(f"  {name}: OK ({np.isfinite(scores).sum()} valid points)")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    return X_enr, dates_enr, spy_close, all_scores


# =============================================================================
# Q46: Detection Delay vs Cohen's d
# =============================================================================

def run_q46(dates_enr, all_scores):
    """
    Q46: Is detection delay a better primary metric than Cohen's d?

    Compute detection delay (trading days from crisis onset to alarm) for each
    top-5 observable across 4 crises. Compare ranking by mean detection delay
    vs ranking by Cohen's d.

    Detection delay < 0 is "lead time" (alarm before crisis = earlier = better).
    We report lead_time = -delay so higher = better (to match Cohen's d direction).
    """
    print("\n" + "=" * 70)
    print("Q46: DETECTION DELAY vs COHEN'S d")
    print("=" * 70)

    lead_times = {}  # name -> {crisis: lead_days}
    for name in TOP5_CONFIGS:
        scores = all_scores.get(name)
        if scores is None:
            continue
        lead_times[name] = {}
        for crisis_key, crisis_info in FOUR_CRISES.items():
            crisis_start = pd.Timestamp(crisis_info['start'])
            lt = compute_lead_time(
                scores, dates_enr, crisis_start,
                lookback_days=252, min_persistence=3,
                threshold_quantile=0.95, min_expanding=60,
            )
            lead_times[name][crisis_key] = lt['lead_time_days']

    # Build summary
    print(f"\n{'Method':<25} {'Mean Lead':>10} {'n_det':>6} {'d_rank':>7}")
    print("-" * 52)
    summary = {}
    for name, lt_dict in lead_times.items():
        valid = [v for v in lt_dict.values() if v is not None and np.isfinite(v)]
        mean_lt = np.mean(valid) if valid else np.nan
        summary[name] = {
            'mean_lead_time': mean_lt,
            'n_detected': len(valid),
            'cohens_d_rank': TOP5_CONFIGS[name]['cohens_d_rank'],
            'cohens_d': TOP5_CONFIGS[name]['cohens_d'],
            'per_crisis': lt_dict,
        }
        print(f"{name:<25} {mean_lt:>10.1f} {len(valid):>6} "
              f"{TOP5_CONFIGS[name]['cohens_d_rank']:>7}")

    # Rank by lead time
    valid_methods = [n for n in summary if np.isfinite(summary[n]['mean_lead_time'])]
    lead_time_rank = sorted(valid_methods,
                            key=lambda n: summary[n]['mean_lead_time'],
                            reverse=True)

    print("\nLead-time ranking vs Cohen's d ranking:")
    print(f"{'Lead-time rank':<5} {'Method':<25} {'Mean Lead':>10} {'d rank':>7}")
    for lt_rank, name in enumerate(lead_time_rank, 1):
        d_rank = summary[name]['cohens_d_rank']
        agree = "AGREE" if abs(lt_rank - d_rank) <= 1 else f"DIFF ({lt_rank - d_rank:+d})"
        print(f"{lt_rank:<5} {name:<25} {summary[name]['mean_lead_time']:>10.1f} "
              f"{d_rank:>7}  {agree}")

    # Kendall tau rank correlation
    if len(valid_methods) >= 3:
        lt_ranks_arr = np.array([
            next(i+1 for i, n in enumerate(lead_time_rank) if n == name)
            for name in valid_methods
        ])
        d_ranks_arr = np.array([summary[name]['cohens_d_rank'] for name in valid_methods])
        tau, tau_p = stats.kendalltau(lt_ranks_arr, d_ranks_arr)
        print(f"\nKendall tau (lead-time rank vs Cohen's d rank): "
              f"tau={tau:.3f}, p={tau_p:.3f}")
    else:
        tau, tau_p = np.nan, np.nan
        print("\nInsufficient methods for Kendall tau")

    # Per-crisis breakdown
    print("\nPer-crisis lead times (days before crisis onset):")
    header = f"{'Method':<25} " + " ".join(f"{k[-4:][:8]:>10}" for k in FOUR_CRISES)
    print(header)
    for name in valid_methods:
        row = f"{name:<25} "
        for ck in FOUR_CRISES:
            lt_val = lead_times[name].get(ck, np.nan)
            if lt_val is None or not np.isfinite(lt_val):
                row += f"{'N/A':>10} "
            else:
                row += f"{lt_val:>10.0f} "
        print(row)

    return {
        'summary': summary,
        'lead_time_ranking': lead_time_rank,
        'kendall_tau': tau,
        'kendall_p': tau_p,
        'verdict': 'KEEP' if abs(tau) < 0.8 else 'REJECT',
    }


# =============================================================================
# Q47: Conformal Prediction
# =============================================================================

def run_q47(dates_enr, all_scores):
    """
    Q47: Can conformal prediction give valid coverage for our detectors?

    Split conformal prediction:
    - Calibration set: randomly sampled from non-crisis periods
    - Test set: held-out non-crisis + crisis periods
    - Score: |predicted_quantile - actual indicator|
    - Claim: 90% coverage band should contain 90% of test points

    We use SpectralEntropy (top detector) and BerryPhaseRate.
    The conformal score is the normalized z-score threshold.
    """
    print("\n" + "=" * 70)
    print("Q47: CONFORMAL PREDICTION CALIBRATION")
    print("=" * 70)

    # Build crisis mask for 4 crises
    T = len(dates_enr)
    crisis_mask = np.zeros(T, dtype=bool)
    for ck, cinfo in FOUR_CRISES.items():
        cs = pd.Timestamp(cinfo['start'])
        ce = pd.Timestamp(cinfo['end'])
        for i, d in enumerate(dates_enr):
            if cs <= d <= ce:
                crisis_mask[i] = True

    n_crisis = crisis_mask.sum()
    n_normal = (~crisis_mask).sum()
    print(f"Crisis days: {n_crisis}, Normal days: {n_normal}")

    results = {}
    for det_name in ['Spectral Entropy', 'Berry Phase Rate']:
        scores = all_scores.get(det_name)
        if scores is None:
            print(f"  {det_name}: no scores")
            continue

        # Remove NaN
        valid_mask = np.isfinite(scores)
        s = scores[valid_mask]
        cm = crisis_mask[valid_mask]

        # Normalize scores to [0,1] range via ranking (distribution-free)
        score_ranks = stats.rankdata(s) / len(s)

        # Split calibration: use first 60% of normal days as calibration
        normal_idx = np.where(~cm)[0]
        n_cal = int(0.6 * len(normal_idx))
        cal_idx = normal_idx[:n_cal]
        test_normal_idx = normal_idx[n_cal:]

        # Conformal nonconformity scores on calibration set
        # Score: 1 - rank (higher score = more anomalous = nonconforming)
        cal_scores = score_ranks[cal_idx]
        # Nonconformity measure: we want to predict "normal" (label=0)
        # For normal points: nonconformity = 1 - (1 - rank) = rank (anomaly level)
        nonconformity_cal = cal_scores  # lower rank = more typical

        # Find conformal quantile for 90% coverage
        alpha = 0.10  # 1 - target coverage
        n_cal_size = len(nonconformity_cal)
        q_level = np.ceil((n_cal_size + 1) * (1 - alpha)) / n_cal_size
        q_level = min(q_level, 1.0)
        conformal_threshold = np.quantile(nonconformity_cal, q_level)

        # Test coverage on held-out normal days (should be >= 90%)
        test_normal_scores = score_ranks[test_normal_idx]
        normal_coverage = np.mean(test_normal_scores <= conformal_threshold)

        # Test coverage on crisis days (expect < 90% — crisis should be "outside" the band)
        crisis_idx_arr = np.where(cm)[0]
        crisis_scores_arr = score_ranks[crisis_idx_arr]
        crisis_coverage = np.mean(crisis_scores_arr <= conformal_threshold)

        print(f"\n{det_name}:")
        print(f"  Calibration set size: {n_cal_size}")
        print(f"  Conformal threshold (90% level): {conformal_threshold:.4f}")
        print(f"  Normal day coverage: {normal_coverage:.3f} (target >= 0.90)")
        print(f"  Crisis day coverage: {crisis_coverage:.3f} (want < 0.90 = crisis detectable)")
        print(f"  Calibration VALID: {normal_coverage >= 0.88}")
        print(f"  Crisis detection power: {1 - crisis_coverage:.3f} "
              f"(fraction of crisis days outside band)")

        results[det_name] = {
            'conformal_threshold': float(conformal_threshold),
            'normal_coverage': float(normal_coverage),
            'crisis_coverage': float(crisis_coverage),
            'crisis_detection_power': float(1 - crisis_coverage),
            'calibration_valid': bool(normal_coverage >= 0.88),
        }

    # Verdict
    valid_results = [r for r in results.values()]
    if valid_results:
        avg_normal_cov = np.mean([r['normal_coverage'] for r in valid_results])
        avg_crisis_power = np.mean([r['crisis_detection_power'] for r in valid_results])
        print(f"\nAverage normal coverage: {avg_normal_cov:.3f}")
        print(f"Average crisis detection power: {avg_crisis_power:.3f}")
        verdict = 'KEEP' if avg_normal_cov >= 0.88 and avg_crisis_power > 0.20 else 'REJECT'
    else:
        verdict = 'REJECT'

    results['verdict'] = verdict
    return results


# =============================================================================
# Q48: Survival Analysis
# =============================================================================

def run_q48(dates_enr, spy_close, all_scores):
    """
    Q48: Survival analysis (Cox PH) framing for lead time.

    Model "time until crisis onset" as a survival event. Each observable's
    z-score is a time-varying covariate.

    Implementation without lifelines:
    - For each pre-crisis window (up to 252 days before each crisis):
      * Record (time_to_crisis, observable_z_score, event_occurred=1 for crisis)
      * Estimate hazard using Breslow estimator approximation
    - Report: Pearson correlation between z-score and 1/(time_to_crisis+1)
      as a proxy for the Cox log-hazard coefficient

    This is a simplified Cox-like analysis: positive correlation means
    higher z-score -> shorter time-to-crisis (higher hazard).
    """
    print("\n" + "=" * 70)
    print("Q48: SURVIVAL ANALYSIS (COX-LIKE) FRAMING")
    print("=" * 70)

    # Build time-to-crisis series for each crisis
    T = len(dates_enr)

    all_tte = []   # time-to-event (days)
    all_event = [] # 1 = crisis actually occurred (all = 1 here)
    all_z = {name: [] for name in TOP5_CONFIGS if name in all_scores}

    # SPY realized vol (20-day) as time series
    spy_log_ret = np.log(spy_close / spy_close.shift(1)).fillna(0).values

    for ck, cinfo in FOUR_CRISES.items():
        crisis_start = pd.Timestamp(cinfo['start'])
        crisis_idx = int(np.searchsorted(dates_enr, crisis_start))

        # Only look at 252 days before crisis
        lookback = min(252, crisis_idx - 60)
        if lookback < 30:
            continue

        window_start = crisis_idx - lookback

        for t in range(window_start, crisis_idx):
            tte = crisis_idx - t  # days to crisis
            all_tte.append(tte)
            all_event.append(1)

            # z-score of each observable at time t
            for name in all_z:
                scores = all_scores[name]
                # Expanding z-score: use only past data
                past = scores[max(0, t-60):t]
                past_valid = past[np.isfinite(past)]
                if len(past_valid) < 10 or not np.isfinite(scores[t]):
                    all_z[name].append(np.nan)
                else:
                    z = (scores[t] - np.mean(past_valid)) / (np.std(past_valid) + 1e-8)
                    all_z[name].append(z)

    all_tte = np.array(all_tte)
    all_event = np.array(all_event)

    print(f"\nTotal pre-crisis observations: {len(all_tte)}")
    print(f"Time-to-crisis range: {all_tte.min():.0f} to {all_tte.max():.0f} days")

    # Cox-like: log(hazard) ~ beta * z_score
    # Proxy: correlation between z_score and log(1/tte) = -log(tte)
    # (higher hazard = shorter tte = negative log_tte)
    log_hazard_proxy = -np.log(all_tte + 1)

    print(f"\n{'Method':<25} {'beta (Pearson r)':>16} {'p-value':>10} {'Interpretation'}")
    print("-" * 70)

    cox_results = {}
    for name in all_z:
        z_arr = np.array(all_z[name])
        valid = np.isfinite(z_arr) & np.isfinite(log_hazard_proxy)
        if valid.sum() < 30:
            print(f"{name:<25} {'insufficient data':>16}")
            continue
        r, p = stats.pearsonr(z_arr[valid], log_hazard_proxy[valid])
        interp = "Higher z -> crisis sooner" if r > 0 else "Higher z -> crisis later"
        print(f"{name:<25} {r:>16.4f} {p:>10.4f}  {interp}")
        cox_results[name] = {'beta_proxy': float(r), 'p_value': float(p)}

    # Rank by |beta| (strength of hazard association)
    ranked = sorted(cox_results.items(), key=lambda x: abs(x[1]['beta_proxy']), reverse=True)
    print("\nRanking by |beta| (hazard ratio strength):")
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank}. {name}: beta={r['beta_proxy']:.4f}, p={r['p_value']:.4f}")

    # Verdict: does survival framing add insight vs Cohen's d?
    sig_methods = [name for name, r in cox_results.items() if r['p_value'] < 0.05]
    print(f"\nMethods with significant hazard association (p<0.05): {len(sig_methods)}")
    verdict = 'KEEP' if len(sig_methods) >= 2 else 'REJECT'

    return {
        'cox_results': cox_results,
        'n_observations': len(all_tte),
        'significant_methods': sig_methods,
        'verdict': verdict,
    }


# =============================================================================
# Q49: Granger Causality
# =============================================================================

def granger_causality_f_test(x, y, max_lag, min_periods=30):
    """
    Granger causality F-test: does x at lag(s) 1..max_lag predict y?

    Restricted model: y ~ y_{t-1..t-max_lag}
    Unrestricted model: y ~ y_{t-1..t-max_lag} + x_{t-1..t-max_lag}

    Returns F-statistic and p-value.
    """
    from sklearn.linear_model import LinearRegression

    T = len(y)
    if T < min_periods + max_lag:
        return np.nan, np.nan

    # Build lagged design matrices
    start = max_lag
    n = T - start

    Y = y[start:]
    Y_lags = np.column_stack([y[start - k:T - k] for k in range(1, max_lag + 1)])
    X_lags = np.column_stack([x[start - k:T - k] for k in range(1, max_lag + 1)])

    # Filter NaN
    valid = np.all(np.isfinite(Y_lags), axis=1) & np.isfinite(Y) & \
            np.all(np.isfinite(X_lags), axis=1)
    if valid.sum() < min_periods:
        return np.nan, np.nan

    Y_c = Y[valid]
    Y_lags_c = Y_lags[valid]
    X_lags_c = X_lags[valid]
    n_valid = valid.sum()

    # Restricted model
    lr_r = LinearRegression().fit(Y_lags_c, Y_c)
    resid_r = Y_c - lr_r.predict(Y_lags_c)
    rss_r = np.sum(resid_r ** 2)

    # Unrestricted model
    Z = np.hstack([Y_lags_c, X_lags_c])
    lr_u = LinearRegression().fit(Z, Y_c)
    resid_u = Y_c - lr_u.predict(Z)
    rss_u = np.sum(resid_u ** 2)

    k = max_lag  # number of restrictions
    df_u = n_valid - 2 * max_lag - 1
    if df_u <= 0 or rss_u < 1e-12:
        return np.nan, np.nan

    f_stat = ((rss_r - rss_u) / k) / (rss_u / df_u)
    p_val = 1 - stats.f.cdf(f_stat, k, df_u)
    return float(f_stat), float(p_val)


def run_q49(dates_enr, spy_close, all_scores):
    """
    Q49: Granger causality from observables to realized 20-day volatility.

    For each of top-5 observables, test: does observable z-score at lag t
    predict realized vol at t+k for k in {1, 5, 10, 20} days?

    Report F-statistic and p-value per observable, per lag.
    """
    print("\n" + "=" * 70)
    print("Q49: GRANGER CAUSALITY: OBSERVABLES -> REALIZED VOLATILITY")
    print("=" * 70)

    T = len(dates_enr)

    # Realized 20-day volatility from SPY (aligned with enriched dates)
    spy_log_ret = np.log(spy_close / spy_close.shift(1)).fillna(0).values
    # 20-day realized vol (annualized)
    realized_vol = np.full(T, np.nan)
    for t in range(20, T):
        realized_vol[t] = np.std(spy_log_ret[t-20:t]) * np.sqrt(252)

    # Z-score each observable's raw scores (expanding)
    def expanding_zscore(scores, min_window=60):
        z = np.full_like(scores, np.nan)
        for t in range(min_window, len(scores)):
            past = scores[max(0, t-252):t]
            past_valid = past[np.isfinite(past)]
            if len(past_valid) >= min_window // 2 and np.isfinite(scores[t]):
                z[t] = (scores[t] - np.mean(past_valid)) / (np.std(past_valid) + 1e-8)
        return z

    lags = [1, 5, 10, 20]

    print(f"\nTarget: Realized 20-day vol (n={np.isfinite(realized_vol).sum()} valid points)")
    print(f"Lags tested: {lags} days\n")

    results = {}
    for name in TOP5_CONFIGS:
        scores = all_scores.get(name)
        if scores is None:
            continue

        z = expanding_zscore(scores)
        results[name] = {}

        row_parts = [f"{name:<25}"]
        for lag in lags:
            # Build lead-lag series: x[t] predicts y[t+lag]
            x = z[:-lag] if lag > 0 else z
            y = realized_vol[lag:] if lag > 0 else realized_vol

            f_stat, p_val = granger_causality_f_test(x, y, max_lag=lag)
            results[name][lag] = {'f_stat': f_stat, 'p_value': p_val}

            sig = "*" if (p_val is not None and np.isfinite(p_val) and p_val < 0.05) else " "
            f_str = f"{f_stat:.2f}" if np.isfinite(f_stat) else "N/A"
            p_str = f"{p_val:.3f}" if np.isfinite(p_val) else "N/A"
            row_parts.append(f"F={f_str} p={p_str}{sig}")

        print(" | ".join(row_parts))

    print("\n(* = p < 0.05)")

    # Summary: which observables Granger-cause realized vol at any lag?
    print("\nGranger causality summary (p < 0.05 at any lag):")
    granger_positive = []
    for name, lag_results in results.items():
        sig_lags = [lag for lag, r in lag_results.items()
                    if r['f_stat'] is not None and np.isfinite(r['f_stat'])
                    and r['p_value'] < 0.05]
        if sig_lags:
            granger_positive.append(name)
            print(f"  {name}: significant at lags {sig_lags}")
        else:
            print(f"  {name}: not significant at any lag")

    verdict = 'KEEP' if len(granger_positive) >= 2 else 'REJECT'
    return {
        'results': results,
        'granger_positive': granger_positive,
        'verdict': verdict,
    }


# =============================================================================
# Q50: Continuous Crisis Intensity vs Binary Labels
# =============================================================================

def run_q50(dates_enr, spy_close, all_scores):
    """
    Q50: Continuous crisis intensity (realized vol) vs binary crisis labels.

    Current approach: binary label (crisis window = 1).
    Test: use realized 20-day vol as continuous intensity proxy.
    Metric: Spearman correlation between observable z-score and realized vol.

    Compare rankings:
    - Spearman rho ranking (continuous intensity)
    - Cohen's d ranking (binary label)
    """
    print("\n" + "=" * 70)
    print("Q50: CONTINUOUS CRISIS INTENSITY vs BINARY LABELS")
    print("=" * 70)

    T = len(dates_enr)

    # Realized 20-day volatility
    spy_log_ret = np.log(spy_close / spy_close.shift(1)).fillna(0).values
    realized_vol = np.full(T, np.nan)
    for t in range(20, T):
        realized_vol[t] = np.std(spy_log_ret[t-20:t]) * np.sqrt(252)

    # Binary crisis mask (all defined crises, not just 4)
    crisis_mask = np.zeros(T, dtype=bool)
    for ck, cinfo in ALL_CRISES.items():
        cs = pd.Timestamp(cinfo['start'])
        ce = pd.Timestamp(cinfo['end'])
        for i, d in enumerate(dates_enr):
            if cs <= d <= ce:
                crisis_mask[i] = True

    print(f"\nCrisis days: {crisis_mask.sum()}, Normal days: {(~crisis_mask).sum()}")
    print(f"Realized vol range: {np.nanmin(realized_vol):.3f} to {np.nanmax(realized_vol):.3f}")

    # Expanding z-score
    def expanding_zscore(scores, min_window=60):
        z = np.full_like(scores, np.nan)
        for t in range(min_window, len(scores)):
            past = scores[max(0, t-252):t]
            past_valid = past[np.isfinite(past)]
            if len(past_valid) >= 30 and np.isfinite(scores[t]):
                z[t] = (scores[t] - np.mean(past_valid)) / (np.std(past_valid) + 1e-8)
        return z

    print(f"\n{'Method':<25} {'Spearman rho':>13} {'p-value':>10} "
          f"{'rho rank':>9} {'d rank':>7} {'rank diff':>10}")
    print("-" * 80)

    spearman_results = {}
    for name in TOP5_CONFIGS:
        scores = all_scores.get(name)
        if scores is None:
            continue
        z = expanding_zscore(scores)
        valid = np.isfinite(z) & np.isfinite(realized_vol)
        if valid.sum() < 100:
            continue
        rho, p_val = stats.spearmanr(z[valid], realized_vol[valid])
        spearman_results[name] = {
            'rho': float(rho),
            'p_value': float(p_val),
            'cohens_d_rank': TOP5_CONFIGS[name]['cohens_d_rank'],
            'cohens_d': TOP5_CONFIGS[name]['cohens_d'],
        }

    # Rank by |rho|
    valid_methods = [n for n in spearman_results if np.isfinite(spearman_results[n]['rho'])]
    rho_rank_order = sorted(valid_methods,
                            key=lambda n: abs(spearman_results[n]['rho']), reverse=True)

    for rho_rank, name in enumerate(rho_rank_order, 1):
        r = spearman_results[name]
        d_rank = r['cohens_d_rank']
        rank_diff = rho_rank - d_rank
        sig = "*" if r['p_value'] < 0.05 else " "
        diff_str = f"{rank_diff:+d}"
        print(f"{name:<25} {r['rho']:>13.4f}{sig} {r['p_value']:>10.4f} "
              f"{rho_rank:>9} {d_rank:>7} {diff_str:>10}")

    print("\n(* = p < 0.05)")

    # Kendall tau between rho-rank and d-rank
    if len(valid_methods) >= 3:
        rho_ranks_arr = np.array([
            next(i+1 for i, n in enumerate(rho_rank_order) if n == name)
            for name in valid_methods
        ])
        d_ranks_arr = np.array([spearman_results[name]['cohens_d_rank']
                                 for name in valid_methods])
        tau, tau_p = stats.kendalltau(rho_ranks_arr, d_ranks_arr)
        print(f"\nKendall tau (Spearman rank vs Cohen's d rank): "
              f"tau={tau:.3f}, p={tau_p:.3f}")
    else:
        tau, tau_p = np.nan, np.nan
        print("Insufficient methods for Kendall tau")

    # Also compute Cohen's d for crisis vs non-crisis (binary)
    # and compare to Spearman
    print("\nBinary vs Continuous comparison for each method:")
    print(f"{'Method':<25} {'Cohen d':>9} {'Spearman rho':>13} {'Consistent?':>12}")
    for name in valid_methods:
        scores = all_scores.get(name)
        if scores is None:
            continue
        z = expanding_zscore(scores)
        valid = np.isfinite(z)
        crisis_z = z[valid & crisis_mask[:len(z)]]
        normal_z = z[valid & ~crisis_mask[:len(z)]]
        if len(crisis_z) > 5 and len(normal_z) > 5:
            d_val, _, _ = compute_cohens_d_with_ci(
                crisis_z, normal_z, n_bootstrap=1000
            )
        else:
            d_val = np.nan
        rho = spearman_results[name]['rho']
        # Consistent means both agree: higher scores during crisis
        consistent = "YES" if d_val > 0 and rho > 0 else "NO/MIXED"
        print(f"{name:<25} {d_val:>9.3f} {rho:>13.4f}  {consistent:>12}")

    sig_spearman = [n for n in valid_methods if spearman_results[n]['p_value'] < 0.05]
    verdict = 'KEEP' if len(sig_spearman) >= 2 and abs(tau) < 0.8 else 'REJECT'

    return {
        'spearman_results': spearman_results,
        'rho_ranking': rho_rank_order,
        'kendall_tau': tau,
        'kendall_p': tau_p,
        'verdict': verdict,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Q46-Q50: STATISTICAL/EVALUATION METHODOLOGY INVESTIGATION")
    print("=" * 70)

    # Load shared data once
    X_enr, dates_enr, spy_close, all_scores = load_shared_data()

    # Run all questions
    results = {}

    results['Q46'] = run_q46(dates_enr, all_scores)
    results['Q47'] = run_q47(dates_enr, all_scores)
    results['Q48'] = run_q48(dates_enr, spy_close, all_scores)
    results['Q49'] = run_q49(dates_enr, spy_close, all_scores)
    results['Q50'] = run_q50(dates_enr, spy_close, all_scores)

    # ==========================================================================
    # Final summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Q46-Q50")
    print("=" * 70)

    print("""
Q46: Detection Delay vs Cohen's d
- Key finding: see above table for lead-time rank vs Cohen's d rank
- Evidence: Kendall tau and per-crisis lead times
- Verdict: """, results['Q46']['verdict'])

    print("""
Q47: Conformal Prediction Coverage
- Key finding: conformal threshold calibration on normal days
- Evidence: normal coverage vs target 0.90, crisis detection power
- Verdict: """, results['Q47']['verdict'])

    print("""
Q48: Survival Analysis (Cox-like)
- Key finding: which observables have strongest hazard association
- Evidence: Pearson beta proxy and p-values
- Verdict: """, results['Q48']['verdict'])

    print("""
Q49: Granger Causality
- Key finding: observables that Granger-cause realized vol
- Evidence: F-stats and p-values at lags 1,5,10,20
- Granger-positive methods: """, results['Q49']['granger_positive'])
    print("- Verdict: ", results['Q49']['verdict'])

    print("""
Q50: Continuous Crisis Intensity
- Key finding: Spearman rho ranking vs Cohen's d ranking
- Evidence: Kendall tau between rankings
- Verdict: """, results['Q50']['verdict'])

    return results


if __name__ == '__main__':
    main()
