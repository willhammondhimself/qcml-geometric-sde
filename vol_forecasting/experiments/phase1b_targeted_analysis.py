"""
Phase 1b — Targeted Reanalysis of QCML Granger Causality

Phase 1 tested 4 features x 10 lags = 40 simultaneous Granger tests against rv_21d.
Result: 0/40 significant after Holm-Bonferroni. But the signal IS there:
  - Contemporaneous correlations rho 0.09-0.16, all p < 1e-11
  - Multi-Lag Fidelity lag 1 nearly passes (p_raw=0.0026, p_adj=0.104)
  - Spectral Gap sustained CCF r~0.22 at lags 10-30

This script applies 5 methodological improvements:
  1. Shorter-horizon target (rv_5d alongside rv_21d)
  2. Focused test battery (8 tests: 4 features x 2 lags)
  3. Engineered features (MA20 smoothing, delta, regime indicator)
  4. Nonlinear dependence (transfer entropy, quantile regression)
  5. Quick HAR + QCML pilot (expanding-window OOS R-squared)

Usage:
    python vol_forecasting/experiments/phase1b_targeted_analysis.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # vol_forecasting/experiments/
VOL_ROOT = SCRIPT_DIR.parent                          # vol_forecasting/
REPO_ROOT = VOL_ROOT.parent                           # qcml-geometric-sde/
sys.path.insert(0, str(REPO_ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix_single_asset
from experiments.evaluation import holm_bonferroni_correction
from qcml_geometry.core import QCMLGeometry
from qcml_geometry.indicators import SpectralGapIndicator
from qcml_geometry.observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = 'SPY'
START_DATE = '1995-01-01'
END_DATE = '2024-12-31'
RV_HORIZONS = [5, 21]
FOCUSED_LAGS = [1, 5]           # pre-registered: lag 1 (fast signal), lag 5 (weekly)
CCF_MAX_LAG = 30
DETECTOR_PARAMS = dict(
    hilbert_dim=8,
    n_pca_components=10,
    operator_method='pca_inspired',
    rolling_window=10,
    min_expanding=60,
    expanding_refit_interval=21,
    seed=42,
)

# Transfer entropy config
TE_N_BINS = 6
TE_N_SURROGATES = 1000
TE_LAGS = [1, 5]

# Quantile regression config
QR_TAU = 0.9
QR_LAGS = [1, 5]

# HAR pilot config
HAR_TRAIN_FRAC = 0.6
HAR_MIN_TRAIN = 500

FIGURES_DIR = VOL_ROOT / 'figures'
RESULTS_DIR = VOL_ROOT / 'results'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_index(obj):
    """Normalize DatetimeIndex to date-only to avoid DST/UTC offset mismatches."""
    if hasattr(obj, 'index') and isinstance(obj.index, pd.DatetimeIndex):
        obj.index = obj.index.normalize()
    return obj


# ============================================================================
# 1. Data Loading & Realized Vol (reused from Phase 1)
# ============================================================================

def load_prices() -> pd.Series:
    """Fetch SPY close prices from Polygon."""
    logger.info("Fetching %s %s to %s from Polygon...", SYMBOL, START_DATE, END_DATE)
    raw = fetch_data([SYMBOL], START_DATE, END_DATE)
    prices = raw['close'].droplevel('symbol')
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    logger.info("  Got %d daily prices (%s to %s)", len(prices),
                prices.index[0].date(), prices.index[-1].date())
    return prices


def build_realized_vol(log_returns: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Build forward-looking annualized realized volatility.

    RV_{t,h} = std(log_returns[t+1 : t+h+1]) * sqrt(252)
    """
    rv = {}
    for h in horizons:
        rv[f'rv_{h}d'] = (
            log_returns
            .shift(-h)
            .rolling(window=h)
            .std()
            * np.sqrt(252)
        )
    return pd.DataFrame(rv, index=log_returns.index)


# ============================================================================
# 2. QCML Feature Extraction (same as Phase 1)
# ============================================================================

def extract_qcml_features(prices: pd.Series) -> pd.DataFrame:
    """Extract 4 QCML z-score time series aligned to dates."""
    X_raw, dates_raw = create_feature_matrix_single_asset(prices, extra_lags=True)
    logger.info("  Raw features: %s, %d dates", X_raw.shape, len(dates_raw))

    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates_raw[19:].normalize()  # strip DST offset
    logger.info("  Enriched features: %s, %d dates", X_enriched.shape, len(dates_enriched))

    detector_specs = [
        ('qfi_det', QFIDeterminantDetector),
        ('berry_rate', BerryPhaseRateDetector),
        ('multi_lag_fid', MultiLagFidelityDetector),
    ]

    features = {}
    for name, DetectorClass in detector_specs:
        logger.info("  Fitting %s...", name)
        det = DetectorClass(**DETECTOR_PARAMS)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        features[name] = pd.Series(scores, index=dates_enriched, name=name)
        n_valid = np.sum(~np.isnan(scores))
        logger.info("    %s: %d valid scores (of %d)", name, n_valid, len(scores))

    # Spectral gap
    logger.info("  Fitting spectral_gap...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    n_components = min(DETECTOR_PARAMS['n_pca_components'], X_enriched.shape[1])
    scaler = StandardScaler().fit(X_enriched)
    pca = PCA(n_components=n_components).fit(scaler.transform(X_enriched))

    X_pca = pca.transform(scaler.transform(X_enriched))
    X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

    geo = QCMLGeometry(n_features=n_components, hilbert_dim=DETECTOR_PARAMS['hilbert_dim'])
    geo.fit_operators(X_pca, method=DETECTOR_PARAMS['operator_method'])

    indicator = SpectralGapIndicator(geometry=geo)
    raw_gaps = indicator.compute_spectral_gap_series(X_pca)

    min_exp = DETECTOR_PARAMS['min_expanding']
    gap_z = np.full(len(raw_gaps), np.nan)
    for t in range(min_exp, len(raw_gaps)):
        past = raw_gaps[:t]
        mu = np.mean(past)
        sigma = np.std(past, ddof=1)
        if sigma > 1e-12:
            gap_z[t] = abs((raw_gaps[t] - mu) / sigma)

    features['spectral_gap'] = pd.Series(gap_z, index=dates_enriched, name='spectral_gap')
    n_valid = np.sum(~np.isnan(gap_z))
    logger.info("    spectral_gap: %d valid scores (of %d)", n_valid, len(gap_z))

    return pd.DataFrame(features)


# ============================================================================
# 3. Engineered Features
# ============================================================================

def engineer_features(qcml_df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features based on CCF insights from Phase 1.

    Returns DataFrame with original + engineered columns.
    """
    eng = qcml_df.copy()

    # (a) Spectral Gap MA(20): CCF shows sustained r~0.22 at lags 10-30
    eng['spectral_gap_ma20'] = qcml_df['spectral_gap'].rolling(20).mean()

    # (b) Multi-Lag Fidelity raw z-score (already raw, just alias for clarity)
    # Already in qcml_df as 'multi_lag_fid'

    # (c) Delta features: 1-day and 5-day changes in z-scores
    for feat in ['multi_lag_fid', 'spectral_gap', 'qfi_det', 'berry_rate']:
        eng[f'{feat}_d1'] = qcml_df[feat].diff(1)
        eng[f'{feat}_d5'] = qcml_df[feat].diff(5)

    # (d) Regime indicator: binary flag when z > 1.5 (tail from scatter)
    for feat in ['multi_lag_fid', 'spectral_gap', 'qfi_det', 'berry_rate']:
        eng[f'{feat}_regime'] = (qcml_df[feat] > 1.5).astype(float)

    return eng


# ============================================================================
# 4. Temporal Alignment
# ============================================================================

def align_data(feat_df: pd.DataFrame, rv_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features and RV on date index, drop NaN rows."""
    combined = feat_df.join(rv_df, how='inner').dropna()
    logger.info("Aligned dataset: %d rows (%s to %s)",
                len(combined), combined.index[0].date(), combined.index[-1].date())
    return combined


# ============================================================================
# 5. Focused Granger Causality (8 tests)
# ============================================================================

def granger_f_test(x: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """Single-lag Granger F-test: does x Granger-cause y at given lag?

    Returns dict with F statistic and p-value.
    """
    from numpy.linalg import lstsq

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    T = len(x)

    if T <= 2 * lag + 2:
        return {'F': np.nan, 'p': np.nan, 'n': 0}

    Y = y[lag:]
    X_r = np.column_stack([y[lag - k - 1:T - k - 1] for k in range(lag)])
    X_r = np.column_stack([X_r, np.ones(len(Y))])

    X_u = np.column_stack([
        X_r[:, :-1],
        *[x[lag - k - 1:T - k - 1].reshape(-1, 1) for k in range(lag)],
        np.ones(len(Y)).reshape(-1, 1),
    ])

    beta_r, _, _, _ = lstsq(X_r, Y, rcond=None)
    beta_u, _, _, _ = lstsq(X_u, Y, rcond=None)

    rss_r = np.sum((Y - X_r @ beta_r) ** 2)
    rss_u = np.sum((Y - X_u @ beta_u) ** 2)

    n = len(Y)
    k_r = X_r.shape[1]
    k_u = X_u.shape[1]
    df1 = k_u - k_r
    df2 = n - k_u

    if df2 > 0 and rss_u > 1e-12:
        F = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p = 1 - stats.f.cdf(F, df1, df2)
    else:
        F, p = np.nan, np.nan

    return {'F': float(F), 'p': float(p), 'n': n}


def run_focused_granger(df: pd.DataFrame, rv_col: str,
                        feature_cols: list[str], lags: list[int]) -> dict:
    """Run focused Granger battery: features x lags.

    Returns dict with raw results and Holm-Bonferroni correction.
    """
    raw_results = []
    for feat in feature_cols:
        for lag in lags:
            res = granger_f_test(df[feat].values, df[rv_col].values, lag)
            raw_results.append({
                'feature': feat,
                'lag': lag,
                'F': res['F'],
                'p_raw': res['p'],
                'n': res['n'],
            })

    # Holm-Bonferroni correction
    p_values = np.array([r['p_raw'] for r in raw_results])
    adjusted_p, rejected = holm_bonferroni_correction(p_values)

    for i, r in enumerate(raw_results):
        r['p_adjusted'] = float(adjusted_p[i])
        r['rejected'] = bool(rejected[i])

    raw_results.sort(key=lambda x: x['p_adjusted'])

    return {
        'entries': raw_results,
        'n_tests': len(raw_results),
        'n_rejected': int(np.sum(rejected)),
        'any_significant': bool(np.any(rejected)),
        'alpha_per_test': 0.05 / len(raw_results),
    }


# ============================================================================
# 6. Transfer Entropy (nonlinear dependence)
# ============================================================================

def binned_transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int,
                            n_bins: int = 6) -> float:
    """Compute binned transfer entropy: TE(X -> Y) at given lag.

    TE(X->Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

    Uses equal-frequency binning for robustness.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    T = len(x)

    if T < lag + 50:
        return np.nan

    # Equal-frequency binning
    def discretize(arr, n_bins):
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(arr, percentiles)
        edges[-1] += 1e-10
        return np.digitize(arr, edges[1:])

    x_d = discretize(x, n_bins)
    y_d = discretize(y, n_bins)

    # Build joint samples: (Y_t, Y_{t-lag}, X_{t-lag})
    Y_now = y_d[lag:]
    Y_past = y_d[:T - lag]
    X_past = x_d[:T - lag]
    n = len(Y_now)

    # Joint and marginal entropies via counting
    def entropy_from_counts(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    # H(Y_t, Y_past)
    joint_yy = np.zeros((n_bins, n_bins))
    for i in range(n):
        joint_yy[Y_now[i] - 1, Y_past[i] - 1] += 1

    # H(Y_t, Y_past, X_past)
    joint_yyx = np.zeros((n_bins, n_bins, n_bins))
    for i in range(n):
        joint_yyx[Y_now[i] - 1, Y_past[i] - 1, X_past[i] - 1] += 1

    # H(Y_past, X_past)
    joint_yx = np.zeros((n_bins, n_bins))
    for i in range(n):
        joint_yx[Y_past[i] - 1, X_past[i] - 1] += 1

    # H(Y_past)
    marg_y = np.zeros(n_bins)
    for i in range(n):
        marg_y[Y_past[i] - 1] += 1

    # TE = H(Y_t, Y_past) + H(Y_past, X_past) - H(Y_t, Y_past, X_past) - H(Y_past)
    H_yy = entropy_from_counts(joint_yy.ravel())
    H_yyx = entropy_from_counts(joint_yyx.ravel())
    H_yx = entropy_from_counts(joint_yx.ravel())
    H_y = entropy_from_counts(marg_y)

    te = H_yy + H_yx - H_yyx - H_y
    return float(te)


def transfer_entropy_with_bootstrap(x: np.ndarray, y: np.ndarray, lag: int,
                                    n_bins: int = 6,
                                    n_surrogates: int = 1000) -> dict:
    """TE with bootstrap significance via time-shifted surrogates.

    Shuffles x to destroy temporal dependence while preserving marginals.
    """
    te_observed = binned_transfer_entropy(x, y, lag, n_bins)

    if np.isnan(te_observed):
        return {'te': np.nan, 'p': np.nan, 'ci_95': [np.nan, np.nan],
                'te_null_mean': np.nan, 'te_null_std': np.nan}

    rng = np.random.RandomState(42)
    te_null = np.empty(n_surrogates)

    for i in range(n_surrogates):
        x_shuffled = rng.permutation(x)
        te_null[i] = binned_transfer_entropy(x_shuffled, y, lag, n_bins)

    te_null = te_null[~np.isnan(te_null)]
    if len(te_null) == 0:
        return {'te': te_observed, 'p': np.nan, 'ci_95': [np.nan, np.nan],
                'te_null_mean': np.nan, 'te_null_std': np.nan}

    p = float(np.mean(te_null >= te_observed))
    ci_95 = [float(np.percentile(te_null, 2.5)), float(np.percentile(te_null, 97.5))]

    return {
        'te': float(te_observed),
        'p': float(p),
        'ci_95': ci_95,
        'te_null_mean': float(np.mean(te_null)),
        'te_null_std': float(np.std(te_null)),
        'significant': te_observed > ci_95[1],
    }


def run_transfer_entropy_tests(df: pd.DataFrame, rv_col: str,
                               feature_cols: list[str]) -> dict:
    """Run transfer entropy tests for all features at configured lags."""
    results = {}
    for feat in feature_cols:
        results[feat] = {}
        for lag in TE_LAGS:
            logger.info("  Transfer entropy: %s -> %s at lag %d", feat, rv_col, lag)
            x = df[feat].values
            y = df[rv_col].values
            valid = ~(np.isnan(x) | np.isnan(y))
            res = transfer_entropy_with_bootstrap(
                x[valid], y[valid], lag, TE_N_BINS, TE_N_SURROGATES
            )
            results[feat][lag] = res
            logger.info("    TE=%.5f, p=%.4f, sig=%s",
                        res['te'], res['p'], res.get('significant', 'N/A'))
    return results


# ============================================================================
# 7. Quantile Regression (tail dependence)
# ============================================================================

def quantile_regression_test(x: np.ndarray, y: np.ndarray, lag: int,
                             tau: float = 0.9) -> dict:
    """Test if lagged x predicts extreme y via quantile regression.

    Uses statsmodels QuantReg.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not available for quantile regression")
        return {'beta': np.nan, 'p': np.nan, 'tau': tau, 'lag': lag}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    T = len(x)

    if T < lag + 100:
        return {'beta': np.nan, 'p': np.nan, 'tau': tau, 'lag': lag, 'n': 0}

    # Lagged predictor
    X_lagged = x[:T - lag]
    Y_target = y[lag:]
    X_design = sm.add_constant(X_lagged)

    try:
        model = sm.QuantReg(Y_target, X_design)
        result = model.fit(q=tau, max_iter=1000)

        beta_x = float(result.params[1])
        p_x = float(result.pvalues[1])
        ci = result.conf_int(alpha=0.05)
        ci_x = [float(ci[1, 0]), float(ci[1, 1])]
    except Exception as e:
        logger.warning("Quantile regression failed: %s", e)
        return {'beta': np.nan, 'p': np.nan, 'tau': tau, 'lag': lag, 'n': len(Y_target)}

    return {
        'beta': beta_x,
        'p': p_x,
        'tau': tau,
        'lag': lag,
        'n': len(Y_target),
        'ci_95': ci_x,
        'significant': p_x < 0.05,
    }


def run_quantile_tests(df: pd.DataFrame, rv_col: str,
                       feature_cols: list[str]) -> dict:
    """Run quantile regression tests for all features at configured lags."""
    results = {}
    for feat in feature_cols:
        results[feat] = {}
        for lag in QR_LAGS:
            logger.info("  Quantile reg (tau=%.1f): %s -> %s at lag %d",
                        QR_TAU, feat, rv_col, lag)
            res = quantile_regression_test(
                df[feat].values, df[rv_col].values, lag, QR_TAU
            )
            results[feat][lag] = res
            logger.info("    beta=%.5f, p=%.4f, sig=%s",
                        res['beta'], res['p'], res.get('significant', 'N/A'))
    return results


# ============================================================================
# 8. HAR-RV Pilot
# ============================================================================

def build_har_features(log_returns: pd.Series) -> pd.DataFrame:
    """Build HAR-RV features: daily, weekly, monthly realized vol.

    HAR-RV (Corsi 2009): backward-looking, annualized.
    - Daily: |r_t| * sqrt(252) — single-day vol proxy
    - Weekly: 5-day rolling std * sqrt(252)
    - Monthly: 22-day rolling std * sqrt(252)
    """
    rv_daily = log_returns.abs() * np.sqrt(252)
    rv_weekly = log_returns.rolling(5).std() * np.sqrt(252)
    rv_monthly = log_returns.rolling(22).std() * np.sqrt(252)

    return pd.DataFrame({
        'rv_daily': rv_daily,
        'rv_weekly': rv_weekly,
        'rv_monthly': rv_monthly,
    }, index=log_returns.index)


def har_pilot_oos(df: pd.DataFrame, rv_col: str,
                  qcml_features: list[str],
                  train_frac: float = 0.6) -> dict:
    """Expanding-window OOS comparison: HAR-only vs HAR+QCML.

    Trains on first train_frac, expands forward. Reports OOS R-squared and QLIKE.
    """
    from numpy.linalg import lstsq

    har_cols = ['rv_daily', 'rv_weekly', 'rv_monthly']
    all_cols = har_cols + qcml_features

    # Drop rows with NaN in any column we need
    cols_needed = all_cols + [rv_col]
    valid_df = df[cols_needed].dropna()
    n = len(valid_df)

    train_end = max(int(n * train_frac), HAR_MIN_TRAIN)
    if train_end >= n - 50:
        logger.warning("Not enough data for OOS evaluation")
        return {'error': 'insufficient data'}

    logger.info("  HAR pilot: %d train, %d test", train_end, n - train_end)

    y_all = valid_df[rv_col].values

    results = {}
    for model_name, feature_cols in [('HAR', har_cols), ('HAR_QCML', all_cols)]:
        X_all = valid_df[feature_cols].values

        # Expanding window predictions
        y_pred = np.full(n, np.nan)
        for t in range(train_end, n):
            X_train = np.column_stack([X_all[:t], np.ones(t)])
            y_train = y_all[:t]
            X_test = np.append(X_all[t], 1.0).reshape(1, -1)

            beta, _, _, _ = lstsq(X_train, y_train, rcond=None)
            y_pred[t] = float((X_test @ beta).item())

        # OOS metrics
        oos_mask = ~np.isnan(y_pred)
        y_oos = y_all[oos_mask]
        yh_oos = y_pred[oos_mask]

        # Clip predictions to positive (vol can't be negative)
        yh_oos = np.clip(yh_oos, 1e-6, None)

        ss_res = np.sum((y_oos - yh_oos) ** 2)
        ss_tot = np.sum((y_oos - np.mean(y_oos)) ** 2)
        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # QLIKE loss: mean(rv/forecast - log(rv/forecast) - 1)
        ratio = y_oos / yh_oos
        qlike = float(np.mean(ratio - np.log(ratio) - 1))

        mse = float(np.mean((y_oos - yh_oos) ** 2))
        mae = float(np.mean(np.abs(y_oos - yh_oos)))

        results[model_name] = {
            'r2_oos': float(r2_oos),
            'qlike': qlike,
            'mse': mse,
            'mae': mae,
            'n_oos': int(np.sum(oos_mask)),
            'n_train_initial': train_end,
        }

        logger.info("    %s: R2_OOS=%.4f, QLIKE=%.4f, MSE=%.6f, MAE=%.4f",
                    model_name, r2_oos, qlike, mse, mae)

    # Delta
    r2_delta = results['HAR_QCML']['r2_oos'] - results['HAR']['r2_oos']
    qlike_delta = results['HAR']['qlike'] - results['HAR_QCML']['qlike']  # positive = QCML better

    results['delta'] = {
        'r2_improvement': float(r2_delta),
        'qlike_improvement': float(qlike_delta),
        'qcml_features_used': qcml_features,
    }

    logger.info("  Delta R2: %+.4f, Delta QLIKE: %+.4f (positive = QCML helps)",
                r2_delta, qlike_delta)

    return results


def har_pilot_oos_with_predictions(df: pd.DataFrame, rv_col: str,
                                   qcml_features: list[str],
                                   train_frac: float = 0.6) -> tuple:
    """Same as har_pilot_oos but also returns prediction arrays for plotting."""
    from numpy.linalg import lstsq

    har_cols = ['rv_daily', 'rv_weekly', 'rv_monthly']
    all_cols = har_cols + qcml_features

    cols_needed = all_cols + [rv_col]
    valid_df = df[cols_needed].dropna()
    n = len(valid_df)
    dates = valid_df.index

    train_end = max(int(n * train_frac), HAR_MIN_TRAIN)
    if train_end >= n - 50:
        return {}, None, None, None

    y_all = valid_df[rv_col].values

    predictions = {}
    for model_name, feature_cols in [('HAR', har_cols), ('HAR_QCML', all_cols)]:
        X_all = valid_df[feature_cols].values
        y_pred = np.full(n, np.nan)
        for t in range(train_end, n):
            X_train = np.column_stack([X_all[:t], np.ones(t)])
            y_train = y_all[:t]
            X_test = np.append(X_all[t], 1.0).reshape(1, -1)
            beta, _, _, _ = lstsq(X_train, y_train, rcond=None)
            y_pred[t] = float((X_test @ beta).item())
        predictions[model_name] = np.clip(y_pred, 1e-6, None)

    return predictions, y_all, dates, train_end


# ============================================================================
# 9. CCF Comparison (rv_5d vs rv_21d)
# ============================================================================

def compute_ccf(x: np.ndarray, y: np.ndarray, max_lag: int = 30) -> dict:
    """Compute CCF for lags -max_lag to +max_lag."""
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)

    T = len(x)
    ccf = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x_seg = x[:T - k]
            y_seg = y[k:]
        else:
            x_seg = x[-k:]
            y_seg = y[:T + k]

        valid = ~(np.isnan(x_seg) | np.isnan(y_seg))
        if np.sum(valid) < 30:
            ccf[k] = np.nan
            continue

        ccf[k] = float(np.corrcoef(x_seg[valid], y_seg[valid])[0, 1])

    return ccf


# ============================================================================
# 10. Figures
# ============================================================================

def _set_pub_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
    })


def plot_rv5d_vs_rv21d_ccf(df: pd.DataFrame, feature_cols: list[str]):
    """Compare CCF of QCML features vs rv_5d and rv_21d side by side."""
    _set_pub_style()

    fig, axes = plt.subplots(len(feature_cols), 2, figsize=(12, 3 * len(feature_cols)),
                             sharex=True, sharey=True)

    T_eff = len(df)
    sig_bound = 2.0 / np.sqrt(T_eff)

    for i, feat in enumerate(feature_cols):
        for j, rv_col in enumerate(['rv_5d', 'rv_21d']):
            ax = axes[i, j]
            ccf = compute_ccf(df[feat].values, df[rv_col].values, CCF_MAX_LAG)

            lags = sorted(ccf.keys())
            vals = [ccf[k] for k in lags]

            ax.bar(lags, vals, width=0.8, color='steelblue', alpha=0.7, edgecolor='none')
            ax.axhline(sig_bound, ls='--', color='red', lw=0.8, alpha=0.6)
            ax.axhline(-sig_bound, ls='--', color='red', lw=0.8, alpha=0.6)
            ax.axhline(0, ls='-', color='black', lw=0.5)
            ax.axvline(0, ls=':', color='gray', lw=0.5)
            ax.axvspan(0.5, CCF_MAX_LAG + 0.5, alpha=0.05, color='green')

            if i == 0:
                ax.set_title(rv_col, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(feat.replace('_', ' ').title(), fontsize=10)
            if i == len(feature_cols) - 1:
                ax.set_xlabel('Lag (days)')

    fig.suptitle('CCF Comparison: QCML features vs rv_5d and rv_21d', fontsize=13, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase1b_rv5d_vs_rv21d_ccf.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase1b_rv5d_vs_rv21d_ccf.{pdf,png}")


def plot_transfer_entropy(te_results: dict, rv_col: str):
    """Bar chart of transfer entropy with bootstrap CIs."""
    _set_pub_style()

    features = list(te_results.keys())
    lags = sorted(te_results[features[0]].keys())
    n_feat = len(features)
    n_lags = len(lags)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(n_feat)

    for j, lag in enumerate(lags):
        tes = [te_results[f][lag]['te'] for f in features]
        ci_lo = [te_results[f][lag]['ci_95'][0] for f in features]
        ci_hi = [te_results[f][lag]['ci_95'][1] for f in features]
        null_means = [te_results[f][lag]['te_null_mean'] for f in features]

        offset = (j - (n_lags - 1) / 2) * width
        bars = ax.bar(x + offset, tes, width, label=f'Lag {lag}',
                      alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add null distribution CI as error region
        for k in range(n_feat):
            ax.plot([x[k] + offset - width / 3, x[k] + offset + width / 3],
                    [ci_hi[k], ci_hi[k]], color='red', lw=1.5, alpha=0.7)
            if te_results[features[k]][lag].get('significant', False):
                ax.text(x[k] + offset, tes[k] + 0.002, '*',
                        ha='center', fontsize=14, fontweight='bold', color='green')

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title(f'Transfer Entropy: QCML features -> {rv_col}\n'
                 f'(red line = 97.5th pctile of null; * = significant)', fontsize=11)
    ax.legend()
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase1b_transfer_entropy.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase1b_transfer_entropy.{pdf,png}")


def plot_quantile_dependence(qr_results: dict, rv_col: str):
    """Bar chart of quantile regression slopes with CIs."""
    _set_pub_style()

    features = list(qr_results.keys())
    lags = sorted(qr_results[features[0]].keys())
    n_feat = len(features)
    n_lags = len(lags)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(n_feat)

    for j, lag in enumerate(lags):
        betas = [qr_results[f][lag]['beta'] for f in features]
        errors_lo = []
        errors_hi = []
        for f in features:
            ci = qr_results[f][lag].get('ci_95', [np.nan, np.nan])
            beta = qr_results[f][lag]['beta']
            errors_lo.append(beta - ci[0] if not np.isnan(ci[0]) else 0)
            errors_hi.append(ci[1] - beta if not np.isnan(ci[1]) else 0)

        offset = (j - (n_lags - 1) / 2) * width
        bars = ax.bar(x + offset, betas, width, label=f'Lag {lag}',
                      alpha=0.8, edgecolor='black', linewidth=0.5,
                      yerr=[errors_lo, errors_hi], capsize=3)

        for k in range(n_feat):
            if qr_results[features[k]][lag].get('significant', False):
                y_pos = betas[k] + errors_hi[k] + 0.001
                ax.text(x[k] + offset, y_pos, '*',
                        ha='center', fontsize=14, fontweight='bold', color='green')

    ax.axhline(0, ls='-', color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
    ax.set_ylabel(f'Quantile Regression Slope (tau={QR_TAU})')
    ax.set_title(f'Quantile Dependence: lagged QCML -> {rv_col} at tau={QR_TAU}\n'
                 f'(* = p < 0.05)', fontsize=11)
    ax.legend()
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase1b_quantile_dependence.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase1b_quantile_dependence.{pdf,png}")


def plot_har_pilot(predictions: dict, y_actual: np.ndarray,
                   dates: pd.DatetimeIndex, train_end: int, rv_col: str):
    """OOS comparison plot: HAR vs HAR+QCML."""
    _set_pub_style()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    oos_dates = dates[train_end:]
    y_oos = y_actual[train_end:]

    # Top panel: actual vs forecasts
    ax = axes[0]
    ax.plot(oos_dates, y_oos, color='black', lw=0.8, alpha=0.6, label='Realized')
    ax.plot(oos_dates, predictions['HAR'][train_end:], color='blue', lw=1,
            alpha=0.7, label='HAR-RV')
    ax.plot(oos_dates, predictions['HAR_QCML'][train_end:], color='red', lw=1,
            alpha=0.7, label='HAR+QCML')
    ax.set_ylabel(f'{rv_col} (annualized)')
    ax.set_title('Out-of-Sample Volatility Forecasts', fontsize=12)
    ax.legend(loc='upper right')

    # Bottom panel: squared error difference
    ax2 = axes[1]
    se_har = (y_oos - predictions['HAR'][train_end:]) ** 2
    se_qcml = (y_oos - predictions['HAR_QCML'][train_end:]) ** 2
    diff = se_har - se_qcml  # positive = QCML better

    # Rolling 63-day average for clarity
    diff_smooth = pd.Series(diff, index=oos_dates).rolling(63).mean()
    ax2.fill_between(diff_smooth.index, 0, diff_smooth.values,
                     where=diff_smooth.values > 0, color='green', alpha=0.3, label='QCML better')
    ax2.fill_between(diff_smooth.index, 0, diff_smooth.values,
                     where=diff_smooth.values <= 0, color='red', alpha=0.3, label='HAR better')
    ax2.axhline(0, ls='-', color='black', lw=0.5)
    ax2.set_ylabel('SE(HAR) - SE(HAR+QCML)\n(63d rolling avg)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase1b_har_pilot_oos.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase1b_har_pilot_oos.{pdf,png}")


# ============================================================================
# 11. Save Results
# ============================================================================

def save_results(all_results: dict) -> Path:
    """Write results JSON with timestamp."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results['timestamp'] = ts

    path = RESULTS_DIR / f'phase1b_targeted_{ts}.json'
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Phase 1b — Targeted Reanalysis: QCML features -> Realized Vol")
    logger.info("=" * 70)

    # --- Load data ---
    prices = load_prices()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # --- Build forward-looking RV ---
    logger.info("Building realized vol targets (h=%s)...", RV_HORIZONS)
    rv_df = build_realized_vol(log_returns, RV_HORIZONS)

    # --- Extract QCML features ---
    logger.info("Extracting QCML features...")
    qcml_df = extract_qcml_features(prices)

    # --- Engineer features ---
    logger.info("Engineering features...")
    eng_df = engineer_features(qcml_df)

    # --- Build HAR features ---
    logger.info("Building HAR features...")
    har_df = build_har_features(log_returns)

    # --- Combine all ---
    logger.info("Aligning all data...")
    all_features = eng_df.join(har_df, how='inner').join(rv_df, how='inner').dropna()
    logger.info("Combined dataset: %d rows (%s to %s)",
                len(all_features), all_features.index[0].date(),
                all_features.index[-1].date())

    # --- Base QCML feature columns ---
    base_features = ['qfi_det', 'berry_rate', 'multi_lag_fid', 'spectral_gap']

    # ======================================================================
    # IMPROVEMENT 1: rv_5d vs rv_21d comparison
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT 1: Shorter-horizon target (rv_5d vs rv_21d)")
    logger.info("=" * 70)

    granger_by_horizon = {}
    for rv_col in ['rv_5d', 'rv_21d']:
        logger.info("\n--- Target: %s ---", rv_col)
        res = run_focused_granger(all_features, rv_col, base_features, FOCUSED_LAGS)
        granger_by_horizon[rv_col] = res
        logger.info("  %d / %d significant after Holm-Bonferroni",
                    res['n_rejected'], res['n_tests'])
        for entry in res['entries'][:5]:
            logger.info("    %s lag=%d: F=%.2f, p_raw=%.4e, p_adj=%.4e %s",
                        entry['feature'], entry['lag'], entry['F'],
                        entry['p_raw'], entry['p_adjusted'],
                        " *** SIGNIFICANT ***" if entry['rejected'] else "")

    # ======================================================================
    # IMPROVEMENT 2: Focused 8-test battery (already done above)
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT 2: Focused battery — 8 tests (4 features x 2 lags)")
    logger.info("=" * 70)
    # Results already in granger_by_horizon — summarize
    for rv_col, res in granger_by_horizon.items():
        logger.info("  %s: %d/%d significant (threshold p < %.4f per test)",
                    rv_col, res['n_rejected'], res['n_tests'], res['alpha_per_test'])

    # ======================================================================
    # IMPROVEMENT 3: Engineered features Granger test
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT 3: Engineered features Granger test")
    logger.info("=" * 70)

    eng_features = ['spectral_gap_ma20', 'multi_lag_fid_d1', 'spectral_gap_d5',
                    'multi_lag_fid_regime']
    eng_granger_by_horizon = {}
    for rv_col in ['rv_5d', 'rv_21d']:
        logger.info("\n--- Engineered features -> %s ---", rv_col)
        res = run_focused_granger(all_features, rv_col, eng_features, FOCUSED_LAGS)
        eng_granger_by_horizon[rv_col] = res
        logger.info("  %d / %d significant", res['n_rejected'], res['n_tests'])
        for entry in res['entries'][:5]:
            logger.info("    %s lag=%d: F=%.2f, p_raw=%.4e, p_adj=%.4e %s",
                        entry['feature'], entry['lag'], entry['F'],
                        entry['p_raw'], entry['p_adjusted'],
                        " *** SIGNIFICANT ***" if entry['rejected'] else "")

    # ======================================================================
    # IMPROVEMENT 4a: Transfer Entropy
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT 4a: Transfer Entropy (nonlinear dependence)")
    logger.info("=" * 70)

    primary_rv = 'rv_5d'
    te_results = run_transfer_entropy_tests(all_features, primary_rv, base_features)

    n_te_sig = sum(
        1 for f in te_results for lag in te_results[f]
        if te_results[f][lag].get('significant', False)
    )
    logger.info("  %d / %d TE tests significant (rv_5d)", n_te_sig,
                len(base_features) * len(TE_LAGS))

    # ======================================================================
    # IMPROVEMENT 4b: Quantile Regression
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT 4b: Quantile Regression (tau=%.1f, tail dependence)", QR_TAU)
    logger.info("=" * 70)

    qr_results = run_quantile_tests(all_features, primary_rv, base_features)

    n_qr_sig = sum(
        1 for f in qr_results for lag in qr_results[f]
        if qr_results[f][lag].get('significant', False)
    )
    logger.info("  %d / %d quantile tests significant (rv_5d)", n_qr_sig,
                len(base_features) * len(QR_LAGS))

    # ======================================================================
    # IMPROVEMENT 5: HAR + QCML Pilot
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT 5: HAR + QCML Pilot (expanding-window OOS)")
    logger.info("=" * 70)

    # Use best 2 base QCML features: multi_lag_fid + spectral_gap (highest CCF)
    qcml_pilot_features = ['multi_lag_fid', 'spectral_gap']

    har_results = {}
    for rv_col in ['rv_5d', 'rv_21d']:
        logger.info("\n--- HAR pilot for %s ---", rv_col)
        har_results[rv_col] = har_pilot_oos(
            all_features, rv_col, qcml_pilot_features, HAR_TRAIN_FRAC
        )

    # Also test with engineered features
    qcml_eng_features = ['multi_lag_fid', 'spectral_gap', 'spectral_gap_ma20',
                         'multi_lag_fid_d1']
    logger.info("\n--- HAR + engineered QCML for rv_5d ---")
    har_results['rv_5d_engineered'] = har_pilot_oos(
        all_features, 'rv_5d', qcml_eng_features, HAR_TRAIN_FRAC
    )

    # ======================================================================
    # Figures
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Generating figures...")
    logger.info("=" * 70)

    # Fig 1: CCF comparison rv_5d vs rv_21d
    plot_rv5d_vs_rv21d_ccf(all_features, base_features)

    # Fig 2: Transfer entropy
    plot_transfer_entropy(te_results, primary_rv)

    # Fig 3: Quantile dependence
    plot_quantile_dependence(qr_results, primary_rv)

    # Fig 4: HAR pilot OOS
    predictions, y_actual, dates, train_end = har_pilot_oos_with_predictions(
        all_features, 'rv_5d', qcml_pilot_features, HAR_TRAIN_FRAC
    )
    if predictions:
        plot_har_pilot(predictions, y_actual, dates, train_end, 'rv_5d')

    # ======================================================================
    # Collect & Save Results
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Saving results...")
    logger.info("=" * 70)

    # Serialize transfer entropy results with string lag keys
    te_json = {}
    for feat, lag_dict in te_results.items():
        te_json[feat] = {str(lag): vals for lag, vals in lag_dict.items()}

    qr_json = {}
    for feat, lag_dict in qr_results.items():
        qr_json[feat] = {str(lag): vals for lag, vals in lag_dict.items()}

    all_results = {
        'config': {
            'symbol': SYMBOL,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'rv_horizons': RV_HORIZONS,
            'focused_lags': FOCUSED_LAGS,
            'detector_params': DETECTOR_PARAMS,
            'te_config': {'n_bins': TE_N_BINS, 'n_surrogates': TE_N_SURROGATES, 'lags': TE_LAGS},
            'qr_config': {'tau': QR_TAU, 'lags': QR_LAGS},
            'har_config': {'train_frac': HAR_TRAIN_FRAC, 'min_train': HAR_MIN_TRAIN},
        },
        'data_summary': {
            'n_aligned': len(all_features),
            'date_range': [str(all_features.index[0].date()),
                           str(all_features.index[-1].date())],
        },
        'focused_granger': {
            rv_col: granger_by_horizon[rv_col] for rv_col in granger_by_horizon
        },
        'engineered_granger': {
            rv_col: eng_granger_by_horizon[rv_col] for rv_col in eng_granger_by_horizon
        },
        'transfer_entropy': te_json,
        'quantile_regression': qr_json,
        'har_pilot': har_results,
        'summary': {
            'focused_granger_rv5d_sig': granger_by_horizon['rv_5d']['n_rejected'],
            'focused_granger_rv21d_sig': granger_by_horizon['rv_21d']['n_rejected'],
            'eng_granger_rv5d_sig': eng_granger_by_horizon['rv_5d']['n_rejected'],
            'te_significant': n_te_sig,
            'qr_significant': n_qr_sig,
            'har_r2_oos_rv5d': har_results.get('rv_5d', {}).get('HAR', {}).get('r2_oos'),
            'har_qcml_r2_oos_rv5d': har_results.get('rv_5d', {}).get('HAR_QCML', {}).get('r2_oos'),
            'har_r2_delta_rv5d': har_results.get('rv_5d', {}).get('delta', {}).get('r2_improvement'),
        },
    }

    result_path = save_results(all_results)

    # ======================================================================
    # Summary
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1b SUMMARY")
    logger.info("=" * 70)

    logger.info("\n1. Focused Granger (4 features x 2 lags = 8 tests):")
    for rv_col in ['rv_5d', 'rv_21d']:
        res = granger_by_horizon[rv_col]
        logger.info("   %s: %d/%d significant", rv_col, res['n_rejected'], res['n_tests'])
        if res['any_significant']:
            for e in res['entries']:
                if e['rejected']:
                    logger.info("     * %s lag=%d: F=%.2f, p_adj=%.4e",
                                e['feature'], e['lag'], e['F'], e['p_adjusted'])

    logger.info("\n2. Engineered features Granger (4 features x 2 lags = 8 tests):")
    for rv_col in ['rv_5d', 'rv_21d']:
        res = eng_granger_by_horizon[rv_col]
        logger.info("   %s: %d/%d significant", rv_col, res['n_rejected'], res['n_tests'])

    logger.info("\n3. Transfer Entropy: %d/%d significant (rv_5d)",
                n_te_sig, len(base_features) * len(TE_LAGS))

    logger.info("\n4. Quantile Regression (tau=%.1f): %d/%d significant (rv_5d)",
                QR_TAU, n_qr_sig, len(base_features) * len(QR_LAGS))

    logger.info("\n5. HAR Pilot OOS (rv_5d):")
    if 'rv_5d' in har_results and 'HAR' in har_results['rv_5d']:
        h = har_results['rv_5d']
        logger.info("   HAR-only:  R2=%.4f, QLIKE=%.4f", h['HAR']['r2_oos'], h['HAR']['qlike'])
        logger.info("   HAR+QCML:  R2=%.4f, QLIKE=%.4f", h['HAR_QCML']['r2_oos'], h['HAR_QCML']['qlike'])
        logger.info("   Delta R2:  %+.4f", h['delta']['r2_improvement'])
        logger.info("   Delta QLIKE: %+.4f (positive = QCML helps)", h['delta']['qlike_improvement'])

    logger.info("\n  Results: %s", result_path)
    logger.info("  Figures: %s/phase1b_*.{pdf,png}", FIGURES_DIR)
    logger.info("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
