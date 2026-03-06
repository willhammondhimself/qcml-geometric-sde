"""
Phase 2b — Vol Dynamics Forecasting

Phase 2 found QCML features don't improve vol *level* forecasting over HAR
(DM p=0.36 for quantile, QCML *hurts* RF at p=0.006). rv_monthly alone captures
82% of RF importance. But Phase 1b confirmed the QCML signal IS real — it's
nonlinear, tail-concentrated, and operates at weekly scale.

Key insight: HAR predicts where vol IS (level). QCML detects regime *transitions*
(where vol is GOING). The natural target is vol **change**, not vol level.

This script tests:
  - Richer features (all 4 QCML z-scores + nonlinear interactions)
  - Vol dynamics targets (delta_rv, vol_spike)
  - 3 model families: Regime-Switching HAR, GBM Quantile, Geometric ARCH

Walk-forward: 15 annual test folds (2010-2024), expanding window from 2004.

Usage:
    python vol_forecasting/experiments/phase2b_vol_dynamics.py
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # vol_forecasting/experiments/
VOL_ROOT = SCRIPT_DIR.parent                          # vol_forecasting/
REPO_ROOT = VOL_ROOT.parent                           # qcml-geometric-sde/
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluation import holm_bonferroni_correction

# Reuse data pipeline from phase1b
from vol_forecasting.experiments.phase1b_targeted_analysis import (
    load_prices,
    build_realized_vol,
    extract_qcml_features,
    engineer_features,
    build_har_features,
)

# Reuse evaluation utilities from phase2
from vol_forecasting.experiments.phase2_tail_vol_forecasting import (
    diebold_mariano_test,
    bootstrap_metric,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RV_HORIZON = 5
INITIAL_TRAIN_YEARS = 5          # train 2004-2009, test from 2010
SPIKE_PCTILE = 95                # vol_spike threshold (expanding window)
RS_THRESHOLD_PCTILE = 75         # regime-switch: top 25% = stressed
RS_MIN_REGIME_OBS = 50           # minimum obs in stressed regime to fit separate model

# Feature tier definitions
HAR_DELTA_COLS = ['rv_daily', 'rv_weekly', 'rv_monthly',
                  'delta_rv_daily', 'delta_rv_weekly', 'delta_rv_monthly']

QCML_BASE_COLS = ['spectral_gap_ma20', 'berry_rate', 'qfi_det', 'multi_lag_fid']

QCML_INTERACT_COLS = ['berry_rate_x_rv_monthly', 'spectral_gap_ma20_x_rv_monthly',
                      'qfi_det_x_rv_monthly']

T1_COLS = HAR_DELTA_COLS
T2_COLS = HAR_DELTA_COLS + QCML_BASE_COLS
T3_COLS = HAR_DELTA_COLS + QCML_BASE_COLS + QCML_INTERACT_COLS

GBM_PARAMS = dict(n_estimators=300, max_depth=5, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.8, learning_rate=0.05,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)

N_BOOTSTRAP = 1000
DM_BANDWIDTH = 5

FIGURES_DIR = VOL_ROOT / 'figures'
RESULTS_DIR = VOL_ROOT / 'results'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRISIS_PERIODS = [
    ('2008-09-01', '2009-03-31', 'GFC'),
    ('2010-05-01', '2010-07-31', 'Flash Crash'),
    ('2011-08-01', '2011-10-31', 'EU Debt'),
    ('2015-08-01', '2015-09-30', 'China Deval'),
    ('2018-02-01', '2018-04-30', 'Volmageddon'),
    ('2020-02-01', '2020-04-30', 'COVID'),
    ('2022-01-01', '2022-10-31', 'Rate Hikes'),
]


# ============================================================================
# Step 1: Data pipeline
# ============================================================================

def prepare_dataset() -> pd.DataFrame:
    """Load prices, build RV, extract QCML, engineer all features, build targets.

    Returns aligned DataFrame with ~5100 rows, all feature tiers + targets.
    """
    logger.info("=" * 70)
    logger.info("Step 1: Preparing dataset")
    logger.info("=" * 70)

    prices = load_prices()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Build backward-looking and forward-looking RV
    rv_fwd = build_realized_vol(log_returns, [RV_HORIZON])
    har_df = build_har_features(log_returns)

    # Extract QCML features
    logger.info("Extracting QCML features...")
    qcml_df = extract_qcml_features(prices)
    eng_df = engineer_features(qcml_df)

    # Combine HAR + engineered QCML + forward RV
    combined = eng_df.join(har_df, how='inner').join(rv_fwd, how='inner')

    # --- Delta-HAR features (change in backward-looking vol) ---
    combined['delta_rv_daily'] = combined['rv_daily'].diff()
    combined['delta_rv_weekly'] = combined['rv_weekly'].diff()
    combined['delta_rv_monthly'] = combined['rv_monthly'].diff()

    # --- Interaction features (QCML x backward vol) ---
    combined['berry_rate_x_rv_monthly'] = combined['berry_rate'] * combined['rv_monthly']
    combined['spectral_gap_ma20_x_rv_monthly'] = combined['spectral_gap_ma20'] * combined['rv_monthly']
    combined['qfi_det_x_rv_monthly'] = combined['qfi_det'] * combined['rv_monthly']

    # --- Primary target: delta_rv_5d (vol change) ---
    rv_col = f'rv_{RV_HORIZON}d'
    combined['delta_rv_5d'] = combined[rv_col].shift(-RV_HORIZON) - combined[rv_col]

    # --- Secondary target: vol_spike (binary, expanding window threshold) ---
    # Expanding 95th percentile of rv_5d up to time t
    rv_series = combined[rv_col]
    spike_threshold = rv_series.expanding(min_periods=252).quantile(SPIKE_PCTILE / 100.0)
    combined['vol_spike'] = (combined[rv_col].shift(-RV_HORIZON) > spike_threshold).astype(float)

    # --- Keep only needed columns and drop NaN ---
    all_cols = list(set(T3_COLS + ['delta_rv_5d', 'vol_spike', rv_col,
                                    'spectral_gap_ma20', 'log_returns_col']))
    # Actually just keep everything that exists and drop NaN on the features we need
    keep_cols = T3_COLS + ['delta_rv_5d', 'vol_spike', rv_col]
    for c in keep_cols:
        if c not in combined.columns:
            logger.warning("Missing column: %s", c)

    existing_cols = [c for c in keep_cols if c in combined.columns]
    combined = combined[existing_cols].dropna()

    logger.info("Dataset: %d rows (%s to %s)",
                len(combined), combined.index[0].date(), combined.index[-1].date())
    logger.info("  Features: %s", T3_COLS)
    logger.info("  Targets: delta_rv_5d (mean=%.4f, std=%.4f), vol_spike (rate=%.3f)",
                combined['delta_rv_5d'].mean(), combined['delta_rv_5d'].std(),
                combined['vol_spike'].mean())

    return combined


# ============================================================================
# Step 2: Model implementations
# ============================================================================

def _fit_ols(X_train, y_train, X_test):
    """Fit OLS and return predictions."""
    X_tr = np.column_stack([X_train, np.ones(len(X_train))])
    X_te = np.column_stack([X_test, np.ones(len(X_test))])
    beta, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
    return X_te @ beta


def _fit_regime_switching_har(X_train, y_train, X_test,
                              regime_train, regime_test):
    """Fit separate OLS in calm/stressed regimes, predict by current regime.

    Args:
        X_train, y_train: Training data (T1 features only).
        X_test: Test data.
        regime_train: Boolean array (True = stressed) for training.
        regime_test: Boolean array for test.

    Returns:
        Predictions array, or NaN array if insufficient stressed obs.
    """
    n_stressed_train = np.sum(regime_train)
    n_calm_train = np.sum(~regime_train)

    if n_stressed_train < RS_MIN_REGIME_OBS or n_calm_train < RS_MIN_REGIME_OBS:
        return _fit_ols(X_train, y_train, X_test)

    # Fit separate OLS in each regime
    X_tr_c = np.column_stack([X_train[~regime_train], np.ones(n_calm_train)])
    X_tr_s = np.column_stack([X_train[regime_train], np.ones(n_stressed_train)])
    beta_calm, _, _, _ = np.linalg.lstsq(X_tr_c, y_train[~regime_train], rcond=None)
    beta_stressed, _, _, _ = np.linalg.lstsq(X_tr_s, y_train[regime_train], rcond=None)

    # Predict
    X_te = np.column_stack([X_test, np.ones(len(X_test))])
    preds = np.where(
        regime_test,
        X_te @ beta_stressed,
        X_te @ beta_calm,
    )
    return preds


def _fit_gbm_quantile(X_train, y_train, X_test, alpha=0.5):
    """Fit LightGBM quantile regression."""
    import lightgbm as lgb

    params = dict(GBM_PARAMS)
    params['objective'] = 'quantile'
    params['alpha'] = alpha

    dtrain = lgb.Dataset(X_train, label=y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.train(params, dtrain, num_boost_round=params.pop('n_estimators', 300))

    preds = model.predict(X_test)
    importances = model.feature_importance(importance_type='gain')
    return preds, importances


def _fit_logistic(X_train, y_train, X_test):
    """Fit logistic regression for binary vol_spike target."""
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        model.fit(X_train, y_train.astype(int))

    probs = model.predict_proba(X_test)[:, 1]
    return probs


def _geoarch_log_likelihood(params, returns, berry, spectral):
    """Negative log-likelihood for Geometric ARCH model.

    sigma2_t = omega + beta * sigma2_{t-1} + gamma1 * berry_{t-1} + gamma2 * spectral_{t-1}
    """
    omega, beta, gamma1, gamma2 = params
    T = len(returns)
    sigma2 = np.full(T, np.var(returns))

    for t in range(1, T):
        sigma2[t] = omega + beta * sigma2[t - 1] + gamma1 * berry[t - 1] + gamma2 * spectral[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)

    # Gaussian log-likelihood
    ll = -0.5 * np.sum(np.log(sigma2) + returns ** 2 / sigma2)
    return -ll  # negative for minimization


def _fit_geoarch(returns_train, berry_train, spectral_train):
    """Fit Geometric ARCH via MLE. Returns fitted params or None on failure."""
    var0 = np.var(returns_train)
    x0 = [var0 * 0.1, 0.8, 0.01, 0.01]
    bounds = [(1e-10, None), (1e-10, 0.999), (0.0, None), (0.0, None)]

    try:
        result = minimize(
            _geoarch_log_likelihood, x0,
            args=(returns_train, berry_train, spectral_train),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10},
        )
        if result.success:
            return result.x
    except Exception:
        pass

    # Fallback: try Nelder-Mead (no bounds, clip after)
    try:
        result = minimize(
            _geoarch_log_likelihood, x0,
            args=(returns_train, berry_train, spectral_train),
            method='Nelder-Mead',
            options={'maxiter': 1000},
        )
        params = result.x
        params[0] = max(params[0], 1e-10)
        params[1] = np.clip(params[1], 1e-10, 0.999)
        params[2] = max(params[2], 0.0)
        params[3] = max(params[3], 0.0)
        return params
    except Exception:
        return None


def _geoarch_predict_sigma(params, returns, berry, spectral):
    """Compute model-implied conditional volatility given params."""
    omega, beta, gamma1, gamma2 = params
    T = len(returns)
    sigma2 = np.full(T, np.var(returns))

    for t in range(1, T):
        sigma2[t] = omega + beta * sigma2[t - 1] + gamma1 * berry[t - 1] + gamma2 * spectral[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)

    return np.sqrt(sigma2)


def _fit_garch11(returns_train):
    """Fit standard GARCH(1,1) via arch library. Returns model result or None."""
    from arch import arch_model

    returns_pct = returns_train * 100
    try:
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp='off', show_warning=False)
        return result
    except Exception:
        return None


# ============================================================================
# Step 3: Walk-forward engine
# ============================================================================

def walk_forward_evaluation(df: pd.DataFrame) -> dict:
    """Run expanding-window walk-forward for all models.

    15 annual test folds (2010-2024), train starts at first year in data.
    """
    logger.info("=" * 70)
    logger.info("Step 3: Walk-forward evaluation")
    logger.info("=" * 70)

    rv_col = f'rv_{RV_HORIZON}d'
    log_returns_col = None  # We'll compute from rv_daily

    years = df.index.year
    unique_years = sorted(years.unique())
    data_start_year = unique_years[0]
    first_test_year = data_start_year + INITIAL_TRAIN_YEARS + 1
    test_years = [y for y in unique_years if y >= first_test_year]

    logger.info("Data years: %d-%d, test years: %s",
                data_start_year, unique_years[-1], test_years)

    # Model list for delta_rv_5d
    delta_models = [
        'HAR_delta',      # OLS on T1
        'RS_HAR',         # Regime-switching OLS on T1
        'GBM_HAR',        # GBM quantile (alpha=0.5) on T1
        'GBM_QCML',       # GBM quantile on T2
        'GBM_NL',         # GBM quantile on T3
        'GBM_NL_q90',     # GBM quantile (alpha=0.9) on T3
        'GBM_NL_q10',     # GBM quantile (alpha=0.1) on T3
        'GeoARCH',        # Geometric ARCH
        'GARCH11',        # Standard GARCH(1,1) baseline
    ]

    spike_models = ['Logistic_HAR', 'Logistic_HAR_QCML']

    # Storage
    fold_results = []
    oos_preds = {m: [] for m in delta_models + spike_models}
    oos_actuals_delta = []
    oos_actuals_spike = []
    oos_dates = []
    gbm_importances = {m: [] for m in ['GBM_HAR', 'GBM_QCML', 'GBM_NL']}
    geoarch_params_per_fold = []
    garch_params_per_fold = []
    regime_counts = []

    for test_year in test_years:
        train_mask = years < test_year
        test_mask = years == test_year

        n_train = int(np.sum(train_mask))
        n_test = int(np.sum(test_mask))
        if n_test == 0:
            continue

        logger.info("  Fold: Train [%d-%d] (%d), Test [%d] (%d)",
                    data_start_year, test_year - 1, n_train, test_year, n_test)

        # Extract arrays
        y_train_delta = df.loc[train_mask, 'delta_rv_5d'].values
        y_test_delta = df.loc[test_mask, 'delta_rv_5d'].values
        y_train_spike = df.loc[train_mask, 'vol_spike'].values
        y_test_spike = df.loc[test_mask, 'vol_spike'].values
        test_dates = df.index[test_mask]

        oos_actuals_delta.append(y_test_delta)
        oos_actuals_spike.append(y_test_spike)
        oos_dates.append(test_dates)

        fold_info = {'test_year': test_year, 'n_train': n_train, 'n_test': n_test, 'models': {}}

        # --- Feature arrays ---
        def _get_features(mask, cols):
            return df.loc[mask, cols].values

        X_train_t1 = _get_features(train_mask, T1_COLS)
        X_test_t1 = _get_features(test_mask, T1_COLS)
        X_train_t2 = _get_features(train_mask, T2_COLS)
        X_test_t2 = _get_features(test_mask, T2_COLS)
        X_train_t3 = _get_features(train_mask, T3_COLS)
        X_test_t3 = _get_features(test_mask, T3_COLS)

        # --- Regime indicator for RS-HAR ---
        sg_train = df.loc[train_mask, 'spectral_gap_ma20'].values
        sg_test = df.loc[test_mask, 'spectral_gap_ma20'].values
        rs_threshold = np.percentile(sg_train, RS_THRESHOLD_PCTILE)
        regime_train = sg_train > rs_threshold
        regime_test = sg_test > rs_threshold
        n_stressed = int(np.sum(regime_train))
        regime_counts.append({'test_year': test_year, 'n_stressed_train': n_stressed,
                              'n_calm_train': n_train - n_stressed,
                              'threshold': float(rs_threshold)})

        # --- Model 1: HAR-delta (OLS) ---
        preds = _fit_ols(X_train_t1, y_train_delta, X_test_t1)
        oos_preds['HAR_delta'].append(preds)
        fold_info['models']['HAR_delta'] = {'mse': float(np.mean((y_test_delta - preds) ** 2))}

        # --- Model 2: Regime-Switching HAR ---
        preds = _fit_regime_switching_har(X_train_t1, y_train_delta, X_test_t1,
                                          regime_train, regime_test)
        oos_preds['RS_HAR'].append(preds)
        fold_info['models']['RS_HAR'] = {'mse': float(np.mean((y_test_delta - preds) ** 2)),
                                          'n_stressed_train': n_stressed}

        # --- Model 3-7: GBM variants ---
        for model_name, X_tr, X_te, alpha in [
            ('GBM_HAR', X_train_t1, X_test_t1, 0.5),
            ('GBM_QCML', X_train_t2, X_test_t2, 0.5),
            ('GBM_NL', X_train_t3, X_test_t3, 0.5),
            ('GBM_NL_q90', X_train_t3, X_test_t3, 0.9),
            ('GBM_NL_q10', X_train_t3, X_test_t3, 0.1),
        ]:
            preds, importances = _fit_gbm_quantile(X_tr, y_train_delta, X_te, alpha=alpha)
            oos_preds[model_name].append(preds)
            fold_info['models'][model_name] = {'mse': float(np.mean((y_test_delta - preds) ** 2))}

            if model_name in gbm_importances:
                cols = T1_COLS if model_name == 'GBM_HAR' else (T2_COLS if model_name == 'GBM_QCML' else T3_COLS)
                imp = dict(zip(cols, importances.tolist()))
                imp['test_year'] = test_year
                gbm_importances[model_name].append(imp)

        # --- Model 8: Geometric ARCH ---
        # Use rv_daily as proxy for daily returns (annualized vol -> returns)
        rv_daily_train = df.loc[train_mask, 'rv_daily'].values / np.sqrt(252)
        rv_daily_test = df.loc[test_mask, 'rv_daily'].values / np.sqrt(252)
        berry_train_arr = df.loc[train_mask, 'berry_rate'].values
        berry_test_arr = df.loc[test_mask, 'berry_rate'].values
        sg_train_arr = df.loc[train_mask, 'spectral_gap_ma20'].values
        sg_test_arr = df.loc[test_mask, 'spectral_gap_ma20'].values

        geoarch_p = _fit_geoarch(rv_daily_train, berry_train_arr, sg_train_arr)
        if geoarch_p is not None:
            # Compute model-implied sigma on combined train+test to initialize
            all_returns = np.concatenate([rv_daily_train, rv_daily_test])
            all_berry = np.concatenate([berry_train_arr, berry_test_arr])
            all_sg = np.concatenate([sg_train_arr, sg_test_arr])
            sigma_all = _geoarch_predict_sigma(geoarch_p, all_returns, all_berry, all_sg)

            sigma_test = sigma_all[n_train:]
            # delta_sigma as proxy for delta_rv
            rv_test_model = sigma_test * np.sqrt(252)
            rv_train_end = sigma_all[n_train - 1] * np.sqrt(252)
            delta_sigma = np.diff(np.concatenate([[rv_train_end], rv_test_model]))

            oos_preds['GeoARCH'].append(delta_sigma)
            fold_info['models']['GeoARCH'] = {
                'mse': float(np.mean((y_test_delta - delta_sigma) ** 2)),
                'params': {'omega': float(geoarch_p[0]), 'beta': float(geoarch_p[1]),
                           'gamma1_berry': float(geoarch_p[2]), 'gamma2_spectral': float(geoarch_p[3])},
            }
            geoarch_params_per_fold.append({
                'test_year': test_year,
                'omega': float(geoarch_p[0]), 'beta': float(geoarch_p[1]),
                'gamma1_berry': float(geoarch_p[2]), 'gamma2_spectral': float(geoarch_p[3]),
            })
        else:
            oos_preds['GeoARCH'].append(np.zeros(n_test))
            fold_info['models']['GeoARCH'] = {'mse': np.nan, 'params': None}
            geoarch_params_per_fold.append({'test_year': test_year, 'failed': True})

        # --- Model 9: GARCH(1,1) baseline ---
        garch_result = _fit_garch11(rv_daily_train)
        if garch_result is not None:
            # Compute GARCH conditional vol on combined data
            from arch import arch_model
            all_returns_pct = np.concatenate([rv_daily_train, rv_daily_test]) * 100
            garch_full = arch_model(all_returns_pct, vol='Garch', p=1, q=1, dist='normal')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_full = garch_full.fit(disp='off', show_warning=False,
                                              starting_values=garch_result.params.values)
                cond_vol = res_full.conditional_volatility / 100 * np.sqrt(252)
                sigma_test_g = cond_vol[n_train:]
                rv_train_end_g = cond_vol[n_train - 1]
                delta_garch = np.diff(np.concatenate([[rv_train_end_g], sigma_test_g]))
            except Exception:
                delta_garch = np.zeros(n_test)

            oos_preds['GARCH11'].append(delta_garch)
            fold_info['models']['GARCH11'] = {
                'mse': float(np.mean((y_test_delta - delta_garch) ** 2)),
            }
            garch_params_per_fold.append({
                'test_year': test_year,
                'omega': float(garch_result.params.get('omega', np.nan)),
                'alpha': float(garch_result.params.get('alpha[1]', np.nan)),
                'beta': float(garch_result.params.get('beta[1]', np.nan)),
            })
        else:
            oos_preds['GARCH11'].append(np.zeros(n_test))
            fold_info['models']['GARCH11'] = {'mse': np.nan}
            garch_params_per_fold.append({'test_year': test_year, 'failed': True})

        # --- Spike models: Logistic regression ---
        probs = _fit_logistic(X_train_t1, y_train_spike, X_test_t1)
        oos_preds['Logistic_HAR'].append(probs)
        fold_info['models']['Logistic_HAR'] = {'spike_rate_test': float(np.mean(y_test_spike))}

        probs = _fit_logistic(X_train_t2, y_train_spike, X_test_t2)
        oos_preds['Logistic_HAR_QCML'].append(probs)
        fold_info['models']['Logistic_HAR_QCML'] = {'spike_rate_test': float(np.mean(y_test_spike))}

        fold_results.append(fold_info)

    # Concatenate OOS
    y_oos_delta = np.concatenate(oos_actuals_delta)
    y_oos_spike = np.concatenate(oos_actuals_spike)
    dates_oos = pd.DatetimeIndex(np.concatenate([d.values for d in oos_dates]))
    preds_oos = {m: np.concatenate(oos_preds[m]) for m in oos_preds}

    logger.info("Total OOS: %d observations (%s to %s)",
                len(y_oos_delta), dates_oos[0].date(), dates_oos[-1].date())

    return {
        'fold_results': fold_results,
        'y_oos_delta': y_oos_delta,
        'y_oos_spike': y_oos_spike,
        'dates_oos': dates_oos,
        'preds_oos': preds_oos,
        'gbm_importances': gbm_importances,
        'geoarch_params': geoarch_params_per_fold,
        'garch_params': garch_params_per_fold,
        'regime_counts': regime_counts,
        'test_years': test_years,
    }


# ============================================================================
# Step 4: Aggregate evaluation
# ============================================================================

def compute_aggregate_metrics(y_oos_delta, y_oos_spike, preds_oos, dates_oos, df) -> dict:
    """Compute aggregate metrics for all models."""
    logger.info("=" * 70)
    logger.info("Step 4: Aggregate evaluation")
    logger.info("=" * 70)

    results = {'delta_rv': {}, 'vol_spike': {}, 'dm_tests': {}}

    # --- Delta-RV models ---
    delta_models = ['HAR_delta', 'RS_HAR', 'GBM_HAR', 'GBM_QCML', 'GBM_NL', 'GeoARCH', 'GARCH11']
    naive_mse = float(np.mean(y_oos_delta ** 2))  # naive: predict delta=0

    for model_name in delta_models:
        preds = preds_oos[model_name]
        valid = ~(np.isnan(preds) | np.isnan(y_oos_delta))
        y_v = y_oos_delta[valid]
        p_v = preds[valid]

        mse = float(np.mean((y_v - p_v) ** 2))
        mae = float(np.mean(np.abs(y_v - p_v)))
        r2_vs_naive = 1 - mse / naive_mse if naive_mse > 0 else np.nan

        # Directional accuracy
        dir_acc = float(np.mean(np.sign(p_v) == np.sign(y_v))) if len(y_v) > 0 else np.nan

        # Pearson correlation
        corr = float(np.corrcoef(y_v, p_v)[0, 1]) if len(y_v) > 1 else np.nan

        results['delta_rv'][model_name] = {
            'mse': mse,
            'mae': mae,
            'r2_vs_naive': r2_vs_naive,
            'directional_accuracy': dir_acc,
            'pearson_r': corr,
            'n_valid': int(np.sum(valid)),
        }

        logger.info("  %s: MSE=%.6f, MAE=%.4f, R2_naive=%.4f, DirAcc=%.3f, r=%.3f",
                    model_name, mse, mae, r2_vs_naive, dir_acc, corr)

    results['delta_rv']['naive_mse'] = naive_mse

    # --- Pre-registered DM tests (3 comparisons) ---
    dm_pairs = [
        ('RS_HAR', 'HAR_delta', 'RS-HAR vs HAR-delta (QCML regime gate)'),
        ('GBM_QCML', 'GBM_HAR', 'GBM-QCML vs GBM-HAR (QCML features)'),
        ('GeoARCH', 'GARCH11', 'GeoARCH vs GARCH(1,1) (QCML in variance eq)'),
    ]

    dm_results = []
    for m_test, m_base, label in dm_pairs:
        p_test = preds_oos[m_test]
        p_base = preds_oos[m_base]
        valid = ~(np.isnan(p_test) | np.isnan(p_base) | np.isnan(y_oos_delta))
        loss_test = (y_oos_delta[valid] - p_test[valid]) ** 2
        loss_base = (y_oos_delta[valid] - p_base[valid]) ** 2
        dm = diebold_mariano_test(loss_base, loss_test, h=DM_BANDWIDTH)
        dm['label'] = label
        dm['model_test'] = m_test
        dm['model_base'] = m_base
        dm_results.append(dm)
        logger.info("  DM %s: stat=%.3f, p=%.4f", label, dm['DM'], dm['p'])

    # Holm-Bonferroni on 3 tests
    dm_pvals = np.array([d['p'] for d in dm_results])
    adj_p, rejected = holm_bonferroni_correction(dm_pvals)
    for i, dm in enumerate(dm_results):
        dm['p_adjusted'] = float(adj_p[i])
        dm['rejected'] = bool(rejected[i])

    results['dm_tests']['pre_registered'] = dm_results

    # --- Vol spike models ---
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    for model_name in ['Logistic_HAR', 'Logistic_HAR_QCML']:
        probs = preds_oos[model_name]
        valid = ~(np.isnan(probs) | np.isnan(y_oos_spike))
        y_v = y_oos_spike[valid].astype(int)
        p_v = probs[valid]

        if len(np.unique(y_v)) < 2:
            results['vol_spike'][model_name] = {'error': 'single class in OOS'}
            continue

        auc_roc = float(roc_auc_score(y_v, p_v))
        auc_pr = float(average_precision_score(y_v, p_v))
        brier = float(brier_score_loss(y_v, p_v))

        results['vol_spike'][model_name] = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'brier_score': brier,
            'n_spikes': int(np.sum(y_v)),
            'n_total': len(y_v),
        }
        logger.info("  %s: AUC-ROC=%.3f, AUC-PR=%.3f, Brier=%.4f",
                    model_name, auc_roc, auc_pr, brier)

    # DM on Brier scores
    p_har = preds_oos['Logistic_HAR']
    p_qcml = preds_oos['Logistic_HAR_QCML']
    valid = ~(np.isnan(p_har) | np.isnan(p_qcml) | np.isnan(y_oos_spike))
    loss_har = (y_oos_spike[valid] - p_har[valid]) ** 2
    loss_qcml = (y_oos_spike[valid] - p_qcml[valid]) ** 2
    dm_spike = diebold_mariano_test(loss_har, loss_qcml, h=DM_BANDWIDTH)
    dm_spike['label'] = 'Logistic HAR+QCML vs Logistic HAR (Brier)'
    results['dm_tests']['vol_spike'] = dm_spike
    logger.info("  DM Spike: stat=%.3f, p=%.4f", dm_spike['DM'], dm_spike['p'])

    return results


# ============================================================================
# Step 5: Regime-conditional analysis
# ============================================================================

def compute_regime_conditional(y_oos_delta, preds_oos, df, dates_oos) -> dict:
    """Split OOS into pre-transition vs continuation, report metrics in each."""
    logger.info("=" * 70)
    logger.info("Step 5: Regime-conditional analysis (exploratory)")
    logger.info("=" * 70)

    rv_col = f'rv_{RV_HORIZON}d'

    # Rolling 21d absolute change in rv
    rv_series = df[rv_col]
    delta_rv_21d = rv_series.diff(21).abs()
    threshold = delta_rv_21d.quantile(0.75)

    # Get delta_rv_21d for OOS dates
    delta_rv_oos = delta_rv_21d.reindex(dates_oos)
    pre_transition = (delta_rv_oos > threshold).values
    continuation = ~pre_transition

    # Handle NaN in the regime assignment
    valid_regime = ~np.isnan(delta_rv_oos.values)
    pre_transition = pre_transition & valid_regime
    continuation = continuation & valid_regime

    results = {
        'threshold_delta_rv_21d': float(threshold),
        'n_pre_transition': int(np.sum(pre_transition)),
        'n_continuation': int(np.sum(continuation)),
        'regimes': {},
    }

    delta_models = ['HAR_delta', 'RS_HAR', 'GBM_HAR', 'GBM_QCML', 'GBM_NL']
    naive_mse_full = float(np.mean(y_oos_delta ** 2))

    for regime_name, mask in [('pre_transition', pre_transition), ('continuation', continuation)]:
        y_r = y_oos_delta[mask]
        naive_mse = float(np.mean(y_r ** 2)) if len(y_r) > 0 else np.nan

        regime_results = {'n': len(y_r), 'naive_mse': naive_mse, 'models': {}}

        for model_name in delta_models:
            preds = preds_oos[model_name][mask]
            valid = ~np.isnan(preds)
            y_v = y_r[valid]
            p_v = preds[valid]

            mse = float(np.mean((y_v - p_v) ** 2))
            r2 = 1 - mse / naive_mse if naive_mse > 0 else np.nan
            regime_results['models'][model_name] = {'mse': mse, 'r2_vs_naive': r2}

        results['regimes'][regime_name] = regime_results
        logger.info("  %s (%d obs): naive_MSE=%.6f", regime_name, len(y_r), naive_mse)
        for m in delta_models:
            r2 = regime_results['models'][m]['r2_vs_naive']
            logger.info("    %s: R2=%.4f", m, r2)

    return results


# ============================================================================
# Step 6: Figures
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


def _add_crisis_shading(ax, dates):
    for start, end, _ in CRISIS_PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if s <= dates[-1] and e >= dates[0]:
            ax.axvspan(max(s, dates[0]), min(e, dates[-1]),
                       alpha=0.15, color='gray', zorder=0)


def plot_delta_rv_forecasts(y_oos, preds_oos, dates_oos):
    """Fig 1: Time series of actual vs predicted delta_rv_5d."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    # Rolling 21d average for readability
    window = 21
    actual_smooth = pd.Series(y_oos, index=dates_oos).rolling(window).mean()
    ax.plot(dates_oos, actual_smooth, color='black', lw=0.8, alpha=0.6, label='Actual (21d avg)')

    for model, color, ls in [('HAR_delta', 'steelblue', '-'),
                               ('RS_HAR', 'green', '--'),
                               ('GBM_NL', 'firebrick', '-')]:
        smooth = pd.Series(preds_oos[model], index=dates_oos).rolling(window).mean()
        ax.plot(dates_oos, smooth, color=color, lw=1, alpha=0.8, ls=ls, label=model)

    _add_crisis_shading(ax, dates_oos)
    ax.axhline(0, ls=':', color='gray', lw=0.5)
    ax.set_ylabel('delta_rv_5d (21d rolling avg)')
    ax.set_xlabel('Date')
    ax.set_title('Vol Change Forecasts: HAR-delta vs RS-HAR vs GBM-NL', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2b_delta_rv_forecasts.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2b_delta_rv_forecasts.{pdf,png}")


def plot_geoarch_params(geoarch_params):
    """Fig 2: GeoARCH gamma1 (berry) and gamma2 (spectral) across folds."""
    _set_pub_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    valid_folds = [p for p in geoarch_params if 'failed' not in p]
    if not valid_folds:
        logger.warning("No valid GeoARCH folds — skipping param plot")
        plt.close(fig)
        return

    years = [p['test_year'] for p in valid_folds]
    gamma1 = [p['gamma1_berry'] for p in valid_folds]
    gamma2 = [p['gamma2_spectral'] for p in valid_folds]

    axes[0].bar(years, gamma1, color='firebrick', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('gamma1 (Berry Rate)')
    axes[0].set_title('Geometric ARCH Parameters Across Walk-Forward Folds', fontsize=12)
    axes[0].axhline(0, ls=':', color='gray', lw=0.5)

    axes[1].bar(years, gamma2, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('gamma2 (Spectral Gap)')
    axes[1].set_xlabel('Test Year')
    axes[1].axhline(0, ls=':', color='gray', lw=0.5)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2b_geoarch_params.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2b_geoarch_params.{pdf,png}")


def plot_gbm_importance(gbm_importances):
    """Fig 3: Feature importances for GBM-HAR vs GBM-NL."""
    _set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model_name in zip(axes, ['GBM_HAR', 'GBM_NL']):
        imp_list = gbm_importances.get(model_name, [])
        if not imp_list:
            continue

        imp_df = pd.DataFrame(imp_list).drop(columns=['test_year'], errors='ignore')
        means = imp_df.mean().sort_values(ascending=True)
        stds = imp_df.std()

        def _feat_color(feat):
            if feat in QCML_BASE_COLS:
                return 'firebrick'
            elif feat in QCML_INTERACT_COLS:
                return 'darkorange'
            return 'steelblue'

        colors = [_feat_color(f) for f in means.index]
        ax.barh(range(len(means)), means.values,
                xerr=stds[means.index].values,
                color=colors, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels([f.replace('_', ' ').replace(' x ', '\nx ') for f in means.index],
                           fontsize=8)
        ax.set_xlabel('Mean Gain Importance')
        ax.set_title(model_name.replace('_', ' '), fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='HAR features'),
        Patch(facecolor='firebrick', label='QCML features'),
        Patch(facecolor='darkorange', label='Interactions'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.suptitle('GBM Feature Importances: HAR-only vs Full Nonlinear (mean +/- std)', fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2b_gbm_importance.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2b_gbm_importance.{pdf,png}")


def plot_regime_conditional(regime_results):
    """Fig 4: Grouped bars — model MSE in pre-transition vs continuation."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['HAR_delta', 'RS_HAR', 'GBM_HAR', 'GBM_QCML', 'GBM_NL']
    regimes = ['pre_transition', 'continuation']
    n_models = len(models)
    width = 0.35
    x = np.arange(n_models)

    colors = {'pre_transition': 'firebrick', 'continuation': 'steelblue'}
    labels = {'pre_transition': 'Pre-transition', 'continuation': 'Continuation'}

    for j, regime in enumerate(regimes):
        vals = []
        for m in models:
            v = regime_results['regimes'].get(regime, {}).get('models', {}).get(m, {}).get('mse', np.nan)
            vals.append(v)
        offset = (j - 0.5) * width
        ax.bar(x + offset, vals, width, label=labels[regime],
               color=colors[regime], alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add naive MSE lines
    for regime, ls in [('pre_transition', '--'), ('continuation', ':')]:
        naive = regime_results['regimes'].get(regime, {}).get('naive_mse', np.nan)
        if not np.isnan(naive):
            ax.axhline(naive, ls=ls, color=colors[regime], lw=1, alpha=0.5,
                       label=f'Naive (delta=0) {labels[regime]}')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax.set_ylabel('MSE')
    ax.set_title('Regime-Conditional MSE: Pre-transition vs Continuation\n'
                 '(Pre-transition = |delta_rv_21d| > 75th pctile)', fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2b_regime_conditional.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2b_regime_conditional.{pdf,png}")


# ============================================================================
# Step 7: Save results
# ============================================================================

def save_results(all_results: dict) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results['timestamp'] = ts

    path = RESULTS_DIR / f'phase2b_dynamics_{ts}.json'
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Phase 2b — Vol Dynamics Forecasting")
    logger.info("  Target: delta_rv_5d (vol change), vol_spike (binary)")
    logger.info("  Feature tiers: T1=%d, T2=%d, T3=%d", len(T1_COLS), len(T2_COLS), len(T3_COLS))
    logger.info("=" * 70)

    # --- Step 1: Data ---
    df = prepare_dataset()

    # --- Step 3: Walk-forward ---
    wf = walk_forward_evaluation(df)

    y_delta = wf['y_oos_delta']
    y_spike = wf['y_oos_spike']
    dates_oos = wf['dates_oos']
    preds_oos = wf['preds_oos']

    # --- Step 4: Aggregate evaluation ---
    agg = compute_aggregate_metrics(y_delta, y_spike, preds_oos, dates_oos, df)

    # --- Step 5: Regime-conditional ---
    regime = compute_regime_conditional(y_delta, preds_oos, df, dates_oos)

    # --- Step 6: Figures ---
    logger.info("=" * 70)
    logger.info("Step 6: Generating figures")
    logger.info("=" * 70)

    plot_delta_rv_forecasts(y_delta, preds_oos, dates_oos)
    plot_geoarch_params(wf['geoarch_params'])
    plot_gbm_importance(wf['gbm_importances'])
    plot_regime_conditional(regime)

    # --- Step 7: Save ---
    all_results = {
        'config': {
            'rv_horizon': RV_HORIZON,
            'initial_train_years': INITIAL_TRAIN_YEARS,
            'spike_pctile': SPIKE_PCTILE,
            'rs_threshold_pctile': RS_THRESHOLD_PCTILE,
            'feature_tiers': {
                'T1_HAR_delta': T1_COLS,
                'T2_plus_QCML': T2_COLS,
                'T3_plus_interactions': T3_COLS,
            },
            'gbm_params': GBM_PARAMS,
            'n_bootstrap': N_BOOTSTRAP,
            'dm_bandwidth': DM_BANDWIDTH,
        },
        'data_summary': {
            'n_total': len(df),
            'n_oos': len(y_delta),
            'date_range': [str(df.index[0].date()), str(df.index[-1].date())],
            'oos_date_range': [str(dates_oos[0].date()), str(dates_oos[-1].date())],
            'test_years': wf['test_years'],
            'delta_rv_stats': {
                'mean': float(np.mean(y_delta)),
                'std': float(np.std(y_delta)),
                'median': float(np.median(y_delta)),
            },
            'spike_rate': float(np.mean(y_spike)),
        },
        'fold_results': wf['fold_results'],
        'aggregate_metrics': agg,
        'regime_conditional': regime,
        'geoarch_params': wf['geoarch_params'],
        'garch_params': wf['garch_params'],
        'regime_counts': wf['regime_counts'],
    }

    result_path = save_results(all_results)

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2b SUMMARY")
    logger.info("=" * 70)

    logger.info("\nDelta-RV models (vs naive delta=0, MSE=%.6f):", agg['delta_rv']['naive_mse'])
    for m in ['HAR_delta', 'RS_HAR', 'GBM_HAR', 'GBM_QCML', 'GBM_NL', 'GeoARCH', 'GARCH11']:
        r = agg['delta_rv'].get(m, {})
        logger.info("  %s: MSE=%.6f, R2=%.4f, DirAcc=%.3f, r=%.3f",
                    m, r.get('mse', np.nan), r.get('r2_vs_naive', np.nan),
                    r.get('directional_accuracy', np.nan), r.get('pearson_r', np.nan))

    logger.info("\nPre-registered DM tests (Holm-Bonferroni corrected):")
    for dm in agg['dm_tests']['pre_registered']:
        sig = "SIGNIFICANT" if dm['rejected'] else "not sig."
        logger.info("  %s: DM=%.3f, p_raw=%.4f, p_adj=%.4f (%s)",
                    dm['label'], dm['DM'], dm['p'], dm['p_adjusted'], sig)

    logger.info("\nVol spike models:")
    for m in ['Logistic_HAR', 'Logistic_HAR_QCML']:
        r = agg['vol_spike'].get(m, {})
        logger.info("  %s: AUC-ROC=%.3f, AUC-PR=%.3f, Brier=%.4f",
                    m, r.get('auc_roc', np.nan), r.get('auc_pr', np.nan),
                    r.get('brier_score', np.nan))

    dm_spike = agg['dm_tests'].get('vol_spike', {})
    logger.info("  DM Spike: stat=%.3f, p=%.4f", dm_spike.get('DM', np.nan), dm_spike.get('p', np.nan))

    logger.info("\nRegime-conditional R2 (pre-transition vs continuation):")
    for regime_name in ['pre_transition', 'continuation']:
        rr = regime['regimes'].get(regime_name, {})
        logger.info("  %s (%d obs):", regime_name, rr.get('n', 0))
        for m in ['HAR_delta', 'RS_HAR', 'GBM_HAR', 'GBM_QCML', 'GBM_NL']:
            r2 = rr.get('models', {}).get(m, {}).get('r2_vs_naive', np.nan)
            logger.info("    %s: R2=%.4f", m, r2)

    logger.info("\nGeoARCH parameter stability:")
    valid_geo = [p for p in wf['geoarch_params'] if 'failed' not in p]
    if valid_geo:
        g1s = [p['gamma1_berry'] for p in valid_geo]
        g2s = [p['gamma2_spectral'] for p in valid_geo]
        logger.info("  gamma1_berry: mean=%.6f, std=%.6f, %d/%d folds converged",
                    np.mean(g1s), np.std(g1s), len(valid_geo), len(wf['geoarch_params']))
        logger.info("  gamma2_spectral: mean=%.6f, std=%.6f",
                    np.mean(g2s), np.std(g2s))

    logger.info("\nGBM feature importances (GBM-NL):")
    nl_imps = wf['gbm_importances'].get('GBM_NL', [])
    if nl_imps:
        imp_df = pd.DataFrame(nl_imps).drop(columns=['test_year'], errors='ignore')
        means = imp_df.mean().sort_values(ascending=False)
        for feat, val in means.items():
            tag = "QCML" if feat in QCML_BASE_COLS else ("INTERACT" if feat in QCML_INTERACT_COLS else "HAR")
            logger.info("  %s: %.1f (%s)", feat, val, tag)

    logger.info("\n  Results: %s", result_path)
    logger.info("  Figures: %s/phase2b_*.{pdf,png}", FIGURES_DIR)
    logger.info("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
