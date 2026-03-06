"""
Phase 2 — Tail Volatility Forecasting

Phase 1b proved the QCML-vol signal is nonlinear and tail-concentrated:
- Linear HAR + QCML: Delta R^2 ~ 0 (linear augmentation is a dead end)
- Transfer entropy at lag 5: 4/4 significant (nonlinear weekly-scale signal)
- Quantile regression tau=0.9: 8/8 significant (all QCML features predict extreme vol)
- Spectral gap MA(20) Granger-causes rv_5d (p_adj=0.003)

Pivot: Instead of predicting mean vol, predict extreme vol (90th percentile).
This directly targets where the QCML signal lives.

Walk-forward design:
  Fold  1: Train [2004-2009], Test [2010]
  Fold  2: Train [2004-2010], Test [2011]
  ...
  Fold 14: Train [2004-2022], Test [2023]

4 models per fold:
  1. Quantile HAR       — QuantReg(tau=0.9) on HAR features only
  2. Quantile HAR+QCML  — QuantReg(tau=0.9) on HAR + QCML features
  3. RF HAR             — RandomForestRegressor on HAR features only
  4. RF HAR+QCML        — RandomForestRegressor on HAR + QCML features

Usage:
    python vol_forecasting/experiments/phase2_tail_vol_forecasting.py
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
    _normalize_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TAU = 0.9                        # quantile target (90th percentile)
RV_HORIZON = 5                   # 5-day forward RV (strongest nonlinear signal)
INITIAL_TRAIN_YEARS = 5          # first fold trains on ~2004-2009
HAR_COLS = ['rv_daily', 'rv_weekly', 'rv_monthly']
QCML_COLS = ['multi_lag_fid', 'spectral_gap', 'spectral_gap_ma20']
RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42)
N_BOOTSTRAP = 1000
DM_BANDWIDTH = 5                 # Newey-West bandwidth = forecast horizon
BLOCK_BOOTSTRAP_SIZE = 20        # block size for bootstrap CIs

FIGURES_DIR = VOL_ROOT / 'figures'
RESULTS_DIR = VOL_ROOT / 'results'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Crisis shading periods for plots
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
    """Load prices, build RV, extract QCML features, align everything.

    Returns:
        Single DataFrame with HAR + QCML + rv_5d target, ~5185 rows 2004-2024.
    """
    logger.info("=" * 70)
    logger.info("Step 1: Preparing dataset")
    logger.info("=" * 70)

    # Load prices
    prices = load_prices()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Build forward-looking RV (5-day only)
    rv_df = build_realized_vol(log_returns, [RV_HORIZON])

    # Extract QCML features
    logger.info("Extracting QCML features...")
    qcml_df = extract_qcml_features(prices)

    # Engineer features (adds spectral_gap_ma20, deltas, regime indicators)
    eng_df = engineer_features(qcml_df)

    # Build HAR features
    har_df = build_har_features(log_returns)

    # Combine and align
    combined = eng_df.join(har_df, how='inner').join(rv_df, how='inner')

    # Select only columns we need
    target_col = f'rv_{RV_HORIZON}d'
    keep_cols = HAR_COLS + QCML_COLS + [target_col]
    combined = combined[keep_cols].dropna()

    logger.info("Dataset: %d rows (%s to %s), columns: %s",
                len(combined), combined.index[0].date(), combined.index[-1].date(),
                list(combined.columns))

    return combined


# ============================================================================
# Step 2: Evaluation metrics
# ============================================================================

def pinball_loss(y_actual: np.ndarray, y_quantile: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Element-wise pinball (quantile) loss.

    L = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)

    Args:
        y_actual: Realized values, shape (n,).
        y_quantile: Quantile forecast, shape (n,).
        tau: Quantile level (0.9 for 90th percentile).

    Returns:
        Element-wise losses, shape (n,).
    """
    diff = y_actual - y_quantile
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


def exceedance_rate(y_actual: np.ndarray, y_quantile: np.ndarray) -> float:
    """Fraction of observations where actual exceeds the quantile forecast.

    For a well-calibrated tau=0.9 forecast, this should be ~10%.

    Args:
        y_actual: Realized values, shape (n,).
        y_quantile: Quantile forecast, shape (n,).

    Returns:
        Exceedance rate in [0, 1].
    """
    return float(np.mean(y_actual > y_quantile))


def christoffersen_test(y_actual: np.ndarray, y_quantile: np.ndarray,
                        alpha: float = 0.10) -> dict:
    """Christoffersen (1998) conditional coverage test.

    Tests both unconditional coverage (Kupiec LR) and independence (Markov LR).
    Joint LR_cc ~ chi2(2) under the null of correct conditional coverage.

    Args:
        y_actual: Realized values, shape (n,).
        y_quantile: Quantile forecast, shape (n,).
        alpha: Nominal exceedance probability (1 - tau). Default 0.10 for tau=0.9.

    Returns:
        Dict with LR_uc, LR_ind, LR_cc, p_uc, p_ind, p_cc.
    """
    # Hit sequence: 1 if actual > quantile (exceedance), 0 otherwise
    hits = (y_actual > y_quantile).astype(int)
    n = len(hits)
    n1 = int(np.sum(hits))
    n0 = n - n1

    if n1 == 0 or n0 == 0:
        return {'LR_uc': np.nan, 'LR_ind': np.nan, 'LR_cc': np.nan,
                'p_uc': np.nan, 'p_ind': np.nan, 'p_cc': np.nan,
                'exceedance_rate': n1 / n if n > 0 else np.nan}

    pi_hat = n1 / n

    # Kupiec LR (unconditional coverage)
    # H0: pi = alpha
    log_L0 = n1 * np.log(alpha) + n0 * np.log(1 - alpha)
    log_L1 = n1 * np.log(pi_hat) + n0 * np.log(1 - pi_hat)
    LR_uc = -2 * (log_L0 - log_L1)
    p_uc = 1 - stats.chi2.cdf(LR_uc, df=1)

    # Markov LR (independence)
    # Build 2x2 transition matrix
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        if hits[t - 1] == 0 and hits[t] == 0: n00 += 1
        elif hits[t - 1] == 0 and hits[t] == 1: n01 += 1
        elif hits[t - 1] == 1 and hits[t] == 0: n10 += 1
        elif hits[t - 1] == 1 and hits[t] == 1: n11 += 1

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    # Under independence: pi01 = pi11 = pi_hat
    # LR_ind tests H0: pi01 = pi11
    if pi01 > 0 and pi01 < 1 and pi11 > 0 and pi11 < 1:
        log_L_ind = (n01 * np.log(pi01) + n00 * np.log(1 - pi01) +
                     n11 * np.log(pi11) + n10 * np.log(1 - pi11))
        log_L_ind0 = (n01 + n11) * np.log(pi_hat) + (n00 + n10) * np.log(1 - pi_hat)
        LR_ind = -2 * (log_L_ind0 - log_L_ind)
    else:
        LR_ind = 0.0

    p_ind = 1 - stats.chi2.cdf(LR_ind, df=1)

    # Joint test
    LR_cc = LR_uc + LR_ind
    p_cc = 1 - stats.chi2.cdf(LR_cc, df=2)

    return {
        'LR_uc': float(LR_uc), 'LR_ind': float(LR_ind), 'LR_cc': float(LR_cc),
        'p_uc': float(p_uc), 'p_ind': float(p_ind), 'p_cc': float(p_cc),
        'exceedance_rate': float(pi_hat),
        'n_exceedances': int(n1), 'n_total': n,
    }


def diebold_mariano_test(loss1: np.ndarray, loss2: np.ndarray, h: int = DM_BANDWIDTH) -> dict:
    """Diebold-Mariano test for equal predictive accuracy.

    Uses Newey-West HAC standard errors to handle serial correlation
    from overlapping h-day forecasts.

    Args:
        loss1: Losses from model 1, shape (n,).
        loss2: Losses from model 2, shape (n,).
        h: Forecast horizon / Newey-West bandwidth.

    Returns:
        Dict with DM statistic, p-value, mean loss differential.
    """
    d = loss1 - loss2
    n = len(d)
    d_bar = np.mean(d)

    if n < 2 * h:
        return {'DM': np.nan, 'p': np.nan, 'mean_diff': float(d_bar)}

    # Newey-West HAC variance estimator
    gamma_0 = np.mean((d - d_bar) ** 2)
    nw_sum = 0.0
    for k in range(1, h):
        weight = 1 - k / h  # Bartlett kernel
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        nw_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + nw_sum) / n

    if var_d <= 0:
        return {'DM': np.nan, 'p': np.nan, 'mean_diff': float(d_bar)}

    DM = d_bar / np.sqrt(var_d)
    p = 2 * (1 - stats.norm.cdf(abs(DM)))  # two-sided

    return {
        'DM': float(DM),
        'p': float(p),
        'mean_diff': float(d_bar),
        'se': float(np.sqrt(var_d)),
    }


def bootstrap_metric(y_actual: np.ndarray, y_pred: np.ndarray,
                     metric_fn, n_boot: int = N_BOOTSTRAP,
                     block_size: int = BLOCK_BOOTSTRAP_SIZE) -> dict:
    """Block bootstrap confidence intervals for a forecast metric.

    Uses circular block bootstrap to handle serial correlation
    from overlapping forecast windows.

    Args:
        y_actual: Realized values, shape (n,).
        y_pred: Forecasted values, shape (n,).
        metric_fn: Callable(y_actual, y_pred) -> scalar.
        n_boot: Number of bootstrap samples.
        block_size: Block size for circular block bootstrap.

    Returns:
        Dict with point estimate, 95% CI, and bootstrap std.
    """
    n = len(y_actual)
    point = metric_fn(y_actual, y_pred)

    rng = np.random.RandomState(42)
    boot_vals = np.empty(n_boot)

    n_blocks = int(np.ceil(n / block_size))

    for b in range(n_boot):
        # Circular block bootstrap
        starts = rng.randint(0, n, size=n_blocks)
        indices = []
        for s in starts:
            indices.extend(range(s, s + block_size))
        indices = np.array(indices[:n]) % n  # wrap around

        boot_vals[b] = metric_fn(y_actual[indices], y_pred[indices])

    ci_lo = float(np.percentile(boot_vals, 2.5))
    ci_hi = float(np.percentile(boot_vals, 97.5))

    return {
        'point': float(point),
        'ci_95': [ci_lo, ci_hi],
        'std': float(np.std(boot_vals)),
    }


# ============================================================================
# Step 3: Walk-forward engine
# ============================================================================

def _fit_quantile_reg(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Fit quantile regression with solver fallbacks.

    Tries simplex, then Powell, then BFGS. Returns NaN on total failure.

    Args:
        X_train: Training features, shape (n_train, d).
        y_train: Training target, shape (n_train,).
        X_test: Test features, shape (n_test, d).
        tau: Quantile level.

    Returns:
        Predictions, shape (n_test,).
    """
    import statsmodels.api as sm

    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)

    model = sm.QuantReg(y_train, X_train_c)

    for method in ['simplex', 'powell', 'bfgs']:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(q=tau, max_iter=2000, method=method)
            preds = result.predict(X_test_c)
            return np.clip(preds, 1e-6, None)
        except Exception:
            continue

    logger.warning("QuantReg failed with all solvers, returning NaN")
    return np.full(len(X_test), np.nan)


def walk_forward_evaluation(df: pd.DataFrame) -> dict:
    """Run expanding-window walk-forward evaluation with 4 models.

    Fold structure: train starts at 2004, test years 2010-2023 (14 folds).

    Args:
        df: Aligned DataFrame with HAR + QCML + rv_5d columns.

    Returns:
        Dict with per-fold results and concatenated OOS predictions.
    """
    from sklearn.ensemble import RandomForestRegressor

    logger.info("=" * 70)
    logger.info("Step 3: Walk-forward evaluation")
    logger.info("=" * 70)

    target_col = f'rv_{RV_HORIZON}d'
    all_feature_cols = HAR_COLS + QCML_COLS

    # Determine fold boundaries by year
    years = df.index.year
    unique_years = sorted(years.unique())
    data_start_year = unique_years[0]

    # First test year: data_start_year + INITIAL_TRAIN_YEARS + 1
    first_test_year = data_start_year + INITIAL_TRAIN_YEARS + 1
    test_years = [y for y in unique_years if y >= first_test_year]

    logger.info("Data years: %d-%d, test years: %s",
                data_start_year, unique_years[-1], test_years)

    # Model specifications
    model_specs = {
        'QR_HAR': {'features': HAR_COLS, 'type': 'quantile'},
        'QR_HAR_QCML': {'features': all_feature_cols, 'type': 'quantile'},
        'RF_HAR': {'features': HAR_COLS, 'type': 'rf'},
        'RF_HAR_QCML': {'features': all_feature_cols, 'type': 'rf'},
    }

    # Storage
    fold_results = []
    oos_predictions = {m: [] for m in model_specs}
    oos_actuals = []
    oos_dates = []
    rf_importances = {m: [] for m in model_specs if 'RF' in m}

    for test_year in test_years:
        train_mask = years < test_year
        test_mask = years == test_year

        X_train_all = df.loc[train_mask, all_feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test_all = df.loc[test_mask, all_feature_cols].values
        y_test = df.loc[test_mask, target_col].values
        test_dates = df.index[test_mask]

        n_train = len(y_train)
        n_test = len(y_test)

        if n_test == 0:
            continue

        logger.info("  Fold: Train [%d-%d] (%d obs), Test [%d] (%d obs)",
                    data_start_year, test_year - 1, n_train, test_year, n_test)

        fold_info = {
            'test_year': test_year,
            'n_train': n_train,
            'n_test': n_test,
            'models': {},
        }

        oos_actuals.append(y_test)
        oos_dates.append(test_dates)

        for model_name, spec in model_specs.items():
            feat_cols = spec['features']
            feat_idx = [all_feature_cols.index(c) for c in feat_cols]

            X_tr = X_train_all[:, feat_idx]
            X_te = X_test_all[:, feat_idx]

            if spec['type'] == 'quantile':
                preds = _fit_quantile_reg(X_tr, y_train, X_te, TAU)

            elif spec['type'] == 'rf':
                rf = RandomForestRegressor(**RF_PARAMS)
                rf.fit(X_tr, y_train)
                preds = np.clip(rf.predict(X_te), 1e-6, None)

                # Collect feature importances
                imp = dict(zip(feat_cols, rf.feature_importances_))
                imp['test_year'] = test_year
                rf_importances[model_name].append(imp)

            oos_predictions[model_name].append(preds)

            # Per-fold metrics
            pl = float(np.mean(pinball_loss(y_test, preds, TAU)))
            exc = exceedance_rate(y_test, preds)
            fold_info['models'][model_name] = {
                'pinball_loss': pl,
                'exceedance_rate': exc,
                'n_nan': int(np.sum(np.isnan(preds))),
            }

        fold_results.append(fold_info)

    # Concatenate OOS results
    y_oos = np.concatenate(oos_actuals)
    dates_oos = pd.DatetimeIndex(np.concatenate([d.values for d in oos_dates]))
    preds_oos = {m: np.concatenate(oos_predictions[m]) for m in model_specs}

    logger.info("Total OOS: %d observations (%s to %s)",
                len(y_oos), dates_oos[0].date(), dates_oos[-1].date())

    return {
        'fold_results': fold_results,
        'y_oos': y_oos,
        'dates_oos': dates_oos,
        'preds_oos': preds_oos,
        'rf_importances': rf_importances,
        'test_years': test_years,
    }


# ============================================================================
# Step 4: Aggregate evaluation
# ============================================================================

def compute_aggregate_metrics(y_oos: np.ndarray, preds_oos: dict,
                              rf_importances: dict) -> dict:
    """Compute aggregate metrics across all OOS observations.

    Per model: pinball loss with bootstrap CI, exceedance rate, Christoffersen test.
    Cross-model: DM tests (quantile pair and RF pair), Holm-Bonferroni corrected.

    Args:
        y_oos: Realized OOS values, shape (n,).
        preds_oos: Dict of model_name -> OOS predictions, shape (n,).
        rf_importances: Dict of RF model_name -> list of per-fold importance dicts.

    Returns:
        Dict with per-model and cross-model results.
    """
    logger.info("=" * 70)
    logger.info("Step 4: Aggregate evaluation")
    logger.info("=" * 70)

    results = {'per_model': {}, 'cross_model': {}, 'rf_importance': {}}

    # --- Per-model metrics ---
    for model_name, preds in preds_oos.items():
        valid = ~np.isnan(preds)
        y_v = y_oos[valid]
        p_v = preds[valid]

        # Pinball loss with bootstrap CI
        def mean_pinball(y, p):
            return float(np.mean(pinball_loss(y, p, TAU)))

        pl_boot = bootstrap_metric(y_v, p_v, mean_pinball)

        # Exceedance rate
        exc = exceedance_rate(y_v, p_v)

        # Christoffersen test
        chris = christoffersen_test(y_v, p_v, alpha=1 - TAU)

        model_result = {
            'pinball_loss': pl_boot,
            'exceedance_rate': exc,
            'christoffersen': chris,
            'n_valid': int(np.sum(valid)),
        }

        # For RF models: also compute OOS R^2 and QLIKE
        if 'RF' in model_name:
            ss_res = np.sum((y_v - p_v) ** 2)
            ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            ratio = y_v / p_v
            qlike = float(np.mean(ratio - np.log(ratio) - 1))

            def compute_r2(y, p):
                ss_r = np.sum((y - p) ** 2)
                ss_t = np.sum((y - np.mean(y)) ** 2)
                return 1 - ss_r / ss_t if ss_t > 0 else np.nan

            def compute_qlike(y, p):
                r = y / np.clip(p, 1e-8, None)
                return float(np.mean(r - np.log(r) - 1))

            r2_boot = bootstrap_metric(y_v, p_v, compute_r2)
            qlike_boot = bootstrap_metric(y_v, p_v, compute_qlike)

            model_result['r2_oos'] = r2_boot
            model_result['qlike'] = qlike_boot

        results['per_model'][model_name] = model_result

        logger.info("  %s: pinball=%.4f [%.4f, %.4f], exc_rate=%.3f, Chris p_cc=%.3f",
                    model_name, pl_boot['point'], pl_boot['ci_95'][0], pl_boot['ci_95'][1],
                    exc, chris['p_cc'])
        if 'r2_oos' in model_result:
            logger.info("    R2_OOS=%.4f [%.4f, %.4f], QLIKE=%.4f",
                        model_result['r2_oos']['point'],
                        model_result['r2_oos']['ci_95'][0],
                        model_result['r2_oos']['ci_95'][1],
                        model_result['qlike']['point'])

    # --- Cross-model DM tests ---
    # Primary: QR_HAR vs QR_HAR_QCML
    # Secondary: RF_HAR vs RF_HAR_QCML
    dm_tests = []
    dm_pairs = [
        ('QR_HAR', 'QR_HAR_QCML', 'Quantile HAR vs Quantile HAR+QCML'),
        ('RF_HAR', 'RF_HAR_QCML', 'RF HAR vs RF HAR+QCML'),
    ]

    for m1, m2, label in dm_pairs:
        p1 = preds_oos[m1]
        p2 = preds_oos[m2]
        valid = ~(np.isnan(p1) | np.isnan(p2))

        loss1 = pinball_loss(y_oos[valid], p1[valid], TAU)
        loss2 = pinball_loss(y_oos[valid], p2[valid], TAU)

        dm = diebold_mariano_test(loss1, loss2, h=DM_BANDWIDTH)
        dm['label'] = label
        dm['model1'] = m1
        dm['model2'] = m2
        dm_tests.append(dm)

        logger.info("  DM %s: stat=%.3f, p=%.4f, mean_diff=%.5f",
                    label, dm['DM'], dm['p'], dm['mean_diff'])

    # Holm-Bonferroni correction on 2 DM tests
    dm_pvals = np.array([d['p'] for d in dm_tests])
    adj_p, rejected = holm_bonferroni_correction(dm_pvals)
    for i, dm in enumerate(dm_tests):
        dm['p_adjusted'] = float(adj_p[i])
        dm['rejected'] = bool(rejected[i])

    results['cross_model']['dm_tests'] = dm_tests
    results['cross_model']['holm_bonferroni_threshold'] = 0.025

    # --- RF feature importances ---
    for model_name, imp_list in rf_importances.items():
        if not imp_list:
            continue
        imp_df = pd.DataFrame(imp_list)
        imp_df = imp_df.drop(columns=['test_year'])
        mean_imp = imp_df.mean().to_dict()
        std_imp = imp_df.std().to_dict()
        results['rf_importance'][model_name] = {
            'mean': mean_imp,
            'std': std_imp,
            'n_folds': len(imp_list),
        }

    return results


# ============================================================================
# Step 5: Regime-conditional analysis
# ============================================================================

def compute_regime_conditional(y_oos: np.ndarray, preds_oos: dict,
                               df: pd.DataFrame, dates_oos: pd.DatetimeIndex) -> dict:
    """Split OOS into calm vs stressed regimes, compute metrics in each.

    Split based on rv_daily at median (backward-looking, no leakage).
    This is labeled as exploratory analysis.

    Args:
        y_oos: Realized OOS values, shape (n,).
        preds_oos: Dict of model_name -> OOS predictions.
        df: Full aligned DataFrame (for rv_daily).
        dates_oos: OOS dates.

    Returns:
        Dict with calm/stressed metrics and DM tests within each regime.
    """
    logger.info("=" * 70)
    logger.info("Step 5: Regime-conditional analysis (exploratory)")
    logger.info("=" * 70)

    # Get rv_daily for OOS dates
    rv_daily_oos = df.loc[dates_oos, 'rv_daily'].values
    median_rv = np.nanmedian(rv_daily_oos)

    calm_mask = rv_daily_oos <= median_rv
    stressed_mask = rv_daily_oos > median_rv

    results = {'median_rv_daily': float(median_rv), 'regimes': {}}

    for regime_name, mask in [('calm', calm_mask), ('stressed', stressed_mask)]:
        y_r = y_oos[mask]
        n_r = len(y_r)

        regime_results = {'n': n_r, 'models': {}}

        for model_name, preds in preds_oos.items():
            p_r = preds[mask]
            valid = ~np.isnan(p_r)
            y_v = y_r[valid]
            p_v = p_r[valid]

            pl = float(np.mean(pinball_loss(y_v, p_v, TAU)))
            exc = exceedance_rate(y_v, p_v)
            regime_results['models'][model_name] = {
                'pinball_loss': pl,
                'exceedance_rate': exc,
                'n_valid': int(np.sum(valid)),
            }

        # DM test within regime: QR pair
        for m1, m2, label in [('QR_HAR', 'QR_HAR_QCML', 'Quantile'),
                               ('RF_HAR', 'RF_HAR_QCML', 'RF')]:
            p1 = preds_oos[m1][mask]
            p2 = preds_oos[m2][mask]
            valid = ~(np.isnan(p1) | np.isnan(p2))

            loss1 = pinball_loss(y_r[valid], p1[valid], TAU)
            loss2 = pinball_loss(y_r[valid], p2[valid], TAU)

            dm = diebold_mariano_test(loss1, loss2, h=DM_BANDWIDTH)
            regime_results[f'dm_{label.lower()}'] = dm

        results['regimes'][regime_name] = regime_results

        logger.info("  %s (%d obs): QR_HAR pinball=%.4f, QR_HAR_QCML pinball=%.4f",
                    regime_name, n_r,
                    regime_results['models']['QR_HAR']['pinball_loss'],
                    regime_results['models']['QR_HAR_QCML']['pinball_loss'])

    # Compute improvement ratios
    for regime_name in ['calm', 'stressed']:
        r = results['regimes'][regime_name]
        qr_base = r['models']['QR_HAR']['pinball_loss']
        qr_qcml = r['models']['QR_HAR_QCML']['pinball_loss']
        r['qr_improvement_pct'] = float((qr_base - qr_qcml) / qr_base * 100) if qr_base > 0 else 0
        rf_base = r['models']['RF_HAR']['pinball_loss']
        rf_qcml = r['models']['RF_HAR_QCML']['pinball_loss']
        r['rf_improvement_pct'] = float((rf_base - rf_qcml) / rf_base * 100) if rf_base > 0 else 0

    logger.info("  QR improvement: calm=%.1f%%, stressed=%.1f%%",
                results['regimes']['calm']['qr_improvement_pct'],
                results['regimes']['stressed']['qr_improvement_pct'])

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


def _add_crisis_shading(ax, dates_oos):
    """Add gray vertical bands for crisis periods."""
    for start, end, _ in CRISIS_PERIODS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if start_dt <= dates_oos[-1] and end_dt >= dates_oos[0]:
            ax.axvspan(max(start_dt, dates_oos[0]), min(end_dt, dates_oos[-1]),
                       alpha=0.15, color='gray', zorder=0)


def plot_quantile_forecasts(y_oos: np.ndarray, preds_oos: dict,
                            dates_oos: pd.DatetimeIndex):
    """Fig 1: Time series of actual rv_5d with 90th percentile bands."""
    _set_pub_style()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates_oos, y_oos, color='black', lw=0.6, alpha=0.5, label='Realized rv_5d')
    ax.plot(dates_oos, preds_oos['QR_HAR'], color='steelblue', lw=1, alpha=0.8,
            label='Q90 HAR')
    ax.plot(dates_oos, preds_oos['QR_HAR_QCML'], color='firebrick', lw=1, alpha=0.8,
            label='Q90 HAR+QCML')

    _add_crisis_shading(ax, dates_oos)

    ax.set_ylabel('Annualized Realized Vol (5-day)')
    ax.set_xlabel('Date')
    ax.set_title(f'90th Percentile Volatility Forecasts (Walk-Forward OOS)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2_quantile_forecasts.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2_quantile_forecasts.{pdf,png}")


def plot_rolling_pinball(y_oos: np.ndarray, preds_oos: dict,
                         dates_oos: pd.DatetimeIndex):
    """Fig 2: Rolling pinball loss and differential."""
    _set_pub_style()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [2, 1]})

    # Top: 63-day rolling pinball loss
    ax = axes[0]
    window = 63

    for model_name, color, ls in [('QR_HAR', 'steelblue', '-'),
                                   ('QR_HAR_QCML', 'firebrick', '-')]:
        losses = pinball_loss(y_oos, preds_oos[model_name], TAU)
        rolling = pd.Series(losses, index=dates_oos).rolling(window).mean()
        ax.plot(rolling.index, rolling.values, color=color, lw=1.2, ls=ls,
                label=model_name.replace('_', ' '))

    _add_crisis_shading(ax, dates_oos)
    ax.set_ylabel(f'Pinball Loss (tau={TAU}, {window}d rolling)')
    ax.set_title('Rolling Pinball Loss: HAR vs HAR+QCML Quantile Models', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)

    # Bottom: loss differential
    ax2 = axes[1]
    loss_har = pinball_loss(y_oos, preds_oos['QR_HAR'], TAU)
    loss_qcml = pinball_loss(y_oos, preds_oos['QR_HAR_QCML'], TAU)
    diff = loss_har - loss_qcml  # positive = QCML better

    diff_smooth = pd.Series(diff, index=dates_oos).rolling(window).mean()
    ax2.fill_between(diff_smooth.index, 0, diff_smooth.values,
                     where=diff_smooth.values > 0, color='green', alpha=0.3, label='QCML better')
    ax2.fill_between(diff_smooth.index, 0, diff_smooth.values,
                     where=diff_smooth.values <= 0, color='red', alpha=0.3, label='HAR better')
    ax2.axhline(0, ls='-', color='black', lw=0.5)
    _add_crisis_shading(ax2, dates_oos)
    ax2.set_ylabel(f'Loss Differential ({window}d avg)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2_rolling_pinball.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2_rolling_pinball.{pdf,png}")


def plot_rf_importance(rf_importances: dict):
    """Fig 3: Horizontal bar chart of RF feature importances."""
    _set_pub_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (model_name, imp_list) in zip(axes, rf_importances.items()):
        if not imp_list:
            continue

        imp_df = pd.DataFrame(imp_list).drop(columns=['test_year'])
        means = imp_df.mean().sort_values(ascending=True)
        stds = imp_df.std()

        colors = ['firebrick' if feat in QCML_COLS else 'steelblue' for feat in means.index]
        bars = ax.barh(range(len(means)), means.values, xerr=stds[means.index].values,
                       color=colors, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(means)))
        ax.set_yticklabels([f.replace('_', ' ') for f in means.index], fontsize=9)
        ax.set_xlabel('Mean Feature Importance')
        ax.set_title(model_name.replace('_', ' '), fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='HAR features'),
                       Patch(facecolor='firebrick', label='QCML features')]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.suptitle('Random Forest Feature Importances (mean +/- std across folds)', fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2_rf_importance.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2_rf_importance.{pdf,png}")


def plot_regime_conditional(regime_results: dict):
    """Fig 4: Grouped bars for calm vs stressed regime performance."""
    _set_pub_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['QR_HAR', 'QR_HAR_QCML', 'RF_HAR', 'RF_HAR_QCML']
    regimes = ['calm', 'stressed']
    n_models = len(models)
    n_regimes = len(regimes)

    x = np.arange(n_models)
    width = 0.35

    colors = {'calm': 'steelblue', 'stressed': 'firebrick'}

    for j, regime in enumerate(regimes):
        vals = [regime_results['regimes'][regime]['models'][m]['pinball_loss'] for m in models]
        offset = (j - (n_regimes - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=f'{regime.capitalize()}',
                      color=colors[regime], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax.set_ylabel(f'Mean Pinball Loss (tau={TAU})')
    ax.set_title('Regime-Conditional Pinball Loss\n(Calm: rv_daily <= median, Stressed: rv_daily > median)',
                 fontsize=11)
    ax.legend()

    # Annotate DM p-values
    for regime in regimes:
        r = regime_results['regimes'][regime]
        dm_qr = r.get('dm_quantile', {})
        if 'p' in dm_qr and not np.isnan(dm_qr['p']):
            # Position annotation between QR_HAR and QR_HAR_QCML
            y_max = max(r['models']['QR_HAR']['pinball_loss'],
                        r['models']['QR_HAR_QCML']['pinball_loss'])
            ax.text(0.5, y_max * 1.05,
                    f'DM p={dm_qr["p"]:.3f}',
                    ha='center', fontsize=8, style='italic',
                    color=colors[regime])

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'phase2_regime_conditional.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved: phase2_regime_conditional.{pdf,png}")


# ============================================================================
# Step 7: Save results & main
# ============================================================================

def save_results(all_results: dict) -> Path:
    """Write results JSON with timestamp."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results['timestamp'] = ts

    path = RESULTS_DIR / f'phase2_tail_{ts}.json'
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return path


def main():
    logger.info("=" * 70)
    logger.info("Phase 2 — Tail Volatility Forecasting")
    logger.info("  Quantile target: tau=%.1f (90th percentile)", TAU)
    logger.info("  RV horizon: %d days", RV_HORIZON)
    logger.info("=" * 70)

    # --- Step 1: Data pipeline ---
    df = prepare_dataset()

    # --- Step 3: Walk-forward evaluation ---
    wf = walk_forward_evaluation(df)

    y_oos = wf['y_oos']
    dates_oos = wf['dates_oos']
    preds_oos = wf['preds_oos']

    # --- Step 4: Aggregate evaluation ---
    agg = compute_aggregate_metrics(y_oos, preds_oos, wf['rf_importances'])

    # --- Step 5: Regime-conditional analysis ---
    regime = compute_regime_conditional(y_oos, preds_oos, df, dates_oos)

    # --- Step 6: Figures ---
    logger.info("=" * 70)
    logger.info("Step 6: Generating figures")
    logger.info("=" * 70)

    plot_quantile_forecasts(y_oos, preds_oos, dates_oos)
    plot_rolling_pinball(y_oos, preds_oos, dates_oos)
    plot_rf_importance(wf['rf_importances'])
    plot_regime_conditional(regime)

    # --- Step 7: Save results ---
    logger.info("=" * 70)
    logger.info("Step 7: Saving results")
    logger.info("=" * 70)

    all_results = {
        'config': {
            'tau': TAU,
            'rv_horizon': RV_HORIZON,
            'initial_train_years': INITIAL_TRAIN_YEARS,
            'har_cols': HAR_COLS,
            'qcml_cols': QCML_COLS,
            'rf_params': RF_PARAMS,
            'n_bootstrap': N_BOOTSTRAP,
            'dm_bandwidth': DM_BANDWIDTH,
            'block_bootstrap_size': BLOCK_BOOTSTRAP_SIZE,
        },
        'data_summary': {
            'n_total': len(df),
            'n_oos': len(y_oos),
            'date_range': [str(df.index[0].date()), str(df.index[-1].date())],
            'oos_date_range': [str(dates_oos[0].date()), str(dates_oos[-1].date())],
            'test_years': wf['test_years'],
        },
        'fold_results': wf['fold_results'],
        'aggregate_metrics': agg,
        'regime_conditional': regime,
    }

    result_path = save_results(all_results)

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2 SUMMARY")
    logger.info("=" * 70)

    logger.info("\nPer-model aggregate metrics:")
    for model_name, m in agg['per_model'].items():
        pl = m['pinball_loss']
        logger.info("  %s: pinball=%.4f [%.4f, %.4f], exc=%.3f",
                    model_name, pl['point'], pl['ci_95'][0], pl['ci_95'][1],
                    m['exceedance_rate'])
        chris = m['christoffersen']
        logger.info("    Christoffersen: p_uc=%.3f, p_ind=%.3f, p_cc=%.3f",
                    chris['p_uc'], chris['p_ind'], chris['p_cc'])
        if 'r2_oos' in m:
            logger.info("    R2_OOS=%.4f, QLIKE=%.4f",
                        m['r2_oos']['point'], m['qlike']['point'])

    logger.info("\nDiebold-Mariano tests (Holm-Bonferroni corrected):")
    for dm in agg['cross_model']['dm_tests']:
        sig = "SIGNIFICANT" if dm['rejected'] else "not sig."
        logger.info("  %s: DM=%.3f, p_raw=%.4f, p_adj=%.4f (%s)",
                    dm['label'], dm['DM'], dm['p'], dm['p_adjusted'], sig)

    logger.info("\nRegime-conditional improvement (QCML vs baseline):")
    for regime_name in ['calm', 'stressed']:
        r = regime['regimes'][regime_name]
        logger.info("  %s: QR improvement=%.1f%%, RF improvement=%.1f%%",
                    regime_name, r['qr_improvement_pct'], r['rf_improvement_pct'])

    logger.info("\nRF feature importances (HAR+QCML model):")
    if 'RF_HAR_QCML' in agg['rf_importance']:
        imp = agg['rf_importance']['RF_HAR_QCML']
        sorted_imp = sorted(imp['mean'].items(), key=lambda x: x[1], reverse=True)
        for feat, val in sorted_imp:
            is_qcml = "QCML" if feat in QCML_COLS else "HAR"
            logger.info("  %s: %.4f +/- %.4f (%s)", feat, val,
                        imp['std'].get(feat, 0), is_qcml)

    logger.info("\n  Results: %s", result_path)
    logger.info("  Figures: %s/phase2_*.{pdf,png}", FIGURES_DIR)
    logger.info("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
