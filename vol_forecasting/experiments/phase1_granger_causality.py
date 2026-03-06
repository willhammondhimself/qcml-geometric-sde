"""
Phase 1 — Granger Causality: Do QCML geometric observables predict realized vol?

Tests whether QFI determinant, Berry curvature rate, multi-lag fidelity, and
spectral gap z-scores Granger-cause forward-looking realized volatility.

Success criterion: at least one QCML feature Granger-causes RV at >= 1 lag,
p < 0.05 after Holm-Bonferroni correction across all 40 forward tests.

Outputs:
    results/phase1_granger_YYYYMMDD_HHMMSS.json
    figures/phase1_scatter_qcml_vs_rv.{pdf,png}
    figures/phase1_ccf_lags.{pdf,png}

Usage:
    python vol_forecasting/experiments/phase1_granger_causality.py
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
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Path setup — reach the repo root so we can import qcml_geometry + experiments
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
RV_HORIZONS = [5, 21]               # weekly and monthly
MAX_GRANGER_LAG = 10
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

FIGURES_DIR = VOL_ROOT / 'figures'
RESULTS_DIR = VOL_ROOT / 'results'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Data Loading & Realized Vol
# ============================================================================

def load_prices() -> pd.Series:
    """Fetch SPY close prices from Polygon."""
    logger.info("Fetching %s %s to %s from Polygon...", SYMBOL, START_DATE, END_DATE)
    raw = fetch_data([SYMBOL], START_DATE, END_DATE)
    prices = raw['close'].droplevel('symbol')
    prices.index = pd.DatetimeIndex(prices.index)
    logger.info("  Got %d daily prices (%s to %s)", len(prices),
                prices.index[0].date(), prices.index[-1].date())
    return prices


def build_realized_vol(log_returns: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Build forward-looking annualized realized volatility.

    RV_{t,h} = std(log_returns[t+1 : t+h+1]) * sqrt(252)

    Last h rows are NaN because we lack future data.
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
# 2. QCML Feature Extraction
# ============================================================================

def extract_qcml_features(prices: pd.Series) -> pd.DataFrame:
    """Extract 4 QCML z-score time series aligned to dates.

    Returns DataFrame with columns: qfi_det, berry_rate, multi_lag_fid, spectral_gap.

    Caveat: scaler/PCA fitted on full data (non-causal). Acceptable for Phase 1
    hypothesis screening; causal extraction deferred to Phase 3.
    """
    # Feature matrix from single asset
    X_raw, dates_raw = create_feature_matrix_single_asset(prices, extra_lags=True)
    logger.info("  Raw features: %s, %d dates", X_raw.shape, len(dates_raw))

    # Enriched features (rolling mean/std/min/max over 20-day lookback)
    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates_raw[19:]  # lookback=20 drops first 19
    logger.info("  Enriched features: %s, %d dates", X_enriched.shape, len(dates_enriched))

    # --- Detectors: QFI, Berry, MultiLag ---
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

    # --- Spectral gap (different API: needs QCMLGeometry + PCA) ---
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

    # Z-score with expanding window (same min_expanding as detectors)
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
# 3. Temporal Alignment
# ============================================================================

def align_data(qcml_df: pd.DataFrame, rv_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join QCML features and RV on date index, drop NaN rows."""
    combined = qcml_df.join(rv_df, how='inner').dropna()
    logger.info("Aligned dataset: %d rows (%s to %s)",
                len(combined), combined.index[0].date(), combined.index[-1].date())
    return combined


# ============================================================================
# 4. Stationarity Testing
# ============================================================================

def test_stationarity(series: pd.Series, name: str) -> dict:
    """Run ADF test and return results dict."""
    result = adfuller(series.values, autolag='AIC')
    return {
        'name': name,
        'adf_stat': float(result[0]),
        'p_value': float(result[1]),
        'n_lags_used': int(result[2]),
        'n_obs': int(result[3]),
        'critical_values': {k: float(v) for k, v in result[4].items()},
        'stationary_at_5pct': result[1] < 0.05,
    }


# ============================================================================
# 5. Granger Causality (OLS F-test)
# ============================================================================

def granger_causality_test(x: np.ndarray, y: np.ndarray, max_lag: int = 10) -> dict:
    """Test if x Granger-causes y via OLS F-test.

    Replicates the pattern from experiments/options_comparison.py.

    Returns:
        Dict mapping lag -> {F, p}.
    """
    from numpy.linalg import lstsq

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    T = len(x)

    results = {}
    for lag in range(1, max_lag + 1):
        if T <= 2 * lag + 2:
            continue

        # Restricted: y_t ~ y_{t-1}, ..., y_{t-lag}, const
        Y = y[lag:]
        X_r = np.column_stack([y[lag - k - 1:T - k - 1] for k in range(lag)])
        X_r = np.column_stack([X_r, np.ones(len(Y))])

        # Unrestricted: add lagged x
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

        results[lag] = {'F': float(F), 'p': float(p)}

    return results


def run_granger_tests(df: pd.DataFrame, rv_col: str,
                      feature_cols: list[str]) -> dict:
    """Run forward and reverse Granger tests for all features.

    Returns:
        Dict with 'forward' and 'reverse' sub-dicts, each mapping
        feature_name -> {lag -> {F, p}}.
    """
    results = {'forward': {}, 'reverse': {}}

    for feat in feature_cols:
        x = df[feat].values
        y = df[rv_col].values

        logger.info("  Granger: %s -> %s", feat, rv_col)
        results['forward'][feat] = granger_causality_test(x, y, MAX_GRANGER_LAG)

        logger.info("  Granger: %s -> %s (reverse)", rv_col, feat)
        results['reverse'][feat] = granger_causality_test(y, x, MAX_GRANGER_LAG)

    return results


def apply_multiple_comparison_correction(granger_results: dict) -> dict:
    """Apply Holm-Bonferroni to all forward-direction p-values.

    Collects all (feature, lag) p-values, corrects jointly, then maps back.
    """
    entries = []
    for feat, lag_dict in granger_results['forward'].items():
        for lag, vals in lag_dict.items():
            entries.append((feat, lag, vals['p']))

    if not entries:
        return {'entries': [], 'any_significant': False}

    p_values = np.array([e[2] for e in entries])
    adjusted_p, rejected = holm_bonferroni_correction(p_values)

    corrected = []
    for i, (feat, lag, raw_p) in enumerate(entries):
        corrected.append({
            'feature': feat,
            'lag': int(lag),
            'p_raw': float(raw_p),
            'p_adjusted': float(adjusted_p[i]),
            'rejected': bool(rejected[i]),
            'F': float(granger_results['forward'][feat][lag]['F']),
        })

    corrected.sort(key=lambda x: x['p_adjusted'])

    return {
        'entries': corrected,
        'n_tests': len(corrected),
        'n_rejected': int(np.sum(rejected)),
        'any_significant': bool(np.any(rejected)),
    }


# ============================================================================
# 6. Cross-Correlation Function
# ============================================================================

def compute_ccf(x: np.ndarray, y: np.ndarray, max_lag: int = 30) -> dict:
    """Compute CCF for lags -max_lag to +max_lag.

    Positive lag k means x at time t is correlated with y at time t+k,
    i.e., x leads y by k steps.

    Returns dict mapping lag -> Pearson r.
    """
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
# 7. Figures
# ============================================================================

def _set_pub_style():
    """Apply publication-quality plot settings."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
    })


def plot_ccf(df: pd.DataFrame, rv_col: str, feature_cols: list[str],
             output_stem: str):
    """4-panel CCF figure (one per QCML feature vs RV)."""
    _set_pub_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    T_eff = len(df)
    sig_bound = 2.0 / np.sqrt(T_eff)

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        ccf = compute_ccf(df[feat].values, df[rv_col].values, CCF_MAX_LAG)

        lags = sorted(ccf.keys())
        vals = [ccf[k] for k in lags]

        ax.bar(lags, vals, width=0.8, color='steelblue', alpha=0.7, edgecolor='none')
        ax.axhline(sig_bound, ls='--', color='red', lw=0.8, alpha=0.6)
        ax.axhline(-sig_bound, ls='--', color='red', lw=0.8, alpha=0.6)
        ax.axhline(0, ls='-', color='black', lw=0.5)
        ax.axvline(0, ls=':', color='gray', lw=0.5)

        # Shade positive-lag region where QCML leads RV
        ax.axvspan(0.5, CCF_MAX_LAG + 0.5, alpha=0.05, color='green')

        ax.set_title(feat.replace('_', ' ').title(), fontsize=11)
        if i >= 2:
            ax.set_xlabel('Lag (days)')
        if i % 2 == 0:
            ax.set_ylabel('Pearson r')

    fig.suptitle(f'Cross-Correlation: QCML features vs {rv_col}', fontsize=13, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'{output_stem}.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved CCF figure: %s.{pdf,png}", output_stem)


def plot_scatter(df: pd.DataFrame, rv_col: str, feature_cols: list[str],
                 output_stem: str):
    """2x2 scatter: QCML z-score at t vs RV at t+21."""
    _set_pub_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        x = df[feat].values
        y = df[rv_col].values

        valid = ~(np.isnan(x) | np.isnan(y))
        x_v, y_v = x[valid], y[valid]

        ax.scatter(x_v, y_v, s=2, alpha=0.15, color='steelblue', rasterized=True)

        # LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(y_v, x_v, frac=0.2, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color='red', lw=2, label='LOWESS')
        except ImportError:
            pass

        rho, p = stats.spearmanr(x_v, y_v)
        ax.set_title(f'{feat.replace("_", " ").title()}\n'
                     f'Spearman r = {rho:.3f} (p = {p:.2e})', fontsize=10)
        ax.set_xlabel('QCML z-score (t)')
        ax.set_ylabel(f'{rv_col} (t)')

    fig.suptitle(f'QCML z-scores vs {rv_col}', fontsize=13, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'{output_stem}.{ext}', dpi=300)
    plt.close(fig)
    logger.info("Saved scatter figure: %s.{pdf,png}", output_stem)


# ============================================================================
# 8. Save Results
# ============================================================================

def save_results(config: dict, data_summary: dict, stationarity: list,
                 granger: dict, correction: dict, spearman: dict,
                 ccf_peaks: dict, success: bool) -> Path:
    """Write results JSON with timestamp."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp': ts,
        'config': config,
        'data_summary': data_summary,
        'stationarity': stationarity,
        'granger_causality': granger,
        'holm_bonferroni': correction,
        'spearman_correlations': spearman,
        'ccf_peak_lags': ccf_peaks,
        'success': success,
        'caveat': ('Scaler/PCA fitted on full data (non-causal). '
                   'Phase 1 hypothesis test only; causal extraction in Phase 3.'),
    }

    path = RESULTS_DIR / f'phase1_granger_{ts}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Phase 1 — Granger Causality: QCML features -> Realized Vol")
    logger.info("=" * 60)

    # --- Load data ---
    prices = load_prices()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # --- Build forward-looking RV ---
    logger.info("Building realized vol targets (h=%s)...", RV_HORIZONS)
    rv_df = build_realized_vol(log_returns, RV_HORIZONS)

    # --- Extract QCML features ---
    logger.info("Extracting QCML features...")
    qcml_df = extract_qcml_features(prices)

    # --- Align ---
    logger.info("Aligning QCML features with RV...")
    combined = align_data(qcml_df, rv_df)

    feature_cols = ['qfi_det', 'berry_rate', 'multi_lag_fid', 'spectral_gap']
    rv_col = 'rv_21d'

    data_summary = {
        'n_prices': len(prices),
        'n_aligned': len(combined),
        'date_range': [str(combined.index[0].date()), str(combined.index[-1].date())],
        'feature_cols': feature_cols,
        'rv_cols': [f'rv_{h}d' for h in RV_HORIZONS],
    }

    # --- Stationarity ---
    logger.info("Running ADF stationarity tests...")
    stationarity_results = []
    use_log_rv = {}

    for col in feature_cols + [f'rv_{h}d' for h in RV_HORIZONS]:
        adf = test_stationarity(combined[col], col)
        stationarity_results.append(adf)
        logger.info("  %s: ADF=%.3f, p=%.4f, stationary=%s",
                     col, adf['adf_stat'], adf['p_value'], adf['stationary_at_5pct'])

        # If RV level is non-stationary, switch to log(RV)
        if col.startswith('rv_') and not adf['stationary_at_5pct']:
            log_col = f'log_{col}'
            combined[log_col] = np.log(combined[col].clip(lower=1e-6))
            adf_log = test_stationarity(combined[log_col], log_col)
            stationarity_results.append(adf_log)
            use_log_rv[col] = log_col
            logger.info("  %s: ADF=%.3f, p=%.4f, stationary=%s (log transform)",
                         log_col, adf_log['adf_stat'], adf_log['p_value'],
                         adf_log['stationary_at_5pct'])

    # Use log(RV) if level RV failed ADF
    effective_rv_col = use_log_rv.get(rv_col, rv_col)
    logger.info("Using %s as target for Granger tests", effective_rv_col)

    # --- Granger causality ---
    logger.info("Running Granger causality tests (max_lag=%d)...", MAX_GRANGER_LAG)
    granger_results = run_granger_tests(combined, effective_rv_col, feature_cols)

    # --- Multiple comparison correction ---
    logger.info("Applying Holm-Bonferroni correction...")
    correction = apply_multiple_comparison_correction(granger_results)

    n_sig = correction['n_rejected']
    logger.info("  %d / %d tests significant after correction", n_sig, correction['n_tests'])

    if correction['any_significant']:
        logger.info("  *** SUCCESS: At least one QCML feature Granger-causes RV ***")
        for entry in correction['entries']:
            if entry['rejected']:
                logger.info("    %s lag=%d: F=%.2f, p_raw=%.4e, p_adj=%.4e",
                             entry['feature'], entry['lag'], entry['F'],
                             entry['p_raw'], entry['p_adjusted'])
    else:
        logger.info("  No significant Granger causality after correction.")

    # --- Spearman correlations ---
    logger.info("Computing Spearman correlations...")
    spearman = {}
    for feat in feature_cols:
        rho, p = stats.spearmanr(combined[feat].values, combined[rv_col].values,
                                  nan_policy='omit')
        spearman[feat] = {'rho': float(rho), 'p': float(p)}
        logger.info("  %s vs %s: rho=%.4f, p=%.2e", feat, rv_col, rho, p)

    # --- CCF peak lags ---
    logger.info("Computing CCF peak lags...")
    ccf_peaks = {}
    for feat in feature_cols:
        ccf = compute_ccf(combined[feat].values, combined[rv_col].values, CCF_MAX_LAG)
        # Find peak positive-lag correlation (where QCML leads RV)
        pos_lags = {k: v for k, v in ccf.items() if k > 0 and not np.isnan(v)}
        if pos_lags:
            peak_lag = max(pos_lags, key=lambda k: abs(pos_lags[k]))
            ccf_peaks[feat] = {
                'peak_lag': int(peak_lag),
                'peak_r': float(pos_lags[peak_lag]),
            }
        else:
            ccf_peaks[feat] = {'peak_lag': None, 'peak_r': None}

    # --- Figures ---
    logger.info("Generating figures...")
    plot_ccf(combined, rv_col, feature_cols, 'phase1_ccf_lags')
    plot_scatter(combined, rv_col, feature_cols, 'phase1_scatter_qcml_vs_rv')

    # --- Save ---
    config = {
        'symbol': SYMBOL,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'rv_horizons': RV_HORIZONS,
        'max_granger_lag': MAX_GRANGER_LAG,
        'ccf_max_lag': CCF_MAX_LAG,
        'detector_params': DETECTOR_PARAMS,
        'effective_rv_target': effective_rv_col,
    }

    # Serialize granger results with string keys for JSON
    granger_json = {}
    for direction in ('forward', 'reverse'):
        granger_json[direction] = {}
        for feat, lag_dict in granger_results[direction].items():
            granger_json[direction][feat] = {
                str(lag): vals for lag, vals in lag_dict.items()
            }

    success = correction['any_significant']
    result_path = save_results(
        config=config,
        data_summary=data_summary,
        stationarity=stationarity_results,
        granger=granger_json,
        correction=correction,
        spearman=spearman,
        ccf_peaks=ccf_peaks,
        success=success,
    )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Phase 1 Summary")
    logger.info("=" * 60)
    logger.info("  Aligned samples: %d", len(combined))
    logger.info("  RV target: %s", effective_rv_col)
    logger.info("  Granger tests: %d forward, %d reverse",
                correction['n_tests'], correction['n_tests'])
    logger.info("  Significant after Holm-Bonferroni: %d / %d",
                correction['n_rejected'], correction['n_tests'])
    logger.info("  SUCCESS: %s", success)
    logger.info("  Results: %s", result_path)
    logger.info("=" * 60)

    return success


if __name__ == '__main__':
    main()
