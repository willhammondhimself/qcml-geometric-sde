"""
Implied Volatility vs Geometric Curvature Comparison.

Compares geometric observables (Berry curvature, QFI, fidelity) against
options-based regime indicators (VIX level, VIX term structure, put-call skew)
to answer: does geometric curvature capture information NOT in implied vol?

Approach:
    1. Fetch equity prices (SPY, DIA) and VIX proxy (^VIX via Polygon, or VIXY ETF)
    2. Compute QCML geometric signals on equity data
    3. Compute options-based indicators (VIX level, VIX z-score, VIX term slope)
    4. Correlation analysis: Spearman rank correlation between each pair
    5. Complementarity test: does combining geometric + VIX improve detection?
    6. Granger causality: does curvature lead VIX changes or vice versa?

Usage:
    python experiments/options_comparison.py
    python experiments/options_comparison.py --quick
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.data_loader import (
    fetch_polygon_data,
    create_feature_matrix,
    ALL_CRISES,
)
from experiments.baselines import RollingVolatilityDetector
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# VIX-based Regime Indicators
# =============================================================================

class VIXLevelDetector(BaseRegimeDetector):
    """Regime detection via VIX level z-score.

    Score = expanding z-score of VIX daily close.
    """

    def __init__(self, min_expanding=60):
        self.min_expanding = min_expanding

    @property
    def name(self):
        return "VIX Level"

    def fit(self, X, **kwargs):
        return self

    def compute_regime_scores(self, X):
        # X[:, 0] is assumed to be VIX close or first feature
        vix = X[:, 0] if X.ndim > 1 else X
        T = len(vix)
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(vix[:t])
            sigma = np.std(vix[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (vix[t] - mu) / sigma
        return z_scores


class VIXTermSlopeDetector(BaseRegimeDetector):
    """Regime detection via VIX term structure inversion.

    Term structure slope = VIX_3m - VIX_1m. Negative slope (backwardation)
    indicates acute stress. Score = negative of expanding z-score of slope.
    """

    def __init__(self, min_expanding=60):
        self.min_expanding = min_expanding

    @property
    def name(self):
        return "VIX Term Slope"

    def fit(self, X, **kwargs):
        return self

    def compute_regime_scores(self, X):
        # Expects X with columns [..., slope] where slope = col -1 or col 1
        slope = X[:, -1] if X.ndim > 1 else X
        T = len(slope)
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(slope[:t])
            sigma = np.std(slope[:t], ddof=1)
            if sigma > 1e-12:
                # Negative slope = stress, so negate for positive crisis signal
                z_scores[t] = -(slope[t] - mu) / sigma
        return z_scores


# =============================================================================
# Granger Causality (bivariate)
# =============================================================================

def granger_causality_test(x, y, max_lag=5):
    """Test if x Granger-causes y.

    Uses OLS F-test: does adding lagged x improve prediction of y over
    lagged y alone?

    Args:
        x: Potential cause series (T,).
        y: Potential effect series (T,).
        max_lag: Maximum lag to test.

    Returns:
        Dict with F-stat and p-value per lag.
    """
    from numpy.linalg import lstsq

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    T = len(x)

    results = {}
    for lag in range(1, max_lag + 1):
        if T <= 2 * lag + 2:
            continue

        # Restricted model: y_t ~ y_{t-1}, ..., y_{t-lag}
        Y = y[lag:]
        X_r = np.column_stack([y[lag - k - 1:T - k - 1] for k in range(lag)])
        X_r = np.column_stack([X_r, np.ones(len(Y))])

        # Unrestricted: y_t ~ y_{t-1}, ..., y_{t-lag}, x_{t-1}, ..., x_{t-lag}
        X_u = np.column_stack([
            X_r[:, :-1],  # lagged y (without intercept)
            *[x[lag - k - 1:T - k - 1].reshape(-1, 1) for k in range(lag)],
            np.ones(len(Y)).reshape(-1, 1),
        ])

        # OLS
        beta_r, res_r, _, _ = lstsq(X_r, Y, rcond=None)
        beta_u, res_u, _, _ = lstsq(X_u, Y, rcond=None)

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


# =============================================================================
# Main Pipeline
# =============================================================================

def run_options_comparison(quick=False):
    """Run geometric curvature vs implied vol comparison.

    Args:
        quick: If True, use fewer crises.

    Returns:
        Dict with correlation, complementarity, and Granger results.
    """
    logger.info("=" * 70)
    logger.info("Geometric Curvature vs Implied Volatility Comparison")
    logger.info("=" * 70)

    # ---- Fetch equity data ----
    logger.info("\n[1] Fetching equity data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]

    # ---- Fetch VIX proxy ----
    logger.info("\n[2] Fetching VIX proxy (VIXY or computing from SPY vol)...")
    # Use 20-day realized vol as VIX proxy (always available)
    spy_close = prices_df['SPY']
    spy_ret = np.log(spy_close / spy_close.shift(1))
    vix_proxy = spy_ret.rolling(20).std() * np.sqrt(252) * 100  # annualized %
    vix_proxy = vix_proxy.dropna()

    # Also try VIXY ETF if available
    try:
        vixy_raw = fetch_polygon_data(['VIXY'], '2011-01-01', '2024-12-31')
        vixy_close = vixy_raw['close'].unstack('symbol')['VIXY'].dropna()
        logger.info(f"  VIXY data: {len(vixy_close)} days")
    except Exception:
        vixy_close = None
        logger.info("  VIXY not available, using realized vol proxy")

    # ---- Compute geometric signals ----
    logger.info("\n[3] Computing geometric observables...")
    common = dict(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
        rolling_window=20, seed=42,
    )
    geo_detectors = [
        ('Berry Phase Rate', BerryPhaseRateDetector(**common)),
        ('QFI Determinant', QFIDeterminantDetector(**common)),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector(**common)),
    ]

    geo_scores = {}
    for name, det in geo_detectors:
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        geo_scores[name] = scores

    # Align VIX proxy to enriched dates
    vix_aligned = vix_proxy.reindex(dates_enriched).values
    geo_scores['VIX Proxy (RV20)'] = vix_aligned

    # Vol Z-score
    vol_det = RollingVolatilityDetector(vol_window=20, min_expanding=60)
    vol_det.fit(X_enriched)
    geo_scores['Rolling Vol Z'] = vol_det.compute_regime_scores(X_enriched)

    # ---- Correlation analysis ----
    logger.info("\n[4] Computing Spearman correlations...")
    method_names = list(geo_scores.keys())
    n_methods = len(method_names)
    corr_matrix = np.full((n_methods, n_methods), np.nan)

    for i in range(n_methods):
        for j in range(n_methods):
            s1 = geo_scores[method_names[i]]
            s2 = geo_scores[method_names[j]]
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if np.sum(valid) > 30:
                rho, p = stats.spearmanr(s1[valid], s2[valid])
                corr_matrix[i, j] = rho

    logger.info(f"\n  {'':25s}" + "".join(f"{n[:10]:>12s}" for n in method_names))
    for i, name in enumerate(method_names):
        row = f"  {name:25s}"
        for j in range(n_methods):
            row += f"{corr_matrix[i,j]:12.3f}" if not np.isnan(corr_matrix[i,j]) else f"{'N/A':>12s}"
        logger.info(row)

    # ---- Complementarity test: Cohen's d comparison ----
    logger.info("\n[5] Complementarity test (Cohen's d per crisis)...")
    crisis_keys = list(ALL_CRISES.keys())
    if quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']

    complement_results = {}
    for ck in crisis_keys:
        ci = ALL_CRISES[ck]
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=10)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=10)

        if cs > dates_enriched[-1] or ce < dates_enriched[0]:
            continue

        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        crisis_d = {}
        for mname, scores in geo_scores.items():
            d, _, _ = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=2000,
            )
            crisis_d[mname] = float(d) if not np.isnan(d) else None

        complement_results[ck] = crisis_d
        vix_d = crisis_d.get('VIX Proxy (RV20)')
        berry_d = crisis_d.get('Berry Phase Rate')
        logger.info(f"  {ck:20s}: VIX d={vix_d:.2f}, Berry d={berry_d:.2f}"
                   if vix_d and berry_d else f"  {ck:20s}: insufficient data")

    # ---- Granger causality ----
    logger.info("\n[6] Granger causality tests...")
    granger_results = {}
    for geo_name in ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity']:
        geo_sig = geo_scores.get(geo_name)
        vix_sig = geo_scores.get('VIX Proxy (RV20)')
        if geo_sig is None or vix_sig is None:
            continue

        # Forward: does geometric curvature Granger-cause VIX changes?
        vix_diff = np.diff(vix_sig)
        geo_trimmed = geo_sig[:-1]

        forward = granger_causality_test(geo_trimmed, vix_diff, max_lag=5)
        reverse = granger_causality_test(vix_diff, geo_trimmed, max_lag=5)

        granger_results[geo_name] = {
            'forward': forward,  # geo → VIX
            'reverse': reverse,  # VIX → geo
        }

        for lag, r in forward.items():
            sig = "*" if r['p'] < 0.05 else ""
            logger.info(f"  {geo_name:25s} → VIX (lag {lag}): "
                       f"F={r['F']:.2f}, p={r['p']:.4f} {sig}")
        for lag, r in reverse.items():
            sig = "*" if r['p'] < 0.05 else ""
            logger.info(f"  VIX → {geo_name:25s} (lag {lag}): "
                       f"F={r['F']:.2f}, p={r['p']:.4f} {sig}")

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {'quick': quick},
        'correlation_matrix': {
            'methods': method_names,
            'values': corr_matrix.tolist(),
        },
        'complementarity': complement_results,
        'granger': granger_results,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'options_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'options_comparison_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Geometric curvature vs implied vol comparison')
    parser.add_argument('--quick', action='store_true', help='Fewer crises')
    args = parser.parse_args()
    run_options_comparison(quick=args.quick)


if __name__ == '__main__':
    main()
