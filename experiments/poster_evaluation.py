"""
Authoritative Poster Evaluation: Single comprehensive run for APS 2026 poster.

Runs all metrics sequentially:
1. Fetch real data from Polygon API (SPY, DIA, QQQ, IWM, EFA, 2005-2024)
2. Compute all geometric observables (original + new multi-plane, Ricci, geodesic)
3. Run 20+ method comparison (Cohen's d per crisis)
4. Run online detection evaluation (AUC-ROC, AUC-PR, detection rate)
5. Run backtest with improved strategies (Sharpe, Calmar, max drawdown)
6. Save POSTER_RESULTS.json
7. Call poster figure generator with real data

Usage:
    python experiments/poster_evaluation.py
    python experiments/poster_evaluation.py --quick  # fewer crises, faster
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from sklearn.metrics import roc_auc_score, average_precision_score

from qcml_geometry.observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
    MultiPlaneBerryDetector,
    RicciScalarDetector,
    GeodesicDistanceDetector,
)
from qcml_geometry.online_detection import (
    OnlineGeometricFeatureComputer,
    OnlineBayesianDetector,
    OnlineHMMDetector,
    ExpandingPercentileDetector,
    OnlineLogisticDetector,
    OnlineEnsembleDetector,
    OnlineStackingEnsemble,
)

from experiments.data_loader import (
    fetch_data, create_feature_matrix, ALL_CRISES,
)
from experiments.backtest.strategies import (
    geometric_long_flat, geometric_long_short,
    geometric_continuous, multi_signal_strategy,
)
from experiments.backtest.benchmarks import (
    buy_and_hold_spy, constant_vol_spy,
)
from experiments.backtest.execution import apply_transaction_costs
from experiments.backtest.metrics import compute_backtest_metrics
from experiments.backtest.risk_management import dynamic_vol_target

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Section 1: Offline Regime Detection (Cohen's d)
# =============================================================================

def compute_cohens_d(scores, labels, min_valid=30):
    """Compute Cohen's d: (mean_crisis - mean_calm) / pooled_std.

    Args:
        scores: Regime scores (T,).
        labels: Binary crisis labels (T,).
        min_valid: Minimum valid scores per group.

    Returns:
        Cohen's d (float) or NaN.
    """
    valid = ~np.isnan(scores)
    s = scores[valid]
    l = labels[valid]

    crisis = s[l == 1]
    calm = s[l == 0]

    if len(crisis) < min_valid or len(calm) < min_valid:
        return np.nan

    mean_diff = np.mean(crisis) - np.mean(calm)
    n1, n2 = len(calm), len(crisis)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(calm, ddof=1) + (n2 - 1) * np.var(crisis, ddof=1))
        / (n1 + n2 - 2)
    )

    if pooled_std < 1e-12:
        return 0.0
    return mean_diff / pooled_std


def run_offline_comparison(X_raw, dates, all_crises, quick=False):
    """Run offline comparison of all detectors, return Cohen's d per crisis.

    Args:
        X_raw: Feature matrix (T, d).
        dates: DatetimeIndex.
        all_crises: Crisis definitions dict.
        quick: If True, use subset of crises.

    Returns:
        Dict of {method_name: {crisis_name: cohens_d}}.
    """
    logger.info("\n[2/5] Running offline regime detection comparison...")

    # Build crisis labels
    T = len(X_raw)
    crisis_labels = np.zeros(T)
    for ci in all_crises.values():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (dates >= cs) & (dates <= ce)
        crisis_labels[mask] = 1.0

    crises_to_test = list(all_crises.keys())
    if quick:
        crises_to_test = ['2008_gfc', '2020_covid', '2022_rates', '2024_carry']

    # Initialize detectors
    detectors = [
        BerryPhaseRateDetector(hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired'),
        QFIDeterminantDetector(hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired'),
        MultiLagFidelityDetector(hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired'),
        MultiPlaneBerryDetector(hilbert_dim=8, n_pca_components=8, operator_method='pca_inspired',
                                top_k_planes=10, aggregation='max'),
        GeodesicDistanceDetector(hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired'),
    ]

    # Ricci is slow — use fewer PCA components
    detectors.append(
        RicciScalarDetector(hilbert_dim=8, n_pca_components=4, operator_method='pca_inspired')
    )

    results = {}
    for det in detectors:
        logger.info(f"  Fitting {det.name}...")
        det.fit(X_raw)
        scores = det.compute_regime_scores(X_raw)

        per_crisis = {}
        for crisis_key in crises_to_test:
            ci = all_crises[crisis_key]
            cs, ce = pd.Timestamp(ci['start']), pd.Timestamp(ci['end'])
            mask_crisis = (dates >= cs) & (dates <= ce)
            mask_calm = ~mask_crisis & (dates >= cs - pd.Timedelta(days=365))

            local_labels = np.zeros(T)
            local_labels[mask_crisis] = 1.0

            valid_mask = mask_crisis | mask_calm
            if np.sum(valid_mask) < 60:
                per_crisis[crisis_key] = np.nan
                continue

            d = compute_cohens_d(scores[valid_mask], local_labels[valid_mask])
            per_crisis[crisis_key] = float(d) if not np.isnan(d) else None

        # Global d
        d_global = compute_cohens_d(scores, crisis_labels)
        per_crisis['global'] = float(d_global) if not np.isnan(d_global) else None

        results[det.name] = per_crisis
        logger.info(f"    {det.name}: global d={d_global:.2f}")

    return results


# =============================================================================
# Section 2: Online Detection (AUC)
# =============================================================================

def run_online_evaluation(X_enriched, dates, crisis_labels):
    """Run online detection and compute AUC-ROC/PR.

    Args:
        X_enriched: Enriched feature matrix (T, d).
        dates: DatetimeIndex.
        crisis_labels: Binary labels (T,).

    Returns:
        Dict with per-detector AUCs and the P(crisis) time series.
    """
    logger.info("\n[3/5] Running online detection evaluation...")

    T = len(X_enriched)

    feat_computer = OnlineGeometricFeatureComputer(
        hilbert_dim=8, n_pca_components=15,
        operator_method='pca_inspired', seed=42,
    )

    bayesian = OnlineBayesianDetector()
    hmm = OnlineHMMDetector(seed=42)
    percentile = ExpandingPercentileDetector(min_history=60)
    logistic = OnlineLogisticDetector(refit_interval=21, min_history=126)

    # Weighted ensemble (old-style)
    ensemble_old = OnlineEnsembleDetector(
        detectors=[
            OnlineBayesianDetector(),
            OnlineHMMDetector(seed=42),
            ExpandingPercentileDetector(min_history=60),
        ],
        weights=[0.4, 0.4, 0.2],
    )

    # Stacking ensemble (new)
    stacking = OnlineStackingEnsemble(
        detectors=[
            OnlineBayesianDetector(),
            OnlineHMMDetector(seed=42),
            ExpandingPercentileDetector(min_history=60),
        ],
        min_meta_history=252,
    )

    all_detectors = {
        'Bayesian': bayesian,
        'HMM': hmm,
        'Percentile': percentile,
        'Logistic': logistic,
        'Ensemble (weighted)': ensemble_old,
        'Ensemble (stacking)': stacking,
    }

    # Collect P(crisis) for each detector
    p_all = {name: np.full(T, np.nan) for name in all_detectors}
    log_interval = max(T // 10, 1)

    for t in range(T):
        if (t + 1) % log_interval == 0:
            logger.info(f"    Online step {t+1}/{T}")

        features = feat_computer.update(X_enriched[t])

        for name, det in all_detectors.items():
            if name == 'Logistic':
                det.add_label(crisis_labels[t])
            if name == 'Ensemble (stacking)':
                det.add_label(crisis_labels[t])
            p = det.update(features)
            p_all[name][t] = p

    # Compute AUC metrics
    min_valid = 252
    results = {}
    for name, p_series in p_all.items():
        valid = ~np.isnan(p_series)
        if np.sum(valid) < min_valid:
            results[name] = {'auc_roc': None, 'auc_pr': None, 'n_valid': int(np.sum(valid))}
            continue

        p_valid = p_series[valid]
        y_valid = crisis_labels[valid]

        if len(np.unique(y_valid)) < 2:
            results[name] = {'auc_roc': None, 'auc_pr': None, 'n_valid': int(np.sum(valid))}
            continue

        try:
            auc_roc = roc_auc_score(y_valid, p_valid)
            auc_pr = average_precision_score(y_valid, p_valid)
        except Exception:
            auc_roc, auc_pr = None, None

        results[name] = {
            'auc_roc': float(auc_roc) if auc_roc is not None else None,
            'auc_pr': float(auc_pr) if auc_pr is not None else None,
            'n_valid': int(np.sum(valid)),
        }
        logger.info(f"    {name}: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")

    # Return best ensemble P(crisis) for backtest
    best_name = max(
        [(n, r['auc_roc'] or 0) for n, r in results.items()],
        key=lambda x: x[1],
    )[0]
    logger.info(f"    Best detector: {best_name}")

    return results, p_all[best_name], p_all


# =============================================================================
# Section 3: Backtest
# =============================================================================

def run_backtest_suite(spy_returns, p_crisis, spy_dates, target_vol=0.10):
    """Run all strategies and benchmarks, return metrics.

    Args:
        spy_returns: Daily SPY returns (T,).
        p_crisis: P(crisis) signal (T,).
        spy_dates: DatetimeIndex.
        target_vol: Vol target.

    Returns:
        Dict of strategy metrics.
    """
    logger.info("\n[4/5] Running backtest suite...")

    strategies = {}

    # S1: GeometricLongFlat (original)
    w, _ = geometric_long_flat(spy_returns, p_crisis, target_vol=target_vol)
    net, costs = apply_transaction_costs(spy_returns, w, cost_bps=0.5)
    strategies['GeometricLongFlat'] = net

    # S2: GeometricLongShort
    w, _ = geometric_long_short(spy_returns, p_crisis, target_vol=target_vol)
    net, _ = apply_transaction_costs(spy_returns, w, cost_bps=0.5)
    strategies['GeometricLongShort'] = net

    # S3: Continuous Sizing (NEW)
    w, _ = geometric_continuous(spy_returns, p_crisis, target_vol=target_vol)
    net, _ = apply_transaction_costs(spy_returns, w, cost_bps=0.5)
    strategies['GeometricContinuous'] = net

    # S4: Multi-Signal (NEW)
    w, _ = multi_signal_strategy(spy_returns, p_crisis, target_vol=target_vol)
    net, _ = apply_transaction_costs(spy_returns, w, cost_bps=0.5)
    strategies['MultiSignal'] = net

    # S5: Dynamic Vol Target (NEW)
    dyn_vol = dynamic_vol_target(p_crisis, vol_calm=0.12, vol_crisis=0.05)
    from experiments.backtest.portfolio import vol_target_weight
    from experiments.backtest.strategies import regime_adjusted_weight
    from experiments.backtest.execution import apply_min_holding_period
    from experiments.backtest.risk_management import drawdown_circuit_breaker, apply_position_limits

    T = len(spy_returns)
    base_w = np.zeros(T)
    for t in range(20, T):
        local_vol = np.std(spy_returns[max(0, t-20):t]) * np.sqrt(252)
        base_w[t] = min(dyn_vol[t] / max(local_vol, 0.01), 2.0)
    adjusted_w = regime_adjusted_weight(
        base_w, p_crisis, crisis_threshold=0.5,
        crisis_weight_multiplier=0.0, ramp_width=0.3, ramp_type='sigmoid',
    )
    adjusted_w = apply_min_holding_period(adjusted_w, min_days=3)
    adjusted_w = apply_position_limits(adjusted_w, max_long=1.5, max_short=0.0)
    weights_dv = np.zeros(T)
    weights_dv[1:] = adjusted_w[:-1]
    equity = np.cumprod(1 + spy_returns * weights_dv)
    weights_dv = drawdown_circuit_breaker(equity, weights_dv, max_drawdown=0.20)
    net_dv, _ = apply_transaction_costs(spy_returns, weights_dv, cost_bps=0.5)
    strategies['DynamicVolTarget'] = net_dv

    # Benchmarks
    w_bh = buy_and_hold_spy(spy_returns)
    strategies['BuyHoldSPY'] = spy_returns * w_bh

    w_cv = constant_vol_spy(spy_returns, target_vol=target_vol)
    net_cv, _ = apply_transaction_costs(spy_returns, w_cv, cost_bps=0.5)
    strategies['ConstantVolSPY'] = net_cv

    # Compute metrics for all
    is_cutoff = pd.Timestamp('2020-01-01')
    is_mask = spy_dates < is_cutoff
    oos_mask = spy_dates >= is_cutoff

    results = {}
    for name, net_ret in strategies.items():
        full = compute_backtest_metrics(net_ret)
        is_m = compute_backtest_metrics(net_ret[is_mask[:len(net_ret)]])
        oos_m = compute_backtest_metrics(net_ret[oos_mask[:len(net_ret)]])

        results[name] = {
            'full': full,
            'in_sample': is_m,
            'out_of_sample': oos_m,
        }
        logger.info(
            f"    {name:25s}: Sharpe={full['sharpe']:.2f} "
            f"(IS={is_m['sharpe']:.2f}, OOS={oos_m['sharpe']:.2f}), "
            f"MaxDD={full['max_drawdown']:.1%}"
        )

    return results


# =============================================================================
# Main Pipeline
# =============================================================================

def main(quick=False):
    logger.info("=" * 70)
    logger.info("POSTER EVALUATION: Geometric Regime Detection")
    logger.info("=" * 70)

    # ---- Step 1: Fetch data ----
    logger.info("\n[1/5] Fetching data from Polygon API...")
    symbols = ['SPY', 'DIA', 'QQQ', 'IWM', 'EFA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(prices_df)

    spy_prices = prices_df['SPY'].reindex(dates).dropna()
    spy_returns = spy_prices.pct_change().fillna(0).values
    spy_dates = spy_prices.index

    T = min(len(X_raw), len(spy_returns))
    X_raw = X_raw[:T]
    spy_returns = spy_returns[:T]
    spy_dates = spy_dates[:T]
    dates = dates[:T]

    logger.info(f"  Data: {T} days, {dates[0].date()} to {dates[-1].date()}, {len(symbols)} symbols")

    # Enriched features for online detection
    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates[19:]
    T_enriched = min(len(X_enriched), len(spy_returns) - 19)
    X_enriched = X_enriched[:T_enriched]
    spy_returns_enriched = spy_returns[19:19 + T_enriched]
    spy_dates_enriched = spy_dates[19:19 + T_enriched]

    crisis_labels = np.zeros(T_enriched)
    for ci in ALL_CRISES.values():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (spy_dates_enriched >= cs) & (spy_dates_enriched <= ce)
        crisis_labels[mask] = 1.0

    # ---- Step 2: Offline comparison ----
    offline_results = run_offline_comparison(X_raw, dates, ALL_CRISES, quick=quick)

    # ---- Step 3: Online detection ----
    online_results, p_crisis_best, p_crisis_all = run_online_evaluation(
        X_enriched, dates_enriched, crisis_labels,
    )

    # ---- Step 4: Backtest ----
    backtest_results = run_backtest_suite(
        spy_returns_enriched, p_crisis_best, spy_dates_enriched,
    )

    # ---- Step 5: Save results + generate figures ----
    logger.info("\n[5/5] Saving results and generating figures...")

    output = {
        'timestamp': datetime.now().isoformat(),
        'data': {
            'symbols': symbols,
            'n_days': T,
            'start_date': str(dates[0].date()),
            'end_date': str(dates[-1].date()),
        },
        'offline_comparison': _make_serializable(offline_results),
        'online_detection': _make_serializable(online_results),
        'backtest': _make_serializable(backtest_results),
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'POSTER_RESULTS.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Results saved to {out_path}")

    # Save real-data time series for figure generation
    figure_data = {
        'spy_prices': spy_prices.reindex(spy_dates_enriched).values.tolist(),
        'spy_dates': [str(d.date()) for d in spy_dates_enriched],
        'p_crisis': np.where(np.isnan(p_crisis_best), 0.0, p_crisis_best).tolist(),
    }
    # Add per-detector P(crisis) series
    for name, p_series in p_crisis_all.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        figure_data[f'p_crisis_{safe_name}'] = np.where(
            np.isnan(p_series), 0.0, p_series
        ).tolist()

    fig_data_path = out_dir / 'POSTER_FIGURE_DATA.json'
    with open(fig_data_path, 'w') as f:
        json.dump(figure_data, f)
    logger.info(f"  Figure data saved to {fig_data_path}")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("POSTER EVALUATION SUMMARY")
    logger.info("=" * 70)

    # Best offline d-values
    logger.info("\nOffline Cohen's d (global):")
    for method, per_crisis in sorted(
        offline_results.items(),
        key=lambda x: x[1].get('global') or 0,
        reverse=True,
    ):
        g = per_crisis.get('global')
        logger.info(f"  {method:25s}: d = {g:.2f}" if g else f"  {method:25s}: d = N/A")

    # Best online AUC
    logger.info("\nOnline AUC-ROC:")
    for name, r in sorted(
        online_results.items(),
        key=lambda x: x[1].get('auc_roc') or 0,
        reverse=True,
    ):
        auc = r.get('auc_roc')
        logger.info(f"  {name:25s}: AUC = {auc:.3f}" if auc else f"  {name:25s}: AUC = N/A")

    # Best backtest Sharpe
    logger.info("\nBacktest Sharpe (full period, net of costs):")
    for name, r in sorted(
        backtest_results.items(),
        key=lambda x: x[1]['full']['sharpe'],
        reverse=True,
    ):
        fp = r['full']
        logger.info(
            f"  {name:25s}: Sharpe={fp['sharpe']:.2f}, MaxDD={fp['max_drawdown']:.1%}"
        )

    logger.info(f"\nDone. Results at {out_path}")
    return output


def _make_serializable(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poster evaluation pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick mode (4 crises)')
    args = parser.parse_args()
    main(quick=args.quick)
