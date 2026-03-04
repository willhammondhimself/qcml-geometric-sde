"""
Main backtest orchestrator: runs all strategies and benchmarks.

Signal at close(t) affects weight at open(t+1) — strictly causal.
Reports IS (2005-2019) and OOS (2020-2024) results separately.

Usage:
    python experiments/backtest/run_backtest.py
    python experiments/backtest/run_backtest.py --quick
    python experiments/backtest/run_backtest.py --target-vol 0.15

Outputs:
    experiments/outputs/regime_detection/backtest/backtest_YYYYMMDD_HHMMSS.json
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.observables import BaseRegimeDetector
from qcml_geometry.online_detection import (
    OnlineGeometricFeatureComputer,
    OnlineBayesianDetector,
    OnlineHMMDetector,
    ExpandingPercentileDetector,
    OnlineEnsembleDetector,
)

from experiments.data_loader import (
    fetch_data, create_feature_matrix, ALL_CRISES,
)
from experiments.backtest.strategies import (
    geometric_long_flat,
    geometric_multi_asset,
    geometric_long_short,
)
from experiments.backtest.benchmarks import (
    buy_and_hold_spy,
    buy_and_hold_equal_weight,
    sixty_forty,
    constant_vol_spy,
    constant_vol_multi_asset,
)
from experiments.backtest.execution import apply_transaction_costs, compute_turnover
from experiments.backtest.metrics import (
    compute_backtest_metrics,
    bootstrap_sharpe_ci,
    ledoit_wolf_sharpe_test,
    crisis_period_returns,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Online P(crisis) Signal Generator
# =============================================================================

def generate_online_pcris(X_enriched, dates, crisis_labels):
    """Generate causal P(crisis) signal using online ensemble detector.

    Runs the full online pipeline point-by-point. Uses a weighted ensemble
    of Bayesian (0.4), HMM (0.4), and Percentile (0.2) detectors.

    Args:
        X_enriched: Enriched feature matrix (T, d).
        dates: DatetimeIndex.
        crisis_labels: Binary labels (for supervised model).

    Returns:
        p_crisis: Array (T,) of P(crisis) values.
    """
    T = len(X_enriched)

    feat_computer = OnlineGeometricFeatureComputer(
        hilbert_dim=8, n_pca_components=15,
        operator_method='pca_inspired',
        seed=42,
    )

    bayesian = OnlineBayesianDetector(
        transition_prob=0.02, persistence=0.95,
    )
    hmm = OnlineHMMDetector(seed=42)
    percentile = ExpandingPercentileDetector(min_history=60)

    ensemble = OnlineEnsembleDetector(
        detectors=[bayesian, hmm, percentile],
        weights=[0.4, 0.4, 0.2],
    )

    p_crisis = np.full(T, np.nan)
    log_interval = max(T // 10, 1)

    for t in range(T):
        if (t + 1) % log_interval == 0:
            logger.info(f"    Signal generation: step {t+1}/{T}")

        features = feat_computer.update(X_enriched[t])
        p = ensemble.update(features)
        p_crisis[t] = p

    return p_crisis


# =============================================================================
# Main Backtest Pipeline
# =============================================================================

def run_backtest(
    target_vol=0.10,
    crisis_threshold=0.5,
    quick=False,
    n_bootstrap=10000,
):
    """Run the full backtest pipeline.

    Args:
        target_vol: Vol target for strategies.
        crisis_threshold: P(crisis) threshold for regime adjustment.
        quick: Skip multi-asset strategies.
        n_bootstrap: Bootstrap resamples for Sharpe CI.

    Returns:
        Full results dict.
    """
    logger.info("=" * 70)
    logger.info("PnL BACKTEST: GEOMETRIC REGIME DETECTION")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1/6] Fetching data...")
    symbols = ['SPY', 'DIA']
    multi_symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'DIA']

    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(prices_df)

    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates[19:]

    # SPY returns aligned to enriched dates
    spy_prices = prices_df['SPY'].reindex(dates_enriched).dropna()
    spy_returns = spy_prices.pct_change().fillna(0).values
    spy_dates = spy_prices.index

    # Align lengths
    T = min(len(X_enriched), len(spy_returns))
    X_enriched = X_enriched[:T]
    spy_returns = spy_returns[:T]
    spy_dates = spy_dates[:T]
    dates_enriched = dates_enriched[:T]

    logger.info(f"  SPY: {T} days, {spy_dates[0].date()} to {spy_dates[-1].date()}")

    # Multi-asset returns
    multi_returns = None
    if not quick:
        try:
            raw_multi = fetch_data(multi_symbols, '2005-01-01', '2024-12-31')
            multi_prices = raw_multi['close'].unstack('symbol').dropna()
            multi_rets_df = multi_prices.pct_change().dropna()
            # Align to SPY dates
            common_dates = multi_rets_df.index.intersection(spy_dates)
            multi_returns = multi_rets_df.reindex(common_dates).values
            logger.info(f"  Multi-asset: {multi_returns.shape[0]} days, {len(multi_symbols)} assets")
        except Exception as e:
            logger.warning(f"  Multi-asset data unavailable: {e}")

    # Crisis labels
    crisis_labels = np.zeros(T)
    for ci in ALL_CRISES.values():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (spy_dates >= cs) & (spy_dates <= ce)
        crisis_labels[mask] = 1.0

    # ---- Generate P(crisis) signal ----
    logger.info("\n[2/6] Generating online P(crisis) signal...")
    p_crisis = generate_online_pcris(X_enriched, dates_enriched, crisis_labels)

    valid_count = np.sum(~np.isnan(p_crisis))
    logger.info(f"  Signal: {valid_count}/{T} valid values, mean={np.nanmean(p_crisis):.3f}")

    # ---- Run strategies ----
    logger.info("\n[3/6] Running strategies...")

    all_results = {}

    # Strategy 1: GeometricLongFlat
    logger.info("  GeometricLongFlat...")
    w_glf, _ = geometric_long_flat(
        spy_returns, p_crisis,
        target_vol=target_vol, crisis_threshold=crisis_threshold,
    )
    ret_glf_gross = spy_returns * w_glf
    ret_glf_net, cost_glf = apply_transaction_costs(
        spy_returns, w_glf, cost_bps=0.5,
    )
    all_results['GeometricLongFlat'] = {
        'weights': w_glf,
        'gross_returns': ret_glf_gross,
        'net_returns': ret_glf_net,
        'costs': cost_glf,
        'asset': 'SPY',
    }

    # Strategy 2: GeometricLongShort
    logger.info("  GeometricLongShort...")
    w_gls, _ = geometric_long_short(
        spy_returns, p_crisis,
        target_vol=target_vol, crisis_threshold=crisis_threshold,
    )
    ret_gls_gross = spy_returns * w_gls
    ret_gls_net, cost_gls = apply_transaction_costs(
        spy_returns, w_gls, cost_bps=0.5,
    )
    all_results['GeometricLongShort'] = {
        'weights': w_gls,
        'gross_returns': ret_gls_gross,
        'net_returns': ret_gls_net,
        'costs': cost_gls,
        'asset': 'SPY',
    }

    # Strategy 3: GeometricMultiAsset (if data available)
    if multi_returns is not None:
        logger.info("  GeometricMultiAsset...")
        T_multi = multi_returns.shape[0]
        p_crisis_multi = p_crisis[:T_multi]
        w_gma, _ = geometric_multi_asset(
            multi_returns, p_crisis_multi, multi_symbols,
            target_vol=target_vol, crisis_threshold=crisis_threshold,
        )
        ret_gma_gross = np.sum(multi_returns * w_gma, axis=1)
        ret_gma_net, cost_gma = apply_transaction_costs(
            multi_returns, w_gma, symbols=multi_symbols,
        )
        all_results['GeometricMultiAsset'] = {
            'weights': w_gma,
            'gross_returns': ret_gma_gross,
            'net_returns': ret_gma_net,
            'costs': cost_gma,
            'asset': 'Multi',
        }

    # ---- Run benchmarks ----
    logger.info("\n[4/6] Running benchmarks...")

    # Benchmark 1: Buy-and-hold SPY
    w_bh = buy_and_hold_spy(spy_returns)
    ret_bh = spy_returns * w_bh
    all_results['BuyHoldSPY'] = {
        'weights': w_bh,
        'gross_returns': ret_bh,
        'net_returns': ret_bh,  # no turnover
        'costs': np.zeros(T),
        'asset': 'SPY',
    }

    # Benchmark 2: 60/40
    w_6040 = sixty_forty(spy_returns)
    ret_6040 = spy_returns * w_6040
    all_results['SixtyForty'] = {
        'weights': w_6040,
        'gross_returns': ret_6040,
        'net_returns': ret_6040,
        'costs': np.zeros(T),
        'asset': 'SPY',
    }

    # Benchmark 3: ConstantVolSPY (THE KEY BENCHMARK)
    w_cvs = constant_vol_spy(spy_returns, target_vol=target_vol)
    ret_cvs_gross = spy_returns * w_cvs
    ret_cvs_net, cost_cvs = apply_transaction_costs(
        spy_returns, w_cvs, cost_bps=0.5,
    )
    all_results['ConstantVolSPY'] = {
        'weights': w_cvs,
        'gross_returns': ret_cvs_gross,
        'net_returns': ret_cvs_net,
        'costs': cost_cvs,
        'asset': 'SPY',
    }

    # Multi-asset benchmarks
    if multi_returns is not None:
        w_bh_ew = buy_and_hold_equal_weight(multi_returns)
        ret_bh_ew = np.sum(multi_returns * w_bh_ew, axis=1)
        all_results['BuyHoldEqualWeight'] = {
            'weights': w_bh_ew,
            'gross_returns': ret_bh_ew,
            'net_returns': ret_bh_ew,
            'costs': np.zeros(len(multi_returns)),
            'asset': 'Multi',
        }

        w_cv_ew = constant_vol_multi_asset(multi_returns, target_vol=target_vol)
        ret_cv_ew_gross = np.sum(multi_returns * w_cv_ew, axis=1)
        ret_cv_ew_net, cost_cv_ew = apply_transaction_costs(
            multi_returns, w_cv_ew, symbols=multi_symbols,
        )
        all_results['ConstantVolMultiAsset'] = {
            'weights': w_cv_ew,
            'gross_returns': ret_cv_ew_gross,
            'net_returns': ret_cv_ew_net,
            'costs': cost_cv_ew,
            'asset': 'Multi',
        }

    # ---- Compute metrics ----
    logger.info("\n[5/6] Computing metrics...")

    # IS/OOS split
    is_cutoff = pd.Timestamp('2020-01-01')
    is_mask = spy_dates < is_cutoff
    oos_mask = spy_dates >= is_cutoff

    output_results = {}
    for name, r in all_results.items():
        net_ret = r['net_returns']
        gross_ret = r['gross_returns']

        # Full period
        full_metrics = compute_backtest_metrics(net_ret)
        full_sharpe, full_ci_lo, full_ci_hi = bootstrap_sharpe_ci(
            net_ret, n_bootstrap=n_bootstrap,
        )

        # IS period
        if r['asset'] == 'SPY':
            is_metrics = compute_backtest_metrics(net_ret[is_mask])
            oos_metrics = compute_backtest_metrics(net_ret[oos_mask])
        else:
            # Multi-asset may have different length
            is_metrics = compute_backtest_metrics(net_ret[:min(np.sum(is_mask), len(net_ret))])
            oos_idx = max(np.sum(is_mask), 0)
            oos_metrics = compute_backtest_metrics(net_ret[oos_idx:])

        # Turnover
        w = r['weights']
        if w.ndim == 1:
            avg_turnover = float(np.mean(np.abs(np.diff(w))))
        else:
            avg_turnover = float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1)))
        annual_turnover = avg_turnover * 252

        # Crisis returns
        if r['asset'] == 'SPY':
            crisis_rets = crisis_period_returns(net_ret, spy_dates, ALL_CRISES)
        else:
            crisis_rets = {}

        output_results[name] = {
            'full_period': full_metrics,
            'in_sample': is_metrics,
            'out_of_sample': oos_metrics,
            'sharpe_ci': {
                'point': full_sharpe,
                'ci_lo': float(full_ci_lo) if not np.isnan(full_ci_lo) else None,
                'ci_hi': float(full_ci_hi) if not np.isnan(full_ci_hi) else None,
            },
            'avg_daily_turnover': avg_turnover,
            'annual_turnover': annual_turnover,
            'total_cost_bps': float(np.sum(r['costs'])) * 10000 / max(len(r['costs']), 1),
            'crisis_returns': {k: float(v) if not np.isnan(v) else None for k, v in crisis_rets.items()},
        }

        logger.info(
            f"  {name:25s}: Sharpe={full_metrics['sharpe']:.2f} "
            f"(IS={is_metrics['sharpe']:.2f}, OOS={oos_metrics['sharpe']:.2f}), "
            f"MaxDD={full_metrics['max_drawdown']:.1%}, "
            f"Turnover={annual_turnover:.0f}x/yr"
        )

    # ---- Statistical comparisons ----
    logger.info("\n[6/6] Statistical comparisons vs ConstantVolSPY...")
    comparisons = {}
    benchmark_net = all_results['ConstantVolSPY']['net_returns']

    for name in ['GeometricLongFlat', 'GeometricLongShort']:
        strat_net = all_results[name]['net_returns']
        delta_sr, p_val = ledoit_wolf_sharpe_test(strat_net, benchmark_net)
        comparisons[name] = {
            'vs': 'ConstantVolSPY',
            'delta_sharpe': float(delta_sr),
            'p_value': float(p_val),
            'significant': p_val < 0.05,
        }
        logger.info(
            f"  {name} vs ConstantVolSPY: ΔSharpe={delta_sr:+.3f}, "
            f"p={p_val:.4f} {'*' if p_val < 0.05 else ''}"
        )

    # Break-even cost analysis
    logger.info("\n  Break-even cost analysis:")
    for name in ['GeometricLongFlat', 'GeometricLongShort']:
        gross = all_results[name]['gross_returns']
        bench_net = benchmark_net

        # At what cost level does strategy Sharpe equal benchmark Sharpe?
        turnover = compute_turnover(all_results[name]['weights'])
        bench_sharpe = output_results['ConstantVolSPY']['full_period']['sharpe']

        for test_cost in range(1, 50):
            cost_drag = turnover * test_cost / 10000.0
            test_net = gross - cost_drag
            test_sharpe = _quick_sharpe(test_net)
            if test_sharpe < bench_sharpe:
                logger.info(f"  {name}: break-even at ~{test_cost} bps")
                comparisons[name]['breakeven_bps'] = test_cost
                break
        else:
            logger.info(f"  {name}: break-even > 50 bps")
            comparisons[name]['breakeven_bps'] = 50

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':25s} {'Full SR':>8s} {'IS SR':>8s} {'OOS SR':>8s} {'MaxDD':>8s}")
    logger.info("-" * 60)

    for name, r in sorted(
        output_results.items(),
        key=lambda x: x[1]['full_period']['sharpe'],
        reverse=True,
    ):
        fp = r['full_period']
        is_p = r['in_sample']
        oos_p = r['out_of_sample']
        logger.info(
            f"  {name:25s} {fp['sharpe']:8.2f} {is_p['sharpe']:8.2f} "
            f"{oos_p['sharpe']:8.2f} {fp['max_drawdown']:7.1%}"
        )

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'target_vol': target_vol,
            'crisis_threshold': crisis_threshold,
            'is_cutoff': '2020-01-01',
            'symbols': symbols,
            'multi_symbols': multi_symbols if multi_returns is not None else None,
            'n_bootstrap': n_bootstrap,
            'cost_bps': {'SPY': 0.5, 'commission': 0.5},
        },
        'results': _make_serializable(output_results),
        'statistical_comparisons': comparisons,
        'signal_stats': {
            'n_valid': int(np.sum(~np.isnan(p_crisis))),
            'mean_p_crisis': float(np.nanmean(p_crisis)),
            'std_p_crisis': float(np.nanstd(p_crisis)),
            'pct_above_threshold': float(np.nanmean(p_crisis > crisis_threshold)),
        },
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'backtest'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'backtest_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def _quick_sharpe(returns):
    """Fast annualized Sharpe ratio computation."""
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return 0.0
    s = np.std(returns, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(returns) / s * np.sqrt(252))


def _make_serializable(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return None  # don't serialize large arrays
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PnL backtest for geometric regime detection')
    parser.add_argument('--target-vol', type=float, default=0.10,
                        help='Annualized vol target (default: 0.10)')
    parser.add_argument('--crisis-threshold', type=float, default=0.5,
                        help='P(crisis) threshold for regime adjustment (default: 0.5)')
    parser.add_argument('--quick', action='store_true',
                        help='Skip multi-asset strategies')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Bootstrap resamples (default: 10000)')
    args = parser.parse_args()

    run_backtest(
        target_vol=args.target_vol,
        crisis_threshold=args.crisis_threshold,
        quick=args.quick,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == '__main__':
    main()
