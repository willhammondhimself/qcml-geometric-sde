"""
Backtest sensitivity analysis: parameter sweep for robustness.

Sweeps over vol target and crisis threshold to show Sharpe isn't
cherry-picked from one set of parameters.

Usage:
    python experiments/backtest/sensitivity.py
"""

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

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.backtest.strategies import geometric_long_flat
from experiments.backtest.benchmarks import constant_vol_spy
from experiments.backtest.execution import apply_transaction_costs
from experiments.backtest.metrics import compute_backtest_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def run_sensitivity():
    """Sweep vol target x crisis threshold -> net Sharpe heatmap."""
    logger.info("=" * 70)
    logger.info("BACKTEST SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    # ---- Data ----
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates[19:]

    spy_prices = prices_df['SPY'].reindex(dates_enriched).dropna()
    spy_returns = spy_prices.pct_change().fillna(0).values
    spy_dates = spy_prices.index

    T = min(len(X_enriched), len(spy_returns))
    X_enriched = X_enriched[:T]
    spy_returns = spy_returns[:T]
    spy_dates = spy_dates[:T]
    dates_enriched = dates_enriched[:T]

    crisis_labels = np.zeros(T)
    for ci in ALL_CRISES.values():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (spy_dates >= cs) & (spy_dates <= ce)
        crisis_labels[mask] = 1.0

    # ---- Generate signal once using ensemble ----
    logger.info("Generating P(crisis) signal (ensemble)...")
    feat_computer = OnlineGeometricFeatureComputer(
        hilbert_dim=8, n_pca_components=15,
        operator_method='pca_inspired', seed=42,
    )
    bayesian = OnlineBayesianDetector(transition_prob=0.02, persistence=0.95)
    hmm = OnlineHMMDetector(seed=42)
    percentile = ExpandingPercentileDetector(min_history=60)
    ensemble = OnlineEnsembleDetector(
        detectors=[bayesian, hmm, percentile],
        weights=[0.4, 0.4, 0.2],
    )

    p_crisis = np.full(T, np.nan)
    for t in range(T):
        features = feat_computer.update(X_enriched[t])
        p_crisis[t] = ensemble.update(features)

    # ---- Parameter grid ----
    vol_targets = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # IS / OOS split
    is_cutoff = pd.Timestamp('2020-01-01')
    is_mask = spy_dates < is_cutoff
    oos_mask = spy_dates >= is_cutoff

    results = {}
    logger.info(f"Sweeping {len(vol_targets)} vol targets x {len(thresholds)} thresholds...")

    for vt in vol_targets:
        for ct in thresholds:
            w, _ = geometric_long_flat(
                spy_returns, p_crisis,
                target_vol=vt, crisis_threshold=ct,
            )
            gross = spy_returns * w
            net, _ = apply_transaction_costs(spy_returns, w, cost_bps=0.5)

            full = compute_backtest_metrics(net)
            is_m = compute_backtest_metrics(net[is_mask])
            oos_m = compute_backtest_metrics(net[oos_mask])

            # Benchmark: constant vol at same target
            w_bm = constant_vol_spy(spy_returns, target_vol=vt)
            bm_net, _ = apply_transaction_costs(spy_returns, w_bm, cost_bps=0.5)
            bm_full = compute_backtest_metrics(bm_net)

            key = f"vt{vt:.2f}_ct{ct:.2f}"
            results[key] = {
                'vol_target': vt,
                'crisis_threshold': ct,
                'net_sharpe': full['sharpe'],
                'is_sharpe': is_m['sharpe'],
                'oos_sharpe': oos_m['sharpe'],
                'max_drawdown': full['max_drawdown'],
                'benchmark_sharpe': bm_full['sharpe'],
                'alpha_sharpe': full['sharpe'] - bm_full['sharpe'],
            }

    # ---- Summary ----
    logger.info("\nSharpe (net) by vol_target x crisis_threshold:")
    logger.info(f"{'':12s}" + "".join(f"ct={ct:.1f}  " for ct in thresholds))

    for vt in vol_targets:
        row = f"vt={vt:.2f}    "
        for ct in thresholds:
            key = f"vt{vt:.2f}_ct{ct:.2f}"
            sr = results[key]['net_sharpe']
            row += f"{sr:7.2f} "
        logger.info(row)

    logger.info("\nAlpha Sharpe (vs ConstantVol benchmark):")
    logger.info(f"{'':12s}" + "".join(f"ct={ct:.1f}  " for ct in thresholds))

    for vt in vol_targets:
        row = f"vt={vt:.2f}    "
        for ct in thresholds:
            key = f"vt{vt:.2f}_ct{ct:.2f}"
            alpha = results[key]['alpha_sharpe']
            row += f"{alpha:+7.2f} "
        logger.info(row)

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'grid': {'vol_targets': vol_targets, 'thresholds': thresholds},
        'results': results,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'backtest'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'sensitivity_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {out_path}")
    return output


if __name__ == '__main__':
    run_sensitivity()
