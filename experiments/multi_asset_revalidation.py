"""
Multi-asset re-validation using the current causal pipeline.

Runs Berry/QFI/MLF/RF on 5 equity ETFs (SPY, QQQ, IWM, EFA, DIA) × 4 crises
using per-crisis causal preprocessing (scaler/PCA/operators fitted pre-crisis).

Usage:
    python experiments/multi_asset_revalidation.py
"""

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
    fetch_data,
    create_feature_matrix,
    create_feature_matrix_single_asset,
)
from experiments.baselines import RandomForestRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# Same HPO configs as regime_comparison.py
QCML_CONFIGS = {
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': dict(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
            normalization='sphere', berry_aggregation='f01',
        ),
    },
    'QFI Determinant': {
        'class': QFIDeterminantDetector,
        'params': dict(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            normalization='soft', qfi_mode='logdet',
            adaptive_epsilon=True,
        ),
    },
    'Multi-Lag Fidelity': {
        'class': MultiLagFidelityDetector,
        'params': dict(
            hilbert_dim=4, n_pca_components=8, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            normalization='sphere',
        ),
    },
}

# 4 representative crises for multi-asset test
CRISES = {
    '2008_gfc': {'start': '2008-09-15', 'end': '2009-03-09'},
    '2020_covid': {'start': '2020-02-20', 'end': '2020-03-23'},
    '2022_rates': {'start': '2022-01-03', 'end': '2022-10-12'},
    '2023_svb': {'start': '2023-03-08', 'end': '2023-03-20'},
}

# All 12 crises for RF LOCO labels
ALL_CRISES_FOR_LABELS = {
    '2007_quant': {'start': '2007-08-01', 'end': '2007-08-16'},
    '2008_gfc': {'start': '2008-09-15', 'end': '2009-03-09'},
    '2010_flash': {'start': '2010-05-06', 'end': '2010-05-06'},
    '2011_euro': {'start': '2011-08-05', 'end': '2011-10-03'},
    '2015_china': {'start': '2015-08-18', 'end': '2015-08-25'},
    '2018_volmageddon': {'start': '2018-02-02', 'end': '2018-02-09'},
    '2018_q4': {'start': '2018-10-03', 'end': '2018-12-24'},
    '2019_repo': {'start': '2019-09-16', 'end': '2019-09-17'},
    '2020_covid': {'start': '2020-02-20', 'end': '2020-03-23'},
    '2022_rates': {'start': '2022-01-03', 'end': '2022-10-12'},
    '2023_svb': {'start': '2023-03-08', 'end': '2023-03-20'},
    '2024_carry': {'start': '2024-07-31', 'end': '2024-08-05'},
}

ASSETS = ['SPY', 'QQQ', 'IWM', 'EFA', 'DIA']
WINDOW_SIZE = 10


def compute_cohens_d(crisis_scores, normal_scores):
    """Cohen's d between crisis and normal score distributions."""
    c = crisis_scores[~np.isnan(crisis_scores)]
    n = normal_scores[~np.isnan(normal_scores)]
    if len(c) < 2 or len(n) < 2:
        return 0.0
    pooled_std = np.sqrt(((len(c) - 1) * np.var(c, ddof=1) +
                           (len(n) - 1) * np.var(n, ddof=1)) /
                          (len(c) + len(n) - 2))
    if pooled_std < 1e-12:
        return 0.0
    return abs(np.mean(c) - np.mean(n)) / pooled_std


def run_single_asset(symbol):
    """Run all methods on a single asset across 4 crises."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Asset: {symbol}")
    logger.info(f"{'='*60}")

    # Fetch data — use 2-symbol universe for feature matrix if possible
    # But for non-SPY assets, use single-asset feature matrix
    try:
        if symbol in ('SPY', 'DIA'):
            raw = fetch_data(['SPY', 'DIA'], '1995-01-01', '2024-12-31')
            prices_df = raw['close'].unstack('symbol').dropna()
            X, dates = create_feature_matrix(prices_df)
        else:
            # Single-asset: fetch with SPY as pair for cross-sectional features
            raw = fetch_data([symbol, 'SPY'], '1995-01-01', '2024-12-31')
            prices_df = raw['close'].unstack('symbol').dropna()
            X, dates = create_feature_matrix(prices_df)
    except Exception as e:
        logger.warning(f"  Multi-symbol failed for {symbol}: {e}, trying single-asset")
        raw = fetch_data([symbol], '2000-01-01', '2024-12-31')
        prices_df = raw['close'].unstack('symbol').dropna()
        X, dates = create_feature_matrix_single_asset(prices_df[symbol])

    # Build enriched features
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Features: {X.shape} -> enriched: {X_enriched.shape}")

    results = {}

    for crisis_key, crisis_info in CRISES.items():
        crisis_start = pd.Timestamp(crisis_info['start'])
        crisis_end = pd.Timestamp(crisis_info['end'])

        # Causal cutoff
        cutoff_date = crisis_start - pd.Timedelta(days=WINDOW_SIZE)
        fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

        if fit_end_idx < 100:
            logger.warning(f"  {crisis_key}: insufficient pre-crisis data ({fit_end_idx}), skipping")
            continue

        # Crisis mask on enriched dates
        crisis_mask = (dates_enriched >= crisis_start) & (dates_enriched <= crisis_end)
        if crisis_mask.sum() == 0:
            logger.warning(f"  {crisis_key}: no data in crisis window for {symbol}")
            continue

        normal_mask = ~crisis_mask & (np.arange(len(dates_enriched)) >= fit_end_idx)

        # -- QCML methods --
        for method_name, config in QCML_CONFIGS.items():
            try:
                params = {**config['params'], 'causal_fit_length': fit_end_idx}
                det = config['class'](**params)
                det.fit(X_enriched)
                scores = det.compute_regime_scores(X_enriched)

                d = compute_cohens_d(scores[crisis_mask], scores[normal_mask])
                results.setdefault(method_name, {})[crisis_key] = round(d, 2)
                logger.info(f"  {method_name} / {crisis_key}: d={d:.2f}")
            except Exception as e:
                logger.warning(f"  {method_name} / {crisis_key} failed: {e}")
                results.setdefault(method_name, {})[crisis_key] = 0.0

        # -- RF (LOCO) --
        try:
            # Build labels from all crises except held-out
            y = np.zeros(len(X))
            for ck, ci in ALL_CRISES_FOR_LABELS.items():
                if ck == crisis_key:
                    continue
                mask = (dates >= pd.Timestamp(ci['start'])) & (dates <= pd.Timestamp(ci['end']))
                y[mask] = 1.0

            fit_end_raw = int(np.searchsorted(dates, cutoff_date))
            rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
            y_train = y[:fit_end_raw]

            if np.sum(y_train) > 0:
                rf.fit_with_labels(X[:fit_end_raw], y_train)
                scores = rf.compute_regime_scores(X)
                rf_scores = scores[19:]  # Align with enriched

                d = compute_cohens_d(rf_scores[crisis_mask], rf_scores[normal_mask])
            else:
                d = 0.0

            results.setdefault('Random Forest', {})[crisis_key] = round(d, 2)
            logger.info(f"  Random Forest / {crisis_key}: d={d:.2f}")
        except Exception as e:
            logger.warning(f"  Random Forest / {crisis_key} failed: {e}")
            results.setdefault('Random Forest', {})[crisis_key] = 0.0

    return results


def main():
    all_results = {}

    for symbol in ASSETS:
        all_results[symbol] = run_single_asset(symbol)

    # Build summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Mean Cohen's d by Asset")
    logger.info("=" * 70)

    methods = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity', 'Random Forest']
    header = f"{'Asset':<8}" + "".join(f"{m:<20}" for m in methods)
    logger.info(header)

    method_means_all = {m: [] for m in methods}

    for symbol in ASSETS:
        row = f"{symbol:<8}"
        for m in methods:
            crises_d = all_results[symbol].get(m, {})
            vals = [v for v in crises_d.values() if v is not None]
            mean_d = np.mean(vals) if vals else 0.0
            method_means_all[m].append(mean_d)
            row += f"{mean_d:<20.2f}"
        logger.info(row)

    logger.info("-" * 70)
    row = f"{'Overall':<8}"
    for m in methods:
        overall = np.mean(method_means_all[m])
        row += f"{overall:<20.2f}"
    logger.info(row)

    # Wilcoxon test: MLF vs RF
    mlf_vals = method_means_all['Multi-Lag Fidelity']
    rf_vals = method_means_all['Random Forest']
    if len(mlf_vals) >= 5:
        stat, p = stats.wilcoxon(mlf_vals, rf_vals)
        logger.info(f"\nWilcoxon MLF vs RF: stat={stat:.1f}, p={p:.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'causal_per_crisis',
        'assets': ASSETS,
        'crises': list(CRISES.keys()),
        'per_asset': all_results,
        'summary': {
            m: {
                'per_asset': dict(zip(ASSETS, method_means_all[m])),
                'overall_mean': float(np.mean(method_means_all[m])),
            }
            for m in methods
        },
    }

    outdir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = outdir / f'multi_asset_revalidation_{ts}.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to {outfile}")
    return output


if __name__ == '__main__':
    main()
