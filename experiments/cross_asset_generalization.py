"""
Cross-asset generalization test: geometric methods on bonds, commodities, FX.

Tests whether QCML geometric observables generalize to fundamentally different
asset classes where Random Forest (trained on equities) has no labeled data.

Asset Universes:
    1. Bond ETFs:     AGG, TLT, HYG, LQD  (2003-2024)
    2. Commodities:   GLD, USO             (2004-2024)
    3. FX/Dollar:     FXE, UUP             (2005-2024)
    4. Equity (ctrl): SPY, DIA             (2005-2024)

For each universe:
    - Fetch prices from Polygon
    - Build feature matrix
    - Run 3 QCML detectors + RF (trained on equity crises) + Vol Z
    - Evaluate Cohen's d on applicable crises
    - Report per-asset-class detection quality

Key hypothesis: Geometric methods transfer better because they are unsupervised
— no equity-specific training labels needed.

Usage:
    python experiments/cross_asset_generalization.py
    python experiments/cross_asset_generalization.py --quick
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
    ALL_CRISES,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    RandomForestRegimeDetector,
)
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
# Asset Universe Definitions
# =============================================================================

ASSET_UNIVERSES = {
    'equity': {
        'symbols': ['SPY', 'DIA'],
        'start': '2005-01-01',
        'end': '2024-12-31',
        'label': 'US Equity ETFs',
        'crises': list(ALL_CRISES.keys()),
    },
    'bonds': {
        'symbols': ['AGG', 'TLT', 'HYG', 'LQD'],
        'start': '2007-01-01',  # HYG launched 2007
        'end': '2024-12-31',
        'label': 'Bond ETFs',
        'crises': [
            '2008_gfc', '2011_euro', '2013_taper', '2015_china',
            '2018_q4', '2020_covid', '2022_rates', '2023_svb',
        ],
    },
    'commodities': {
        'symbols': ['GLD', 'USO'],
        'start': '2006-06-01',  # USO launched 2006
        'end': '2024-12-31',
        'label': 'Commodity ETFs',
        'crises': [
            '2008_gfc', '2014_oil', '2015_china', '2020_covid',
            '2022_rates',
        ],
    },
    'fx': {
        'symbols': ['FXE', 'UUP'],
        'start': '2007-03-01',  # UUP launched 2007
        'end': '2024-12-31',
        'label': 'FX ETFs',
        'crises': [
            '2008_gfc', '2011_euro', '2015_china', '2020_covid',
            '2022_rates', '2024_carry',
        ],
    },
}

# Asset-class-specific crisis definitions (supplement ALL_CRISES)
ASSET_SPECIFIC_CRISES = {
    '2013_taper': {
        'start': '2013-05-22', 'end': '2013-09-30',
        'label': 'Taper Tantrum 2013',
    },
    '2014_oil': {
        'start': '2014-06-20', 'end': '2015-01-31',
        'label': 'Oil Price Crash 2014',
    },
}


def get_crisis_def(crisis_key):
    """Look up crisis definition from ALL_CRISES or asset-specific."""
    if crisis_key in ALL_CRISES:
        return ALL_CRISES[crisis_key]
    if crisis_key in ASSET_SPECIFIC_CRISES:
        return ASSET_SPECIFIC_CRISES[crisis_key]
    return None


# =============================================================================
# Core Pipeline
# =============================================================================

def build_qcml_detectors():
    """Build QCML detectors with default settings."""
    common = dict(
        hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired',
        rolling_window=20, seed=42,
    )
    return [
        ('Berry Phase Rate', BerryPhaseRateDetector(**common)),
        ('QFI Determinant', QFIDeterminantDetector(**common)),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector(**common)),
    ]


def evaluate_universe(
    universe_key,
    universe_config,
    equity_rf_model=None,
    n_bootstrap=5000,
    window_size=10,
):
    """Run detection evaluation on one asset universe.

    Args:
        universe_key: Key like 'equity', 'bonds', etc.
        universe_config: Dict with symbols, start, end, crises.
        equity_rf_model: Pre-trained RF model (for cross-asset transfer test).
        n_bootstrap: Bootstrap resamples for CIs.
        window_size: Crisis window extension in days.

    Returns:
        Dict of results per method × crisis.
    """
    symbols = universe_config['symbols']
    start = universe_config['start']
    end = universe_config['end']
    crisis_keys = universe_config['crises']

    logger.info(f"\n{'='*60}")
    logger.info(f"Universe: {universe_config['label']} ({symbols})")
    logger.info(f"{'='*60}")

    # Fetch data
    try:
        raw = fetch_data(symbols, start, end)
        prices_df = raw['close'].unstack('symbol').dropna()
    except Exception as e:
        logger.error(f"  Failed to fetch {symbols}: {e}")
        return {}

    # Build features
    if len(symbols) >= 2:
        X, dates = create_feature_matrix(prices_df)
    else:
        X, dates = create_feature_matrix_single_asset(prices_df)

    if len(X) < 200:
        logger.warning(f"  Insufficient data: {len(X)} rows")
        return {}

    logger.info(f"  Feature matrix: {X.shape}, dates: {dates[0]} to {dates[-1]}")

    # Adjust n_pca_components if features are limited
    n_pca = min(15, X.shape[1] - 1)

    # Build enriched features
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]

    # Build QCML detectors (adjusted for feature count)
    common = dict(
        hilbert_dim=8, n_pca_components=n_pca, operator_method='pca_inspired',
        rolling_window=20, seed=42,
    )
    detectors = [
        ('Berry Phase Rate', BerryPhaseRateDetector(**common)),
        ('QFI Determinant', QFIDeterminantDetector(**common)),
        ('Multi-Lag Fidelity', MultiLagFidelityDetector(**common)),
        ('Rolling Vol Z', RollingVolatilityDetector(vol_window=20, min_expanding=60)),
    ]

    # Fit and score
    all_scores = {}
    for name, det in detectors:
        try:
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)
            all_scores[name] = scores
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # RF transfer test: use equity-trained RF on this universe's features
    if equity_rf_model is not None:
        try:
            rf_scores = equity_rf_model.compute_regime_scores(X)
            # Align to enriched dates
            if len(rf_scores) > len(dates_enriched):
                rf_scores = rf_scores[19:]
            all_scores['RF (equity-trained)'] = rf_scores[:len(dates_enriched)]
        except Exception as e:
            logger.warning(f"  RF transfer failed: {e}")

    # Evaluate Cohen's d per method × crisis
    results = {}
    for method_name, scores in all_scores.items():
        method_results = {}
        for ck in crisis_keys:
            ci = get_crisis_def(ck)
            if ci is None:
                continue

            cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
            ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)

            # Check crisis falls within data range
            if cs > dates_enriched[-1] or ce < dates_enriched[0]:
                continue

            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            if np.sum(crisis_mask) < 5 or np.sum(normal_mask) < 30:
                continue

            crisis_scores = scores[crisis_mask]
            normal_scores = scores[normal_mask]

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_scores, normal_scores, n_bootstrap=n_bootstrap,
            )

            method_results[ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
                'n_crisis': int(np.sum(crisis_mask)),
                'n_normal': int(np.sum(normal_mask)),
            }

        if method_results:
            ds = [v['d'] for v in method_results.values() if v['d'] is not None]
            median_d = float(np.median(ds)) if ds else None
            logger.info(f"  {method_name:25s}: median d = {median_d:.3f} "
                       f"({len(ds)} crises)" if median_d else
                       f"  {method_name:25s}: no valid crises")
            results[method_name] = {
                'per_crisis': method_results,
                'median_d': median_d,
                'n_crises_evaluated': len(ds),
            }

    return results


def train_equity_rf():
    """Train an RF model on equity data for cross-asset transfer test.

    Returns:
        Fitted RandomForestRegimeDetector.
    """
    logger.info("\n[Pre] Training RF on equity crises for transfer test...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)

    # Build labels from ALL equity crises
    y = np.zeros(len(X))
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (dates >= cs) & (dates <= ce)
        y[mask] = 1.0

    rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=42, lookback=20)
    rf.fit_with_labels(X, y)
    logger.info(f"  RF trained on {int(y.sum())} crisis days / {len(y)} total")
    return rf


# =============================================================================
# Main
# =============================================================================

def run_cross_asset(quick=False):
    """Run cross-asset generalization test.

    Args:
        quick: If True, only test equity and bonds universes.

    Returns:
        Dict with all results.
    """
    logger.info("=" * 70)
    logger.info("Cross-Asset Generalization Test")
    logger.info("=" * 70)

    # Train equity RF for transfer test
    equity_rf = train_equity_rf()

    # Select universes
    if quick:
        universe_keys = ['equity', 'bonds']
    else:
        universe_keys = list(ASSET_UNIVERSES.keys())

    all_results = {}
    for uk in universe_keys:
        config = ASSET_UNIVERSES[uk]
        results = evaluate_universe(
            uk, config,
            equity_rf_model=equity_rf if uk != 'equity' else None,
        )
        all_results[uk] = {
            'config': config,
            'results': results,
        }

    # Summary: cross-asset heatmap data
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-ASSET SUMMARY (median Cohen's d)")
    logger.info("=" * 70)

    method_names = set()
    for uk_data in all_results.values():
        method_names.update(uk_data.get('results', {}).keys())

    header = f"{'Method':30s}" + "".join(f"{uk:>12s}" for uk in universe_keys)
    logger.info(header)
    logger.info("-" * len(header))

    heatmap_data = {}
    for mname in sorted(method_names):
        row = f"{mname:30s}"
        for uk in universe_keys:
            md = all_results.get(uk, {}).get('results', {}).get(mname, {}).get('median_d')
            row += f"{md:12.3f}" if md is not None else f"{'N/A':>12s}"
        logger.info(row)
        heatmap_data[mname] = {
            uk: all_results.get(uk, {}).get('results', {}).get(mname, {}).get('median_d')
            for uk in universe_keys
        }

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick': quick,
            'universes': universe_keys,
        },
        'results': all_results,
        'heatmap': heatmap_data,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'cross_asset'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'cross_asset_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Results saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Cross-asset generalization test')
    parser.add_argument('--quick', action='store_true', help='Only test equity + bonds')
    args = parser.parse_args()
    run_cross_asset(quick=args.quick)


if __name__ == '__main__':
    main()
