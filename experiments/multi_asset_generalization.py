#!/usr/bin/env python3
"""
Multi-Asset Generalization Experiment

Tests whether QCML geometric regime detection generalizes beyond SPY
by running the top 3 methods + RF baseline on 5 broad market ETFs
individually across 4 major crises.

Key question: Are topological regime signals a property of market
microstructure, not a quirk of a single asset?

Assets tested:
    SPY  - S&P 500 (large-cap US equities)
    QQQ  - Nasdaq 100 (tech-heavy US equities)
    IWM  - Russell 2000 (small-cap US equities)
    EFA  - MSCI EAFE (international developed markets)
    DIA  - Dow Jones Industrial Average

Crises tested:
    2008 GFC, 2015 China Devaluation, 2020 COVID, 2022 Rate Hikes

Output:
    - Per-asset × per-crisis Cohen's d matrix
    - Cross-asset consistency (Spearman correlation of detection scores)
    - Statistical test: Friedman ranking of assets
    - Paired comparison: QCML vs RF across all asset-crisis pairs

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.data import PolygonDataSource, MinimalFeatureEngine
from qcml_geometry import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.baselines import RandomForestRegimeDetector
from experiments.crisis_config import (
    CrisisDefinition,
    CrisisType,
    ValidationConfig,
    get_default_validation_config,
)
from experiments.regime_comparison import evaluate_method

load_dotenv(project_root / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / 'experiments' / 'outputs' / 'regime_detection' / 'multi_asset'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Configuration
# ============================================================================

# Assets to test (broad market ETFs with sufficient history)
ASSETS = ['SPY', 'QQQ', 'IWM', 'EFA', 'DIA']

# Major crises where all ETFs have data
GENERALIZATION_CRISES = [
    CrisisDefinition(
        name="2008_crisis",
        crisis_date="2008-09-15",
        description="Lehman Brothers collapse - GFC",
        universe=["SPY"],  # Will be replaced per-asset
        crisis_type=CrisisType.FINANCIAL,
        expected_lead_days=10,
        lookback_months=6,
        lookahead_months=6,
    ),
    CrisisDefinition(
        name="2015_china",
        crisis_date="2015-08-24",
        description="China devaluation selloff",
        universe=["SPY"],
        crisis_type=CrisisType.GEOPOLITICAL,
        expected_lead_days=10,
        lookback_months=6,
        lookahead_months=6,
    ),
    CrisisDefinition(
        name="2020_covid",
        crisis_date="2020-03-16",
        description="COVID-19 pandemic crash",
        universe=["SPY"],
        crisis_type=CrisisType.PANDEMIC,
        expected_lead_days=10,
        lookback_months=6,
        lookahead_months=6,
    ),
    CrisisDefinition(
        name="2022_rates",
        crisis_date="2022-03-16",
        description="Fed rate hike regime shift",
        universe=["SPY"],
        crisis_type=CrisisType.MONETARY,
        expected_lead_days=10,
        lookback_months=6,
        lookahead_months=6,
    ),
]

# Causal-optimized hyperparameters (from Optuna Phase A)
QCML_CONFIGS = {
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': {
            'hilbert_dim': 8,
            'n_pca_components': 15,
            'operator_method': 'pca_inspired',
            'rolling_window': 30,
            'seed': 42,
        },
    },
    'QFI Determinant': {
        'class': QFIDeterminantDetector,
        'params': {
            'hilbert_dim': 8,
            'n_pca_components': 10,
            'operator_method': 'pca_inspired',
            'rolling_window': 30,
            'seed': 42,
        },
    },
    'Multi-Lag Fidelity': {
        'class': MultiLagFidelityDetector,
        'params': {
            'hilbert_dim': 8,
            'n_pca_components': 15,
            'operator_method': 'pca_inspired',
            'rolling_window': 30,
            'seed': 42,
        },
    },
}


# ============================================================================
# Data fetching — single asset per crisis
# ============================================================================

def fetch_single_asset_data(
    asset: str,
    crisis: CrisisDefinition,
    config: ValidationConfig,
    enriched_lookback: int = 20,
) -> Optional[Dict[str, Any]]:
    """Fetch and prepare data for a single asset around a crisis period.

    Creates features from the single asset's price series (returns, volatility,
    volume features) rather than the multi-asset universe.

    Returns:
        Dict with X_enriched, times_enriched, crisis_idx_enriched, or None.
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found")

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    source = PolygonDataSource(api_key=api_key)
    try:
        raw_data = source.fetch_equities(
            [asset],
            str(start_date.date()),
            str(end_date.date()),
            timeframe="1d",
        )
    except Exception as e:
        logger.warning(f"Failed to fetch {asset} for {crisis.name}: {e}")
        return None

    if raw_data.empty:
        logger.warning(f"No data for {asset} during {crisis.name}")
        return None

    prices = raw_data['close'].unstack(level=0)
    prices = prices.ffill()

    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=asset)
    features = features.dropna()

    if len(features) < 50:
        logger.warning(f"Insufficient data for {asset}/{crisis.name}: {len(features)} rows")
        return None

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X_raw = features.values
    times = features.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(config.n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=enriched_lookback)

    trim = enriched_lookback - 1
    times_enriched = times[trim:]
    crisis_ts = pd.Timestamp(crisis.crisis_date)
    crisis_idx = int((times >= crisis_ts).argmax())
    crisis_idx_enriched = max(0, crisis_idx - trim)

    return {
        'X_enriched': X_enriched,
        'times_enriched': times_enriched,
        'crisis_idx_enriched': crisis_idx_enriched,
        'crisis': crisis,
        'asset': asset,
        'n_features': X_raw.shape[1],
        'n_pca': n_components,
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_single_pair(
    asset: str,
    crisis: CrisisDefinition,
    method_name: str,
    method_config: Dict,
    data: Dict,
    config: ValidationConfig,
) -> Dict[str, Any]:
    """Evaluate one method on one asset-crisis pair."""
    DetectorClass = method_config['class']
    params = method_config['params'].copy()

    detector = DetectorClass(**params)
    detector.fit(data['X_enriched'])

    result = evaluate_method(
        detector,
        data['X_enriched'],
        data['times_enriched'],
        data['crisis_idx_enriched'],
        crisis,
        config,
        n_bootstrap=500,
        n_permutations=200,
        seed=42,
    )

    return {
        'asset': asset,
        'crisis': crisis.name,
        'method': method_name,
        'effect_size_d': result.get('effect_size_d_normalized', 0.0),
        'p_value': result.get('p_value', 1.0),
    }


def evaluate_rf_single(
    asset: str,
    crisis: CrisisDefinition,
    all_data: Dict[str, Dict[str, Dict]],
    config: ValidationConfig,
) -> Dict[str, Any]:
    """Evaluate RF baseline on one asset-crisis pair (leave-one-crisis-out)."""
    # Train on all other crises for this asset
    train_X = []
    train_y = []

    for other_crisis_name, crisis_data in all_data.get(asset, {}).items():
        if other_crisis_name == crisis.name or crisis_data is None:
            continue
        cd = crisis_data
        X_e = cd['X_enriched']
        ci = cd['crisis_idx_enriched']
        n = len(X_e)

        y = np.zeros(n)
        crisis_start = max(0, ci - config.analysis_window_days)
        crisis_end = min(n, ci + config.analysis_window_days)
        y[crisis_start:crisis_end] = 1.0

        train_X.append(X_e)
        train_y.append(y)

    if len(train_X) < 2:
        return {
            'asset': asset, 'crisis': crisis.name, 'method': 'Random Forest',
            'effect_size_d': 0.0, 'p_value': 1.0,
        }

    X_train = np.vstack(train_X)
    y_train = np.concatenate(train_y)

    rf = RandomForestRegimeDetector(seed=42)
    rf.fit_with_labels(X_train, y_train)

    test_data = all_data[asset][crisis.name]
    result = evaluate_method(
        rf,
        test_data['X_enriched'],
        test_data['times_enriched'],
        test_data['crisis_idx_enriched'],
        crisis,
        config,
        n_bootstrap=500,
        n_permutations=200,
        seed=42,
    )

    return {
        'asset': asset,
        'crisis': crisis.name,
        'method': 'Random Forest',
        'effect_size_d': result.get('effect_size_d_normalized', 0.0),
        'p_value': result.get('p_value', 1.0),
    }


# ============================================================================
# Main pipeline
# ============================================================================

def run_multi_asset_experiment(quick: bool = False):
    """Run the full multi-asset generalization experiment."""
    config = get_default_validation_config()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    assets = ASSETS[:3] if quick else ASSETS
    crises = GENERALIZATION_CRISES[:2] if quick else GENERALIZATION_CRISES

    print("=" * 70)
    print("MULTI-ASSET GENERALIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Assets: {assets}")
    print(f"Crises: {[c.name for c in crises]}")
    print(f"Methods: {list(QCML_CONFIGS.keys())} + Random Forest")
    print(f"Total evaluations: {len(assets)} × {len(crises)} × 4 = {len(assets) * len(crises) * 4}")
    print()

    # Phase 1: Fetch all data
    print("Phase 1: Fetching data for all asset-crisis pairs...")
    all_data: Dict[str, Dict[str, Optional[Dict]]] = {}
    t0 = time.time()

    for asset in assets:
        all_data[asset] = {}
        for crisis in crises:
            data = fetch_single_asset_data(asset, crisis, config)
            all_data[asset][crisis.name] = data
            status = f"T={len(data['X_enriched'])}" if data else "FAILED"
            print(f"  {asset}/{crisis.name}: {status}")

    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    # Phase 2: Evaluate QCML methods
    print("Phase 2: Evaluating QCML methods...")
    results = []

    for asset in assets:
        for crisis in crises:
            data = all_data[asset].get(crisis.name)
            if data is None:
                continue

            for method_name, method_config in QCML_CONFIGS.items():
                try:
                    result = evaluate_single_pair(
                        asset, crisis, method_name, method_config, data, config
                    )
                    results.append(result)
                    print(f"  {asset}/{crisis.name}/{method_name}: d={result['effect_size_d']:.3f}")
                except Exception as e:
                    logger.warning(f"  {asset}/{crisis.name}/{method_name} failed: {e}")
                    results.append({
                        'asset': asset, 'crisis': crisis.name,
                        'method': method_name, 'effect_size_d': 0.0, 'p_value': 1.0,
                    })

    # Phase 3: Evaluate RF baseline
    print("\nPhase 3: Evaluating RF baseline (leave-one-crisis-out)...")
    for asset in assets:
        for crisis in crises:
            if all_data[asset].get(crisis.name) is None:
                continue
            try:
                result = evaluate_rf_single(asset, crisis, all_data, config)
                results.append(result)
                print(f"  {asset}/{crisis.name}/RF: d={result['effect_size_d']:.3f}")
            except Exception as e:
                logger.warning(f"  {asset}/{crisis.name}/RF failed: {e}")
                results.append({
                    'asset': asset, 'crisis': crisis.name,
                    'method': 'Random Forest', 'effect_size_d': 0.0, 'p_value': 1.0,
                })

    # Phase 4: Statistical analysis
    print("\n" + "=" * 70)
    print("MULTI-ASSET GENERALIZATION RESULTS")
    print("=" * 70)

    df = pd.DataFrame(results)

    # Per-method mean d across all assets × crises
    method_summary = df.groupby('method')['effect_size_d'].agg(['mean', 'median', 'std', 'count'])
    print("\n--- Method Summary (all assets × crises) ---")
    print(method_summary.to_string())

    # Per-asset mean d (aggregating across crises and methods)
    asset_summary = df.groupby(['asset', 'method'])['effect_size_d'].mean().unstack('method')
    print("\n--- Per-Asset Performance (mean d) ---")
    print(asset_summary.to_string(float_format='{:.3f}'.format))

    # Cross-asset consistency: for each method, compute Spearman correlation
    # of per-crisis d values between SPY and other assets
    print("\n--- Cross-Asset Consistency (Spearman rho vs SPY) ---")
    spy_data = df[df['asset'] == 'SPY']
    for method in QCML_CONFIGS:
        spy_d = spy_data[spy_data['method'] == method].set_index('crisis')['effect_size_d']
        for other_asset in [a for a in assets if a != 'SPY']:
            other_data = df[(df['asset'] == other_asset) & (df['method'] == method)]
            other_d = other_data.set_index('crisis')['effect_size_d']
            common = spy_d.index.intersection(other_d.index)
            if len(common) >= 3:
                rho, p = stats.spearmanr(spy_d[common], other_d[common])
                print(f"  {method} SPY↔{other_asset}: rho={rho:.3f}, p={p:.3f}")

    # Wilcoxon: QCML vs RF across all asset-crisis pairs
    print("\n--- QCML vs RF (Wilcoxon signed-rank, all asset-crisis pairs) ---")
    rf_d = df[df['method'] == 'Random Forest'].set_index(['asset', 'crisis'])['effect_size_d']

    for method in QCML_CONFIGS:
        qcml_d = df[df['method'] == method].set_index(['asset', 'crisis'])['effect_size_d']
        common = rf_d.index.intersection(qcml_d.index)
        if len(common) >= 5:
            stat, p = stats.wilcoxon(qcml_d[common], rf_d[common], alternative='greater')
            wins = (qcml_d[common] > rf_d[common]).sum()
            losses = (qcml_d[common] < rf_d[common]).sum()
            ties = (qcml_d[common] == rf_d[common]).sum()
            mean_diff = (qcml_d[common] - rf_d[common]).mean()
            print(
                f"  {method}: W={wins}, L={losses}, T={ties}, "
                f"mean_diff={mean_diff:+.3f}, p={p:.4f} "
                f"{'*' if p < 0.05 else '†' if p < 0.1 else 'n.s.'}"
            )

    # Friedman test across all methods
    print("\n--- Friedman Test (method ranking across asset-crisis pairs) ---")
    pivot = df.pivot_table(
        values='effect_size_d',
        index=['asset', 'crisis'],
        columns='method',
    ).dropna()

    if len(pivot) >= 4:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[pivot[col].values for col in pivot.columns]
        )
        print(f"  chi-sq={friedman_stat:.2f}, p={friedman_p:.4f}")

        # Mean ranks
        ranks = pivot.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean()
        print(f"  Mean ranks (lower=better):")
        for method, rank in mean_ranks.sort_values().items():
            print(f"    {method}: {rank:.2f}")

    # SPY-only vs multi-asset comparison
    print("\n--- SPY-Only vs Multi-Asset Mean d ---")
    for method in list(QCML_CONFIGS.keys()) + ['Random Forest']:
        spy_mean = df[(df['method'] == method) & (df['asset'] == 'SPY')]['effect_size_d'].mean()
        all_mean = df[df['method'] == method]['effect_size_d'].mean()
        non_spy = df[(df['method'] == method) & (df['asset'] != 'SPY')]['effect_size_d'].mean()
        print(f"  {method}: SPY={spy_mean:.3f}, non-SPY={non_spy:.3f}, all={all_mean:.3f}")

    # Save results
    output = {
        'timestamp': timestamp,
        'assets': assets,
        'crises': [c.name for c in crises],
        'methods': list(QCML_CONFIGS.keys()) + ['Random Forest'],
        'results': results,
        'method_summary': method_summary.to_dict(),
        'asset_method_matrix': asset_summary.to_dict() if asset_summary is not None else {},
    }

    out_path = OUTPUT_DIR / f'multi_asset_results_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Multi-Asset Generalization Experiment')
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: 3 assets × 2 crises (vs 5 × 4)',
    )
    args = parser.parse_args()

    run_multi_asset_experiment(quick=args.quick)


if __name__ == '__main__':
    main()
