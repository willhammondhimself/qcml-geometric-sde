"""
Feature engineering ablation study for QCML regime detection.

Tests how feature set, symbol universe, and Hilbert dimension affect
detection performance. Uses per-crisis causal fitting throughout.

Ablation axes:
    1. Feature set: minimal (current) vs. enriched (multi-horizon vol, skew, kurt)
    2. Symbol universe: 2 symbols (SPY+DIA) vs. 5 (SPY+DIA+QQQ+IWM+EFA)
    3. Hilbert dimension: 4, 6, 8, 12
    4. Operator method: random vs. pca_inspired

Usage:
    python experiments/feature_engineering_ablation.py
    python experiments/feature_engineering_ablation.py --quick
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

EXTENSION_DAYS = 10


# =============================================================================
# Feature Engineering Functions
# =============================================================================

def create_minimal_features(prices_df):
    """Original minimal feature set: ret, vol5, vol20, mom5, mom20 + cross-sectional.

    This is the current default from data_loader.create_feature_matrix.
    """
    log_ret = np.log(prices_df / prices_df.shift(1))

    features = {}
    for col in prices_df.columns:
        features[f'{col}_ret'] = log_ret[col]
        features[f'{col}_vol5'] = log_ret[col].rolling(5).std()
        features[f'{col}_vol20'] = log_ret[col].rolling(20).std()
        features[f'{col}_mom5'] = prices_df[col].pct_change(5)
        features[f'{col}_mom20'] = prices_df[col].pct_change(20)

    if len(prices_df.columns) > 1:
        features['cross_corr5'] = (
            log_ret.rolling(5).corr().groupby(level=0).mean().mean(axis=1)
        )
        features['cross_vol_disp'] = log_ret.rolling(20).std().std(axis=1)
        features['avg_ret'] = log_ret.mean(axis=1)

    feat_df = pd.DataFrame(features)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feat_df.values, feat_df.index


def create_enriched_features(prices_df):
    """Enriched feature set: minimal + multi-horizon vol, skew, kurtosis, range.

    Adds:
    - vol10, vol40, vol60 (multi-horizon realized volatility)
    - skew20 (rolling skewness — tail asymmetry)
    - kurt20 (rolling kurtosis — tail heaviness)
    - range20 (max/min ratio — intra-period range)
    - volume profile features if volume data available
    - cross-sectional dispersion at multiple horizons
    """
    log_ret = np.log(prices_df / prices_df.shift(1))

    features = {}
    for col in prices_df.columns:
        # Base features (same as minimal)
        features[f'{col}_ret'] = log_ret[col]
        features[f'{col}_vol5'] = log_ret[col].rolling(5).std()
        features[f'{col}_vol20'] = log_ret[col].rolling(20).std()
        features[f'{col}_mom5'] = prices_df[col].pct_change(5)
        features[f'{col}_mom20'] = prices_df[col].pct_change(20)

        # Additional horizons
        features[f'{col}_vol10'] = log_ret[col].rolling(10).std()
        features[f'{col}_vol40'] = log_ret[col].rolling(40).std()
        features[f'{col}_vol60'] = log_ret[col].rolling(60).std()
        features[f'{col}_mom10'] = prices_df[col].pct_change(10)
        features[f'{col}_mom40'] = prices_df[col].pct_change(40)

        # Higher-order moments
        features[f'{col}_skew20'] = log_ret[col].rolling(20).skew()
        features[f'{col}_kurt20'] = log_ret[col].rolling(20).kurt()

        # Range
        features[f'{col}_range20'] = (
            prices_df[col].rolling(20).max() / prices_df[col].rolling(20).min() - 1
        )

        # Squared returns (proxy for realized variance)
        features[f'{col}_ret_sq'] = log_ret[col] ** 2

    if len(prices_df.columns) > 1:
        features['cross_corr5'] = (
            log_ret.rolling(5).corr().groupby(level=0).mean().mean(axis=1)
        )
        features['cross_corr20'] = (
            log_ret.rolling(20).corr().groupby(level=0).mean().mean(axis=1)
        )
        features['cross_vol_disp5'] = log_ret.rolling(5).std().std(axis=1)
        features['cross_vol_disp20'] = log_ret.rolling(20).std().std(axis=1)
        features['cross_vol_disp60'] = log_ret.rolling(60).std().std(axis=1)
        features['avg_ret'] = log_ret.mean(axis=1)
        features['ret_dispersion'] = log_ret.std(axis=1)

    feat_df = pd.DataFrame(features)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feat_df.values, feat_df.index


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_config(
    X_enriched, dates_enriched, detector_class, params, crisis_keys,
    n_bootstrap=1000,
):
    """Evaluate a detector config with per-crisis causal fitting.

    Args:
        X_enriched: Enriched feature matrix.
        dates_enriched: DatetimeIndex.
        detector_class: Detector class.
        params: Base detector params (causal_fit_length will be set per-crisis).
        crisis_keys: List of crisis keys to evaluate.
        n_bootstrap: Bootstrap resamples.

    Returns:
        results: Dict {crisis_key: d}.
    """
    results = {}
    for ck in crisis_keys:
        ci = ALL_CRISES[ck]
        crisis_start = pd.Timestamp(ci['start'])
        cutoff_date = crisis_start - pd.Timedelta(days=EXTENSION_DAYS)
        fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

        if fit_end_idx < 100:
            continue

        try:
            det = detector_class(**{**params, 'causal_fit_length': fit_end_idx})
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)

            cs = crisis_start - pd.Timedelta(days=EXTENSION_DAYS)
            ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS)
            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            d, _, _ = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
            )
            if not np.isnan(d):
                results[ck] = float(d)
        except Exception as e:
            logger.debug(f"  Failed {ck}: {e}")

    return results


def run_ablation(quick=False):
    """Run feature engineering ablation study."""
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING ABLATION STUDY")
    logger.info("=" * 70)

    # --- Ablation configurations ---
    symbol_universes = {
        '2_sym': ['SPY', 'DIA'],
        '5_sym': ['SPY', 'DIA', 'QQQ', 'IWM', 'EFA'],
    }

    feature_sets = {
        'minimal': create_minimal_features,
        'enriched': create_enriched_features,
    }

    hilbert_dims = [4, 6, 8] if quick else [4, 6, 8, 12]

    operator_methods = ['random', 'pca_inspired']

    # Reference detectors: Berry (best individual), MLF (most robust)
    detectors = {
        'berry': {
            'class': BerryPhaseRateDetector,
            'base_params': dict(
                n_pca_components=8, rolling_window=15, seed=42,
                normalization='sphere', berry_aggregation='f01',
            ),
        },
        'mlf': {
            'class': MultiLagFidelityDetector,
            'base_params': dict(
                n_pca_components=8, rolling_window=20, seed=42,
                normalization='sphere',
            ),
        },
    }

    # Crisis selection
    if quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = [k for k in ALL_CRISES if int(k[:4]) >= 2005]

    # --- Fetch all symbol universes ---
    logger.info("\n[1] Fetching data for all symbol universes...")
    all_data = {}
    for uni_name, symbols in symbol_universes.items():
        logger.info(f"  {uni_name}: {symbols}")
        raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
        prices_df = raw['close'].unstack('symbol').dropna()
        all_data[uni_name] = prices_df

    # --- Run ablation ---
    logger.info(f"\n[2] Running ablation ({len(symbol_universes)} universes × "
                f"{len(feature_sets)} feature sets × {len(hilbert_dims)} dims × "
                f"{len(operator_methods)} operators × {len(detectors)} detectors)...")

    results = []
    total_configs = (
        len(symbol_universes) * len(feature_sets) * len(hilbert_dims) *
        len(operator_methods) * len(detectors)
    )
    config_idx = 0

    for uni_name in symbol_universes:
        prices_df = all_data[uni_name]

        for feat_name, feat_fn in feature_sets.items():
            X, dates = feat_fn(prices_df)
            X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
            dates_enriched = dates[19:]

            for h_dim in hilbert_dims:
                for op_method in operator_methods:
                    for det_key, det_config in detectors.items():
                        config_idx += 1
                        if config_idx % 5 == 0:
                            logger.info(f"  Config {config_idx}/{total_configs}...")

                        # Cap n_pca to feature dimension
                        n_pca = min(
                            det_config['base_params']['n_pca_components'],
                            X_enriched.shape[1],
                        )

                        params = {
                            **det_config['base_params'],
                            'hilbert_dim': h_dim,
                            'n_pca_components': n_pca,
                            'operator_method': op_method,
                        }

                        crisis_ds = evaluate_config(
                            X_enriched, dates_enriched,
                            det_config['class'], params,
                            crisis_keys, n_bootstrap=500,
                        )

                        if crisis_ds:
                            ds = list(crisis_ds.values())
                            results.append({
                                'universe': uni_name,
                                'features': feat_name,
                                'hilbert_dim': h_dim,
                                'operator': op_method,
                                'detector': det_key,
                                'n_features_raw': X.shape[1],
                                'n_features_enriched': X_enriched.shape[1],
                                'n_crises': len(crisis_ds),
                                'mean_d': float(np.mean(ds)),
                                'median_d': float(np.median(ds)),
                                'std_d': float(np.std(ds)),
                                'per_crisis': crisis_ds,
                            })

    # --- Analysis ---
    logger.info("\n[3] Analyzing results...")

    if not results:
        logger.error("  No results! Something went wrong.")
        return {}

    df = pd.DataFrame(results)

    # Best overall config
    best = df.loc[df['median_d'].idxmax()]
    logger.info(f"\n  Best config: {best['detector']} / {best['features']} / "
                f"{best['universe']} / h={best['hilbert_dim']} / {best['operator']}")
    logger.info(f"  Best median d = {best['median_d']:.3f}")

    # Feature set comparison (marginal)
    logger.info("\n  Feature set comparison (marginal median d):")
    for feat in df['features'].unique():
        sub = df[df['features'] == feat]
        logger.info(f"    {feat:15s}  median d = {sub['median_d'].median():.3f}  "
                     f"(n={len(sub)} configs)")

    # Universe comparison
    logger.info("\n  Symbol universe comparison (marginal median d):")
    for uni in df['universe'].unique():
        sub = df[df['universe'] == uni]
        logger.info(f"    {uni:15s}  median d = {sub['median_d'].median():.3f}  "
                     f"(n={len(sub)} configs)")

    # Hilbert dimension comparison
    logger.info("\n  Hilbert dimension comparison (marginal median d):")
    for h in sorted(df['hilbert_dim'].unique()):
        sub = df[df['hilbert_dim'] == h]
        logger.info(f"    h={h:2d}            median d = {sub['median_d'].median():.3f}  "
                     f"(n={len(sub)} configs)")

    # Operator comparison
    logger.info("\n  Operator comparison (marginal median d):")
    for op in df['operator'].unique():
        sub = df[df['operator'] == op]
        logger.info(f"    {op:15s}  median d = {sub['median_d'].median():.3f}  "
                     f"(n={len(sub)} configs)")

    # --- Save ---
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'feature_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'feature_ablation_{ts}.json'

    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick': quick,
            'crisis_keys': crisis_keys,
            'symbol_universes': {k: v for k, v in symbol_universes.items()},
            'feature_sets': list(feature_sets.keys()),
            'hilbert_dims': hilbert_dims,
            'operator_methods': operator_methods,
        },
        'results': results,
        'summary': {
            'best_config': {
                'detector': best['detector'],
                'features': best['features'],
                'universe': best['universe'],
                'hilbert_dim': int(best['hilbert_dim']),
                'operator': best['operator'],
                'median_d': float(best['median_d']),
            },
            'feature_set_marginal': {
                feat: float(df[df['features'] == feat]['median_d'].median())
                for feat in df['features'].unique()
            },
            'universe_marginal': {
                uni: float(df[df['universe'] == uni]['median_d'].median())
                for uni in df['universe'].unique()
            },
            'hilbert_dim_marginal': {
                int(h): float(df[df['hilbert_dim'] == h]['median_d'].median())
                for h in sorted(df['hilbert_dim'].unique())
            },
            'operator_marginal': {
                op: float(df[df['operator'] == op]['median_d'].median())
                for op in df['operator'].unique()
            },
        },
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert_numpy)

    logger.info(f"\n  Saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Feature engineering ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick run (4 crises)')
    args = parser.parse_args()

    run_ablation(quick=args.quick)


if __name__ == '__main__':
    main()
