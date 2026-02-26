"""
Verification script for QCML method improvements.

Runs V1-V4 from the improvement plan on real Polygon data.
"""

import logging
import sys
import warnings
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
    SpectralGapDetector,
    MetricConditionDetector,
    GeometricEnsembleDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

EXTENSION_DAYS = 10
CRISIS_KEY = '2020_covid'

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        logger.info(f"  PASS: {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        logger.error(f"  FAIL: {name} — {detail}")


def get_crisis_normal_scores(scores, dates, crisis_key):
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def main():
    global passed, failed

    logger.info("=" * 70)
    logger.info("QCML Improvement Verification (V1-V4)")
    logger.info("=" * 70)

    # ---- Fetch data ----
    logger.info("\nFetching data from Polygon...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"Feature matrix: {X_enriched.shape}, dates: {dates_enriched[0]} to {dates_enriched[-1]}")

    # Causal cutoff for 2020 COVID
    ci = ALL_CRISES[CRISIS_KEY]
    crisis_start = pd.Timestamp(ci['start'])
    cutoff_date = crisis_start - pd.Timedelta(days=EXTENSION_DAYS)
    fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))
    logger.info(f"Causal fit end: index {fit_end_idx} ({dates_enriched[fit_end_idx - 1].date()})")

    # ================================================================
    # V1: Backward Compatibility
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("V1: Backward Compatibility (default params)")
    logger.info("=" * 50)

    v1_configs = {
        'Berry Phase Rate': {
            'class': BerryPhaseRateDetector,
            'params': dict(
                hilbert_dim=6, n_pca_components=8, rolling_window=15,
                operator_method='random', seed=42,
                causal_fit_length=fit_end_idx,
                normalization='sphere', berry_aggregation='f01', adaptive_epsilon=False,
            ),
            'expected_d_range': (0.3, 2.5),
        },
        'QFI Determinant': {
            'class': QFIDeterminantDetector,
            'params': dict(
                hilbert_dim=8, n_pca_components=15, rolling_window=20,
                operator_method='pca_inspired', seed=42,
                causal_fit_length=fit_end_idx,
                normalization='sphere', qfi_mode='logdet', adaptive_epsilon=False,
            ),
            # d=0.93 was MEDIAN across 12 crises; single-crisis can be lower
            'expected_d_range': (0.05, 2.5),
        },
        'Multi-Lag Fidelity': {
            'class': MultiLagFidelityDetector,
            'params': dict(
                hilbert_dim=4, n_pca_components=8, rolling_window=20,
                operator_method='pca_inspired', seed=42,
                causal_fit_length=fit_end_idx,
                normalization='sphere', adaptive_epsilon=False,
            ),
            'expected_d_range': (0.2, 2.5),
        },
    }

    v1_d_values = {}
    for name, cfg in v1_configs.items():
        det = cfg['class'](**cfg['params'])
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        crisis_s, normal_s = get_crisis_normal_scores(scores, dates_enriched, CRISIS_KEY)
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=2000)
        v1_d_values[name] = d

        lo, hi = cfg['expected_d_range']
        check(
            f"{name} d={d:.3f} in [{lo}, {hi}]",
            lo <= d <= hi,
            f"d={d:.3f} outside expected range",
        )
        logger.info(f"    {name}: d = {d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ================================================================
    # V2: Normalization Modes Produce Different d-Values
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("V2: Normalization Modes (QFI Determinant)")
    logger.info("=" * 50)

    norm_modes = ['sphere', 'none', 'soft', 'clip']
    norm_d_values = {}
    for nm in norm_modes:
        det = QFIDeterminantDetector(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            causal_fit_length=fit_end_idx,
            normalization=nm,
            adaptive_epsilon=(nm != 'sphere'),
        )
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        crisis_s, normal_s = get_crisis_normal_scores(scores, dates_enriched, CRISIS_KEY)
        d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=2000)
        norm_d_values[nm] = d
        logger.info(f"    normalization={nm:8s}  d = {d:.3f}")

    # Check that d-values are not all identical
    d_vals = list(norm_d_values.values())
    d_range = max(d_vals) - min(d_vals)
    check(
        f"Normalization modes produce different d-values (range={d_range:.3f})",
        d_range > 0.01,
        f"All 4 modes gave nearly identical d-values (range={d_range:.4f})",
    )
    for nm, d in norm_d_values.items():
        check(
            f"normalization='{nm}' produces finite d={d:.3f}",
            np.isfinite(d),
            f"Got NaN or inf",
        )

    # ================================================================
    # V3: Aggregation/Mode Features Work
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("V3: Aggregation and Mode Features")
    logger.info("=" * 50)

    # Berry aggregation modes
    logger.info("  Berry Phase Rate aggregation modes:")
    berry_d_values = {}
    for agg in ['f01', 'frobenius', 'max']:
        det = BerryPhaseRateDetector(
            hilbert_dim=6, n_pca_components=8, rolling_window=15,
            operator_method='random', seed=42,
            causal_fit_length=fit_end_idx,
            normalization='none', berry_aggregation=agg,
            adaptive_epsilon=True,
        )
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        crisis_s, normal_s = get_crisis_normal_scores(scores, dates_enriched, CRISIS_KEY)
        d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=2000)
        berry_d_values[agg] = d
        has_nans = np.sum(np.isnan(scores[60:])) / len(scores[60:])
        logger.info(f"    berry_aggregation={agg:12s}  d = {d:.3f}  (NaN ratio: {has_nans:.3f})")
        check(
            f"Berry agg='{agg}' d={d:.3f} is finite",
            np.isfinite(d),
            f"Got non-finite d",
        )

    # QFI modes
    logger.info("  QFI Determinant modes:")
    qfi_d_values = {}
    for mode in ['logdet', 'trace', 'max_eig', 'condition', 'entropy']:
        det = QFIDeterminantDetector(
            hilbert_dim=8, n_pca_components=15, rolling_window=20,
            operator_method='pca_inspired', seed=42,
            causal_fit_length=fit_end_idx,
            normalization='none', qfi_mode=mode,
            adaptive_epsilon=True,
        )
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        crisis_s, normal_s = get_crisis_normal_scores(scores, dates_enriched, CRISIS_KEY)
        d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=2000)
        qfi_d_values[mode] = d
        has_nans = np.sum(np.isnan(scores[60:])) / len(scores[60:])
        logger.info(f"    qfi_mode={mode:12s}  d = {d:.3f}  (NaN ratio: {has_nans:.3f})")
        check(
            f"QFI mode='{mode}' d={d:.3f} is finite",
            np.isfinite(d),
            f"Got non-finite d",
        )

    # ================================================================
    # V4: New Detectors Return Valid Scores
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("V4: New Detectors (SpectralGap, MetricCondition, GeometricEnsemble)")
    logger.info("=" * 50)

    new_configs = {
        'Spectral Gap': {
            'class': SpectralGapDetector,
            'params': dict(
                hilbert_dim=8, n_pca_components=15, rolling_window=20,
                operator_method='random', seed=42,
                causal_fit_length=fit_end_idx,
                normalization='none', adaptive_epsilon=True,
            ),
        },
        'Metric Condition': {
            'class': MetricConditionDetector,
            'params': dict(
                hilbert_dim=8, n_pca_components=15, rolling_window=20,
                operator_method='random', seed=42,
                causal_fit_length=fit_end_idx,
                normalization='none', adaptive_epsilon=True,
            ),
        },
        'Geometric Ensemble': {
            'class': GeometricEnsembleDetector,
            'params': dict(
                hilbert_dim=8, n_pca_components=15, rolling_window=20,
                operator_method='random', seed=42,
                causal_fit_length=fit_end_idx,
                normalization='none', adaptive_epsilon=True,
            ),
        },
    }

    for name, cfg in new_configs.items():
        det = cfg['class'](**cfg['params'])
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        crisis_s, normal_s = get_crisis_normal_scores(scores, dates_enriched, CRISIS_KEY)
        d, ci_lo, ci_hi = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=2000)
        has_nans = np.sum(np.isnan(scores[60:])) / len(scores[60:])
        logger.info(f"    {name:25s}  d = {d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  (NaN ratio: {has_nans:.3f})")

        check(f"{name} d={d:.3f} is finite", np.isfinite(d), "Got non-finite d")
        check(f"{name} d={d:.3f} > 0", d > 0, f"d={d:.3f} is not positive")

    # ================================================================
    # Summary
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"VERIFICATION SUMMARY: {passed} passed, {failed} failed")
    logger.info("=" * 70)

    if errors:
        logger.error("FAILURES:")
        for e in errors:
            logger.error(f"  - {e}")

    # Best normalization/aggregation discovery
    logger.info("\n--- Best Configurations Found ---")
    logger.info(f"  QFI best normalization: {max(norm_d_values, key=norm_d_values.get)} "
                f"(d={max(norm_d_values.values()):.3f})")
    logger.info(f"  Berry best aggregation: {max(berry_d_values, key=berry_d_values.get)} "
                f"(d={max(berry_d_values.values()):.3f})")
    logger.info(f"  QFI best mode: {max(qfi_d_values, key=qfi_d_values.get)} "
                f"(d={max(qfi_d_values.values()):.3f})")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
