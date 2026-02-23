"""
RF Augmentation Experiment: Do geometric features carry novel signal?

Tests whether adding geometric observables to Random Forest improves regime
detection beyond what standard enriched features provide.

Four evaluation pipelines:
1. LOCO comparison — baseline RF vs augmented RF, leave-one-crisis-out
2. Temporal OOS — pre-2020 train, post-2020 test
3. Feature ablation — drop-one and add-one analysis of 5 geometric features
4. Statistical tests — Wilcoxon, bootstrap CI, permutation feature importance

Usage:
    python experiments/rf_augmentation_experiment.py
    python experiments/rf_augmentation_experiment.py --quick
    python experiments/rf_augmentation_experiment.py --n-bootstrap 1000

Outputs:
    experiments/outputs/regime_detection/rf_augmentation_YYYYMMDD_HHMMSS.json
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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import BaseRegimeDetector

from experiments.data_loader import fetch_polygon_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import (
    compute_cohens_d_with_ci,
    friedman_test,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# GeometricFeatureExtractor
# =============================================================================

class GeometricFeatureExtractor:
    """Extract geometric time series from enriched features for RF augmentation.

    Fits scaler/PCA/QCML geometry (unsupervised, no labels) and computes 5
    geometric features at each timestep:
        berry_rate      — smoothed |diff(Berry curvature)|
        qfi_logdet      — smoothed log|det(quantum metric)|
        multilag_infid  — smoothed weighted multi-lag (1-F)
        inv_spectral_gap — 1 / (spectral gap + eps)
        log_condition   — log(condition number of metric tensor)

    Args:
        hilbert_dim: Hilbert space dimension for QCML geometry.
        n_pca_components: Number of PCA components.
        operator_method: Operator fitting method ('pca_inspired' or 'random').
        rolling_window: Smoothing window for raw geometric series.
        lags: Fidelity lags for multi-lag infidelity.
        lag_weights: Weights for each fidelity lag.
        seed: Random seed.
    """

    FEATURE_NAMES = [
        'berry_rate',
        'qfi_logdet',
        'multilag_infid',
        'inv_spectral_gap',
        'log_condition',
    ]

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=15,
        operator_method='pca_inspired',
        rolling_window=20,
        lags=None,
        lag_weights=None,
        seed=42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.lags = lags or [1, 3, 5, 10]
        self.lag_weights = lag_weights or [0.4, 0.3, 0.2, 0.1]
        self.seed = seed
        self._scaler = None
        self._pca = None
        self._geometry = None

    def fit(self, X_enriched):
        """Fit scaler, PCA, and geometry on enriched features (unsupervised).

        Args:
            X_enriched: Enriched feature matrix (T, d_enriched).

        Returns:
            self
        """
        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X_enriched.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X_enriched)

        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X_enriched)
        self._pca.fit(X_scaled)

        X_pca = self._pca.transform(X_scaled)
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        X_pca = X_pca / (norms + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)

        return self

    def transform(self, X_enriched):
        """Compute 5 geometric feature time series.

        Args:
            X_enriched: Enriched feature matrix (T, d_enriched).

        Returns:
            geo_features: np.ndarray (T, 5) — columns in FEATURE_NAMES order.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before transform().")

        X_scaled = self._scaler.transform(X_enriched)
        X_pca = self._pca.transform(X_scaled)
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        Xt = X_pca / (norms + 1e-8)

        T = len(Xt)
        eig_tol = 1e-10

        berry_vals = np.empty(T)
        log_pseudodet = np.empty(T)
        inv_gap = np.empty(T)
        log_cond = np.empty(T)
        states = []

        for t in range(T):
            xt = Xt[t]

            # Berry curvature (2D, indices 0,1)
            berry_vals[t] = self._geometry.berry_curvature_2d(xt, indices=(0, 1))

            # Metric tensor for logdet and condition number
            g_ij = self._geometry.quantum_metric(xt)
            eigenvalues = np.linalg.eigvalsh(g_ij)
            nonzero_eigs = eigenvalues[eigenvalues > eig_tol]

            if len(nonzero_eigs) > 0:
                log_pseudodet[t] = np.sum(np.log(nonzero_eigs))
                log_cond[t] = np.log(nonzero_eigs[-1] / nonzero_eigs[0] + 1e-12)
            else:
                log_pseudodet[t] = np.log(eig_tol) * len(eigenvalues)
                log_cond[t] = 0.0

            # Spectral gap
            gap = self._geometry.spectral_gap(xt)
            inv_gap[t] = 1.0 / (gap + 1e-8)

            # Coherent state for fidelity
            psi = self._geometry.quasi_coherent_state(xt)
            states.append(psi)

        # Berry rate = |diff(berry)|
        berry_rate = np.abs(np.diff(berry_vals))
        berry_rate = np.concatenate([[0.0], berry_rate])

        # Multi-lag infidelity
        max_lag = max(self.lags)
        infidelity = np.full(T, 0.0)
        for t in range(max_lag, T):
            weighted = 0.0
            for lag, w in zip(self.lags, self.lag_weights):
                if t >= lag:
                    overlap = np.abs(np.vdot(states[t], states[t - lag]))
                    weighted += w * (1.0 - overlap ** 2)
            infidelity[t] = weighted

        # Smooth all 5 features
        rw = self.rolling_window
        berry_rate_s = pd.Series(berry_rate).rolling(rw, min_periods=1).mean().values
        logdet_s = pd.Series(log_pseudodet).rolling(rw, min_periods=1).mean().values
        infid_s = pd.Series(infidelity).rolling(rw, min_periods=1).mean().values
        inv_gap_s = pd.Series(inv_gap).rolling(rw, min_periods=1).mean().values
        log_cond_s = pd.Series(log_cond).rolling(rw, min_periods=1).mean().values

        return np.column_stack([
            berry_rate_s,
            logdet_s,
            infid_s,
            inv_gap_s,
            log_cond_s,
        ])

    def fit_transform(self, X_enriched):
        """Fit and transform in one call."""
        self.fit(X_enriched)
        return self.transform(X_enriched)


# =============================================================================
# RF Helpers
# =============================================================================

def train_rf(X_train, y_train, n_estimators=200, max_depth=6, seed=42):
    """Train a Random Forest classifier.

    Args:
        X_train: Feature matrix (N, d).
        y_train: Binary labels (N,).
        n_estimators: Number of trees.
        max_depth: Max tree depth.
        seed: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def build_crisis_labels(dates, crises_dict, window_size=10):
    """Build binary crisis labels for a date index.

    Args:
        dates: DatetimeIndex.
        crises_dict: Dict of crisis definitions with 'start'/'end' keys.
        window_size: Extension in calendar days (±).

    Returns:
        y: Binary array (T,) where 1 = crisis period.
    """
    y = np.zeros(len(dates))
    for ci in crises_dict.values():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=window_size)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=window_size)
        mask = (dates >= cs) & (dates <= ce)
        y[mask] = 1.0
    return y


def score_rf_on_crisis(rf, X_test, dates, crisis_info, window_size=10):
    """Score RF predictions for a single crisis via Cohen's d.

    Args:
        rf: Fitted RandomForestClassifier.
        X_test: Full feature matrix (T, d).
        dates: DatetimeIndex aligned with X_test.
        crisis_info: Dict with 'start' and 'end'.
        window_size: Crisis window extension in calendar days.

    Returns:
        (d, ci_lo, ci_hi): Cohen's d with 95% bootstrap CI.
    """
    proba = rf.predict_proba(X_test)[:, 1]

    cs = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=window_size)
    ce = pd.Timestamp(crisis_info['end']) + pd.Timedelta(days=window_size)
    crisis_mask = (dates >= cs) & (dates <= ce)

    crisis_scores = proba[crisis_mask]
    normal_scores = proba[~crisis_mask]

    return compute_cohens_d_with_ci(crisis_scores, normal_scores, n_bootstrap=5000)


# =============================================================================
# Pipeline 1: LOCO Comparison
# =============================================================================

def run_loco_comparison(
    X_enriched, dates_enriched, X_raw, geo_extractor,
    crises, window_size=10,
):
    """Leave-one-crisis-out comparison: baseline RF vs augmented RF.

    For each held-out crisis:
    - Train baseline RF on enriched features (other crises as labels)
    - Train augmented RF on enriched + geometric features
    - Compare per-crisis Cohen's d

    Args:
        X_enriched: Enriched feature matrix (T, d).
        dates_enriched: DatetimeIndex for enriched features.
        X_raw: Raw feature matrix for legacy RF lookback.
        geo_extractor: Fitted GeometricFeatureExtractor.
        crises: Dict of crises to evaluate.
        window_size: Crisis window extension.

    Returns:
        Dict with per-crisis results for baseline and augmented RF.
    """
    logger.info("\n[LOCO] Running leave-one-crisis-out comparison...")

    # Compute geometric features once (unsupervised, no labels used)
    geo_features = geo_extractor.transform(X_enriched)
    X_augmented = np.hstack([X_enriched, geo_features])

    results = {'baseline': {}, 'augmented': {}}

    for held_out_key, held_out_info in crises.items():
        logger.info(f"  Held out: {held_out_key}")

        # Build labels: 1 during all crises EXCEPT held_out
        train_crises = {k: v for k, v in ALL_CRISES.items() if k != held_out_key}
        y = build_crisis_labels(dates_enriched, train_crises, window_size)

        # Train baseline RF on enriched features
        rf_base = train_rf(X_enriched, y)
        d_base, ci_lo_base, ci_hi_base = score_rf_on_crisis(
            rf_base, X_enriched, dates_enriched, held_out_info, window_size
        )

        # Train augmented RF on enriched + geometric features
        rf_aug = train_rf(X_augmented, y)
        d_aug, ci_lo_aug, ci_hi_aug = score_rf_on_crisis(
            rf_aug, X_augmented, dates_enriched, held_out_info, window_size
        )

        results['baseline'][held_out_key] = {
            'd': _safe_float(d_base),
            'ci_lo': _safe_float(ci_lo_base),
            'ci_hi': _safe_float(ci_hi_base),
        }
        results['augmented'][held_out_key] = {
            'd': _safe_float(d_aug),
            'ci_lo': _safe_float(ci_lo_aug),
            'ci_hi': _safe_float(ci_hi_aug),
        }

        logger.info(
            f"    Baseline d={d_base:.3f}  Augmented d={d_aug:.3f}  "
            f"Δ={d_aug - d_base:+.3f}"
        )

    return results


# =============================================================================
# Pipeline 2: Temporal OOS
# =============================================================================

def run_temporal_oos(X_enriched, dates_enriched, window_size=10):
    """Temporal out-of-sample: train on pre-2020, test on post-2020.

    GeometricFeatureExtractor.fit() strictly on pre-2020 data.

    Args:
        X_enriched: Enriched feature matrix (T, d).
        dates_enriched: DatetimeIndex.
        window_size: Crisis window extension.

    Returns:
        Dict with per-crisis results for baseline and augmented RF.
    """
    logger.info("\n[Temporal OOS] Pre-2020 train, post-2020 test...")

    split_date = pd.Timestamp('2020-01-01')
    train_mask = dates_enriched < split_date
    test_mask = dates_enriched >= split_date

    X_train_enr = X_enriched[train_mask]
    X_test_enr = X_enriched[test_mask]
    dates_test = dates_enriched[test_mask]

    # Fit geometric extractor strictly on train data
    geo_oos = GeometricFeatureExtractor(
        hilbert_dim=8, n_pca_components=15,
        operator_method='pca_inspired', seed=42,
    )
    geo_oos.fit(X_train_enr)

    # Transform both train and test
    geo_train = geo_oos.transform(X_train_enr)
    geo_test = geo_oos.transform(X_test_enr)

    X_train_aug = np.hstack([X_train_enr, geo_train])
    X_test_aug = np.hstack([X_test_enr, geo_test])

    # Labels: pre-2020 crises for training
    pre2020_crises = {
        k: v for k, v in ALL_CRISES.items()
        if pd.Timestamp(v['end']) < split_date
    }
    y_train = build_crisis_labels(
        dates_enriched[train_mask], pre2020_crises, window_size
    )

    # Train both RFs
    rf_base = train_rf(X_train_enr, y_train)
    rf_aug = train_rf(X_train_aug, y_train)

    # Evaluate on post-2020 crises
    post2020_crises = {
        k: v for k, v in ALL_CRISES.items()
        if pd.Timestamp(v['start']) >= split_date
    }

    results = {'baseline': {}, 'augmented': {}}
    for ck, ci in post2020_crises.items():
        d_base, ci_lo_b, ci_hi_b = score_rf_on_crisis(
            rf_base, X_test_enr, dates_test, ci, window_size
        )
        d_aug, ci_lo_a, ci_hi_a = score_rf_on_crisis(
            rf_aug, X_test_aug, dates_test, ci, window_size
        )

        results['baseline'][ck] = {
            'd': _safe_float(d_base),
            'ci_lo': _safe_float(ci_lo_b),
            'ci_hi': _safe_float(ci_hi_b),
        }
        results['augmented'][ck] = {
            'd': _safe_float(d_aug),
            'ci_lo': _safe_float(ci_lo_a),
            'ci_hi': _safe_float(ci_hi_a),
        }

        logger.info(
            f"  {ck}: Baseline d={d_base:.3f}  Augmented d={d_aug:.3f}  "
            f"Δ={d_aug - d_base:+.3f}"
        )

    return results


# =============================================================================
# Pipeline 3: Feature Ablation
# =============================================================================

def run_feature_ablation(
    X_enriched, dates_enriched, geo_extractor,
    crises, window_size=10,
):
    """Feature ablation: drop-one and add-one analysis of 5 geometric features.

    Args:
        X_enriched: Enriched feature matrix (T, d).
        dates_enriched: DatetimeIndex.
        geo_extractor: Fitted GeometricFeatureExtractor.
        crises: Dict of crises.
        window_size: Crisis window extension.

    Returns:
        Dict with drop-one and add-one results.
    """
    logger.info("\n[Ablation] Running feature ablation analysis...")

    geo_features = geo_extractor.transform(X_enriched)
    feature_names = GeometricFeatureExtractor.FEATURE_NAMES

    # Full labels (all crises)
    y_all = build_crisis_labels(dates_enriched, ALL_CRISES, window_size)

    # Baseline: RF with enriched features only
    rf_base = train_rf(X_enriched, y_all)
    base_ds = _compute_mean_d(rf_base, X_enriched, dates_enriched, crises, window_size)

    # Full augmented
    X_full_aug = np.hstack([X_enriched, geo_features])
    rf_full = train_rf(X_full_aug, y_all)
    full_ds = _compute_mean_d(rf_full, X_full_aug, dates_enriched, crises, window_size)

    results = {
        'baseline_mean_d': base_ds,
        'full_augmented_mean_d': full_ds,
        'drop_one': {},
        'add_one': {},
    }

    # Drop-one: remove each geometric feature from full set
    for i, fname in enumerate(feature_names):
        keep_cols = [j for j in range(5) if j != i]
        geo_subset = geo_features[:, keep_cols]
        X_drop = np.hstack([X_enriched, geo_subset])
        rf_drop = train_rf(X_drop, y_all)
        mean_d = _compute_mean_d(rf_drop, X_drop, dates_enriched, crises, window_size)
        drop_impact = full_ds - mean_d  # positive = dropping hurts
        results['drop_one'][fname] = {
            'mean_d': mean_d,
            'impact': drop_impact,
        }
        logger.info(f"  Drop {fname}: mean_d={mean_d:.3f} (impact={drop_impact:+.3f})")

    # Add-one: each geometric feature individually added to enriched
    for i, fname in enumerate(feature_names):
        X_add = np.hstack([X_enriched, geo_features[:, i:i+1]])
        rf_add = train_rf(X_add, y_all)
        mean_d = _compute_mean_d(rf_add, X_add, dates_enriched, crises, window_size)
        add_impact = mean_d - base_ds  # positive = adding helps
        results['add_one'][fname] = {
            'mean_d': mean_d,
            'impact': add_impact,
        }
        logger.info(f"  Add {fname}: mean_d={mean_d:.3f} (impact={add_impact:+.3f})")

    return results


def _compute_mean_d(rf, X, dates, crises, window_size):
    """Compute mean Cohen's d across all crises for a fitted RF."""
    d_vals = []
    for ck, ci in crises.items():
        d, _, _ = score_rf_on_crisis(rf, X, dates, ci, window_size)
        if not np.isnan(d):
            d_vals.append(d)
    return float(np.mean(d_vals)) if d_vals else 0.0


# =============================================================================
# Pipeline 4: Statistical Tests
# =============================================================================

def run_statistical_tests(loco_results, n_bootstrap=10000, seed=42):
    """Run paired statistical tests on LOCO results.

    Tests:
    - Paired Wilcoxon signed-rank test on per-crisis d-values
    - Bootstrap CI on mean improvement (augmented - baseline)
    - Win/loss/tie count

    Args:
        loco_results: Dict from run_loco_comparison with 'baseline' and 'augmented'.
        n_bootstrap: Bootstrap resamples for CI.
        seed: Random seed.

    Returns:
        Dict with test results.
    """
    logger.info("\n[Stats] Running statistical tests on LOCO results...")

    baseline_ds = []
    augmented_ds = []
    crisis_keys = []

    for ck in loco_results['baseline']:
        d_base = loco_results['baseline'][ck].get('d')
        d_aug = loco_results['augmented'][ck].get('d')
        if d_base is not None and d_aug is not None:
            baseline_ds.append(d_base)
            augmented_ds.append(d_aug)
            crisis_keys.append(ck)

    baseline_ds = np.array(baseline_ds)
    augmented_ds = np.array(augmented_ds)
    diffs = augmented_ds - baseline_ds

    n = len(diffs)
    results = {
        'n_crises': n,
        'mean_baseline_d': float(np.mean(baseline_ds)),
        'mean_augmented_d': float(np.mean(augmented_ds)),
        'mean_improvement': float(np.mean(diffs)),
        'median_improvement': float(np.median(diffs)),
    }

    # Win/loss/tie
    wins = int(np.sum(diffs > 0.01))
    losses = int(np.sum(diffs < -0.01))
    ties = n - wins - losses
    results['wins'] = wins
    results['losses'] = losses
    results['ties'] = ties

    # Paired Wilcoxon
    if n >= 6:
        stat, p_val = stats.wilcoxon(augmented_ds, baseline_ds, alternative='greater')
        results['wilcoxon_stat'] = float(stat)
        results['wilcoxon_p'] = float(p_val)
        logger.info(f"  Wilcoxon: stat={stat:.2f}, p={p_val:.4f}")
    else:
        results['wilcoxon_stat'] = None
        results['wilcoxon_p'] = None
        logger.info(f"  Wilcoxon: insufficient data (n={n})")

    # Bootstrap CI on mean improvement
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(diffs[idx])

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    results['bootstrap_ci_lo'] = float(ci_lo)
    results['bootstrap_ci_hi'] = float(ci_hi)
    results['bootstrap_ci_excludes_zero'] = bool(ci_lo > 0)
    logger.info(
        f"  Bootstrap CI on mean improvement: [{ci_lo:.3f}, {ci_hi:.3f}]"
    )

    # Per-crisis detail
    results['per_crisis'] = {}
    for i, ck in enumerate(crisis_keys):
        results['per_crisis'][ck] = {
            'baseline_d': float(baseline_ds[i]),
            'augmented_d': float(augmented_ds[i]),
            'improvement': float(diffs[i]),
        }

    # Permutation feature importance (from full augmented RF)
    logger.info(
        f"  Results: {wins}W/{losses}L/{ties}T, "
        f"mean Δd={np.mean(diffs):+.3f}"
    )

    return results


# =============================================================================
# Permutation Feature Importance
# =============================================================================

def run_permutation_importance(
    X_enriched, dates_enriched, geo_extractor,
    crises, window_size=10, n_repeats=10, seed=42,
):
    """Permutation importance of each geometric feature in the augmented RF.

    Shuffles each geometric feature column and measures Cohen's d degradation.

    Args:
        X_enriched: Enriched features (T, d).
        dates_enriched: DatetimeIndex.
        geo_extractor: Fitted GeometricFeatureExtractor.
        crises: Crisis dict.
        window_size: Crisis window extension.
        n_repeats: Number of shuffle repeats.
        seed: Random seed.

    Returns:
        Dict mapping feature names to importance scores.
    """
    logger.info("\n[Importance] Running permutation feature importance...")

    geo_features = geo_extractor.transform(X_enriched)
    X_augmented = np.hstack([X_enriched, geo_features])
    feature_names = GeometricFeatureExtractor.FEATURE_NAMES

    y_all = build_crisis_labels(dates_enriched, ALL_CRISES, window_size)
    rf = train_rf(X_augmented, y_all)
    base_d = _compute_mean_d(rf, X_augmented, dates_enriched, crises, window_size)

    rng = np.random.default_rng(seed)
    n_enriched = X_enriched.shape[1]
    results = {}

    for i, fname in enumerate(feature_names):
        degradations = []
        for rep in range(n_repeats):
            X_perm = X_augmented.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, n_enriched + i] = X_perm[perm_idx, n_enriched + i]

            perm_d = _compute_mean_d(rf, X_perm, dates_enriched, crises, window_size)
            degradations.append(base_d - perm_d)

        mean_deg = float(np.mean(degradations))
        std_deg = float(np.std(degradations, ddof=1))
        results[fname] = {
            'mean_degradation': mean_deg,
            'std_degradation': std_deg,
        }
        logger.info(f"  {fname}: mean degradation={mean_deg:.4f} ± {std_deg:.4f}")

    return results


# =============================================================================
# Utilities
# =============================================================================

def _safe_float(val):
    """Convert to float, returning None for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_experiment(quick=False, n_bootstrap=10000, window_size=10):
    """Run the full RF augmentation experiment.

    Args:
        quick: Only use 4 representative crises.
        n_bootstrap: Bootstrap resamples.
        window_size: Crisis window extension.

    Returns:
        Full results dict.
    """
    logger.info("=" * 70)
    logger.info("RF AUGMENTATION EXPERIMENT")
    logger.info("Do geometric features carry novel signal beyond enriched features?")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("\n[1/7] Fetching data from Polygon...")
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(prices_df)
    logger.info(f"  Raw features: {X_raw.shape}, dates: {dates[0]} to {dates[-1]}")

    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"  Enriched features: {X_enriched.shape}")

    # ---- Crisis selection ----
    if quick:
        crisis_keys = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
    else:
        crisis_keys = list(ALL_CRISES.keys())

    crises = {k: ALL_CRISES[k] for k in crisis_keys}
    logger.info(f"  Evaluating {len(crises)} crises")

    # ---- Fit GeometricFeatureExtractor (unsupervised) ----
    logger.info("\n[2/7] Fitting GeometricFeatureExtractor (unsupervised)...")
    geo = GeometricFeatureExtractor(
        hilbert_dim=8, n_pca_components=15,
        operator_method='pca_inspired', seed=42,
    )
    geo.fit(X_enriched)
    logger.info("  Fitted. Computing geometric features...")
    geo_features = geo.transform(X_enriched)
    logger.info(f"  Geometric features: {geo_features.shape}")

    # Sanity check: are geometric features non-trivial?
    for i, fname in enumerate(GeometricFeatureExtractor.FEATURE_NAMES):
        col = geo_features[:, i]
        valid = col[~np.isnan(col)]
        logger.info(
            f"    {fname}: mean={np.mean(valid):.4f}, "
            f"std={np.std(valid):.4f}, range=[{np.min(valid):.4f}, {np.max(valid):.4f}]"
        )

    # ---- Pipeline 1: LOCO ----
    logger.info("\n[3/7] Pipeline 1: LOCO Comparison")
    loco_results = run_loco_comparison(
        X_enriched, dates_enriched, X_raw, geo,
        crises, window_size,
    )

    # ---- Pipeline 2: Temporal OOS ----
    logger.info("\n[4/7] Pipeline 2: Temporal OOS")
    temporal_results = run_temporal_oos(
        X_enriched, dates_enriched, window_size,
    )

    # ---- Pipeline 3: Feature Ablation ----
    logger.info("\n[5/7] Pipeline 3: Feature Ablation")
    ablation_results = run_feature_ablation(
        X_enriched, dates_enriched, geo,
        crises, window_size,
    )

    # ---- Pipeline 4: Statistical Tests ----
    logger.info("\n[6/7] Pipeline 4: Statistical Tests")
    stat_results = run_statistical_tests(
        loco_results, n_bootstrap=n_bootstrap,
    )

    # ---- Pipeline 5: Permutation Importance ----
    logger.info("\n[7/7] Pipeline 5: Permutation Feature Importance")
    importance_results = run_permutation_importance(
        X_enriched, dates_enriched, geo,
        crises, window_size, n_repeats=10,
    )

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n  LOCO: Baseline mean d = {stat_results['mean_baseline_d']:.3f}")
    logger.info(f"  LOCO: Augmented mean d = {stat_results['mean_augmented_d']:.3f}")
    logger.info(f"  LOCO: Mean improvement  = {stat_results['mean_improvement']:+.3f}")
    logger.info(f"  LOCO: {stat_results['wins']}W/{stat_results['losses']}L/{stat_results['ties']}T")

    if stat_results.get('wilcoxon_p') is not None:
        logger.info(f"  Wilcoxon p = {stat_results['wilcoxon_p']:.4f}")
    logger.info(
        f"  Bootstrap CI: [{stat_results['bootstrap_ci_lo']:.3f}, "
        f"{stat_results['bootstrap_ci_hi']:.3f}]"
    )

    logger.info(f"\n  Ablation: Baseline mean d = {ablation_results['baseline_mean_d']:.3f}")
    logger.info(f"  Ablation: Full augmented   = {ablation_results['full_augmented_mean_d']:.3f}")

    logger.info("\n  Feature importance (permutation degradation):")
    sorted_imp = sorted(
        importance_results.items(),
        key=lambda x: x[1]['mean_degradation'],
        reverse=True,
    )
    for fname, imp in sorted_imp:
        logger.info(f"    {fname:20s}: {imp['mean_degradation']:+.4f}")

    # Temporal OOS summary
    if temporal_results['baseline'] and temporal_results['augmented']:
        base_oos = [
            v['d'] for v in temporal_results['baseline'].values()
            if v['d'] is not None
        ]
        aug_oos = [
            v['d'] for v in temporal_results['augmented'].values()
            if v['d'] is not None
        ]
        if base_oos and aug_oos:
            logger.info(
                f"\n  Temporal OOS: Baseline mean d = {np.mean(base_oos):.3f}"
            )
            logger.info(
                f"  Temporal OOS: Augmented mean d = {np.mean(aug_oos):.3f}"
            )

    # ---- Save ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick': quick,
            'n_bootstrap': n_bootstrap,
            'window_size': window_size,
            'n_crises': len(crises),
            'geo_config': {
                'hilbert_dim': 8,
                'n_pca_components': 15,
                'operator_method': 'pca_inspired',
                'rolling_window': 20,
            },
        },
        'loco': loco_results,
        'temporal_oos': temporal_results,
        'ablation': ablation_results,
        'statistical_tests': stat_results,
        'permutation_importance': importance_results,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'rf_augmentation_{ts}.json'

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to {out_path}")
    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RF Augmentation Experiment: geometric features as RF input'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Only use 4 representative crises',
    )
    parser.add_argument(
        '--n-bootstrap', type=int, default=10000,
        help='Bootstrap resamples (default: 10000)',
    )
    parser.add_argument(
        '--window-size', type=int, default=10,
        help='Crisis window extension ± days (default: 10)',
    )
    args = parser.parse_args()

    run_experiment(
        quick=args.quick,
        n_bootstrap=args.n_bootstrap,
        window_size=args.window_size,
    )


if __name__ == '__main__':
    main()
