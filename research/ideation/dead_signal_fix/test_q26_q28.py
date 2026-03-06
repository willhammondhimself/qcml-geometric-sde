"""
Dead Signal Resurrection Smoke Test: Q26, Q27, Q28

Tests whether simple parameter fixes can resurrect three near-zero signals:

Q26: QGT Phase Rigidity (d=0.013) — Kramers degeneracy from pca_inspired operators
  Fix: Switch to operator_method='random', try hilbert_dim=8 and hilbert_dim=16.

Q27: Berry Velocity Coupling (d=0.012) — coupling ||iota_v F||_g may need longer timescale
  Fix: Vary velocity_lag (1, 5, 10, 20) and rolling_window (30, 60, 120).
  Note: current detector uses lag=1 (adjacent timesteps). Testing longer lags by
  passing a lagged version of X to berry_velocity_coupling().

Q28: Curvature Rate (d=0.013) — |dR/dt| is noisy raw rate
  Fix (a): Smoothed rate via longer rolling_window (40, 80)
  Fix (b): Accumulated curvature (cumulative sum of |dR/dt|, then re-differentiate at slower rate)
  Fix (c): Sign changes in Ricci scalar (zero-crossing count)
  Fix (d): Switch pca_inspired -> random (same Kramers fix as Q26)

Protocol:
  - 4 crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
  - SPY + DIA data from yfinance
  - n_bootstrap=500 (quick-test quality, not paper-quality)
  - Keep if median d > 0.3

Usage:
    cd /Users/willhammond/Will\\ x\\ Average\\ Research/qcml-geometric-sde
    python research/ideation/dead_signal_fix/test_q26_q28.py
"""

import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import (
    QGTPhaseRigidityDetector,
    BerryVelocityCouplingDetector,
    CurvatureRateDetector,
    BaseRegimeDetector,
)
from qcml_geometry.observables import (
    _standard_qcml_fit,
    _standard_init,
    _expanding_zscore,
    _transform_array,
    ExpandingWindowMixin,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CRISIS_KEYS = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 500
SEED = 42
KEEP_THRESHOLD = 0.3  # median d > 0.3 -> keep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crisis_masks(dates_enriched, crisis_info):
    """Return (crisis_mask, normal_mask) boolean arrays."""
    cs = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    ce = pd.Timestamp(crisis_info['end']) + pd.Timedelta(days=10)
    crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
    return crisis_mask, ~crisis_mask


def _causal_fit_end(dates_enriched, crisis_info):
    """Index of last date >= 10 days before crisis start."""
    cutoff = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    return max(100, int(np.searchsorted(dates_enriched, cutoff)))


def _eval_detector(det, X_enriched, dates_enriched, crises, label):
    """Fit detector and compute Cohen's d per crisis.

    Returns:
        per_crisis: dict {crisis_key: d}
        median_d: float
    """
    per_crisis = {}
    for ck, ci in crises.items():
        fit_end = _causal_fit_end(dates_enriched, ci)
        det.causal_fit_length = fit_end
        crisis_mask, normal_mask = _crisis_masks(dates_enriched, ci)

        try:
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask],
                n_bootstrap=N_BOOTSTRAP, seed=SEED,
            )
            per_crisis[ck] = float(d) if np.isfinite(d) else np.nan
        except Exception as e:
            per_crisis[ck] = np.nan
            print(f"    ERROR for {label} on {ck}: {e}")

    vals = [v for v in per_crisis.values() if np.isfinite(v)]
    median_d = float(np.median(vals)) if vals else np.nan
    return per_crisis, median_d


def _print_result(label, per_crisis, median_d, crises):
    """Print one row of results."""
    crisis_parts = " | ".join(
        f"{ck}: {per_crisis.get(ck, np.nan):.3f}" if np.isfinite(per_crisis.get(ck, np.nan))
        else f"{ck}: N/A"
        for ck in CRISIS_KEYS
    )
    verdict = "KEEP" if median_d >= KEEP_THRESHOLD else "reject"
    print(f"  [{verdict}] {label:<45}  median_d={median_d:.3f}  |  {crisis_parts}")


# ---------------------------------------------------------------------------
# Q27 custom detector: BerryVelocityCouplingDetector with configurable lag
# ---------------------------------------------------------------------------

class LaggedBerryVelocityCouplingDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Berry Velocity Coupling with configurable velocity lag.

    Instead of v = x(t) - x(t-1), computes v = x(t) - x(t-lag).
    Longer lags smooth out noise and may reveal structural coupling.
    """

    def __init__(self, hilbert_dim=8, n_pca_components=8, operator_method='random',
                 scale_exponent=None, rolling_window=20, min_expanding=60, seed=42,
                 causal_fit_length=None, expanding_refit_interval=None,
                 normalization='soft', adaptive_epsilon=True, custom_operators=None,
                 velocity_lag=1):
        _standard_init(self, hilbert_dim=hilbert_dim, n_pca_components=n_pca_components,
                        operator_method=operator_method, scale_exponent=scale_exponent,
                        rolling_window=rolling_window, min_expanding=min_expanding,
                        seed=seed, causal_fit_length=causal_fit_length,
                        expanding_refit_interval=expanding_refit_interval,
                        normalization=normalization, adaptive_epsilon=adaptive_epsilon,
                        custom_operators=custom_operators, velocity_lag=velocity_lag)

    @property
    def name(self):
        return f"Berry Velocity Coupling (lag={self.velocity_lag})"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(X, self._scaler, self._pca,
                              normalization=self.normalization,
                              train_norms=self._train_norms, train_std=self._train_std)
        T = len(Xt)
        lag = self.velocity_lag
        eps = self._epsilon
        vals = np.full(T, np.nan)
        for t in range(lag, T):
            geo = self._geometry
            xt = Xt[t]
            xt_prev = Xt[t - lag]
            vals[t] = geo.berry_velocity_coupling(xt, xt_prev, epsilon=eps)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, lag)


# ---------------------------------------------------------------------------
# Q28 variants
# ---------------------------------------------------------------------------

class CurvatureRateSmoothedDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Curvature rate with longer rolling_window for smoothing."""

    def __init__(self, hilbert_dim=6, n_pca_components=3, operator_method='random',
                 scale_exponent=None, rolling_window=40, min_expanding=60, seed=42,
                 causal_fit_length=None, expanding_refit_interval=None,
                 normalization='soft', adaptive_epsilon=True, custom_operators=None,
                 subsample=10):
        _standard_init(self, hilbert_dim=hilbert_dim, n_pca_components=n_pca_components,
                        operator_method=operator_method, scale_exponent=scale_exponent,
                        rolling_window=rolling_window, min_expanding=min_expanding,
                        seed=seed, causal_fit_length=causal_fit_length,
                        expanding_refit_interval=expanding_refit_interval,
                        normalization=normalization, adaptive_epsilon=adaptive_epsilon,
                        custom_operators=custom_operators, subsample=subsample)

    @property
    def name(self):
        return f"Curvature Rate Smoothed (rw={self.rolling_window})"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(X, self._scaler, self._pca,
                              normalization=self.normalization,
                              train_norms=self._train_norms, train_std=self._train_std)
        T = len(Xt)
        ricci_vals = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            geo = self._geometry
            xt = Xt[t]
            ricci_vals[t] = geo.ricci_scalar(xt)
        if self.subsample > 1:
            ricci_vals = pd.Series(ricci_vals).interpolate(method='linear').values
        rate = np.full(T, np.nan)
        for t in range(1, T):
            if not np.isnan(ricci_vals[t]) and not np.isnan(ricci_vals[t - 1]):
                rate[t] = abs(ricci_vals[t] - ricci_vals[t - 1])
        return _expanding_zscore(rate, self.rolling_window, self.min_expanding, T, 1)


class AccumulatedCurvatureDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Accumulated |dR/dt| over a window — cumulative curvature change.

    Instead of instantaneous |dR/dt|, sums |dR/dt| over a window_accum period.
    This captures sustained curvature changes that build up during crises.
    """

    def __init__(self, hilbert_dim=6, n_pca_components=3, operator_method='random',
                 scale_exponent=None, rolling_window=20, min_expanding=60, seed=42,
                 causal_fit_length=None, expanding_refit_interval=None,
                 normalization='soft', adaptive_epsilon=True, custom_operators=None,
                 subsample=10, window_accum=20):
        _standard_init(self, hilbert_dim=hilbert_dim, n_pca_components=n_pca_components,
                        operator_method=operator_method, scale_exponent=scale_exponent,
                        rolling_window=rolling_window, min_expanding=min_expanding,
                        seed=seed, causal_fit_length=causal_fit_length,
                        expanding_refit_interval=expanding_refit_interval,
                        normalization=normalization, adaptive_epsilon=adaptive_epsilon,
                        custom_operators=custom_operators, subsample=subsample,
                        window_accum=window_accum)

    @property
    def name(self):
        return f"Accumulated Curvature (accum={self.window_accum})"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(X, self._scaler, self._pca,
                              normalization=self.normalization,
                              train_norms=self._train_norms, train_std=self._train_std)
        T = len(Xt)
        ricci_vals = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            geo = self._geometry
            xt = Xt[t]
            ricci_vals[t] = geo.ricci_scalar(xt)
        if self.subsample > 1:
            ricci_vals = pd.Series(ricci_vals).interpolate(method='linear').values

        # Compute instantaneous rate
        rate = np.full(T, np.nan)
        for t in range(1, T):
            if not np.isnan(ricci_vals[t]) and not np.isnan(ricci_vals[t - 1]):
                rate[t] = abs(ricci_vals[t] - ricci_vals[t - 1])

        # Accumulated curvature: rolling sum of rate
        accum = pd.Series(rate).rolling(window=self.window_accum, min_periods=1).sum().values

        return _expanding_zscore(accum, self.rolling_window, self.min_expanding, T, 1)


class CurvatureSignChangeDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Count sign changes in Ricci scalar over a window.

    Sign flips in R indicate the manifold is transitioning between
    positive and negative curvature regimes — a topological marker.
    """

    def __init__(self, hilbert_dim=6, n_pca_components=3, operator_method='random',
                 scale_exponent=None, rolling_window=20, min_expanding=60, seed=42,
                 causal_fit_length=None, expanding_refit_interval=None,
                 normalization='soft', adaptive_epsilon=True, custom_operators=None,
                 subsample=10, sign_window=30):
        _standard_init(self, hilbert_dim=hilbert_dim, n_pca_components=n_pca_components,
                        operator_method=operator_method, scale_exponent=scale_exponent,
                        rolling_window=rolling_window, min_expanding=min_expanding,
                        seed=seed, causal_fit_length=causal_fit_length,
                        expanding_refit_interval=expanding_refit_interval,
                        normalization=normalization, adaptive_epsilon=adaptive_epsilon,
                        custom_operators=custom_operators, subsample=subsample,
                        sign_window=sign_window)

    @property
    def name(self):
        return f"Curvature Sign Changes (win={self.sign_window})"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(X, self._scaler, self._pca,
                              normalization=self.normalization,
                              train_norms=self._train_norms, train_std=self._train_std)
        T = len(Xt)
        ricci_vals = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            geo = self._geometry
            xt = Xt[t]
            ricci_vals[t] = geo.ricci_scalar(xt)
        if self.subsample > 1:
            ricci_vals = pd.Series(ricci_vals).interpolate(method='linear').values

        # Sign change count over rolling sign_window
        signs = np.sign(ricci_vals)
        sign_changes = np.full(T, np.nan)
        w = self.sign_window
        for t in range(w, T):
            window_signs = signs[t - w:t]
            valid = window_signs[~np.isnan(window_signs)]
            if len(valid) >= 2:
                sign_changes[t] = float(np.sum(np.diff(valid) != 0))
            else:
                sign_changes[t] = 0.0

        return _expanding_zscore(sign_changes, self.rolling_window, self.min_expanding, T, w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Dead Signal Resurrection Smoke Test: Q26, Q27, Q28")
    print(f"Crises: {CRISIS_KEYS}")
    print(f"n_bootstrap={N_BOOTSTRAP} | keep_threshold={KEEP_THRESHOLD}")
    print("=" * 80)

    # ---- Load data ----
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31', use_cache=True)
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    print(f"Data: {X_enriched.shape[0]} days, {X_enriched.shape[1]} features\n")

    crises = {k: ALL_CRISES[k] for k in CRISIS_KEYS if k in ALL_CRISES}

    # =========================================================================
    # Q26: QGT Phase Rigidity — Kramers fix
    # =========================================================================
    print("\n" + "=" * 80)
    print("Q26: QGT Phase Rigidity Fix (pca_inspired -> random, dim sweep)")
    print("=" * 80)

    q26_results = {}

    # Baseline: original pca_inspired, dim=8 (what was tested before)
    # Note: original used dim=8, but may have used pca_inspired. Check the default.
    # From code: QGTPhaseRigidityDetector defaults to operator_method='random' already!
    # But the original bad run may have used pca_inspired. Test both explicitly.

    q26_variants = [
        ("baseline_pca_dim4",
         QGTPhaseRigidityDetector(hilbert_dim=4, n_pca_components=8,
                                   operator_method='pca_inspired', rolling_window=20,
                                   min_expanding=60, seed=SEED, normalization='soft',
                                   adaptive_epsilon=True)),
        ("random_dim8",
         QGTPhaseRigidityDetector(hilbert_dim=8, n_pca_components=8,
                                   operator_method='random', rolling_window=20,
                                   min_expanding=60, seed=SEED, normalization='soft',
                                   adaptive_epsilon=True)),
        ("random_dim8_rw40",
         QGTPhaseRigidityDetector(hilbert_dim=8, n_pca_components=8,
                                   operator_method='random', rolling_window=40,
                                   min_expanding=60, seed=SEED, normalization='soft',
                                   adaptive_epsilon=True)),
        ("random_dim16",
         QGTPhaseRigidityDetector(hilbert_dim=16, n_pca_components=8,
                                   operator_method='random', rolling_window=20,
                                   min_expanding=60, seed=SEED, normalization='soft',
                                   adaptive_epsilon=True)),
        ("random_dim16_rw40",
         QGTPhaseRigidityDetector(hilbert_dim=16, n_pca_components=8,
                                   operator_method='random', rolling_window=40,
                                   min_expanding=60, seed=SEED, normalization='soft',
                                   adaptive_epsilon=True)),
    ]

    for label, det in q26_variants:
        per_crisis, median_d = _eval_detector(det, X_enriched, dates_enriched, crises, label)
        q26_results[label] = {'per_crisis': per_crisis, 'median_d': median_d}
        _print_result(label, per_crisis, median_d, crises)

    # Best Q26
    best_q26_label = max(q26_results, key=lambda k: q26_results[k]['median_d'])
    best_q26 = q26_results[best_q26_label]

    # =========================================================================
    # Q27: Berry Velocity Coupling — lag and window sweep
    # =========================================================================
    print("\n" + "=" * 80)
    print("Q27: Berry Velocity Coupling Fix (velocity_lag and rolling_window sweep)")
    print("=" * 80)

    q27_results = {}

    q27_variants = []
    for lag in [1, 5, 10, 20]:
        for rw in [30, 60, 120]:
            label = f"lag{lag}_rw{rw}"
            det = LaggedBerryVelocityCouplingDetector(
                hilbert_dim=8, n_pca_components=8, operator_method='random',
                rolling_window=rw, min_expanding=60, seed=SEED,
                normalization='soft', adaptive_epsilon=True,
                velocity_lag=lag,
            )
            q27_variants.append((label, det))

    for label, det in q27_variants:
        per_crisis, median_d = _eval_detector(det, X_enriched, dates_enriched, crises, label)
        q27_results[label] = {'per_crisis': per_crisis, 'median_d': median_d}
        _print_result(label, per_crisis, median_d, crises)

    best_q27_label = max(q27_results, key=lambda k: q27_results[k]['median_d'])
    best_q27 = q27_results[best_q27_label]

    # =========================================================================
    # Q28: Curvature Rate — alternative computations
    # =========================================================================
    print("\n" + "=" * 80)
    print("Q28: Curvature Rate Fix (smoothing, accumulation, sign changes, random operators)")
    print("=" * 80)

    q28_results = {}

    q28_variants = [
        # Fix (d): switch pca_inspired -> random, keep other params default
        ("random_ops_default",
         CurvatureRateDetector(hilbert_dim=6, n_pca_components=3,
                               operator_method='random', rolling_window=20,
                               min_expanding=60, seed=SEED, normalization='soft',
                               adaptive_epsilon=True, subsample=10)),
        # Fix (a): smoothed rate, longer rolling_window
        ("smoothed_rw40",
         CurvatureRateSmoothedDetector(hilbert_dim=6, n_pca_components=3,
                                       operator_method='random', rolling_window=40,
                                       min_expanding=60, seed=SEED, normalization='soft',
                                       adaptive_epsilon=True, subsample=10)),
        ("smoothed_rw80",
         CurvatureRateSmoothedDetector(hilbert_dim=6, n_pca_components=3,
                                       operator_method='random', rolling_window=80,
                                       min_expanding=60, seed=SEED, normalization='soft',
                                       adaptive_epsilon=True, subsample=10)),
        # Fix (b): accumulated curvature
        ("accumulated_accum20",
         AccumulatedCurvatureDetector(hilbert_dim=6, n_pca_components=3,
                                      operator_method='random', rolling_window=20,
                                      min_expanding=60, seed=SEED, normalization='soft',
                                      adaptive_epsilon=True, subsample=10,
                                      window_accum=20)),
        ("accumulated_accum60",
         AccumulatedCurvatureDetector(hilbert_dim=6, n_pca_components=3,
                                      operator_method='random', rolling_window=20,
                                      min_expanding=60, seed=SEED, normalization='soft',
                                      adaptive_epsilon=True, subsample=10,
                                      window_accum=60)),
        # Fix (c): sign changes
        ("sign_changes_win30",
         CurvatureSignChangeDetector(hilbert_dim=6, n_pca_components=3,
                                     operator_method='random', rolling_window=20,
                                     min_expanding=60, seed=SEED, normalization='soft',
                                     adaptive_epsilon=True, subsample=10,
                                     sign_window=30)),
        ("sign_changes_win60",
         CurvatureSignChangeDetector(hilbert_dim=6, n_pca_components=3,
                                     operator_method='random', rolling_window=20,
                                     min_expanding=60, seed=SEED, normalization='soft',
                                     adaptive_epsilon=True, subsample=10,
                                     sign_window=60)),
    ]

    for label, det in q28_variants:
        per_crisis, median_d = _eval_detector(det, X_enriched, dates_enriched, crises, label)
        q28_results[label] = {'per_crisis': per_crisis, 'median_d': median_d}
        _print_result(label, per_crisis, median_d, crises)

    best_q28_label = max(q28_results, key=lambda k: q28_results[k]['median_d'])
    best_q28 = q28_results[best_q28_label]

    # =========================================================================
    # Final Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL STRUCTURED SUMMARY")
    print("=" * 80)

    def _format_crisis_dict(per_crisis):
        parts = []
        for ck in CRISIS_KEYS:
            d = per_crisis.get(ck, np.nan)
            short = ck.replace('_gfc', '').replace('_covid', '').replace('_rates', '').replace('_svb', '')
            short_map = {
                '2008': 'gfc',
                '2020': 'covid',
                '2022': 'rates',
                '2023': 'svb',
            }
            key = ck.split('_')[0]
            tag = {'2008': 'gfc', '2020': 'covid', '2022': 'rates', '2023': 'svb'}.get(key, ck)
            parts.append(f"{tag}: {d:.3f}" if np.isfinite(d) else f"{tag}: N/A")
        return "{" + ", ".join(parts) + "}"

    # Q26
    q26_best = q26_results[best_q26_label]
    q26_verdict = "KEEP" if q26_best['median_d'] >= KEEP_THRESHOLD else "REJECT"
    print(f"""
Q26: QGT Phase Rigidity Fix
- Best variant: {best_q26_label}
- Cohen's d per crisis: {_format_crisis_dict(q26_best['per_crisis'])}
- Median d: {q26_best['median_d']:.3f}
- Verdict: {q26_verdict}
- All variants tested:""")
    for k, v in sorted(q26_results.items(), key=lambda x: -x[1]['median_d']):
        print(f"    {k:<45} median_d={v['median_d']:.3f}  {_format_crisis_dict(v['per_crisis'])}")

    # Q27
    q27_best = q27_results[best_q27_label]
    q27_verdict = "KEEP" if q27_best['median_d'] >= KEEP_THRESHOLD else "REJECT"
    print(f"""
Q27: Berry Velocity Coupling Fix
- Best variant: {best_q27_label}
- Cohen's d per crisis: {_format_crisis_dict(q27_best['per_crisis'])}
- Median d: {q27_best['median_d']:.3f}
- Verdict: {q27_verdict}
- Top-5 variants by median d:""")
    top5_q27 = sorted(q27_results.items(), key=lambda x: -x[1]['median_d'])[:5]
    for k, v in top5_q27:
        print(f"    {k:<45} median_d={v['median_d']:.3f}  {_format_crisis_dict(v['per_crisis'])}")

    # Q28
    q28_best = q28_results[best_q28_label]
    q28_verdict = "KEEP" if q28_best['median_d'] >= KEEP_THRESHOLD else "REJECT"
    print(f"""
Q28: Curvature Rate Fix
- Best variant: {best_q28_label}
- Cohen's d per crisis: {_format_crisis_dict(q28_best['per_crisis'])}
- Median d: {q28_best['median_d']:.3f}
- Verdict: {q28_verdict}
- All variants tested:""")
    for k, v in sorted(q28_results.items(), key=lambda x: -x[1]['median_d']):
        print(f"    {k:<45} median_d={v['median_d']:.3f}  {_format_crisis_dict(v['per_crisis'])}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
