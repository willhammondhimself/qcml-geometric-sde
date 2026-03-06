"""
Q18: Does time-delay embedding (Takens' theorem) before quantum embedding
improve dynamical fidelity?

Takens' theorem: a scalar time series x(t) can be embedded in d-dimensional
space using time delays:
    [x(t), x(t-1), x(t-2), ..., x(t-(d-1))]

This captures the geometry of the underlying dynamical attractor, which may
change shape more dramatically during crises than the standard cross-sectional
multi-feature approach.

Experimental design
--------------------
- Asset: SPY only (single asset, delay-embedded)
- Period: 2005-01-01 to 2024-12-31
- 4 crises: 2008_gfc, 2011_euro, 2020_covid, 2022_rates
- Takens: d=6 delay dimensions, tau=1 (consecutive days)
- Detector: SpectralEntropy (the best single QCML channel in Paper 1)
- Comparison: standard pipeline (returns + vol5 + vol20 + mom5 + mom20 on SPY)
- n_bootstrap: 1000 (smoke test, not publication quality)

Output
------
    research/ideation/takens_embedding/smoke_results.json
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

np.random.seed(42)

from experiments.data_loader import fetch_data, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import (
    SpectralEntropyDetector,
    ExpandingWindowMixin,
    BaseRegimeDetector,
    _standard_init,
    _standard_qcml_fit,
    _apply_normalization,
    _transform_array,
    _expanding_zscore,
)

# =============================================================================
# Constants
# =============================================================================

DELAY_DIM = 6        # embedding dimension d
TAU = 1              # time delay tau (daily steps)
HILBERT_DIM = 8
N_PCA = 6            # PCA components: at most DELAY_DIM, keep full signal
ROLLING_WINDOW = 20
MIN_EXPANDING = 60
SEED = 42
N_BOOTSTRAP = 1000   # smoke test
WINDOW_SIZE = 10     # crisis window extension in trading days

SMOKE_CRISES = ["2008_gfc", "2011_euro", "2020_covid", "2022_rates"]

# =============================================================================
# Takens embedding
# =============================================================================


def takens_embed(log_returns: np.ndarray, d: int = DELAY_DIM, tau: int = TAU) -> np.ndarray:
    """Construct Takens delay-embedding matrix from a scalar return series.

    For time index t (with t >= (d-1)*tau), the embedding vector is:
        v(t) = [r(t), r(t-tau), r(t-2*tau), ..., r(t-(d-1)*tau)]

    The result captures the geometry of the dynamical attractor of the
    underlying return-generating process (Takens 1981).

    Args:
        log_returns: 1-D array of log returns, length T.
        d: Embedding dimension (number of delay coordinates).
        tau: Time delay in samples.

    Returns:
        X_delay: 2-D array of shape (T - (d-1)*tau, d).
            Row i corresponds to time t = (d-1)*tau + i.
    """
    T = len(log_returns)
    warmup = (d - 1) * tau
    n_valid = T - warmup
    if n_valid <= 0:
        raise ValueError(f"Series too short ({T}) for d={d}, tau={tau}: need >= {warmup + 1}")

    X_delay = np.empty((n_valid, d), dtype=np.float64)
    for i in range(n_valid):
        t = warmup + i
        for k in range(d):
            X_delay[i, k] = log_returns[t - k * tau]
    return X_delay


def takens_dates(dates: pd.DatetimeIndex, d: int = DELAY_DIM, tau: int = TAU) -> pd.DatetimeIndex:
    """Return the date index aligned with the Takens embedding output.

    Args:
        dates: Full DatetimeIndex corresponding to log_returns.
        d: Embedding dimension.
        tau: Time delay.

    Returns:
        DatetimeIndex of length T - (d-1)*tau.
    """
    warmup = (d - 1) * tau
    return dates[warmup:]


# =============================================================================
# TakensSpectralEntropyDetector
# =============================================================================


class TakensSpectralEntropyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """SpectralEntropy detector operating on Takens delay-embedded features.

    Instead of the standard enriched feature matrix (returns + rolling stats),
    the input to QCMLGeometry is the Takens delay-embedded return vector:
        [r(t), r(t-1), ..., r(t-d+1)]

    This directly encodes the dynamical attractor geometry of the return
    series. The hypothesis (Q18) is that attractor shape changes more
    dramatically during financial crises than the standard cross-sectional
    feature set.

    The detector is otherwise identical to SpectralEntropyDetector:
    PCA + StandardScaler preprocessing, QCMLGeometry with random operators,
    spectral entropy scored via expanding z-score.

    Args:
        hilbert_dim: Hilbert space dimension for QCMLGeometry.
        n_pca_components: PCA components after StandardScaler.
        operator_method: Operator construction method ('random' recommended).
        rolling_window: Rolling mean window before z-scoring.
        min_expanding: Minimum history before computing z-scores.
        seed: Random seed for reproducibility.
        causal_fit_length: If set, fit only on first causal_fit_length rows.
        normalization: Post-PCA normalization ('soft' matches observatory default).
        adaptive_epsilon: Use adaptive finite-difference step.
    """

    def __init__(
        self,
        hilbert_dim: int = HILBERT_DIM,
        n_pca_components: int = N_PCA,
        operator_method: str = "random",
        scale_exponent=None,
        rolling_window: int = ROLLING_WINDOW,
        min_expanding: int = MIN_EXPANDING,
        seed: int = SEED,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        custom_operators=None,
        c: float = 1.0,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            c=c,
        )

    @property
    def name(self) -> str:
        return "Takens Spectral Entropy"

    def fit(self, X: np.ndarray, **kwargs) -> "TakensSpectralEntropyDetector":
        """Fit preprocessing (scaler + PCA + operators) on Takens-embedded X.

        Args:
            X: Takens delay-embedded feature matrix, shape (T, d).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute spectral entropy regime scores on Takens-embedded X.

        Args:
            X: Takens delay-embedded feature matrix, shape (T, d).

        Returns:
            scores: 1-D array of expanding z-scores, length T.
                NaN in the warmup period (first min_expanding rows).
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        vals = np.empty(T)
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t)
                if self._snapshots
                else (self._geometry, Xt[t])
            )
            vals[t] = geo.spectral_entropy(xt, c=self.c)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)


# =============================================================================
# Standard pipeline (for comparison)
# =============================================================================


def build_standard_features_single_asset(prices_series: pd.Series):
    """Build standard single-asset feature matrix (returns + rolling stats).

    This replicates the approach used in the main pipeline for single-asset
    analysis: log returns, 5/20-day volatility, 5/20-day momentum.

    Args:
        prices_series: pd.Series of close prices with DatetimeIndex.

    Returns:
        X: np.ndarray of shape (T', 5).
        dates: pd.DatetimeIndex of length T'.
    """
    log_ret = np.log(prices_series / prices_series.shift(1))
    feat_df = pd.DataFrame({
        "ret": log_ret,
        "vol5": log_ret.rolling(5).std(),
        "vol20": log_ret.rolling(20).std(),
        "mom5": prices_series.pct_change(5),
        "mom20": prices_series.pct_change(20),
    })
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feat_df.values, feat_df.index


# =============================================================================
# Per-crisis causal evaluation
# =============================================================================


def evaluate_detector_on_crisis(
    detector,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_key: str,
    n_bootstrap: int = N_BOOTSTRAP,
    window_size: int = WINDOW_SIZE,
):
    """Fit detector causally (pre-crisis data only) and compute Cohen's d.

    Args:
        detector: An unfitted detector instance (will be reset).
        X: Feature matrix aligned with dates.
        dates: DatetimeIndex of length T.
        crisis_key: Key into ALL_CRISES dict.
        n_bootstrap: Bootstrap resamples.
        window_size: Crisis window extension in trading days.

    Returns:
        dict with keys: d, ci_lo, ci_hi, n_crisis, n_normal.
        Returns None if insufficient pre-crisis data.
    """
    ci = ALL_CRISES[crisis_key]
    crisis_start = pd.Timestamp(ci["start"])
    crisis_end = pd.Timestamp(ci["end"])

    # Causal cutoff: fit only on data before crisis window
    cutoff_date = crisis_start - pd.Timedelta(days=window_size)
    fit_end_idx = int(np.searchsorted(dates, cutoff_date))

    if fit_end_idx < MIN_EXPANDING + 20:
        logger.warning(
            f"  Skipping {crisis_key}: insufficient pre-crisis data "
            f"({fit_end_idx} rows, need > {MIN_EXPANDING + 20})"
        )
        return None

    # Set causal fit length and fit
    detector.causal_fit_length = fit_end_idx
    detector.fit(X)
    scores = detector.compute_regime_scores(X)

    # Build crisis / normal masks
    cs = crisis_start - pd.Timedelta(days=window_size)
    ce = crisis_end + pd.Timedelta(days=window_size)
    crisis_mask = (dates >= cs) & (dates <= ce)
    normal_mask = ~crisis_mask

    crisis_scores = scores[crisis_mask]
    normal_scores = scores[normal_mask]

    d, ci_lo, ci_hi = compute_cohens_d_with_ci(
        crisis_scores, normal_scores,
        n_bootstrap=n_bootstrap, seed=SEED,
    )

    n_crisis = int(np.sum(~np.isnan(crisis_scores)))
    n_normal = int(np.sum(~np.isnan(normal_scores)))

    logger.info(
        f"    {crisis_key:20s}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
        f"  n_crisis={n_crisis}, n_normal={n_normal}"
    )
    return {
        "d": float(d) if not np.isnan(d) else None,
        "ci_lo": float(ci_lo) if not np.isnan(ci_lo) else None,
        "ci_hi": float(ci_hi) if not np.isnan(ci_hi) else None,
        "n_crisis": n_crisis,
        "n_normal": n_normal,
    }


# =============================================================================
# Main smoke test
# =============================================================================


def run_smoke_test():
    """Run Q18 empirical test: Takens vs standard pipeline on SPY.

    Fetches SPY data 2005-2024, builds two feature representations:
        1. Takens delay-embedded returns (d=6, tau=1)
        2. Standard single-asset features (ret + vol5 + vol20 + mom5 + mom20)

    Evaluates SpectralEntropyDetector on 4 crises with per-crisis causal fit.
    Saves results to research/ideation/takens_embedding/smoke_results.json.

    Returns:
        dict of results.
    """
    logger.info("=" * 60)
    logger.info("Q18: Takens Embedding Smoke Test")
    logger.info(f"  Delay dim d={DELAY_DIM}, tau={TAU}")
    logger.info(f"  Detector: SpectralEntropy (hilbert_dim={HILBERT_DIM})")
    logger.info(f"  Crises: {SMOKE_CRISES}")
    logger.info(f"  n_bootstrap={N_BOOTSTRAP}")
    logger.info("=" * 60)

    # ---- Fetch data ----
    logger.info("\n[1] Fetching SPY data 2005-2025...")
    raw = fetch_data(["SPY"], "2005-01-01", "2024-12-31", use_cache=True)
    prices_df = raw["close"].unstack("symbol").dropna()
    spy_prices = prices_df["SPY"]
    spy_log_ret = np.log(spy_prices / spy_prices.shift(1)).dropna()
    spy_log_ret_arr = spy_log_ret.values
    spy_dates = spy_log_ret.index

    logger.info(f"  SPY log returns: {len(spy_log_ret_arr)} days, "
                f"{spy_dates[0].date()} to {spy_dates[-1].date()}")

    # ---- Takens embedding ----
    logger.info(f"\n[2] Takens embedding (d={DELAY_DIM}, tau={TAU})...")
    X_takens = takens_embed(spy_log_ret_arr, d=DELAY_DIM, tau=TAU)
    dates_takens = takens_dates(spy_dates, d=DELAY_DIM, tau=TAU)
    logger.info(f"  Takens matrix: {X_takens.shape}, "
                f"{dates_takens[0].date()} to {dates_takens[-1].date()}")

    # ---- Standard features ----
    logger.info("\n[3] Building standard single-asset features...")
    X_standard, dates_standard = build_standard_features_single_asset(spy_prices)
    logger.info(f"  Standard matrix: {X_standard.shape}, "
                f"{dates_standard[0].date()} to {dates_standard[-1].date()}")

    # ---- Evaluate both pipelines per crisis ----
    takens_results = {}
    standard_results = {}

    logger.info("\n[4] Evaluating Takens pipeline...")
    for crisis_key in SMOKE_CRISES:
        if crisis_key not in ALL_CRISES:
            logger.warning(f"  Unknown crisis key: {crisis_key}, skipping.")
            continue
        det = TakensSpectralEntropyDetector(
            hilbert_dim=HILBERT_DIM,
            n_pca_components=N_PCA,
            operator_method="random",
            rolling_window=ROLLING_WINDOW,
            min_expanding=MIN_EXPANDING,
            seed=SEED,
            normalization="soft",
            adaptive_epsilon=True,
        )
        res = evaluate_detector_on_crisis(
            det, X_takens, dates_takens, crisis_key, N_BOOTSTRAP,
        )
        takens_results[crisis_key] = res

    logger.info("\n[5] Evaluating standard pipeline (SpectralEntropy on standard features)...")
    for crisis_key in SMOKE_CRISES:
        if crisis_key not in ALL_CRISES:
            continue
        # Use the stock SpectralEntropyDetector from observables
        det = SpectralEntropyDetector(
            hilbert_dim=HILBERT_DIM,
            n_pca_components=min(N_PCA, X_standard.shape[1]),
            operator_method="random",
            rolling_window=ROLLING_WINDOW,
            min_expanding=MIN_EXPANDING,
            seed=SEED,
            normalization="soft",
            adaptive_epsilon=True,
        )
        res = evaluate_detector_on_crisis(
            det, X_standard, dates_standard, crisis_key, N_BOOTSTRAP,
        )
        standard_results[crisis_key] = res

    # ---- Summarise ----
    logger.info("\n[6] Computing summary statistics...")

    def median_d(results_dict):
        vals = [
            v["d"] for v in results_dict.values()
            if v is not None and v["d"] is not None
        ]
        return float(np.median(vals)) if vals else float("nan")

    takens_d_per_crisis = {
        k: (v["d"] if v else None) for k, v in takens_results.items()
    }
    standard_d_per_crisis = {
        k: (v["d"] if v else None) for k, v in standard_results.items()
    }

    takens_median = median_d(takens_results)
    standard_median = median_d(standard_results)

    if not np.isnan(standard_median) and standard_median != 0:
        pct_change = (takens_median - standard_median) / abs(standard_median) * 100
        improvement_str = f"{pct_change:+.1f}%"
    else:
        improvement_str = "N/A"

    passes = (
        not np.isnan(takens_median)
        and not np.isnan(standard_median)
        and takens_median > standard_median
    )

    results = {
        "experiment": "Q18_takens_embedding_smoke_test",
        "config": {
            "delay_dim": DELAY_DIM,
            "tau": TAU,
            "hilbert_dim": HILBERT_DIM,
            "n_pca_components": N_PCA,
            "detector": "SpectralEntropyDetector",
            "crises": SMOKE_CRISES,
            "n_bootstrap": N_BOOTSTRAP,
            "period": "2005-01-01 to 2024-12-31",
            "asset": "SPY",
        },
        "takens_d_per_crisis": takens_d_per_crisis,
        "standard_d_per_crisis": standard_d_per_crisis,
        "takens_ci_per_crisis": {
            k: {"lo": v["ci_lo"], "hi": v["ci_hi"]} if v else None
            for k, v in takens_results.items()
        },
        "standard_ci_per_crisis": {
            k: {"lo": v["ci_lo"], "hi": v["ci_hi"]} if v else None
            for k, v in standard_results.items()
        },
        "takens_median_d": round(takens_median, 4),
        "standard_median_d": round(standard_median, 4),
        "improvement": improvement_str,
        "passes_threshold": passes,
        "interpretation": (
            "Takens embedding IMPROVES spectral entropy detection vs standard features."
            if passes else
            "Takens embedding DOES NOT improve spectral entropy detection vs standard features."
        ),
    }

    # ---- Print YAML summary ----
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    print("\ntakens_d_per_crisis:")
    for k, v in takens_d_per_crisis.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: null")
    print("standard_d_per_crisis:")
    for k, v in standard_d_per_crisis.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: null")
    print(f"takens_median_d: {takens_median:.4f}")
    print(f"standard_median_d: {standard_median:.4f}")
    print(f"improvement: {improvement_str}")
    print(f"passes_threshold: {passes}")
    print(f"interpretation: {results['interpretation']}")

    # ---- Save results ----
    output_dir = Path(__file__).parent
    output_path = output_dir / "smoke_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_smoke_test()
