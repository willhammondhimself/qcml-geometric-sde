#!/usr/bin/env python3
"""
Granger Causality Analysis: QCML Geometric Observables vs Market Volatility

Tests whether QCML geometric regime scores (Berry phase rate, QFI determinant,
multi-lag fidelity) Granger-cause market volatility and return measures, or
vice versa. Granger causality establishes temporal precedence: if QCML scores
at lag t predict future volatility beyond what past volatility predicts, the
QCML framework provides genuinely *leading* information.

Methodology:
    For each (QCML_method, target_variable, lag) triple:
    1. Verify stationarity via Augmented Dickey-Fuller test; first-difference
       any non-stationary series.
    2. Run bivariate Granger causality (statsmodels grangercausalitytests):
       - Forward: does QCML_score(t-k) predict target(t)?
       - Reverse: does target(t-k) predict QCML_score(t)?
    3. Apply Bonferroni correction for multiple comparisons.

Target variables:
    - Realized volatility: 20-day rolling std of log returns
    - Absolute log returns: |log(P_t / P_{t-1})|
    - VIX proxy: expanding-window z-score of 20-day rolling volatility

Outputs:
    - granger_results.json with F-statistics, p-values, significance flags
    - Printed summary table

Usage:
    python experiments/granger_causality.py
    python experiments/granger_causality.py --output-dir experiments/outputs/regime_detection/granger

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from qcml_geometry import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.data import PolygonDataSource, MinimalFeatureEngine

load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("experiments/outputs/regime_detection/granger")

# Symbols used for feature engineering (MinimalFeatureEngine requires >= 2)
UNIVERSE = ["SPY", "XLF", "QQQ", "TLT"]
BENCHMARK = "SPY"

# Date range: long history for robust Granger tests
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"

# Granger test lags
GRANGER_LAGS = [1, 2, 3, 5, 10]

# Detector hyperparameters (match project defaults)
HILBERT_DIM = 8
N_PCA_COMPONENTS = 15
OPERATOR_METHOD = "random"
ROLLING_WINDOW = 20
MIN_EXPANDING = 60
SEED = 42

# Stationarity threshold
ADF_ALPHA = 0.05

# Volatility computation
VOL_WINDOW = 20


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def fetch_long_history(
    symbols: List[str],
    start_date: str,
    end_date: str,
    n_pca: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Series]:
    """Fetch multi-symbol data, build features, PCA-transform, and extract SPY close.

    Args:
        symbols: Ticker symbols for feature engineering.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        n_pca: Number of PCA components.

    Returns:
        X_enriched: Enriched feature matrix (T_trimmed, 4*n_pca).
        X_pca: PCA-transformed feature matrix (T, n_pca), normalized.
        times: DatetimeIndex aligned with X_pca.
        spy_close: SPY close price series aligned with times.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    logger.info(f"Fetching {symbols} from {start_date} to {end_date}")
    source = PolygonDataSource(api_key=api_key)
    raw = source.fetch_equities(symbols, start_date=start_date, end_date=end_date)

    if raw.empty:
        raise ValueError("No data returned from Polygon API")

    prices = raw["close"].unstack(level=0)
    prices = prices.ffill()

    # Keep SPY close for target construction
    spy_close = prices[BENCHMARK].copy()

    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=BENCHMARK)
    features = features.dropna()
    times = features.index

    X_raw = features.values
    logger.info(f"Raw features: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

    # Standardize and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(n_pca, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

    # Build enriched features (rolling mean/std/min/max of PCA components)
    enriched_lookback = 20
    X_enriched = BaseRegimeDetector.build_enriched_features(
        X_pca, lookback=enriched_lookback
    )

    # Trim times and spy_close to match enriched features
    trim = enriched_lookback - 1
    times_enriched = times[trim:]
    spy_close_aligned = spy_close.reindex(times_enriched)

    logger.info(
        f"After PCA: {X_pca.shape}, enriched: {X_enriched.shape}, "
        f"times: {len(times_enriched)}"
    )

    return X_enriched, X_pca[trim:], times_enriched, spy_close_aligned


# ---------------------------------------------------------------------------
# QCML score computation
# ---------------------------------------------------------------------------


def compute_qcml_scores(
    X_enriched: np.ndarray,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Fit and compute regime scores for the 3 primary QCML detectors.

    Args:
        X_enriched: Enriched feature matrix (T, n_features).
        seed: Random seed.

    Returns:
        Dictionary mapping method name to 1-D score array of length T.
    """
    detectors = {
        "Berry Phase Rate": BerryPhaseRateDetector(
            hilbert_dim=HILBERT_DIM,
            n_pca_components=N_PCA_COMPONENTS,
            operator_method=OPERATOR_METHOD,
            rolling_window=ROLLING_WINDOW,
            min_expanding=MIN_EXPANDING,
            seed=seed,
        ),
        "QFI Determinant": QFIDeterminantDetector(
            hilbert_dim=HILBERT_DIM,
            n_pca_components=N_PCA_COMPONENTS,
            operator_method=OPERATOR_METHOD,
            rolling_window=ROLLING_WINDOW,
            min_expanding=MIN_EXPANDING,
            seed=seed,
        ),
        "Multi-Lag Fidelity": MultiLagFidelityDetector(
            hilbert_dim=HILBERT_DIM,
            n_pca_components=N_PCA_COMPONENTS,
            operator_method=OPERATOR_METHOD,
            rolling_window=ROLLING_WINDOW,
            min_expanding=MIN_EXPANDING,
            seed=seed,
        ),
    }

    scores = {}
    for name, det in detectors.items():
        logger.info(f"Computing {name} scores...")
        np.random.seed(seed)
        det.fit(X_enriched)
        s = det.compute_regime_scores(X_enriched)
        scores[name] = s
        n_valid = np.sum(~np.isnan(s))
        logger.info(f"  {name}: {n_valid}/{len(s)} valid scores")

    return scores


# ---------------------------------------------------------------------------
# Target variable construction
# ---------------------------------------------------------------------------


def compute_target_variables(
    spy_close: pd.Series,
    vol_window: int = VOL_WINDOW,
    min_expanding: int = MIN_EXPANDING,
) -> Dict[str, np.ndarray]:
    """Compute market target variables from SPY close prices.

    Args:
        spy_close: SPY daily close prices, DatetimeIndex.
        vol_window: Window for realized volatility.
        min_expanding: Minimum observations for z-score computation.

    Returns:
        Dictionary mapping target name to 1-D array aligned with spy_close index.
    """
    log_returns = np.log(spy_close / spy_close.shift(1))

    # 1. Realized volatility: 20-day rolling std of log returns
    realized_vol = log_returns.rolling(window=vol_window, min_periods=vol_window).std()

    # 2. Absolute log returns
    abs_returns = log_returns.abs()

    # 3. VIX proxy: expanding z-score of 20-day rolling volatility
    vix_proxy = pd.Series(np.full(len(spy_close), np.nan), index=spy_close.index)
    rv_vals = realized_vol.values
    for t in range(min_expanding, len(rv_vals)):
        if np.isnan(rv_vals[t]):
            continue
        past = rv_vals[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma = np.std(past_valid, ddof=1)
        if sigma > 1e-12:
            vix_proxy.iloc[t] = (rv_vals[t] - mu) / sigma

    return {
        "realized_vol": realized_vol.values,
        "abs_returns": abs_returns.values,
        "vix_proxy": vix_proxy.values,
    }


# ---------------------------------------------------------------------------
# Stationarity and differencing
# ---------------------------------------------------------------------------


def adf_test(series: np.ndarray, name: str = "") -> Tuple[float, bool]:
    """Augmented Dickey-Fuller stationarity test.

    Args:
        series: 1-D array (NaN-free).
        name: Label for logging.

    Returns:
        (p_value, is_stationary) where is_stationary = p_value < ADF_ALPHA.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, autolag="AIC")
    p_value = result[1]
    is_stationary = p_value < ADF_ALPHA
    logger.info(
        f"  ADF {name}: stat={result[0]:.4f}, p={p_value:.6f}, "
        f"{'stationary' if is_stationary else 'NON-STATIONARY'}"
    )
    return p_value, is_stationary


def ensure_stationary(
    series: np.ndarray,
    name: str = "",
) -> Tuple[np.ndarray, bool]:
    """Check stationarity; first-difference if non-stationary.

    Args:
        series: 1-D array (may contain NaN).
        name: Label for logging.

    Returns:
        (stationary_series, was_differenced).
    """
    # Remove NaN for ADF test
    valid_mask = ~np.isnan(series)
    valid = series[valid_mask]

    if len(valid) < 50:
        logger.warning(f"  {name}: too few valid observations ({len(valid)})")
        return series, False

    _, is_stationary = adf_test(valid, name)

    if is_stationary:
        return series, False

    # First-difference
    diffed = np.full_like(series, np.nan)
    for i in range(1, len(series)):
        if not np.isnan(series[i]) and not np.isnan(series[i - 1]):
            diffed[i] = series[i] - series[i - 1]

    valid_diffed = diffed[~np.isnan(diffed)]
    if len(valid_diffed) >= 50:
        _, is_now_stationary = adf_test(valid_diffed, f"{name} (differenced)")
        if not is_now_stationary:
            logger.warning(f"  {name}: still non-stationary after differencing")

    return diffed, True


# ---------------------------------------------------------------------------
# Granger causality testing
# ---------------------------------------------------------------------------


def run_granger_test(
    cause: np.ndarray,
    effect: np.ndarray,
    lags: List[int],
    cause_name: str = "cause",
    effect_name: str = "effect",
) -> Dict[int, Dict[str, float]]:
    """Run Granger causality test: does `cause` Granger-cause `effect`?

    Uses statsmodels grangercausalitytests. The function expects a 2-column
    array where column 0 is the *response* (effect) and column 1 is the
    *predictor* (cause).

    Args:
        cause: Potential causal series (1-D, may contain NaN).
        effect: Response series (1-D, may contain NaN).
        lags: List of lag orders to test.
        cause_name: Label for the cause variable.
        effect_name: Label for the effect variable.

    Returns:
        Dictionary: lag -> {f_stat, p_value} from the ssr_ftest.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Align and drop NaN
    combined = np.column_stack([effect, cause])
    valid_mask = ~np.isnan(combined).any(axis=1)
    combined_valid = combined[valid_mask]

    max_lag = max(lags)
    if len(combined_valid) < max_lag + 50:
        logger.warning(
            f"  Granger {cause_name} -> {effect_name}: insufficient data "
            f"({len(combined_valid)} obs, need {max_lag + 50})"
        )
        return {}

    results = {}
    try:
        # grangercausalitytests prints output; suppress it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_results = grangercausalitytests(
                combined_valid, maxlag=max_lag, verbose=False
            )

        for lag in lags:
            if lag in gc_results:
                # Extract SSR F-test results: (F-stat, p-value, df_denom, df_num)
                f_test = gc_results[lag][0]["ssr_ftest"]
                results[lag] = {
                    "f_stat": float(f_test[0]),
                    "p_value": float(f_test[1]),
                }

    except Exception as e:
        logger.error(f"  Granger test failed ({cause_name} -> {effect_name}): {e}")

    return results


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_granger_analysis(
    X_enriched: np.ndarray,
    spy_close: pd.Series,
    times: pd.DatetimeIndex,
    seed: int = SEED,
    lags: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run full Granger causality analysis.

    For each QCML method x target variable x lag:
      - Forward test: QCML -> target
      - Reverse test: target -> QCML

    Args:
        X_enriched: Enriched feature matrix.
        spy_close: SPY close prices aligned with X_enriched.
        times: DatetimeIndex.
        seed: Random seed.
        lags: Lag orders to test.

    Returns:
        Nested results dictionary.
    """
    if lags is None:
        lags = GRANGER_LAGS

    # Step 1: Compute QCML scores
    logger.info("=" * 60)
    logger.info("Step 1: Computing QCML regime scores")
    logger.info("=" * 60)
    qcml_scores = compute_qcml_scores(X_enriched, seed=seed)

    # Step 2: Compute target variables
    logger.info("=" * 60)
    logger.info("Step 2: Computing target variables")
    logger.info("=" * 60)
    targets = compute_target_variables(spy_close)

    # Step 3: Stationarity checks and differencing
    logger.info("=" * 60)
    logger.info("Step 3: Stationarity tests (ADF)")
    logger.info("=" * 60)

    stationary_scores = {}
    score_differenced = {}
    for name, s in qcml_scores.items():
        s_stat, was_diff = ensure_stationary(s, name=name)
        stationary_scores[name] = s_stat
        score_differenced[name] = was_diff

    stationary_targets = {}
    target_differenced = {}
    for name, t in targets.items():
        t_stat, was_diff = ensure_stationary(t, name=name)
        stationary_targets[name] = t_stat
        target_differenced[name] = was_diff

    # Step 4: Granger causality tests
    logger.info("=" * 60)
    logger.info("Step 4: Granger causality tests")
    logger.info("=" * 60)

    # Count total tests for Bonferroni correction
    n_methods = len(stationary_scores)
    n_targets = len(stationary_targets)
    n_lags = len(lags)
    n_directions = 2  # forward + reverse
    n_total_tests = n_methods * n_targets * n_lags * n_directions
    bonferroni_alpha = 0.05 / n_total_tests

    logger.info(f"Total tests: {n_total_tests}")
    logger.info(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

    forward_results = {}  # QCML -> target
    reverse_results = {}  # target -> QCML

    for method_name, method_scores in stationary_scores.items():
        forward_results[method_name] = {}
        reverse_results[method_name] = {}

        for target_name, target_series in stationary_targets.items():
            logger.info(f"  Testing: {method_name} <-> {target_name}")

            # Forward: QCML Granger-causes target
            fwd = run_granger_test(
                cause=method_scores,
                effect=target_series,
                lags=lags,
                cause_name=method_name,
                effect_name=target_name,
            )

            fwd_formatted = {}
            for lag, res in fwd.items():
                fwd_formatted[f"lag_{lag}"] = {
                    "f_stat": res["f_stat"],
                    "p_value": res["p_value"],
                    "significant_nominal": res["p_value"] < 0.05,
                    "significant_bonferroni": res["p_value"] < bonferroni_alpha,
                }
            forward_results[method_name][target_name] = fwd_formatted

            # Reverse: target Granger-causes QCML
            rev = run_granger_test(
                cause=target_series,
                effect=method_scores,
                lags=lags,
                cause_name=target_name,
                effect_name=method_name,
            )

            rev_formatted = {}
            for lag, res in rev.items():
                rev_formatted[f"lag_{lag}"] = {
                    "f_stat": res["f_stat"],
                    "p_value": res["p_value"],
                    "significant_nominal": res["p_value"] < 0.05,
                    "significant_bonferroni": res["p_value"] < bonferroni_alpha,
                }
            reverse_results[method_name][target_name] = rev_formatted

    # Step 5: Compile stationarity metadata
    stationarity_info = {
        "scores": {
            name: {"differenced": diff}
            for name, diff in score_differenced.items()
        },
        "targets": {
            name: {"differenced": diff}
            for name, diff in target_differenced.items()
        },
    }

    return {
        "forward": forward_results,
        "reverse": reverse_results,
        "stationarity": stationarity_info,
        "bonferroni_alpha": bonferroni_alpha,
        "n_total_tests": n_total_tests,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print a formatted summary of Granger causality results.

    Args:
        results: Output from run_granger_analysis.
    """
    bonf_alpha = results["bonferroni_alpha"]

    print("\n" + "=" * 90)
    print("GRANGER CAUSALITY RESULTS")
    print("=" * 90)
    print(f"Bonferroni-corrected alpha: {bonf_alpha:.6f} ({results['n_total_tests']} tests)")
    print()

    # Stationarity summary
    print("Stationarity (ADF test, alpha=0.05):")
    for category in ["scores", "targets"]:
        for name, info in results["stationarity"][category].items():
            status = "differenced" if info["differenced"] else "stationary"
            print(f"  {name}: {status}")
    print()

    for direction_label, direction_key in [
        ("FORWARD: QCML -> Market (QCML predicts market)", "forward"),
        ("REVERSE: Market -> QCML (market predicts QCML)", "reverse"),
    ]:
        print("-" * 90)
        print(direction_label)
        print("-" * 90)

        header = f"{'Method':<22s} {'Target':<16s}"
        for lag in GRANGER_LAGS:
            header += f"  lag={lag:<3d}"
        print(header)
        print("-" * 90)

        direction_data = results[direction_key]
        for method_name, targets in direction_data.items():
            for target_name, lag_results in targets.items():
                row = f"{method_name:<22s} {target_name:<16s}"
                for lag in GRANGER_LAGS:
                    key = f"lag_{lag}"
                    if key in lag_results:
                        p = lag_results[key]["p_value"]
                        sig_bonf = lag_results[key]["significant_bonferroni"]
                        sig_nom = lag_results[key]["significant_nominal"]
                        if sig_bonf:
                            marker = "***"
                        elif sig_nom:
                            marker = " * "
                        else:
                            marker = "   "
                        row += f"  {p:.4f}{marker}"
                    else:
                        row += "      N/A  "
                print(row)
        print()

    # Count significant results
    print("=" * 90)
    print("SIGNIFICANCE SUMMARY")
    print("=" * 90)

    for direction_label, direction_key in [
        ("Forward (QCML -> Market)", "forward"),
        ("Reverse (Market -> QCML)", "reverse"),
    ]:
        n_nominal = 0
        n_bonferroni = 0
        n_total = 0

        for method_name, targets in results[direction_key].items():
            for target_name, lag_results in targets.items():
                for lag_key, lr in lag_results.items():
                    n_total += 1
                    if lr["significant_nominal"]:
                        n_nominal += 1
                    if lr["significant_bonferroni"]:
                        n_bonferroni += 1

        print(
            f"  {direction_label}: "
            f"{n_bonferroni}/{n_total} Bonferroni-significant, "
            f"{n_nominal}/{n_total} nominally significant (p<0.05)"
        )

    # Per-method summary for forward direction
    print()
    print("Per-method forward results (QCML -> Market):")
    for method_name, targets in results["forward"].items():
        best_p = 1.0
        best_combo = ""
        n_sig = 0
        for target_name, lag_results in targets.items():
            for lag_key, lr in lag_results.items():
                if lr["significant_bonferroni"]:
                    n_sig += 1
                if lr["p_value"] < best_p:
                    best_p = lr["p_value"]
                    lag_num = lag_key.replace("lag_", "")
                    best_combo = f"{target_name} lag={lag_num}"

        sig_marker = "***" if best_p < bonf_alpha else ("*" if best_p < 0.05 else "")
        print(
            f"  {method_name:<22s}: best p={best_p:.6f} {sig_marker} "
            f"({best_combo}), {n_sig} Bonferroni-significant"
        )

    print()
    print("Legend: *** = Bonferroni-significant, * = nominally significant (p<0.05)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Granger causality analysis."""
    parser = argparse.ArgumentParser(
        description="Granger Causality: QCML Geometric Observables vs Market Volatility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory for output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE,
        help="Data start date (default: 2005-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=END_DATE,
        help="Data end date (default: 2024-12-31)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GRANGER CAUSALITY ANALYSIS")
    print("QCML Geometric Observables vs Market Volatility/Returns")
    print("=" * 70)
    print(f"Universe: {UNIVERSE}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Lags tested: {GRANGER_LAGS}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Fetch data
    X_enriched, X_pca, times, spy_close = fetch_long_history(
        symbols=UNIVERSE,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"\nData loaded: {len(times)} trading days, {X_enriched.shape[1]} enriched features")
    print(f"Date range: {times[0].date()} to {times[-1].date()}")

    # Run analysis
    results = run_granger_analysis(
        X_enriched=X_enriched,
        spy_close=spy_close,
        times=times,
        seed=args.seed,
    )

    # Print summary
    print_summary_table(results)

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "granger_causality",
        "parameters": {
            "seed": args.seed,
            "universe": UNIVERSE,
            "benchmark": BENCHMARK,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "granger_lags": GRANGER_LAGS,
            "hilbert_dim": HILBERT_DIM,
            "n_pca_components": N_PCA_COMPONENTS,
            "operator_method": OPERATOR_METHOD,
            "vol_window": VOL_WINDOW,
            "adf_alpha": ADF_ALPHA,
        },
        "n_trading_days": int(len(times)),
        "date_range": [str(times[0].date()), str(times[-1].date())],
        "forward_results": results["forward"],
        "reverse_results": results["reverse"],
        "stationarity": results["stationarity"],
        "bonferroni_alpha": results["bonferroni_alpha"],
        "n_total_tests": results["n_total_tests"],
    }

    results_path = output_dir / "granger_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
