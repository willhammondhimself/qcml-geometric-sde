"""Berry Phase Rate vs realized-volatility correlation.

Computes Pearson r and Spearman rho between the Berry Phase Rate causal
score series and SPY's 20-day rolling realized volatility, on the same
SPY/DIA feature matrix the paper uses.  Saves results JSON for the
"QCML is not just a vol proxy" referee preempt.

Output: experiments/outputs/orthogonality/berry_vol_correlation.json
"""

from __future__ import annotations

import json
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

load_dotenv(ROOT / ".env")

from experiments.data_loader import fetch_data, create_feature_matrix
from qcml_geometry.observables import BerryPhaseRateDetector

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"
SYMBOLS = ["SPY", "DIA"]
VOL_WINDOW = 20

# Default Berry params used in regime_comparison.py (same as paper canonical run)
BERRY_PARAMS = dict(
    hilbert_dim=6,
    n_pca_components=8,
    rolling_window=15,
    operator_method="random",
    seed=42,
    normalization="sphere",
    berry_aggregation="f01",
)

OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "outputs"
    / "orthogonality"
    / "berry_vol_correlation.json"
)


def realized_vol_20d(spy_close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling 20-day realized vol of SPY log returns, annualized."""
    log_ret = np.log(spy_close / spy_close.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv


def main() -> None:
    print(f"Fetching {SYMBOLS} {START}..{END}")
    raw = fetch_data(SYMBOLS, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    print(f"  prices_df shape: {prices_df.shape}, columns: {list(prices_df.columns)}")

    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features shape: {feat_arr.shape}")

    print("Fitting Berry detector and computing causal scores...")
    detector = BerryPhaseRateDetector(**BERRY_PARAMS)
    detector.fit(feat_arr)
    scores = detector.compute_regime_scores(feat_arr)
    score_series = pd.Series(scores, index=feat_index, name="berry_score")
    print(f"  score series: {len(score_series)} obs, NaN: {score_series.isna().sum()}")

    rv = realized_vol_20d(prices_df["SPY"].astype(float), VOL_WINDOW)
    rv = rv.reindex(feat_index)
    print(f"  rv series: {len(rv)} obs, NaN: {rv.isna().sum()}")

    # Align and drop NaN
    df = pd.concat([score_series, rv.rename("realized_vol_20d")], axis=1).dropna()
    n = len(df)
    print(f"  aligned (after NaN drop): {n} obs")

    # Correlations on raw series
    pearson_r, pearson_p = stats.pearsonr(df["berry_score"], df["realized_vol_20d"])
    spearman_r, spearman_p = stats.spearmanr(df["berry_score"], df["realized_vol_20d"])

    # Correlation on first-differences (control for nonstationarity)
    d_berry = df["berry_score"].diff().dropna()
    d_rv = df["realized_vol_20d"].diff().dropna()
    aligned = pd.concat([d_berry, d_rv], axis=1).dropna()
    pearson_r_diff, pearson_p_diff = stats.pearsonr(
        aligned.iloc[:, 0], aligned.iloc[:, 1]
    )

    # Lead/lag at +/- 30 days (does Berry lead vol?)
    lag_corrs = {}
    for lag in range(-30, 31, 5):
        if lag == 0:
            lag_corrs[lag] = float(pearson_r)
            continue
        if lag > 0:
            shifted = df["berry_score"].shift(lag)
        else:
            shifted = df["berry_score"].shift(lag)
        pair = pd.concat([shifted, df["realized_vol_20d"]], axis=1).dropna()
        if len(pair) < 100:
            continue
        r, _ = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        lag_corrs[lag] = float(r)

    results = {
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "vol_window_days": VOL_WINDOW,
            "berry_params": BERRY_PARAMS,
        },
        "summary": {
            "n_obs_aligned": int(n),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r_first_diff": float(pearson_r_diff),
            "pearson_p_first_diff": float(pearson_p_diff),
            "abs_pearson_r": float(abs(pearson_r)),
            "interpretation": (
                "negligible (|r|<0.1)" if abs(pearson_r) < 0.1
                else "small (0.1<=|r|<0.3)" if abs(pearson_r) < 0.3
                else "moderate (0.3<=|r|<0.5)" if abs(pearson_r) < 0.5
                else "large (|r|>=0.5)"
            ),
        },
        "lag_correlations": lag_corrs,
        "timestamp": datetime.now().isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=== RESULTS ===")
    print(f"  N aligned obs:         {n}")
    print(f"  Pearson r:             {pearson_r:+.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman rho:          {spearman_r:+.4f}  (p={spearman_p:.2e})")
    print(f"  Pearson on first diff: {pearson_r_diff:+.4f}  (p={pearson_p_diff:.2e})")
    print(f"  Interpretation:        {results['summary']['interpretation']}")
    print()
    print("Lag correlations (Berry shifted by +/-N days):")
    for lag in sorted(lag_corrs):
        print(f"  lag {lag:+3d}d: r = {lag_corrs[lag]:+.4f}")
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
