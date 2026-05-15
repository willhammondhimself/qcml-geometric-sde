"""Does Berry Phase Rate add information beyond realized vol?

The raw Pearson correlation between Berry's causal z-score and realized vol
is +0.52 — both are crisis detectors so positive correlation is expected.
The defensible "QCML is not just a vol proxy" question is whether Berry
provides INCREMENTAL information beyond vol, conditioned on vol level.

This script runs four tests:

1. Logistic regression of crisis label on {vol, Berry}: is Berry coefficient
   significant after controlling for vol?
2. McNemar test on crisis alarms (Berry-only vs vol-only thresholding).
3. Partial Pearson r between Berry and crisis label, controlling for vol.
4. Lead-time analysis: in pre-crisis windows, does Berry rise before vol?

Output: experiments/outputs/orthogonality/berry_vs_vol_incremental.json
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

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from qcml_geometry.observables import BerryPhaseRateDetector

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"
SYMBOLS = ["SPY", "DIA"]
VOL_WINDOW = 20
ALARM_Z = 2.0  # detection threshold (paper convention)

BERRY_PARAMS = dict(
    hilbert_dim=6,
    n_pca_components=8,
    rolling_window=15,
    operator_method="random",
    seed=42,
    normalization="sphere",
    berry_aggregation="f01",
    adaptive_z_window=252,  # paper's canonical adaptive window
)

OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "outputs"
    / "orthogonality"
    / "berry_vs_vol_incremental.json"
)


def build_crisis_label(dates: pd.DatetimeIndex) -> pd.Series:
    """1 inside any historical crisis window, 0 otherwise."""
    label = pd.Series(0, index=dates, dtype=int)
    for crisis in ALL_CRISES.values():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        mask = (dates >= start) & (dates <= end)
        label.loc[mask] = 1
    return label


def realized_vol_z(spy_close: pd.Series, vol_window: int, z_window: int = 252) -> pd.Series:
    """Realized vol followed by causal expanding |z|, matching Berry's z."""
    log_ret = np.log(spy_close / spy_close.shift(1))
    rv = log_ret.rolling(vol_window).std() * np.sqrt(252)
    z = pd.Series(np.nan, index=rv.index)
    rv_arr = rv.values
    for t in range(len(rv_arr)):
        if t < z_window:
            window = rv_arr[: t + 1]
        else:
            window = rv_arr[t - z_window : t]
        if np.isnan(rv_arr[t]) or len(window[~np.isnan(window)]) < 30:
            continue
        mu = np.nanmedian(window)
        sigma = 1.4826 * np.nanmedian(np.abs(window - mu))
        if sigma > 1e-12:
            z.iloc[t] = abs((rv_arr[t] - mu) / sigma)
    return z


def main() -> None:
    print(f"Fetching {SYMBOLS} {START}..{END}")
    raw = fetch_data(SYMBOLS, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    spy_close = prices_df["SPY"].astype(float)

    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features shape: {feat_arr.shape}")

    print("Fitting Berry detector and computing causal z-scores...")
    detector = BerryPhaseRateDetector(**BERRY_PARAMS)
    detector.fit(feat_arr)
    berry_z = pd.Series(
        detector.compute_regime_scores(feat_arr), index=feat_index, name="berry_z"
    )

    rv_levels = (
        np.log(spy_close / spy_close.shift(1)).rolling(VOL_WINDOW).std() * np.sqrt(252)
    )
    rv_levels = rv_levels.reindex(feat_index).rename("rv_levels")

    rv_z = realized_vol_z(spy_close, VOL_WINDOW).reindex(feat_index).rename("rv_z")

    label = build_crisis_label(feat_index).rename("crisis")

    df = pd.concat([berry_z, rv_levels, rv_z, label], axis=1).dropna()
    n = len(df)
    n_crisis = int(df["crisis"].sum())
    print(f"  aligned: {n} obs, {n_crisis} crisis days ({100*n_crisis/n:.1f}%)")

    # Test 1 — incremental logistic regression: P(crisis | rv_z, berry_z)
    from sklearn.linear_model import LogisticRegression
    X = df[["rv_z", "berry_z"]].values
    y = df["crisis"].values

    full = LogisticRegression(max_iter=1000).fit(X, y)
    vol_only = LogisticRegression(max_iter=1000).fit(df[["rv_z"]].values, y)
    berry_only = LogisticRegression(max_iter=1000).fit(df[["berry_z"]].values, y)

    full_ll = -np.log(np.clip(full.predict_proba(X)[np.arange(n), y], 1e-12, 1)).sum()
    vol_only_ll = -np.log(np.clip(vol_only.predict_proba(df[["rv_z"]].values)[np.arange(n), y], 1e-12, 1)).sum()
    berry_only_ll = -np.log(np.clip(berry_only.predict_proba(df[["berry_z"]].values)[np.arange(n), y], 1e-12, 1)).sum()

    # Likelihood ratio test: full vs vol_only (extra Berry coefficient)
    lr_stat_berry = 2 * (vol_only_ll - full_ll)
    lr_p_berry = float(stats.chi2.sf(lr_stat_berry, df=1))
    lr_stat_vol = 2 * (berry_only_ll - full_ll)
    lr_p_vol = float(stats.chi2.sf(lr_stat_vol, df=1))

    print()
    print("=== Test 1: incremental logistic regression P(crisis | rv_z, berry_z) ===")
    print(f"  Vol-only NLL:    {vol_only_ll:.1f}")
    print(f"  Berry-only NLL:  {berry_only_ll:.1f}")
    print(f"  Full NLL:        {full_ll:.1f}")
    print(f"  LR test for Berry|vol:  chi2(1) = {lr_stat_berry:.2f}  p = {lr_p_berry:.2e}")
    print(f"  LR test for vol|Berry:  chi2(1) = {lr_stat_vol:.2f}  p = {lr_p_vol:.2e}")
    print(f"  Coefficients (full): rv_z={full.coef_[0,0]:+.3f}, berry_z={full.coef_[0,1]:+.3f}")

    # Test 2 — McNemar on alarms (z > ALARM_Z)
    berry_alarm = (df["berry_z"] > ALARM_Z).astype(int).values
    vol_alarm = (df["rv_z"] > ALARM_Z).astype(int).values
    n_b1_v0 = int(((berry_alarm == 1) & (vol_alarm == 0)).sum())
    n_b0_v1 = int(((berry_alarm == 0) & (vol_alarm == 1)).sum())
    n_b1_v1 = int(((berry_alarm == 1) & (vol_alarm == 1)).sum())
    # McNemar's chi-squared (with continuity correction)
    if n_b1_v0 + n_b0_v1 > 0:
        mcnemar_stat = (abs(n_b1_v0 - n_b0_v1) - 1) ** 2 / (n_b1_v0 + n_b0_v1)
        mcnemar_p = float(stats.chi2.sf(mcnemar_stat, df=1))
    else:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0

    print()
    print("=== Test 2: McNemar on alarms (z > 2.0) ===")
    print(f"  Berry-only alarms: {n_b1_v0}")
    print(f"  Vol-only alarms:   {n_b0_v1}")
    print(f"  Both alarmed:      {n_b1_v1}")
    print(f"  McNemar chi2 = {mcnemar_stat:.2f}, p = {mcnemar_p:.2e}")
    # In-crisis / out-of-crisis breakdown
    in_crisis = df["crisis"] == 1
    n_b1_v0_crisis = int(((berry_alarm == 1) & (vol_alarm == 0) & in_crisis).sum())
    n_b0_v1_crisis = int(((berry_alarm == 0) & (vol_alarm == 1) & in_crisis).sum())
    print(f"  Of Berry-only alarms inside crisis windows:  {n_b1_v0_crisis}/{n_b1_v0}  ({100*n_b1_v0_crisis/max(n_b1_v0,1):.1f}%)")
    print(f"  Of Vol-only   alarms inside crisis windows:  {n_b0_v1_crisis}/{n_b0_v1}  ({100*n_b0_v1_crisis/max(n_b0_v1,1):.1f}%)")

    # Test 3 — partial correlation Berry vs crisis-label, controlling for rv_z
    def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
        # Regress x and y on z, take residual correlation
        zz = z.reshape(-1, 1)
        x_resid = x - np.linalg.lstsq(np.c_[zz, np.ones_like(z)], x, rcond=None)[0][0] * z
        y_resid = y - np.linalg.lstsq(np.c_[zz, np.ones_like(z)], y, rcond=None)[0][0] * z
        return stats.pearsonr(x_resid, y_resid)

    pc_berry_crisis_given_vol = partial_corr(
        df["berry_z"].values.astype(float),
        df["crisis"].values.astype(float),
        df["rv_z"].values.astype(float),
    )
    pc_vol_crisis_given_berry = partial_corr(
        df["rv_z"].values.astype(float),
        df["crisis"].values.astype(float),
        df["berry_z"].values.astype(float),
    )

    print()
    print("=== Test 3: partial correlations (vs crisis label) ===")
    print(f"  partial r(Berry, crisis | vol):  {pc_berry_crisis_given_vol[0]:+.4f}  p = {pc_berry_crisis_given_vol[1]:.2e}")
    print(f"  partial r(vol, crisis | Berry):  {pc_vol_crisis_given_berry[0]:+.4f}  p = {pc_vol_crisis_given_berry[1]:.2e}")

    # Test 4 — lead/lag at +/-30d on raw z-score series
    lag_corrs = {}
    for lag in range(-30, 31, 5):
        shifted = df["berry_z"].shift(lag)
        pair = pd.concat([shifted, df["rv_z"]], axis=1).dropna()
        if len(pair) < 100:
            continue
        r, _ = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        lag_corrs[lag] = float(r)
    best_lag = max(lag_corrs, key=lambda k: lag_corrs[k])

    print()
    print("=== Test 4: lead/lag z-score correlation ===")
    for lag in sorted(lag_corrs):
        marker = " <-- peak" if lag == best_lag else ""
        print(f"  lag {lag:+3d}d: r = {lag_corrs[lag]:+.4f}{marker}")

    # Pearson on z vs z and z vs levels (for record)
    pearson_z_z, _ = stats.pearsonr(df["berry_z"], df["rv_z"])
    pearson_z_level, _ = stats.pearsonr(df["berry_z"], df["rv_levels"])
    spearman_z_z, _ = stats.spearmanr(df["berry_z"], df["rv_z"])

    results = {
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "vol_window_days": VOL_WINDOW,
            "alarm_z": ALARM_Z,
            "berry_params": BERRY_PARAMS,
        },
        "summary": {
            "n_obs": int(n),
            "n_crisis_days": n_crisis,
            "pct_crisis": float(100 * n_crisis / n),
            "pearson_berry_z_vs_rv_z": float(pearson_z_z),
            "pearson_berry_z_vs_rv_levels": float(pearson_z_level),
            "spearman_berry_z_vs_rv_z": float(spearman_z_z),
        },
        "test1_logistic_regression": {
            "full_NLL": float(full_ll),
            "vol_only_NLL": float(vol_only_ll),
            "berry_only_NLL": float(berry_only_ll),
            "LR_chi2_berry_given_vol": float(lr_stat_berry),
            "LR_p_berry_given_vol": lr_p_berry,
            "LR_chi2_vol_given_berry": float(lr_stat_vol),
            "LR_p_vol_given_berry": lr_p_vol,
            "coef_rv_z": float(full.coef_[0, 0]),
            "coef_berry_z": float(full.coef_[0, 1]),
        },
        "test2_mcnemar": {
            "berry_only_alarms": n_b1_v0,
            "vol_only_alarms": n_b0_v1,
            "both_alarmed": n_b1_v1,
            "mcnemar_chi2": float(mcnemar_stat),
            "mcnemar_p": mcnemar_p,
            "berry_only_alarms_inside_crisis": n_b1_v0_crisis,
            "vol_only_alarms_inside_crisis": n_b0_v1_crisis,
            "berry_only_precision": float(n_b1_v0_crisis / max(n_b1_v0, 1)),
            "vol_only_precision": float(n_b0_v1_crisis / max(n_b0_v1, 1)),
        },
        "test3_partial_correlations": {
            "pcorr_berry_crisis_given_vol_r": float(pc_berry_crisis_given_vol[0]),
            "pcorr_berry_crisis_given_vol_p": float(pc_berry_crisis_given_vol[1]),
            "pcorr_vol_crisis_given_berry_r": float(pc_vol_crisis_given_berry[0]),
            "pcorr_vol_crisis_given_berry_p": float(pc_vol_crisis_given_berry[1]),
        },
        "test4_lag_correlations": lag_corrs,
        "test4_best_lag_days": int(best_lag),
        "timestamp": datetime.now().isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
