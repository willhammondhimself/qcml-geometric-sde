"""Angle 5: peak Berry curvature in pre-crisis window predicts subsequent peak drawdown.

For each crisis (17-19 historical events from ALL_CRISES):
  crisis_start = t0
  crisis_end = t1
  For each lookback L in {20, 40, 60, 120, 252}:
    peak_berry[L]  = max(Berry curvature Frobenius norm) over [t0 - L, t0]
    peak_vol[L]    = max(20-day realized vol)            over [t0 - L, t0]
  eventual_dd      = (min SPY close in [t0, t1]) / (max SPY close in [t0, t1]) - 1
                     (negative; e.g., -0.55 = 55% peak-to-trough drawdown)

Then for each L:
  Regression A: dd ~ peak_berry[L]
  Regression B: dd ~ peak_vol[L]
  Regression C: dd ~ peak_berry[L] + peak_vol[L]    (joint)
  Report R^2 for each, slope significance, LOOCV MAE.
  Incremental F-test: does Berry add to vol model?

Pass criteria:
  - R^2(Berry) > 0.40 for at least one L
  - Incremental F-test p < 0.05 (Berry adds beyond vol)
  - LOOCV MAE(Berry) < 0.08

Uses canonical-paper Berry params: hd=6, n_pca=8, op=random (seed=0 default), causal embedding fit on full series (not walk-forward — this is a regression, not a detector evaluation).

Output: experiments/outputs/curvature_drawdown/curvature_drawdown_regression.json
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

from experiments.data_loader import (
    fetch_data,
    create_feature_matrix,
    ALL_CRISES,
)
from qcml_geometry.core import QCMLGeometry
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"
SYMBOLS = ["SPY", "DIA"]

HILBERT_DIM = 6
N_PCA = 8
SEED = 0  # canonical hd=6 random operator basis (matches the d=-1.14 endo/exo finding)
LOOKBACK_DAYS = [20, 40, 60, 120, 252]
VOL_WINDOW = 20

OUTPUT_PATH = (
    ROOT / "experiments" / "outputs" / "curvature_drawdown" / "curvature_drawdown_regression.json"
)


def fit_geometry(X: np.ndarray, hilbert_dim: int = HILBERT_DIM) -> tuple:
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=min(N_PCA, X.shape[1])).fit(scaler.transform(X))
    X_pca = pca.transform(scaler.transform(X))
    geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
    np.random.seed(SEED)
    geo.fit_operators(X_pca, method="random")
    return geo, scaler, pca


def berry_curvature_series(geo: QCMLGeometry, X_pca: np.ndarray) -> np.ndarray:
    """Compute Frobenius norm of Berry curvature tensor at each timestep."""
    T = len(X_pca)
    out = np.full(T, np.nan)
    for t in range(T):
        try:
            F = geo.berry_curvature(X_pca[t], epsilon=1e-5)
            out[t] = np.linalg.norm(F, ord="fro")
        except Exception:
            out[t] = np.nan
    return out


def realized_vol(spy_close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    log_ret = np.log(spy_close / spy_close.shift(1))
    return (log_ret.rolling(window).std() * np.sqrt(252))


def peak_drawdown(price: pd.Series) -> float:
    """Most negative peak-to-trough drawdown over the window."""
    if price.empty:
        return np.nan
    cummax = price.cummax()
    dd = (price - cummax) / cummax
    return float(dd.min())


def compute_loocv_mae(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out MAE with linear regression."""
    n = len(y)
    if n < 3:
        return float("nan")
    errors = np.empty(n)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        m = LinearRegression().fit(X[train_mask], y[train_mask])
        pred = m.predict(X[i : i + 1])
        errors[i] = abs(pred[0] - y[i])
    return float(np.mean(errors))


def regression_summary(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    m = LinearRegression().fit(X, y)
    pred = m.predict(X)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    # slope significance via residual SE
    n, p = X.shape
    if n - p - 1 > 0:
        sigma2 = ss_res / (n - p - 1)
        cov = sigma2 * np.linalg.pinv(X.T @ X - n * np.outer(X.mean(0), X.mean(0)))
        # t-stat for first coefficient
        slope_se = float(np.sqrt(np.diag(cov))[0]) if cov.shape[0] >= 1 else float("nan")
        t_stat = float(m.coef_[0] / slope_se) if slope_se > 1e-12 else float("nan")
        slope_p = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - p - 1)))
    else:
        slope_se, t_stat, slope_p = float("nan"), float("nan"), float("nan")
    loocv = compute_loocv_mae(X, y)
    return {
        "label": label,
        "n": int(n),
        "n_features": int(p),
        "intercept": float(m.intercept_),
        "coefficients": [float(c) for c in m.coef_],
        "r2": float(r2),
        "slope_se_first": slope_se,
        "slope_t_first": t_stat,
        "slope_p_first": slope_p,
        "loocv_mae": loocv,
        "ss_res": float(ss_res),
        "ss_tot": float(ss_tot),
    }


def incremental_f_test(full: dict, restricted: dict) -> dict:
    """F-test: does adding berry to vol-only model significantly reduce SS_res?"""
    n = full["n"]
    p_full = full["n_features"]
    p_rest = restricted["n_features"]
    rss_full = full["ss_res"]
    rss_rest = restricted["ss_res"]
    df_diff = p_full - p_rest
    df_resid_full = n - p_full - 1
    if df_diff <= 0 or df_resid_full <= 0 or rss_full <= 1e-12:
        return {"f_stat": float("nan"), "f_p": float("nan")}
    f_stat = ((rss_rest - rss_full) / df_diff) / (rss_full / df_resid_full)
    f_p = float(1 - stats.f.cdf(f_stat, df_diff, df_resid_full))
    return {"f_stat": float(f_stat), "f_p": f_p}


def main():
    print(f"Fetching {SYMBOLS} {START}..{END}")
    raw = fetch_data(SYMBOLS, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    spy_close = prices_df["SPY"].astype(float)
    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features: {feat_arr.shape}")

    print("Fitting geometry and computing Berry curvature time series...")
    geo, scaler, pca = fit_geometry(feat_arr)
    X_pca = pca.transform(scaler.transform(feat_arr))
    berry = berry_curvature_series(geo, X_pca)
    berry_s = pd.Series(berry, index=feat_index, name="berry_fro")
    print(f"  berry: {berry_s.dropna().shape[0]} valid, "
          f"min={berry_s.min():.4e}, mean={berry_s.mean():.4e}, max={berry_s.max():.4e}")

    rv = realized_vol(spy_close).reindex(feat_index)

    # Per-crisis (peak_berry[L], peak_vol[L], eventual_dd)
    crisis_records = []
    spy_full = spy_close.reindex(feat_index).ffill()
    for crisis_id, crisis in ALL_CRISES.items():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        if start not in feat_index and end not in feat_index:
            crisis_mask = (feat_index >= start) & (feat_index <= end)
            if not crisis_mask.any():
                continue
        # Crisis window for drawdown
        crisis_mask = (feat_index >= start) & (feat_index <= end)
        if not crisis_mask.any():
            continue
        dd = peak_drawdown(spy_full[crisis_mask])
        # t0 = first feature_index point on/after start
        t0_idx = feat_index.get_indexer([start], method="bfill")[0]
        if t0_idx < 0:
            continue
        peaks = {}
        for L in LOOKBACK_DAYS:
            lo = max(0, t0_idx - L)
            window_berry = berry_s.iloc[lo:t0_idx]
            window_rv = rv.iloc[lo:t0_idx]
            if window_berry.dropna().empty or window_rv.dropna().empty:
                continue
            peaks[f"peak_berry_L{L}"] = float(window_berry.max())
            peaks[f"peak_vol_L{L}"] = float(window_rv.max())
        if not peaks:
            continue
        crisis_records.append({
            "crisis": crisis_id,
            "label": crisis.get("label", crisis_id),
            "start": str(start.date()),
            "end": str(end.date()),
            "eventual_drawdown": dd,
            **peaks,
        })

    df = pd.DataFrame(crisis_records).set_index("crisis")
    print()
    print(f"Crises with full data: {len(df)}")
    print(df[["eventual_drawdown"] + [f"peak_berry_L60" for _ in [0]] + [f"peak_vol_L60" for _ in [0]]])

    # Regressions for each lookback L
    results_by_L = {}
    for L in LOOKBACK_DAYS:
        b_col = f"peak_berry_L{L}"
        v_col = f"peak_vol_L{L}"
        sub = df[[b_col, v_col, "eventual_drawdown"]].dropna()
        if len(sub) < 4:
            continue
        y = sub["eventual_drawdown"].values

        reg_berry = regression_summary(sub[[b_col]].values, y, label=f"berry_L{L}")
        reg_vol = regression_summary(sub[[v_col]].values, y, label=f"vol_L{L}")
        reg_joint = regression_summary(sub[[b_col, v_col]].values, y, label=f"joint_L{L}")
        f_test = incremental_f_test(reg_joint, reg_vol)

        results_by_L[L] = {
            "n_crises": len(sub),
            "berry": reg_berry,
            "vol": reg_vol,
            "joint": reg_joint,
            "incremental_f_berry_given_vol": f_test,
            "delta_r2_joint_vs_vol": reg_joint["r2"] - reg_vol["r2"],
        }

        print()
        print(f"--- L = {L} days, n = {len(sub)} ---")
        print(f"  Berry-only:  R^2 = {reg_berry['r2']:.3f}  slope p = {reg_berry['slope_p_first']:.3f}  "
              f"LOOCV MAE = {reg_berry['loocv_mae']:.3f}")
        print(f"  Vol-only:    R^2 = {reg_vol['r2']:.3f}  slope p = {reg_vol['slope_p_first']:.3f}  "
              f"LOOCV MAE = {reg_vol['loocv_mae']:.3f}")
        print(f"  Joint:       R^2 = {reg_joint['r2']:.3f}  LOOCV MAE = {reg_joint['loocv_mae']:.3f}")
        print(f"  Berry|vol incremental F = {f_test['f_stat']:.3f}  p = {f_test['f_p']:.3f}  "
              f"ΔR² = {results_by_L[L]['delta_r2_joint_vs_vol']:+.3f}")

    # Pass-criteria check
    pass_criteria = []
    for L, r in results_by_L.items():
        passes = (
            r["berry"]["r2"] > 0.40
            and r["incremental_f_berry_given_vol"]["f_p"] < 0.05
            and r["berry"]["loocv_mae"] < 0.08
        )
        pass_criteria.append({"L": L, "passes": passes})

    print()
    print("=" * 60)
    print("PASS CRITERIA (R^2 berry > 0.40 AND incremental F p < 0.05 AND LOOCV MAE < 0.08):")
    for p in pass_criteria:
        print(f"  L={p['L']:>3}: {'PASS' if p['passes'] else 'fail'}")

    out = {
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "hilbert_dim": HILBERT_DIM,
            "n_pca": N_PCA,
            "seed": SEED,
            "lookback_days": LOOKBACK_DAYS,
            "vol_window": VOL_WINDOW,
            "berry_scalar": "frobenius_norm_of_berry_curvature_tensor",
        },
        "per_crisis": df.reset_index().to_dict(orient="records"),
        "results_by_lookback": {str(k): v for k, v in results_by_L.items()},
        "pass_criteria": pass_criteria,
        "timestamp": datetime.now().isoformat(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
