"""Multi-asset extension of Angle 5: curvature → drawdown severity.

Repeats the per-crisis (peak Berry curvature, peak vol) → eventual drawdown
regression on each of 5 equity ETFs (SPY, QQQ, IWM, EFA, DIA). For each
asset, the QCML embedding is built from the asset paired with SPY (so DIA
gets paired with SPY, QQQ with SPY, etc.; SPY itself gets paired with DIA).

For each (asset, crisis) row in the pooled dataset (~75 rows), we collect:
  - peak_berry[L]   = max Berry-curvature Frobenius norm in [t0-L, t0]
  - peak_vol[L]     = max 20-d realized vol of asset price in [t0-L, t0]
  - eventual_dd     = peak-to-trough drawdown of asset price in [t0, t1]

Then for each L in {20, 40, 60, 120, 252}:
  - dd ~ peak_berry alone
  - dd ~ peak_vol alone
  - dd ~ peak_berry + peak_vol (joint)
  - incremental F-test: does Berry add to vol-only?
  - LOOCV MAE

Critical question: does the L=120 signal (R²=0.37 in single-asset, F p=0.041)
survive at N≈75 with similar magnitude? If yes, the headline strengthens. If
it collapses, the original was sample-size artifact.

Output: experiments/outputs/curvature_drawdown/curvature_drawdown_multi_asset.json
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
ASSETS = ["SPY", "QQQ", "IWM", "EFA", "DIA"]
HILBERT_DIM = 6
N_PCA = 8
SEED = 0
LOOKBACK_DAYS = [20, 40, 60, 120, 252]
VOL_WINDOW = 20

OUTPUT_PATH = (
    ROOT / "experiments" / "outputs" / "curvature_drawdown" / "curvature_drawdown_multi_asset.json"
)


def fit_geometry(X: np.ndarray):
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=min(N_PCA, X.shape[1])).fit(scaler.transform(X))
    X_pca = pca.transform(scaler.transform(X))
    geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=HILBERT_DIM)
    np.random.seed(SEED)
    geo.fit_operators(X_pca, method="random")
    return geo, scaler, pca


def berry_curvature_series(geo: QCMLGeometry, X_pca: np.ndarray) -> np.ndarray:
    T = len(X_pca)
    out = np.full(T, np.nan)
    for t in range(T):
        try:
            F = geo.berry_curvature(X_pca[t], epsilon=1e-5)
            out[t] = np.linalg.norm(F, ord="fro")
        except Exception:
            out[t] = np.nan
    return out


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return (log_ret.rolling(window).std() * np.sqrt(252))


def peak_drawdown(price: pd.Series) -> float:
    if price.empty:
        return float("nan")
    cm = price.cummax()
    dd = (price - cm) / cm
    return float(dd.min())


def run_for_asset(symbol: str) -> dict:
    """Build QCML embedding for `symbol` paired with SPY (or DIA if symbol is SPY),
    compute Berry curvature time series, then per-crisis (peak_berry, peak_vol, dd).
    Returns dict of crisis_id → record. Skips crises with insufficient data."""
    pair_anchor = "DIA" if symbol == "SPY" else "SPY"
    pair = [symbol, pair_anchor]
    print(f"  fetching {pair} ...")
    raw = fetch_data(pair, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    if symbol not in prices_df.columns:
        print(f"  WARNING: {symbol} not found in fetched data")
        return {}
    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features: {feat_arr.shape}")

    print(f"  fitting geometry...")
    geo, scaler, pca = fit_geometry(feat_arr)
    X_pca = pca.transform(scaler.transform(feat_arr))

    print(f"  computing Berry curvature ...")
    berry = berry_curvature_series(geo, X_pca)
    berry_s = pd.Series(berry, index=feat_index, name="berry_fro")
    print(f"    berry: {berry_s.dropna().shape[0]} valid")

    asset_close = prices_df[symbol].astype(float).reindex(feat_index).ffill()
    rv = realized_vol(asset_close).reindex(feat_index)

    records = {}
    for crisis_id, crisis in ALL_CRISES.items():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        crisis_mask = (feat_index >= start) & (feat_index <= end)
        if not crisis_mask.any():
            continue
        dd = peak_drawdown(asset_close[crisis_mask])
        if pd.isna(dd):
            continue
        t0_idx = feat_index.get_indexer([start], method="bfill")[0]
        if t0_idx < 0:
            continue
        rec = {
            "asset": symbol,
            "crisis": crisis_id,
            "label": crisis.get("label", crisis_id),
            "eventual_drawdown": dd,
        }
        ok = True
        for L in LOOKBACK_DAYS:
            lo = max(0, t0_idx - L)
            wb = berry_s.iloc[lo:t0_idx].dropna()
            wv = rv.iloc[lo:t0_idx].dropna()
            if wb.empty or wv.empty:
                ok = False
                break
            rec[f"peak_berry_L{L}"] = float(wb.max())
            rec[f"peak_vol_L{L}"] = float(wv.max())
        if ok:
            records[crisis_id] = rec

    print(f"  → {len(records)} (asset, crisis) records")
    return records


def compute_loocv_mae(X: np.ndarray, y: np.ndarray) -> float:
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
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    n, p = X.shape
    if n - p - 1 > 0:
        sigma2 = ss_res / (n - p - 1)
        try:
            xtx_inv = np.linalg.pinv(X.T @ X - n * np.outer(X.mean(0), X.mean(0)))
        except Exception:
            xtx_inv = np.eye(p) * float("nan")
        slope_se = float(np.sqrt(np.diag(sigma2 * xtx_inv))[0]) if xtx_inv.shape[0] >= 1 else float("nan")
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
        "slope_p_first": slope_p,
        "loocv_mae": loocv,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


def incremental_f_test(full: dict, restricted: dict) -> dict:
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
    all_records = []
    for sym in ASSETS:
        print(f"\n=== {sym} ===")
        recs = run_for_asset(sym)
        all_records.extend(list(recs.values()))

    df = pd.DataFrame(all_records)
    print(f"\nPooled dataset: {len(df)} (asset, crisis) rows")
    print(df.groupby("asset").size())

    # Regressions for each lookback L on pooled dataset
    results_by_L = {}
    for L in LOOKBACK_DAYS:
        b_col = f"peak_berry_L{L}"
        v_col = f"peak_vol_L{L}"
        sub = df[[b_col, v_col, "eventual_drawdown"]].dropna()
        if len(sub) < 5:
            continue
        y = sub["eventual_drawdown"].values

        reg_berry = regression_summary(sub[[b_col]].values, y, label=f"berry_L{L}")
        reg_vol = regression_summary(sub[[v_col]].values, y, label=f"vol_L{L}")
        reg_joint = regression_summary(sub[[b_col, v_col]].values, y, label=f"joint_L{L}")
        f_test = incremental_f_test(reg_joint, reg_vol)

        results_by_L[L] = {
            "n_rows": len(sub),
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

    pass_criteria = []
    for L, r in results_by_L.items():
        passes = (
            r["joint"]["r2"] > 0.40
            and r["incremental_f_berry_given_vol"]["f_p"] < 0.05
        )
        pass_criteria.append({"L": L, "passes": passes,
                              "joint_r2": r["joint"]["r2"],
                              "incremental_f_p": r["incremental_f_berry_given_vol"]["f_p"]})

    print()
    print("=" * 60)
    print("PASS CRITERIA (joint R^2 > 0.40 AND incremental F p < 0.05):")
    for p in pass_criteria:
        marker = "PASS" if p["passes"] else "fail"
        print(f"  L={p['L']:>3}: {marker}   joint R²={p['joint_r2']:.3f}, F p={p['incremental_f_p']:.3f}")
    print("=" * 60)

    out = {
        "config": {
            "assets": ASSETS,
            "start": START,
            "end": END,
            "hilbert_dim": HILBERT_DIM,
            "n_pca": N_PCA,
            "seed": SEED,
            "lookback_days": LOOKBACK_DAYS,
            "vol_window": VOL_WINDOW,
        },
        "n_records": int(len(df)),
        "per_record": all_records,
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
