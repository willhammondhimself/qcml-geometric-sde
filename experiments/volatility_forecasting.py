"""Fair test: does market geometry help forecast volatility out-of-sample?

Reframes from "detect crises" to the legitimate, easier target of predicting
forward realized volatility. We run an honest expanding-window horse race:

    baseline  = HAR-RV features (current short/medium/long realized vol)
    +geometry = HAR-RV + the full set of (causal) geometric observable scores

with both a linear (Ridge) and a nonlinear (gradient boosting) model. If geometry
carries volatility information beyond vol's own persistence — especially nonlinear
or multivariate — then +geometry beats baseline in out-of-sample R². If not, it
doesn't, and that's the honest answer.

Everything is causal: geometry operators are fit only on an early prefix and emit
expanding-z (causal) scores; the forecaster is retrained on expanding windows and
only ever predicts the future.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import experiments.walk_forward_hpo as wf  # noqa: E402
from experiments.data_loader import fetch_data  # noqa: E402
from qcml_geometry import (  # noqa: E402
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
    ReducedPurityDetector,
    SpectralEntropyDetector,
    SpectralGapDetector,
)

HORIZON = 20  # forecast 20-trading-day forward realized vol

GEO = {
    "berry": (BerryPhaseRateDetector, dict(hilbert_dim=6, n_pca_components=8, rolling_window=15,
              operator_method="random", seed=42, normalization="sphere", berry_aggregation="f01")),
    "mlf": (MultiLagFidelityDetector, dict(hilbert_dim=4, n_pca_components=8, rolling_window=20,
            operator_method="pca_inspired", seed=42, normalization="sphere")),
    "sgap": (SpectralGapDetector, dict(hilbert_dim=8, n_pca_components=12, rolling_window=20,
             operator_method="random", seed=42, normalization="soft", adaptive_epsilon=True)),
    "sent": (SpectralEntropyDetector, dict(hilbert_dim=8, n_pca_components=8, rolling_window=20,
             operator_method="random", seed=42, normalization="soft", adaptive_epsilon=True)),
    "purity": (ReducedPurityDetector, dict(hilbert_dim=8, n_pca_components=8, rolling_window=20,
               operator_method="random", seed=42, normalization="soft", adaptive_epsilon=True)),
    "qfi": (QFIDeterminantDetector, dict(hilbert_dim=8, n_pca_components=12, rolling_window=20,
            operator_method="pca_inspired", seed=42, normalization="soft", qfi_mode="logdet",
            adaptive_epsilon=True)),
}


def oos_r2(y, yhat):
    sse = np.nansum((y - yhat) ** 2)
    sst = np.nansum((y - np.nanmean(y)) ** 2)
    return 1.0 - sse / sst if sst > 0 else np.nan


def delta_r2_pvalue(y, yhat_a, yhat_b, n_boot=2000, block=63, seed=0):
    """ΔR² of model B over A + one-sided block-bootstrap p (B no better than A)."""
    m = np.isfinite(y) & np.isfinite(yhat_a) & np.isfinite(yhat_b)
    y, ya, yb = y[m], yhat_a[m], yhat_b[m]
    sst = np.sum((y - np.mean(y)) ** 2)
    if sst <= 0 or len(y) < 100:
        return np.nan, np.nan
    d = (y - ya) ** 2 - (y - yb) ** 2  # >0 where B better
    dr2 = float(np.sum(d) / sst)
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, nb)])[
            :n
        ]
        boots[b] = np.sum(d[idx]) / sst
    p = float((np.sum(boots <= 0) + 1) / (n_boot + 1))
    return dr2, p


def walk_forward_predict(Xf, y, model_fn, first, step=63, purge=HORIZON):
    """Expanding-window: train on [:i-purge], predict [i:i+step]. Returns OOS yhat aligned to y.

    ``purge`` drops the last ``purge`` training rows before each test block:
    y[t] spans returns in (t, t+HORIZON], so unpurged labels at t in
    [i-HORIZON, i) overlap the test window — a lookahead leak (López de Prado
    2018, Ch. 7). Purging deflates absolute OOS R² slightly; comparisons
    between feature sets are unaffected since all see the same windows.
    """
    T = len(y)
    yhat = np.full(T, np.nan)
    i = first
    while i < T:
        tr = slice(0, max(0, i - purge))
        te = slice(i, min(i + step, T))
        mtr = np.isfinite(y[tr]) & np.all(np.isfinite(Xf[tr]), axis=1)
        if mtr.sum() < 250:
            i += step
            continue
        sc = StandardScaler().fit(Xf[tr][mtr])
        model = model_fn()
        model.fit(sc.transform(Xf[tr][mtr]), y[tr][mtr])
        Xte = Xf[te]
        ok = np.all(np.isfinite(Xte), axis=1)
        if ok.any():
            pred = np.full(Xte.shape[0], np.nan)
            pred[ok] = model.predict(sc.transform(Xte[ok]))
            yhat[te] = pred
        i += step
    return yhat


def main():
    print("Preparing data + geometry features (causal)...")
    Xe, dates = wf.prepare_data()
    raw = fetch_data(["SPY", "DIA"], "1995-01-01", "2024-12-31")
    close = raw["close"].unstack("symbol").dropna()
    spy = close["SPY"].reindex(pd.to_datetime([str(d.date()) for d in dates]))
    ret = np.concatenate([[np.nan], np.diff(np.log(spy.values.astype(float)))])
    T = len(dates)

    rv = pd.Series(ret)
    # target: log forward realized vol
    fwd = np.full(T, np.nan)
    for t in range(T - HORIZON):
        fr = ret[t + 1 : t + 1 + HORIZON]
        fr = fr[np.isfinite(fr)]
        if len(fr) >= HORIZON // 2:
            fwd[t] = np.log(np.std(fr) + 1e-6)

    # HAR-RV baseline features: current realized vol at short/medium/long lags (log)
    har = np.column_stack([
        np.log(rv.rolling(5).std().values + 1e-6),
        np.log(rv.rolling(22).std().values + 1e-6),
        np.log(rv.rolling(66).std().values + 1e-6),
    ])

    # geometry features: causal detector scores
    fit_end = int(0.30 * T)
    geo_cols, geo_names = [], []
    for name, (cls, p) in GEO.items():
        try:
            det = cls(causal_fit_length=fit_end, **p)
            det.fit(Xe)
            geo_cols.append(np.asarray(det.compute_regime_scores(Xe), dtype=float))
            geo_names.append(name)
        except Exception as exc:
            print(f"  geo {name} failed: {type(exc).__name__}: {exc}")
    geo = np.column_stack(geo_cols)
    print(f"  geometry features: {geo_names}")

    first = max(fit_end + 250, 1000)
    y = fwd

    models = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "GBM": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05),
    }
    print(f"\nOut-of-sample forward-{HORIZON}d log-vol R² (expanding window, refit every 63d, "
          f"purge={HORIZON}d):")
    print(f"{'model':6s} {'HAR baseline':>14s} {'HAR + geometry':>16s} {'geometry only':>15s} "
          f"{'Δ(geo adds)':>12s} {'p(Δ≤0)':>8s}")
    for mname, mfn in models.items():
        yh_base = walk_forward_predict(har, y, mfn, first)
        yh_both = walk_forward_predict(np.column_stack([har, geo]), y, mfn, first)
        yh_geo = walk_forward_predict(geo, y, mfn, first)
        m = np.isfinite(yh_base) & np.isfinite(yh_both) & np.isfinite(yh_geo) & np.isfinite(y)
        r2_base = oos_r2(y[m], yh_base[m])
        r2_both = oos_r2(y[m], yh_both[m])
        r2_geo = oos_r2(y[m], yh_geo[m])
        dr2, p = delta_r2_pvalue(y, yh_base, yh_both)
        print(f"{mname:6s} {r2_base:14.3f} {r2_both:16.3f} {r2_geo:15.3f} {dr2:+12.3f} {p:8.3f}")


if __name__ == "__main__":
    main()
