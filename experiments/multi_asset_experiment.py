"""Multi-asset prediction horse race: does correlation-manifold geometry add OOS
predictive power for forward systemic risk beyond the Absorption Ratio?

For a pre-registered slice (experiments/multi_asset_data.SLICES), build:
  * the N-asset returns matrix → Absorption-Ratio score + panel-vol (HAR) baseline,
  * the correlation-manifold features → causal geometry-detector scores,
  * forward systemic-risk target (default: forward equal-weight max drawdown),
then run expanding-window OOS R² for feature sets {vol, AR, AR+geo, geo, all} with
both Ridge and gradient boosting, and block-bootstrap the ΔR² of "AR+geo" over "AR".
"""

from __future__ import annotations

import argparse
import json
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

from experiments.baselines import AbsorptionRatioDetector  # noqa: E402
from experiments.multi_asset_data import (  # noqa: E402
    SLICES,
    correlation_features,
    fetch_panel,
    returns_matrix,
    slice_diversity,
)
from experiments.systemic_risk_targets import TARGETS  # noqa: E402
from experiments.volatility_forecasting import oos_r2, walk_forward_predict  # noqa: E402
from qcml_geometry import (  # noqa: E402
    BerryPhaseRateDetector,
    DimensionalityCollapseDetector,
    QFIDeterminantDetector,
    ReducedPurityDetector,
    SectionalCurvatureDetector,
)

HORIZON = 20

# Geometry detectors fed the correlation manifold (cross-asset-sensitive).
GEO = {
    "reduced_purity": (
        ReducedPurityDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=8,
            operator_method="random",
            seed=42,
            normalization="soft",
            adaptive_epsilon=True,
            partition=(2, 4),
        ),
    ),
    "dim_collapse": (
        DimensionalityCollapseDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=8,
            operator_method="random",
            seed=42,
            normalization="soft",
            adaptive_epsilon=True,
            subsample=5,
        ),
    ),
    "sectional_curv": (
        SectionalCurvatureDetector,
        dict(
            hilbert_dim=6,
            n_pca_components=3,
            operator_method="pca_inspired",
            seed=42,
            normalization="soft",
            adaptive_epsilon=True,
            score_mode="neg_fraction",
            subsample=10,
        ),
    ),
    "qfi_det": (
        QFIDeterminantDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=12,
            operator_method="pca_inspired",
            seed=42,
            normalization="soft",
            qfi_mode="logdet",
            adaptive_epsilon=True,
        ),
    ),
    "berry": (
        BerryPhaseRateDetector,
        dict(
            hilbert_dim=6,
            n_pca_components=8,
            operator_method="random",
            seed=42,
            normalization="sphere",
            berry_aggregation="f01",
        ),
    ),
}


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


def build_slice_frame(slice_name, horizon=HORIZON, fit_frac=0.30, corr_window=20):
    members = SLICES[slice_name]
    prices = fetch_panel(members)
    R, rdates = returns_matrix(prices)
    C, cdates = correlation_features(prices, window=corr_window)
    div = slice_diversity(prices)

    # all three forward targets + baselines on the returns-date index
    ew = pd.Series(R.mean(axis=1), index=rdates)
    cols = {
        f"target_{name}": pd.Series(fn(R, horizon), index=rdates) for name, fn in TARGETS.items()
    }
    cols.update(
        {
            "vol5": np.log(ew.rolling(5).std() + 1e-6),
            "vol22": np.log(ew.rolling(22).std() + 1e-6),
            "vol66": np.log(ew.rolling(66).std() + 1e-6),
            "ar": pd.Series(
                AbsorptionRatioDetector(rolling_window=252, n_components=2).compute_regime_scores(
                    R
                ),
                index=rdates,
            ),
        }
    )
    base = pd.DataFrame(cols)

    # causal geometry-detector scores on the correlation manifold
    fit_end = int(fit_frac * len(C))
    geo = {}
    for name, (cls, params) in GEO.items():
        try:
            det = cls(causal_fit_length=fit_end, **params)
            det.fit(C)
            geo[name] = pd.Series(
                np.asarray(det.compute_regime_scores(C), dtype=float), index=cdates
            )
        except Exception as exc:  # a config invalid for this slice's dimensionality
            print(f"  [{slice_name}] geo {name} failed: {type(exc).__name__}: {exc}")
    geo_df = pd.DataFrame(geo)

    full = base.join(geo_df, how="inner").dropna()
    return full, div, list(geo.keys())


def run_slice(slice_name):
    full, div, geo_names = build_slice_frame(slice_name)
    feat = {
        "vol": ["vol5", "vol22", "vol66"],
        "ar": ["ar"],
        "ar+geo": ["ar"] + geo_names,
        "geo": geo_names,
        "all": ["vol5", "vol22", "vol66", "ar"] + geo_names,
    }
    first = int(0.45 * len(full))
    models = {
        "ridge": lambda: Ridge(alpha=1.0),
        "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05),
    }
    targets = [c for c in full.columns if c.startswith("target_")]

    out = {
        "slice": slice_name,
        "diversity": div,
        "n_rows": int(len(full)),
        "geo_detectors": geo_names,
        "targets": {},
    }
    print(f"\n=== {slice_name}  (diversity={div:.3f}, rows={len(full)}, geo={geo_names}) ===")
    for tname in targets:
        y = full[tname].values
        out["targets"][tname] = {}
        print(f"  target={tname}")
        for mname, mfn in models.items():
            preds = {
                k: walk_forward_predict(full[c].values, y, mfn, first) for k, c in feat.items()
            }
            r2 = {k: oos_r2(y[np.isfinite(p)], p[np.isfinite(p)]) for k, p in preds.items()}
            dr2_ar, p_ar = delta_r2_pvalue(y, preds["ar"], preds["ar+geo"])  # geo adds over AR?
            dr2_vol, p_vol = delta_r2_pvalue(y, preds["vol"], preds["all"])  # stack beats vol?
            out["targets"][tname][mname] = {
                **{f"r2_{k}": float(v) for k, v in r2.items()},
                "delta_r2_geo_over_ar": dr2_ar,
                "delta_r2_geo_over_ar_p": p_ar,
                "delta_r2_all_over_vol": dr2_vol,
                "delta_r2_all_over_vol_p": p_vol,
            }
            dr2, p = dr2_ar, p_ar
            print(
                f"    {mname:5s} R2: vol={r2['vol']:+.3f} ar={r2['ar']:+.3f} "
                f"ar+geo={r2['ar+geo']:+.3f} geo={r2['geo']:+.3f} all={r2['all']:+.3f}  "
                f"| Δ(geo|ar)={dr2:+.3f} (p={p:.3f})"
            )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", nargs="+", default=["sectors+macro"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = [run_slice(s) for s in args.slices]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
