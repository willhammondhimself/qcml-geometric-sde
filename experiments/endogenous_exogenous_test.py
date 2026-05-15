"""Formal test: does the QCML spectral gap distinguish endogenous from
exogenous crises? Pre-registered classification in `crisis_classification.py`.

Pipeline:
1. Compute spectral-gap time series with CAUSAL (expanding-window) embedding fit.
2. For each crisis: compute min(gap)/trailing_252d_mean (already-defined metric).
3. For each crisis: also compute min(realized_vol)/trailing_252d_mean as control —
   if vol distinguishes endogenous from exogenous as cleanly as gap does, the
   QCML claim isn't unique.
4. Two-group test (endogenous vs exogenous):
   - Welch's t-test (parametric, robust to unequal variances)
   - Mann-Whitney U (rank-based, robust to non-normality and small N)
   - Cohen's d effect size
   - Bootstrap CI on group-mean difference (10,000 resamples, percentile)
5. Vol-baseline test using identical metrics — must NOT find significance.
6. Sensitivity check: drop borderline crises, re-run the test.

Output: experiments/outputs/diagnostics/endo_exo_test.json
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
from experiments.crisis_classification import (
    CRISIS_CLASSIFICATION,
    get_classification_summary,
)
from qcml_geometry.core import QCMLGeometry
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"
SYMBOLS = ["SPY", "DIA"]

HILBERT_DIM = 6
N_PCA = 8
SEED = 42
TRAILING_WINDOW = 252
REFIT_INTERVAL = 252  # refit embedding every N days for causal fit
N_BOOT = 10_000

OUTPUT_PATH = (
    ROOT / "experiments" / "outputs" / "diagnostics" / "endo_exo_test.json"
)


def fit_geometry_at(X: np.ndarray, fit_end: int) -> tuple[QCMLGeometry, StandardScaler, PCA]:
    """Fit on X[:fit_end] only — strictly causal."""
    X_train = X[:fit_end]
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=min(N_PCA, X.shape[1])).fit(scaler.transform(X_train))
    X_pca = pca.transform(scaler.transform(X_train))
    geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=HILBERT_DIM)
    np.random.seed(SEED)
    geo.fit_operators(X_pca, method="random")
    return geo, scaler, pca


def causal_gap_series(feat_arr: np.ndarray) -> np.ndarray:
    """Compute spectral gap at each timestep using expanding-window embedding refit
    every REFIT_INTERVAL days. The embedding at time t depends only on data <= t."""
    T = len(feat_arr)
    gap = np.full(T, np.nan)
    # Initial fit point — need at least TRAILING_WINDOW observations
    fit_points = list(range(TRAILING_WINDOW, T, REFIT_INTERVAL))
    if fit_points[-1] != T:
        fit_points.append(T)

    print(f"  causal refit at {len(fit_points)} points")
    geo, scaler, pca = None, None, None
    next_refit = fit_points[0]
    fit_idx = 0
    for t in range(T):
        if t >= next_refit:
            geo, scaler, pca = fit_geometry_at(feat_arr, t)
            fit_idx += 1
            next_refit = fit_points[fit_idx] if fit_idx < len(fit_points) else T + 1
            if fit_idx % 5 == 0 or fit_idx == 1:
                print(f"    refit at t={t} (idx {fit_idx}/{len(fit_points)})")
        if geo is None:
            continue
        x_pca = pca.transform(scaler.transform(feat_arr[t : t + 1]))[0]
        try:
            gap[t] = geo.spectral_gap(x_pca)
        except Exception:
            gap[t] = np.nan
    return gap


def per_crisis_collapse_ratio(
    series: pd.Series, trailing_window: int
) -> dict[str, dict]:
    """Compute min(series)/trailing-mean inside each crisis window."""
    trailing = series.rolling(trailing_window, min_periods=30).mean()
    pre_mean = series.expanding(min_periods=30).mean()  # fallback if trailing nan
    out = {}
    for crisis_id, crisis in ALL_CRISES.items():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        mask = (series.index >= start) & (series.index <= end)
        if not mask.any():
            continue
        vals = series[mask]
        if vals.dropna().empty:
            continue
        baseline = trailing.loc[vals.index[0]]
        if pd.isna(baseline) or baseline < 1e-12:
            baseline = pre_mean.loc[vals.index[0]] if vals.index[0] in pre_mean.index else np.nan
        if pd.isna(baseline) or baseline < 1e-12:
            continue
        min_val = vals.min()
        out[crisis_id] = {
            "min": float(min_val),
            "baseline": float(baseline),
            "ratio": float(min_val / baseline),
        }
    return out


def two_group_test(endo: list[float], exo: list[float]) -> dict:
    """Welch t-test, Mann-Whitney U, Cohen's d, bootstrap CI on mean diff."""
    e = np.array(endo)
    x = np.array(exo)

    t_stat, t_p = stats.ttest_ind(e, x, equal_var=False, alternative="less")
    u_stat, u_p = stats.mannwhitneyu(e, x, alternative="less")
    pooled = np.sqrt(
        ((len(e) - 1) * np.var(e, ddof=1) + (len(x) - 1) * np.var(x, ddof=1))
        / (len(e) + len(x) - 2)
    )
    cohen_d = (np.mean(e) - np.mean(x)) / pooled if pooled > 1e-12 else float("nan")

    # Bootstrap CI on (mean_exo - mean_endo)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs[i] = np.mean(rng.choice(x, size=len(x), replace=True)) - np.mean(
            rng.choice(e, size=len(e), replace=True)
        )
    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])

    return {
        "n_endogenous": len(e),
        "n_exogenous": len(x),
        "endo_mean": float(np.mean(e)),
        "endo_std": float(np.std(e, ddof=1)) if len(e) > 1 else None,
        "exo_mean": float(np.mean(x)),
        "exo_std": float(np.std(x, ddof=1)) if len(x) > 1 else None,
        "mean_diff_exo_minus_endo": float(np.mean(x) - np.mean(e)),
        "welch_t": float(t_stat),
        "welch_p_one_sided_endo_lt_exo": float(t_p),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p_one_sided": float(u_p),
        "cohens_d_endo_vs_exo": float(cohen_d),
        "bootstrap_ci95_diff": [float(ci_lo), float(ci_hi)],
    }


def realized_vol_series(spy_close: pd.Series, vol_window: int = 20) -> pd.Series:
    log_ret = np.log(spy_close / spy_close.shift(1))
    return (log_ret.rolling(vol_window).std() * np.sqrt(252))


def main() -> None:
    print(f"Fetching {SYMBOLS} {START}..{END}")
    raw = fetch_data(SYMBOLS, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features: {feat_arr.shape}")

    print("Computing causal-fit gap series (this is slow)...")
    gap_arr = causal_gap_series(feat_arr)
    gap = pd.Series(gap_arr, index=feat_index, name="spectral_gap")
    print(f"  gap series: {gap.dropna().shape[0]} valid obs, "
          f"min={gap.dropna().min():.4e}, max={gap.dropna().max():.4e}")

    print("Computing realized-vol baseline series...")
    spy_close = prices_df["SPY"].astype(float)
    rv = realized_vol_series(spy_close).reindex(feat_index)
    print(f"  rv series: {rv.dropna().shape[0]} valid obs")

    print("Per-crisis collapse ratios...")
    gap_ratios = per_crisis_collapse_ratio(gap, TRAILING_WINDOW)
    # Vol "collapse" is actually a SPIKE — invert: use max(rv)/baseline so the
    # "lower means more reaction" semantics reverses cleanly. We'll test both
    # max(rv)/baseline and min(rv)/baseline; vol direction is the OPPOSITE of gap.
    rv_ratios_min = per_crisis_collapse_ratio(rv, TRAILING_WINDOW)
    rv_max_ratios = {}
    for crisis_id, crisis in ALL_CRISES.items():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        mask = (rv.index >= start) & (rv.index <= end)
        if not mask.any() or rv[mask].dropna().empty:
            continue
        baseline = rv.rolling(TRAILING_WINDOW, min_periods=30).mean().loc[rv[mask].index[0]]
        if pd.isna(baseline) or baseline < 1e-12:
            continue
        rv_max_ratios[crisis_id] = {
            "max": float(rv[mask].max()),
            "baseline": float(baseline),
            "ratio": float(rv[mask].max() / baseline),
        }

    # Group ratios by classification
    endo_gap, exo_gap = [], []
    endo_rv, exo_rv = [], []
    rows = []
    for crisis_id, entry in CRISIS_CLASSIFICATION.items():
        cls = entry["class"]
        gap_r = gap_ratios.get(crisis_id, {}).get("ratio")
        rv_r = rv_max_ratios.get(crisis_id, {}).get("ratio")
        if gap_r is None:
            continue
        if cls == "endogenous":
            endo_gap.append(gap_r)
            if rv_r is not None:
                endo_rv.append(rv_r)
        else:
            exo_gap.append(gap_r)
            if rv_r is not None:
                exo_rv.append(rv_r)
        rows.append({
            "crisis": crisis_id,
            "class": cls,
            "borderline": entry["borderline"],
            "gap_ratio": gap_r,
            "rv_max_ratio": rv_r,
        })

    print()
    print("Per-crisis (sorted by class):")
    print(f"{'Crisis':<22}{'Class':<11}{'BL':<5}{'Gap ratio':>10}{'Vol max ratio':>15}")
    rows_sorted = sorted(rows, key=lambda r: (r["class"], r["gap_ratio"]))
    for r in rows_sorted:
        print(f"{r['crisis']:<22}{r['class']:<11}{('Y' if r['borderline'] else ''):<5}"
              f"{r['gap_ratio']:>10.3f}"
              f"{(r['rv_max_ratio'] if r['rv_max_ratio'] is not None else float('nan')):>15.3f}")

    # Primary test: gap distinguishes endogenous from exogenous
    print()
    print("=" * 60)
    print("PRIMARY TEST: gap_ratio (endogenous < exogenous)")
    print("=" * 60)
    primary = two_group_test(endo_gap, exo_gap)
    for k, v in primary.items():
        print(f"  {k}: {v}")

    # Vol baseline: does vol also distinguish them?
    print()
    print("=" * 60)
    print("VOL BASELINE: rv_max_ratio (does vol distinguish endo vs exo?)")
    print("=" * 60)
    vol_test = two_group_test(endo_rv, exo_rv)  # same direction (lower vol response = endogenous?)
    for k, v in vol_test.items():
        print(f"  {k}: {v}")

    # Sensitivity: drop borderline crises
    print()
    print("=" * 60)
    print("SENSITIVITY: drop borderline crises")
    print("=" * 60)
    endo_strict = [r["gap_ratio"] for r in rows if r["class"] == "endogenous" and not r["borderline"]]
    exo_strict = [r["gap_ratio"] for r in rows if r["class"] == "exogenous" and not r["borderline"]]
    print(f"  n_endo_strict={len(endo_strict)}, n_exo_strict={len(exo_strict)}")
    if len(endo_strict) >= 2 and len(exo_strict) >= 2:
        strict = two_group_test(endo_strict, exo_strict)
        for k, v in strict.items():
            print(f"  {k}: {v}")
    else:
        strict = None
        print("  insufficient strict-classification crises for test")

    results = {
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "hilbert_dim": HILBERT_DIM,
            "n_pca": N_PCA,
            "seed": SEED,
            "trailing_window_days": TRAILING_WINDOW,
            "refit_interval_days": REFIT_INTERVAL,
            "n_bootstrap": N_BOOT,
            "classification_summary": get_classification_summary(),
        },
        "per_crisis": rows_sorted,
        "primary_test_gap_ratio_endo_lt_exo": primary,
        "vol_baseline_test": vol_test,
        "sensitivity_strict_classification": strict,
        "timestamp": datetime.now().isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
