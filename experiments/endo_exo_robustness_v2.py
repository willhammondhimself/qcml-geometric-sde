"""Robustness sweep v2 for the endogenous-vs-exogenous gap-collapse finding.

V1 was misconfigured: setting np.random.seed() does NOT vary the operator basis
because qcml_geometry.core.fit_operators(method='random') seeds each operator k
with rng(k) regardless of any global seed (see core.py:231). V1's "5 seeds"
were 5 copies of the same experiment.

V2 varies the operator basis the same way `experiments/seed_sensitivity.py` does:
build custom Hermitian operators with k seeded by (k + offset). For each (offset,
hilbert_dim), bypass fit_operators and assign geo.operators directly.

Sweep:
  - seed_offset in {0, 100, 200, 300, 400}   -> 5 truly distinct bases
  - hilbert_dim in {4, 6, 8}                 -> 3 dimensions
  -> 15 configs

The v1 result that hd=6 + canonical operators gives d=-1.14 is THE finding to
stress-test. This sweep asks: under hd=6, does varying the operator basis
preserve the endo/exo distinction? And does the effect appear at hd=4 / hd=8
under any basis?

Output: experiments/outputs/diagnostics/endo_exo_robustness_v2.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from itertools import product
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
from experiments.crisis_classification import CRISIS_CLASSIFICATION
from qcml_geometry.core import QCMLGeometry
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"
SYMBOLS = ["SPY", "DIA"]

SEED_OFFSETS = [0, 100, 200, 300, 400]
HILBERT_DIMS = [4, 6, 8]
N_PCA = 8
TRAILING_WINDOW = 252
REFIT_INTERVAL = 252
N_BOOT = 5_000

OUTPUT_PATH = (
    ROOT / "experiments" / "outputs" / "diagnostics" / "endo_exo_robustness_v2.json"
)


def make_hermitian(rng: np.random.Generator, hd: int) -> np.ndarray:
    A = rng.standard_normal((hd, hd)) + 1j * rng.standard_normal((hd, hd))
    return (A + A.conj().T) / 2.0


def make_basis(offset: int, n_ops: int, hd: int) -> list[np.ndarray]:
    """Operator k seeded by (k + offset), distinct from canonical seeding when offset > 0."""
    return [make_hermitian(np.random.default_rng(k + offset), hd) for k in range(n_ops)]


def fit_geometry_at(X: np.ndarray, fit_end: int, hilbert_dim: int, offset: int):
    X_train = X[:fit_end]
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=min(N_PCA, X.shape[1])).fit(scaler.transform(X_train))
    X_pca = pca.transform(scaler.transform(X_train))
    geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
    # Bypass fit_operators; assign custom operators built with offset-shifted seeds
    n_ops = X_pca.shape[1]
    geo.operators = make_basis(offset, n_ops, hilbert_dim)
    geo.is_fitted = True  # required by spectral_gap guard
    return geo, scaler, pca


def causal_gap_series(feat_arr: np.ndarray, hilbert_dim: int, offset: int) -> np.ndarray:
    T = len(feat_arr)
    gap = np.full(T, np.nan)
    fit_points = list(range(TRAILING_WINDOW, T, REFIT_INTERVAL))
    if fit_points[-1] != T:
        fit_points.append(T)

    geo, scaler, pca = None, None, None
    next_refit = fit_points[0]
    fit_idx = 0
    for t in range(T):
        if t >= next_refit:
            geo, scaler, pca = fit_geometry_at(feat_arr, t, hilbert_dim, offset)
            fit_idx += 1
            next_refit = fit_points[fit_idx] if fit_idx < len(fit_points) else T + 1
        if geo is None:
            continue
        x_pca = pca.transform(scaler.transform(feat_arr[t : t + 1]))[0]
        try:
            gap[t] = geo.spectral_gap(x_pca)
        except Exception:
            gap[t] = np.nan
    return gap


def per_crisis_ratios(series: pd.Series) -> dict:
    trailing = series.rolling(TRAILING_WINDOW, min_periods=30).mean()
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
            continue
        out[crisis_id] = float(vals.min() / baseline)
    return out


def two_group_test(endo, exo, n_boot=N_BOOT, seed=0):
    e, x = np.array(endo), np.array(exo)
    if len(e) < 2 or len(x) < 2:
        return None
    t_stat, t_p = stats.ttest_ind(e, x, equal_var=False, alternative="less")
    u_stat, u_p = stats.mannwhitneyu(e, x, alternative="less")
    pooled = np.sqrt(
        ((len(e) - 1) * np.var(e, ddof=1) + (len(x) - 1) * np.var(x, ddof=1))
        / (len(e) + len(x) - 2)
    )
    cohen_d = (np.mean(e) - np.mean(x)) / pooled if pooled > 1e-12 else float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = np.mean(rng.choice(x, size=len(x), replace=True)) - np.mean(
            rng.choice(e, size=len(e), replace=True)
        )
    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "n_endo": len(e),
        "n_exo": len(x),
        "endo_mean": float(np.mean(e)),
        "exo_mean": float(np.mean(x)),
        "welch_p": float(t_p),
        "mannwhitney_p": float(u_p),
        "cohens_d": float(cohen_d),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "ci_excludes_zero": bool(ci_lo > 0),
    }


def main():
    print(f"Fetching {SYMBOLS} {START}..{END}")
    raw = fetch_data(SYMBOLS, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features: {feat_arr.shape}")

    configs = list(product(SEED_OFFSETS, HILBERT_DIMS))
    print(f"\n{len(configs)} configurations to run\n")

    results = []
    t_start = time.time()
    for i, (offset, hd) in enumerate(configs, 1):
        cfg_id = f"offset={offset} hd={hd}"
        elapsed = time.time() - t_start
        eta = (elapsed / max(i - 1, 1)) * (len(configs) - i + 1) if i > 1 else None
        eta_str = f", ETA {eta/60:.1f} min" if eta else ""
        print(f"[{i}/{len(configs)}] {cfg_id}  (elapsed {elapsed/60:.1f} min{eta_str})")

        try:
            gap_arr = causal_gap_series(feat_arr, hd, offset)
            gap = pd.Series(gap_arr, index=feat_index)
            ratios = per_crisis_ratios(gap)

            endo_vals, exo_vals = [], []
            for cid, entry in CRISIS_CLASSIFICATION.items():
                if cid not in ratios:
                    continue
                if entry["class"] == "endogenous":
                    endo_vals.append(ratios[cid])
                else:
                    exo_vals.append(ratios[cid])

            test = two_group_test(endo_vals, exo_vals, seed=offset or 1)
            if test is None:
                print(f"  insufficient data, skipping")
                continue
            print(f"  d={test['cohens_d']:+.2f}  Welch p={test['welch_p']:.3f}  "
                  f"M-W p={test['mannwhitney_p']:.3f}  "
                  f"CI=[{test['bootstrap_ci_lo']:+.3f}, {test['bootstrap_ci_hi']:+.3f}]")

            results.append({
                "config": {
                    "seed_offset": offset,
                    "hilbert_dim": hd,
                },
                "test": test,
                "ratios": ratios,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    welch_ps = [r["test"]["welch_p"] for r in results]
    ds = [r["test"]["cohens_d"] for r in results]
    ci_excludes = [r["test"]["ci_excludes_zero"] for r in results]
    n_total = len(results)
    n_welch_sig = sum(1 for p in welch_ps if p < 0.05)
    n_ci_excl = sum(ci_excludes)
    n_d_large = sum(1 for d in ds if d <= -0.8)

    pct_welch = 100 * n_welch_sig / n_total if n_total else 0
    pct_ci = 100 * n_ci_excl / n_total if n_total else 0
    median_d = float(np.median(ds)) if ds else None

    if pct_welch >= 70 and median_d is not None and median_d <= -0.8:
        verdict = "GREEN"
    elif pct_welch >= 40 or (median_d is not None and median_d <= -0.5):
        verdict = "YELLOW"
    else:
        verdict = "RED"

    summary = {
        "n_configurations": n_total,
        "n_welch_significant_p005": n_welch_sig,
        "pct_welch_significant": pct_welch,
        "n_bootstrap_ci_excludes_zero": n_ci_excl,
        "pct_bootstrap_ci_excludes_zero": pct_ci,
        "n_cohens_d_large": n_d_large,
        "median_cohens_d": median_d,
        "min_cohens_d": float(np.min(ds)) if ds else None,
        "max_cohens_d": float(np.max(ds)) if ds else None,
        "verdict": verdict,
    }

    print()
    print("=" * 70)
    print(f"AGGREGATE: {n_total} configs run")
    print(f"  Welch significant (p<0.05):    {n_welch_sig}/{n_total} ({pct_welch:.0f}%)")
    print(f"  Bootstrap CI excludes zero:     {n_ci_excl}/{n_total} ({pct_ci:.0f}%)")
    print(f"  Cohen's d <= -0.8:              {n_d_large}/{n_total}")
    print(f"  median Cohen's d:               {median_d:+.3f}")
    print(f"  min/max d:                      {summary['min_cohens_d']:+.3f} / {summary['max_cohens_d']:+.3f}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    out = {
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "n_pca": N_PCA,
            "trailing_window_days": TRAILING_WINDOW,
            "refit_interval_days": REFIT_INTERVAL,
            "n_bootstrap": N_BOOT,
            "seed_offsets": SEED_OFFSETS,
            "hilbert_dims": HILBERT_DIMS,
            "operator_construction": "custom_offset_shifted_random_hermitian",
            "note": "v1 was buggy: np.random.seed did not vary the basis. v2 builds operators directly.",
        },
        "results": results,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
