"""Bundled endogenous-vs-exogenous robustness suite (Path A+B+C).

Builds a unified 3D matrix testing whether the QCML spectral-gap endo/exo
finding generalizes across:
  A. Multiple classical baselines (5 baselines)
  B. Multiple assets (5 ETFs)
  C. Multiple QCML observables (5 observables, including original spectral gap)

For each (detector, asset) cell:
  1. Fit detector on the asset's feature matrix (canonical HPO_CONFIGS params).
  2. Compute regime-score time series (z-scored, higher = stronger crisis signal).
  3. Per crisis: intensity = max(score in crisis) / median(trailing 252d non-crisis score).
  4. Test endo vs exo: Welch t (two-sided), Mann-Whitney U (two-sided),
     Cohen's d, 5000-bootstrap CI on (endo - exo) mean difference.

Aggregate readouts:
  - Fraction of QCML cells with two-sided p < 0.05
  - Fraction of classical-baseline cells with two-sided p < 0.05
  - Holm-Bonferroni adjusted p-values across the matrix
  - Direction-consistency tally per detector

Pre-registered success criteria:
  - STRONG: ≥50% QCML cells significant AND ≤20% classical cells significant
            → JFE-Oxford-grade headline
  - MEDIUM: 30-50% QCML cells; classical mostly null
            → QF reframe
  - WEAK/NULL: <30% QCML OR classical also significant
               → drop endo/exo from headline; defer to Paper 2

Output: experiments/outputs/diagnostics/endo_exo_suite.json
"""

from __future__ import annotations

import json
import sys
import time
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
from experiments.crisis_classification import CRISIS_CLASSIFICATION
from experiments.regime_comparison import HPO_CONFIGS, CLASSICAL_CONFIGS

DETECTOR_CONFIGS = {**HPO_CONFIGS, **CLASSICAL_CONFIGS}

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"

ASSETS = ["SPY", "QQQ", "IWM", "EFA", "DIA"]

QCML_DETECTORS = [
    "Spectral Gap",
    "Berry Phase Rate",
    "Spectral Entropy",
    "Reduced Purity",
    "Hamiltonian Sensitivity",
]

CLASSICAL_DETECTORS = [
    "Rolling Vol Z",
    "CUSUM",
    "HMM",
    "BOCPD",
    "GARCH(1,1)",
]

TRAILING_WINDOW = 252
N_BOOT = 5_000

OUTPUT_PATH = (
    ROOT / "experiments" / "outputs" / "diagnostics" / "endo_exo_suite.json"
)


def fetch_features(symbol: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Per-asset feature matrix using the same pairing convention as
    multi_asset_revalidation.py: SPY/DIA for SPY, asset/SPY for others."""
    pair_anchor = "DIA" if symbol == "SPY" else "SPY"
    pair = [symbol, pair_anchor]
    raw = fetch_data(pair, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    if symbol not in prices_df.columns:
        raise RuntimeError(f"{symbol} not in fetched data")
    feat_arr, feat_index = create_feature_matrix(prices_df)
    return feat_arr, feat_index


def fit_and_score(detector_name: str, feat_arr: np.ndarray) -> np.ndarray:
    """Instantiate detector with HPO_CONFIGS params, fit, return regime-score series."""
    cfg = DETECTOR_CONFIGS.get(detector_name)
    if cfg is None:
        raise KeyError(f"{detector_name} not in DETECTOR_CONFIGS")
    cls = cfg["class"]
    params = dict(cfg["params"])
    det = cls(**params)
    det.fit(feat_arr)
    scores = det.compute_regime_scores(feat_arr)
    return np.asarray(scores)


def per_crisis_intensity(score: pd.Series, trailing_window: int = TRAILING_WINDOW) -> dict:
    """For each crisis: max(score) inside window / median(score) in trailing
    non-crisis baseline. Returns dict {crisis_id: ratio}."""
    out = {}
    # Build trailing median series (rolling) — but we want pre-crisis baseline
    rolling_median = score.rolling(trailing_window, min_periods=30).median()
    # Pre-crisis baseline: median of trailing window ending at crisis start
    for crisis_id, crisis in ALL_CRISES.items():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        mask = (score.index >= start) & (score.index <= end)
        if not mask.any():
            continue
        crisis_score = score[mask].dropna()
        if crisis_score.empty:
            continue
        # Baseline = median of detection scores in [start - trailing_window, start]
        baseline_window = score.loc[
            (score.index < start) & (score.index >= start - pd.Timedelta(days=int(trailing_window * 1.45)))
        ].dropna()
        if len(baseline_window) < 30:
            continue
        baseline = baseline_window.median()
        if baseline is np.nan or abs(baseline) < 1e-12:
            # Detection scores can be near zero; use mean abs as fallback denominator
            baseline = baseline_window.abs().mean()
            if baseline < 1e-12:
                continue
        intensity = float(crisis_score.max() / baseline)
        out[crisis_id] = intensity
    return out


def two_group_test(group_a: list[float], group_b: list[float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Welch t (two-sided), Mann-Whitney (two-sided), Cohen's d, bootstrap CI on (a - b) mean diff."""
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    cohen_d = (np.mean(a) - np.mean(b)) / pooled if pooled > 1e-12 else float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = np.mean(rng.choice(a, size=len(a), replace=True)) - np.mean(
            rng.choice(b, size=len(b), replace=True)
        )
    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_mean": float(np.mean(a) - np.mean(b)),
        "welch_t": float(t_stat),
        "welch_p_two_sided": float(t_p),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p_two_sided": float(u_p),
        "cohens_d_a_minus_b": float(cohen_d),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
    }


def holm_bonferroni(ps: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down. Returns boolean reject-array."""
    ps = np.asarray(ps)
    n = len(ps)
    sort_idx = np.argsort(ps)
    sorted_p = ps[sort_idx]
    reject = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p < alpha / (n - i):
            reject[sort_idx[i]] = True
        else:
            break
    return reject.tolist()


def main():
    cells = []
    t0 = time.time()
    n_total = len(ASSETS) * (len(QCML_DETECTORS) + len(CLASSICAL_DETECTORS))
    n_done = 0

    for asset in ASSETS:
        print(f"\n=== Asset: {asset} ===")
        try:
            feat_arr, feat_index = fetch_features(asset)
            print(f"  features: {feat_arr.shape}")
        except Exception as e:
            print(f"  FETCH ERROR: {e}")
            continue

        for detector_name in QCML_DETECTORS + CLASSICAL_DETECTORS:
            n_done += 1
            elapsed = time.time() - t0
            eta = (elapsed / max(n_done - 1, 1)) * (n_total - n_done) if n_done > 1 else None
            eta_str = f", ETA {eta/60:.1f} min" if eta else ""
            print(f"  [{n_done}/{n_total}] {detector_name}  (elapsed {elapsed/60:.1f} min{eta_str})")

            try:
                scores = fit_and_score(detector_name, feat_arr)
                score_s = pd.Series(scores, index=feat_index)
                intensities = per_crisis_intensity(score_s)
                # Group by classification
                endo_vals, exo_vals, per_crisis_records = [], [], []
                for cid, entry in CRISIS_CLASSIFICATION.items():
                    if cid not in intensities:
                        continue
                    val = intensities[cid]
                    per_crisis_records.append({
                        "crisis": cid,
                        "class": entry["class"],
                        "borderline": entry["borderline"],
                        "intensity": val,
                    })
                    if entry["class"] == "endogenous":
                        endo_vals.append(val)
                    else:
                        exo_vals.append(val)

                test = two_group_test(endo_vals, exo_vals)
                if test is None:
                    print(f"    insufficient data")
                    continue

                family = "QCML" if detector_name in QCML_DETECTORS else "Classical"
                d_str = test["cohens_d_a_minus_b"]
                p_str = test["welch_p_two_sided"]
                print(f"    family={family}  d={d_str:+.2f}  Welch p={p_str:.3f}")

                cells.append({
                    "detector": detector_name,
                    "family": family,
                    "asset": asset,
                    "test": test,
                    "per_crisis": per_crisis_records,
                })
            except Exception as e:
                import traceback
                print(f"    ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

    print(f"\n{len(cells)} cells completed (of {n_total})")

    # Aggregate
    qcml_cells = [c for c in cells if c["family"] == "QCML"]
    classical_cells = [c for c in cells if c["family"] == "Classical"]

    n_qcml_sig = sum(1 for c in qcml_cells if c["test"]["welch_p_two_sided"] < 0.05)
    n_classical_sig = sum(1 for c in classical_cells if c["test"]["welch_p_two_sided"] < 0.05)
    n_qcml_ci_excl = sum(1 for c in qcml_cells if c["test"]["ci_excludes_zero"])
    n_classical_ci_excl = sum(1 for c in classical_cells if c["test"]["ci_excludes_zero"])

    pct_qcml = 100 * n_qcml_sig / max(len(qcml_cells), 1)
    pct_classical = 100 * n_classical_sig / max(len(classical_cells), 1)

    # Holm-Bonferroni across all cells
    all_ps = [c["test"]["welch_p_two_sided"] for c in cells]
    rejects = holm_bonferroni(all_ps, alpha=0.05) if all_ps else []
    n_holm_qcml = sum(1 for c, r in zip(cells, rejects) if r and c["family"] == "QCML")
    n_holm_classical = sum(1 for c, r in zip(cells, rejects) if r and c["family"] == "Classical")

    # Direction consistency (endo > exo on intensity = positive d, since intensity is max/baseline and gap collapses → 1/gap z-spike → larger intensity for endogenous)
    n_qcml_pos_d = sum(1 for c in qcml_cells if c["test"]["cohens_d_a_minus_b"] > 0)
    n_classical_pos_d = sum(1 for c in classical_cells if c["test"]["cohens_d_a_minus_b"] > 0)

    # Verdict
    if pct_qcml >= 50 and pct_classical <= 20:
        verdict = "STRONG"
    elif pct_qcml >= 30 and pct_classical <= 30:
        verdict = "MEDIUM"
    else:
        verdict = "WEAK_OR_NULL"

    summary = {
        "n_qcml_cells": len(qcml_cells),
        "n_classical_cells": len(classical_cells),
        "n_qcml_significant_p005": n_qcml_sig,
        "n_classical_significant_p005": n_classical_sig,
        "pct_qcml_significant": pct_qcml,
        "pct_classical_significant": pct_classical,
        "n_qcml_ci_excludes_zero": n_qcml_ci_excl,
        "n_classical_ci_excludes_zero": n_classical_ci_excl,
        "n_qcml_holm_significant": n_holm_qcml,
        "n_classical_holm_significant": n_holm_classical,
        "n_qcml_d_positive": n_qcml_pos_d,
        "n_classical_d_positive": n_classical_pos_d,
        "verdict": verdict,
    }

    print()
    print("=" * 70)
    print(f"AGGREGATE — {len(cells)} cells")
    print(f"  QCML cells significant @ p<0.05:        {n_qcml_sig}/{len(qcml_cells)} ({pct_qcml:.0f}%)")
    print(f"  Classical cells significant @ p<0.05:   {n_classical_sig}/{len(classical_cells)} ({pct_classical:.0f}%)")
    print(f"  QCML CI excludes zero:                  {n_qcml_ci_excl}/{len(qcml_cells)}")
    print(f"  Classical CI excludes zero:              {n_classical_ci_excl}/{len(classical_cells)}")
    print(f"  Holm-Bonferroni QCML survivors:          {n_holm_qcml}")
    print(f"  Holm-Bonferroni Classical survivors:     {n_holm_classical}")
    print(f"  QCML d > 0 (endo > exo):                {n_qcml_pos_d}/{len(qcml_cells)}")
    print(f"  Classical d > 0:                        {n_classical_pos_d}/{len(classical_cells)}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    out = {
        "config": {
            "assets": ASSETS,
            "qcml_detectors": QCML_DETECTORS,
            "classical_detectors": CLASSICAL_DETECTORS,
            "trailing_window_days": TRAILING_WINDOW,
            "n_bootstrap": N_BOOT,
            "start": START,
            "end": END,
            "intensity_statistic": "max(score) in crisis / median(score) in trailing 252d non-crisis baseline",
            "test_direction": "two-sided",
        },
        "cells": cells,
        "summary": summary,
        "elapsed_minutes": (time.time() - t0) / 60,
        "timestamp": datetime.now().isoformat(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
