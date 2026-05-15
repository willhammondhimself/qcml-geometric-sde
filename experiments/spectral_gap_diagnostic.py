"""Day-1 diagnostic: does the QCML spectral gap actually approach zero during crises?

Pulls the raw spectral-gap time series across 2005-2024 SPY/DIA, then for each
of 17 historical crises computes (a) the minimum gap inside the crisis window
and (b) the ratio of that minimum to the trailing 252-day mean.

Decision rule (per ~/tooling/plans/here-s-a-prompt-you-scalable-stonebraker.md):
    min(gap)/trailing_mean < 0.10 in >= 10 of 17 crises  -> commit to Angle 3
    min(gap)/trailing_mean > 0.50 in most crises          -> commit to Angle 5
    in between                                            -> try power-law version of Angle 3

Output: experiments/outputs/diagnostics/spectral_gap_diagnostic.json + console table
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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

warnings.filterwarnings("ignore")

START = "2005-01-01"
END = "2024-12-31"
SYMBOLS = ["SPY", "DIA"]

# Match the canonical Berry params used in the paper for an apples-to-apples comparison
HILBERT_DIM = 6
N_PCA = 8
SEED = 42
TRAILING_WINDOW = 252  # 1 trading year

OUTPUT_PATH = (
    ROOT / "experiments" / "outputs" / "diagnostics" / "spectral_gap_diagnostic.json"
)


def fit_geometry(X: np.ndarray) -> tuple[QCMLGeometry, StandardScaler, PCA]:
    """Fit the QCML embedding once on the full series (causal preprocessing not
    needed for a diagnostic of gap behavior)."""
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=min(N_PCA, X.shape[1])).fit(scaler.transform(X))
    X_pca = pca.transform(scaler.transform(X))
    geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=HILBERT_DIM)
    np.random.seed(SEED)
    geo.fit_operators(X_pca, method="random")
    return geo, scaler, pca


def main() -> None:
    print(f"Fetching {SYMBOLS} {START}..{END}")
    raw = fetch_data(SYMBOLS, START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    feat_arr, feat_index = create_feature_matrix(prices_df)
    print(f"  features: {feat_arr.shape}")

    print("Fitting QCML geometry (random operators, seed=42)...")
    geo, scaler, pca = fit_geometry(feat_arr)
    X_pca = pca.transform(scaler.transform(feat_arr))

    print("Computing spectral gap at every timestep (this may take a minute)...")
    gap_series = np.empty(len(X_pca))
    for t in range(len(X_pca)):
        gap_series[t] = geo.spectral_gap(X_pca[t])
        if t % 1000 == 0:
            print(f"  t={t}/{len(X_pca)}, gap={gap_series[t]:.4e}")

    gap = pd.Series(gap_series, index=feat_index, name="spectral_gap")
    print(f"  gap series stats: min={gap.min():.4e}, max={gap.max():.4e}, mean={gap.mean():.4e}")

    # Trailing 252-day mean
    trailing_mean = gap.rolling(TRAILING_WINDOW, min_periods=30).mean()

    # Per-crisis analysis
    rows = []
    for crisis_id, crisis in ALL_CRISES.items():
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])
        mask = (gap.index >= start) & (gap.index <= end)
        if not mask.any():
            continue

        crisis_gap = gap[mask]
        crisis_trailing = trailing_mean[mask]
        # Use the trailing mean at the START of the crisis window
        baseline = trailing_mean.loc[crisis_trailing.index[0]]
        if pd.isna(baseline) or baseline < 1e-12:
            # Use overall pre-crisis mean if trailing not available
            pre_mask = (gap.index < start)
            baseline = gap[pre_mask].mean() if pre_mask.any() else gap.mean()

        min_gap = crisis_gap.min()
        ratio = min_gap / baseline if baseline > 1e-12 else np.nan

        rows.append({
            "crisis": crisis_id,
            "label": crisis.get("label", crisis_id),
            "start": str(start.date()),
            "end": str(end.date()),
            "n_days": int(mask.sum()),
            "baseline_gap": float(baseline),
            "min_gap": float(min_gap),
            "min_to_baseline_ratio": float(ratio) if not pd.isna(ratio) else None,
        })

    # Sort by ratio (smallest first = strongest gap collapse)
    rows = sorted(
        rows, key=lambda r: r["min_to_baseline_ratio"] if r["min_to_baseline_ratio"] is not None else 1e9
    )

    # Decision rule
    valid = [r for r in rows if r["min_to_baseline_ratio"] is not None]
    n_strong = sum(1 for r in valid if r["min_to_baseline_ratio"] < 0.10)
    n_moderate = sum(1 for r in valid if 0.10 <= r["min_to_baseline_ratio"] < 0.50)
    n_weak = sum(1 for r in valid if r["min_to_baseline_ratio"] >= 0.50)

    if n_strong >= 10:
        verdict = "ANGLE_3_ALIVE"
        rationale = f"{n_strong}/{len(valid)} crises drop below 10% of baseline -> gap closure story is real"
    elif n_weak >= len(valid) - 3:
        verdict = "ANGLE_5_PIVOT"
        rationale = f"{n_weak}/{len(valid)} crises stay above 50% of baseline -> gap doesn't close meaningfully"
    else:
        verdict = "POWER_LAW_PROBE"
        rationale = (
            f"messy middle: {n_strong} <10%, {n_moderate} 10-50%, {n_weak} >=50% -> "
            f"try power-law fit before committing"
        )

    results = {
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "hilbert_dim": HILBERT_DIM,
            "n_pca": N_PCA,
            "seed": SEED,
            "trailing_window_days": TRAILING_WINDOW,
        },
        "per_crisis": rows,
        "summary": {
            "n_crises_analyzed": len(valid),
            "n_min_to_baseline_lt_010": n_strong,
            "n_min_to_baseline_010_to_050": n_moderate,
            "n_min_to_baseline_gte_050": n_weak,
            "verdict": verdict,
            "rationale": rationale,
        },
        "timestamp": datetime.now().isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("PER-CRISIS GAP COLLAPSE TABLE (sorted strongest collapse first)")
    print("=" * 80)
    print(f"{'Crisis':<22} {'Baseline':>12} {'Min(gap)':>12} {'Ratio':>10}  Verdict")
    for r in rows:
        ratio = r["min_to_baseline_ratio"]
        flag = "STRONG" if ratio < 0.10 else "moderate" if ratio < 0.50 else "weak"
        print(
            f"{r['crisis']:<22} {r['baseline_gap']:>12.4e} {r['min_gap']:>12.4e}"
            f" {ratio:>10.3f}  {flag}"
        )
    print()
    print("=" * 80)
    print(f"VERDICT: {verdict}")
    print(f"  {rationale}")
    print(f"  strong  (<10% of baseline):   {n_strong}/{len(valid)}")
    print(f"  moderate (10-50% of baseline): {n_moderate}/{len(valid)}")
    print(f"  weak    (>=50% of baseline):  {n_weak}/{len(valid)}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
