"""E_0 ground state energy: full-panel evaluation and gate check.

Pipeline:
  1. Load SPY/DIA features 2005-2024 (matching null_model_test.py).
  2. Fit a QCMLGeometry with random Hermitian operators (deterministic
     per-index seeding to match the canonical Berry basis).
  3. Compute E_0(t) = ground state energy at each time step.
  4. Z-score over an expanding window (causal).
  5. Compute per-crisis Cohen's d on the post-2005 panel.
  6. Compute Pearson r against 20-day rolling realized volatility.
  7. Apply the decision gate from research/e0_correlation_check.md:
        promote if median_d > 0.5 AND |r_vol| < 0.3.

Output:
  experiments/outputs/e0_promotion/e0_full_panel_results.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.data_loader import (
    ALL_CRISES,
    create_feature_matrix,
    fetch_data,
)
from experiments.evaluation import _cohens_d
from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import BaseRegimeDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

START = "2005-01-01"
END = "2024-12-31"
HILBERT_DIM = 6
N_PCA = 8
ROLLING_WINDOW = 20
MIN_EXPANDING = 252  # 1 year warmup before z-scoring
OUTPUT_DIR = Path(__file__).parent / "outputs" / "e0_promotion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Decision gate from research/e0_correlation_check.md
GATE_MIN_D = 0.5
GATE_MAX_R_VOL = 0.3

POST_2005_CRISES = {
    k: v for k, v in ALL_CRISES.items()
    if pd.Timestamp(v["start"]) >= pd.Timestamp(START)
}


def per_window_d(scores: np.ndarray, windows: list[tuple[int, int]]) -> list[float]:
    ds = []
    for s, e in windows:
        in_w = scores[s:e]
        out_mask = np.ones(len(scores), dtype=bool)
        out_mask[s:e] = False
        out_w = scores[out_mask]
        in_clean = in_w[~np.isnan(in_w)]
        out_clean = out_w[~np.isnan(out_w)]
        if len(in_clean) < 2 or len(out_clean) < 2:
            ds.append(0.0)
            continue
        ds.append(_cohens_d(in_clean, out_clean))
    return ds


def expanding_zscore(values: np.ndarray, min_periods: int) -> np.ndarray:
    n = len(values)
    z = np.full(n, np.nan)
    for t in range(min_periods, n):
        mu = np.nanmean(values[:t])
        sigma = np.nanstd(values[:t], ddof=1)
        if sigma > 1e-12:
            z[t] = (values[t] - mu) / sigma
    return z


def main():
    logger.info("Fetching SPY/DIA data %s to %s", START, END)
    raw = fetch_data(["SPY", "DIA"], START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    X, dates = create_feature_matrix(prices_df)
    X = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    date_index = dates[19:]
    logger.info("Feature matrix: %s", X.shape)

    # PCA + sphere normalization (matching Berry's pre-processing)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=N_PCA).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_pca_sphere = X_pca / np.where(norms > 1e-12, norms, 1.0)
    logger.info("PCA-projected feature matrix: %s", X_pca_sphere.shape)

    # Build QCMLGeometry with the canonical random basis
    geo = QCMLGeometry(n_features=N_PCA, hilbert_dim=HILBERT_DIM)
    geo.fit_operators(X_pca_sphere, method="random")
    logger.info("QCMLGeometry fitted with %d random Hermitian operators", N_PCA)

    # Compute E_0(x) at each time step
    logger.info("Computing E_0(x) over %d time steps...", X_pca_sphere.shape[0])
    energies = np.empty(X_pca_sphere.shape[0])
    for t in range(X_pca_sphere.shape[0]):
        _, energy = geo.quasi_coherent_state(X_pca_sphere[t], return_energy=True)
        energies[t] = energy

    # Z-score over expanding window, then take rolling mean as a smoother.
    z_scores = expanding_zscore(energies, MIN_EXPANDING)
    z_smooth = pd.Series(z_scores).rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean().values
    logger.info("E_0 z-score range: [%.3f, %.3f]",
                np.nanmin(z_smooth), np.nanmax(z_smooth))

    # Per-crisis Cohen's d
    real_starts = [pd.Timestamp(POST_2005_CRISES[k]["start"]) for k in POST_2005_CRISES]
    real_ends = [pd.Timestamp(POST_2005_CRISES[k]["end"]) for k in POST_2005_CRISES]
    real_windows: list[tuple[int, int]] = []
    crisis_keys: list[str] = []
    for key, s, e in zip(POST_2005_CRISES.keys(), real_starts, real_ends):
        in_range = (date_index >= s) & (date_index <= e)
        if not in_range.any():
            continue
        idxs = np.where(in_range)[0]
        real_windows.append((int(idxs[0]), int(idxs[-1]) + 1))
        crisis_keys.append(key)

    per_crisis = per_window_d(z_smooth, real_windows)
    median_d = float(np.median(per_crisis))
    logger.info("E_0 median per-crisis Cohen's d (%d crises) = %.3f",
                len(per_crisis), median_d)

    # Pearson r against 20-day rolling realized vol
    spy = prices_df["SPY"].reindex(date_index)
    log_ret = np.log(spy / spy.shift(1))
    vol_20d = log_ret.rolling(20).std()
    z_series = pd.Series(z_smooth, index=date_index)
    common = pd.concat([z_series, vol_20d], axis=1).dropna()
    r_vol = float(common.iloc[:, 0].corr(common.iloc[:, 1]))
    logger.info("E_0 z-score correlation with 20d realized vol: r = %.3f", r_vol)

    # Decision gate
    gate_pass = (median_d > GATE_MIN_D) and (abs(r_vol) < GATE_MAX_R_VOL)
    decision = "PROMOTE" if gate_pass else "DEFER"
    logger.info("Gate: median_d > %.2f -> %s, |r_vol| < %.2f -> %s. Decision: %s",
                GATE_MIN_D, median_d > GATE_MIN_D,
                GATE_MAX_R_VOL, abs(r_vol) < GATE_MAX_R_VOL,
                decision)

    results = {
        "config": {
            "start": START, "end": END,
            "hilbert_dim": HILBERT_DIM, "n_pca": N_PCA,
            "rolling_window": ROLLING_WINDOW,
            "min_expanding": MIN_EXPANDING,
            "n_crises": len(real_windows),
            "crisis_keys": crisis_keys,
        },
        "per_crisis_d": dict(zip(crisis_keys, per_crisis)),
        "summary": {
            "median_d": median_d,
            "mean_d": float(np.mean(per_crisis)),
            "min_d": float(np.min(per_crisis)),
            "max_d": float(np.max(per_crisis)),
            "r_vol_20d": r_vol,
            "gate_min_d": GATE_MIN_D,
            "gate_max_r_vol": GATE_MAX_R_VOL,
            "gate_passes": gate_pass,
            "decision": decision,
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"e0_full_panel_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    stable = OUTPUT_DIR / "e0_full_panel_results.json"
    with open(stable, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s and %s", json_path, stable)


if __name__ == "__main__":
    main()
