"""Seed sensitivity probe for Berry Phase Rate.

Re-runs Berry Phase Rate with 5 different random-operator bases
(generated with seed_offset in {0, 100, 200, 300, 400}) and reports
per-window median Cohen's d for each basis.  Tests whether the
d=0.72 walk-forward result is fragile to operator basis choice.

Note: the production `operator_method='random'` always seeds operator
k with seed=k (deterministic), so the detector's own `seed` parameter
does not change the basis.  To actually vary the basis we generate
custom Hermitian operators with a per-basis offset.

Same fit/eval methodology as null_model_test.py: single global fit
over 2005-2024 SPY/DIA, per-crisis Cohen's d on each post-2005 crisis
window, median across crises.

Output:
  experiments/outputs/seed_sensitivity/seed_sensitivity_results.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data_loader import (
    ALL_CRISES,
    create_feature_matrix,
    fetch_data,
)
from experiments.evaluation import _cohens_d
from experiments.regime_comparison import HPO_CONFIGS
from qcml_geometry.observables import BaseRegimeDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

SEED_OFFSETS = [0, 100, 200, 300, 400]
START = "2005-01-01"
END = "2024-12-31"
METHOD = "Berry Phase Rate"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "seed_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def main():
    logger.info("Fetching SPY/DIA data %s to %s", START, END)
    raw = fetch_data(["SPY", "DIA"], START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    X, dates = create_feature_matrix(prices_df)
    X = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    date_index = dates[19:]
    logger.info("Feature matrix shape: %s", X.shape)

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
    logger.info("N crises: %d", len(real_windows))

    cfg = HPO_CONFIGS[METHOD]
    base_params = dict(cfg["params"])

    # Build n_pca_components Hermitian operators for each seed offset.
    # Match the canonical scheme: operator k uses np.random.default_rng(k+offset).
    hilbert_dim = base_params["hilbert_dim"]
    n_ops = base_params["n_pca_components"]

    def make_hermitian(rng: np.random.Generator) -> np.ndarray:
        A = rng.standard_normal((hilbert_dim, hilbert_dim)) + 1j * rng.standard_normal(
            (hilbert_dim, hilbert_dim)
        )
        return (A + A.conj().T) / 2.0

    def make_basis(offset: int) -> list[np.ndarray]:
        """Operator k seeded by (k + offset), matching canonical scheme."""
        return [make_hermitian(np.random.default_rng(k + offset)) for k in range(n_ops)]

    results = {
        "config": {
            "method": METHOD,
            "seed_offsets": SEED_OFFSETS,
            "n_crises": len(real_windows),
            "crisis_keys": crisis_keys,
            "base_params": base_params,
            "note": (
                "operator_method='random' uses deterministic per-index seeds, "
                "so we override with custom_operators built from a fresh RNG "
                "per offset in order to actually vary the basis."
            ),
        },
        "per_offset": {},
    }

    medians = []
    for offset in SEED_OFFSETS:
        operators = make_basis(offset)
        params = dict(base_params)
        params["custom_operators"] = operators
        params.pop("operator_method", None)
        det = cfg["class"](**params)
        det.fit(X)
        scores = det.compute_regime_scores(X)
        if len(scores) < len(date_index):
            pad = np.full(len(date_index) - len(scores), np.nan)
            scores = np.concatenate([pad, scores])
        ds = per_window_d(scores, real_windows)
        median_d = float(np.median(ds))
        medians.append(median_d)
        results["per_offset"][str(offset)] = {
            "per_window_d": ds,
            "median_d": median_d,
            "mean_d": float(np.mean(ds)),
        }
        logger.info("  offset=%d: median d = %.3f", offset, median_d)

    arr = np.array(medians)
    results["summary"] = {
        "n_offsets": len(SEED_OFFSETS),
        "median_of_medians": float(np.median(arr)),
        "mean_of_medians": float(np.mean(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "range": float(arr.max() - arr.min()),
    }
    logger.info(
        "Across %d offsets: median %.3f, range [%.3f, %.3f], std %.3f",
        len(SEED_OFFSETS),
        results["summary"]["median_of_medians"],
        results["summary"]["min"],
        results["summary"]["max"],
        results["summary"]["std"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"seed_sensitivity_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    stable = OUTPUT_DIR / "seed_sensitivity_results.json"
    with open(stable, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s and %s", json_path, stable)


if __name__ == "__main__":
    main()
