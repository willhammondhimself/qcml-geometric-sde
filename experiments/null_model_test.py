"""Null-model test: shuffle crisis labels and verify median Cohen's d collapses.

For each permutation:
  1. Sample K non-overlapping random windows of total length matching the real
     17-crisis panel from the SPY/DIA timeline (2005-01-01 to 2024-12-31).
  2. Treat those random windows as "fake crisis" labels.
  3. Compute Cohen's d for each detector using the fake labels.
  4. Record per-method median across the 17 fake windows.

Repeat N_PERMUTATIONS times and report the resulting null distribution.

If the detectors are picking up genuine regime signal (and not arbitrary
score structure), the null median d should collapse to ~0 with a tight 95% CI.

Output: experiments/outputs/null_model/null_model_results.json
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

N_PERMUTATIONS = 200
SEED = 42
START = "2005-01-01"
END = "2024-12-31"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "null_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POST_2005_CRISES = {
    k: v for k, v in ALL_CRISES.items()
    if pd.Timestamp(v["start"]) >= pd.Timestamp(START)
}

# Two representative geometric detectors: the headline (Berry Phase Rate) and
# the offline-best (Reduced Purity).  Adding a third would just inflate runtime
# without changing the story.
TARGET_METHODS = ["Berry Phase Rate", "Reduced Purity"]


def compute_score_series(method_name: str, X: np.ndarray) -> np.ndarray:
    """Run the detector once on the full feature matrix and return scores."""
    cfg = HPO_CONFIGS[method_name]
    cls = cfg["class"]
    params = cfg.get("params", {}).copy()

    detector = cls(**params)
    detector.fit(X)
    scores = detector.compute_regime_scores(X)
    return np.asarray(scores)


def crisis_mask_from_dates(
    crisis_starts: list[pd.Timestamp],
    crisis_ends: list[pd.Timestamp],
    date_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Build a boolean mask: True where date is inside any crisis window."""
    mask = np.zeros(len(date_index), dtype=bool)
    for s, e in zip(crisis_starts, crisis_ends):
        mask |= (date_index >= s) & (date_index <= e)
    return mask


def per_window_d(scores: np.ndarray, windows: list[tuple[int, int]]) -> list[float]:
    """For each window (start_idx, end_idx) compute Cohen's d vs everything else."""
    ds = []
    for s, e in windows:
        in_w = scores[s:e]
        out_mask = np.ones(len(scores), dtype=bool)
        out_mask[s:e] = False
        out_w = scores[out_mask]
        if len(in_w) < 2 or len(out_w) < 2:
            ds.append(0.0)
            continue
        in_clean = in_w[~np.isnan(in_w)]
        out_clean = out_w[~np.isnan(out_w)]
        if len(in_clean) < 2 or len(out_clean) < 2:
            ds.append(0.0)
            continue
        ds.append(_cohens_d(in_clean, out_clean))
    return ds


def sample_random_windows(
    n_dates: int,
    window_lengths: list[int],
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> list[tuple[int, int]]:
    """Place K windows of given lengths at random non-overlapping positions.

    Returns a list of (start, end) index pairs.
    """
    placed: list[tuple[int, int]] = []
    occupied = np.zeros(n_dates, dtype=bool)
    for length in window_lengths:
        for _ in range(max_attempts):
            start = int(rng.integers(0, max(1, n_dates - length)))
            if not occupied[start:start + length].any():
                occupied[start:start + length] = True
                placed.append((start, start + length))
                break
    return placed


def main():
    logger.info("Fetching SPY/DIA data %s to %s", START, END)
    raw = fetch_data(["SPY", "DIA"], START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    X, dates = create_feature_matrix(prices_df)
    # Build the same enriched features the comparison runner uses
    X = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    date_index = dates[19:]
    logger.info("Feature matrix shape: %s, dates: %s to %s",
                X.shape, date_index[0].date(), date_index[-1].date())

    real_starts = [pd.Timestamp(POST_2005_CRISES[k]["start"]) for k in POST_2005_CRISES]
    real_ends = [pd.Timestamp(POST_2005_CRISES[k]["end"]) for k in POST_2005_CRISES]

    real_windows: list[tuple[int, int]] = []
    for s, e in zip(real_starts, real_ends):
        in_range = (date_index >= s) & (date_index <= e)
        if not in_range.any():
            continue
        idxs = np.where(in_range)[0]
        real_windows.append((int(idxs[0]), int(idxs[-1]) + 1))
    window_lengths = [end - start for start, end in real_windows]
    logger.info(
        "Real crises: %d windows, total %d days, lengths min=%d max=%d median=%d",
        len(window_lengths), sum(window_lengths),
        min(window_lengths), max(window_lengths), int(np.median(window_lengths)),
    )

    rng = np.random.default_rng(SEED)
    results = {
        "config": {
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "start": START,
            "end": END,
            "n_dates": len(date_index),
            "n_crises": len(window_lengths),
            "window_lengths": window_lengths,
            "methods": TARGET_METHODS,
        },
        "real_d": {},
        "null_d_distribution": {},
        "summary": {},
    }

    score_series = {}
    for method in TARGET_METHODS:
        logger.info("Computing %s scores on full timeline...", method)
        scores = compute_score_series(method, X)
        if len(scores) < len(date_index):
            pad = np.full(len(date_index) - len(scores), np.nan)
            scores = np.concatenate([pad, scores])
        elif len(scores) > len(date_index):
            scores = scores[-len(date_index):]
        score_series[method] = scores

        real_per_window = per_window_d(scores, real_windows)
        d_median = float(np.median(real_per_window))
        results["real_d"][method] = d_median
        results.setdefault("real_per_window", {})[method] = real_per_window
        logger.info("  %s real median d (across %d crises) = %.3f",
                    method, len(real_windows), d_median)

    # Two complementary null distributions:
    #   (A) Random non-overlapping windows of matched lengths.
    #   (B) Circular shift of the score series by a random offset, then
    #       compute d on the original crisis windows.
    # (A) tests whether the crisis dates are special; (B) tests whether the
    # score series has temporal structure aligned with crises specifically.
    for method in TARGET_METHODS:
        logger.info("Null sampling (random windows) for %s ...", method)
        scores = score_series[method]
        n_dates = len(scores)

        null_random_windows = []
        for _ in range(N_PERMUTATIONS):
            fake_windows = sample_random_windows(n_dates, window_lengths, rng)
            if len(fake_windows) < len(window_lengths):
                continue
            ds = per_window_d(scores, fake_windows)
            null_random_windows.append(float(np.median(ds)))

        logger.info("Null sampling (circular shift) for %s ...", method)
        null_circ_shift = []
        for _ in range(N_PERMUTATIONS):
            offset = int(rng.integers(window_lengths and max(window_lengths) or 1, n_dates))
            shifted = np.concatenate([scores[offset:], scores[:offset]])
            ds = per_window_d(shifted, real_windows)
            null_circ_shift.append(float(np.median(ds)))

        arr_rw = np.array(null_random_windows)
        arr_cs = np.array(null_circ_shift)
        results["null_d_distribution"][method] = {
            "random_windows": arr_rw.tolist(),
            "circular_shift": arr_cs.tolist(),
        }
        real = float(results["real_d"][method])
        results["summary"][method] = {
            "real_median_d": real,
            "random_windows": {
                "null_median": float(np.median(arr_rw)),
                "null_q025": float(np.quantile(arr_rw, 0.025)),
                "null_q975": float(np.quantile(arr_rw, 0.975)),
                "p_value": float((arr_rw >= real).mean()),
            },
            "circular_shift": {
                "null_median": float(np.median(arr_cs)),
                "null_q025": float(np.quantile(arr_cs, 0.025)),
                "null_q975": float(np.quantile(arr_cs, 0.975)),
                "p_value": float((arr_cs >= real).mean()),
            },
        }
        for null_type in ("random_windows", "circular_shift"):
            s = results["summary"][method][null_type]
            logger.info(
                "  %s: null median=%.3f, 95%% CI [%.3f, %.3f], p=%.3f",
                null_type,
                s["null_median"], s["null_q025"], s["null_q975"], s["p_value"],
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"null_model_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", output_path)

    # Also write a stable filename for the paper
    stable_path = OUTPUT_DIR / "null_model_results.json"
    with open(stable_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s (stable)", stable_path)


if __name__ == "__main__":
    main()
