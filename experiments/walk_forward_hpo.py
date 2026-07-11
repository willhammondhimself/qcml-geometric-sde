"""Honest (nested, walk-forward) hyperparameter optimization.

Motivation
----------
``optuna_hpo.py`` selects hyperparameters by maximizing median Cohen's d over
``OPT_CRISES`` — the *same* post-2005 crisis panel the leaderboard reports on.
The per-crisis causal cutoff only prevents within-day preprocessing leakage; it
does nothing about the *model-selection* objective having seen every evaluation
crisis. That is look-ahead in model selection and invites overfitting.

This module removes it with an expanding-window nested protocol:

  For each out-of-sample (OOS) test crisis, select hyperparameters via Optuna
  using ONLY the crises that occurred strictly before it (chronologically);
  freeze the winning config; then evaluate Cohen's d on the held-out crisis.

The per-window OOS d-values it produces are exactly what
``walk_forward_bootstrap_ci.py`` previously hardcoded from the LaTeX table — so
this also de-circularizes that "verification".

It additionally computes the *in-sample* optuna baseline (select on the full
panel, score per crisis) so the overfitting gap (in-sample d − nested OOS d) is
quantified directly.

Usage
-----
    python experiments/walk_forward_hpo.py --quick            # fast smoke run
    python experiments/walk_forward_hpo.py \
        --methods "Berry Phase Rate" "Multi-Lag Fidelity" "Spectral Gap" \
        --n-trials 25
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python experiments/walk_forward_hpo.py`

from experiments.data_loader import ALL_CRISES, create_feature_matrix, fetch_data  # noqa: E402
from experiments.evaluation import _cohens_d  # noqa: E402
from experiments.optuna_hpo import SEARCH_SPACES  # noqa: E402
from qcml_geometry.observables import BaseRegimeDetector  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "experiments" / "outputs" / "regime_detection" / "walk_forward"

# OOS windows reported in the paper's walk-forward table (chronological).
DEFAULT_OOS_KEYS = [
    "2010_flash",
    "2011_euro",
    "2015_china",
    "2018_volmageddon",
    "2018_q4",
    "2019_repo",
    "2020_covid",
    "2022_rates",
    "2023_svb",
]


# --------------------------------------------------------------------------- #
# Data + evaluation primitives (mirror regime_comparison.py exactly)
# --------------------------------------------------------------------------- #


def prepare_data(symbols=("SPY", "DIA"), start="1995-01-01", end="2024-12-31"):
    """Fetch prices → feature matrix → enriched features, as in the main pipeline."""
    raw = fetch_data(list(symbols) + ["^VIX"], start, end)
    prices = raw["close"].unstack("symbol").dropna()
    if "^VIX" in prices.columns:
        prices = prices.drop(columns=["^VIX"])
    X, dates = create_feature_matrix(prices)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    return X_enriched, dates_enriched


def crisis_cohens_d(
    detector_class,
    params,
    X_enriched,
    dates_enriched,
    crisis,
    window_size=10,
    normal_mode="global",
    local_days=60,
):
    """Causal Cohen's d for one crisis: crisis-window scores vs a pre-crisis normal.

    Preprocessing/operators are fit only on data before the crisis
    (``causal_fit_length``). ``normal_mode`` selects the comparison group:
    ``"global"`` = all pre-cutoff history (the original metric, but confounded by
    non-stationarity — a late window separates from old history regardless of
    crisis); ``"local"`` = the ``local_days`` bars immediately before the cutoff
    (a matched baseline that isolates the regime change). Returns NaN if the
    crisis lacks enough data or the config raises.
    """
    cs, ce = pd.Timestamp(crisis["start"]), pd.Timestamp(crisis["end"])
    cutoff = cs - pd.Timedelta(days=window_size)
    fit_end = int(np.searchsorted(dates_enriched, cutoff))

    crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
    if normal_mode == "local":
        idx = np.arange(len(dates_enriched))
        normal_mask = (idx >= max(0, fit_end - local_days)) & (idx < fit_end) & ~crisis_mask
    else:
        normal_mask = ~crisis_mask & (np.arange(len(dates_enriched)) < fit_end)
    if crisis_mask.sum() < 5 or fit_end < 60 or normal_mask.sum() < 30:
        return np.nan

    try:
        # Geometric detectors need the causal preprocessing cutoff; classical
        # baselines (expanding/rolling stats) are causal by construction and
        # don't take the kwarg.
        if "causal_fit_length" in inspect.signature(detector_class.__init__).parameters:
            det = detector_class(**params, causal_fit_length=fit_end)
        else:
            det = detector_class(**params)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
    except Exception as exc:  # invalid config in the search space → skip
        logger.debug("config failed on %s: %s", crisis["start"], exc)
        return np.nan

    crisis_scores = scores[crisis_mask]
    normal_scores = scores[normal_mask]
    crisis_scores = crisis_scores[np.isfinite(crisis_scores)]
    normal_scores = normal_scores[np.isfinite(normal_scores)]
    if len(crisis_scores) < 3 or len(normal_scores) < 3:
        return np.nan
    return float(_cohens_d(crisis_scores, normal_scores))


def _compute_d(detector_class, params, X_enriched, dates_enriched, crisis_key, window_size=10):
    """Keyed entry point for one (detector, crisis) Cohen's d.

    Indirection so a memoizing cache can be installed by rebinding ``_D_FN``
    (see experiments/hpo_cache.py). Default is the direct computation.
    """
    return crisis_cohens_d(
        detector_class, params, X_enriched, dates_enriched, ALL_CRISES[crisis_key], window_size
    )


# Rebindable hook: hpo_cache.CohensDCache.install() swaps this for a memoizing
# version. Defaults to the direct computation (no caching).
_D_FN = _compute_d


def median_d_over(detector_class, params, X_enriched, dates_enriched, crises):
    ds = [
        d
        for ck in crises
        if np.isfinite(d := _D_FN(detector_class, params, X_enriched, dates_enriched, ck))
    ]
    return float(np.median(ds)) if ds else 0.0


def _full_params(method, best_params):
    """Reconstruct the complete detector kwargs from Optuna's suggested params."""
    return SEARCH_SPACES[method]["params"](optuna.trial.FixedTrial(best_params))


def select_params(method, X_enriched, dates_enriched, selection_crises, n_trials, seed):
    """Optuna search maximizing median d over the selection (prior) crises only."""
    space = SEARCH_SPACES[method]
    cls, param_fn = space["class"], space["params"]

    def objective(trial):
        params = param_fn(trial)
        return median_d_over(cls, params, X_enriched, dates_enriched, selection_crises)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


def chronological_keys(keys):
    return sorted(keys, key=lambda k: pd.Timestamp(ALL_CRISES[k]["start"]))


def selection_pool(test_key, candidate_keys=None):
    """Crises usable to tune for ``test_key``: those that ENDED strictly before
    the test crisis STARTS. This is the leak-free invariant — no crisis at or
    after the test window can influence hyperparameter selection.
    """
    candidate_keys = candidate_keys if candidate_keys is not None else list(ALL_CRISES.keys())
    test_start = pd.Timestamp(ALL_CRISES[test_key]["start"])
    return {
        k: ALL_CRISES[k]
        for k in chronological_keys(candidate_keys)
        if k != test_key and pd.Timestamp(ALL_CRISES[k]["end"]) < test_start
    }


def run_nested(method, X_enriched, dates_enriched, oos_keys, n_trials, seed):
    """Expanding-window nested HPO: select on prior crises, evaluate OOS."""
    cls = SEARCH_SPACES[method]["class"]
    oos_ordered = chronological_keys(oos_keys)

    per_window = {}
    for test_key in oos_ordered:
        selection = selection_pool(test_key)
        if not selection:
            logger.info("  %s %s: no prior crises, skipping OOS window", method, test_key)
            continue

        best_params, sel_d = select_params(
            method, X_enriched, dates_enriched, selection, n_trials, seed
        )
        full = _full_params(method, best_params)
        oos_d = _D_FN(cls, full, X_enriched, dates_enriched, test_key)
        per_window[test_key] = {
            "oos_d": None if not np.isfinite(oos_d) else oos_d,
            "selection_median_d": sel_d,
            "n_selection_crises": len(selection),
            "best_params": best_params,
        }
        logger.info(
            "  %-20s %-16s OOS d=%5s  (selected on %d prior crises, sel median d=%.3f)",
            method,
            test_key,
            "n/a" if not np.isfinite(oos_d) else f"{oos_d:.3f}",
            len(selection),
            sel_d,
        )
    return per_window


def run_in_sample(method, X_enriched, dates_enriched, oos_keys, n_trials, seed):
    """Optimistic baseline: select on the FULL OOS panel, then score per crisis."""
    cls = SEARCH_SPACES[method]["class"]
    panel = {k: ALL_CRISES[k] for k in oos_keys}
    best_params, _ = select_params(method, X_enriched, dates_enriched, panel, n_trials, seed)
    full = _full_params(method, best_params)
    per_window = {}
    for k in chronological_keys(oos_keys):
        d = _D_FN(cls, full, X_enriched, dates_enriched, k)
        per_window[k] = None if not np.isfinite(d) else d
    return per_window, best_params


def _median(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    return float(np.median(arr)) if len(arr) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--methods", nargs="+", default=["Berry Phase Rate", "Multi-Lag Fidelity", "Spectral Gap"]
    )
    ap.add_argument("--oos-keys", nargs="+", default=DEFAULT_OOS_KEYS)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symbols", nargs="+", default=["SPY", "DIA"])
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument(
        "--quick", action="store_true", help="fast smoke: SPY-only, 6 trials, 4 OOS windows"
    )
    ap.add_argument(
        "--no-in-sample", action="store_true", help="skip the in-sample optuna baseline comparison"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        args.symbols = ["SPY"]
        args.n_trials = 6
        args.oos_keys = ["2015_china", "2020_covid", "2022_rates", "2023_svb"]

    logger.info("Preparing data (%s, %s→2024)...", args.symbols, args.start)
    X_enriched, dates_enriched = prepare_data(tuple(args.symbols), start=args.start)
    logger.info(
        "Enriched features: %s, %s → %s",
        X_enriched.shape,
        dates_enriched[0].date(),
        dates_enriched[-1].date(),
    )

    results = {
        "config": {
            "methods": args.methods,
            "oos_keys": args.oos_keys,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "symbols": args.symbols,
            "start": args.start,
            "protocol": "expanding-window nested HPO (select on strictly-prior crises)",
        },
        "nested": {},
        "in_sample": {},
        "overfitting_gap": {},
    }

    for method in args.methods:
        logger.info("\n=== %s — nested walk-forward HPO ===", method)
        nested = run_nested(
            method, X_enriched, dates_enriched, args.oos_keys, args.n_trials, args.seed
        )
        results["nested"][method] = nested
        nested_median = _median([w["oos_d"] for w in nested.values()])

        in_sample_median = None
        if not args.no_in_sample:
            logger.info("=== %s — in-sample optuna baseline ===", method)
            in_sample, in_params = run_in_sample(
                method, X_enriched, dates_enriched, args.oos_keys, args.n_trials, args.seed
            )
            results["in_sample"][method] = {"per_window": in_sample, "best_params": in_params}
            in_sample_median = _median(in_sample.values())

        results["overfitting_gap"][method] = {
            "nested_oos_median_d": nested_median,
            "in_sample_median_d": in_sample_median,
            "gap": None if in_sample_median is None else in_sample_median - nested_median,
        }
        gap = results["overfitting_gap"][method]
        logger.info(
            "  %s: nested OOS median d=%.3f | in-sample median d=%s | gap=%s",
            method,
            nested_median,
            "n/a" if in_sample_median is None else f"{in_sample_median:.3f}",
            "n/a" if gap["gap"] is None else f"{gap['gap']:+.3f}",
        )

    results["timestamp"] = datetime.now().isoformat()
    out = Path(args.out) if args.out else OUTPUT_DIR / "wf_nested_hpo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nWrote %s", out)


if __name__ == "__main__":
    main()
