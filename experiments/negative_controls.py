"""Negative controls for the nested walk-forward HPO harness.

The headline check: a label-permutation null. If we replace the real crisis
windows with random windows placed in *non-crisis* periods and re-run the exact
nested protocol, the out-of-sample Cohen's d must collapse to ~0. If it doesn't,
the harness is leaking and any positive result is suspect.

This deliberately bypasses ALL_CRISES / the cache and drives the protocol over an
arbitrary chronological set of (start, end) windows, reusing crisis_cohens_d and
the SEARCH_SPACES objective.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.data_loader import ALL_CRISES  # noqa: E402
from experiments.optuna_hpo import SEARCH_SPACES  # noqa: E402
from experiments.walk_forward_hpo import (  # noqa: E402
    _full_params,
    crisis_cohens_d,
    prepare_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)
OUTPUT_DIR = ROOT / "experiments" / "outputs" / "regime_detection" / "overfitting"


def _trading_len(dates_enriched, crisis):
    cs, ce = pd.Timestamp(crisis["start"]), pd.Timestamp(crisis["end"])
    return int(((dates_enriched >= cs) & (dates_enriched <= ce)).sum())


def random_window_crises(dates_enriched, real_crises, seed, warmup=250, gap_days=20):
    """One random window per real crisis, matched in trading-day length, placed in
    a non-crisis region (post-warmup, not overlapping any real crisis ± buffer)."""
    rng = np.random.default_rng(seed)
    T = len(dates_enriched)
    # mask of dates inside any real crisis (± buffer) — forbidden for random windows
    forbidden = np.zeros(T, dtype=bool)
    for cr in ALL_CRISES.values():
        cs = pd.Timestamp(cr["start"]) - pd.Timedelta(days=gap_days)
        ce = pd.Timestamp(cr["end"]) + pd.Timedelta(days=gap_days)
        forbidden |= (dates_enriched >= cs) & (dates_enriched <= ce)

    fake = {}
    for i, cr in enumerate(real_crises.values()):
        L = max(_trading_len(dates_enriched, cr), 5)
        for _ in range(200):  # rejection sampling
            start = int(rng.integers(warmup, T - L - 1))
            if not forbidden[start : start + L].any():
                fake[f"rand_{i}"] = {
                    "start": str(dates_enriched[start].date()),
                    "end": str(dates_enriched[start + L - 1].date()),
                }
                forbidden[start : start + L] = True  # don't reuse
                break
    return fake


def _chrono(crises):
    return dict(sorted(crises.items(), key=lambda kv: pd.Timestamp(kv[1]["start"])))


def _memo_cd(cls, params, X, dates, cr, memo, normal_mode="global"):
    """crisis_cohens_d with a local memo (selection pools overlap → big speedup)."""
    key = (
        cls.__name__,
        json.dumps(params, sort_keys=True, default=str),
        cr["start"],
        cr["end"],
        normal_mode,
    )
    if key in memo:
        return memo[key]
    d = crisis_cohens_d(cls, params, X, dates, cr, normal_mode=normal_mode)
    memo[key] = d
    return d


def nested_oos_median(method, X, dates, crises, n_trials, seed, memo=None, normal_mode="global"):
    """Run the nested protocol over an arbitrary chronological crises dict.

    For each test window, tune on windows that ended strictly before it, then
    evaluate OOS on the held-out window. Returns the median OOS d.
    """
    cls = SEARCH_SPACES[method]["class"]
    pfn = SEARCH_SPACES[method]["params"]
    memo = memo if memo is not None else {}
    crises = _chrono(crises)
    keys = list(crises)
    oos = []
    for i, test_key in enumerate(keys):
        test_start = pd.Timestamp(crises[test_key]["start"])
        selection = {k: crises[k] for k in keys[:i] if pd.Timestamp(crises[k]["end"]) < test_start}
        if not selection:
            continue

        def objective(trial):
            params = pfn(trial)
            ds = [
                d
                for cr in selection.values()
                if np.isfinite(d := _memo_cd(cls, params, X, dates, cr, memo, normal_mode))
            ]
            return float(np.median(ds)) if ds else 0.0

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        full = _full_params(method, study.best_params)
        d = _memo_cd(cls, full, X, dates, crises[test_key], memo, normal_mode)
        if np.isfinite(d):
            oos.append(d)
    return float(np.median(oos)) if oos else float("nan")


def leak_test(method, oos_keys, n_trials=25, seed=42, n_perm=10, data=None, normal_mode="global"):
    data = data or prepare_data()
    X, dates = data
    real_crises = {k: ALL_CRISES[k] for k in oos_keys}

    real_median = nested_oos_median(
        method, X, dates, real_crises, n_trials, seed, normal_mode=normal_mode
    )
    logger.info("%s [%s]: REAL nested-OOS median d = %.3f", method, normal_mode, real_median)

    null_medians = []
    for p in range(n_perm):
        fake = random_window_crises(dates, real_crises, seed=1000 + p)
        m = nested_oos_median(method, X, dates, fake, n_trials, seed, normal_mode=normal_mode)
        null_medians.append(m)
        logger.info("  null perm %d: median d = %.3f", p, m)

    null_arr = np.array([m for m in null_medians if np.isfinite(m)])
    pval = float((np.sum(null_arr >= real_median) + 1) / (len(null_arr) + 1))
    return {
        "method": method,
        "real_nested_oos_median_d": real_median,
        "null_medians": null_medians,
        "null_mean": float(np.mean(null_arr)) if len(null_arr) else None,
        "null_q95": float(np.quantile(null_arr, 0.95)) if len(null_arr) else None,
        "p_value": pval,
        "n_perm": int(len(null_arr)),
        "passes_leak_test": bool(np.mean(null_arr) < 0.2) if len(null_arr) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["Berry Phase Rate"])
    ap.add_argument(
        "--oos-keys",
        nargs="+",
        default=[
            "2010_flash",
            "2011_euro",
            "2015_china",
            "2018_volmageddon",
            "2018_q4",
            "2019_repo",
            "2020_covid",
            "2022_rates",
            "2023_svb",
        ],
    )
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal-mode", default="global", choices=["global", "local"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = prepare_data()
    results = {
        m: leak_test(
            m, args.oos_keys, args.n_trials, args.seed, args.n_perm, data, args.normal_mode
        )
        for m in args.methods
    }
    out = Path(args.out) if args.out else OUTPUT_DIR / "leak_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "results": results, "timestamp": datetime.now().isoformat()}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    for m, r in results.items():
        logger.info(
            "%s: real=%.3f null_mean=%.3f p=%.3f passes=%s",
            m,
            r["real_nested_oos_median_d"],
            r["null_mean"],
            r["p_value"],
            r["passes_leak_test"],
        )
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
