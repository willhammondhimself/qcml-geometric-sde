"""Rigorous quantification of HPO/backtest overfitting for the nested protocol.

Pure functions (matrix/vector in → stats out) are separated from data-driven
drivers so the statistics are unit-testable without network or detectors.

Methods (see methodology brief / citations in docstrings):
* Overfitting curve — in-sample vs nested-OOS d across trial budgets.
* PBO via CSCV — Bailey, Borwein, López de Prado & Zhu (2017),
  "The Probability of Backtest Overfitting", J. Computational Finance 20(4).
* Deflated effect size — extreme-value (Gumbel) max-under-null à la Bailey &
  López de Prado (2014) Deflated Sharpe Ratio, adapted to Cohen's d, with an
  *effective* N from the trial-correlation eigenvalues (trials are correlated).
* Gap significance — paired bootstrap + sign-flip permutation on per-window gaps.
* Multiplicity — Holm (1979) FWER + Benjamini-Hochberg (1995) FDR over detectors.

Small-sample caveat (≈9-17 crises): PBO/deflation point estimates are coarse;
we report bootstrap bands and prefer resampling over asymptotic Gaussian forms.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import combinations
from math import e as EULER_E
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python experiments/...`

from experiments.evaluation import bh_fdr_correction, holm_bonferroni_correction  # noqa: E402

logger = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Pure statistics (no I/O, no detectors)
# --------------------------------------------------------------------------- #


def pbo_from_matrix(M: np.ndarray, n_partitions: int = 8, max_splits: int | None = None) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    Args:
        M: performance matrix, shape (n_configs, n_crises), higher = better
           (here Cohen's d). Columns assumed in chronological order.
        n_partitions: S, number of contiguous crisis blocks (forced even,
           clamped to n_crises).
    Returns dict with pbo, logits, n_splits, and the small-sample caveat fields.
    """
    M = np.asarray(M, dtype=float)
    n_configs, n_crises = M.shape
    keep = ~np.any(np.isnan(M), axis=1)  # CSCV needs complete config rows
    M = M[keep]
    n_configs = M.shape[0]
    if n_configs < 2 or n_crises < 4:
        return {"pbo": None, "reason": "insufficient configs/crises", "n_configs": int(n_configs)}

    S = min(n_partitions, n_crises)
    if S % 2 == 1:
        S -= 1
    # contiguous chronological blocks of column indices
    blocks = [b for b in np.array_split(np.arange(n_crises), S) if len(b)]
    S = len(blocks)
    if S < 2:
        return {"pbo": None, "reason": "too few blocks", "n_configs": int(n_configs)}

    logits = []
    combos = list(combinations(range(S), S // 2))
    if max_splits is not None and len(combos) > max_splits:
        combos = combos[:max_splits]
    eps = 1e-6
    for is_blocks in combos:
        is_cols = np.concatenate([blocks[b] for b in is_blocks])
        oos_cols = np.concatenate([blocks[b] for b in range(S) if b not in is_blocks])
        is_perf = np.median(M[:, is_cols], axis=1)
        oos_perf = np.median(M[:, oos_cols], axis=1)
        n_star = int(np.argmax(is_perf))  # IS-best config
        # OOS relative rank of the IS-best (1=worst .. n_configs=best)
        ranks = rankdata(oos_perf, method="average")
        omega = ranks[n_star] / (n_configs + 1)
        omega = min(max(omega, eps), 1 - eps)
        logits.append(float(np.log(omega / (1 - omega))))

    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits < 0)),  # P(IS-best below OOS median)
        "logit_mean": float(np.mean(logits)),
        "logit_q05": float(np.quantile(logits, 0.05)),
        "logit_q95": float(np.quantile(logits, 0.95)),
        "n_splits": int(len(logits)),
        "n_partitions": int(S),
        "n_configs": int(n_configs),
        "n_crises": int(n_crises),
        "caveat": "small panel → PBO is a qualitative red-flag, not a precise probability",
    }


def effective_n_trials(M: np.ndarray) -> float:
    """Effective independent-trial count from the config-correlation eigenvalues.

    n_eff = (Σλ)² / Σλ² of the config×config correlation matrix (participation
    ratio). Correlated trials → n_eff << n_configs, avoiding an over-conservative
    deflation bar.
    """
    M = np.asarray(M, dtype=float)
    keep = ~np.any(np.isnan(M), axis=1)
    M = M[keep]
    if M.shape[0] < 2 or M.shape[1] < 2:
        return float(max(M.shape[0], 1))
    C = np.corrcoef(M)  # configs × configs
    C = np.nan_to_num(C, nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    if lam.sum() <= 0:
        return float(M.shape[0])
    return float((lam.sum() ** 2) / np.sum(lam**2))


def deflate_d(observed_d: float, config_median_ds: np.ndarray, n_eff: float | None = None) -> dict:
    """Deflate a selected Cohen's d for the number of (effective) trials.

    Uses the Gumbel expected-maximum-under-null:
        E[max Z_n] ≈ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))
    scaled by the across-trial dispersion of d. Clamped ≥ 0. (Bailey & López de
    Prado 2014, adapted to an effect size.)
    """
    ds = np.asarray(config_median_ds, dtype=float)
    ds = ds[np.isfinite(ds)]
    N = float(len(ds)) if n_eff is None else float(n_eff)
    sigma = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
    if N < 2 or sigma == 0.0:
        return {"observed_d": observed_d, "expected_max_under_null": 0.0, "deflated_d": observed_d}
    emax_z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / N) + EULER_GAMMA * norm.ppf(1 - 1 / (N * EULER_E))
    expected_max = sigma * emax_z
    return {
        "observed_d": float(observed_d),
        "sigma_d_across_trials": sigma,
        "n_eff_trials": N,
        "expected_max_under_null": float(expected_max),
        "deflated_d": float(max(0.0, observed_d - expected_max)),
    }


def gap_stats(
    in_sample: dict, oos: dict, n_boot: int = 10000, n_perm: int = 5000, seed: int = 42
) -> dict:
    """Per-window (in-sample − OOS) gap: paired bootstrap CI + sign-flip permutation.

    in_sample/oos are {window: d_or_None}; aligned on windows present (non-None) in both.
    """
    rng = np.random.default_rng(seed)
    keys = [k for k in in_sample if in_sample.get(k) is not None and oos.get(k) is not None]
    gaps = np.array([in_sample[k] - oos[k] for k in keys], dtype=float)
    if len(gaps) < 2:
        return {"n_windows": int(len(gaps)), "mean_gap": None, "reason": "too few paired windows"}

    boot = np.array(
        [np.mean(rng.choice(gaps, size=len(gaps), replace=True)) for _ in range(n_boot)]
    )
    lo, hi = np.quantile(boot, [0.025, 0.975])

    observed = abs(np.mean(gaps))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(gaps)))
    perm_means = np.abs((signs * gaps).mean(axis=1))
    p = float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))

    return {
        "n_windows": int(len(gaps)),
        "gap_per_window": {k: float(in_sample[k] - oos[k]) for k in keys},
        "mean_gap": float(np.mean(gaps)),
        "gap_ci95": [float(lo), float(hi)],
        "paired_permutation_p": p,
        "significant": bool(p < 0.05),
    }


def multiplicity(pvalues_by_method: dict, alpha: float = 0.05) -> dict:
    """Holm (FWER) + Benjamini-Hochberg (FDR) over per-detector p-values."""
    methods = list(pvalues_by_method)
    pvec = [pvalues_by_method[m] for m in methods]
    holm_adj, holm_rej = holm_bonferroni_correction(pvec)
    bh_adj, bh_rej = bh_fdr_correction(pvec, alpha=alpha)
    return {
        m: {
            "raw_p": float(pvec[i]),
            "holm_adjusted_p": float(holm_adj[i]),
            "holm_rejected": bool(holm_rej[i]),
            "bh_adjusted_p": float(bh_adj[i]),
            "bh_rejected": bool(bh_rej[i]),
        }
        for i, m in enumerate(methods)
    }


# --------------------------------------------------------------------------- #
# Data-driven drivers (import walk_forward_hpo lazily to keep pure fns importable)
# --------------------------------------------------------------------------- #


def _wf():
    import experiments.walk_forward_hpo as wf

    return wf


def perf_matrix(method, oos_keys, n_trials, seed, data):
    """(configs, M, keys): M[i,j] = Cohen's d of trial-config i on crisis keys[j]."""
    import optuna

    wf = _wf()
    from experiments.optuna_hpo import SEARCH_SPACES

    cls, pfn = SEARCH_SPACES[method]["class"], SEARCH_SPACES[method]["params"]
    X_enriched, dates_enriched = data
    panel = {k: wf.ALL_CRISES[k] for k in oos_keys}

    def objective(trial):
        return wf.median_d_over(cls, pfn(trial), X_enriched, dates_enriched, panel)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    configs = [t.params for t in study.trials if t.values is not None]
    keys = wf.chronological_keys(oos_keys)
    M = np.full((len(configs), len(keys)), np.nan)
    for i, p in enumerate(configs):
        full = wf._full_params(method, p)
        for j, ck in enumerate(keys):
            M[i, j] = wf._D_FN(cls, full, X_enriched, dates_enriched, ck)
    return configs, M, keys


def overfitting_curve(method, trial_budgets=(8, 25, 50, 100), oos_keys=None, seed=42, data=None):
    wf = _wf()
    oos_keys = oos_keys or wf.DEFAULT_OOS_KEYS
    data = data or wf.prepare_data()
    X_enriched, dates_enriched = data
    curve = []
    for b in trial_budgets:
        nested = wf.run_nested(method, X_enriched, dates_enriched, oos_keys, b, seed)
        nested_med = wf._median([w["oos_d"] for w in nested.values()])
        in_sample, _ = wf.run_in_sample(method, X_enriched, dates_enriched, oos_keys, b, seed)
        in_med = wf._median(list(in_sample.values()))
        curve.append(
            {
                "trial_budget": int(b),
                "in_sample_median_d": in_med,
                "nested_oos_median_d": nested_med,
                "gap": (
                    (in_med - nested_med)
                    if (np.isfinite(in_med) and np.isfinite(nested_med))
                    else None
                ),
            }
        )
    # OLS slope of gap vs log2(budget)
    pts = [(np.log2(c["trial_budget"]), c["gap"]) for c in curve if c["gap"] is not None]
    slope = None
    if len(pts) >= 2:
        xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        slope = float(np.polyfit(xs, ys, 1)[0])
    return {"method": method, "curve": curve, "gap_slope_per_log2_trial": slope}


def run_suite(
    methods,
    oos_keys=None,
    trial_budgets=(8, 25),
    n_trials=25,
    seed=42,
    out=None,
    data=None,
    use_cache=True,
):
    wf = _wf()
    oos_keys = oos_keys or wf.DEFAULT_OOS_KEYS
    data = data or wf.prepare_data()
    if use_cache:
        from experiments.hpo_cache import CohensDCache

        CohensDCache().attach_data(data[0]).install()

    per_method = {}
    for method in methods:
        logger.info("overfitting suite: %s", method)
        curve = overfitting_curve(method, trial_budgets, oos_keys, seed, data)
        configs, M, keys = perf_matrix(method, oos_keys, n_trials, seed, data)
        nested = wf.run_nested(method, *data, oos_keys, n_trials, seed)
        in_sample, _ = wf.run_in_sample(method, *data, oos_keys, n_trials, seed)
        observed = wf._median([w["oos_d"] for w in nested.values()])
        config_medians = np.nanmedian(M, axis=1) if M.size else np.array([])
        per_method[method] = {
            "curve": curve,
            "pbo": pbo_from_matrix(M),
            "deflated": deflate_d(observed, config_medians, n_eff=effective_n_trials(M)),
            "gap_significance": gap_stats(in_sample, {k: v["oos_d"] for k, v in nested.items()}),
        }

    results = {
        "config": {
            "methods": methods,
            "oos_keys": oos_keys,
            "trial_budgets": list(trial_budgets),
            "n_trials": n_trials,
            "seed": seed,
        },
        "per_method": per_method,
        "timestamp": datetime.now().isoformat(),
    }
    out = (
        Path(out)
        if out
        else (
            wf.ROOT
            / "experiments"
            / "outputs"
            / "regime_detection"
            / "overfitting"
            / "overfitting_stats.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote %s", out)
    return results


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--methods", nargs="+", default=["Berry Phase Rate", "Multi-Lag Fidelity", "Spectral Gap"]
    )
    ap.add_argument("--oos-keys", nargs="+", default=None)
    ap.add_argument("--trial-budgets", nargs="+", type=int, default=[8, 25])
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_suite(
        args.methods, args.oos_keys, tuple(args.trial_budgets), args.n_trials, args.seed, args.out
    )


if __name__ == "__main__":
    main()
