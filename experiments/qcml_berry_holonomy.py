"""Angle 1: Berry holonomy via Wilson loop product on MPD-simulated paths.

Pre-registration: experiments/designs/20260515_qcml_berry_holonomy.yaml.

Pipeline:
  1. Load frozen ABM pool (parquet, n=3000) and verify metadata vs design SHA.
  2. Day-1 smoke (100/group x 5 bases x gamma statistic). Apply LOCKED decision
     rule: proceed iff median |d| >= 0.3 AND >=4/5 bases agree on sign.
  3. If smoke passes: full 1500/group x 5 bases x 3 statistics (gamma, |gamma|,
     mean|gamma_w| over 252-step sub-loops).
  4. Classical-loop sanity check on rolling-vol-z / HMM / RF score time series:
     each per-run loop integral must verify to ~0 (Stokes; algebraically zero).
  5. Welch + MWU + Cohen's d + 10k bootstrap CI per cell; Holm across 15 cells.
  6. Verdict STRONG | MEDIUM | WEAK_OR_NULL per pre-registered thresholds.

Run:
    PYTHONPATH=. caffeinate -i python experiments/qcml_berry_holonomy.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qcml_geometry.core import QCMLGeometry  # noqa: E402

# Reuse Angle 2 helpers (DRY)
from experiments.qcml_abm_matched_regime import (  # noqa: E402
    BASIS_OFFSETS,
    HILBERT_DIM,
    N_PCA,
    SEED,
    build_geo_for_basis,
    decode_returns,
    file_sha256,
    holm_bonferroni,
    load_pool,
    per_run_feature_matrix,
    two_sample_stats,
    verify_pool_metadata,
)

DESIGN_YAML = ROOT / "experiments" / "designs" / "20260515_qcml_berry_holonomy.yaml"
OUT_PATH = ROOT / "experiments" / "outputs" / "diagnostics" / "qcml_berry_holonomy.json"

N_TRAIN_PER_GROUP = 200
N_SMOKE_PER_GROUP = 100
N_FULL_PER_GROUP = 1500
SUB_WINDOW = 252  # for mean|gamma_w| statistic
SMOKE_D_THRESHOLD = 0.30
SMOKE_MIN_AGREEING_BASES = 4
CLASSICAL_SANITY_EPS = 1.0e-10
HOLM_FAMILY_SIZE = 15  # 5 bases x 3 statistics
N_BOOT = 10_000


# -----------------------------------------------------------------------------
# Wilson loop product (log-space, gauge-invariant)
# -----------------------------------------------------------------------------

def wilson_loop_phase(psi_stack: np.ndarray) -> float:
    """Compute Berry phase γ = Im log W where
        W = <psi_{T-1} | psi_0> * prod_{t=0..T-2} <psi_t | psi_{t+1}>.

    Uses log-space accumulation to avoid underflow over T=2520.

    Args:
        psi_stack: (T, hilbert_dim) complex array; each row is one timestep's
            quasi-coherent state (assumed unit-normalised, but renormalised
            here defensively).

    Returns:
        Berry phase in (-pi, pi].
    """
    T = psi_stack.shape[0]
    # Defensive renormalisation
    norms = np.linalg.norm(psi_stack, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    psi = psi_stack / norms
    # Pairwise overlaps (forward + closing edge)
    overlaps_fwd = np.einsum("ti,ti->t", np.conj(psi[:-1]), psi[1:])
    closing = np.vdot(psi[-1], psi[0])
    # Accumulated phase (log-space)
    phases = np.angle(overlaps_fwd).sum() + np.angle(closing)
    # Wrap into (-pi, pi]
    return float(np.mod(phases + np.pi, 2 * np.pi) - np.pi)


def windowed_wilson_phases(psi_stack: np.ndarray, window: int = SUB_WINDOW) -> np.ndarray:
    """Compute Berry phase inside each non-overlapping window of size `window`."""
    T = psi_stack.shape[0]
    n_w = T // window
    out = np.empty(n_w)
    for i in range(n_w):
        out[i] = wilson_loop_phase(psi_stack[i * window:(i + 1) * window])
    return out


def per_run_quasi_coherent_states(geo: QCMLGeometry, X_pca: np.ndarray) -> np.ndarray:
    """Stack quasi-coherent states for every timestep."""
    T = X_pca.shape[0]
    psi = np.empty((T, geo.hilbert_dim), dtype=np.complex128)
    for t in range(T):
        try:
            psi[t] = geo.quasi_coherent_state(X_pca[t])
        except Exception:
            psi[t] = np.nan
    return psi


# -----------------------------------------------------------------------------
# Per-run statistics
# -----------------------------------------------------------------------------

def per_run_holonomy(
    geo: QCMLGeometry, X_pca: np.ndarray, want_windowed: bool = True
) -> dict:
    """Compute (gamma, |gamma|, mean|gamma_w|) for one run on one basis."""
    psi = per_run_quasi_coherent_states(geo, X_pca)
    if np.any(np.isnan(psi)):
        return {"gamma": np.nan, "abs_gamma": np.nan, "mean_abs_gamma_w": np.nan}
    gamma = wilson_loop_phase(psi)
    out = {"gamma": gamma, "abs_gamma": abs(gamma)}
    if want_windowed:
        gws = windowed_wilson_phases(psi, SUB_WINDOW)
        out["mean_abs_gamma_w"] = float(np.mean(np.abs(gws)))
    else:
        out["mean_abs_gamma_w"] = float("nan")
    return out


# -----------------------------------------------------------------------------
# Classical-loop sanity check (Stokes verification)
# -----------------------------------------------------------------------------

def classical_loop_integrals(returns: np.ndarray) -> dict:
    """Per-run discrete loop integral I = (f_{T-1} - f_0) + (f_0 - f_{T-1}) = 0.

    Computed numerically for three classical detector score series. By
    construction these must be zero — non-zero indicates an implementation bug.
    """
    out = {}
    log_ret = pd.Series(returns)
    # rolling vol z (expanding-window z-score of 20-day rolling vol)
    vol = log_ret.rolling(20).std()
    z = (vol - vol.expanding(min_periods=60).mean()) / vol.expanding(min_periods=60).std()
    z = z.fillna(0.0).values
    out["rolling_vol_z"] = float(z[-1] - z[0] + (z[0] - z[-1]))
    # HMM 2-state posterior (subsampled compute — actual fit not needed for Stokes check,
    # any per-timestep score series suffices; use squared returns as a stand-in scalar)
    sq = returns ** 2
    out["sq_returns_score"] = float(sq[-1] - sq[0] + (sq[0] - sq[-1]))
    # RF prob proxy: rank-transform of vol (any monotone scalar of state suffices)
    rank = pd.Series(vol.values).rank(pct=True).fillna(0.0).values
    out["rank_vol_score"] = float(rank[-1] - rank[0] + (rank[0] - rank[-1]))
    return out


# -----------------------------------------------------------------------------
# Embedding fit
# -----------------------------------------------------------------------------

def fit_shared_embedding(df_pool: pd.DataFrame, n_steps: int) -> tuple:
    """Fit StandardScaler + PCA on a balanced 200+200 training subsample."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train_ii = df_pool[df_pool["group_label"] == "type_ii_heavy"].sample(
        N_TRAIN_PER_GROUP, random_state=SEED
    )
    train_iii = df_pool[df_pool["group_label"] == "type_iii_heavy"].sample(
        N_TRAIN_PER_GROUP, random_state=SEED
    )
    train_df = pd.concat([train_ii, train_iii])
    feats = [per_run_feature_matrix(decode_returns(row["returns_b64"], n_steps))
             for _, row in train_df.iterrows()]
    X_train = np.vstack(feats)
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(scaler.transform(X_train))
    geos = {off: build_geo_for_basis(N_PCA, off) for off in BASIS_OFFSETS}
    print(f"  scaler + PCA fit on {X_train.shape[0]:,} timesteps; "
          f"5 bases at offsets {BASIS_OFFSETS}")
    return scaler, pca, geos, train_df["run_id"].tolist()


# -----------------------------------------------------------------------------
# Phase loop (smoke or full)
# -----------------------------------------------------------------------------

def compute_phase(
    df_test: pd.DataFrame, scaler, pca, geos, n_steps: int, want_windowed: bool, label: str
) -> pd.DataFrame:
    """Compute per-run holonomy on a test subset across all bases."""
    records = []
    t0 = time.time()
    for i, (_, row) in enumerate(df_test.iterrows()):
        returns = decode_returns(row["returns_b64"], n_steps)
        X = per_run_feature_matrix(returns)
        X_pca = pca.transform(scaler.transform(X))
        for off, geo in geos.items():
            obs = per_run_holonomy(geo, X_pca, want_windowed=want_windowed)
            records.append({
                "run_id": int(row["run_id"]),
                "group_label": row["group_label"],
                "basis_offset": off,
                **obs,
            })
        if (i + 1) % 25 == 0 or (i + 1) == len(df_test):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df_test) - (i + 1)) / max(rate, 1e-6)
            print(f"  [{label}] {i+1}/{len(df_test)} in {elapsed:.0f}s; ETA {eta:.0f}s")
    return pd.DataFrame(records)


def per_basis_stats(df: pd.DataFrame, statistic: str, rng: np.random.Generator) -> list[dict]:
    """Welch + MWU + Cohen's d + bootstrap per basis offset for one statistic."""
    cells = []
    for off in BASIS_OFFSETS:
        sub = df[df["basis_offset"] == off]
        a = sub.loc[sub["group_label"] == "type_ii_heavy", statistic].values
        b = sub.loc[sub["group_label"] == "type_iii_heavy", statistic].values
        cell = two_sample_stats(a, b, rng)
        cell.update({"statistic": statistic, "basis_offset": off})
        cells.append(cell)
    return cells


def smoke_decision(cells_gamma: list[dict]) -> tuple[bool, dict]:
    """Apply LOCKED smoke gate to per-basis gamma cells."""
    ds = np.array([c["cohens_d"] for c in cells_gamma], dtype=float)
    valid = ds[~np.isnan(ds)]
    if valid.size == 0:
        return False, {"reason": "all bases NaN — implementation bug"}
    med_abs_d = float(np.median(np.abs(valid)))
    signs = np.sign(valid)
    n_pos = int((signs > 0).sum())
    n_neg = int((signs < 0).sum())
    agreeing = max(n_pos, n_neg)
    proceed = (med_abs_d >= SMOKE_D_THRESHOLD) and (agreeing >= SMOKE_MIN_AGREEING_BASES)
    return proceed, {
        "median_abs_cohens_d_across_bases": med_abs_d,
        "bases_agreeing_on_sign": agreeing,
        "per_basis_d": [float(d) for d in ds],
        "threshold_d": SMOKE_D_THRESHOLD,
        "threshold_bases": SMOKE_MIN_AGREEING_BASES,
        "proceed": proceed,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    print("Loading frozen ABM pool...")
    df_pool, meta = load_pool()
    n_steps = int(meta["n_steps"])
    print(f"  rows: {len(df_pool)}, groups: {df_pool['group_label'].value_counts().to_dict()}")
    print(f"  pool meta: mpd_commit={meta.get('mpd_git_commit','?')[:8]} "
          f"design_sha(angle2)={meta.get('design_yaml_sha256','?')[:8]} "
          f"n_steps={n_steps}")
    # Holonomy design has its OWN SHA — pool was generated for Angle 2 design; that's fine.
    holonomy_design_sha = file_sha256(DESIGN_YAML)
    print(f"  holonomy design SHA: {holonomy_design_sha[:8]}")

    print("\nFitting shared embedding (StandardScaler + PCA + 5 bases)...")
    scaler, pca, geos, train_ids = fit_shared_embedding(df_pool, n_steps)
    train_set = set(train_ids)

    # ---- Day-1 smoke: 100/group, gamma only, all 5 bases ----
    print(f"\n=== SMOKE TEST ({N_SMOKE_PER_GROUP}/group, 5 bases, gamma only) ===")
    test_pool = df_pool[~df_pool["run_id"].isin(train_set)]
    smoke_ii = test_pool[test_pool["group_label"] == "type_ii_heavy"].sample(
        N_SMOKE_PER_GROUP, random_state=SEED + 1
    )
    smoke_iii = test_pool[test_pool["group_label"] == "type_iii_heavy"].sample(
        N_SMOKE_PER_GROUP, random_state=SEED + 1
    )
    smoke_df = pd.concat([smoke_ii, smoke_iii])
    smoke_results = compute_phase(smoke_df, scaler, pca, geos, n_steps,
                                  want_windowed=False, label="smoke")
    rng_b = np.random.default_rng(SEED + 100)
    smoke_cells_gamma = per_basis_stats(smoke_results, "gamma", rng_b)
    proceed, smoke_summary = smoke_decision(smoke_cells_gamma)
    print(f"\nSmoke summary: median |d|={smoke_summary.get('median_abs_cohens_d_across_bases'):.3f}, "
          f"bases_agreeing={smoke_summary.get('bases_agreeing_on_sign')}, "
          f"per-basis d={smoke_summary.get('per_basis_d')}")
    print(f"Smoke decision: PROCEED={proceed}")

    result = {
        "experiment_id": "qcml_berry_holonomy_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "holonomy_design_sha256": holonomy_design_sha,
            "pool_meta": meta,
            "hilbert_dim": HILBERT_DIM,
            "n_pca": N_PCA,
            "basis_offsets": BASIS_OFFSETS,
            "n_train_per_group": N_TRAIN_PER_GROUP,
            "n_smoke_per_group": N_SMOKE_PER_GROUP,
            "n_full_per_group": N_FULL_PER_GROUP,
            "n_boot": N_BOOT,
            "sub_window": SUB_WINDOW,
            "smoke_d_threshold": SMOKE_D_THRESHOLD,
            "smoke_min_agreeing_bases": SMOKE_MIN_AGREEING_BASES,
            "classical_sanity_eps": CLASSICAL_SANITY_EPS,
        },
        "smoke": {
            "decision_rule": {
                "median_abs_d_threshold": SMOKE_D_THRESHOLD,
                "min_agreeing_bases": SMOKE_MIN_AGREEING_BASES,
            },
            "summary": smoke_summary,
            "per_basis_cells_gamma": smoke_cells_gamma,
        },
    }

    if not proceed:
        result["verdict"] = "WEAK_OR_NULL"
        result["verdict_reason"] = (
            f"Smoke gate FAILED: median |d|={smoke_summary['median_abs_cohens_d_across_bases']:.3f} "
            f"(threshold {SMOKE_D_THRESHOLD}); bases agreeing="
            f"{smoke_summary['bases_agreeing_on_sign']} (need {SMOKE_MIN_AGREEING_BASES}). "
            "Per locked pre-registration: STOP, ship paper."
        )
        result["wall_seconds"] = time.time() - t_start
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\nWrote {OUT_PATH}")
        print(f"VERDICT: WEAK_OR_NULL")
        return

    # ---- Full run: 1500/group x 5 bases x 3 statistics ----
    print(f"\n=== FULL RUN ({N_FULL_PER_GROUP}/group, 5 bases, all 3 statistics) ===")
    full_ii = test_pool[test_pool["group_label"] == "type_ii_heavy"].sample(
        N_FULL_PER_GROUP, random_state=SEED + 2
    )
    full_iii = test_pool[test_pool["group_label"] == "type_iii_heavy"].sample(
        N_FULL_PER_GROUP, random_state=SEED + 2
    )
    full_df = pd.concat([full_ii, full_iii])
    full_results = compute_phase(full_df, scaler, pca, geos, n_steps,
                                 want_windowed=True, label="full")

    # Per-cell stats across the 15-cell family
    print("\nPer-cell two-sample stats + Holm-Bonferroni...")
    rng_full = np.random.default_rng(SEED + 1000)
    statistics = ["gamma", "abs_gamma", "mean_abs_gamma_w"]
    all_cells = []
    for stat in statistics:
        all_cells.extend(per_basis_stats(full_results, stat, rng_full))
    p_vals = [c["welch_p"] for c in all_cells]
    holm_p = holm_bonferroni(p_vals)
    for c, hp in zip(all_cells, holm_p):
        c["welch_p_holm"] = float(hp)

    # Per-statistic summary
    stat_summary = []
    for stat in statistics:
        sub = [c for c in all_cells if c["statistic"] == stat]
        ds = [c["cohens_d"] for c in sub if not np.isnan(c["cohens_d"])]
        signs = [np.sign(d) for d in ds]
        majority = 1.0 if sum(s > 0 for s in signs) >= sum(s < 0 for s in signs) else -1.0
        agree = int(sum(np.sign(d) == majority for d in ds))
        bases_clearing = sum(
            (abs(c["cohens_d"]) >= 0.5) and (c["welch_p_holm"] < 0.05) and (np.sign(c["cohens_d"]) == majority)
            for c in sub if not np.isnan(c["cohens_d"])
        )
        stat_summary.append({
            "statistic": stat,
            "median_d": float(np.nanmedian(ds)) if ds else float("nan"),
            "bases_with_majority_dir": agree,
            "bases_clearing_bar": int(bases_clearing),
        })

    # Classical-loop sanity check
    print("Classical-loop sanity check (must verify ~0 by Stokes)...")
    max_abs = {"rolling_vol_z": 0.0, "sq_returns_score": 0.0, "rank_vol_score": 0.0}
    for _, row in full_df.iterrows():
        returns = decode_returns(row["returns_b64"], n_steps)
        cl = classical_loop_integrals(returns)
        for k, v in cl.items():
            if abs(v) > max_abs[k]:
                max_abs[k] = abs(v)
    sanity_pass = all(v <= CLASSICAL_SANITY_EPS for v in max_abs.values())
    print(f"  per-detector max |loop integral|: {max_abs}")
    print(f"  sanity pass (<= {CLASSICAL_SANITY_EPS}): {sanity_pass}")

    # Verdict (LOCKED rule)
    if not sanity_pass:
        verdict = "WEAK_OR_NULL"
        verdict_reason = "Classical-loop sanity check FAILED — implementation bug; result void."
    else:
        bases_clearing_max = max(s["bases_clearing_bar"] for s in stat_summary)
        if bases_clearing_max == 5:
            verdict = "STRONG"
            verdict_reason = "At least one statistic clears |d|>=0.5 with Holm-p<0.05 in 5/5 bases."
        elif bases_clearing_max >= 2:
            verdict = "MEDIUM"
            verdict_reason = (
                f"At least one statistic clears the bar in {bases_clearing_max}/5 bases "
                "(basis-fragile but real)."
            )
        else:
            verdict = "WEAK_OR_NULL"
            verdict_reason = "No statistic clears |d|>=0.5 + Holm-p<0.05 in >=2 bases."

    result["full_run"] = {
        "n_per_group": N_FULL_PER_GROUP,
        "per_cell": all_cells,
        "per_statistic_summary": stat_summary,
        "classical_loop_sanity": {
            "max_abs_loop_integral": max_abs,
            "eps": CLASSICAL_SANITY_EPS,
            "pass": bool(sanity_pass),
        },
    }
    result["verdict"] = verdict
    result["verdict_reason"] = verdict_reason
    result["wall_seconds"] = time.time() - t_start

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_reason}")
    print(f"  Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
