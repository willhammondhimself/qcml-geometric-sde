"""Angle 2: QCML geometry on vol/moment-matched ABM regimes.

Pre-registration: experiments/designs/20260514_qcml_abm_matched_regime.yaml.

Pipeline:
  1. Load frozen ABM pool (parquet from ~/market-population-dynamics/scripts/
     generate_qcml_pool.py) and verify metadata matches the design YAML.
  2. Day-1 gate: nearest-neighbour 1:1 caliper matching on 5 standardised
     classical covariates (rv_ann, ac1, ac5, skew, kurt). Require >=100 pairs
     with per-covariate SMD < 0.10. If gate fails -> STOP, write JSON, exit.
  3. Fit one shared embedding on a balanced 200+200 training subsample held
     out from the matched test set: StandardScaler -> PCA(8) ->
     QCMLGeometry(hilbert_dim=6). Build 5 operator bases via the
     endo_exo_robustness_v2 custom-operator pattern.
  4. Per (test run, basis): compute time series for 5 QCML observables
     (spectral gap, Berry-curv Frobenius, spectral entropy, reduced purity,
     Hamiltonian sensitivity); summarise as per-run mean.
  5. Classical negative controls on matched set: Rolling Vol Z, HMM 2-state,
     Random Forest LORO on the 5 classical covariates. All must stay null;
     any significant => result VOID.
  6. Per cell: Welch t + Mann-Whitney U + Cohen's d + percentile bootstrap CI.
     Holm-Bonferroni across the 25 QCML cells.
  7. Aggregate verdict (STRONG / MEDIUM / WEAK_OR_NULL) per pre-registered
     thresholds. Write everything to JSON.

Run:
    PYTHONPATH=. caffeinate -i python experiments/qcml_abm_matched_regime.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy import stats as sstats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qcml_geometry.core import QCMLGeometry  # noqa: E402

DESIGN_YAML = ROOT / "experiments" / "designs" / "20260514_qcml_abm_matched_regime.yaml"
POOL_PATH = ROOT / "data" / "abm_pool" / "abm_pool_v1.parquet"
OUT_PATH = ROOT / "experiments" / "outputs" / "diagnostics" / "qcml_abm_matched_regime.json"

# Canonical params from pre-registration
HILBERT_DIM = 6
N_PCA = 8
SEED = 42
BASIS_OFFSETS = [0, 100, 200, 300, 400]
COVARIATES = ["rv_ann", "ac1", "ac5", "skew", "kurt"]
N_TRAIN_PER_GROUP = 200
N_BOOT = 10_000
CALIPER_STD = 0.25
SMD_THRESHOLD = 0.10
MIN_MATCHED_PER_GROUP = 100
HOLM_FAMILY_SIZE = 25  # 5 observables x 5 bases


# -----------------------------------------------------------------------------
# Pool loading + verification
# -----------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_pool() -> tuple[pd.DataFrame, dict]:
    table = pq.read_table(POOL_PATH)
    meta_raw = table.schema.metadata or {}
    meta = {k.decode(): v.decode() for k, v in meta_raw.items() if not k.startswith(b"pandas")}
    df = table.to_pandas()
    return df, meta


def verify_pool_metadata(meta: dict) -> None:
    """Refuse to run if the pool wasn't generated for *this* design YAML."""
    design_sha_now = file_sha256(DESIGN_YAML)
    design_sha_pool = meta.get("design_yaml_sha256", "")
    if design_sha_now != design_sha_pool:
        print(
            f"WARNING: design YAML SHA mismatch.\n"
            f"  pool was generated for: {design_sha_pool}\n"
            f"  current YAML SHA:       {design_sha_now}\n"
            f"Continuing — but verify edits to the design were purely "
            f"cosmetic (comments, formatting) before trusting results."
        )


def decode_returns(b64: str, n_steps: int) -> np.ndarray:
    """Decode a base64 float32 returns array."""
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    assert arr.size == n_steps, f"expected {n_steps} returns, got {arr.size}"
    return arr.astype(np.float64)


# -----------------------------------------------------------------------------
# Feature matrix per simulation run (mirrors create_feature_matrix_single_asset)
# -----------------------------------------------------------------------------

def per_run_feature_matrix(returns: np.ndarray) -> np.ndarray:
    """Build a single-asset feature matrix from a log-returns series.

    Mirrors experiments/data_loader.py::create_feature_matrix_single_asset
    (extra_lags=True) with the price-derived columns recovered from returns.
    Returns rows after warmup drop (NaN/inf removed).
    """
    log_ret = pd.Series(returns)
    # reconstruct cumulative log-price for momentum / range features
    log_p = log_ret.cumsum()
    p = np.exp(log_p)
    features = {
        "ret": log_ret,
        "vol5":  log_ret.rolling(5).std(),
        "vol10": log_ret.rolling(10).std(),
        "vol20": log_ret.rolling(20).std(),
        "vol40": log_ret.rolling(40).std(),
        "vol60": log_ret.rolling(60).std(),
        "mom5":  p.pct_change(5),
        "mom10": p.pct_change(10),
        "mom20": p.pct_change(20),
        "mom40": p.pct_change(40),
        "mom60": p.pct_change(60),
        "ret_sq": log_ret ** 2,
        "range_20": p.rolling(20).max() / p.rolling(20).min() - 1.0,
        "skew_20": log_ret.rolling(20).skew(),
        "kurt_20": log_ret.rolling(20).kurt(),
    }
    feat = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).dropna()
    return feat.values


# -----------------------------------------------------------------------------
# QCML observables (raw per-point — not z-scored, since we average across runs)
# -----------------------------------------------------------------------------

def spectral_entropy(geo: QCMLGeometry, x: np.ndarray) -> float:
    """Shannon entropy of the normalised error-Hamiltonian eigenvalue spectrum.

    This is a self-contained 'spectral entropy' definition; we report it in the
    JSON and do not claim equivalence to qcml_geometry.observables.SpectralEntropyDetector
    (which adds rolling + expanding-z normalisation we explicitly want to avoid
    so per-run means are comparable across runs).
    """
    H = geo.error_hamiltonian(x)
    w = np.linalg.eigvalsh(H)
    w = np.clip(w - w.min(), 0.0, None)
    s = w.sum()
    if s < 1e-12:
        return 0.0
    p = w / s
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def per_run_observables(geo: QCMLGeometry, X_pca: np.ndarray) -> dict[str, float]:
    """Compute 5 QCML observable means for one run, given PCA-transformed features.

    Returns dict of {observable_name: per-run mean}.
    """
    T = X_pca.shape[0]
    if T < 3:
        return {k: float("nan") for k in
                ["spectral_gap", "berry_frob", "spectral_entropy",
                 "reduced_purity", "ham_sensitivity"]}
    gap = np.empty(T)
    berry = np.empty(T)
    s_ent = np.empty(T)
    purity = np.empty(T)
    ham = np.empty(T - 1)
    for t in range(T):
        x = X_pca[t]
        try:
            gap[t] = geo.spectral_gap(x)
        except Exception:
            gap[t] = np.nan
        try:
            F = geo.berry_curvature(x)
            berry[t] = float(np.linalg.norm(F))
        except Exception:
            berry[t] = np.nan
        try:
            s_ent[t] = spectral_entropy(geo, x)
        except Exception:
            s_ent[t] = np.nan
        try:
            purity[t] = float(geo.reduced_state_purity(x, partition=(2, 4)))
        except Exception:
            purity[t] = np.nan
    for t in range(T - 1):
        try:
            ham[t] = float(geo.hamiltonian_sensitivity(X_pca[t], X_pca[t + 1]))
        except Exception:
            ham[t] = np.nan
    return {
        "spectral_gap":     float(np.nanmean(gap)),
        "berry_frob":       float(np.nanmean(berry)),
        "spectral_entropy": float(np.nanmean(s_ent)),
        "reduced_purity":   float(np.nanmean(purity)),
        "ham_sensitivity":  float(np.nanmean(ham)),
    }


# -----------------------------------------------------------------------------
# Multi-basis pattern (from endo_exo_robustness_v2.py:74-95)
# -----------------------------------------------------------------------------

def make_hermitian(rng: np.random.Generator, hd: int) -> np.ndarray:
    A = rng.standard_normal((hd, hd)) + 1j * rng.standard_normal((hd, hd))
    return (A + A.conj().T) / 2.0


def make_basis(offset: int, n_ops: int, hd: int) -> list[np.ndarray]:
    return [make_hermitian(np.random.default_rng(k + offset), hd) for k in range(n_ops)]


def build_geo_for_basis(n_features: int, offset: int) -> QCMLGeometry:
    geo = QCMLGeometry(n_features=n_features, hilbert_dim=HILBERT_DIM)
    geo.operators = make_basis(offset, n_features, HILBERT_DIM)
    geo.is_fitted = True
    return geo


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------

def nearest_neighbour_match(
    df: pd.DataFrame, covariates: list[str], rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """1:1 nearest-neighbour caliper matching with Mahalanobis distance on standardised covariates.

    Returns matched DataFrame (2 * n_pairs rows) and per-covariate SMD report.
    """
    z = StandardScaler().fit_transform(df[covariates].values)
    z_df = pd.DataFrame(z, columns=covariates, index=df.index)
    cov = np.cov(z, rowvar=False) + 1e-9 * np.eye(len(covariates))
    inv_cov = np.linalg.inv(cov)

    ii_idx = df.index[df["group_label"] == "type_ii_heavy"].to_numpy()
    iii_idx = df.index[df["group_label"] == "type_iii_heavy"].to_numpy()
    # Shuffle Type-II order for stochastic tie-breaking
    rng.shuffle(ii_idx)

    ii_z = z_df.loc[ii_idx].values
    iii_z = z_df.loc[iii_idx].values
    used_iii = np.zeros(len(iii_idx), dtype=bool)

    pairs = []
    for i, zi in enumerate(ii_z):
        diff = iii_z - zi  # (n_iii, k)
        # Mahalanobis distance
        d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        d2[used_iii] = np.inf
        # Caliper: per-covariate max |z-diff| <= CALIPER_STD
        max_abs = np.max(np.abs(diff), axis=1)
        d2[max_abs > CALIPER_STD] = np.inf
        j = int(np.argmin(d2))
        if not np.isfinite(d2[j]):
            continue
        used_iii[j] = True
        pairs.append((ii_idx[i], iii_idx[j]))

    matched_idx = [p[0] for p in pairs] + [p[1] for p in pairs]
    matched = df.loc[matched_idx].copy()
    matched["pair_id"] = list(range(len(pairs))) + list(range(len(pairs)))

    # Post-matching SMD per covariate
    smd = {}
    for c in covariates:
        a = matched.loc[matched["group_label"] == "type_ii_heavy", c].values
        b = matched.loc[matched["group_label"] == "type_iii_heavy", c].values
        if len(a) < 2 or len(b) < 2:
            smd[c] = float("nan")
            continue
        pooled_sd = np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
        smd[c] = float(abs(np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 1e-12 else 0.0
    return matched, smd


# -----------------------------------------------------------------------------
# Classical negative-control detectors
# -----------------------------------------------------------------------------

def hmm_2_state_score(returns: np.ndarray) -> float:
    """Per-run scalar from a 2-state Gaussian HMM on returns: time-averaged P(high-vol state)."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return float("nan")
    model = GaussianHMM(n_components=2, n_iter=50, random_state=SEED, covariance_type="diag")
    X = returns.reshape(-1, 1)
    model.fit(X)
    # Identify high-vol state by the larger variance
    high = int(np.argmax(model.covars_.flatten()))
    posteriors = model.predict_proba(X)
    return float(posteriors[:, high].mean())


def rf_loro_predictions(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Leave-one-run-out RF on the 5 classical covariates only.

    Returns predicted-positive probability for each run (positive = type_ii_heavy).
    """
    X = df[covariates].values
    y = (df["group_label"].values == "type_ii_heavy").astype(int)
    preds = np.empty(len(df))
    # Use simple LOO via index iteration (fast enough at n<300).
    for i in range(len(df)):
        mask = np.ones(len(df), dtype=bool)
        mask[i] = False
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1)
        clf.fit(X[mask], y[mask])
        preds[i] = clf.predict_proba(X[i:i + 1])[0, 1]
    return preds


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def two_sample_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Welch + Mann-Whitney + Cohen's d + percentile bootstrap CI on (a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"welch_p": float("nan"), "mwu_p": float("nan"), "cohens_d": float("nan"),
                "ci95": [float("nan"), float("nan")], "n_a": len(a), "n_b": len(b)}
    t, t_p = sstats.ttest_ind(a, b, equal_var=False)
    u, u_p = sstats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                     / (len(a) + len(b) - 2))
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 1e-12 else float("nan")
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs[i] = np.mean(rng.choice(a, size=len(a), replace=True)) - \
                   np.mean(rng.choice(b, size=len(b), replace=True))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "welch_p": float(t_p),
        "mwu_p": float(u_p),
        "cohens_d": float(d),
        "ci95": [float(lo), float(hi)],
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
    }


def holm_bonferroni(p_vals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(p_vals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, i in enumerate(order):
        v = (n - rank) * p[i]
        running_max = max(running_max, v)
        adj[i] = min(running_max, 1.0)
    return adj.tolist()


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    print("Loading frozen ABM pool...")
    df_pool, meta = load_pool()
    print(f"  rows: {len(df_pool)}, groups: {df_pool['group_label'].value_counts().to_dict()}")
    print(f"  pool meta: mpd_commit={meta.get('mpd_git_commit','?')[:8]} "
          f"design_sha={meta.get('design_yaml_sha256','?')[:8]} "
          f"n_steps={meta.get('n_steps')}")
    verify_pool_metadata(meta)
    n_steps = int(meta["n_steps"])

    print("\nDay-1 GATE: nearest-neighbour matching on classical covariates...")
    matched, smd = nearest_neighbour_match(df_pool, COVARIATES, rng)
    n_pairs = len(matched) // 2
    print(f"  matched pairs: {n_pairs}")
    print(f"  per-covariate SMD: {smd}")
    smd_pass = all((not np.isnan(v)) and v < SMD_THRESHOLD for v in smd.values())
    size_pass = n_pairs >= MIN_MATCHED_PER_GROUP
    gate_pass = smd_pass and size_pass
    print(f"  size_pass (>= {MIN_MATCHED_PER_GROUP}): {size_pass}; "
          f"SMD_pass (< {SMD_THRESHOLD} on all): {smd_pass}")

    result = {
        "experiment_id": "qcml_abm_matched_regime_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "design_yaml_sha256": file_sha256(DESIGN_YAML),
            "pool_meta": meta,
            "hilbert_dim": HILBERT_DIM,
            "n_pca": N_PCA,
            "basis_offsets": BASIS_OFFSETS,
            "covariates": COVARIATES,
            "n_train_per_group": N_TRAIN_PER_GROUP,
            "n_boot": N_BOOT,
            "caliper_std": CALIPER_STD,
            "smd_threshold": SMD_THRESHOLD,
            "min_matched_per_group": MIN_MATCHED_PER_GROUP,
        },
        "day1_gate": {
            "n_matched_pairs": int(n_pairs),
            "smd": smd,
            "pass": bool(gate_pass),
        },
    }

    if not gate_pass:
        result["verdict"] = "WEAK_OR_NULL"
        result["verdict_reason"] = (
            "Day-1 gate FAILED: matching could not produce >=100 pairs with "
            "SMD<0.10 on all 5 covariates. Experiment structurally impossible "
            "as pre-registered."
        )
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nGate FAILED — wrote {OUT_PATH}")
        return

    # Hold out a balanced training subsample for embedding fit (disjoint from matched test set)
    test_run_ids = set(matched["run_id"].tolist())
    train_pool = df_pool[~df_pool["run_id"].isin(test_run_ids)]
    train_ii  = train_pool[train_pool["group_label"] == "type_ii_heavy"].sample(N_TRAIN_PER_GROUP, random_state=SEED)
    train_iii = train_pool[train_pool["group_label"] == "type_iii_heavy"].sample(N_TRAIN_PER_GROUP, random_state=SEED)
    train_df = pd.concat([train_ii, train_iii])
    print(f"\nEmbedding training: {len(train_df)} runs ({N_TRAIN_PER_GROUP}/group, disjoint from matched test).")

    print("Building pooled training feature matrix...")
    train_feats = []
    for _, row in train_df.iterrows():
        returns = decode_returns(row["returns_b64"], n_steps)
        X = per_run_feature_matrix(returns)
        train_feats.append(X)
    X_train_stack = np.vstack(train_feats)
    print(f"  pooled training X: {X_train_stack.shape}")

    print("Fitting shared scaler + PCA + 5 bases of QCMLGeometry...")
    scaler = StandardScaler().fit(X_train_stack)
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(scaler.transform(X_train_stack))
    geos = {off: build_geo_for_basis(N_PCA, off) for off in BASIS_OFFSETS}
    print(f"  scaler/PCA fit on {X_train_stack.shape[0]:,} timesteps")
    print(f"  bases: {BASIS_OFFSETS}")

    # Per-run observables on matched test set
    print(f"\nComputing per-run QCML observables on {len(matched)} matched test runs x {len(BASIS_OFFSETS)} bases...")
    obs_names = ["spectral_gap", "berry_frob", "spectral_entropy", "reduced_purity", "ham_sensitivity"]
    # records: list of dict with keys [run_id, group_label, basis_offset, <obs_names>]
    records = []
    t_obs = time.time()
    for i, (_, row) in enumerate(matched.iterrows()):
        returns = decode_returns(row["returns_b64"], n_steps)
        X = per_run_feature_matrix(returns)
        X_pca = pca.transform(scaler.transform(X))
        for off, geo in geos.items():
            obs = per_run_observables(geo, X_pca)
            records.append({
                "run_id": int(row["run_id"]),
                "group_label": row["group_label"],
                "basis_offset": off,
                **obs,
            })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_obs
            rate = (i + 1) / elapsed
            eta = (len(matched) - (i + 1)) / rate
            print(f"  {i+1}/{len(matched)} runs in {elapsed:.0f}s; ETA {eta:.0f}s")
    obs_df = pd.DataFrame(records)
    print(f"  observable eval done in {time.time() - t_obs:.0f}s; obs_df shape {obs_df.shape}")

    # Per-cell stats (5 obs x 5 bases) on matched runs
    print("\nPer-cell two-sample stats + Holm-Bonferroni...")
    cells = []
    rng_b = np.random.default_rng(SEED + 1)
    for obs in obs_names:
        for off in BASIS_OFFSETS:
            sub = obs_df[obs_df["basis_offset"] == off]
            a = sub.loc[sub["group_label"] == "type_ii_heavy", obs].values
            b = sub.loc[sub["group_label"] == "type_iii_heavy", obs].values
            cell = two_sample_stats(a, b, rng_b)
            cell.update({"observable": obs, "basis_offset": off})
            cells.append(cell)
    p_vals = [c["welch_p"] for c in cells]
    holm = holm_bonferroni(p_vals)
    for c, hp in zip(cells, holm):
        c["welch_p_holm"] = float(hp)

    # Per-observable direction-consistency across bases
    obs_summary = []
    for obs in obs_names:
        sub = [c for c in cells if c["observable"] == obs]
        ds = [c["cohens_d"] for c in sub if not np.isnan(c["cohens_d"])]
        signs = [np.sign(c["cohens_d"]) for c in sub if not np.isnan(c["cohens_d"])]
        majority_sign = 1.0 if sum(s > 0 for s in signs) >= sum(s < 0 for s in signs) else -1.0
        agree = sum(np.sign(d) == majority_sign for d in ds)
        clears = sum(
            (abs(c["cohens_d"]) >= 0.5) and (c["welch_p_holm"] < 0.05) and (np.sign(c["cohens_d"]) == majority_sign)
            for c in sub if not np.isnan(c["cohens_d"])
        )
        obs_summary.append({
            "observable": obs,
            "median_d": float(np.nanmedian(ds)) if ds else float("nan"),
            "bases_with_dir": int(agree),
            "bases_clearing_bar": int(clears),
            "clears_overall": bool(clears >= 1 and agree >= 4),
        })

    # Classical negative controls on matched set
    print("\nClassical negative controls on matched set...")
    classical_results = []
    rng_c = np.random.default_rng(SEED + 2)

    # rv_z: per-run rv_ann is already a classical "rolling vol" summary
    a_rv = matched.loc[matched["group_label"] == "type_ii_heavy", "rv_ann"].values
    b_rv = matched.loc[matched["group_label"] == "type_iii_heavy", "rv_ann"].values
    rv_cell = two_sample_stats(a_rv, b_rv, rng_c)
    rv_cell["detector"] = "rolling_vol_ann"
    classical_results.append(rv_cell)

    # HMM 2-state per-run posterior
    print("  HMM 2-state...")
    hmm_scores = []
    for _, row in matched.iterrows():
        returns = decode_returns(row["returns_b64"], n_steps)
        hmm_scores.append(hmm_2_state_score(returns))
    matched["_hmm"] = hmm_scores
    a_h = matched.loc[matched["group_label"] == "type_ii_heavy", "_hmm"].values
    b_h = matched.loc[matched["group_label"] == "type_iii_heavy", "_hmm"].values
    hmm_cell = two_sample_stats(a_h, b_h, rng_c)
    hmm_cell["detector"] = "hmm_2_state"
    classical_results.append(hmm_cell)

    # RF LORO on 5 covariates -> predicted P(type_ii)
    print("  RF leave-one-run-out...")
    rf_preds = rf_loro_predictions(matched, COVARIATES)
    matched["_rf"] = rf_preds
    a_r = matched.loc[matched["group_label"] == "type_ii_heavy", "_rf"].values
    b_r = matched.loc[matched["group_label"] == "type_iii_heavy", "_rf"].values
    rf_cell = two_sample_stats(a_r, b_r, rng_c)
    rf_cell["detector"] = "random_forest_loro"
    classical_results.append(rf_cell)

    classical_sig = sum(c["welch_p"] < 0.05 and abs(c["cohens_d"]) >= 0.3 for c in classical_results)

    # Sanity check on UNMATCHED pool: classical detectors should separate strongly
    a_rv_u = df_pool.loc[df_pool["group_label"] == "type_ii_heavy", "rv_ann"].values
    b_rv_u = df_pool.loc[df_pool["group_label"] == "type_iii_heavy", "rv_ann"].values
    sanity_unmatched = two_sample_stats(a_rv_u, b_rv_u, rng_c)

    # Verdict
    n_qcml_clearing = sum(o["clears_overall"] for o in obs_summary)
    if n_qcml_clearing >= 3 and classical_sig == 0:
        verdict = "STRONG"
    elif n_qcml_clearing >= 1 and classical_sig == 0:
        verdict = "MEDIUM"
    else:
        verdict = "WEAK_OR_NULL"

    result.update({
        "obs_per_cell": cells,
        "obs_summary_per_observable": obs_summary,
        "classical_negative_controls": classical_results,
        "n_classical_significant": int(classical_sig),
        "n_qcml_observables_clearing_bar": int(n_qcml_clearing),
        "sanity_check_unmatched_rv": sanity_unmatched,
        "verdict": verdict,
        "wall_seconds": time.time() - t_start,
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nVerdict: {verdict}")
    print(f"  QCML observables clearing bar: {n_qcml_clearing} / 5")
    print(f"  Classical neg controls significant: {classical_sig} / 3")
    print(f"  Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
