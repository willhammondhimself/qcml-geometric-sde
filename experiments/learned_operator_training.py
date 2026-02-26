"""
Gradient-Descent Operator Learning for QCML Regime Detection.

Trains Hermitian operators to maximize crisis/non-crisis separability
measured by Cohen's d. Uses leave-one-crisis-out (LOCO) cross-validation
to avoid overfitting.

Evaluation uses the FULL detector pipeline (scaler → PCA → operators →
rolling window → z-score normalization), producing effect sizes comparable
to the main paper results (d ~ 0.8–1.2).

Usage:
    python experiments/learned_operator_training.py              # LOCO on 4 crises, berry only
    python experiments/learned_operator_training.py --full       # all 16 crises, all 3 detectors
    python experiments/learned_operator_training.py --stability  # 20-seed variance analysis
    python experiments/learned_operator_training.py --detector berry  # single detector

Outputs:
    experiments/outputs/operator_learning/
        operator_learning_{detector}_{timestamp}.json
        loco_comparison_{detector}.pdf
        stability_{detector}.pdf
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
    BaseRegimeDetector,
)
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent / "outputs" / "operator_learning"
QUICK_CRISES = ["2008_gfc", "2020_covid", "2022_rates", "2018_volmageddon"]
ALL_DETECTOR_TYPES = ["berry", "qfi", "fidelity"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


DETECTOR_CLASSES = {
    "berry": BerryPhaseRateDetector,
    "qfi": QFIDeterminantDetector,
    "fidelity": MultiLagFidelityDetector,
}


# ══════════════════════════════════════════════════════════════════════
# Learned Operator Module
# ══════════════════════════════════════════════════════════════════════


class HermitianParameterization:
    """Parameterize Hermitian operators via real parameters.

    A d×d Hermitian matrix has d² real degrees of freedom.
    We store: d diagonal (real) + d(d-1)/2 upper-triangle (complex = 2 real).
    Total: d² parameters.
    """

    def __init__(self, d):
        self.d = d
        self.n_params = d * d

    def params_to_hermitian(self, params):
        """Convert real parameter vector to Hermitian matrix."""
        d = self.d
        H = np.zeros((d, d), dtype=np.complex128)

        # diagonal
        H[np.diag_indices(d)] = params[:d]

        # upper triangle (real + imag parts)
        idx = d
        for i in range(d):
            for j in range(i + 1, d):
                H[i, j] = params[idx] + 1j * params[idx + 1]
                H[j, i] = params[idx] - 1j * params[idx + 1]
                idx += 2

        return H

    def hermitian_to_params(self, H):
        """Convert Hermitian matrix to real parameter vector."""
        d = self.d
        params = np.zeros(self.n_params)
        params[:d] = np.real(np.diag(H))

        idx = d
        for i in range(d):
            for j in range(i + 1, d):
                params[idx] = np.real(H[i, j])
                params[idx + 1] = np.imag(H[i, j])
                idx += 2

        return params


def _compute_scores_fast(operators, X_pca, hilbert_dim, detector_type="berry"):
    """Fast raw-score computation for gradient descent objective.

    Bypasses the full detector pipeline (no scaler/PCA/rolling/z-score)
    since X_pca is already preprocessed. Used only inside the gradient
    descent loop for speed.
    """
    geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
    geo.set_operators(operators)

    T = X_pca.shape[0]

    if detector_type == "berry":
        berry = np.empty(T)
        for t in range(T):
            berry[t] = geo.berry_curvature_2d(X_pca[t], indices=(0, 1))
        return np.abs(np.diff(berry, prepend=berry[0]))

    elif detector_type == "qfi":
        logdet = np.empty(T)
        for t in range(T):
            g = geo.quantum_metric(X_pca[t])
            eigs = np.linalg.eigvalsh(g)
            pos = eigs[eigs > 1e-10]
            logdet[t] = np.sum(np.log(pos)) if len(pos) > 0 else -20.0
        return np.abs(logdet)

    elif detector_type == "fidelity":
        states = []
        for t in range(T):
            states.append(geo.quasi_coherent_state(X_pca[t]))
        infidelity = np.zeros(T)
        for t in range(1, T):
            overlap = np.abs(np.vdot(states[t], states[t - 1]))
            infidelity[t] = 1.0 - overlap**2
        return infidelity

    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")


def _compute_scores_full_pipeline(
    operators, X_enriched, hilbert_dim, detector_type="berry",
):
    """Compute regime scores using the full detector pipeline with custom operators.

    Creates a real detector with custom_operators, runs the complete
    scaler → PCA → operator → rolling window → z-score pipeline.
    Returns z-scored regime scores comparable to the main paper results.

    Args:
        operators: List of Hermitian matrices (hilbert_dim, hilbert_dim).
        X_enriched: Enriched feature matrix (T, d) — output of build_enriched_features().
        hilbert_dim: Hilbert space dimension.
        detector_type: 'berry', 'qfi', or 'fidelity'.

    Returns:
        scores: 1-D array of z-scored regime scores, length T.
    """
    det_cls = DETECTOR_CLASSES[detector_type]
    # n_pca_components must match len(operators) — one operator per PCA dimension
    det = det_cls(
        hilbert_dim=hilbert_dim,
        custom_operators=operators,
        n_pca_components=len(operators),
    )
    det.fit(X_enriched)
    return det.compute_regime_scores(X_enriched)


def _cohens_d(crisis_scores, normal_scores):
    """Cohen's d (absolute) between two score distributions."""
    n1, n2 = len(crisis_scores), len(normal_scores)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(crisis_scores, ddof=1)
    var2 = np.var(normal_scores, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return abs(np.mean(crisis_scores) - np.mean(normal_scores)) / pooled_std


def _eval_full_pipeline(operators, X_enriched, dates, hilbert_dim, detector_type,
                        ho_mask, normal_mask):
    """Evaluate operators via full pipeline, return Cohen's d on held-out crisis."""
    scores = _compute_scores_full_pipeline(
        operators, X_enriched, hilbert_dim, detector_type,
    )
    valid = ~np.isnan(scores)
    return _cohens_d(
        scores[ho_mask & valid],
        scores[normal_mask & valid],
    )


def learn_operators_gradient(
    X_pca,
    dates,
    crisis_keys,
    hilbert_dim=8,
    n_operators=8,
    detector_type="berry",
    n_steps=200,
    lr=0.01,
    seed=42,
    init_method="random",
):
    """Learn Hermitian operators by gradient descent to maximize Cohen's d.

    Uses fast raw-score objective for gradient steps (no rolling/z-score).
    Final operators should be evaluated with _compute_scores_full_pipeline().

    Args:
        X_pca: Preprocessed feature matrix (T, n_features) for fast scoring.
        dates: DatetimeIndex aligned to X_pca.
        crisis_keys: List of crisis keys to optimize over.
        hilbert_dim: Hilbert space dimension.
        n_operators: Number of operators to learn.
        detector_type: 'berry', 'qfi', or 'fidelity'.
        n_steps: Optimization steps.
        lr: Learning rate.
        seed: Random seed for initialization.
        init_method: 'random' or 'pca_inspired' for initial operators.

    Returns:
        operators: List of learned Hermitian operators.
        history: List of Cohen's d values during optimization.
    """
    rng = np.random.default_rng(seed)
    param = HermitianParameterization(hilbert_dim)

    # build crisis mask
    crisis_mask = np.zeros(len(dates), dtype=bool)
    for key in crisis_keys:
        info = ALL_CRISES[key]
        s, e = pd.Timestamp(info["start"]), pd.Timestamp(info["end"])
        crisis_mask |= (dates >= s) & (dates <= e)

    # initialize operators
    geo_init = QCMLGeometry(n_features=n_operators, hilbert_dim=hilbert_dim)
    geo_init.fit_operators(X_pca, method=init_method)
    operators = [op.copy() for op in geo_init.operators[:n_operators]]

    # flatten all operators into a single parameter vector
    all_params = np.concatenate([param.hermitian_to_params(op) for op in operators])
    params_per_op = param.n_params

    # subsample for efficiency (use ~200 points from each class)
    max_per_class = 200
    crisis_idx = np.where(crisis_mask)[0]
    normal_idx = np.where(~crisis_mask)[0]
    if len(crisis_idx) > max_per_class:
        crisis_idx = rng.choice(crisis_idx, max_per_class, replace=False)
    if len(normal_idx) > max_per_class:
        normal_idx = rng.choice(normal_idx, max_per_class, replace=False)
    eval_idx = np.sort(np.concatenate([crisis_idx, normal_idx]))
    eval_crisis = np.isin(eval_idx, np.where(crisis_mask)[0])

    X_eval = X_pca[eval_idx]

    def objective(p):
        """Negative Cohen's d (we minimize)."""
        ops = []
        for k in range(n_operators):
            start = k * params_per_op
            ops.append(param.params_to_hermitian(p[start : start + params_per_op]))

        scores = _compute_scores_fast(ops, X_eval, hilbert_dim, detector_type)
        valid = ~np.isnan(scores)
        if valid.sum() < 10:
            return 0.0
        crisis_s = scores[eval_crisis & valid]
        normal_s = scores[~eval_crisis & valid]
        return -_cohens_d(crisis_s, normal_s)

    history = []
    eps_grad = 0.01

    logger.info(f"  Optimizing {n_operators} operators ({detector_type}), "
                f"{n_steps} steps, lr={lr}")

    best_params = all_params.copy()
    best_obj = objective(all_params)
    history.append(-best_obj)

    for step in range(n_steps):
        n_grad_params = min(10, len(all_params))
        grad_idx = rng.choice(len(all_params), n_grad_params, replace=False)

        grad = np.zeros_like(all_params)
        for i in grad_idx:
            p_plus = all_params.copy()
            p_minus = all_params.copy()
            p_plus[i] += eps_grad
            p_minus[i] -= eps_grad
            grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * eps_grad)

        all_params -= lr * grad

        obj = objective(all_params)
        history.append(-obj)

        if obj < best_obj:
            best_obj = obj
            best_params = all_params.copy()

        if (step + 1) % 50 == 0:
            logger.info(f"    Step {step + 1}/{n_steps}: d = {-obj:.4f} "
                        f"(best = {-best_obj:.4f})")

    # reconstruct best operators
    learned_ops = []
    for k in range(n_operators):
        start = k * params_per_op
        learned_ops.append(
            param.params_to_hermitian(best_params[start : start + params_per_op])
        )

    return learned_ops, history


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════


def evaluate_operators_loco(
    X_enriched, X_pca, dates, all_crisis_keys, hilbert_dim=8,
    detector_type="berry", n_steps=150, lr=0.01,
):
    """Leave-one-crisis-out evaluation of learned vs baseline operators.

    Gradient descent uses X_pca (fast). Final evaluation uses X_enriched
    through the full detector pipeline (scaler → PCA → rolling → z-score).

    Args:
        X_enriched: Enriched feature matrix for full-pipeline evaluation.
        X_pca: Pre-PCA'd matrix for fast gradient descent.
        dates: DatetimeIndex aligned with both matrices.
        all_crisis_keys: List of crisis keys for LOCO.
        hilbert_dim: Hilbert space dimension.
        detector_type: 'berry', 'qfi', or 'fidelity'.
        n_steps: Gradient descent steps per LOCO fold.
        lr: Learning rate.

    Returns:
        dict mapping method -> {crisis_key: Cohen's d}.
    """
    results = {
        "random": {},
        "pca_inspired": {},
        "learned_from_random": {},
        "learned_from_pca": {},
    }

    for held_out in all_crisis_keys:
        logger.info(f"\n  LOCO: held out = {held_out}")
        train_crises = [k for k in all_crisis_keys if k != held_out]

        # held-out crisis mask
        info = ALL_CRISES[held_out]
        s, e = pd.Timestamp(info["start"]), pd.Timestamp(info["end"])
        ho_mask = (dates >= s) & (dates <= e)
        normal_mask = ~ho_mask

        # exclude other crises from "normal"
        for k in all_crisis_keys:
            if k != held_out:
                ki = ALL_CRISES[k]
                normal_mask &= ~((dates >= pd.Timestamp(ki["start"])) &
                                 (dates <= pd.Timestamp(ki["end"])))

        # baseline 1: random operators — generate and evaluate via full pipeline
        np.random.seed(42)
        geo_rand = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
        geo_rand.fit_operators(X_pca, method="random")
        d_rand = _eval_full_pipeline(
            geo_rand.operators, X_enriched, dates, hilbert_dim,
            detector_type, ho_mask, normal_mask,
        )
        results["random"][held_out] = d_rand

        # baseline 2: pca_inspired operators
        geo_pca = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
        geo_pca.fit_operators(X_pca, method="pca_inspired")
        d_pca = _eval_full_pipeline(
            geo_pca.operators, X_enriched, dates, hilbert_dim,
            detector_type, ho_mask, normal_mask,
        )
        results["pca_inspired"][held_out] = d_pca

        # learned 1: initialized from random
        ops_lr, _ = learn_operators_gradient(
            X_pca, dates, train_crises, hilbert_dim=hilbert_dim,
            detector_type=detector_type, n_steps=n_steps, lr=lr,
            seed=42, init_method="random",
        )
        d_lr = _eval_full_pipeline(
            ops_lr, X_enriched, dates, hilbert_dim,
            detector_type, ho_mask, normal_mask,
        )
        results["learned_from_random"][held_out] = d_lr

        # learned 2: initialized from pca_inspired
        ops_lp, _ = learn_operators_gradient(
            X_pca, dates, train_crises, hilbert_dim=hilbert_dim,
            detector_type=detector_type, n_steps=n_steps, lr=lr,
            seed=42, init_method="pca_inspired",
        )
        d_lp = _eval_full_pipeline(
            ops_lp, X_enriched, dates, hilbert_dim,
            detector_type, ho_mask, normal_mask,
        )
        results["learned_from_pca"][held_out] = d_lp

        logger.info(
            f"    {held_out}: random d={d_rand:.3f}, pca d={d_pca:.3f}, "
            f"learned_rand d={d_lr:.3f}, learned_pca d={d_lp:.3f}"
        )

    return results


def stability_analysis(
    X_enriched, X_pca, dates, crisis_keys, hilbert_dim=8, detector_type="berry",
    n_seeds=20, n_steps=100, lr=0.01,
):
    """Run learned operator training with different seeds to measure stability.

    Evaluation uses the full detector pipeline for comparable effect sizes.

    Returns:
        dict mapping method -> list of Cohen's d values across seeds.
    """
    crisis_mask = np.zeros(len(dates), dtype=bool)
    for key in crisis_keys:
        info = ALL_CRISES[key]
        s, e = pd.Timestamp(info["start"]), pd.Timestamp(info["end"])
        crisis_mask |= (dates >= s) & (dates <= e)
    normal_mask = ~crisis_mask

    results = {"random": [], "pca_inspired": [], "learned": []}

    for seed in range(n_seeds):
        # random baseline (varies with seed)
        np.random.seed(seed)
        geo = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
        geo.fit_operators(X_pca, method="random")

        scores_r = _compute_scores_full_pipeline(
            geo.operators, X_enriched, hilbert_dim, detector_type,
        )
        valid_r = ~np.isnan(scores_r)
        d_r = _cohens_d(scores_r[crisis_mask & valid_r], scores_r[normal_mask & valid_r])
        results["random"].append(d_r)

        # learned (varies with seed via initialization + gradient noise)
        ops, _ = learn_operators_gradient(
            X_pca, dates, crisis_keys, hilbert_dim=hilbert_dim,
            detector_type=detector_type, n_steps=n_steps, lr=lr,
            seed=seed, init_method="random",
        )
        scores_l = _compute_scores_full_pipeline(
            ops, X_enriched, hilbert_dim, detector_type,
        )
        valid_l = ~np.isnan(scores_l)
        d_l = _cohens_d(scores_l[crisis_mask & valid_l], scores_l[normal_mask & valid_l])
        results["learned"].append(d_l)

        if (seed + 1) % 5 == 0:
            logger.info(f"  Stability: {seed + 1}/{n_seeds} seeds complete")

    # pca_inspired is deterministic
    geo_pca = QCMLGeometry(n_features=X_pca.shape[1], hilbert_dim=hilbert_dim)
    geo_pca.fit_operators(X_pca, method="pca_inspired")
    scores_pca = _compute_scores_full_pipeline(
        geo_pca.operators, X_enriched, hilbert_dim, detector_type,
    )
    valid_pca = ~np.isnan(scores_pca)
    d_pca = _cohens_d(
        scores_pca[crisis_mask & valid_pca], scores_pca[normal_mask & valid_pca]
    )
    results["pca_inspired"] = [d_pca] * n_seeds

    return results


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════


def plot_loco_comparison(results, detector_type, out_dir):
    """Bar chart comparing 4 operator methods across held-out crises."""
    methods = list(results.keys())
    crises = list(results[methods[0]].keys())

    fig, ax = plt.subplots(figsize=(max(10, len(crises) * 1.2), 5))

    x = np.arange(len(crises))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, method in enumerate(methods):
        vals = [results[method][c] for c in crises]
        ax.bar(x + i * width, vals, width, label=method.replace("_", " ").title(),
               color=colors[i], alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([ALL_CRISES[c]["label"] for c in crises],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cohen's d (held-out crisis)")
    ax.set_title(f"LOCO Operator Comparison — {detector_type.title()} Detector")
    ax.legend(fontsize=8)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / f"loco_comparison_{detector_type}.pdf")
    fig.savefig(out_dir / f"loco_comparison_{detector_type}.png")
    plt.close(fig)


def plot_stability(results, detector_type, out_dir):
    """Box plots of Cohen's d across seeds for each method."""
    fig, ax = plt.subplots(figsize=(6, 5))

    data = [results["random"], results["learned"], results["pca_inspired"]]
    labels = ["Random", "Learned", "PCA-Inspired"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Operator Stability ({detector_type.title()}, {len(results['random'])} seeds)")

    for i, (d, label) in enumerate(zip(data, labels)):
        mean_val = np.mean(d)
        std_val = np.std(d)
        cv = std_val / (mean_val + 1e-12) * 100
        ax.text(i + 1, max(d) + 0.02, f"CV={cv:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"stability_{detector_type}.pdf")
    fig.savefig(out_dir / f"stability_{detector_type}.png")
    plt.close(fig)


def plot_summary_table(all_detector_results, out_dir):
    """Summary bar chart: median d across crises for each method × detector."""
    detectors = list(all_detector_results.keys())
    methods = list(all_detector_results[detectors[0]].keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(detectors))
    width = 0.18
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, method in enumerate(methods):
        medians = []
        for det in detectors:
            vals = list(all_detector_results[det][method].values())
            medians.append(np.median(vals))
        ax.bar(x + i * width, medians, width,
               label=method.replace("_", " ").title(),
               color=colors[i], alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([d.title() for d in detectors])
    ax.set_ylabel("Median Cohen's d")
    ax.set_title("Learned Operators: Median Effect Size by Detector")
    ax.legend(fontsize=8)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "loco_summary_all_detectors.pdf")
    fig.savefig(out_dir / "loco_summary_all_detectors.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Learned operator training")
    parser.add_argument("--full", action="store_true", help="All 16 crises, all 3 detectors")
    parser.add_argument("--stability", action="store_true", help="Run 20-seed stability analysis")
    parser.add_argument("--detector", type=str, default=None,
                        choices=["berry", "qfi", "fidelity"],
                        help="Single detector (default: all 3 with --full)")
    parser.add_argument("--hilbert-dim", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crises = list(ALL_CRISES.keys()) if args.full else QUICK_CRISES

    # determine which detectors to run
    if args.detector:
        detector_types = [args.detector]
    elif args.full:
        detector_types = ALL_DETECTOR_TYPES
    else:
        detector_types = ["berry"]

    # ── data preparation ─────────────────────────────────────────────
    logger.info("Preparing data...")
    prices = fetch_data(["SPY", "DIA"], "2005-01-01", "2025-06-30")
    close = prices["close"].unstack("symbol")
    X_raw, dates_raw = create_feature_matrix(close)

    # build enriched features for full-pipeline evaluation
    X_enriched = BaseRegimeDetector.build_enriched_features(X_raw, lookback=20)
    dates = dates_raw[19:]  # trim dates to match enriched features
    assert len(X_enriched) == len(dates), (
        f"Length mismatch: X_enriched={len(X_enriched)}, dates={len(dates)}"
    )

    # build X_pca for fast gradient descent (same preprocessing as before)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=min(8, X_scaled.shape[1]))
    X_pca_full = pca.fit_transform(X_scaled)

    # soft normalization
    norms = np.linalg.norm(X_pca_full, axis=1, keepdims=True)
    median_norm = np.median(norms)
    X_pca_full = X_pca_full / (norms + median_norm)

    # trim X_pca to match enriched features (drop first 19 rows)
    X_pca = X_pca_full[19:]

    # filter crises to those overlapping the data date range
    date_min, date_max = dates.min(), dates.max()
    valid_crises = []
    for k in crises:
        cs, ce = pd.Timestamp(ALL_CRISES[k]["start"]), pd.Timestamp(ALL_CRISES[k]["end"])
        if ce >= date_min and cs <= date_max:
            valid_crises.append(k)
        else:
            logger.info(f"  Skipping {k} (outside data range {date_min.date()}–{date_max.date()})")
    crises = valid_crises

    logger.info(f"Data ready: T={len(dates)}, enriched={X_enriched.shape[1]} features, "
                f"pca={X_pca.shape[1]} components, {len(crises)} crises in range")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stability:
        # ── stability analysis ───────────────────────────────────────
        for det_type in detector_types:
            logger.info(f"\n=== Stability Analysis ({det_type}, 20 seeds) ===")
            stab_results = stability_analysis(
                X_enriched, X_pca, dates, crises,
                hilbert_dim=args.hilbert_dim,
                detector_type=det_type,
                n_seeds=20,
                n_steps=args.n_steps,
                lr=args.lr,
            )

            stab_output = {
                "timestamp": datetime.now().isoformat(),
                "detector": det_type,
                "config": {
                    "hilbert_dim": args.hilbert_dim,
                    "n_steps": args.n_steps,
                    "lr": args.lr,
                    "crises": crises,
                    "n_seeds": 20,
                },
                "stability": {
                    method: {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "cv": float(np.std(vals) / (np.mean(vals) + 1e-12)),
                        "values": [float(v) for v in vals],
                    }
                    for method, vals in stab_results.items()
                },
            }

            plot_stability(stab_results, det_type, OUTPUT_DIR)

            results_path = OUTPUT_DIR / f"stability_{det_type}_{timestamp}.json"
            with open(results_path, "w") as f:
                json.dump(stab_output, f, indent=2, default=str)

            logger.info(f"Stability results ({det_type}):")
            for method, vals in stab_results.items():
                logger.info(f"  {method}: mean={np.mean(vals):.3f}, "
                            f"std={np.std(vals):.3f}, "
                            f"CV={np.std(vals) / (np.mean(vals) + 1e-12) * 100:.1f}%")
    else:
        # ── LOCO evaluation ──────────────────────────────────────────
        all_detector_results = {}

        for det_type in detector_types:
            logger.info(f"\n=== LOCO Evaluation: {det_type} ({len(crises)} crises) ===")
            loco_results = evaluate_operators_loco(
                X_enriched, X_pca, dates, crises,
                hilbert_dim=args.hilbert_dim,
                detector_type=det_type,
                n_steps=args.n_steps,
                lr=args.lr,
            )
            all_detector_results[det_type] = loco_results
            plot_loco_comparison(loco_results, det_type, OUTPUT_DIR)

            # per-detector summary
            logger.info(f"\n--- {det_type.title()} LOCO Summary ---")
            for method, per_crisis in loco_results.items():
                vals = list(per_crisis.values())
                logger.info(f"  {method}: median d={np.median(vals):.3f}, "
                            f"mean d={np.mean(vals):.3f}")

            # save per-detector results
            det_output = {
                "timestamp": datetime.now().isoformat(),
                "detector": det_type,
                "config": {
                    "hilbert_dim": args.hilbert_dim,
                    "n_steps": args.n_steps,
                    "lr": args.lr,
                    "crises": crises,
                },
                "loco": loco_results,
            }
            det_path = OUTPUT_DIR / f"operator_learning_{det_type}_{timestamp}.json"
            with open(det_path, "w") as f:
                json.dump(det_output, f, indent=2, default=str)
            logger.info(f"  Saved to {det_path}")

        # cross-detector summary plot
        if len(all_detector_results) > 1:
            plot_summary_table(all_detector_results, OUTPUT_DIR)

        # grand summary
        logger.info("\n=== Grand Summary (Median Cohen's d) ===")
        header = f"{'Method':<25}"
        for det in detector_types:
            header += f" {det:>10}"
        logger.info(header)
        for method in ["random", "pca_inspired", "learned_from_random", "learned_from_pca"]:
            row = f"  {method:<23}"
            for det in detector_types:
                vals = list(all_detector_results[det][method].values())
                row += f" {np.median(vals):>10.3f}"
            logger.info(row)


if __name__ == "__main__":
    main()
