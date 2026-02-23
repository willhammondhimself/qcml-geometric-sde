"""
Modern statistical comparison framework for multi-method regime detection evaluation.

Implements:
    build_d_matrix                        — Extract d-value matrix from comparison JSON
    compute_model_confidence_set          — Hansen-Lunde-Nason (2011) Model Confidence Set
    bayesian_signed_rank                  — Benavoli et al. (2017) Bayesian signed-rank test
    generate_cd_diagram                   — Critical difference diagram via autorank
    bootstrap_rank_cis                    — Bootstrap confidence intervals for mean ranks
    compute_oracle_and_complementarity    — Oracle score and method complementarity analysis
    compute_win_rates                     — Pairwise win-rate matrix
    run_statistical_comparison            — Full pipeline (CLI entry point)

Usage:
    python experiments/statistical_comparison.py <comparison_results.json>
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import rankdata, spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A. Model Confidence Set (Hansen, Lunde, Nason 2011)
# ---------------------------------------------------------------------------


def compute_model_confidence_set(d_matrix, alpha=0.10, n_bootstrap=10000, seed=42):
    """Compute the Model Confidence Set via bootstrap range-statistic elimination.

    Hansen, Lunde, Nason (2011): identifies the set of methods that are
    statistically indistinguishable from the best. Uses the Range statistic
    with a bootstrap elimination procedure.

    Higher d-values are better. The method with the lowest mean d is the
    candidate for elimination at each step.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.
        alpha: Significance level for elimination (default 0.10).
        n_bootstrap: Number of bootstrap resamples (default 10000).
        seed: Random seed for reproducibility.

    Returns:
        List of method indices in the Model Confidence Set.
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    n_crises, n_methods = d_matrix.shape
    rng = np.random.default_rng(seed)

    active = list(range(n_methods))

    while len(active) > 1:
        d_active = d_matrix[:, active]
        n_active = len(active)

        # Compute mean d per method (over crises)
        mean_d = d_active.mean(axis=0)

        # Range statistic: max absolute pairwise mean difference
        range_stat = 0.0
        for i in range(n_active):
            for j in range(i + 1, n_active):
                diff = abs(mean_d[i] - mean_d[j])
                range_stat = max(range_stat, diff)

        # Bootstrap: resample crises, compute centered range statistic
        # Center each method's d-values around zero for the null distribution
        centered = d_active - mean_d[np.newaxis, :]
        boot_count = 0

        for _ in range(n_bootstrap):
            boot_idx = rng.choice(n_crises, size=n_crises, replace=True)
            boot_d = centered[boot_idx]
            boot_means = boot_d.mean(axis=0)

            boot_range = 0.0
            for i in range(n_active):
                for j in range(i + 1, n_active):
                    diff = abs(boot_means[i] - boot_means[j])
                    boot_range = max(boot_range, diff)

            if boot_range >= range_stat:
                boot_count += 1

        p_value = boot_count / n_bootstrap

        if p_value >= alpha:
            # Cannot reject equal performance — stop, current set is the MCS
            break

        # Eliminate the method with the lowest mean d (worst performer)
        worst_local_idx = int(np.argmin(mean_d))
        active.pop(worst_local_idx)

    return active


# ---------------------------------------------------------------------------
# B. Bayesian Signed-Rank Test (Benavoli et al. 2017)
# ---------------------------------------------------------------------------


def bayesian_signed_rank(d_a, d_b, rope=0.1):
    """Bayesian signed-rank test for paired comparison of two methods.

    Uses baycomp.two_on_multiple when available, otherwise falls back to
    a count-based heuristic.

    Args:
        d_a: 1-D array of d-values for method A (one per crisis).
        d_b: 1-D array of d-values for method B (one per crisis).
        rope: Region of practical equivalence width (default 0.1).

    Returns:
        Dict with keys: p_a_better, p_equivalent, p_b_better.
    """
    d_a = np.asarray(d_a, dtype=float)
    d_b = np.asarray(d_b, dtype=float)

    try:
        import baycomp

        probs = baycomp.two_on_multiple(d_a, d_b, rope=rope)
        # baycomp returns (p_left, p_rope, p_right) when rope > 0
        # p_left = P(first > second), p_rope = P(equivalent), p_right = P(second > first)
        p_left, p_rope, p_right = probs
        return {
            "p_a_better": float(p_left),
            "p_equivalent": float(p_rope),
            "p_b_better": float(p_right),
        }
    except ImportError:
        logger.warning("baycomp not available, using count-based heuristic")
        diffs = d_a - d_b
        n = len(diffs)
        n_better = np.sum(diffs > rope)
        n_equiv = np.sum(np.abs(diffs) <= rope)
        n_worse = np.sum(diffs < -rope)
        total = max(n_better + n_equiv + n_worse, 1)
        return {
            "p_a_better": float(n_better / total),
            "p_equivalent": float(n_equiv / total),
            "p_b_better": float(n_worse / total),
        }


# ---------------------------------------------------------------------------
# C. Critical Difference Diagram (via autorank)
# ---------------------------------------------------------------------------


def generate_cd_diagram(d_matrix, method_names, output_path=None):
    """Generate a critical difference diagram using autorank.

    Autorank expects LOWER = better by default with order='ascending'.
    Since our d-values are HIGHER = better, we use order='descending'.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.
        method_names: List of method name strings.
        output_path: If provided, save the CD diagram as PDF.

    Returns:
        autorank result object.
    """
    import pandas as pd

    import autorank

    d_matrix = np.asarray(d_matrix, dtype=float)

    # Build DataFrame: columns = method names, rows = crises
    df = pd.DataFrame(d_matrix, columns=method_names)

    # order='descending' means higher values get rank 1 (better)
    result = autorank.autorank(df, alpha=0.05, verbose=False, order="descending")

    if output_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        autorank.plot_stats(result, ax=ax, allow_insignificant=True)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("CD diagram saved to %s", output_path)

    return result


# ---------------------------------------------------------------------------
# D. Bootstrap Rank Confidence Intervals
# ---------------------------------------------------------------------------


def bootstrap_rank_cis(d_matrix, n_bootstrap=10000, seed=42):
    """Compute bootstrap confidence intervals for mean ranks.

    For each bootstrap resample of crises, re-ranks methods within that
    resample and records mean ranks. Uses the percentile method for CIs.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.
        n_bootstrap: Number of bootstrap resamples (default 10000).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping method index to {mean_rank, ci_lo, ci_hi}.
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    n_crises, n_methods = d_matrix.shape
    rng = np.random.default_rng(seed)

    # Store bootstrap mean ranks
    boot_mean_ranks = np.zeros((n_bootstrap, n_methods))

    for b in range(n_bootstrap):
        boot_idx = rng.choice(n_crises, size=n_crises, replace=True)
        boot_d = d_matrix[boot_idx]

        # Rank within each bootstrapped crisis (rank 1 = highest d)
        ranks = np.zeros_like(boot_d)
        for i in range(n_crises):
            ranks[i] = rankdata(-boot_d[i])

        boot_mean_ranks[b] = ranks.mean(axis=0)

    # Actual mean ranks
    actual_ranks = np.zeros_like(d_matrix)
    for i in range(n_crises):
        actual_ranks[i] = rankdata(-d_matrix[i])
    actual_mean_ranks = actual_ranks.mean(axis=0)

    result = {}
    for j in range(n_methods):
        ci_lo, ci_hi = np.percentile(boot_mean_ranks[:, j], [2.5, 97.5])
        result[j] = {
            "mean_rank": float(actual_mean_ranks[j]),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
        }

    return result


# ---------------------------------------------------------------------------
# E. Oracle and Complementarity Analysis
# ---------------------------------------------------------------------------


def compute_oracle_and_complementarity(d_matrix, method_names):
    """Compute oracle score and method complementarity.

    The oracle selects the best method per crisis (maximum d-value).
    Complementarity is measured as 1 - mean |Spearman correlation|
    between methods' d-value vectors.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.
        method_names: List of method name strings.

    Returns:
        Dict with oracle_median_d, oracle_mean_d, best_single_median_d,
        oracle_improvement, correlation_matrix, complementarity_score.
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    n_crises, n_methods = d_matrix.shape

    # Oracle: best d per crisis
    oracle_d = d_matrix.max(axis=1)
    oracle_median = float(np.median(oracle_d))
    oracle_mean = float(np.mean(oracle_d))

    # Best single method by median d
    method_medians = np.median(d_matrix, axis=0)
    best_single_idx = int(np.argmax(method_medians))
    best_single_median = float(method_medians[best_single_idx])

    # Oracle improvement
    oracle_improvement = oracle_median - best_single_median

    # Spearman correlation matrix between methods
    corr_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                rho, _ = spearmanr(d_matrix[:, i], d_matrix[:, j])
                corr_matrix[i, j] = rho

    # Complementarity: 1 - mean |correlation| (excluding diagonal)
    n_pairs = n_methods * (n_methods - 1)
    if n_pairs > 0:
        abs_corr_sum = np.sum(np.abs(corr_matrix)) - n_methods  # subtract diagonal
        complementarity = 1.0 - abs_corr_sum / n_pairs
    else:
        complementarity = 0.0

    # Per-crisis best method
    best_per_crisis = {}
    for i in range(n_crises):
        best_idx = int(np.argmax(d_matrix[i]))
        best_per_crisis[f"crisis_{i}"] = method_names[best_idx]

    return {
        "oracle_median_d": oracle_median,
        "oracle_mean_d": oracle_mean,
        "best_single_median_d": best_single_median,
        "best_single_method": method_names[best_single_idx],
        "oracle_improvement": oracle_improvement,
        "correlation_matrix": corr_matrix,
        "complementarity_score": float(complementarity),
        "best_per_crisis": best_per_crisis,
    }


# ---------------------------------------------------------------------------
# F. Win-Rate Analysis
# ---------------------------------------------------------------------------


def compute_win_rates(d_matrix, method_names):
    """Compute pairwise win rates between methods.

    For each pair (A, B), counts the number of crises where A has a strictly
    higher d-value than B.

    Args:
        d_matrix: 2-D array (n_crises, n_methods) of Cohen's d values.
        method_names: List of method name strings.

    Returns:
        Nested dict: result[name_a][name_b] = number of crises A beats B.
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    n_crises, n_methods = d_matrix.shape

    result = {}
    for i, name_a in enumerate(method_names):
        result[name_a] = {}
        for j, name_b in enumerate(method_names):
            if i == j:
                continue
            wins = int(np.sum(d_matrix[:, i] > d_matrix[:, j]))
            result[name_a][name_b] = wins

    return result


# ---------------------------------------------------------------------------
# Helper: Build d-value matrix from JSON results
# ---------------------------------------------------------------------------


def build_d_matrix(results):
    """Build a d-value matrix from a comparison results dict.

    Args:
        results: Dict of {method_name: {crisis_name: {"d": float, ...}, ...}, ...}.

    Returns:
        (d_matrix, method_names, crisis_names): The d-value matrix (n_crises, n_methods),
        ordered list of method names, and ordered list of crisis names.
        Rows with any NaN are removed.
    """
    method_names = sorted(results.keys())

    # Collect all crisis names across all methods
    all_crises = set()
    for method_data in results.values():
        all_crises.update(method_data.keys())
    crisis_names = sorted(all_crises)

    n_crises = len(crisis_names)
    n_methods = len(method_names)

    d_matrix = np.full((n_crises, n_methods), np.nan)
    for j, method in enumerate(method_names):
        for i, crisis in enumerate(crisis_names):
            if crisis in results[method] and "d" in results[method][crisis]:
                d_matrix[i, j] = results[method][crisis]["d"]

    # Remove rows (crises) with any NaN
    valid_rows = ~np.any(np.isnan(d_matrix), axis=1)
    d_clean = d_matrix[valid_rows]
    crisis_names_clean = [c for c, valid in zip(crisis_names, valid_rows) if valid]

    return d_clean, method_names, crisis_names_clean


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_statistical_comparison(input_path):
    """Run the full statistical comparison pipeline.

    Loads a comparison JSON, builds the d-value matrix, runs all 6 analyses
    (MCS, Bayesian, CD diagram, bootstrap ranks, oracle, win rates), and
    saves results to JSON and a CD diagram PDF.

    Args:
        input_path: Path to comparison results JSON.

    Returns:
        Dict of all results.
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        data = json.load(f)

    results_dict = data["results"]
    d_matrix, method_names, crisis_names = build_d_matrix(results_dict)

    n_crises, n_methods = d_matrix.shape
    print(f"\n{'='*70}")
    print(f"STATISTICAL COMPARISON FRAMEWORK")
    print(f"{'='*70}")
    print(f"Input: {input_path.name}")
    print(f"Methods: {n_methods}, Crises: {n_crises}")
    print(f"Methods: {', '.join(method_names)}")
    print()

    output = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "n_methods": n_methods,
        "n_crises": n_crises,
        "method_names": method_names,
        "crisis_names": crisis_names,
    }

    # --- A. Model Confidence Set ---
    print(f"{'─'*40}")
    print("A. MODEL CONFIDENCE SET (alpha=0.10)")
    print(f"{'─'*40}")
    mcs_indices = compute_model_confidence_set(d_matrix, alpha=0.10, n_bootstrap=10000, seed=42)
    mcs_names = [method_names[i] for i in mcs_indices]
    print(f"  MCS contains {len(mcs_names)} method(s):")
    for name in mcs_names:
        idx = method_names.index(name)
        median_d = float(np.median(d_matrix[:, idx]))
        print(f"    - {name} (median d = {median_d:.3f})")
    output["model_confidence_set"] = {
        "alpha": 0.10,
        "n_bootstrap": 10000,
        "mcs_methods": mcs_names,
        "mcs_indices": [int(i) for i in mcs_indices],
    }
    print()

    # --- B. Bayesian Signed-Rank Tests ---
    print(f"{'─'*40}")
    print("B. BAYESIAN SIGNED-RANK TESTS (rope=0.1)")
    print(f"{'─'*40}")
    qcml_methods = ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity"]
    baselines = ["Random Forest", "CUSUM"]
    bayesian_results = {}

    for qcml_name in qcml_methods:
        if qcml_name not in method_names:
            continue
        qi = method_names.index(qcml_name)
        for baseline_name in baselines:
            if baseline_name not in method_names:
                continue
            bi = method_names.index(baseline_name)
            pair_key = f"{qcml_name} vs {baseline_name}"
            result = bayesian_signed_rank(d_matrix[:, qi], d_matrix[:, bi], rope=0.1)
            bayesian_results[pair_key] = result
            print(f"  {pair_key}:")
            print(
                f"    P({qcml_name} better) = {result['p_a_better']:.3f}, "
                f"P(equiv) = {result['p_equivalent']:.3f}, "
                f"P({baseline_name} better) = {result['p_b_better']:.3f}"
            )

    output["bayesian_signed_rank"] = {
        "rope": 0.1,
        "comparisons": bayesian_results,
    }
    print()

    # --- C. Critical Difference Diagram ---
    print(f"{'─'*40}")
    print("C. CRITICAL DIFFERENCE DIAGRAM")
    print(f"{'─'*40}")
    output_dir = input_path.parent
    cd_path = output_dir / "cd_diagram.pdf"
    try:
        ar_result = generate_cd_diagram(d_matrix, method_names, output_path=str(cd_path))
        print(f"  Saved to: {cd_path}")
        # Extract rank info from autorank result
        if hasattr(ar_result, "rankdf"):
            rank_df = ar_result.rankdf
            print(f"  Rank summary:")
            for _, row in rank_df.iterrows():
                print(
                    f"    {row.name}: mean rank = {row['meanrank']:.2f}, "
                    f"median = {row['median']:.3f}"
                )
            output["cd_diagram"] = {
                "path": str(cd_path),
                "mean_ranks": {
                    row.name: float(row["meanrank"]) for _, row in rank_df.iterrows()
                },
            }
    except Exception as e:
        print(f"  Failed to generate CD diagram: {e}")
        output["cd_diagram"] = {"error": str(e)}
    print()

    # --- D. Bootstrap Rank CIs ---
    print(f"{'─'*40}")
    print("D. BOOTSTRAP RANK CONFIDENCE INTERVALS")
    print(f"{'─'*40}")
    rank_cis = bootstrap_rank_cis(d_matrix, n_bootstrap=10000, seed=42)
    output["bootstrap_rank_cis"] = {}
    for j in range(n_methods):
        name = method_names[j]
        ci = rank_cis[j]
        print(
            f"  {name}: rank = {ci['mean_rank']:.2f} "
            f"[{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]"
        )
        output["bootstrap_rank_cis"][name] = ci
    print()

    # --- E. Oracle and Complementarity ---
    print(f"{'─'*40}")
    print("E. ORACLE & COMPLEMENTARITY")
    print(f"{'─'*40}")
    oracle = compute_oracle_and_complementarity(d_matrix, method_names)
    print(f"  Oracle median d:       {oracle['oracle_median_d']:.3f}")
    print(f"  Best single median d:  {oracle['best_single_median_d']:.3f}")
    print(f"  Best single method:    {oracle['best_single_method']}")
    print(f"  Oracle improvement:    +{oracle['oracle_improvement']:.3f}")
    print(f"  Complementarity score: {oracle['complementarity_score']:.3f}")
    print(f"  Best method per crisis:")
    for crisis, best in oracle["best_per_crisis"].items():
        print(f"    {crisis}: {best}")

    # Convert correlation matrix to serializable format
    output["oracle_and_complementarity"] = {
        "oracle_median_d": oracle["oracle_median_d"],
        "oracle_mean_d": oracle["oracle_mean_d"],
        "best_single_median_d": oracle["best_single_median_d"],
        "best_single_method": oracle["best_single_method"],
        "oracle_improvement": oracle["oracle_improvement"],
        "complementarity_score": oracle["complementarity_score"],
        "best_per_crisis": oracle["best_per_crisis"],
        "correlation_matrix": oracle["correlation_matrix"].tolist(),
    }
    print()

    # --- F. Win Rates ---
    print(f"{'─'*40}")
    print("F. PAIRWISE WIN RATES")
    print(f"{'─'*40}")
    win_rates = compute_win_rates(d_matrix, method_names)
    output["win_rates"] = win_rates

    # Print a compact table
    # Header row
    short_names = [n[:12] for n in method_names]
    header = f"  {'':>20s} | " + " | ".join(f"{s:>6s}" for s in short_names)
    print(header)
    print(f"  {'─'*20}-+-" + "-+-".join("─" * 6 for _ in short_names))
    for i, name in enumerate(method_names):
        row_vals = []
        for j, other in enumerate(method_names):
            if i == j:
                row_vals.append("   -- ")
            else:
                w = win_rates[name][other]
                row_vals.append(f"{w:>4d}/{n_crises:<1d}")
        print(f"  {name:>20s} | " + " | ".join(row_vals))
    print()

    # --- Save output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"statistical_comparison_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    if len(sys.argv) < 2:
        print("Usage: python experiments/statistical_comparison.py <comparison_results.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    run_statistical_comparison(input_file)
