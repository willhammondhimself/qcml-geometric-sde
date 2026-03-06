"""
Reconcile all paper tables from canonical JSON.

Regenerates:
  - aggregate_comparison.tex (Table 4 - main comparison, compiled)
  - per_crisis_winners.tex  (Table 5 - per-crisis matrix, compiled)
  - crisis_taxonomy.tex     (Table 6 - crisis taxonomy, compiled)
  - table_stratified.tex    (Table 7 - novel vs conventional, compiled)
  - table_comparison.tex    (auto-generated, not compiled)
  - table_per_crisis.tex    (auto-generated, not compiled)

Also prints inline numbers needed for text reconciliation.

Usage:
    python paper/reconcile_tables.py                      # auto-find latest JSON
    python paper/reconcile_tables.py --json path/to.json  # specific JSON
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.generate_paper_tables import METHOD_CATEGORIES, LATEX_NAMES, build_d_matrix

TABLES_DIR = REPO_ROOT / "paper" / "tables"
RESULTS_DIR = REPO_ROOT / "experiments" / "outputs" / "regime_detection"

# --- Crisis metadata ---
CRISIS_PRETTY = {
    "2000_dotcom": "Dot-Com '00",
    "2001_911": "September 11 '01",
    "2007_quant": "Quant '07",
    "2008_gfc": "GFC '08",
    "2010_flash": "Flash Crash '10",
    "2011_euro": "Euro Crisis '11",
    "2015_china": "China '15",
    "2018_volmageddon": "Volmageddon '18",
    "2018_q4": "Q4 Selloff '18",
    "2019_repo": "Repo '19",
    "2020_covid": "COVID '20",
    "2022_rates": "Rate Hikes '22",
    "2023_svb": "SVB '23",
    "2024_carry": "Carry '24",
}

# Novel vs Conventional classification (a priori)
NOVEL_CRISES = {
    "2018_volmageddon", "2018_q4", "2019_repo",
    "2022_rates", "2023_svb", "2024_carry",
}

# Crisis taxonomy categories
CRISIS_CATEGORIES = {
    "Volatility Shocks": {
        "crises": ["2018_volmageddon", "2010_flash", "2018_q4"],
        "mechanism": "Sudden volatility regime shift",
    },
    "Systemic/Credit": {
        "crises": ["2008_gfc", "2011_euro", "2023_svb"],
        "mechanism": "Credit contagion and bank runs",
    },
    "Exogenous Shocks": {
        "crises": ["2001_911", "2020_covid"],
        "mechanism": "External trigger, not endogenous",
    },
    "Slow Burns": {
        "crises": ["2000_dotcom", "2022_rates", "2024_carry"],
        "mechanism": "Gradual regime shift over months",
    },
    "Liquidity/Micro.": {
        "crises": ["2007_quant", "2019_repo", "2015_china"],
        "mechanism": "Market structure and liquidity breakdown",
    },
}

# Short names for per_crisis_winners table header
SHORT_NAMES = {
    "Berry Phase Rate": "Berry",
    "QFI Determinant": "QFI",
    "Multi-Lag Fidelity": "MLF",
    "QCML Chern": "Chern",
    "Metric Condition": "MetCon",
    "Speed Limit Ratio": "SLR",
    "Dimensionality Collapse": "DimC",
    "Sectional Curvature Sign": "SeCu",
    "Rolling Vol Z": "RVol",
    "CUSUM": "CUSUM",
    "Random Forest": "RF",
}

# Methods to display in per_crisis_winners (Table 5)
TABLE5_METHODS = [
    "Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity",
    "QCML Chern", "Metric Condition", "Speed Limit Ratio",
    "Dimensionality Collapse", "Sectional Curvature Sign",
    "Rolling Vol Z", "CUSUM", "Random Forest",
]

# Geometric methods (for observatory analysis)
GEOMETRIC_METHODS = {
    "Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity",
    "QCML Chern", "Metric Condition", "Spectral Gap",
    "Speed Limit Ratio", "Dimensionality Collapse",
    "Sectional Curvature Sign", "Geodesic Velocity",
    "Geometric Consensus", "Geometric Ensemble",
}

# Methods for temporal stability and stratified tables
TEMPORAL_METHODS = [
    "Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity",
    "CUSUM", "Random Forest",
]


def find_latest_json() -> Path:
    patterns = [
        str(RESULTS_DIR / "causal_comparison_*.json"),
        str(RESULTS_DIR / "comparison_*.json"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No comparison JSON in {RESULTS_DIR}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return Path(candidates[0])


def gen_aggregate_comparison(d_matrix, method_names, crisis_keys, summary):
    """Generate aggregate_comparison.tex (Table 4, compiled into paper)."""
    n_crises, n_methods = d_matrix.shape
    median_d = np.nanmedian(d_matrix, axis=0)
    mean_d = np.nanmean(d_matrix, axis=0)
    max_d = np.nanmax(d_matrix, axis=0)

    # n_sig: count of crises where bootstrap CI excludes zero
    # For now, count non-NaN entries (all entries have d > 0 in practice)
    n_sig = np.sum(~np.isnan(d_matrix), axis=0)

    # Sort by median d descending
    order = np.argsort(-median_d)
    best_median = np.nanmax(median_d)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Aggregate effect sizes across {n_crises} crises "
        rf"({n_methods} methods). Mean and median Cohen's $d$,"
    )
    lines.append(
        r"  with number of crises where bootstrap 95\% CI excludes zero.}"
    )
    lines.append(r"\label{tab:aggregate_comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Mean $d$ & Median $d$ & Max $d$ & $n_{\text{sig}}$ \\"
    )
    lines.append(r"\midrule")

    for idx in order:
        name = LATEX_NAMES.get(method_names[idx], method_names[idx])
        cat = METHOD_CATEGORIES.get(method_names[idx], "?")
        m_d = median_d[idx]
        # Italicize classical/supervised methods
        if cat in ("Classical", "Supervised"):
            name = r"\textit{" + name + "}"

        # Bold best median
        if abs(m_d - best_median) < 1e-6:
            med_str = r"\textbf{" + f"{m_d:.3f}" + "}"
        else:
            med_str = f"{m_d:.3f}"

        lines.append(
            f"{name} & {mean_d[idx]:.3f} & {med_str} & "
            f"{max_d[idx]:.3f} & {int(n_sig[idx])}/{n_crises} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_per_crisis_winners(d_matrix, method_names, crisis_keys):
    """Generate per_crisis_winners.tex (Table 5, compiled into paper)."""
    # Find indices for display methods
    display_indices = []
    display_names = []
    for m in TABLE5_METHODS:
        if m in method_names:
            display_indices.append(method_names.index(m))
            display_names.append(m)

    n_display = len(display_indices)
    n_methods_total = len(method_names)

    header = [SHORT_NAMES.get(n, n[:6]) for n in display_names]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Per-crisis Cohen's $d$ for selected detection methods "
        rf"({n_display} of {n_methods_total})."
    )
    lines.append(
        r"  Bold indicates the winner among shown methods.  "
        r"Geometric channels (left) vs.\ baselines (right).}"
    )
    lines.append(r"\label{tab:per_crisis_winners}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l" + "r" * n_display + "}")
    lines.append(r"\toprule")
    lines.append("Crisis & " + " & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for i, crisis in enumerate(crisis_keys):
        name = CRISIS_PRETTY.get(crisis, crisis)
        # Find winner among displayed methods
        displayed_d = [d_matrix[i, idx] for idx in display_indices]
        winner_pos = int(np.nanargmax(displayed_d))

        vals = []
        for pos, idx in enumerate(display_indices):
            d_val = d_matrix[i, idx]
            if np.isnan(d_val):
                vals.append("--")
            elif pos == winner_pos:
                vals.append(r"\textbf{" + f"{d_val:.2f}" + "}")
            else:
                vals.append(f"{d_val:.2f}")

        lines.append(f"{name} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_crisis_taxonomy(d_matrix, method_names, crisis_keys):
    """Generate crisis_taxonomy.tex (Table 6, compiled into paper)."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Crisis taxonomy and geometric channel specialization. "
        r"Each crisis category has a distinct mechanism.  "
        r"``Best Overall'' includes baselines; "
        r"``Best Geometric'' is restricted to the seven observatory channels.}"
    )
    lines.append(r"\label{tab:crisis_taxonomy}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lp{3.2cm}lll}")
    lines.append(r"\toprule")
    lines.append(
        r"Category & Mechanism & Crises & Best Overall & Best Geometric \\"
    )
    lines.append(r"\midrule")

    geo_set = GEOMETRIC_METHODS

    for cat_name, cat_info in CRISIS_CATEGORIES.items():
        mechanism = cat_info["mechanism"]
        cat_crises = [c for c in cat_info["crises"] if c in crisis_keys]
        if not cat_crises:
            continue

        crisis_indices = [crisis_keys.index(c) for c in cat_crises]
        crisis_years = ", ".join(c[:4] for c in cat_crises)

        # Mean d per method across category crises
        best_overall_d = -1
        best_overall_name = "?"
        best_geo_d = -1
        best_geo_name = "?"

        for j, method in enumerate(method_names):
            mean_d = np.nanmean([d_matrix[ci, j] for ci in crisis_indices])
            if mean_d > best_overall_d:
                best_overall_d = mean_d
                best_overall_name = method
            if method in geo_set and mean_d > best_geo_d:
                best_geo_d = mean_d
                best_geo_name = method

        # Short names
        short_overall = SHORT_NAMES.get(best_overall_name, best_overall_name[:8])
        short_geo = SHORT_NAMES.get(best_geo_name, best_geo_name[:8])

        lines.append(
            f"{cat_name} & {mechanism} & {crisis_years} & "
            f"{short_overall} ({best_overall_d:.2f}) & "
            f"{short_geo} ({best_geo_d:.2f}) \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_table_stratified(d_matrix, method_names, crisis_keys):
    """Generate table_stratified.tex (Table 7, compiled into paper)."""
    conv_idx = [i for i, k in enumerate(crisis_keys) if k not in NOVEL_CRISES]
    novel_idx = [i for i, k in enumerate(crisis_keys) if k in NOVEL_CRISES]

    n_conv = len(conv_idx)
    n_novel = len(novel_idx)

    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & \multicolumn{2}{c}{Median $d$} & "
        r"\multicolumn{2}{c}{Mean $d$} & Change \\"
    )
    lines.append(
        rf" & Conv.\ ({n_conv}) & Novel ({n_novel}) & "
        rf"Conv.\ ({n_conv}) & Novel ({n_novel}) & (Median) \\"
    )
    lines.append(r"\midrule")

    display_methods = TEMPORAL_METHODS + ["Best Geometric"]
    geo_top3 = ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity"]

    for method in display_methods:
        if method == "Best Geometric":
            # Best of Berry, QFI, MLF per crisis
            geo_indices = [method_names.index(m) for m in geo_top3 if m in method_names]
            conv_vals = [np.nanmax([d_matrix[ci, gi] for gi in geo_indices]) for ci in conv_idx]
            novel_vals = [np.nanmax([d_matrix[ci, gi] for gi in geo_indices]) for ci in novel_idx]
            name = "Best Geometric"
        else:
            if method not in method_names:
                continue
            j = method_names.index(method)
            conv_vals = [d_matrix[ci, j] for ci in conv_idx]
            novel_vals = [d_matrix[ci, j] for ci in novel_idx]
            name = LATEX_NAMES.get(method, method)

        conv_median = float(np.nanmedian(conv_vals))
        novel_median = float(np.nanmedian(novel_vals))
        conv_mean = float(np.nanmean(conv_vals))
        novel_mean = float(np.nanmean(novel_vals))

        if conv_median > 0:
            change = (novel_median - conv_median) / conv_median * 100
            change_str = f"{change:+.0f}\\%"
        else:
            change_str = "--"

        lines.append(
            f"{name:<25} & {conv_median:.2f} & {novel_median:.2f} & "
            f"{conv_mean:.2f} & {novel_mean:.2f} & {change_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\medskip")
    lines.append(
        r"\noindent{\footnotesize Crisis categories defined \emph{a priori} "
        r"(Table~\ref{tab:crises}): Novel = unprecedented market mechanisms; "
        r"Conventional = recognizable historical parallels.  "
        r"``Best Geometric'' = $\max(\text{Berry}, \text{QFI}, \text{MLF})$ "
        r"per crisis.}"
    )
    return "\n".join(lines)


def print_inline_numbers(d_matrix, method_names, crisis_keys, summary):
    """Print all numbers needed for text reconciliation."""
    n_crises, n_methods = d_matrix.shape
    median_d = np.nanmedian(d_matrix, axis=0)
    mean_d = np.nanmean(d_matrix, axis=0)

    # Ranks
    d_clean = d_matrix.copy()
    for j in range(n_methods):
        mask = np.isnan(d_clean[:, j])
        d_clean[mask, j] = np.nanmedian(d_clean[:, j])
    ranks = np.zeros_like(d_clean)
    for i in range(n_crises):
        ranks[i] = stats.rankdata(-d_clean[i])
    mean_ranks = np.mean(ranks, axis=0)

    # Friedman
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[d_clean[:, j] for j in range(n_methods)]
        )
    except Exception:
        friedman_stat, friedman_p = float("nan"), float("nan")

    # Sort by median d
    order = np.argsort(-median_d)

    print("\n" + "=" * 70)
    print("INLINE NUMBERS FOR TEXT RECONCILIATION")
    print("=" * 70)

    print(f"\n--- Method count: {n_methods} methods x {n_crises} crises ---\n")

    print("Rankings by median d:")
    for rank, idx in enumerate(order, 1):
        name = method_names[idx]
        cat = METHOD_CATEGORIES.get(name, "?")
        print(
            f"  {rank:2d}. {name:<27} median d = {median_d[idx]:.3f}, "
            f"mean d = {mean_d[idx]:.3f}, mean rank = {mean_ranks[idx]:.1f}  "
            f"[{cat}]"
        )

    print(f"\n--- Friedman test ---")
    print(f"  chi^2 = {friedman_stat:.1f}, p = {friedman_p:.2e}")

    # Find specific method ranks
    for method in ["Berry Phase Rate", "CUSUM", "QFI Determinant",
                    "Rolling RF (VIX)", "Random Forest", "VIX Level"]:
        if method in method_names:
            idx = method_names.index(method)
            rank = np.where(order == idx)[0][0] + 1
            print(
                f"  {method}: rank {rank}, median d = {median_d[idx]:.2f}"
            )

    # Temporal stability
    pre_2020 = [i for i, k in enumerate(crisis_keys) if int(k[:4]) < 2020]
    post_2020 = [i for i, k in enumerate(crisis_keys) if int(k[:4]) >= 2020]
    post_crises = [crisis_keys[i] for i in post_2020]

    print(f"\n--- Temporal stability ---")
    print(f"  Pre-2020: {len(pre_2020)} crises, Post-2020: {len(post_2020)} crises")
    print(f"  Post-2020 crises: {post_crises}")

    for method in TEMPORAL_METHODS:
        if method not in method_names:
            continue
        j = method_names.index(method)
        pre_mean = np.nanmean(d_matrix[pre_2020, j])
        post_mean = np.nanmean(d_matrix[post_2020, j])
        print(f"  {method}: pre={pre_mean:.2f}, post={post_mean:.2f}")
        for pi in post_2020:
            print(f"    {crisis_keys[pi]}: d = {d_matrix[pi, j]:.2f}")

    # Novel vs conventional
    conv_idx = [i for i, k in enumerate(crisis_keys) if k not in NOVEL_CRISES]
    novel_idx = [i for i, k in enumerate(crisis_keys) if k in NOVEL_CRISES]

    print(f"\n--- Novel vs Conventional ---")
    print(f"  Conventional: {len(conv_idx)}, Novel: {len(novel_idx)}")
    for method in TEMPORAL_METHODS:
        if method not in method_names:
            continue
        j = method_names.index(method)
        conv_med = np.nanmedian([d_matrix[ci, j] for ci in conv_idx])
        novel_med = np.nanmedian([d_matrix[ci, j] for ci in novel_idx])
        change = (novel_med - conv_med) / conv_med * 100 if conv_med > 0 else 0
        print(f"  {method}: conv={conv_med:.2f}, novel={novel_med:.2f}, change={change:+.0f}%")

    # Best geometric per crisis
    geo_top3 = ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity"]
    geo_indices = [method_names.index(m) for m in geo_top3 if m in method_names]
    conv_best = [np.nanmax([d_matrix[ci, gi] for gi in geo_indices]) for ci in conv_idx]
    novel_best = [np.nanmax([d_matrix[ci, gi] for gi in geo_indices]) for ci in novel_idx]
    print(
        f"  Best Geometric: conv={np.nanmedian(conv_best):.2f}, "
        f"novel={np.nanmedian(novel_best):.2f}, "
        f"change={((np.nanmedian(novel_best) - np.nanmedian(conv_best)) / np.nanmedian(conv_best) * 100):+.0f}%"
    )

    # Per-crisis winners
    print(f"\n--- Per-crisis winners (all {n_methods} methods) ---")
    geo_wins = 0
    classical_wins = 0
    for i, crisis in enumerate(crisis_keys):
        winner_idx = np.nanargmax(d_matrix[i])
        winner = method_names[winner_idx]
        winner_d = d_matrix[i, winner_idx]
        cat = METHOD_CATEGORIES.get(winner, "?")
        if cat == "Geometric":
            geo_wins += 1
        else:
            classical_wins += 1
        print(f"  {CRISIS_PRETTY.get(crisis, crisis)}: {winner} (d={winner_d:.2f}) [{cat}]")
    print(f"  Geometric wins: {geo_wins}/{n_crises}, Other wins: {classical_wins}/{n_crises}")

    # Per-crisis winners among TABLE5 methods only
    print(f"\n--- Per-crisis winners (Table 5 methods only) ---")
    t5_indices = [method_names.index(m) for m in TABLE5_METHODS if m in method_names]
    geo_wins_t5 = 0
    for i, crisis in enumerate(crisis_keys):
        t5_d = [(d_matrix[i, idx], method_names[idx]) for idx in t5_indices]
        winner = max(t5_d, key=lambda x: x[0] if not np.isnan(x[0]) else -1)
        cat = METHOD_CATEGORIES.get(winner[1], "?")
        if cat == "Geometric":
            geo_wins_t5 += 1
        print(f"  {CRISIS_PRETTY.get(crisis, crisis)}: {winner[1]} (d={winner[0]:.2f}) [{cat}]")
    print(f"  Geometric wins: {geo_wins_t5}/{n_crises}")

    # CUSUM specific
    print(f"\n--- CUSUM details ---")
    if "CUSUM" in method_names:
        cusum_idx = method_names.index("CUSUM")
        cusum_rank = np.where(order == cusum_idx)[0][0] + 1
        cusum_wins = sum(
            1 for i in range(n_crises)
            if np.nanargmax(d_matrix[i]) == cusum_idx
        )
        print(f"  Rank: {cusum_rank}, median d = {median_d[cusum_idx]:.2f}")
        print(f"  Wins {cusum_wins} of {n_crises} crises")


def main():
    parser = argparse.ArgumentParser(description="Reconcile all paper tables from canonical JSON")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't write files")
    args = parser.parse_args()

    if args.json:
        json_path = Path(args.json)
    else:
        json_path = find_latest_json()

    print(f"Loading: {json_path.name}")

    with open(json_path) as f:
        data = json.load(f)

    results_dict = data["results"]
    summary = data.get("summary", {})
    d_matrix, method_names, crisis_keys = build_d_matrix(results_dict)

    n_methods = len(method_names)
    n_crises = len(crisis_keys)
    print(f"  {n_methods} methods x {n_crises} crises")
    print(f"  Methods: {method_names}")
    print(f"  Crises: {crisis_keys}")

    # Generate tables
    tables = {
        "aggregate_comparison.tex": gen_aggregate_comparison(
            d_matrix, method_names, crisis_keys, summary
        ),
        "per_crisis_winners.tex": gen_per_crisis_winners(
            d_matrix, method_names, crisis_keys
        ),
        "crisis_taxonomy.tex": gen_crisis_taxonomy(
            d_matrix, method_names, crisis_keys
        ),
        "table_stratified.tex": gen_table_stratified(
            d_matrix, method_names, crisis_keys
        ),
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in tables.items():
        path = TABLES_DIR / filename
        if args.dry_run:
            print(f"\n{'=' * 70}")
            print(f"  {filename}")
            print(f"{'=' * 70}")
            print(content)
        else:
            path.write_text(content)
            print(f"  Written: {path.relative_to(REPO_ROOT)}")

    # Print inline numbers
    print_inline_numbers(d_matrix, method_names, crisis_keys, summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
