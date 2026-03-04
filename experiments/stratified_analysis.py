"""
Stratified analysis: Novel vs Conventional crisis performance.

Reads causal_comparison results and CRISIS_CATEGORIES to compute
median/mean Cohen's d by crisis category for each method.
Runs Mann-Whitney U test on RF Novel vs RF Conventional.
Outputs paper/tables/table_stratified.tex.

Usage:
    python experiments/stratified_analysis.py
    python experiments/stratified_analysis.py --json <path_to_comparison.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Crisis categories (mirrors data_loader.py lines 119-140).
# Defined a priori before seeing any results.
NOVEL_CRISES = [
    '2018_volmageddon',
    '2018_q4',
    '2019_repo',
    '2022_rates',
    '2023_svb',
    '2024_carry',
]

CONVENTIONAL_CRISES = [
    '2000_dotcom',
    '2001_911',
    '2007_quant',
    '2008_gfc',
    '2010_flash',
    '2011_euro',
    '2015_china',
    '2020_covid',
]

# Methods to include in the stratified table
METHODS = [
    'Berry Phase Rate',
    'QFI Determinant',
    'Multi-Lag Fidelity',
    'Random Forest',
    'CUSUM',
]

LATEX_NAMES = {
    'Berry Phase Rate': r"Berry Curv.\ Incr.",
    'QFI Determinant': 'QFI Determinant',
    'Multi-Lag Fidelity': 'Multi-Lag Fidelity',
    'Random Forest': 'Random Forest',
    'CUSUM': 'CUSUM',
    'Best Geometric': 'Best Geometric',
}

# Normalize crisis key: the JSON uses '2001911' but categories use '2001_911'
def normalize_key(key):
    """Normalize crisis key to match CRISIS_CATEGORIES format."""
    if key == '2001911':
        return '2001_911'
    return key


def get_d_values(results, method, crisis_list):
    """Extract d-values for a method on a list of crises."""
    d_vals = []
    method_data = results.get(method, {})
    for crisis in crisis_list:
        # Try both formats
        for key_variant in [crisis, crisis.replace('_', '')]:
            if key_variant in method_data:
                entry = method_data[key_variant]
                if isinstance(entry, dict) and 'd' in entry:
                    d_vals.append(entry['d'])
                elif isinstance(entry, (int, float)):
                    d_vals.append(entry)
                break
    return np.array(d_vals)


def best_geometric_d(results, crisis_list):
    """Compute best geometric d per crisis across Berry, QFI, MLF."""
    geo_methods = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity']
    d_vals = []
    for crisis in crisis_list:
        crisis_d = []
        for method in geo_methods:
            method_data = results.get(method, {})
            for key_variant in [crisis, crisis.replace('_', '')]:
                if key_variant in method_data:
                    entry = method_data[key_variant]
                    if isinstance(entry, dict) and 'd' in entry:
                        crisis_d.append(entry['d'])
                    elif isinstance(entry, (int, float)):
                        crisis_d.append(entry)
                    break
        d_vals.append(max(crisis_d) if crisis_d else np.nan)
    return np.array(d_vals)


def generate_table(results, novel_crises, conventional_crises, footnote_text=""):
    """Generate LaTeX table for stratified analysis."""
    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & \multicolumn{2}{c}{Median $d$} & "
        r"\multicolumn{2}{c}{Mean $d$} & Change \\"
    )
    lines.append(
        r" & Conv.\ (%d) & Novel (%d) & Conv.\ (%d) & Novel (%d) & (Median) \\"
        % (len(conventional_crises), len(novel_crises),
           len(conventional_crises), len(novel_crises))
    )
    lines.append(r"\midrule")

    rows = []
    for method in METHODS:
        conv_d = get_d_values(results, method, conventional_crises)
        novel_d = get_d_values(results, method, novel_crises)
        med_conv = np.nanmedian(conv_d) if len(conv_d) > 0 else np.nan
        med_novel = np.nanmedian(novel_d) if len(novel_d) > 0 else np.nan
        mean_conv = np.nanmean(conv_d) if len(conv_d) > 0 else np.nan
        mean_novel = np.nanmean(novel_d) if len(novel_d) > 0 else np.nan
        if med_conv > 0:
            change = (med_novel - med_conv) / med_conv * 100
        else:
            change = np.nan
        rows.append((method, med_conv, med_novel, mean_conv, mean_novel, change))

    # Best Geometric row
    conv_best = best_geometric_d(results, conventional_crises)
    novel_best = best_geometric_d(results, novel_crises)
    med_conv_best = np.nanmedian(conv_best)
    med_novel_best = np.nanmedian(novel_best)
    mean_conv_best = np.nanmean(conv_best)
    mean_novel_best = np.nanmean(novel_best)
    change_best = (med_novel_best - med_conv_best) / med_conv_best * 100
    rows.append(('Best Geometric', med_conv_best, med_novel_best,
                 mean_conv_best, mean_novel_best, change_best))

    for method, med_c, med_n, mean_c, mean_n, chg in rows:
        name = LATEX_NAMES.get(method, method)
        sign = "+" if chg >= 0 else ""
        lines.append(
            f"{name:<20} & {med_c:.2f} & {med_n:.2f} & "
            f"{mean_c:.2f} & {mean_n:.2f} & {sign}{chg:.0f}\\% \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if footnote_text:
        lines.append("")
        lines.append(footnote_text)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Stratified analysis: Novel vs Conventional crisis performance"
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Path to comparison results JSON (auto-detects latest if omitted)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "experiments" / "outputs" / "regime_detection"

    if args.json:
        json_path = Path(args.json)
    else:
        # Find latest causal_comparison JSON
        candidates = sorted(output_dir.glob("causal_comparison_*.json"))
        if not candidates:
            print("ERROR: No causal_comparison JSON found in", output_dir)
            sys.exit(1)
        json_path = candidates[-1]

    print(f"Reading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]

    # Identify which crises are actually in the data
    first_method = list(results.keys())[0]
    available_crises = set(normalize_key(k) for k in results[first_method].keys())

    novel = [c for c in NOVEL_CRISES if c in available_crises]
    conventional = [c for c in CONVENTIONAL_CRISES if c in available_crises]

    print(f"Novel crises ({len(novel)}): {novel}")
    print(f"Conventional crises ({len(conventional)}): {conventional}")
    print()

    # === Main analysis ===
    print("=" * 70)
    print("STRATIFIED ANALYSIS: Novel vs Conventional (a priori classification)")
    print("=" * 70)

    for method in METHODS + ['Best Geometric']:
        if method == 'Best Geometric':
            conv_d = best_geometric_d(results, conventional)
            novel_d = best_geometric_d(results, novel)
        else:
            conv_d = get_d_values(results, method, conventional)
            novel_d = get_d_values(results, method, novel)

        med_c = np.nanmedian(conv_d)
        med_n = np.nanmedian(novel_d)
        mean_c = np.nanmean(conv_d)
        mean_n = np.nanmean(novel_d)
        change = (med_n - med_c) / med_c * 100 if med_c > 0 else float('nan')

        print(f"  {method:<20}  Conv median={med_c:.2f}  Novel median={med_n:.2f}  "
              f"Change={change:+.0f}%")

        if method == 'Random Forest':
            # Mann-Whitney U test
            stat, p = stats.mannwhitneyu(conv_d, novel_d, alternative='two-sided')
            print(f"    Mann-Whitney U: U={stat:.1f}, p={p:.3f}")

    print()

    # === COVID robustness variant ===
    print("=" * 70)
    print("ROBUSTNESS: COVID reclassified as Novel")
    print("=" * 70)

    novel_covid = novel + ['2020_covid']
    conventional_no_covid = [c for c in conventional if c != '2020_covid']

    print(f"Novel crises ({len(novel_covid)}): {novel_covid}")
    print(f"Conventional crises ({len(conventional_no_covid)}): {conventional_no_covid}")

    for method in METHODS + ['Best Geometric']:
        if method == 'Best Geometric':
            conv_d = best_geometric_d(results, conventional_no_covid)
            novel_d = best_geometric_d(results, novel_covid)
        else:
            conv_d = get_d_values(results, method, conventional_no_covid)
            novel_d = get_d_values(results, method, novel_covid)

        med_c = np.nanmedian(conv_d)
        med_n = np.nanmedian(novel_d)
        change = (med_n - med_c) / med_c * 100 if med_c > 0 else float('nan')
        print(f"  {method:<20}  Conv median={med_c:.2f}  Novel median={med_n:.2f}  "
              f"Change={change:+.0f}%")

    print()

    # === Generate LaTeX table ===
    table_dir = repo_root / "paper" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "table_stratified.tex"

    # Main table
    footnote = (
        r"\medskip" "\n"
        r"\noindent{\footnotesize Crisis categories defined \emph{a priori} "
        r"(Table~\ref{tab:crises}): Novel = unprecedented market mechanisms; "
        r"Conventional = recognizable historical parallels.  "
        r"``Best Geometric'' = $\max(\text{Berry}, \text{QFI}, \text{MLF})$ per crisis.  "
    )

    # Add Mann-Whitney result
    rf_conv = get_d_values(results, 'Random Forest', conventional)
    rf_novel = get_d_values(results, 'Random Forest', novel)
    stat, p = stats.mannwhitneyu(rf_conv, rf_novel, alternative='two-sided')
    footnote += (
        f"Mann--Whitney $U$ test on RF Novel vs.\\ Conventional: "
        f"$U = {stat:.0f}$, $p = {p:.2f}$.  "
    )

    # COVID robustness
    rf_conv_no_covid = get_d_values(results, 'Random Forest', conventional_no_covid)
    rf_novel_covid = get_d_values(results, 'Random Forest', novel_covid)
    best_conv_no_covid = best_geometric_d(results, conventional_no_covid)
    best_novel_covid = best_geometric_d(results, novel_covid)

    rf_med_c2 = np.nanmedian(rf_conv_no_covid)
    rf_med_n2 = np.nanmedian(rf_novel_covid)
    rf_chg2 = (rf_med_n2 - rf_med_c2) / rf_med_c2 * 100
    bg_med_c2 = np.nanmedian(best_conv_no_covid)
    bg_med_n2 = np.nanmedian(best_novel_covid)
    bg_chg2 = (bg_med_n2 - bg_med_c2) / bg_med_c2 * 100

    footnote += (
        f"Reclassifying COVID as Novel: RF changes "
        f"Conv.\\ {rf_med_c2:.2f} $\\to$ Novel {rf_med_n2:.2f} ({rf_chg2:+.0f}\\%); "
        f"Best Geometric {bg_med_c2:.2f} $\\to$ {bg_med_n2:.2f} ({bg_chg2:+.0f}\\%)---"
        r"pattern unchanged.}"
    )

    table_tex = generate_table(results, novel, conventional, footnote)

    with open(table_path, 'w') as f:
        f.write(table_tex)

    print(f"Written: {table_path}")
    print()
    print("Generated table:")
    print(table_tex)


if __name__ == "__main__":
    main()
