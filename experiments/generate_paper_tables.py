"""
Generate updated LaTeX tables from comparison results JSON.

Reads enhanced_comparison output and produces LaTeX snippets for:
  - Table 4 (main comparison): Method, Median d, Mean Rank, Category
  - Table 5 (per-crisis breakdown): top methods × all crises
  - Table 6 (temporal OOS): pre-2020 vs post-2020
  - Friedman test statistics for Table 4 caption

Also reads nested_loco output (if available) for unbiased HPO estimates.

Usage:
    python experiments/generate_paper_tables.py <comparison_results.json>
    python experiments/generate_paper_tables.py <comparison_results.json> --nested-loco <nested_loco.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


METHOD_CATEGORIES = {
    'Berry Phase Rate': 'Geometric',
    'Berry Curv. Incr.': 'Geometric',
    'QFI Determinant': 'Geometric',
    'Multi-Lag Fidelity': 'Geometric',
    'QCML Chern': 'Geometric',
    'Metric Condition': 'Geometric',
    'Spectral Gap': 'Geometric',
    'Geo. Consensus': 'Geometric',
    'Geo. Ensemble': 'Geometric',
    'Geometric Consensus': 'Geometric',
    'Geometric Ensemble': 'Geometric',
    'Geodesic Velocity': 'Geometric',
    'Speed Limit Ratio': 'Geometric',
    'Dimensionality Collapse': 'Geometric',
    'Sectional Curvature Sign': 'Geometric',
    'Rolling Vol Z': 'Classical',
    'HMM': 'Classical',
    'CUSUM': 'Classical',
    'HMM 2-state': 'Classical',
    'BOCPD': 'Classical',
    'Isolation Forest': 'Classical',
    'Random Forest': 'Supervised',
    'Rolling RF (VIX)': 'Supervised',
    'VIX Level': 'Classical',
}

LATEX_NAMES = {
    'Berry Phase Rate': r"Berry Curv.\ Incr.",
    'QFI Determinant': 'QFI Determinant',
    'Multi-Lag Fidelity': 'Multi-Lag Fidelity',
    'QCML Chern': 'QCML Chern',
    'Metric Condition': 'Metric Condition',
    'Spectral Gap': 'Spectral Gap',
    'Geo. Consensus': r"Geo.\ Consensus",
    'Geo. Ensemble': r"Geo.\ Ensemble",
    'Geodesic Velocity': 'Geodesic Velocity',
    'Speed Limit Ratio': 'Speed Limit Ratio',
    'Dimensionality Collapse': r"Dim.\ Collapse",
    'Sectional Curvature Sign': r"Sect.\ Curv.\ Sign",
    'Rolling Vol Z': 'Rolling Vol Z',
    'CUSUM': 'CUSUM',
    'HMM 2-state': 'HMM 2-state',
    'BOCPD': 'BOCPD',
    'Isolation Forest': 'Isolation Forest',
    'Random Forest': 'Random Forest',
    'Rolling RF (VIX)': 'Rolling RF (VIX)',
    'VIX Level': 'VIX Level',
}


def build_d_matrix(results_dict, crisis_keys=None):
    """Build d-value matrix from comparison results."""
    method_names = list(results_dict.keys())
    if crisis_keys is None:
        # Get all crises from first method
        first_method = method_names[0]
        crisis_keys = sorted(results_dict[first_method].keys())

    n_crises = len(crisis_keys)
    n_methods = len(method_names)
    d_matrix = np.full((n_crises, n_methods), np.nan)

    for j, method in enumerate(method_names):
        for i, crisis in enumerate(crisis_keys):
            if crisis in results_dict[method]:
                entry = results_dict[method][crisis]
                if isinstance(entry, dict) and 'd' in entry:
                    d_matrix[i, j] = entry['d']
                elif isinstance(entry, (int, float)):
                    d_matrix[i, j] = entry

    return d_matrix, method_names, crisis_keys


def generate_table4(d_matrix, method_names, crisis_keys):
    """Generate Table 4: main comparison table."""
    n_crises, n_methods = d_matrix.shape

    # Median d per method
    median_d = np.nanmedian(d_matrix, axis=0)

    # Friedman test
    # Replace NaN with method median for ranking purposes
    d_clean = d_matrix.copy()
    for j in range(n_methods):
        mask = np.isnan(d_clean[:, j])
        d_clean[mask, j] = np.nanmedian(d_clean[:, j])

    ranks = np.zeros_like(d_clean)
    for i in range(n_crises):
        # Rank in DESCENDING order (higher d = better = lower rank)
        ranks[i] = stats.rankdata(-d_clean[i])

    mean_ranks = np.mean(ranks, axis=0)

    # Friedman
    try:
        stat, p = stats.friedmanchisquare(*[d_clean[:, j] for j in range(n_methods)])
    except Exception:
        stat, p = float('nan'), float('nan')

    # Sort by median d descending
    order = np.argsort(-median_d)

    # Find bottom performers (d < 0.1)
    low_cutoff = 0.1

    print(f"% Table 4: {n_methods} methods × {n_crises} crises")
    print(f"% Friedman chi-sq = {stat:.1f}, p = {p:.1e}")
    print()

    lines = []
    lines.append(r"\begin{tabular}{lccl}")
    lines.append(r"\toprule")
    lines.append(r"Method & Median $d$ & Mean Rank & Category \\")
    lines.append(r"\midrule")

    above_cutoff = [i for i in order if median_d[i] >= low_cutoff]
    below_cutoff = [i for i in order if median_d[i] < low_cutoff]

    best_d = max(median_d[above_cutoff]) if above_cutoff else 0

    for idx in above_cutoff:
        name = LATEX_NAMES.get(method_names[idx], method_names[idx])
        cat = METHOD_CATEGORIES.get(method_names[idx], '?')
        d_val = median_d[idx]
        rank_val = mean_ranks[idx]
        bold = r"\textbf{" + f"{d_val:.2f}" + "}" if d_val == best_d else f"{d_val:.2f}"
        lines.append(f"{name:<25} & {bold} & {rank_val:.1f} & {cat} \\\\")

    if below_cutoff:
        lines.append(r"\midrule")
        for idx in below_cutoff:
            name = LATEX_NAMES.get(method_names[idx], method_names[idx])
            cat = METHOD_CATEGORIES.get(method_names[idx], '?')
            lines.append(f"{name:<25} & {median_d[idx]:.2f} & {mean_ranks[idx]:.1f} & {cat} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))
    print()

    # Print caption stats
    year_range = f"{crisis_keys[0][:4]}--{crisis_keys[-1][:4]}"
    print(f"% Caption: {n_methods} methods across {n_crises} crises ({year_range}).")
    print(f"% Friedman chi^2 = {stat:.1f}, p < {p:.0e}")
    return stat, p


def generate_table5(d_matrix, method_names, crisis_keys):
    """Generate Table 5: per-crisis breakdown for top methods."""
    top_methods = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity',
                   'Random Forest', 'CUSUM']
    top_indices = []
    for m in top_methods:
        if m in method_names:
            top_indices.append(method_names.index(m))

    actual_names = [method_names[i] for i in top_indices]

    # Header
    short = {'Berry Phase Rate': 'Berry', 'QFI Determinant': 'QFI Det.',
             'Multi-Lag Fidelity': 'Multi-Lag', 'Random Forest': 'RF', 'CUSUM': 'CUSUM'}

    header_names = [short.get(n, n[:8]) for n in actual_names]

    print("% Table 5: per-crisis breakdown")
    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(top_indices) + "}")
    lines.append(r"\toprule")
    lines.append("Crisis & " + " & ".join(header_names) + r" \\")
    lines.append(r"\midrule")

    # Pretty crisis names
    crisis_pretty = {
        '1997_asia': '1997 Asia',
        '1998_ltcm': '1998 LTCM',
        '2000_dotcom': '2000 Dotcom',
        '2001_911': '2001 9/11',
        '2001911': '2001 9/11',
        '2007_quant': '2007 Quant',
        '2008_gfc': '2008 GFC',
        '2010_flash': '2010 Flash',
        '2011_euro': '2011 Euro',
        '2015_china': '2015 China',
        '2018_volmageddon': '2018 Volmag.',
        '2018_q4': '2018 Q4',
        '2019_repo': '2019 Repo',
        '2020_covid': '2020 COVID',
        '2022_rates': '2022 Rates',
        '2023_svb': '2023 SVB',
        '2024_carry': '2024 Carry',
    }

    rf_idx = method_names.index('Random Forest') if 'Random Forest' in method_names else -1
    geo_indices = [method_names.index(m) for m in ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity']
                   if m in method_names]

    for i, crisis in enumerate(crisis_keys):
        name = crisis_pretty.get(crisis, crisis)
        vals = []
        # Check if best geometric beats RF
        rf_d = d_matrix[i, rf_idx] if rf_idx >= 0 else -1
        best_geo_d = max(d_matrix[i, gi] for gi in geo_indices) if geo_indices else -1
        best_geo_idx = geo_indices[np.argmax([d_matrix[i, gi] for gi in geo_indices])] if geo_indices else -1

        for idx in top_indices:
            d_val = d_matrix[i, idx]
            if np.isnan(d_val):
                vals.append("--")
            elif idx in geo_indices and idx == best_geo_idx and best_geo_d > rf_d and rf_idx >= 0:
                vals.append(r"\textbf{" + f"{d_val:.2f}" + "}")
            else:
                vals.append(f"{d_val:.2f}")
        lines.append(f"{name} & " + " & ".join(vals) + r" \\")

    # Median row
    lines.append(r"\midrule")
    medians = []
    for idx in top_indices:
        med = np.nanmedian(d_matrix[:, idx])
        medians.append(f"{med:.2f}")
    lines.append("Median & " + " & ".join(medians) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))

    # Count wins
    if rf_idx >= 0 and geo_indices:
        wins = 0
        for i in range(len(crisis_keys)):
            best_geo = max(d_matrix[i, gi] for gi in geo_indices)
            if best_geo > d_matrix[i, rf_idx]:
                wins += 1
        print(f"\n% Best geometric exceeds RF on {wins}/{len(crisis_keys)} crises")


def generate_table6(d_matrix, method_names, crisis_keys):
    """Generate Table 6: temporal stability (pre-2020 vs post-2020)."""
    pre_2020 = [i for i, k in enumerate(crisis_keys) if int(k[:4]) < 2020]
    post_2020 = [i for i, k in enumerate(crisis_keys) if int(k[:4]) >= 2020]

    methods_of_interest = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity',
                           'CUSUM', 'Random Forest']

    print("% Table 6: temporal stability")
    lines = []
    lines.append(r"\begin{tabular}{lcc" + "c" * len(post_2020) + "}")
    lines.append(r"\toprule")

    # Get post-2020 crisis names
    crisis_pretty = {
        '2020_covid': 'COVID', '2022_rates': 'Rates', '2023_svb': 'SVB', '2024_carry': 'Carry',
    }
    post_names = [crisis_pretty.get(crisis_keys[i], crisis_keys[i][:4]) for i in post_2020]
    lines.append("Method & Pre-2020 & Post-2020 & " + " & ".join(post_names) + r" \\")
    lines.append(r"\midrule")

    latex_short = {
        'Berry Phase Rate': r"Berry Curv.\ Incr.",
        'QFI Determinant': 'QFI Determinant',
        'Multi-Lag Fidelity': 'Multi-Lag Fidelity',
        'CUSUM': 'CUSUM',
        'Random Forest': 'Random Forest',
    }

    for method in methods_of_interest:
        if method not in method_names:
            continue
        j = method_names.index(method)
        pre_mean = np.nanmean(d_matrix[pre_2020, j])
        post_mean = np.nanmean(d_matrix[post_2020, j])
        post_vals = [f"{d_matrix[i, j]:.2f}" for i in post_2020]
        name = latex_short.get(method, method)
        lines.append(f"{name} & {pre_mean:.2f} & {post_mean:.2f} & " + " & ".join(post_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from comparison results")
    parser.add_argument("input", help="Path to comparison results JSON")
    parser.add_argument("--nested-loco", help="Path to nested LOCO results JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results_dict = data["results"]
    d_matrix, method_names, crisis_keys = build_d_matrix(results_dict)

    print(f"Loaded: {len(method_names)} methods × {len(crisis_keys)} crises")
    print(f"Methods: {method_names}")
    print(f"Crises: {crisis_keys}")
    print()

    print("=" * 70)
    print("TABLE 4: MAIN COMPARISON")
    print("=" * 70)
    generate_table4(d_matrix, method_names, crisis_keys)
    print()

    print("=" * 70)
    print("TABLE 5: PER-CRISIS BREAKDOWN")
    print("=" * 70)
    generate_table5(d_matrix, method_names, crisis_keys)
    print()

    print("=" * 70)
    print("TABLE 6: TEMPORAL STABILITY")
    print("=" * 70)
    generate_table6(d_matrix, method_names, crisis_keys)

    if args.nested_loco:
        print()
        print("=" * 70)
        print("NESTED LOCO-CV RESULTS (UNBIASED)")
        print("=" * 70)
        with open(args.nested_loco) as f:
            loco_data = json.load(f)

        for key, r in loco_data['results'].items():
            s = r['summary']
            print(f"  {r['name']:<25} ({r['type']}): "
                  f"median d = {s['median_d']:.3f}, "
                  f"mean d = {s['mean_d']:.3f} ± {s['std_d']:.3f}")


if __name__ == "__main__":
    main()
