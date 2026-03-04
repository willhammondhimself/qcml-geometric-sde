"""
Permutation test for channel specialization claim.

Tests: Under a null hypothesis of exchangeable method-crisis performance,
what is the probability that 7 geometric channels produce the observed
specialization pattern?

Null model: Shuffle d-values across methods independently per crisis,
preserving the marginal distribution of each crisis.

Test statistic: Number of distinct geometric channel winners across
crisis categories (higher = more specialized).

Also tests: Heterogeneity of the winner matrix -- do different crises
genuinely have different best channels, or could this arise by chance?
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# Crisis categories (a priori, from paper Table 3)
CRISIS_CATEGORIES = {
    'Volatility Shocks': ['2010_flash', '2018_volmageddon', '2018_q4'],
    'Systemic/Credit': ['2008_gfc', '2011_euro', '2023_svb'],
    'Exogenous Shocks': ['2001_911', '2020_covid'],
    'Slow Burns': ['2000_dotcom', '2022_rates', '2024_carry'],
    'Liquidity/Micro.': ['2007_quant', '2019_repo', '2015_china'],
}

GEOMETRIC_METHODS = [
    'Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity',
    'QCML Chern', 'Metric Condition', 'Spectral Gap',
    'Geo. Consensus', 'Geo. Ensemble', 'Geodesic Velocity',
    'Speed Limit Ratio', 'Dimensionality Collapse',
    'Sectional Curvature Sign',
]


def load_d_matrix(json_path):
    """Load d-value matrix from canonical comparison JSON."""
    with open(json_path) as f:
        raw = json.load(f)

    data = raw['results'] if 'results' in raw else raw

    methods = sorted(data.keys())
    crises = sorted(set().union(*[data[m].keys() for m in methods]))

    d_matrix = np.full((len(methods), len(crises)), np.nan)
    for i, m in enumerate(methods):
        for j, c in enumerate(crises):
            entry = data[m].get(c, {})
            d_val = entry.get('d')
            if d_val is not None:
                d_matrix[i, j] = d_val

    return d_matrix, methods, crises


def count_distinct_geo_category_winners(d_matrix, methods, crises):
    """Count how many distinct geometric methods win at least one category."""
    geo_indices = [i for i, m in enumerate(methods) if m in GEOMETRIC_METHODS]
    if not geo_indices:
        return 0

    geo_matrix = d_matrix[geo_indices, :]
    geo_names = [methods[i] for i in geo_indices]

    category_winners = set()
    for cat, cat_crises in CRISIS_CATEGORIES.items():
        cat_cols = [j for j, c in enumerate(crises) if c in cat_crises]
        if not cat_cols:
            continue
        cat_d = np.nanmean(geo_matrix[:, cat_cols], axis=1)
        best_idx = np.nanargmax(cat_d)
        category_winners.add(geo_names[best_idx])

    return len(category_winners)


def winner_heterogeneity(d_matrix, methods, crises):
    """Entropy of the per-crisis winner distribution among geo channels.

    Higher entropy = more diverse winners = stronger specialization signal.
    """
    geo_indices = [i for i, m in enumerate(methods) if m in GEOMETRIC_METHODS]
    if not geo_indices:
        return 0.0

    geo_matrix = d_matrix[geo_indices, :]

    winner_counts = np.zeros(len(geo_indices))
    for j in range(len(crises)):
        col = geo_matrix[:, j]
        if np.all(np.isnan(col)):
            continue
        best_idx = np.nanargmax(col)
        winner_counts[best_idx] += 1

    # Normalize to probability distribution
    total = winner_counts.sum()
    if total == 0:
        return 0.0
    p = winner_counts / total
    p = p[p > 0]
    return float(stats.entropy(p))


def permutation_test(d_matrix, methods, crises, n_perm=10000, seed=42):
    """Run permutation test for specialization pattern.

    Null: Shuffle method labels within each crisis (preserving
    crisis-level d-value distribution).

    Returns:
        dict with test statistics and p-values.
    """
    rng = np.random.RandomState(seed)

    # Observed statistics
    obs_distinct = count_distinct_geo_category_winners(d_matrix, methods, crises)
    obs_entropy = winner_heterogeneity(d_matrix, methods, crises)

    # Permutation distribution
    perm_distinct = np.zeros(n_perm)
    perm_entropy = np.zeros(n_perm)

    for p in range(n_perm):
        # Shuffle method order within each crisis column
        d_perm = d_matrix.copy()
        for j in range(d_matrix.shape[1]):
            col = d_perm[:, j]
            finite_mask = np.isfinite(col)
            finite_vals = col[finite_mask]
            rng.shuffle(finite_vals)
            d_perm[finite_mask, j] = finite_vals

        perm_distinct[p] = count_distinct_geo_category_winners(d_perm, methods, crises)
        perm_entropy[p] = winner_heterogeneity(d_perm, methods, crises)

    # p-values (one-sided: observed >= permuted)
    p_distinct = float(np.mean(perm_distinct >= obs_distinct))
    p_entropy = float(np.mean(perm_entropy >= obs_entropy))

    return {
        'n_permutations': n_perm,
        'observed_distinct_winners': obs_distinct,
        'null_mean_distinct': float(np.mean(perm_distinct)),
        'null_std_distinct': float(np.std(perm_distinct)),
        'p_value_distinct': p_distinct,
        'observed_entropy': obs_entropy,
        'null_mean_entropy': float(np.mean(perm_entropy)),
        'null_std_entropy': float(np.std(perm_entropy)),
        'p_value_entropy': p_entropy,
    }


def main():
    # Find latest canonical JSON
    output_dir = Path('experiments/outputs/regime_detection')
    jsons = sorted(output_dir.glob('causal_comparison_*.json'))
    if not jsons:
        print("No comparison JSONs found.")
        sys.exit(1)

    json_path = jsons[-1]
    print(f"Using: {json_path.name}")

    d_matrix, methods, crises = load_d_matrix(json_path)
    print(f"Matrix: {len(methods)} methods x {len(crises)} crises")
    print(f"Geometric methods found: {sum(1 for m in methods if m in GEOMETRIC_METHODS)}")

    print("\nRunning permutation test (n=10,000)...")
    results = permutation_test(d_matrix, methods, crises, n_perm=10000)

    print("\n=== SPECIALIZATION PERMUTATION TEST ===")
    print(f"Distinct geo category winners: {results['observed_distinct_winners']}")
    print(f"  Null mean: {results['null_mean_distinct']:.2f} "
          f"(std: {results['null_std_distinct']:.2f})")
    print(f"  p-value: {results['p_value_distinct']:.4f}")
    print()
    print(f"Winner heterogeneity (entropy): {results['observed_entropy']:.3f}")
    print(f"  Null mean: {results['null_mean_entropy']:.3f} "
          f"(std: {results['null_std_entropy']:.3f})")
    print(f"  p-value: {results['p_value_entropy']:.4f}")
    print()

    if results['p_value_distinct'] < 0.05:
        print("CONCLUSION: Specialization pattern is statistically significant.")
    else:
        print("CONCLUSION: Specialization pattern is NOT significant at alpha=0.05.")
        print("  Claim should be weakened to 'suggestive evidence'.")

    # Save results
    out_path = output_dir / 'specialization_permutation_test.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
