#!/usr/bin/env python3
"""
Visualize Optuna Causal Regime Detection Results

Creates publication-quality figures showing:
1. 6-condition comparison across 3 methods (bar chart with significance stars)
2. Per-crisis heatmap showing method performance
3. Phase A vs Phase B trade-off analysis
4. Method ranking across conditions

Author: QCML Research
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

# Use publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
})

# Load results
RESULTS_FILE = "experiments/outputs/regime_detection/causal_optimized/causal_eval_optuna_20260208_141406.json"
OUTPUT_DIR = Path("experiments/outputs/regime_detection/optuna_causal/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

results = data['results']
tests = data['statistical_tests']

# Extract data
methods = ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity']
conditions = [
    '1_original_causal',
    '2_causal_optimized',
    '2b_optuna_phase_b',
    '3_expanding_20',
    '4_expanding_30',
]
condition_labels = [
    'Original\n(causal fit)',
    'Phase A\n(optimized)',
    'Phase B\n(expand win)',
    'Expand-20\n(Phase A)',
    'Expand-30\n(Phase A)',
]

rf_results = results['Random Forest']['5_rf_baseline']
rf_mean = np.nanmean(list(rf_results.values()))

# Crisis names
crisis_names = list(next(iter(next(iter(results.values())).values())).keys())


def get_mean_d(method: str, condition: str) -> float:
    """Get mean Cohen's d for a method-condition pair."""
    d_values = results.get(method, {}).get(condition, {})
    return np.nanmean(list(d_values.values())) if d_values else 0.0


def get_wilcoxon_p(method: str, condition: str) -> float:
    """Get Wilcoxon p-value vs RF."""
    return tests.get(method, {}).get(condition, {}).get('wilcoxon_p', 1.0)


# Figure 1: 6-Condition Comparison (3x2 subplots)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Optuna Causal Regime Detection: 6-Condition Comparison', fontweight='bold')

x = np.arange(len(conditions))
width = 0.6

for idx, method in enumerate(methods):
    ax = axes[idx]

    # Get mean d values and p-values
    d_values = [get_mean_d(method, c) for c in conditions]
    p_values = [get_wilcoxon_p(method, c) for c in conditions]

    # Bar colors based on significance
    colors = ['#2ecc71' if p < 0.05 else '#3498db' if p < 0.1 else '#95a5a6'
              for p in p_values]

    # Create bars
    bars = ax.bar(x, d_values, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add RF baseline
    ax.axhline(rf_mean, color='red', linestyle='--', linewidth=2, label=f'RF (d={rf_mean:.2f})')

    # Add significance stars
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        elif p < 0.1:
            star = '†'
        else:
            star = ''

        if star:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   star, ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height/2,
               f'{height:.2f}', ha='center', va='center', fontweight='bold',
               color='white', fontsize=9)

    ax.set_ylabel("Cohen's d (effect size)", fontweight='bold')
    ax.set_title(method, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=0, ha='center', fontsize=8)
    ax.set_ylim(0, max(d_values) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)

    # Add spines only on left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add legend for significance
legend_elements = [
    mpatches.Patch(color='#2ecc71', label='p < 0.05 (significant)'),
    mpatches.Patch(color='#3498db', label='0.05 ≤ p < 0.1 (marginal)'),
    mpatches.Patch(color='#95a5a6', label='p ≥ 0.1 (n.s.)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
          bbox_to_anchor=(0.5, -0.05), frameon=True, fancybox=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'optuna_6condition_comparison.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'optuna_6condition_comparison.png', bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_DIR / 'optuna_6condition_comparison.pdf'}")
plt.close()


# Figure 2: Per-Crisis Heatmap (Best Condition per Method)
fig, ax = plt.subplots(figsize=(12, 8))

# Get best d value per method per crisis
heatmap_data = np.zeros((len(methods) + 1, len(crisis_names)))  # +1 for RF

for i, method in enumerate(methods):
    for j, crisis in enumerate(crisis_names):
        # Find best condition for this crisis
        best_d = max([
            results[method].get(cond, {}).get(crisis, 0.0)
            for cond in conditions
        ])
        heatmap_data[i, j] = best_d

# Add RF row
for j, crisis in enumerate(crisis_names):
    heatmap_data[-1, j] = rf_results.get(crisis, 0.0)

# Create heatmap
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=5)

# Set ticks
ax.set_xticks(np.arange(len(crisis_names)))
ax.set_yticks(np.arange(len(methods) + 1))
ax.set_xticklabels([c.replace('_', ' ').title() for c in crisis_names],
                   rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(methods + ['RF Baseline'], fontsize=10, fontweight='bold')

# Add text annotations
for i in range(len(methods) + 1):
    for j in range(len(crisis_names)):
        value = heatmap_data[i, j]
        color = 'white' if value > 2.5 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
               color=color, fontweight='bold', fontsize=8)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Cohen's d (effect size)", rotation=270, labelpad=20, fontweight='bold')

ax.set_title('Per-Crisis Performance Heatmap (Best QCML Condition vs RF)',
            fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'optuna_crisis_heatmap.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'optuna_crisis_heatmap.png', bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_DIR / 'optuna_crisis_heatmap.pdf'}")
plt.close()


# Figure 3: Phase A vs Phase B Trade-off Analysis
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Phase A (Single-Shot) vs Phase B (Expanding Window) Trade-off',
            fontweight='bold')

for idx, method in enumerate(methods):
    ax = axes[idx]

    phase_a_d = get_mean_d(method, '2_causal_optimized')
    phase_b_d = get_mean_d(method, '2b_optuna_phase_b')

    # Per-crisis comparison
    phase_a_vals = [results[method]['2_causal_optimized'].get(c, 0.0) for c in crisis_names]
    phase_b_vals = [results[method]['2b_optuna_phase_b'].get(c, 0.0) for c in crisis_names]

    x_pos = np.arange(len(crisis_names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, phase_a_vals, width, label='Phase A (single-shot)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, phase_b_vals, width, label='Phase B (expanding win)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add RF baseline
    ax.axhline(rf_mean, color='gray', linestyle='--', linewidth=1.5,
              label=f'RF (d={rf_mean:.2f})', alpha=0.7)

    ax.set_ylabel("Cohen's d", fontweight='bold')
    ax.set_title(f'{method}\n(Phase A: d={phase_a_d:.2f}, Phase B: d={phase_b_d:.2f})',
                fontweight='bold', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace('_', ' ')[:10] for c in crisis_names],
                       rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'optuna_phase_a_vs_b.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'optuna_phase_a_vs_b.png', bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_DIR / 'optuna_phase_a_vs_b.pdf'}")
plt.close()


# Figure 4: Method Ranking Across Conditions (Friedman-style)
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate mean rank for each method-condition pair
# Lower rank = better performance
rank_matrix = np.zeros((len(methods), len(conditions)))

for crisis_idx, crisis in enumerate(crisis_names):
    # Get all d values for this crisis
    crisis_d = []
    for method in methods:
        for cond in conditions:
            d = results[method].get(cond, {}).get(crisis, 0.0)
            crisis_d.append(d)

    # Rank them (higher d = better = lower rank)
    ranks = np.argsort(np.argsort(-np.array(crisis_d)))  # Descending

    # Assign ranks to matrix
    rank_idx = 0
    for i, method in enumerate(methods):
        for j, cond in enumerate(conditions):
            rank_matrix[i, j] += ranks[rank_idx]
            rank_idx += 1

# Average ranks across crises
rank_matrix /= len(crisis_names)

# Plot as grouped bar chart
x = np.arange(len(conditions))
width = 0.25

for i, method in enumerate(methods):
    offset = (i - 1) * width
    ax.bar(x + offset, rank_matrix[i], width, label=method, alpha=0.8,
           edgecolor='black', linewidth=0.8)

ax.set_ylabel('Mean Rank Across Crises\n(lower = better)', fontweight='bold')
ax.set_xlabel('Condition', fontweight='bold')
ax.set_title('Method Ranking Across Conditions (Friedman-style)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(condition_labels, rotation=0, ha='center', fontsize=9)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'optuna_method_ranking.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'optuna_method_ranking.png', bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_DIR / 'optuna_method_ranking.pdf'}")
plt.close()


# Summary statistics table
print("\n" + "="*80)
print("SUMMARY STATISTICS TABLE")
print("="*80)
print(f"\n{'Method':<25} {'Condition':<20} {'Mean d':>10} {'vs RF':>10} {'p-value':>10} {'Sig':>5}")
print("-" * 80)

for method in methods:
    for i, cond in enumerate(conditions):
        mean_d = get_mean_d(method, cond)
        p_val = get_wilcoxon_p(method, cond)
        delta = mean_d - rf_mean
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else ''

        cond_label = condition_labels[i].replace('\n', ' ')
        print(f"{method:<25} {cond_label:<20} {mean_d:>10.3f} {delta:>+10.3f} {p_val:>10.4f} {sig:>5}")
    print("-" * 80)

print(f"\n{'Random Forest Baseline':<25} {'LOCO':<20} {rf_mean:>10.3f}")
print("="*80)
print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1")
print(f"\nFigures saved to: {OUTPUT_DIR}/")
print("="*80)
