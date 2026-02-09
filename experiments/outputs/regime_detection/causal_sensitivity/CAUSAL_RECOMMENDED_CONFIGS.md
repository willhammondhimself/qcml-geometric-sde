# Causal Sensitivity Analysis: Recommended Configurations

Generated: 2026-02-08 22:54:41

## Context

These configurations are optimized under CAUSAL constraints: PCA, scaler,
and operators are fit only on pre-crisis data (no lookahead). This may
yield different optimal hyperparameters than the non-causal sweep.

## Berry Phase Rate

**Causal-optimal config:**
- hilbert_dim: 12
- n_pca_components: 20
- operator_method: random
- rolling_window: 30
- Mean Cohen's d (12 crises, causal): 1.627
- Median Cohen's d: 1.069
- Std: 1.285, CV: 79.0%

**Comparison to non-causal optimal (h8_p10_pca_inspired_rw10):**
- Same config: NO (different optimal under causal constraints)
- Causal-optimal: h12_p20_random_rw30
- Non-causal-optimal: h8_p10_pca_inspired_rw10

**Top 5 causal configs (Stage 2 validation):**

| Config | Mean d | Median d | Std | CV% |
|--------|--------|----------|-----|-----|
| h12_p20_random_rw30 | 1.627 | 1.069 | 1.285 | 79.0 |
| h8_p8_random_rw20 | 1.602 | 1.541 | 1.006 | 62.8 |
| h16_p20_random_rw20 | 1.313 | 0.967 | 1.118 | 85.2 |
| h16_p15_random_rw30 | 1.283 | 1.069 | 0.895 | 69.8 |
| h16_p20_random_rw30 | 1.256 | 1.003 | 1.033 | 82.2 |

## QFI Determinant

**Causal-optimal config:**
- hilbert_dim: 12
- n_pca_components: 15
- operator_method: pca_inspired
- rolling_window: 20
- Mean Cohen's d (12 crises, causal): 1.352
- Median Cohen's d: 1.162
- Std: 0.913, CV: 67.5%

**Comparison to non-causal optimal (h4_p15_random_rw30):**
- Same config: NO (different optimal under causal constraints)
- Causal-optimal: h12_p15_pca_inspired_rw20
- Non-causal-optimal: h4_p15_random_rw30

**Top 5 causal configs (Stage 2 validation):**

| Config | Mean d | Median d | Std | CV% |
|--------|--------|----------|-----|-----|
| h12_p15_pca_inspired_rw20 | 1.352 | 1.162 | 0.913 | 67.5 |
| h12_p15_pca_inspired_rw30 | 1.264 | 1.182 | 0.926 | 73.3 |
| h12_p8_pca_inspired_rw20 | 1.195 | 0.720 | 1.121 | 93.8 |
| h12_p10_pca_inspired_rw20 | 1.138 | 0.776 | 0.956 | 84.0 |
| h12_p8_pca_inspired_rw30 | 1.107 | 0.903 | 1.073 | 97.0 |

## Multi-Lag Fidelity

**Causal-optimal config:**
- hilbert_dim: 12
- n_pca_components: 20
- operator_method: pca_inspired
- rolling_window: 20
- Mean Cohen's d (12 crises, causal): 1.406
- Median Cohen's d: 1.320
- Std: 1.169, CV: 83.1%

**Comparison to non-causal optimal (h8_p15_pca_inspired_rw20):**
- Same config: NO (different optimal under causal constraints)
- Causal-optimal: h12_p20_pca_inspired_rw20
- Non-causal-optimal: h8_p15_pca_inspired_rw20

**Top 5 causal configs (Stage 2 validation):**

| Config | Mean d | Median d | Std | CV% |
|--------|--------|----------|-----|-----|
| h12_p20_pca_inspired_rw20 | 1.406 | 1.320 | 1.169 | 83.1 |
| h12_p15_pca_inspired_rw20 | 1.373 | 1.345 | 1.137 | 82.8 |
| h12_p10_pca_inspired_rw20 | 1.251 | 1.414 | 1.011 | 80.8 |
| h12_p20_pca_inspired_rw10 | 1.119 | 0.720 | 1.156 | 103.3 |
| h12_p15_pca_inspired_rw10 | 1.110 | 0.699 | 1.100 | 99.2 |

## Operator Method Comparison (Causal)

Aggregated across all hilbert_dim, n_pca, rolling_window:

**Berry Phase Rate:**
- random: mean_d=1.091 (n=48 configs)
- pca_inspired: mean_d=0.993 (n=48 configs)
- pca_inspired advantage: -0.098

**QFI Determinant:**
- random: mean_d=0.934 (n=48 configs)
- pca_inspired: mean_d=1.236 (n=48 configs)
- pca_inspired advantage: +0.302

**Multi-Lag Fidelity:**
- random: mean_d=0.961 (n=48 configs)
- pca_inspired: mean_d=1.300 (n=48 configs)
- pca_inspired advantage: +0.339
