# Extended Sensitivity Analysis: Recommended Configurations

Generated: 2026-02-08 00:22:56

## Summary

This analysis extends the original sensitivity sweep (hilbert_dim x n_pca)
to also vary `operator_method` (random vs pca_inspired) and `rolling_window`
(10, 20, 30). The fused Optuna optimization consistently found
`pca_inspired` + `hilbert_dim=12` superior to the defaults.

## Berry Phase Rate

**Recommended config:**
- hilbert_dim: 8
- n_pca_components: 10
- operator_method: pca_inspired
- rolling_window: 10
- Mean Cohen's d (12 crises): 1.314
- Median Cohen's d: 0.752
- Std: 1.807, CV: 137.5%

**Improvement over default (h8/p15/random/rw20):**
- Default mean d: 1.076
- Improved mean d: 1.314
- Delta: +0.237

**Top 5 configs (Stage 2 validation):**

| Config | Mean d | Median d | Std | CV% |
|--------|--------|----------|-----|-----|
| h8_p10_pca_inspired_rw10 | 1.314 | 0.752 | 1.807 | 137.5 |
| h8_p8_random_rw30 | 1.308 | 1.203 | 0.714 | 54.6 |
| h8_p15_pca_inspired_rw10 | 1.146 | 0.656 | 1.600 | 139.6 |
| h8_p8_pca_inspired_rw20 | 1.128 | 0.750 | 1.643 | 145.7 |
| h4_p15_random_rw30 | 1.112 | 0.963 | 0.851 | 76.6 |

## QFI Determinant

**Recommended config:**
- hilbert_dim: 4
- n_pca_components: 15
- operator_method: random
- rolling_window: 30
- Mean Cohen's d (12 crises): 1.415
- Median Cohen's d: 1.137
- Std: 0.821, CV: 58.0%

**Improvement over default (h8/p15/random/rw20):**
- Default mean d: 1.387
- Improved mean d: 1.415
- Delta: +0.028

**Top 5 configs (Stage 2 validation):**

| Config | Mean d | Median d | Std | CV% |
|--------|--------|----------|-----|-----|
| h4_p15_random_rw30 | 1.415 | 1.137 | 0.821 | 58.0 |
| h16_p20_pca_inspired_rw20 | 1.200 | 0.832 | 1.153 | 96.0 |
| h8_p15_random_rw30 | 1.088 | 0.936 | 0.742 | 68.2 |
| h16_p10_pca_inspired_rw10 | 1.031 | 0.702 | 1.021 | 99.0 |
| h16_p15_pca_inspired_rw20 | 1.006 | 0.617 | 1.100 | 109.4 |

## Multi-Lag Fidelity

**Recommended config:**
- hilbert_dim: 8
- n_pca_components: 15
- operator_method: pca_inspired
- rolling_window: 20
- Mean Cohen's d (12 crises): 1.258
- Median Cohen's d: 0.992
- Std: 1.164, CV: 92.5%

**Improvement over default (h8/p15/random/rw20):**
- Default mean d: 0.634
- Improved mean d: 1.258
- Delta: +0.624

**Top 5 configs (Stage 2 validation):**

| Config | Mean d | Median d | Std | CV% |
|--------|--------|----------|-----|-----|
| h8_p15_pca_inspired_rw20 | 1.258 | 0.992 | 1.164 | 92.5 |
| h8_p10_pca_inspired_rw20 | 1.186 | 0.918 | 1.342 | 113.1 |
| h8_p20_pca_inspired_rw10 | 1.177 | 0.830 | 1.599 | 135.8 |
| h8_p15_pca_inspired_rw10 | 1.148 | 0.933 | 1.435 | 125.0 |
| h8_p10_pca_inspired_rw10 | 1.045 | 0.787 | 1.255 | 120.1 |

## Operator Method Comparison

Aggregated across all hilbert_dim, n_pca, rolling_window:

**Berry Phase Rate:**
- random: mean_d=0.925 (n=48 configs)
- pca_inspired: mean_d=0.987 (n=48 configs)
- pca_inspired advantage: +0.062

**QFI Determinant:**
- random: mean_d=1.045 (n=48 configs)
- pca_inspired: mean_d=1.085 (n=48 configs)
- pca_inspired advantage: +0.040

**Multi-Lag Fidelity:**
- random: mean_d=0.866 (n=48 configs)
- pca_inspired: mean_d=1.092 (n=48 configs)
- pca_inspired advantage: +0.226
