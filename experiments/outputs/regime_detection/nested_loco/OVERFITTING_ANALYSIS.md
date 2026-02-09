# Overfitting Analysis: Fused QCML Optimization

Generated: 2026-02-08 00:58:32

## Context

The original fused QCML optimization (Phase A median d=1.69, Phase B d=1.79)
optimized hyperparameters on ALL 12 crises simultaneously. This analysis
provides unbiased performance estimates via nested cross-validation.

## Part B: True Temporal Out-of-Sample

- Pre-2020 crises (training): ['2007_quant_meltdown', '2008_crisis', '2010_flash_crash', '2011_debt_downgrade', '2015_china', '2016_brexit', '2018_volmageddon', '2018_fed_selloff', '2019_repo_crisis']
- Post-2020 crises (test): ['2020_covid', '2022_rates', '2023_svb']

- Pre-2020 median d (in-sample): 1.704
- **Post-2020 median d (true OOS): 1.823**
- Overfitting gap: -0.119

### Per-Crisis Post-2020 Results

- 2020_covid: d=1.823
- 2022_rates: d=1.948
- 2023_svb: d=1.795

### Comparison to Biased Temporal OOS

- Biased (params optimized on all 12): post-2020 median d = 1.72
- Unbiased (params optimized on pre-2020 only): post-2020 median d = 1.823
- RF post-2020 median d: 0.88

## Recommendations for Paper

1. Replace misleading 'leave-one-crisis-out cross-validation' claim
   with accurate description of the optimization protocol.
2. Report unbiased nested LOCO d alongside the calibration estimate.
3. Discuss the overfitting gap honestly as evidence of the fusion's
   ability to generalize (if gap is modest) or as a limitation.
4. Update the conclusion to reflect unbiased performance numbers.
