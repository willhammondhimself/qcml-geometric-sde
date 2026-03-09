# Research Ideation Swarm — Full Synthesis Report

**Date**: 2026-03-08
**Rounds**: 120
**Protocol**: 3–4 Knights per question, smoke test on 4 crises (GFC, COVID, 2022, SVB), keep if d > 0.3

---

## Executive Summary

The 120-round ideation swarm systematically explored the QCML geometric observatory framework across 11 research themes. Of 120 questions, **82 were kept** (productive directions), **29 rejected** (dead ends), and **9 flagged for further investigation**. The most impactful discoveries are embedding-level improvements that provide multiplicative gains across all observables: higher Hilbert dimension (d=16, +131%) and soft normalization (+0.44d). The strongest new fusion method is hierarchical clustering (d=1.775), and the strongest single detector overall is **ground state energy E_0(x)** (Q107, d=1.411), discovered in the intrinsic dimension round. The spectral gap ratio (Q102, d=1.175) also outperforms the existing IPR benchmark. Several important corrections emerged: Berry Phase Rate lead time is 4 days (not 90), HPO bias is +0.415d, Reduced Purity is unstable on holdout, and the QCML embedding degrades dimension signals by ~0.37 d-units vs classical PCA (Absorption Ratio).

---

## Per-Theme Findings

### Theme 1: Novel Observables (Q1–Q15) — 11 keep, 3 reject, 1 investigate

The richest theme. **OTOC** (Q2, d=1.121) is the standout — a quantum chaos observable that detects financial "scrambling" with a genuinely novel mechanism. **Loschmidt echo** (Q1, d=0.849) and **Wasserstein distance** (Q10, d=0.791) are strong additions. **TDA** (Q9, d=0.656) and **Thouless energy** (Q11, d=0.652) provide complementary geometric perspectives. **Husimi Q-function** (Q8, d=0.541) and **partition function** (Q7, d=0.537) add moderate value. Dead: Chern number requires too many dimensions (Q3), QGT off-diagonal is redundant (Q5), quantum discord is computationally equivalent to mutual information at qubit level (Q14).

### Theme 2: Embedding Architecture (Q16–Q25) — 2 keep, 5 reject, 3 investigate

Low keep rate but the two keeps are transformative. **Hilbert dim=16** (Q16, d=1.327) provides +131% improvement by avoiding Kramers degeneracy — the single highest-impact finding for the existing pipeline. **Soft normalization** (Q24, d=1.144) is a free +0.44d win by replacing sphere normalization. Most other embedding changes (Takens Q18, random features Q19, multi-qubit Q20, gauge-equivariant Q23) were rejected as either impractical or not beneficial. Neural embedding (Q17), Poincaré disk (Q21), and kernel alignment (Q25) merit further investigation.

### Theme 3: Dead Signal Resurrection (Q26–Q35) — 8 keep, 2 reject

Surprisingly productive. **8 of 10 "dead" signals were resurrectable** with simple parameter changes:
- **Geometric Ensemble** (Q30): d=0.013 → d=1.690 by curating to top-3 channels only
- **BOCPD** (Q31): d=0.057 → d=0.898 with shorter hazard prior (λ=100)
- **Transfer Entropy** (Q32): d=0.285 → d=0.763 with bug fix + finer binning
- **Spectral Flow** (Q35): ~0 → d=0.762 with L1 velocity metric
- **Kernel PCA** (Q34): d=0.035 → d=0.669 with gamma tuning
- **Curvature Rate** (Q28): d=0.013 → d=0.539 with sign-change (Ricci flow-inspired)
- **Geometric Consensus** (Q29): d=0.075 → d=0.692 with OR logic instead of AND

Truly dead: QGT Phase Rigidity (Q26, Kramers degeneracy), Berry Velocity Coupling (Q27, algebraically zero at qubit level).

### Theme 4: Fusion & Combination (Q36–Q45) — 8 keep, 2 reject

**Hierarchical clustering** (Q45, d=1.775) is the best fusion method discovered — within-cluster denoising followed by equal cross-cluster weighting. **Hedge/MW online learning** (Q39, d=0.842) beats Regime-Adaptive with zero training, making it the best online fusion. **Switching model** (Q42, d=1.26) and **MI-weighted fusion** (Q44, d=1.009) also strong. **Majority vote** (Q37, d=0.525) is a useful simplicity baseline. Rejected: attention mechanism (Q36, worse than uniform weights), stacking/GBM (Q41, catastrophic overfitting).

### Theme 5: Statistical & Evaluation (Q46–Q55) — 9 keep, 1 reject

Critical methodological findings. **Granger causality confirmed** (Q49, F=14–55) — our observables genuinely predict realized volatility. **HPO bias quantified** (Q54, +0.415d) — material for all paper claims using optimized parameters. **Calibration analysis** (Q55) revealed high-d detectors are anti-calibrated (predicted probability poorly maps to actual crisis frequency). **Detection delay** (Q46) validated as complementary metric. **Window size** (Q52) affects rankings — dim_collapse optimal at 126 days, spectral_entropy at 252. Only reject: LOCO-CV (Q51, redundant with existing holdout design).

### Theme 6: Cross-Asset & Data Expansion (Q56–Q65) — 8 keep, 2 reject

Strong generalization story. **Multi-market embedding** (Q63, d=1.375) is the best cross-asset approach — joint SPY+TLT+GLD geometry. **Market breadth** (Q65, d=1.366) and **regional contagion** (Q64, d=0.816) both strong. **Bonds** (Q56, d=0.619) and **commodities** (Q58, d=0.54) generalize well. **Crypto** (Q57, d=0.387) works but weaker. **FX** (Q60, d=0.282) is weakest — possibly due to mean-reverting dynamics. Rejected: intraday data (Q61, requires fundamentally different pipeline), VIX term structure embedding (Q62, adds complexity without improvement).

### Theme 7: Theoretical Foundations (Q66–Q75) — 9 keep, 1 reject

Strongest theoretical theme. Key results:
- **Berry curvature diverges as Δ⁻²** near level crossings (Q66) — rigorous mathematical basis
- **Adiabatic theorem applies**: speed limit ratio r < 0.1 always (Q67) — validates geometric interpretation
- **QGT is NOT sufficient statistic** for regime detection (Q68) — spectral entropy captures information beyond QGT
- **All observables are gauge-invariant** (Q69) — confirmed via explicit gauge transformation
- **Detection delay lower-bounded** by quantum speed limit (Q71)
- **Crises are crossovers, not phase transitions** (Q73) — spectral gap never closes
- **FDT does not apply** (Q74, rejected) — financial systems too far from equilibrium

### Theme 8: Practical Applications (Q76–Q85) — 8 keep, 2 reject

Mixed results. **Berry Phase Rate lead time corrected** to 4 days median in walk-forward (Q77, was claimed as 90 days). **Dashboard feasible** at 2.3s/day latency (Q78). **Cost reduction possible**: dim=8 retains d>0.5 at 4x speedup (Q79). **Combined vol+regime** is best drawdown protection (Q82, 31% max drawdown reduction). **Geometric fear index** possible but doesn't beat VIX where options exist (Q83). Rejected: risk parity regime overlay (Q80, weight renormalization neutralizes signal), protective put timing (Q84, insufficient precision).

### Theme 9: Competitive Landscape (Q86–Q95) — 8 keep, 2 reject

Important positioning results. **VIX outperforms** (Q93, d=1.631) but requires options market — our advantage is universality. **Absorption ratio** (Q90, d=0.962) validates DimensionalityCollapse from a different angle. **Turbulence index** (Q92, d=1.329) is a strong classical competitor. **Hurst exponent** (Q89, d=0.637) captures complementary multifractal structure. **LightGBM stacking** (Q87) adds +0.15d over raw observables but risks overfitting. **Network risk measures** (Q88, CoVaR/MES/SRISK) are complementary for systemic risk. Rejected: VAEs (Q94, same data starvation as LSTM), GP changepoint (Q95, O(n³) intractable).

### Theme 10: Wild Cards (Q96–Q100) — 4 keep, 1 reject

Exploratory but productive. **Near-miss detection** (Q96) — entropy fires at 35% intensity for near-miss events, validating continuous stress response. **FOMC periodicity** (Q97) — meetings suppress entropy (p=0.014), novel finding. **Severity classification** (Q98) — promising direction (rho=-0.800) but underpowered at n=4. **Flash crash precursors** (Q99) — 2010 has 1-week Berry precursor but n=1. Rejected: universal critical exponent (Q100) — crises are crossovers, not phase transitions, consistent with Q73.

### Theme 11: Intrinsic Dimension (Q101–Q120) — 7 keep, 8 reject, 5 investigate

Motivated by Candelori et al. (2025): QCML quantum metric eigenvalues reveal intrinsic dimension via spectral gap, noise-robust unlike MLE/TwoNN. Explored dimension estimation as regime indicator, reconstruction-loss operators, dimension-informed fusion, noise filtering, theoretical connections, and practical extensions.

**Headline finding**: **Ground state energy E_0(x)** (Q107, d=1.411) is the strongest single detector in the entire project — surpassing Reduced Purity (d=0.835) by 69%. It is operator-independent, computationally trivial (smallest eigenvalue of H(x)), and shows d > 1.0 on all 4 smoke crises. Key caveat: may be a realized-volatility proxy (correlation check needed).

**Dimension as regime indicator (Q101-Q104)**: Intrinsic dimension detectors work (Q101 IntrinsicDim d=0.633, Q102 SpectralGapRatio d=1.175, Q104 EffectiveDim d=0.422) but **all underperform classical Absorption Ratio** (d=0.797). The QCML embedding degrades dimension signals by ~0.37 d-units vs. classical PCA. Δd/Δt rate-of-change (Q103, d=0.269) fails threshold — derivative of discrete dimension is ill-defined. The SpectralGapRatio (Q102) is the standout: d=1.175 with strong RMT connections, though an extreme d=3.97 on 2022 rates needs verification.

**Reconstruction-loss operators (Q105-Q107)**: The coordinate-descent optimizer is **non-functional** (loss delta < 1e-8 over 100 steps — operators unchanged from random initialization). Reconstruction-trained operators ≡ random operators empirically (Q105 reject). Fluctuation weight w∈[0.1,0.2] is inert when the optimizer fails (Q106 reject). However, **E_0(x) itself** — the optimization target — turns out to be an excellent detection signal even with random operators (Q107 keep, d=1.411). This serendipitous finding suggests the quantum structure may be irrelevant; any quadratic form on features could work similarly.

**Dimension-informed fusion (Q108-Q110)**: **Fundamental type mismatch** kills the elegant idea. Quantum metric eigenvectors live in R^8 (feature space), while fusion channels are R^16 (channel space). There is no well-defined projection from one to the other (Q108, Q109 reject). A simpler scalar gate d(t)/d_max to modulate ensemble confidence is viable but has small ceiling (Q110 investigate — estimated +0.02-0.05d improvement).

**Noise filtering (Q111-Q113)**: QCML dimension estimation correctly identifies risk factors only with careful operator choice (Q111 investigate — operator-dependent, degraded vs AR). Normal-direction eigenvalues are embedding artifacts, not noise (Q112 reject — cannot subtract them). Marchenko-Pastur thresholding is **inapplicable** because g_ab is NOT a sample covariance matrix (Q113 reject — no sample size T, no i.i.d. structure).

**Theoretical connections (Q114-Q116)**: The strongest sub-theme. **Renyi entropy unification** (Q114 keep): IPR = Renyi-∞, PR = Renyi-2, Spectral Entropy = Renyi-1 of the eigenvalue distribution. For sharp spectra these are equivalent; for noisy spectra IPR is more robust (explains its superior detection). **Financial interpretation** (Q115 keep): quantum metric rank as "number of active risk factors" is defensible but weak — the factors are entangled with the embedding, not the market. **Herding → dimension collapse** (Q116 keep): sufficient in the large-N limit (herding ⟹ ρ→ρ_∞ ⟹ d decreases) but NOT necessary (stress can decrease d without herding via leverage/margin effects).

**Cross-asset & practical (Q117-Q120)**: Dimension varies by asset class (Q117 investigate — trivially true, but QCML underperforms AR for measuring it). Real-time dashboard is engineering, not research (Q118 reject). Granger causality of d(t) → vol is unlikely given 0/40 prior null results (Q119 investigate). Adaptive Hilbert dim has a circular dependency: need dim to estimate d, need d to set dim (Q120 investigate — simplest version uses PCA variance ratio, which doesn't need QCML).

---

## Statistical Summary

| Theme | Keep | Reject | Investigate | Keep Rate |
|-------|------|--------|-------------|-----------|
| Novel Observables (Q1-15) | 11 | 3 | 1 | 73% |
| Embedding (Q16-25) | 2 | 5 | 3 | 20% |
| Dead Signal Fix (Q26-35) | 8 | 2 | 0 | 80% |
| Fusion (Q36-45) | 8 | 2 | 0 | 80% |
| Statistical (Q46-55) | 9 | 1 | 0 | 90% |
| Cross-Asset (Q56-65) | 8 | 2 | 0 | 80% |
| Theoretical (Q66-75) | 9 | 1 | 0 | 90% |
| Practical (Q76-85) | 8 | 2 | 0 | 80% |
| Competitive (Q86-95) | 8 | 2 | 0 | 80% |
| Wild Cards (Q96-100) | 4 | 1 | 0 | 80% |
| Intrinsic Dimension (Q101-120) | 7 | 8 | 5 | 35% |
| **TOTAL** | **82** | **29** | **9** | **68%** |

---

## Top 17 Discoveries by Effect Size

| Rank | ID | d | Question | Verdict |
|------|----|---|----------|---------|
| 1 | Q33 | 2.018 | LSTM + attention | Keep (but overfitting concern) |
| 2 | Q45 | 1.775 | HClust fusion | Keep — best fusion method |
| 3 | Q30 | 1.690 | Top-3 curated ensemble | Keep — 130x over naive |
| 4 | Q93 | 1.631 | VIX benchmark | Keep — upper bound where options exist |
| 5 | **Q107** | **1.411** | **Ground state energy E_0(x)** | **Keep — strongest single detector (+69% vs Reduced Purity)** |
| 6 | Q63 | 1.375 | Multi-market embedding | Keep — best cross-asset |
| 7 | Q65 | 1.366 | Market breadth embedding | Keep — novel application |
| 8 | Q92 | 1.329 | Turbulence index | Keep — strong classical competitor |
| 9 | Q16 | 1.327 | Hilbert dim=16 | Keep — highest-impact pipeline change |
| 10 | Q42 | 1.260 | Switching model | Keep — best per-regime selector |
| 11 | Q41 | 1.258 | Stacking/GBM | **Reject** — catastrophic overfitting |
| 12 | **Q102** | **1.175** | **Spectral gap ratio** | **Keep — +48% vs IPR, RMT connection** |
| 13 | Q24 | 1.144 | Soft normalization | Keep — free win |
| 14 | Q02 | 1.121 | OTOC detector | Keep — novel quantum chaos observable |
| 15 | Q44 | 1.009 | MI-weighted fusion | Keep — information-theoretic |
| 16 | Q90 | 0.962 | Absorption ratio | Keep — validates DimCollapse |
| 17 | Q31 | 0.898 | BOCPD short hazard | Keep — resurrected from dead |

---

## Critical Corrections to Paper Claims

| Claim | Paper Value | Corrected Value | Source |
|-------|------------|-----------------|--------|
| Berry Phase Rate lead time | 90 days | 4 days (median walk-forward) | Q77 |
| HPO-optimized d-values | As reported | +0.415d optimism bias | Q54 |
| Reduced Purity stability | d=0.834 | d=0.263 on holdout (-66%) | Paper 2 fusion |
| "Phase transition" language | Implied | Crossover, not transition (gap finite) | Q73 |
| Permutation specialization | p significant | p=0.31 (NOT significant) | Previous review |
| QCML dimension > classical | Implied | AR (d=0.797) >> QCML-IPR (d=0.428). Embedding degrades dimension. | Q101-Q104 |
| Reconstruction-loss training works | Assumed | Optimizer non-functional (loss delta < 1e-8). Ops unchanged. | Q105-Q106 |

---

## Cross-References Between Findings

- **Q16 ↔ Q26/Q27**: Kramers degeneracy at dim=4 explains dead QGT Phase Rigidity and Berry Velocity Coupling. Dim=16 fixes this.
- **Q24 ↔ all observables**: Soft normalization is a free +0.44d win — should be applied before all other improvements.
- **Q45 ↔ Q39 ↔ Q42**: Fusion method hierarchy: HClust (1.775) > Switching (1.26) > Hedge/MW (0.842) > Regime-Adaptive (0.774)
- **Q67 ↔ Q66 ↔ Q73**: Adiabatic regime (r<0.1) + Berry curvature divergence + crossover (not transition) = consistent theoretical picture
- **Q54 ↔ Q77**: Both reveal optimism in current claims — HPO bias and lead time inflation
- **Q90 ↔ dim_collapse**: Absorption ratio (Kritzman 2011) is mathematically related to our DimensionalityCollapse — validates approach
- **Q93 ↔ Q83**: VIX outperforms where available, but geometric fear index works universally
- **Q73 ↔ Q100**: Crossover (not phase transition) explains absence of universal critical exponents
- **Q49 ↔ Q76**: Granger causality confirmed → Sharpe improvement is real but modest (+0.071)
- **Q30 ↔ Q45**: Both show that removing/denoising weak channels is more important than sophisticated combination
- **Q107 ↔ Q105/Q106**: Reconstruction-loss optimizer fails, but its target function E_0(x) is the strongest detector — serendipitous discovery
- **Q114 ↔ Q101/Q104**: IPR, PR, SpectralEntropy are Renyi entropies (∞, 2, 1) of eigenvalue distribution — explains why IPR dominates for sharp spectra
- **Q90 ↔ Q101-Q104**: Absorption Ratio (classical, d=0.797) >> all QCML dimension estimators — embedding is the bottleneck
- **Q108/Q109 ↔ fusion channels**: Feature space (R^8) ≠ channel space (R^16) — dimension-weighted fusion is algebraically undefined
- **Q116 ↔ Q73**: Herding → dimension collapse is sufficient but not necessary, consistent with crises being crossovers not transitions

---

## Actionable Next Steps (Priority Order)

### Immediate (Paper 1 improvements)
1. **Switch to soft normalization** (Q24) — free +0.44d, no downside
2. **Increase Hilbert dim to 16** (Q16) — +131%, fixes degeneracy issues
3. **Correct lead time claim** to 4 days (Q77) — avoid reviewer embarrassment
4. **Add HPO bias caveat** (Q54) — +0.415d optimism, report nested CV estimates
5. **Add VIX + Turbulence + Absorption Ratio benchmarks** (Q93, Q92, Q90)

### Near-term (Paper 2 / Observatory expansion)
6. **Integrate OTOC detector** (Q2) — strongest novel observable
7. **Implement HClust fusion** (Q45) — best fusion method
8. **Implement Hedge/MW fusion** (Q39) — best online/no-training fusion
9. **Run curated ensemble** (Q30) — remove dead channels from pipeline
10. **Multi-market embedding** (Q63) — SPY+TLT+GLD joint geometry

### Near-term (Intrinsic Dimension follow-ups)
11. **Integrate E_0(x) detector** (Q107) — strongest single detector, must check vol proxy correlation
12. **Integrate SpectralGapRatio** (Q102) — verify extreme d=3.97 on 2022 rates
13. **Fix reconstruction-loss optimizer** — gradient-based, then re-test Q105/Q106
14. **Add Renyi entropy discussion** (Q114) — IPR/PR/SpEnt unification for paper theory section

### Research directions (Future papers)
15. **Cross-asset generalization study** (Q56–Q65) — bonds, crypto, commodities
16. **Theoretical paper**: Berry curvature divergence + adiabatic regime + crossover dynamics (Q66–Q73)
17. **Dead signal resurrection suite** (Q28–Q35) — 8 channels resurrected
18. **FOMC periodicity** (Q97) — novel finding worth a short note
19. **Near-miss / severity classification** (Q96, Q98) — graded alert system

---

## Bibliography of Key Papers Discovered

- Saatci, Y., Turner, R., & Rasmussen, C.E. (2010). Gaussian process change point models.
- Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2011). Principal components as a measure of systemic risk.
- Chow, G., Jacquier, E., Kritzman, M., & Lowry, K. (1999). Optimal portfolios in good times and bad.
- Gorin, T., Prosen, T., Seligman, T.H., & Žnidarič, M. (2006). Dynamics of Loschmidt echoes and fidelity decay.
- Jacquod, P. & Petitjean, C. (2009). Decoherence, entanglement and irreversibility in quantum dynamical systems.
- Maldacena, J., Shenker, S.H., & Stanford, D. (2016). A bound on chaos (OTOC).
- de Vos, S. & Freund, Y. (2015). Prediction with expert advice (Hedge algorithm).
- Killick, R., Fearnhead, P., & Eckley, I.A. (2012). Optimal detection of changepoints (PELT).
- Carlsson, G. (2009). Topology and data (persistent homology).
- Zanardi, P., Giorda, P., & Cozzini, M. (2007). Information-theoretic differential geometry of quantum phase transitions.
- Candelori, L., et al. (2025). Intrinsic dimension estimation via quantum metric spectral gap (noise-robust vs MLE/TwoNN).
