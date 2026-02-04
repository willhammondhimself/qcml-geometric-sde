# Product Requirements Document (PRD)
# Quantum Mechanics × Quantitative Finance: QCML Research Platform

**Version:** 1.0  
**Date:** February 3, 2026  
**Author:** Will (Quanta Ventures Fellow)  
**Status:** Initial Draft  

---

## Executive Summary

Build a production-grade Quantum Cognition Machine Learning (QCML) research platform that demonstrates groundbreaking applications of quantum mechanical principles to quantitative finance. The project showcases three novel research directions suitable for top-tier quant firm interviews (Jane Street, Two Sigma, DE Shaw, Citadel) and potential academic publication.

**Core Thesis:** Financial markets exhibit quantum-like phenomena (superposition of states, noncommutative observables, uncertainty principles) that QCML can capture more efficiently than classical ML.

---

## 1. Project Vision & Goals

### Primary Objective
Create a unified research platform demonstrating that quantum mechanical formalism applied to financial data yields:
- **Superior predictive performance** on volatility, regime detection, and microstructure
- **Sample efficiency** (fewer parameters, less data needed)
- **Interpretability** through quantum geometric structure
- **Novel theoretical insights** connecting physics and finance

### Success Metrics
1. **Volatility Forecasting**: Sharpe ratio > 1.0 on SPY straddle hedging (2020-2025)
2. **Regime Detection**: Predict factor correlation breakdown 1-2 weeks earlier than HMM
3. **Microstructure**: Beat XGBoost/LGBM on order book tick direction (AUC > 0.58)
4. **Efficiency**: QCML parameters scale ~O(D) vs classical O(2^D) for D features
5. **Publication**: Submit to *Quantitative Finance* or *Journal of Financial Data Science*

### Target Audience
- **Primary**: Quant firms (Jane Street, Two Sigma, DE Shaw) for internship/FTE recruiting
- **Secondary**: Academic finance/ML community (conference/journal publication)
- **Tertiary**: Open-source quant research community (GitHub/arXiv visibility)

---

## 2. Core Technical Architecture

### 2.1 QCML Framework Components

#### Mathematical Foundation
- **State representation**: Financial data vectors → quantum states in Hilbert space H^N
- **Observables**: Learned Hermitian matrices A_k encoding features (vol, momentum, spread, etc.)
- **Error Hamiltonian**: 
  ```
  H(x, {A_k}) = Σ_k L(A_k, x_k)
  ```
  where L is a non-negative Hermitian loss function
- **Training**: Minimize ground state energy via iterative eigensolve + gradient descent
- **Inference**: Forecast = ⟨ψ_0 | B | ψ0⟩ where ψ_0 is ground state, B is forecast observable

#### Implementation Stack
```
┌─────────────────────────────────────────┐
│  User Interface (CLI + Notebooks)       │
├─────────────────────────────────────────┤
│  QCML Core Engine                       │
│  - Hamiltonian constructor              │
│  - Ground state solver (torch.linalg)   │
│  - Observable learner (Adam optimizer)  │
│  - Quasi-coherent state generator       │
├─────────────────────────────────────────┤
│  Data Pipelines                         │
│  - Options data (yfinance, CBOE)        │
│  - Factor returns (Fama-French, AQR)    │
│  - Order book data (sample/simulated)   │
├─────────────────────────────────────────┤
│  Backtesting & Evaluation               │
│  - Walk-forward CV                      │
│  - Sharpe/Calmar/AUC metrics            │
│  - Quantum geometry extractors          │
├─────────────────────────────────────────┤
│  Hyperparameter Optimization (Aster)    │
│  - N (Hilbert space dim)                │
│  - w (quantum fluctuation weight)       │
│  - Loss function variants               │
│  - Observable parameterization          │
└─────────────────────────────────────────┘
```

### 2.2 Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Core ML** | PyTorch 2.x | Autodiff, GPU support, eigensolve ops |
| **Data** | Pandas, NumPy | Standard quant pipeline |
| **Optimization** | Adam, L-BFGS | Stable for Hermitian parameter learning |
| **Backtesting** | Vectorbt, custom | Walk-forward validation, Sharpe tracking |
| **Visualization** | Matplotlib, Plotly | Static + interactive charts |
| **Hyperparameter** | Aster AI | Parallel iterative research agent |
| **Version Control** | Git + DVC | Code + data versioning |
| **Deployment** | Docker, Jupyter | Reproducibility, demo notebooks |

---

## 3. Three Research Pillars

### 3.1 Pillar 1: Quantum Volatility Forecasting

**Problem Statement:**  
Implied volatility and realized volatility are context-dependent, nonlinear, and exhibit clustering/jumps. Classical models (GARCH, Heston) assume Markovian dynamics. Can QCML's noncommutative observables capture their joint distribution more efficiently?

**Approach:**
1. **Data**: SPY/QQQ options chain (2018-2025) + realized vol (5-min intraday)
2. **Features (K=8-12)**:
   - Implied vol (ATM, 30-day)
   - IV skew (25-delta put - call)
   - Term structure (slope)
   - Realized vol (1-day, 5-day, 21-day)
   - Bid-ask spread (ATM)
   - Volume/OI ratio
   - VIX level
   - S&P returns (1-day, 5-day)
3. **QCML Model**:
   - N=8,16,32 Hilbert space
   - Learn observables A_1,...,A_K for each feature
   - Learn forecast observable B_vol for 5-day ahead realized vol
   - Loss: MSE + w * quantum fluctuation penalty
4. **Benchmark**: 
   - Linear regression (IV + RV)
   - GARCH(1,1)
   - Simple feedforward NN (3 layers, 32 nodes)
5. **Evaluation**:
   - Walk-forward backtest (12-month train, 1-month test rolling)
   - Sharpe ratio on delta-hedged straddles (implied > forecast → short, else long)
   - Calmar ratio, max drawdown
6. **Theoretical Contribution**:
   - Prove commutator [A_IV, A_RV] ≠ 0 → uncertainty inequality
   - Show bid-ask spread ~ ΔIV · ΔRV (market maker spread as quantum uncertainty)

**Deliverables:**
- `qcml_vol_forecaster.py` (model class)
- `vol_backtest.ipynb` (end-to-end demo)
- `vol_results.csv` (Sharpe ratios, learning curves)
- Section in final paper

---

### 3.2 Pillar 2: Quantum Market Regimes

**Problem Statement:**  
Markets don't switch discretely between bull/bear/crisis—they're in superpositions with smooth transitions. HMMs assume sharp switches. Can QCML's quantum metric spectrum detect regime shifts earlier?

**Approach:**
1. **Data**: Factor returns (2010-2025)
   - Fama-French 5-factor (market, size, value, profitability, investment)
   - Momentum (UMD)
   - Quality (AQR)
   - Low-vol (BAB)
   - Crypto factors (BTC, ETH momentum/carry)
2. **QCML Model**:
   - Rolling 252-day window
   - N=16,32 Hilbert space
   - Learn observables for each factor return
   - Extract quantum metric g_ij = Re⟨∂ψ/∂θ_i | ∂ψ/∂θ_j⟩
   - Compute eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_N
3. **Regime Signal**:
   - Intrinsic dimension = Σ λ_i / max(λ_i)
   - Rapid drop in dimension → regime shift imminent
   - Eigenvector rotation (Berry phase) → directional shift
4. **Benchmark**:
   - HMM (2-3 states)
   - PCA on factor correlations
   - Rolling correlation clustering
5. **Evaluation**:
   - Precision/recall on labeled regime shifts (COVID, 2018 Q4, etc.)
   - Portfolio: rebalance to equal-weight when shift detected, else momentum-weight
   - Calmar ratio vs fixed strategies
6. **Theoretical Contribution**:
   - Gromov-Wasserstein distance between quantum metrics of adjacent windows
   - Prove regime distance correlates with future portfolio variance

**Deliverables:**
- `qcml_regime_detector.py`
- `regime_backtest.ipynb`
- `regime_signals.csv` (dates, dimension estimates, portfolio weights)
- Section in final paper

---

### 3.3 Pillar 3: Quantum Order Book Microstructure

**Problem Statement:**  
Order book shapes encode simultaneous buy/sell intent (superposition). LOB observables (bid, ask, mid, spread, flow) are noncommutative—cannot be simultaneously measured classically. Can QCML outperform local methods on sparse/extreme regions?

**Approach:**
1. **Data**: 
   - Level-2 order book data (AAPL, TSLA, NVDA) or simulated LOB via agent-based model
   - Features per timestamp (K=10-15):
     - Bid price (L1-L5)
     - Ask price (L1-L5)
     - Bid volume (L1-L5)
     - Ask volume (L1-L5)
     - Order imbalance
     - Microprice
     - Trade flow toxicity
2. **QCML Model**:
   - N=16,32 Hilbert space
   - Learn observables for each LOB feature
   - Forecast: next tick direction (up/down/no-change)
   - Loss: Cross-entropy for classification
3. **Benchmark**:
   - XGBoost on same features
   - LSTM (2 layers, 64 hidden)
   - Logistic regression
4. **Evaluation**:
   - AUC-ROC on tick direction
   - PnL simulation (trade with forecast, pay spread)
   - Performance stratified by volatility regime (calm vs excited)
5. **Theoretical Contribution**:
   - Commutator [A_bid, A_ask] ~ spread prediction
   - Berry curvature as LOB toxicity measure

**Deliverables:**
- `qcml_lob_classifier.py`
- `lob_simulation.ipynb` (if no real data)
- `lob_results.csv` (AUC, PnL, by regime)
- Section in final paper

---

## 4. Development Roadmap

### Phase 0: Setup (Week 1, Feb 3-9)
- [ ] Environment setup (Python 3.10+, PyTorch, Jupyter)
- [ ] Git repo initialization with DVC
- [ ] Data pipeline for Pillar 1 (options + realized vol)
- [ ] Baseline QCML solver (Hamiltonian → ground state → forecast)
- [ ] Synthetic data test (Heston simulation)

### Phase 1: Pillar 1 MVP (Weeks 2-3, Feb 10-23)
- [ ] Implement QCML vol forecaster (N=8, MSE loss)
- [ ] Backtest on SPY (2020-2025)
- [ ] Benchmark vs linear + GARCH + NN
- [ ] Compute Sharpe ratios
- [ ] Document uncertainty inequality (commutator analysis)

### Phase 2: Aster Integration (Week 4, Feb 24-Mar 2)
- [ ] Wrap QCML in Aster-compatible format
- [ ] Define hyperparameter grid (N ∈ [4,8,16,32], w ∈ [0, 0.5, 1])
- [ ] Upload data to Aster
- [ ] Launch 50-100 parallel experiments
- [ ] Analyze top configurations

### Phase 3: Pillar 2 Development (Weeks 5-6, Mar 3-16)
- [ ] Factor return data pipeline
- [ ] QCML regime detector implementation
- [ ] Quantum metric extraction
- [ ] HMM baseline
- [ ] Rolling backtest with rebalancing
- [ ] Evaluate precision/recall

### Phase 4: Pillar 3 Development (Weeks 7-8, Mar 17-30)
- [ ] LOB data (real or simulated)
- [ ] QCML LOB classifier
- [ ] XGBoost/LSTM baselines
- [ ] Tick direction prediction
- [ ] PnL simulation
- [ ] Stratified analysis (vol regimes)

### Phase 5: Integration & Polish (Weeks 9-10, Mar 31-Apr 13)
- [ ] Unified API for all three pillars
- [ ] Master notebook demonstrating all models
- [ ] Quantum geometry visualizations (metric eigenvalues, Berry phase)
- [ ] Clean code, docstrings, tests
- [ ] Docker container for reproducibility

### Phase 6: Paper & Pitch (Weeks 11-12, Apr 14-27)
- [ ] Draft paper (15-20 pages)
- [ ] Results tables, figures
- [ ] Theoretical appendix (uncertainty inequality, metric distance)
- [ ] 10-minute interview pitch deck
- [ ] GitHub README with highlights
- [ ] arXiv submission

---

## 5. Success Criteria & KPIs

### Technical Performance
| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Vol Forecasting Sharpe** | > 1.0 | > 1.5 |
| **Regime Detection Lead Time** | 1 week | 2 weeks |
| **LOB AUC** | > 0.56 | > 0.60 |
| **Parameter Efficiency** | 10x fewer than NN | 50x fewer |
| **Training Time (vs NN)** | < 2x slower | < 1.5x slower |

### Research Impact
- [ ] 1 conference submission (NeurIPS, ICML, or ICLR workshop)
- [ ] 1 journal submission (*Quantitative Finance*, *JFDS*)
- [ ] 100+ GitHub stars within 6 months
- [ ] 3+ quant firm interview callbacks

### Learning Outcomes (Personal)
- [ ] Deep understanding of quantum information theory applied to ML
- [ ] Production-grade PyTorch for financial forecasting
- [ ] Mastery of walk-forward validation, Sharpe analysis
- [ ] Iterative research workflow with Aster
- [ ] Clear communication of quantum concepts to finance audience

---

## 6. Risk Assessment & Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| QCML doesn't beat baselines | Medium | High | Start with synthetic data where QCML provably wins; ensure fair comparison (same features, train period) |
| Eigensolve too slow | Low | Medium | Use GPU, JAX for speed; limit N ≤ 32; explore approximate solvers |
| Data quality issues (options) | Medium | Medium | Use multiple sources (yfinance, CBOE, Polygon); validate prices, filter extreme spreads |
| Hyperparameter sensitivity | High | Medium | Aster's parallel search de-risks; document which configs are stable |
| Overfitting on small test set | Medium | High | Walk-forward with 10+ folds; check consistency across assets (SPY, QQQ, IWM) |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Results not novel enough | Low | High | Emphasize uncertainty inequality proof, quantum metric as regime signal (both theoretically novel) |
| Can't finish all 3 pillars | Medium | Medium | Prioritize Pillar 1 (vol) as MVP; Pillars 2-3 are "bonus" for publication |
| Reviewers unfamiliar with QM | Medium | Medium | Write clear intro explaining Hilbert spaces, Hermitian matrices for finance audience; include classical analogy |
| Code not reproducible | Low | High | Docker, requirements.txt, seed fixing, comprehensive README |

### Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Aster integration takes longer | Medium | Low | Can run experiments manually via grid search if needed; Aster is optimization, not blocker |
| Data collection bottleneck | Low | Medium | Pre-download all data in Phase 0; have backup simulated data ready |
| Scope creep (too many experiments) | High | Medium | Strict feature freeze after Phase 3; track experiments in spreadsheet, limit to top 5 configs |

---

## 7. Dependencies & Resources

### Data Sources
- **Options**: yfinance (free), CBOE DataShop (free delayed), Polygon.io ($200/mo)
- **Factors**: Kenneth French website (free), AQR datasets (free)
- **LOB**: Simulated via agent-based model (no cost), or Databento ($500 sample)

### Compute Resources
- **Local**: MacBook Pro M4 Pro (48GB) for dev, small experiments
- **Cloud**: Aster platform (GPU instances), or Google Colab Pro ($10/mo)
- **Storage**: GitHub (<1GB code), DVC + S3 for data (<10GB)

### Human Resources
- **Solo project** (Will)
- **Advisors**: Evan & Bryan (Quanta Ventures), Adam Landsberg (physics mentor)
- **Code review**: Claude Code, GitHub Copilot

### External Tools
- **Aster**: Iterative hyperparameter search ($0-500 depending on usage)
- **Claude Code**: Implementation partner
- **Perplexity**: Research assistant for literature review

---

## 8. Open Questions & Future Work

### Immediate Questions
1. **Observable parameterization**: Low-rank? Pauli basis? Full Hermitian?
2. **Loss function**: Pure MSE, or add quantum fluctuation term? Weight w?
3. **Benchmark fairness**: How to ensure NN has same "capacity" as QCML?
4. **Statistical significance**: What's threshold for claiming QCML "wins"?

### Future Extensions (Post-Phase 6)
- **Multi-asset QCML**: Joint model across 100+ stocks
- **Quantum hardware**: Run on IBM/Rigetti quantum computer
- **Real-time trading**: Deploy QCML model on live paper trading
- **Stochastic QCML**: Extend to continuous-time SDEs
- **Physics collaboration**: Partner with quantum gravity researchers on metric geometry

---

## 9. Stakeholder Communication

### Weekly Updates
- **Format**: Markdown document in `/reports/week_N.md`
- **Contents**: 
  - Progress (completed tasks)
  - Blockers
  - Next week's goals
  - Key metrics (Sharpe, AUC)
  - Visualizations (learning curves, PnL)

### Milestone Demos
- **Phase 1 Demo** (Feb 23): Vol forecaster beating linear baseline
- **Phase 2 Demo** (Mar 2): Aster results with optimal hyperparameters
- **Phase 4 Demo** (Mar 30): All 3 pillars running end-to-end
- **Final Presentation** (Apr 27): Full paper + interview pitch

### Code Reviews
- **Frequency**: Every 2-3 days
- **Reviewer**: Claude Code (line-by-line), self-review against PRD
- **Focus**: Correctness (eigensolve), efficiency (GPU usage), clarity (docstrings)

---

## 10. Acceptance Criteria

### Minimum Viable Product (MVP)
- [ ] Pillar 1 (vol forecasting) fully implemented and documented
- [ ] Sharpe ratio > 0.8 on walk-forward backtest
- [ ] Code runs end-to-end in Docker container
- [ ] 10-page draft paper with intro, methods, results
- [ ] GitHub repo with README, requirements, example notebook

### Full Success
- [ ] All 3 pillars implemented with positive results
- [ ] At least 1 pillar beats baselines by >20% on primary metric
- [ ] Theoretical contribution (uncertainty inequality OR metric distance) proven
- [ ] 15-20 page paper ready for journal submission
- [ ] Interview pitch deck with compelling narrative
- [ ] 3+ quant firm callbacks after sharing project

### Stretch Goals
- [ ] Publish at NeurIPS/ICML workshop
- [ ] 100+ GitHub stars
- [ ] External replication of results (someone else runs your code)
- [ ] Industry partnership (quant firm funds follow-up research)

---

## 11. Appendix: QCML Primer for Stakeholders

### What is QCML?
Quantum Cognition Machine Learning represents data as vectors in Hilbert space (like quantum states) with learned Hermitian matrices (observables). Instead of storing probabilities for every feature combination (exponentially large), QCML stores ~D×N^2 parameters for D features in N-dimensional Hilbert space.

### Why does it work?
1. **Economy of representation**: Financial features (vol, momentum, spread) interact nonlinearly but lie on low-dimensional manifolds. QCML's global manifold structure captures this efficiently.
2. **Noncommutative observables**: In quantum mechanics, measuring position changes momentum (uncertainty principle). In finance, observing implied vol reveals information about realized vol in context-dependent ways. QCML models this naturally.
3. **Robust to noise**: Classical ML in high dimensions suffers from "curse of dimensionality" (sparse data in 2^D bins). QCML operates in N-dimensional space (N << 2^D), avoiding this.

### How is it different from neural networks?
- **NN**: Local approximations, needs lots of data, black box
- **QCML**: Global manifold, sample efficient, interpretable geometry (quantum metric, Berry curvature)

### What's the physics connection?
- **Quantum state** = portfolio of correlated features
- **Hermitian observable** = tradeable quantity (vol, return, spread)
- **Ground state** = optimal information-theoretic encoding
- **Quantum metric** = geometry of regime space (how "far apart" are bull vs bear markets?)

---

## Document Control

**Version History:**
- v1.0 (Feb 3, 2026): Initial draft

**Approvals:**
- Author: Will (Quanta Ventures Fellow)
- Reviewers: TBD (Evan, Bryan)

**Next Review Date:** Feb 10, 2026 (post-Phase 0)

---

*End of PRD*