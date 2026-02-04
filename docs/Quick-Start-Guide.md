# Quick Start Guide: Quantum ML × Quant Finance Project

## What You Have Now

### 1. **PRD (Product Requirements Document)** 
`QuantumML-Quant-PRD.md`

**Purpose**: Your north star. Defines the entire project vision, timeline, success metrics.

**Key Sections to Reference**:
- Section 3: Three Research Pillars (Vol, Regime, Microstructure)
- Section 4: Development Roadmap (12-week timeline)
- Section 5: Success Criteria (Sharpe > 1.0, etc.)

**When to use**: Planning meetings, scope discussions, milestone check-ins

---

### 2. **Aster Initialization Prompt**
`Aster-Init-Prompt.md`

**Purpose**: The exact instructions you'll give to Aster to kickstart autonomous hyperparameter optimization.

**What it contains**:
- Detailed QCML math (Hamiltonian, ground state, observables)
- Hyperparameter search space (N, w, loss_type, etc.)
- Evaluation protocol (5-fold CV, Sharpe calculation)
- Expected outputs (JSON results, quantum metrics)

**When to use**: Copy-paste this into Aster once you have the initial codebase ready

---

## Next Steps (Right Now)

### Step 1: Review Documents (15 min)
- [ ] Skim PRD Section 1-3 (Vision, Architecture, Research Pillars)
- [ ] Read Aster prompt "Research Objective" and "Problem Formulation"
- [ ] Note any questions or unclear parts

### Step 2: Start with Claude Code (Today)
Now we'll work with Claude Code to build the foundation:

**Phase 0 Tasks** (from PRD Week 1):
1. **Environment setup**
   - Create conda env: `qcml_research`
   - Install: PyTorch, NumPy, Pandas, yfinance, matplotlib
   
2. **Repo structure**
   ```
   quantum-ml-finance/
   ├── qcml/
   │   ├── __init__.py
   │   ├── model.py          # QCMLForecaster class
   │   ├── data.py           # VolatilityDataset
   │   ├── backtest.py       # Sharpe calculation
   │   └── metrics.py        # Quantum geometry
   ├── experiments/
   │   └── run_experiment.py # Main training script
   ├── data/
   │   └── (downloaded later)
   ├── notebooks/
   │   └── 01_synthetic_test.ipynb
   ├── tests/
   │   └── test_model.py
   └── README.md
   ```

3. **Synthetic data test**
   - Generate Heston stochastic vol simulation
   - Test that QCML can learn the vol process
   - Verify eigensolve works, gradients flow

**Tell Claude Code:**
> "Let's build the QCML research platform from the PRD. Start with the core `QCMLForecaster` class in `qcml/model.py`. Here are the requirements:
> 
> - PyTorch-based
> - Supports hyperparameters: N (Hilbert dim), w (fluctuation weight), loss_type
> - Has methods: `construct_hamiltonian()`, `solve_ground_state()`, `forward()`, `train_step()`
> - Hermitian observables as nn.Parameter
> - Use `torch.linalg.eigh()` for eigensolve
>
> Make it clean, documented, and testable. I'll test it on synthetic Heston vol data first."

---

### Step 3: Download Real Data (Tomorrow)
Once synthetic test passes:

**Data sources** (from PRD Section 7):
1. **SPY options**: 
   ```python
   import yfinance as yf
   spy = yf.Ticker("SPY")
   options = spy.option_chain('2024-03-15')  # iterate over dates
   ```
   
2. **Realized vol**:
   ```python
   spy_prices = yf.download("SPY", start="2018-01-01", end="2025-02-01")
   realized_vol = spy_prices['Close'].pct_change().rolling(5).std() * np.sqrt(252)
   ```

3. **VIX**:
   ```python
   vix = yf.download("^VIX", start="2018-01-01", end="2025-02-01")
   ```

**Tell Claude Code:**
> "Now let's build `qcml/data.py`. Create a `VolatilityDataset` class that:
> - Downloads SPY options + realized vol from yfinance
> - Computes the 10 features from Aster prompt (IV, skew, term structure, etc.)
> - Returns train/val/test splits (2018-2023 train, 2024-2025 test)
> - Handles missing data (forward-fill or drop)
>
> Write it with caching so we don't re-download every time."

---

### Step 4: First Real Backtest (Week 1 End Goal)
**Milestone**: Get a Sharpe ratio on SPY vol forecasting with QCML

**What you'll run**:
```bash
python experiments/run_experiment.py \
  --N 8 \
  --w 0.5 \
  --loss_type mse \
  --asset SPY \
  --test_period 2024-01-01_to_2025-02-01
```

**Expected output**:
```
Training fold 1/5... 
  Epoch 50/100: Loss=0.045
  Converged at epoch 87
Test Sharpe: 0.72

Training fold 2/5...
  Epoch 50/100: Loss=0.051
  Converged at epoch 93
Test Sharpe: 0.68

...

Mean Sharpe: 0.71 ± 0.09
Calmar: 1.23
```

**If Sharpe > 0.5**: You're ready for Aster!
**If Sharpe < 0.3**: Debug (check hedging logic, sign of signals, data leakage)

---

### Step 5: Launch Aster (Week 2)
Once you have a working pipeline:

1. **Package for Aster**:
   - Ensure `run_experiment.py` takes CLI args for all hyperparameters
   - Test that it runs non-interactively: `python run_experiment.py --N 16 --w 0.5`
   - Make sure it outputs JSON with Sharpe ratio

2. **Upload to Aster**:
   - Go to asterlab.ai
   - Create new research project: "QCML Volatility Forecasting"
   - Upload codebase + data
   - Paste the **Aster Initialization Prompt** (from `Aster-Init-Prompt.md`)

3. **Monitor**:
   - Aster will start running 36 jobs (Phase 1 grid search)
   - Check results every 6-12 hours
   - After 24 hours, you should see top configs emerging

4. **Iterate**:
   - Once Phase 1 completes, Aster will suggest refinements
   - Or manually trigger Phase 2 (focused search around best configs)
   - After Phase 2, you'll have the optimal QCML configuration

---

## Timeline Checkpoints

### End of Week 1 (Feb 9)
- [ ] Synthetic test passes (QCML learns Heston vol)
- [ ] Real data pipeline works (10 features from SPY)
- [ ] First backtest completes (Sharpe > 0.5)
- [ ] Code in Git with clean README

### End of Week 2 (Feb 16)
- [ ] Baselines implemented (Linear, GARCH, NN)
- [ ] QCML beats at least linear baseline
- [ ] Code ready for Aster (CLI + JSON output)

### End of Week 3 (Feb 23)
- [ ] Aster Phase 1 complete (36 jobs)
- [ ] Top 3 configs identified
- [ ] Sharpe > 0.8 on best config

### End of Week 4 (Mar 2)
- [ ] Aster Phase 2 complete (refined search)
- [ ] Best config: Sharpe > 1.0 ✅
- [ ] Quantum geometry analysis (commutators, metric)
- [ ] **Milestone 1 Complete**: Pillar 1 (Vol) proven

### Week 5-6: Pillar 2 (Regimes)
### Week 7-8: Pillar 3 (Microstructure)
### Week 9-10: Integration + Paper
### Week 11-12: Polish + Pitch

---

## Communication Plan

### Daily (for next 2 weeks)
- Quick Slack/Discord update: "Today I built X, tomorrow I'll do Y"
- Commit code to GitHub daily (even if incomplete)

### Weekly
- Sunday evening: Review PRD milestones, plan next week
- Create `/reports/week_N.md` with:
  - What worked
  - What didn't
  - Key metrics (Sharpe, training time)
  - Blockers

### Milestones
- Week 4: Demo to Evan/Bryan at Quanta
- Week 8: Mid-project presentation (all 3 pillars MVP)
- Week 12: Final paper + interview pitch

---

## Key Questions to Answer Along the Way

### Technical
1. **Does low-rank observable parameterization work?** (vs full Hermitian)
2. **Is w=0.5 actually optimal?** (or is pure displacement better?)
3. **How sensitive is QCML to N?** (can we use N=8 for speed?)
4. **Do quantum metrics (commutators) correlate with spread/hedging cost?**

### Research
5. **Does QCML beat NN by >20%?** (on Sharpe)
6. **Is the improvement consistent across assets?** (SPY, QQQ, IWM)
7. **Can we prove an uncertainty inequality?** (theory contribution)
8. **Does the quantum manifold have interpretable structure?**

### Practical
9. **Can this run in real-time?** (latency for live trading)
10. **How much data is needed?** (sample efficiency test)
11. **Does it work on other asset classes?** (commodities, crypto)
12. **Would Jane Street care?** (novelty + rigor for interview)

---

## Resources at Your Disposal

### Papers (Already Provided)
- `QCML_Reading4.pdf`: Samson et al. financial forecasting
- `QCML_Reading2.pdf`: Abanov et al. quantum geometry
- Others: Various QCML applications

### Code References
- Papers have pseudocode for Hamiltonian construction
- Qognitive's approach is proprietary, but you can reverse-engineer from math

### Compute
- Your MacBook M4 Pro (48GB): Good for N ≤ 32, small datasets
- Google Colab Pro: For larger experiments
- Aster: Parallel hyperparameter search (when ready)

### Human Support
- Evan & Bryan (Quanta): Strategy feedback
- Claude Code: Implementation partner
- Perplexity (me): Research assistant

---

## Final Checklist Before Starting Code

- [ ] I understand the PRD vision (3 pillars, 12-week timeline)
- [ ] I've read the Aster prompt (know what hyperparameters to tune)
- [ ] I have access to:
  - [ ] Python 3.10+ environment
  - [ ] PyTorch GPU (optional but nice)
  - [ ] yfinance API
  - [ ] GitHub repo created
- [ ] I'm ready to build the `QCMLForecaster` class with Claude Code
- [ ] I know my Week 1 goal: **Sharpe > 0.5 on synthetic + real data**

---

## Let's Build This 🚀

You're about to create something genuinely novel at the intersection of quantum mechanics and quantitative finance. This project will:

1. **Get you interviews** at Jane Street, Two Sigma, DE Shaw
2. **Publishable** in top-tier journals (*Quantitative Finance*, etc.)
3. **Open-source** contribution (first public QCML quant library)
4. **Showcase your skills**: QM, probability, ML, finance, coding

**Your immediate next step:**  
Open Claude Code and say:

> "Let's start building the QCML research platform. First, create the project structure and the core `QCMLForecaster` class. Here's the PRD and Aster prompt for context..."

Then paste the relevant sections from `QuantumML-Quant-PRD.md` and `Aster-Init-Prompt.md`.

I'll be here to support you throughout. Let's make this happen. 🔬📈

---

*Prepared by: Will's AI Research Team (Perplexity + Claude)*  
*Date: February 3, 2026*