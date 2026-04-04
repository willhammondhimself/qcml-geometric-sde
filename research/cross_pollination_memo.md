# Cross-pollination memo: QCML, Market Population Dynamics, and PHYS 114

Will Hammond — April 2026

This memo documents structural parallels I've noticed across my three active projects and flags specific places where work in one domain could shortcut work in another.

---

## Overview

All three projects — quantum mechanics coursework, the QCML regime detection paper, and the agent-based market model — share the same mathematical spine: Hamiltonian eigenstructure on a state manifold. The table below lists the concrete overlaps I've confirmed by tracing through the code.

| Concept | PHYS 114 | QCML paper | Market Pop Dynamics |
|---------|----------|------------|---------------------|
| Hamiltonian eigenstates | HW5: diagonalize H, ground state | `core.py:270`: error Hamiltonian H(x) | Fitness matrix; Nash equilibrium as "ground state" |
| Spectral gap | Energy-time uncertainty (HW6) | Theorem 1: gap > 0 implies smoothness | Bifurcation proximity: gap closes at population transition |
| Density matrix / partial trace | HW8: reduced purity of Bell states | Reduced Purity detector (d=0.835) | Population type distribution as mixed state |
| Kronecker products | HW7: multi-particle spin operators | Pauli/Gell-Mann operator basis | Multi-agent interaction matrices |
| Commutators | Ehrenfest theorem (HW6) | Commutator Norm detector | Path-dependence: non-commuting regimes |
| Berry phase | Not covered (beyond PHYS 114) | Berry Phase Rate detector (d=0.608) | Hysteresis in population cycling |
| Adiabatic theorem | Born-Fock (HW5 context) | Theorem 1: smooth geometry when gap open | Two-timescale replicator (fast retail, slow institutional) |

---

## Five things I could actually do with this

### 1. Ground state energy as a standalone detector

The QCML ideation log (Q107) found that E_0(x) — just the smallest eigenvalue of H(x) at each timestep — achieves d = 1.411 on four smoke-test crises. That's the strongest single detector in the entire ideation sweep, stronger than any of the geometric observables in the paper.

It's also the simplest thing to compute: one eigendecomposition, take the minimum. No Berry curvature, no metric tensor, no rolling windows.

I haven't promoted it to the main codebase because it might just be a proxy for realized volatility (the correlation check hasn't been done). But if it holds up, it's embarrassingly effective and should either go in Paper 1 as a baseline or get its own short note.

**What to do:** Run E_0(x) on the full 17-crisis panel. Compute correlation with rolling realized vol. If rho < 0.7, promote it.

### 2. Spectral gap as a bifurcation detector in the agent model

In the QCML paper, the spectral gap Delta(x) = E_1 - E_0 stays positive everywhere (min = 2.34 over 20 years), which is why the smoothness theorem holds. But the gap *opens* during crises (ratio 1.27x), which was a surprise — I expected it to close.

In the agent model, the analog of the spectral gap is the stability margin of the population equilibrium. When all three agent types coexist at roughly equal fractions, the system is near a saddle point — small perturbations can tip the population. The "gap" between the equilibrium eigenvalue and the nearest unstable mode should shrink as you approach the phase boundary on the simplex.

**What to do:** In the endogenous dynamics notebook (02), compute the Jacobian of the replicator at each timestep, extract its eigenvalues, and plot the gap between the leading (stable) eigenvalue and zero. Compare the timing of gap minima with the price crash events in the simulation.

### 3. Prospect theory asymmetry and Berry curvature asymmetry

The Market Pop model's Type II agents have an asymmetric response function (Kahneman-Tversky value function: losses hit 2.25x harder than gains). This creates asymmetric sentiment dynamics — panic sells faster than FOMO buys.

In the QCML data, Berry curvature rate is also asymmetric: the spikes during crisis onset are sharper than during recovery. I don't know if this is the same mechanism or a coincidence. But the math has the same shape: a nonlinear potential with different curvatures on either side of zero.

**What to do:** Compute Berry curvature rate separately for drawdown and recovery periods in the 17-crisis panel. Test whether the onset/recovery asymmetry ratio correlates with the prospect-theory lambda parameter (2.25). If it does, that's a behavioral foundation for why geometric observables work — they're detecting the signature of loss aversion in the manifold curvature.

### 4. Replicator dynamics on the simplex as quantum state evolution

The population fractions (p_1, p_2, p_3) live on a 2-simplex with the constraint that they sum to 1. Quantum states live on the complex projective space CP^{n-1} with the constraint that their norm is 1. Both are constrained manifolds with natural geometric structure.

The replicator equation dp_i/dt = eta * p_i * (pi_i - pi_bar) is a gradient flow on the simplex — agents move toward strategies with above-average fitness. The QCML ground state |psi_0(x)> minimizes the error Hamiltonian. Both are "find the minimum on a manifold" problems.

This is structural, not just analogical. You could literally define a "population metric tensor" g_ab on the simplex by taking finite differences of the population state with respect to model parameters (kappa, alpha, beta, etc.) and computing the Fubini-Study-like object. If that metric's eigenvalues cluster during herding and spread during diversity, you'd have a geometric herding detector that's mathematically identical to the QCML spectral entropy.

**What to do:** Write a 50-line function that computes a 3x3 metric tensor on the population simplex by perturbing the model parameters and measuring population response. Plot its eigenvalue spectrum through a simulated bubble-crash cycle. Compare with QCML spectral entropy on the corresponding price data.

### 5. Level spacing ratio for crisis classification

The Level Spacing Ratio detector (Q123 in ideation) is a specialist: d = 3.27 on GFC, d = 3.44 on COVID, but d = 0.07 on the 2010 Flash Crash. The distinction is between "systemic" crises (many correlated risk factors, GUE-like spectral statistics) and "localized" crises (one or two independent shocks, Poisson-like statistics).

In PHYS 114 terms, this is the difference between a chaotic quantum system (energy levels repel, GUE) and an integrable one (levels independent, Poisson). Random Matrix Theory predicts which statistics you'll see based on the symmetry class of the Hamiltonian.

In the agent model, systemic crises correspond to herding (all agents correlated, one dominant eigenmode) while localized crises correspond to idiosyncratic failures (one agent type blows up, others unaffected). The level spacing ratio might distinguish these directly from market data, without needing the agent model at all.

**What to do:** Compute level spacing ratios on the Market Pop simulation output (the eigenvalues of the rolling covariance of agent demands). Check if the Poisson/GUE classification matches the known crisis type (exogenous shock vs. endogenous cascade).

---

## What I'm not claiming

These parallels are mathematical, not physical. Markets are not quantum systems. Agents are not wavefunctions. The point is narrower: the same eigenvalue/curvature/topology toolkit that describes quantum phase transitions also describes transitions in agent-based population models, because both are driven by changes in the spectral structure of an operator on a manifold. The QCML embedding gives us access to that toolkit for real market data; the agent model lets us test whether the observables have a behavioral interpretation.

---

## References to specific files

- QCML error Hamiltonian: `qcml_geometry/core.py:270-293`
- Ground state energy detector: `qcml_geometry/indicators.py:151-158` (E_0 = min eigenvalue)
- Berry Phase Rate: `qcml_geometry/observables.py:462-620`
- Reduced Purity: `qcml_geometry/observables.py:2968-3034`
- Level Spacing Ratio: `qcml_geometry/observables.py:3253-3434`
- Replicator dynamics: `Market Population Dynamics/src/replicator.py:78-227`
- Prospect theory value function: `Market Population Dynamics/src/agents.py:52-64`
- PHYS 114 density matrix / partial trace: `PHYS114/Homework/HW8_Solutions.tex` (Problem 2)
- PHYS 114 Kronecker product operators: `PHYS114/Homework/HW7_Solutions.nb` (Problem 1)
- Ideation log entry Q107 (E_0): `research/ideation/SYNTHESIS_FULL.md`
- Ideation log entry Q123 (LSR): `research/ideation/SYNTHESIS_FULL.md`
