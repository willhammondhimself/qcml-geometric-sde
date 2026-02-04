# QCML-Geometric SDEs: Topological Market Regime Detection

A very new and experimental framework combining Quantum Cognative Metric Learning (QCML) with stochastic differential equations on learned manifolds and topological invariants for quantitative finance.

## Project Structure

```
Personal-QCML/
├── qcml/                           # Core QCML package
│   ├── __init__.py
│   ├── qcml_geometry.py           # QCML operators, quantum metric, Berry curvature
│   ├── geometric_sde.py           # SDEs on learned manifolds
│   ├── topological_regime.py      # Regime detection via Chern numbers
│   └── trading_signals.py         # Trading signal generation
│
├── notebooks/                      # Research notebooks
│   ├── __init__.py
│   └── Geometric_SDE_QCML.ipynb   # Full pipeline demonstration
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_geometry.py
│   ├── test_sde.py
│   ├── test_regime.py
│   └── test_signals.py
│
├── experiments/                    # Experiment scripts
│   └── __init__.py
│
├── docs/                           # Documentation
│   └── ...
│
├── internal-docs                       # Code quality guidelines
├── PRD.md                          # Product requirements
├── Quick-Start-Guide.md            # Getting started
└── README.md                       # This file
```

## Core Modules

### 1. QCML Geometry (`qcml_geometry.py`)

Learns geometric structure from data using quantum-inspired methods:
- Error Hamiltonian: $H(x) = \frac{1}{2}\sum_k (A_k - x_k \cdot I)^2$
- Quasi-coherent states: ground states of H(x)
- Quantum metric tensor: $g_{ab}$ encoding manifold geometry
- Berry curvature: $F_{ab}$ for topological analysis

```python
from qcml import QCMLGeometry
import numpy as np

# Create and fit geometry
geometry = QCMLGeometry(n_features=3, hilbert_dim=4)
geometry.fit_operators(X, method='pca_inspired')

# Compute quantum metric and Berry curvature
g = geometry.quantum_metric(x)      # Metric tensor
F = geometry.berry_curvature(x)     # Topological structure
d = geometry.quantum_distance(x1, x2)  # Quantum metric distance
```

### 2. Geometric SDEs (`geometric_sde.py`)

Stochastic differential equations that respect the learned geometry:

**Traditional SDE**: $dX = \mu(X)dt + \sigma(X)dW$

**Geometric SDE**: $dX^a = \mu^a(X)dt + \sigma^a_b(X)dW^b$ where $\Sigma^{ab} = \sigma^a_c \sigma^{bc} \propto g^{-1}$

```python
from qcml import GeometricSDE

# Create SDE respecting QCML geometry
sde = GeometricSDE(geometry=geometry)

# Simulate paths
paths, times = sde.simulate_euler_maruyama(
    x0=x0, T=2.0, dt=0.01, n_paths=100,
    use_metric_diffusion=True
)

# Train neural SDE model
from qcml import NeuralGeometricSDE, train_neural_sde
model = NeuralGeometricSDE(n_features=3)
train_neural_sde(model, dataset, n_epochs=100)
```

### 3. Topological Regime Detection (`topological_regime.py`)

Detects market regime changes using topological invariants:
- **Chern number**: Integer topological invariant robust to noise
- **Key insight**: $\Delta C = 0$ → same regime (extreme event); $\Delta C \neq 0$ → topological transition (regime change)

```python
from qcml import TopologicalRegimeDetector

detector = TopologicalRegimeDetector(
    geometry=geometry,
    window_size=50,
    chern_threshold=0.5
)

# Detect transitions
transitions = detector.detect_transitions(X_timeseries)

# Compute regime signatures
signature = detector.compute_regime_signature(X_window)
print(f"Chern number: {signature['chern_number']:.2f}")
```

### 4. Trading Signals (`trading_signals.py`)

Generates trading signals from topological and geometric features:
- Curvature spikes → market stress
- Chern transitions → regime changes
- Metric expansion → volatility shifts
- Spectral gap compression → instability

```python
from qcml import TopologicalTradingStrategy, backtest_topological_strategy

strategy = TopologicalTradingStrategy(
    geometry=geometry,
    lookback=30,
    position_limit=1.0
)

results = backtest_topological_strategy(
    strategy, X, prices,
    transaction_cost=0.001
)

print(f"Sharpe: {results['metrics']['sharpe']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

## Quick Start

### Installation

```bash
cd Personal-QCML
# No external dependencies beyond numpy, torch, scipy
```

### Basic Usage

```python
from qcml import (
    QCMLGeometry,
    create_test_data_sphere,
    GeometricSDE,
    TopologicalRegimeDetector
)

# Create test data
X = create_test_data_sphere(n_samples=500, noise=0.05)

# Fit QCML geometry
qcml = QCMLGeometry(n_features=3, hilbert_dim=4)
qcml.fit_operators(X)

# Create and simulate geometric SDE
sde = GeometricSDE(geometry=qcml)
paths, times = sde.simulate_euler_maruyama(x0=X[0], T=1.0, dt=0.01)

# Detect regime changes
detector = TopologicalRegimeDetector(qcml, window_size=30)
transitions = detector.detect_transitions(X)
```

## Key Papers & References

The framework builds on:
- QCML from Qognitive research (Papers 1-7)
- Your SDE learning notebook (`SDE/Lesson01_Learning_SDE.ipynb`)
- Differential geometry and topological data analysis

## Testing Hypotheses

On real financial data:

1. **2008 Financial Crisis** → Expect Chern number discontinuity (topological)
2. **Flash Crash 2010** → NO Chern change (same topology, extreme point)
3. **COVID March 2020** → Test for topological transition
4. **2022 Rate Hikes** → Gradual Chern shift vs discontinuity

## Next Steps

1. **Apply to real data**: Equities, options, fixed income
2. **Validate on historical crises**: 2008, 2020, etc.
3. **Optimize hyperparameters**: Hilbert dimension, window sizes
4. **Use Astra for numerical optimization**: Matrix exponential, eigenvalue solvers
5. **Write academic paper**: Novel framework + empirical validation

## Quality Standards

- Readable variable names and clear structure
- Physics/math correctness (Hermitian operators, normalized states)
- Reproducibility (fixed random seeds)
- Testing on synthetic data before real applications
- Comprehensive docstrings

## Author

Will Hammond
