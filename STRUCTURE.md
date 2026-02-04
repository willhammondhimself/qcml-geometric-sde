# Project Structure Overview

## Directory Layout

```
Personal-QCML/
├── qcml/                                    # Core QCML package
│   ├── __init__.py                         # Package exports
│   ├── qcml_geometry.py                    # QCML geometry learning
│   ├── geometric_sde.py                    # SDEs on learned manifolds
│   ├── topological_regime.py               # Regime detection via topology
│   └── trading_signals.py                  # Trading signal generation
│
├── notebooks/                               # Research and exploration
│   ├── __init__.py
│   └── Geometric_SDE_QCML.ipynb            # Full pipeline demonstration
│
├── tests/                                   # Unit tests
│   ├── __init__.py
│   ├── test_geometry.py                    # QCML geometry tests
│   ├── test_sde.py                         # SDE module tests
│   ├── test_regime.py                      # Regime detection tests
│   └── test_signals.py                     # Trading signals tests
│
├── experiments/                             # Experiment scripts
│   ├── __init__.py
│   └── example_regime_detection.py         # Example: regime detection
│
├── docs/                                    # Documentation
│   └── Quick-Start-Guide.md                # Getting started guide
│
├── README.md                                # Project overview
├── STRUCTURE.md                             # This file
├── internal-docs                                # Code quality guidelines
└── PRD.md                                   # Product requirements
```

## Module Organization

### `qcml/` - Core Package

#### `qcml_geometry.py` (22 KB)
Implements quantum cognition-inspired metric learning.

**Key Classes:**
- `QCMLGeometry`: Main class for learning and using quantum geometry
  - `fit_operators()`: Learn Hermitian operators from data
  - `quasi_coherent_state()`: Compute ground state
  - `quantum_metric()`: Compute metric tensor
  - `berry_curvature()`: Compute Berry curvature tensor
  - `chern_number()`: Compute topological invariant
  - `quantum_distance()`: Quantum fidelity-based distance
  - `geodesic_distance()`: Geodesic distance on manifold

**Helper Functions:**
- `create_test_data_sphere()`: Generate synthetic sphere data
- `create_test_data_torus()`: Generate synthetic torus data

**Key Concepts:**
- Error Hamiltonian: $H(x) = \frac{1}{2}\sum_k (A_k - x_k \cdot I)^2$
- Quantum metric: $g_{ab} = \text{Re}\langle\partial_a\psi|_b\psi\rangle - ...$
- Berry curvature: $F_{ab} = i(\langle\partial_a\psi|\partial_b\psi\rangle - \text{h.c.})$

#### `geometric_sde.py` (23 KB)
Implements stochastic differential equations on learned manifolds.

**Key Classes:**
- `GeometricSDE`: SDE simulator respecting QCML geometry
  - `simulate_euler_maruyama()`: Euler-Maruyama scheme
  - `simulate_milstein()`: Milstein scheme (higher order)
  - `geodesic_brownian_motion()`: Brownian motion with geodesic drift

- `NeuralGeometricSDE`: Neural network for learning SDE coefficients
- `SDETrajectoryDataset`: PyTorch dataset for SDE data

**Training Functions:**
- `train_neural_sde()`: Train neural SDE on trajectory data
- `simulate_from_neural_sde()`: Simulate using learned model
- `gaussian_nll_loss()`: NLL loss for SDE learning

**Key Concepts:**
- Metric-induced diffusion: $\Sigma^{ab} \propto g^{-1}$
- Geodesic drift correction using Christoffel symbols
- Geometry-aware neural network training

#### `topological_regime.py` (24 KB)
Detects market regime changes using topological invariants.

**Key Classes:**
- `TopologicalRegimeDetector`: Main regime detection class
  - `rolling_chern_number()`: Compute Chern over rolling windows
  - `detect_transitions()`: Find regime changes
  - `compute_regime_signature()`: Characterize regime geometry
  - `classify_event()`: Distinguish regime change from extreme event
  - `online_update()`: Online (streaming) detection

- `MultiScaleRegimeDetector`: Multi-window analysis
- `RegimeTransition`: Data class for detected transitions
- `RegimeState`: Current detector state

**Key Concepts:**
- Chern number: Integer topological invariant
- $\Delta C = 0$ → same regime (extreme event)
- $\Delta C \neq 0$ → topological transition (regime change)
- Multi-scale detection for different time horizons

#### `trading_signals.py` (22 KB)
Generates trading signals from topological/geometric features.

**Key Classes:**
- `TopologicalTradingStrategy`: Signal generation and position management
  - `compute_signal()`: Generate signal from data point
  - `update_position()`: Execute trades based on signals
  - `close_position()`: Exit current position

- `EnsembleTopologicalStrategy`: Combine multiple strategies
- `TradingSignal`: Signal data class
- `SignalType`: Enum of signal types

**Functions:**
- `backtest_topological_strategy()`: Full backtest with metrics
- `generate_synthetic_market_data()`: Synthetic market data with regimes

**Signal Types:**
1. Curvature spike → market stress
2. Topology transition → regime change
3. Metric expansion → volatility shift
4. Spectral gap warning → instability

### `notebooks/` - Research

#### `Geometric_SDE_QCML.ipynb`
Complete interactive demonstration of the framework.

**Sections:**
1. QCML Geometry Learning
2. Geometric SDEs on Learned Manifolds
3. Topological Regime Detection
4. Trading Strategy and Backtesting
5. Neural SDE Learning

**Visualizations:**
- 3D data on spheres/tori
- Quantum metric and Berry curvature
- SDE path evolution
- Chern number over time
- Backtest equity curves

### `tests/` - Unit Tests

#### `test_geometry.py`
Tests for QCML geometry module.

**Test Classes:**
- `TestQCMLGeometry`: Core functionality tests
- `TestTestData`: Synthetic data generation

**Coverage:**
- Operator fitting (PCA, Pauli, random)
- Error Hamiltonian properties
- Ground state computation
- Metric tensor (shape, symmetry, positivity)
- Berry curvature (antisymmetry)
- Distance/similarity metrics
- Sphere vs torus detection

#### `test_sde.py` (placeholder)
Tests for geometric SDE module.

#### `test_regime.py` (placeholder)
Tests for regime detection.

#### `test_signals.py` (placeholder)
Tests for trading signals.

### `experiments/` - Runnable Examples

#### `example_regime_detection.py`
Complete example of regime detection pipeline.

**Steps:**
1. Generate synthetic market data with regime changes
2. Learn QCML geometry
3. Detect regime transitions using Chern numbers
4. Generate trading signals
5. Backtest strategy
6. Visualize results

**Run:**
```bash
python experiments/example_regime_detection.py
```

## Key Files

### Configuration & Documentation
- `internal-docs`: Code quality guidelines and standards
- `PRD.md`: Product requirements and roadmap
- `Quick-Start-Guide.md`: Getting started
- `README.md`: Project overview

### Generated
- `regime_detection_example.png`: Visualization from example script

## Dependencies

**Core:**
- numpy: Numerical computing
- scipy: Scientific functions (eigensolvers, linalg)

**Deep Learning (optional):**
- torch: Neural networks for SDE learning
- torch.nn: Network layers

**Visualization (optional):**
- matplotlib: Plotting and visualization

## Running the Framework

### Quick Test
```bash
cd Personal-QCML
python -c "from qcml import QCMLGeometry; print('QCML loaded!')"
```

### Run Example
```bash
python experiments/example_regime_detection.py
```

### Run Tests (requires pytest)
```bash
pytest tests/ -v
```

### Use in Code
```python
from qcml import (
    QCMLGeometry,
    GeometricSDE,
    TopologicalRegimeDetector,
    TopologicalTradingStrategy
)

# Create and fit geometry
geometry = QCMLGeometry(n_features=5, hilbert_dim=4)
geometry.fit_operators(X)

# Detect regime changes
detector = TopologicalRegimeDetector(geometry)
transitions = detector.detect_transitions(X)

# Generate trading signals
strategy = TopologicalTradingStrategy(geometry)
for t, x in enumerate(X):
    signal = strategy.compute_signal(x, t)
```

## Development Workflow

### Adding a Feature
1. Update relevant module in `qcml/`
2. Add/update tests in `tests/`
3. Document in notebooks
4. Update README/STRUCTURE
5. Create example in `experiments/`

### Code Quality
- Follow internal-docs guidelines
- All classes/functions documented
- Type hints in docstrings
- Physics/math correctness verified
- Reproducibility via seeds

### Testing Strategy
1. Unit tests: Core functionality
2. Integration tests: Full pipeline
3. Synthetic data: Validation
4. Real data: Performance metrics

## Future Expansion

### Phase 2: Real Financial Data
- Load equity/option price data
- Backtest on 2008, 2020, etc.
- Optimize hyperparameters with Astra
- Publish results

### Phase 3: Advanced Features
- Multivariate option pricing
- Cross-asset regime detection
- Risk attribution
- Portfolio optimization

### Phase 4: Production
- Streaming/online signals
- Real-time monitoring
- Risk management integration
- Compliance reporting
