# Geometric Observables for Financial Regime Detection

Academic research code for the paper:

> **Geometric Observables for Financial Regime Detection**
> Will Hammond, Pitzer College

## Overview

This project introduces three unsupervised regime detection observables derived from the metric tensor of a QCML (Quantum Cognition Machine Learning---a spectral metric learning framework) induced geometry on financial data manifolds:

1. **Berry Phase Rate** -- rate of change of Berry curvature signals topological transitions
2. **QFI Pseudo-Determinant** -- metric volume element detects phase boundary crossings
3. **Multi-Lag Fidelity** -- multi-scale state overlap measures regime instability

These geometric observables are competitive with supervised Random Forest baselines (median Cohen's d = 1.67 vs 1.13 via simple weighted combination) and generalize across 5 ETFs (SPY, QQQ, IWM, EFA, DIA).

## Repository Structure

```
qcml_geometry/       Core math library (pure functions, no I/O)
  core.py            QCMLGeometry: error Hamiltonian, quantum metric, Berry curvature
  observables.py     BerryPhaseRateDetector, QFIDeterminantDetector, MultiLagFidelityDetector
  topology.py        TopologicalRegimeDetector, Chern number computation
  indicators.py      Spectral gap, energy evolution, fidelity decay, multi-scale Chern

experiments/         Paper 1 experiment scripts
  data/              PolygonDataSource, MinimalFeatureEngine
  baselines.py       Classical baselines (Vol-Z, CUSUM, HMM, Random Forest)
  additional_detectors.py  Additional QCML detectors (Chern, QFI susceptibility, etc.)
  crisis_config.py   Crisis period definitions
  config.yaml        Default hyperparameters

notebooks/           3 polished notebooks for Paper 1
paper/               LaTeX source
archive/             Non-Paper-1 code (SDE, trading, deep learning, TDA, etc.)
tests/               Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Polygon API key
echo "POLYGON_API_KEY=your_key_here" > .env

# Run example regime detection
python experiments/example_regime_detection.py

# Run tests
pytest tests/ -v
```

## Core Modules

### QCML Geometry (`qcml_geometry/core.py`)

Learns geometric structure from data using spectral metric learning methods:
- Error Hamiltonian: H(x) = 1/2 sum_k (A_k - x_k I)^2
- Quasi-coherent states: ground states of H(x)
- Metric tensor: g_ab encoding manifold geometry
- Berry curvature: F_ab for topological analysis

```python
from qcml_geometry import QCMLGeometry

geometry = QCMLGeometry(n_features=5, hilbert_dim=8)
geometry.fit_operators(X, method='pca_inspired')

g = geometry.quantum_metric(x)       # Metric tensor
F = geometry.berry_curvature(x)      # Topological structure
d = geometry.quantum_distance(x1, x2)  # Fubini-Study distance
```

### Geometric Observables (`qcml_geometry/observables.py`)

Three novel unsupervised regime detectors:

```python
from qcml_geometry import BerryPhaseRateDetector, QFIDeterminantDetector, MultiLagFidelityDetector

detector = BerryPhaseRateDetector(hilbert_dim=8, n_pca_components=15)
detector.fit(X_features)
scores = detector.compute_regime_scores(X_features)
```

## Key Results

| Method | Median Cohen's d | Type |
|--------|-----------------|------|
| Fused QCML (Phase B) | 1.79 | Unsupervised geometric |
| Fused QCML (Phase A) | 1.67 | Unsupervised geometric |
| QFI Determinant | 1.42 | Unsupervised geometric |
| Berry Phase Rate | 1.31 | Unsupervised geometric |
| Multi-Lag Fidelity | 1.26 | Unsupervised geometric |
| Random Forest | 1.13 | Supervised statistical |

## References

This work builds upon the QCML framework developed by Qognitive, Inc. and academic collaborators. See `paper/qcml_geometric_sde.tex` for complete references.

## Citation

```bibtex
@article{hammond2026geometric,
  title={Geometric Observables for Financial Regime Detection},
  author={Hammond, Will},
  year={2026},
  institution={Pitzer College}
}
```

## Author

Will Hammond (whammond@pitzer.edu), Pitzer College
