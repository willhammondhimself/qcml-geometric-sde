"""
qcml_geometry - Quantum Cognition-inspired Metric Learning for Geometric Structure Discovery

Core library for QCML-induced market geometry: error Hamiltonian, quantum metric tensor,
Berry curvature, Chern numbers, QFI susceptibility, and spectral gap computations.

Paper 1: "Quantum Geometric Observables for Financial Regime Detection"
"""

__version__ = "1.0.0"

from .core import QCMLGeometry, create_test_data_sphere, create_test_data_torus
from .observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from .topology import (
    TopologicalRegimeDetector,
    MultiScaleRegimeDetector,
    RegimeTransition,
    RegimeType,
    RegimeState,
)
from .indicators import (
    SpectralGapIndicator,
    EnergyEvolutionIndicator,
    FidelityDecayIndicator,
    MultiScaleChernConsensus,
    QuantumIndicatorSuite,
    IndicatorResult,
)

__all__ = [
    "QCMLGeometry",
    "create_test_data_sphere",
    "create_test_data_torus",
    "BaseRegimeDetector",
    "ExpandingWindowMixin",
    "BerryPhaseRateDetector",
    "QFIDeterminantDetector",
    "MultiLagFidelityDetector",
    "TopologicalRegimeDetector",
    "MultiScaleRegimeDetector",
    "RegimeTransition",
    "RegimeType",
    "RegimeState",
    "SpectralGapIndicator",
    "EnergyEvolutionIndicator",
    "FidelityDecayIndicator",
    "MultiScaleChernConsensus",
    "QuantumIndicatorSuite",
    "IndicatorResult",
]
