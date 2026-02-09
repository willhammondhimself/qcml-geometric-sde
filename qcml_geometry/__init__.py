"""
qcml_geometry - Spectral Metric Learning for Geometric Structure Discovery

Core library for learned market geometry: error Hamiltonian, Fubini-Study metric tensor,
Berry curvature, Chern numbers, Fisher information susceptibility, and spectral gap
computations.  Uses mathematical tools from spectral theory (Hilbert spaces, Hermitian
operators) as a structured nonlinear embedding---no quantum physics involved.

Paper 1: "Geometric Observables for Financial Regime Detection"
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
