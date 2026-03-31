"""
Uhlmann Phase Detector for Mixed-State Regime Detection.

Implements Uhlmann parallel transport phase and fidelity for reduced density
matrices obtained via partial trace of pure QCML states.

Given a pure state |psi> in d=8 (3 qubits), we partition into subsystems
A and B and compute rho_A = Tr_B(|psi><psi|). The Uhlmann phase between
consecutive reduced states measures geometric phase evolution in the
mixed-state setting, potentially capturing noise/decoherence signatures
that pure-state Berry phase misses.

Key formulas:
    Partial trace: rho_A(t) = Tr_B(|psi(t)><psi(t)|)
    Uhlmann fidelity: F(rho_1, rho_2) = (Tr sqrt(sqrt(rho_1) rho_2 sqrt(rho_1)))^2
    Uhlmann phase: phi_U(t) = arg(Tr(sqrt(sqrt(rho_A(t)) rho_A(t+1) sqrt(rho_A(t)))))

References:
    Uhlmann, A. (1976). The "transition probability" in the state space of a *-algebra.
    Jozsa, R. (1994). Fidelity for mixed quantum states.
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple, Optional, List


def partial_trace(psi: np.ndarray, dim_A: int, dim_B: int) -> np.ndarray:
    """Compute reduced density matrix rho_A = Tr_B(|psi><psi|).

    Args:
        psi: State vector of shape (dim_A * dim_B,).
        dim_A: Dimension of subsystem A (kept).
        dim_B: Dimension of subsystem B (traced out).

    Returns:
        rho_A: Reduced density matrix of shape (dim_A, dim_A).
    """
    psi_reshaped = psi.reshape(dim_A, dim_B)
    rho_A = psi_reshaped @ psi_reshaped.conj().T
    return rho_A


def matrix_sqrt_hermitian(rho: np.ndarray) -> np.ndarray:
    """Compute matrix square root of a Hermitian positive-semidefinite matrix.

    Uses eigendecomposition for numerical stability (avoids sqrtm issues
    with near-singular matrices).

    Args:
        rho: Hermitian PSD matrix of shape (d, d).

    Returns:
        sqrt_rho: Matrix square root, shape (d, d).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


def uhlmann_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute Uhlmann fidelity F(rho_1, rho_2).

    F(rho_1, rho_2) = (Tr sqrt(sqrt(rho_1) rho_2 sqrt(rho_1)))^2

    Args:
        rho1: Density matrix of shape (d, d).
        rho2: Density matrix of shape (d, d).

    Returns:
        F: Uhlmann fidelity in [0, 1].
    """
    sqrt_rho1 = matrix_sqrt_hermitian(rho1)
    inner = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_inner = matrix_sqrt_hermitian(inner)
    trace_val = np.real(np.trace(sqrt_inner))
    return float(np.clip(trace_val ** 2, 0.0, 1.0))


def uhlmann_phase(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute Uhlmann parallel transport phase between two density matrices.

    phi_U = arg(Tr(sqrt(sqrt(rho_1) rho_2 sqrt(rho_1))))

    The Uhlmann phase is the phase of the polar decomposition factor in
    the parallel transport condition. It generalizes the Berry phase to
    mixed states.

    Args:
        rho1: Density matrix of shape (d, d).
        rho2: Density matrix of shape (d, d).

    Returns:
        phi: Uhlmann phase in (-pi, pi].
    """
    sqrt_rho1 = matrix_sqrt_hermitian(rho1)
    inner = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_inner = matrix_sqrt_hermitian(inner)
    trace_val = np.trace(sqrt_inner)
    return float(np.angle(trace_val))


def compute_uhlmann_phase_series(
    states: np.ndarray,
    dim_A: int,
    dim_B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Uhlmann phase rate and fidelity time series.

    For each consecutive pair of states, computes:
    1. Reduced density matrices via partial trace
    2. Uhlmann phase between consecutive reduced states
    3. Uhlmann fidelity between consecutive reduced states

    Args:
        states: Array of state vectors, shape (T, hilbert_dim).
        dim_A: Dimension of subsystem A.
        dim_B: Dimension of subsystem B.

    Returns:
        phases: Uhlmann phase rates, shape (T-1,).
        fidelities: Uhlmann fidelities, shape (T-1,).
    """
    T = len(states)
    phases = np.empty(T - 1)
    fidelities = np.empty(T - 1)

    # Precompute all reduced density matrices
    rho_list = [partial_trace(states[t], dim_A, dim_B) for t in range(T)]

    for t in range(T - 1):
        phases[t] = uhlmann_phase(rho_list[t], rho_list[t + 1])
        fidelities[t] = uhlmann_fidelity(rho_list[t], rho_list[t + 1])

    return phases, fidelities


def compute_pure_berry_phase_series(states: np.ndarray) -> np.ndarray:
    """Compute pure-state Berry phase rate for comparison.

    phi_Berry(t) = arg(<psi(t)|psi(t+1)>)

    Args:
        states: Array of state vectors, shape (T, hilbert_dim).

    Returns:
        phases: Berry phase rates, shape (T-1,).
    """
    T = len(states)
    phases = np.empty(T - 1)
    for t in range(T - 1):
        overlap = np.vdot(states[t], states[t + 1])
        phases[t] = np.angle(overlap)
    return phases


def compute_purity_series(
    states: np.ndarray,
    dim_A: int,
    dim_B: int,
) -> np.ndarray:
    """Compute purity Tr(rho_A^2) time series for redundancy check.

    Args:
        states: Array of state vectors, shape (T, hilbert_dim).
        dim_A: Dimension of subsystem A.
        dim_B: Dimension of subsystem B.

    Returns:
        purities: Purity values, shape (T,).
    """
    T = len(states)
    purities = np.empty(T)
    for t in range(T):
        rho_A = partial_trace(states[t], dim_A, dim_B)
        purities[t] = np.real(np.trace(rho_A @ rho_A))
    return purities
