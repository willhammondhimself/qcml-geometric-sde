"""
QCML Geometry Module - Quantum Cognition-inspired Metric Learning

Implements the core QCML framework for learning geometric structure
from financial data using quantum-inspired methods.

Key Components:
- Error Hamiltonian: H(x) = 1/2 * sum_k (A_k - x_k * I)^2
- Quasi-coherent states: |psi(x)> = ground state of H(x)
- Quantum metric tensor: g_ab = Re<d_a psi|d_b psi> - <d_a psi|psi><psi|d_b psi>
- Berry curvature: F_ab = i(<d_a psi|d_b psi> - <d_b psi|d_a psi>)

Reference: Qognitive papers on Quantum Metric Learning
"""

import numpy as np
from typing import Tuple, Optional, List, Union
import warnings


class QCMLGeometry:
    """
    Quantum Cognition-inspired Metric Learning for geometric structure discovery.

    The QCML framework learns a set of Hermitian operators {A_k} from data,
    constructs an error Hamiltonian H(x), and uses its ground state to define
    a natural quantum geometry (metric tensor and Berry curvature).

    Attributes:
        n_features: Number of input features (data dimension)
        hilbert_dim: Dimension of the Hilbert space (typically 2^k for k qubits)
        operators: List of learned Hermitian operators A_k
        is_fitted: Whether the model has been fitted to data
    """

    def __init__(self, n_features: int, hilbert_dim: int = 4,
                 regularization: float = 1e-6):
        """
        Initialize QCML geometry learner.

        Args:
            n_features: Number of data features
            hilbert_dim: Hilbert space dimension (default 4 for 2-qubit system)
            regularization: Small constant for numerical stability
        """
        self.n_features = n_features
        self.hilbert_dim = hilbert_dim
        self.regularization = regularization
        self.operators: List[np.ndarray] = []
        self.is_fitted = False

        self._identity = np.eye(hilbert_dim, dtype=np.complex128)
        self._ground_state_cache = {}

    def _create_random_hermitian(self, seed: Optional[int] = None) -> np.ndarray:
        """Create a random Hermitian matrix."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((self.hilbert_dim, self.hilbert_dim)) + \
            1j * rng.standard_normal((self.hilbert_dim, self.hilbert_dim))
        return (A + A.conj().T) / 2

    def _create_pauli_basis_operator(self, idx: int) -> np.ndarray:
        """
        Create operator from Pauli basis for 2-qubit system.

        For hilbert_dim=4 (2 qubits), uses tensor products of Pauli matrices.
        """
        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        paulis = [I, X, Y, Z]

        if self.hilbert_dim == 4:
            i1, i2 = idx // 4, idx % 4
            return np.kron(paulis[i1], paulis[i2])
        elif self.hilbert_dim == 2:
            return paulis[idx % 4]
        else:
            return self._create_random_hermitian(seed=idx)

    def fit_operators(self, X: np.ndarray, method: str = 'pca_inspired',
                     n_components: Optional[int] = None) -> 'QCMLGeometry':
        """
        Learn Hermitian operators A_k from data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            method: Learning method - 'pca_inspired', 'random', or 'pauli'
            n_components: Number of operators to learn (default: n_features)

        Returns:
            self
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")

        n_ops = n_components if n_components else n_features

        if method == 'pca_inspired':
            X_centered = X - X.mean(axis=0)
            cov = X_centered.T @ X_centered / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]

            self.operators = []
            for k in range(min(n_ops, n_features)):
                base_op = self._create_pauli_basis_operator(k)
                scale = np.sqrt(max(eigenvalues[k], self.regularization))
                self.operators.append(scale * base_op)

        elif method == 'random':
            self.operators = [
                self._create_random_hermitian(seed=k)
                for k in range(n_ops)
            ]

        elif method == 'pauli':
            max_ops = self.hilbert_dim ** 2
            self.operators = [
                self._create_pauli_basis_operator(k)
                for k in range(min(n_ops, max_ops))
            ]
        else:
            raise ValueError(f"Unknown method: {method}")

        while len(self.operators) < n_ops:
            self.operators.append(self._create_random_hermitian())

        self.is_fitted = True
        self._ground_state_cache.clear()

        return self

    def set_operators(self, operators: list) -> 'QCMLGeometry':
        """Set operators directly (e.g., from learned operators).

        Args:
            operators: List of Hermitian matrices, each (hilbert_dim, hilbert_dim).

        Returns:
            self
        """
        self.operators = [np.asarray(op) for op in operators]
        self.is_fitted = True
        self._ground_state_cache.clear()
        return self

    def error_hamiltonian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute error Hamiltonian H(x) = 1/2 * sum_k (A_k - x_k * I)^2.

        Args:
            x: Data point of shape (n_features,)

        Returns:
            H: Hermitian matrix of shape (hilbert_dim, hilbert_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit_operators first")

        x = np.asarray(x).flatten()
        if len(x) != len(self.operators):
            raise ValueError(f"Expected {len(self.operators)} features, got {len(x)}")

        H = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=np.complex128)

        for k, (A_k, x_k) in enumerate(zip(self.operators, x)):
            diff = A_k - x_k * self._identity
            H += 0.5 * (diff @ diff)

        return H

    def quasi_coherent_state(self, x: np.ndarray,
                            return_energy: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Compute quasi-coherent state |psi(x)> = ground state of H(x).

        Args:
            x: Data point of shape (n_features,)
            return_energy: If True, also return the ground state energy

        Returns:
            psi: Ground state vector of shape (hilbert_dim,)
            energy: Ground state energy (if return_energy=True)
        """
        x = np.asarray(x).flatten()
        x_tuple = tuple(x)

        if x_tuple in self._ground_state_cache:
            psi, energy = self._ground_state_cache[x_tuple]
            return (psi, energy) if return_energy else psi

        H = self.error_hamiltonian(x)

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        idx = np.argmin(eigenvalues)

        psi = eigenvectors[:, idx].astype(np.complex128)
        energy = eigenvalues[idx].real

        psi = psi / np.linalg.norm(psi)

        self._ground_state_cache[x_tuple] = (psi, energy)

        if return_energy:
            return psi, energy
        return psi

    def quantum_metric(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute quantum metric tensor g_ab at point x.

        g_ab = Re<d_a psi|d_b psi> - <d_a psi|psi><psi|d_b psi>

        Args:
            x: Data point of shape (n_features,)
            epsilon: Step size for numerical differentiation

        Returns:
            g: Metric tensor of shape (n_features, n_features)
        """
        x = np.asarray(x).flatten()
        n = len(x)

        psi = self.quasi_coherent_state(x)

        dpsi = np.zeros((n, self.hilbert_dim), dtype=np.complex128)

        for a in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[a] += epsilon
            x_minus[a] -= epsilon

            psi_plus = self.quasi_coherent_state(x_plus)
            psi_minus = self.quasi_coherent_state(x_minus)

            dpsi[a] = (psi_plus - psi_minus) / (2 * epsilon)

        g = np.zeros((n, n), dtype=np.float64)

        for a in range(n):
            for b in range(a, n):
                inner1 = np.vdot(dpsi[a], dpsi[b])
                inner2 = np.vdot(dpsi[a], psi) * np.vdot(psi, dpsi[b])

                g[a, b] = np.real(inner1 - inner2)
                g[b, a] = g[a, b]

        eigenvalues = np.linalg.eigvalsh(g)
        if np.min(eigenvalues) < -self.regularization:
            warnings.warn(f"Metric tensor has negative eigenvalue: {np.min(eigenvalues)}")

        return g

    def berry_curvature(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute Berry curvature tensor F_ab at point x.

        F_ab = -2 Im<d_a psi|d_b psi>

        Args:
            x: Data point of shape (n_features,)
            epsilon: Step size for numerical differentiation

        Returns:
            F: Berry curvature tensor of shape (n_features, n_features)
        """
        x = np.asarray(x).flatten()
        n = len(x)

        dpsi = np.zeros((n, self.hilbert_dim), dtype=np.complex128)

        for a in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[a] += epsilon
            x_minus[a] -= epsilon

            psi_plus = self.quasi_coherent_state(x_plus)
            psi_minus = self.quasi_coherent_state(x_minus)

            dpsi[a] = (psi_plus - psi_minus) / (2 * epsilon)

        F = np.zeros((n, n), dtype=np.float64)

        for a in range(n):
            for b in range(a + 1, n):
                inner = np.vdot(dpsi[a], dpsi[b])
                F[a, b] = -2 * np.imag(inner)
                F[b, a] = -F[a, b]

        return F

    def berry_curvature_2d(self, x: np.ndarray, indices: Tuple[int, int] = (0, 1),
                          epsilon: float = 1e-5) -> float:
        """
        Compute Berry curvature for a 2D subspace using the plaquette method.

        Args:
            x: Data point
            indices: Tuple of indices (a, b) for the 2D plane
            epsilon: Step size

        Returns:
            F_ab: Berry curvature component
        """
        x = np.asarray(x).flatten()
        a, b = indices

        x00 = x.copy()
        x10 = x.copy(); x10[a] += epsilon
        x11 = x.copy(); x11[a] += epsilon; x11[b] += epsilon
        x01 = x.copy(); x01[b] += epsilon

        psi00 = self.quasi_coherent_state(x00)
        psi10 = self.quasi_coherent_state(x10)
        psi11 = self.quasi_coherent_state(x11)
        psi01 = self.quasi_coherent_state(x01)

        U01 = np.vdot(psi00, psi10)
        U12 = np.vdot(psi10, psi11)
        U23 = np.vdot(psi11, psi01)
        U30 = np.vdot(psi01, psi00)

        wilson = U01 * U12 * U23 * U30
        berry_phase = np.imag(np.log(wilson))

        return berry_phase / (epsilon ** 2)

    def chern_number(self, X_grid: np.ndarray, indices: Tuple[int, int] = (0, 1),
                    method: str = 'plaquette') -> float:
        """
        Compute Chern number over a 2D region.

        C = (1/2pi) integral F_ab dx^a ^ dx^b

        Args:
            X_grid: Grid of points, shape (n_x, n_y, n_features)
            indices: Tuple of indices for the 2D plane
            method: 'plaquette' (discrete) or 'integrate' (continuous approx)

        Returns:
            C: Chern number (should be close to an integer for closed surfaces)
        """
        X_grid = np.asarray(X_grid)
        n_x, n_y = X_grid.shape[:2]
        a, b = indices

        if method == 'plaquette':
            total_berry_phase = 0.0

            for i in range(n_x - 1):
                for j in range(n_y - 1):
                    psi00 = self.quasi_coherent_state(X_grid[i, j])
                    psi10 = self.quasi_coherent_state(X_grid[i+1, j])
                    psi11 = self.quasi_coherent_state(X_grid[i+1, j+1])
                    psi01 = self.quasi_coherent_state(X_grid[i, j+1])

                    U01 = np.vdot(psi00, psi10)
                    U12 = np.vdot(psi10, psi11)
                    U23 = np.vdot(psi11, psi01)
                    U30 = np.vdot(psi01, psi00)

                    wilson = U01 * U12 * U23 * U30
                    berry_phase = np.imag(np.log(wilson))
                    total_berry_phase += berry_phase

            return total_berry_phase / (2 * np.pi)

        elif method == 'integrate':
            total = 0.0

            for i in range(n_x):
                for j in range(n_y):
                    x = X_grid[i, j]
                    F = self.berry_curvature(x)

                    if i < n_x - 1 and j < n_y - 1:
                        dx_a = np.linalg.norm(X_grid[i+1, j, a] - X_grid[i, j, a])
                        dx_b = np.linalg.norm(X_grid[i, j+1, b] - X_grid[i, j, b])
                        total += F[a, b] * dx_a * dx_b

            return total / (2 * np.pi)

        else:
            raise ValueError(f"Unknown method: {method}")

    def quantum_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute Fubini-Study distance between two points.

        d(x1, x2) = arccos(|<psi(x1)|psi(x2)>|)

        Args:
            x1, x2: Data points

        Returns:
            d: Quantum distance (in [0, pi/2])
        """
        psi1 = self.quasi_coherent_state(x1)
        psi2 = self.quasi_coherent_state(x2)

        fidelity = np.abs(np.vdot(psi1, psi2))
        fidelity = np.clip(fidelity, 0.0, 1.0)

        return np.arccos(fidelity)

    def quantum_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum fidelity (similarity) between two points.

        F = |<psi(x1)|psi(x2)>|^2

        Args:
            x1, x2: Data points

        Returns:
            F: Fidelity in [0, 1]
        """
        psi1 = self.quasi_coherent_state(x1)
        psi2 = self.quasi_coherent_state(x2)

        return np.abs(np.vdot(psi1, psi2)) ** 2

    def geodesic_distance(self, x1: np.ndarray, x2: np.ndarray,
                         n_steps: int = 100) -> float:
        """
        Compute approximate geodesic distance on the learned manifold.

        Args:
            x1, x2: Data points
            n_steps: Number of integration steps

        Returns:
            d: Geodesic distance
        """
        x1 = np.asarray(x1).flatten()
        x2 = np.asarray(x2).flatten()

        t_vals = np.linspace(0, 1, n_steps)
        dt = 1.0 / (n_steps - 1)

        total_distance = 0.0
        dx = x2 - x1

        for t in t_vals[:-1]:
            x = x1 + t * dx
            g = self.quantum_metric(x)

            ds_squared = dx @ g @ dx
            ds = np.sqrt(max(ds_squared, 0)) * dt
            total_distance += ds

        return total_distance

    def clear_cache(self):
        """Clear the ground state cache."""
        self._ground_state_cache.clear()

    def compute_qfi_susceptibility(self, x: np.ndarray, epsilon: float = 1e-5) -> float:
        """
        Compute quantum Fisher information susceptibility chi(x) = tr(g_ab(x)).

        Args:
            x: Data point of shape (n_features,)
            epsilon: Step size for numerical differentiation

        Returns:
            chi: QFI susceptibility (non-negative scalar)
        """
        g = self.quantum_metric(x, epsilon=epsilon)
        return float(np.trace(g))

    def compute_qfi_determinant(self, x: np.ndarray, epsilon: float = 1e-5) -> float:
        """
        Compute quantum metric determinant det(g_ab(x)).

        Args:
            x: Data point of shape (n_features,)
            epsilon: Step size for numerical differentiation

        Returns:
            det_g: Metric determinant (can be zero or positive)
        """
        g = self.quantum_metric(x, epsilon=epsilon)
        return float(np.linalg.det(g))

    def _christoffel_symbols(self, x: np.ndarray, epsilon_metric: float = 1e-5,
                             epsilon_christoffel: float = 1e-4) -> tuple:
        """Compute Christoffel symbols of the Levi-Civita connection.

        Args:
            x: Data point of shape (n_features,)
            epsilon_metric: Step size for quantum_metric differentiation
            epsilon_christoffel: Step size for metric derivative finite differences

        Returns:
            (christoffel, g, g_inv) where christoffel[sigma, mu, nu] = Gamma^sigma_{mu nu}
        """
        x = np.asarray(x).flatten()
        n = len(x)

        g = self.quantum_metric(x, epsilon=epsilon_metric)

        dg = np.zeros((n, n, n))
        for c in range(n):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[c] += epsilon_christoffel
            x_minus[c] -= epsilon_christoffel
            g_plus = self.quantum_metric(x_plus, epsilon=epsilon_metric)
            g_minus = self.quantum_metric(x_minus, epsilon=epsilon_metric)
            dg[:, :, c] = (g_plus - g_minus) / (2 * epsilon_christoffel)

        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        g_inv = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

        christoffel = np.zeros((n, n, n))
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    gamma_first = 0.5 * (dg[:, nu, mu] + dg[:, mu, nu] - dg[mu, nu, :])
                    christoffel[sigma, mu, nu] = g_inv[sigma] @ gamma_first

        return christoffel, g, g_inv

    def ricci_scalar(self, x: np.ndarray, epsilon_metric: float = 1e-5,
                     epsilon_christoffel: float = 1e-4,
                     epsilon_ricci: float = 1e-3) -> float:
        """Compute Ricci scalar curvature R = g^{mu nu} R_{mu nu}.

        Uses hierarchical finite differences: metric (1e-5) -> Christoffel (1e-4)
        -> Ricci (1e-3).

        Args:
            x: Data point of shape (n_features,)
            epsilon_metric: Step size for quantum_metric
            epsilon_christoffel: Step size for Christoffel symbol computation
            epsilon_ricci: Step size for Christoffel derivative computation

        Returns:
            R: Ricci scalar curvature (can be positive, negative, or zero)
        """
        x = np.asarray(x).flatten()
        n = len(x)

        christoffel, g, g_inv = self._christoffel_symbols(x, epsilon_metric, epsilon_christoffel)

        dGamma = np.zeros((n, n, n, n))
        for rho in range(n):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[rho] += epsilon_ricci
            x_minus[rho] -= epsilon_ricci
            G_plus, _, _ = self._christoffel_symbols(x_plus, epsilon_metric, epsilon_christoffel)
            G_minus, _, _ = self._christoffel_symbols(x_minus, epsilon_metric, epsilon_christoffel)
            dGamma[:, :, :, rho] = (G_plus - G_minus) / (2 * epsilon_ricci)

        ricci = np.zeros((n, n))
        for mu in range(n):
            for nu in range(n):
                for sigma in range(n):
                    val = dGamma[sigma, nu, mu, sigma] - dGamma[sigma, sigma, mu, nu]
                    for lam in range(n):
                        val += christoffel[sigma, sigma, lam] * christoffel[lam, nu, mu]
                        val -= christoffel[sigma, nu, lam] * christoffel[lam, sigma, mu]
                    ricci[mu, nu] += val

        return float(np.einsum('ij,ij->', g_inv, ricci))

    def spectral_gap(self, x: np.ndarray) -> float:
        """
        Compute spectral gap at point x.

        The spectral gap is E_1 - E_0 of the error Hamiltonian.

        Args:
            x: Data point

        Returns:
            gap: Spectral gap (E_1 - E_0)
        """
        H = self.error_hamiltonian(x)
        eigenvalues = np.sort(np.linalg.eigvalsh(H))

        if len(eigenvalues) < 2:
            return 0.0

        return eigenvalues[1] - eigenvalues[0]


# ---------------------------------------------------------------------------
# Test data generators (known topology)
# ---------------------------------------------------------------------------

def create_test_data_sphere(n_samples: int = 500, noise: float = 0.1,
                           seed: int = 42) -> np.ndarray:
    """
    Create test data on a 2D sphere embedded in 3D (known topology).

    The sphere has Chern number +/-1 depending on orientation.

    Args:
        n_samples: Number of samples
        noise: Gaussian noise level
        seed: Random seed

    Returns:
        X: Data matrix of shape (n_samples, 3)
    """
    rng = np.random.default_rng(seed)

    # Uniform points on sphere using Fibonacci lattice
    golden_ratio = (1 + np.sqrt(5)) / 2

    i = np.arange(n_samples)
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2 * (i + 0.5) / n_samples)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    X = np.column_stack([x, y, z])

    # Add noise
    X += rng.normal(0, noise, X.shape)

    return X


def create_test_data_torus(n_samples: int = 500, R: float = 2.0, r: float = 0.5,
                          noise: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Create test data on a torus embedded in 3D (zero Chern number).

    Args:
        n_samples: Number of samples
        R: Major radius
        r: Minor radius
        noise: Gaussian noise level
        seed: Random seed

    Returns:
        X: Data matrix of shape (n_samples, 3)
    """
    rng = np.random.default_rng(seed)

    # Uniform points on torus
    n_u = int(np.sqrt(n_samples * R / r))
    n_v = int(np.ceil(n_samples / n_u))

    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    X = np.column_stack([x, y, z])[:n_samples]

    # Add noise
    X += rng.normal(0, noise, X.shape)

    return X
