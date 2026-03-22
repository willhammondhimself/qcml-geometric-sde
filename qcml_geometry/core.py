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

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np


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
        Create operator from n-qubit Pauli tensor products.

        For hilbert_dim = 2^n, constructs tensor products of n Pauli matrices:
            {I, X, Y, Z}^{⊗n} = 4^n operators (complete basis for d×d Hermitian).

        Skips I⊗I⊗...⊗I (identity, idx=0) by offsetting: actual index = idx + 1.

        Falls back to random Hermitian if hilbert_dim is not a power of 2.

        Args:
            idx: Operator index (0-based, maps to Pauli product idx+1).

        Returns:
            Hermitian matrix of shape (hilbert_dim, hilbert_dim).
        """
        eye2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        paulis = [eye2, X, Y, Z]

        n_qubits = int(round(np.log2(self.hilbert_dim)))
        if 2 ** n_qubits != self.hilbert_dim:
            return self._create_random_hermitian(seed=idx)

        # Skip identity (all-I product at index 0) by using idx+1
        pauli_idx = (idx + 1) % (4 ** n_qubits)

        op = np.array([[1.0]], dtype=np.complex128)
        remainder = pauli_idx
        for _ in range(n_qubits):
            op = np.kron(op, paulis[remainder % 4])
            remainder //= 4

        return op

    def _create_gell_mann_operator(self, idx: int) -> np.ndarray:
        """
        Create generalized Gell-Mann matrix (SU(N) generator).

        For dimension N, there are N²-1 generators:
          - N(N-1)/2 symmetric:   (|i><j| + |j><i|) for i < j
          - N(N-1)/2 antisymmetric: -i(|i><j| - |j><i|) for i < j
          - N-1 diagonal:  constructed from first k+1 basis states

        All are Hermitian, traceless, and orthogonal under Hilbert-Schmidt.

        Args:
            idx: Generator index (0-based). Wraps modulo N²-1.

        Returns:
            Hermitian traceless matrix of shape (hilbert_dim, hilbert_dim).
        """
        N = self.hilbert_dim
        n_generators = N * N - 1
        idx = idx % n_generators

        n_symmetric = N * (N - 1) // 2
        n_antisymmetric = n_symmetric

        if idx < n_symmetric:
            # Symmetric: (|i><j| + |j><i|)
            k = idx
            i, j = 0, 0
            for i in range(N):
                for j in range(i + 1, N):
                    if k == 0:
                        break
                    k -= 1
                if k == 0:
                    break
            op = np.zeros((N, N), dtype=np.complex128)
            op[i, j] = 1.0
            op[j, i] = 1.0
            return op

        elif idx < n_symmetric + n_antisymmetric:
            # Antisymmetric: -i(|i><j| - |j><i|)
            k = idx - n_symmetric
            i, j = 0, 0
            for i in range(N):
                for j in range(i + 1, N):
                    if k == 0:
                        break
                    k -= 1
                if k == 0:
                    break
            op = np.zeros((N, N), dtype=np.complex128)
            op[i, j] = -1j
            op[j, i] = 1j
            return op

        else:
            # Diagonal: sqrt(2/(k(k+1))) * (sum_{m=0}^{k-1} |m><m| - k|k><k|)
            k = idx - n_symmetric - n_antisymmetric + 1  # k in 1..N-1
            op = np.zeros((N, N), dtype=np.complex128)
            norm = np.sqrt(2.0 / (k * (k + 1)))
            for m in range(k):
                op[m, m] = norm
            op[k, k] = -k * norm
            return op

    def fit_operators(self, X: np.ndarray, method: str = 'pca_inspired',
                     n_components: Optional[int] = None,
                     scale_exponent: Optional[float] = None) -> 'QCMLGeometry':
        """
        Learn Hermitian operators A_k from data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            method: Learning method:
                - 'pca_inspired': Pauli/Gell-Mann basis scaled by PCA eigenvalues
                - 'random': Seeded random Hermitian matrices
                - 'pauli': N-qubit Pauli tensor products (unscaled)
                - 'gell_mann': Generalized Gell-Mann SU(N) generators (unscaled)
                - 'pca_pauli': Pauli basis scaled by PCA eigenvalues
                - 'pca_gell_mann': Gell-Mann basis scaled by PCA eigenvalues
            n_components: Number of operators to learn (default: n_features)
            scale_exponent: Exponent for PCA eigenvalue scaling in pca_* methods.
                0.0 = equal weight, 0.5 = sqrt (default), 1.0 = full eigenvalue.
                -0.5 = inverse sqrt (emphasize low-variance directions).
                Only used with pca_inspired, pca_pauli, pca_gell_mann.

        Returns:
            self
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")

        n_ops = n_components if n_components else n_features

        # Methods that use PCA eigenvalue scaling
        pca_methods = {'pca_inspired', 'pca_pauli', 'pca_gell_mann'}

        if method in pca_methods:
            exp = scale_exponent if scale_exponent is not None else 0.5

            X_centered = X - X.mean(axis=0)
            cov = X_centered.T @ X_centered / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            idx_sort = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx_sort]
            eigenvalues = eigenvalues[idx_sort]

            # Select base operator constructor
            if method == 'pca_gell_mann':
                base_fn = self._create_gell_mann_operator
            else:
                base_fn = self._create_pauli_basis_operator

            self.operators = []
            for k in range(min(n_ops, n_features)):
                base_op = base_fn(k)
                scale = max(eigenvalues[k], self.regularization) ** exp
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

        elif method == 'gell_mann':
            max_ops = self.hilbert_dim ** 2 - 1
            self.operators = [
                self._create_gell_mann_operator(k)
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

    def quasi_coherent_state(
        self, x: np.ndarray, return_energy: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
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
        x10 = x.copy()
        x10[a] += epsilon
        x11 = x.copy()
        x11[a] += epsilon
        x11[b] += epsilon
        x01 = x.copy()
        x01[b] += epsilon

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

    def full_spectrum(self, x: np.ndarray) -> np.ndarray:
        """Compute the full sorted eigenvalue spectrum of H(x).

        Args:
            x: Data point of shape (n_features,).

        Returns:
            eigenvalues: Sorted eigenvalues of shape (hilbert_dim,).
        """
        H = self.error_hamiltonian(x)
        return np.sort(np.linalg.eigvalsh(H))

    def sectional_curvature(self, x: np.ndarray, i: int = 0, j: int = 1,
                            epsilon_metric: float = 1e-5,
                            epsilon_christoffel: float = 1e-4,
                            epsilon_riemann: float = 1e-3) -> float:
        """Compute sectional curvature K(e_i, e_j) from the Riemann tensor.

        K(e_i, e_j) = R_{ijij} / (g_{ii} g_{jj} - g_{ij}^2)

        Uses finite differences on Christoffel symbols.

        Args:
            x: Data point of shape (n_features,).
            i: First coordinate index.
            j: Second coordinate index.
            epsilon_metric: Step size for quantum_metric.
            epsilon_christoffel: Step size for Christoffel symbols.
            epsilon_riemann: Step size for Christoffel derivative.

        Returns:
            K: Sectional curvature (can be positive, negative, or zero).
        """
        x = np.asarray(x).flatten()
        n = len(x)
        if i == j or i >= n or j >= n:
            return 0.0

        christoffel, g, _ = self._christoffel_symbols(x, epsilon_metric, epsilon_christoffel)

        # Compute dGamma for directions i and j only (not all n)
        dGamma = {}
        for rho in [i, j]:
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[rho] += epsilon_riemann
            x_minus[rho] -= epsilon_riemann
            G_plus, _, _ = self._christoffel_symbols(x_plus, epsilon_metric, epsilon_christoffel)
            G_minus, _, _ = self._christoffel_symbols(x_minus, epsilon_metric, epsilon_christoffel)
            dGamma[rho] = (G_plus - G_minus) / (2 * epsilon_riemann)

        # R^sigma_{jij} for all sigma, using do Carmo convention:
        # R^sigma_{jij} = d_i Gamma^sigma_{jj} - d_j Gamma^sigma_{ij}
        #               + sum_lam Gamma^sigma_{i,lam} Gamma^lam_{jj}
        #               - sum_lam Gamma^sigma_{j,lam} Gamma^lam_{ij}
        R_up = np.zeros(n)
        for sigma in range(n):
            R_up[sigma] += dGamma[i][sigma, j, j] - dGamma[j][sigma, i, j]
            for lam in range(n):
                R_up[sigma] += christoffel[sigma, i, lam] * christoffel[lam, j, j]
                R_up[sigma] -= christoffel[sigma, j, lam] * christoffel[lam, i, j]

        # Full metric lowering: R_{ijij} = sum_sigma g[i, sigma] * R^sigma_{jij}
        R_lower = g[i, :] @ R_up

        denom = g[i, i] * g[j, j] - g[i, j] ** 2
        if abs(denom) < 1e-15:
            return 0.0

        return float(R_lower / denom)

    def spectral_entropy(self, x: np.ndarray, c: float = 1.0) -> float:
        """Compute Shannon entropy of excitation-energy-weighted spectrum.

        Weights w_n = (E_n - E_0) for n >= 1, normalized to sum to 1.
        High entropy = many modes active (normal); low entropy = spectral
        collapse toward ground state (crisis).

        Args:
            x: Data point of shape (n_features,).
            c: Temperature parameter (unused, kept for API symmetry with spectral_complexity).

        Returns:
            S: Shannon entropy of excitation weights (non-negative).
        """
        eigenvalues = self.full_spectrum(x)
        E0 = eigenvalues[0]
        excitations = eigenvalues[1:] - E0
        excitations = np.maximum(excitations, 1e-15)
        weights = excitations / np.sum(excitations)
        return float(-np.sum(weights * np.log(weights + 1e-15)))

    def geometric_phase_rate(self, x_curr: np.ndarray, x_next: np.ndarray) -> float:
        """Compute geometric phase per step with dynamic phase subtracted.

        gamma_geom = arg(<psi_t|psi_{t+1}>) - E_0(t) * dt

        The dynamic phase E_0*dt is subtracted to isolate the purely
        geometric (Berry-like) contribution. Large geometric phase rates
        indicate rapid change in the state's geometric structure.

        Args:
            x_curr: Current data point of shape (n_features,).
            x_next: Next data point of shape (n_features,).

        Returns:
            gamma: Geometric phase rate (radians per step).
        """
        psi_curr, E0 = self.quasi_coherent_state(x_curr, return_energy=True)
        psi_next = self.quasi_coherent_state(x_next)
        overlap = np.vdot(psi_curr, psi_next)
        total_phase = np.angle(overlap)
        dynamic_phase = -E0  # dt = 1 step
        return float(total_phase - dynamic_phase)

    def hamiltonian_sensitivity(self, x_curr: np.ndarray, x_next: np.ndarray) -> float:
        """Compute variance of Hamiltonian perturbation in ground state.

        Var_psi(DeltaH) = <psi|DH^2|psi> - <psi|DH|psi>^2
        where DH = H(x_next) - H(x_curr).

        Large variance indicates the state is highly sensitive to the
        parameter change — a precursor to regime transitions.

        Args:
            x_curr: Current data point of shape (n_features,).
            x_next: Next data point of shape (n_features,).

        Returns:
            var: Variance of DeltaH in ground state (non-negative).
        """
        psi = self.quasi_coherent_state(x_curr)
        H_curr = self.error_hamiltonian(x_curr)
        H_next = self.error_hamiltonian(x_next)
        DH = H_next - H_curr

        DH_psi = DH @ psi
        mean_DH = np.real(np.vdot(psi, DH_psi))
        mean_DH2 = np.real(np.vdot(DH_psi, DH_psi))
        return float(max(mean_DH2 - mean_DH ** 2, 0.0))

    def geodesic_curvature(self, x_prev: np.ndarray, x_curr: np.ndarray,
                           x_next: np.ndarray,
                           epsilon_metric: float = 1e-5,
                           epsilon_christoffel: float = 1e-4) -> float:
        """Compute geodesic curvature (covariant acceleration norm).

        kappa = ||nabla_{gamma'} gamma'||_g where gamma is the discrete
        path x_prev -> x_curr -> x_next. Non-zero geodesic curvature
        means the path deviates from a geodesic — the manifold is
        forcing the trajectory to curve.

        Args:
            x_prev: Previous data point of shape (n_features,).
            x_curr: Current data point of shape (n_features,).
            x_next: Next data point of shape (n_features,).
            epsilon_metric: Step size for quantum_metric.
            epsilon_christoffel: Step size for Christoffel symbols.

        Returns:
            kappa: Geodesic curvature (non-negative).
        """
        x_prev = np.asarray(x_prev).flatten()
        x_curr = np.asarray(x_curr).flatten()
        x_next = np.asarray(x_next).flatten()
        n = len(x_curr)

        christoffel, g, _ = self._christoffel_symbols(
            x_curr, epsilon_metric, epsilon_christoffel
        )

        # Discrete velocity and acceleration
        v = x_next - x_prev  # central difference (2*dt)
        a = x_next - 2 * x_curr + x_prev  # second difference (dt^2)

        # Covariant acceleration: D^2 gamma / dt^2 = a^sigma + Gamma^sigma_{mu nu} v^mu v^nu
        cov_acc = np.zeros(n)
        for sigma in range(n):
            cov_acc[sigma] = a[sigma]
            for mu in range(n):
                for nu in range(n):
                    cov_acc[sigma] += christoffel[sigma, mu, nu] * v[mu] * v[nu]

        # Norm with metric: ||cov_acc||_g = sqrt(g_{ij} a^i a^j)
        norm_sq = cov_acc @ g @ cov_acc
        return float(np.sqrt(max(norm_sq, 0.0)))

    def effective_state_dimension(self, states: List[np.ndarray]) -> float:
        """Compute effective dimension via IPR of time-averaged density matrix.

        D_eff = W^2 / sum_{s,s'} |<psi_s|psi_{s'}>|^2

        where W is the number of states in the window. Low D_eff means
        states are clustered (normal market); high D_eff means states
        explore many directions (crisis/transition).

        Args:
            states: List of W state vectors, each of shape (hilbert_dim,).

        Returns:
            D_eff: Effective state dimension (1 <= D_eff <= W).
        """
        W = len(states)
        if W < 2:
            return 1.0

        # Build Gram matrix G_{ss'} = |<psi_s|psi_{s'}>|^2
        gram_sum = 0.0
        for s in range(W):
            for sp in range(W):
                overlap = np.abs(np.vdot(states[s], states[sp])) ** 2
                gram_sum += overlap

        if gram_sum < 1e-15:
            return float(W)
        return float(W ** 2 / gram_sum)

    def qgt_phase_rigidity(self, x: np.ndarray, epsilon: float = 1e-5) -> float:
        """Compute Berry-to-metric Frobenius ratio ||F||_F / ||g||_F.

        The quantum geometric tensor Q = g + iF/2 splits into symmetric
        (metric) and antisymmetric (Berry curvature) parts. Their ratio
        measures "phase rigidity": how much of the geometry is topological
        vs. metric. During crises, this ratio changes as the Berry
        curvature structure reorganizes.

        Args:
            x: Data point of shape (n_features,).
            epsilon: Step size for numerical differentiation.

        Returns:
            ratio: ||F||_F / (||g||_F + epsilon) in [0, inf).
        """
        g = self.quantum_metric(x, epsilon=epsilon)
        F = self.berry_curvature(x, epsilon=epsilon)
        norm_F = np.linalg.norm(F, 'fro')
        norm_g = np.linalg.norm(g, 'fro')
        return float(norm_F / (norm_g + 1e-10))

    def reduced_state_purity(self, x: np.ndarray,
                             partition: Tuple[int, int] = (2, 4)) -> float:
        """Compute purity of reduced density matrix under bipartition.

        Tr(rho_A^2) where rho_A = Tr_B(|psi><psi|). Low purity indicates
        entanglement between subsystems. For financial data, subsystem
        entanglement changing during crises captures cross-sector coupling.

        Args:
            x: Data point of shape (n_features,).
            partition: (dim_A, dim_B) such that dim_A * dim_B = hilbert_dim.

        Returns:
            purity: Tr(rho_A^2) in [1/dim_A, 1].
        """
        dim_A, dim_B = partition
        if dim_A * dim_B != self.hilbert_dim:
            raise ValueError(
                f"Partition ({dim_A}, {dim_B}) doesn't match hilbert_dim={self.hilbert_dim}"
            )

        psi = self.quasi_coherent_state(x)
        # Reshape state vector into bipartite form
        psi_matrix = psi.reshape(dim_A, dim_B)
        # Reduced density matrix: rho_A = psi_matrix @ psi_matrix^dagger
        rho_A = psi_matrix @ psi_matrix.conj().T
        # Purity = Tr(rho_A^2)
        return float(np.real(np.trace(rho_A @ rho_A)))

    def spectral_complexity(self, x: np.ndarray, c: float = 1.0) -> float:
        """Compute Gibbs entropy with adaptive temperature.

        p_n = exp(-beta * (E_n - E_0)) / Z where beta = c / Delta.
        Delta is the spectral gap. Adaptive temperature ensures the
        partition function is sensitive to spectral structure regardless
        of overall energy scale.

        Args:
            x: Data point of shape (n_features,).
            c: Temperature scaling constant.

        Returns:
            S: Gibbs entropy (non-negative).
        """
        eigenvalues = self.full_spectrum(x)
        E0 = eigenvalues[0]
        gap = eigenvalues[1] - E0 if len(eigenvalues) > 1 else 1.0
        beta = c / max(gap, 1e-10)

        energies = eigenvalues - E0
        log_weights = -beta * energies
        # Numerically stable softmax
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        Z = np.sum(weights)
        probs = weights / Z
        return float(-np.sum(probs * np.log(probs + 1e-15)))

    def berry_velocity_coupling(self, x_curr: np.ndarray, x_prev: np.ndarray,
                                epsilon: float = 1e-5) -> float:
        """Compute Berry curvature contracted with velocity: ||iota_v F||_g.

        (iota_v F)_a = F_{ab} v^b, then ||iota_v F||_g = sqrt(g^{ac} (iota_v F)_a (iota_v F)_c).

        This measures how strongly the Berry curvature couples to the
        current direction of motion. Large values indicate the trajectory
        is crossing a region of strong topological character.

        Args:
            x_curr: Current data point of shape (n_features,).
            x_prev: Previous data point of shape (n_features,).
            epsilon: Step size for metric/curvature computation.

        Returns:
            coupling: Berry-velocity coupling magnitude (non-negative).
        """
        x_curr = np.asarray(x_curr).flatten()
        x_prev = np.asarray(x_prev).flatten()

        v = x_curr - x_prev  # velocity
        F = self.berry_curvature(x_curr, epsilon=epsilon)
        g = self.quantum_metric(x_curr, epsilon=epsilon)

        # iota_v F: contract F with velocity
        iota = F @ v  # (n,) vector

        # Compute g_inv for norm
        eigvals, eigvecs = np.linalg.eigh(g)
        eigvals = np.maximum(eigvals, 1e-8)
        g_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        norm_sq = iota @ g_inv @ iota
        return float(np.sqrt(max(norm_sq, 0.0)))

    def ricci_scalar_rate(self, x_curr: np.ndarray, x_prev: np.ndarray,
                          epsilon_metric: float = 1e-5,
                          epsilon_christoffel: float = 1e-4,
                          epsilon_ricci: float = 1e-3) -> float:
        """Compute absolute rate of change of Ricci scalar curvature.

        |R(t) - R(t-1)| where R is the Ricci scalar. Captures how fast
        the overall manifold curvature is changing. Rapid curvature
        changes indicate the geometry is reorganizing.

        Args:
            x_curr: Current data point of shape (n_features,).
            x_prev: Previous data point of shape (n_features,).
            epsilon_metric: Step size for quantum_metric.
            epsilon_christoffel: Step size for Christoffel symbols.
            epsilon_ricci: Step size for Christoffel derivative.

        Returns:
            rate: |R(t) - R(t-1)| (non-negative).
        """
        R_curr = self.ricci_scalar(x_curr, epsilon_metric, epsilon_christoffel, epsilon_ricci)
        R_prev = self.ricci_scalar(x_prev, epsilon_metric, epsilon_christoffel, epsilon_ricci)
        return float(abs(R_curr - R_prev))

    def hamiltonian_commutator_norm(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Frobenius norm of the commutator [H(x1), H(x2)].

        Measures geometric incompatibility between two data points.
        Zero when Hamiltonians share eigenstates, large when they don't.

        Args:
            x1: First data point of shape (n_features,).
            x2: Second data point of shape (n_features,).

        Returns:
            norm: ||[H(x1), H(x2)]||_F (non-negative).
        """
        H1 = self.error_hamiltonian(x1)
        H2 = self.error_hamiltonian(x2)
        commutator = H1 @ H2 - H2 @ H1
        return float(np.linalg.norm(commutator, 'fro'))


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
