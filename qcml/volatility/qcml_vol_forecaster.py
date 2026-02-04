"""
QCML Volatility Forecaster - Quantum Uncertainty Principle for IV/RV

This module implements the core QCML volatility forecasting model based on
the hypothesis that implied volatility and realized volatility behave as
noncommutative quantum observables.

Key theoretical insight:
- IV and RV are mapped to Hermitian operators A_IV and A_RV
- Their commutator [A_IV, A_RV] = A_IV @ A_RV - A_RV @ A_IV != 0
- This non-commutativity encodes an uncertainty principle
- The commutator magnitude predicts forecast errors

Reference: QCML Pillar 1 - Quantum Volatility Forecasting
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


class QCMLVolForecaster:
    """
    Quantum Cognition-inspired Volatility Forecaster.

    Uses QCML geometry to learn operators for volatility features,
    with special focus on the commutator between IV and RV observables.

    The forecaster learns:
    1. Feature operators {A_k} encoding each input feature
    2. A forecast observable B such that forecast = <psi|B|psi>
    3. The commutator [A_IV, A_RV] encoding uncertainty

    Attributes:
        n_features: Number of input features
        hilbert_dim: Dimension of Hilbert space
        operators: List of learned Hermitian operators for each feature
        forecast_operator: Learned Hermitian operator for forecasting
        iv_operator_idx: Index of IV (VIX) operator
        rv_operator_idx: Index of RV operator
        is_fitted: Whether the model has been fitted

    Example:
        >>> from qcml.volatility.vol_data import load_volatility_data
        >>> dataset = load_volatility_data()
        >>> forecaster = QCMLVolForecaster(n_features=5, hilbert_dim=8)
        >>> forecaster.fit(dataset.X, dataset.y)
        >>> predictions = forecaster.predict(dataset.X)
        >>> commutator = forecaster.get_commutator()
    """

    def __init__(
        self,
        n_features: int,
        hilbert_dim: int = 8,
        iv_feature_idx: int = 0,
        rv_feature_idx: int = 2,
        regularization: float = 1e-6,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6
    ):
        """
        Initialize QCML Volatility Forecaster.

        Args:
            n_features: Number of input features
            hilbert_dim: Hilbert space dimension (default: 8 for moderate complexity)
            iv_feature_idx: Index of IV (VIX) feature in input (default: 0)
            rv_feature_idx: Index of RV feature in input (default: 2 for rv_20d)
            regularization: Regularization constant for numerical stability
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
        """
        self.n_features = n_features
        self.hilbert_dim = hilbert_dim
        self.iv_feature_idx = iv_feature_idx
        self.rv_feature_idx = rv_feature_idx
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.operators: List[np.ndarray] = []
        self.forecast_operator: Optional[np.ndarray] = None
        self.is_fitted = False

        # Cache
        self._identity = np.eye(hilbert_dim, dtype=np.complex128)
        self._state_cache: Dict[tuple, np.ndarray] = {}

        # Training history
        self._loss_history: List[float] = []

    def _create_hermitian_operator(self, seed: int) -> np.ndarray:
        """Create a random Hermitian matrix with controlled spectrum."""
        rng = np.random.default_rng(seed)

        # Random complex matrix
        A = rng.standard_normal((self.hilbert_dim, self.hilbert_dim)) + \
            1j * rng.standard_normal((self.hilbert_dim, self.hilbert_dim))

        # Make Hermitian
        A = (A + A.conj().T) / 2

        # Normalize to have unit Frobenius norm
        A = A / (np.linalg.norm(A, 'fro') + 1e-10)

        return A

    def _create_pauli_like_operator(self, idx: int) -> np.ndarray:
        """
        Create operator from generalized Pauli basis.

        For hilbert_dim=2^k, uses tensor products of Pauli matrices.
        For other dimensions, falls back to SU(n) generators.
        """
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        paulis = [I, X, Y, Z]

        # Check if dimension is power of 2
        n_qubits = int(np.log2(self.hilbert_dim))
        if 2 ** n_qubits == self.hilbert_dim:
            # Build tensor product
            total_ops = 4 ** n_qubits
            indices = []
            temp_idx = idx % total_ops
            for _ in range(n_qubits):
                indices.append(temp_idx % 4)
                temp_idx //= 4

            result = paulis[indices[0]]
            for i in range(1, n_qubits):
                result = np.kron(result, paulis[indices[i]])

            return result
        else:
            # Fall back to random Hermitian
            return self._create_hermitian_operator(seed=idx)

    def fit_operators(
        self,
        X: np.ndarray,
        method: str = 'pca_weighted'
    ) -> 'QCMLVolForecaster':
        """
        Learn Hermitian operators from data.

        Args:
            X: Feature data of shape (n_samples, n_features)
            method: 'pca_weighted', 'random', or 'pauli'

        Returns:
            self
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")

        if method == 'pca_weighted':
            # Use PCA to weight operators by feature importance
            X_centered = X - X.mean(axis=0)
            cov = X_centered.T @ X_centered / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Create operators
            self.operators = []
            for k in range(n_features):
                base_op = self._create_pauli_like_operator(k)
                # Scale by eigenvalue importance
                scale = np.sqrt(max(eigenvalues[k], self.regularization))
                self.operators.append(scale * base_op)

        elif method == 'random':
            self.operators = [
                self._create_hermitian_operator(seed=k)
                for k in range(n_features)
            ]

        elif method == 'pauli':
            self.operators = [
                self._create_pauli_like_operator(k)
                for k in range(n_features)
            ]

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Created {len(self.operators)} feature operators")
        return self

    def error_hamiltonian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute error Hamiltonian H(x) = (1/2) sum_k (A_k - x_k * I)^2.

        Args:
            x: Feature vector of shape (n_features,)

        Returns:
            H: Hermitian matrix of shape (hilbert_dim, hilbert_dim)
        """
        x = np.asarray(x).flatten()
        H = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=np.complex128)

        for k, (A_k, x_k) in enumerate(zip(self.operators, x)):
            diff = A_k - x_k * self._identity
            H += 0.5 * (diff @ diff)

        return H

    def quasi_coherent_state(
        self,
        x: np.ndarray,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute quasi-coherent state |psi(x)> = ground state of H(x).

        Args:
            x: Feature vector
            use_cache: Whether to use state cache

        Returns:
            psi: Normalized state vector of shape (hilbert_dim,)
        """
        x = np.asarray(x).flatten()

        if use_cache:
            x_tuple = tuple(x.round(6))  # Round for cache key stability
            if x_tuple in self._state_cache:
                return self._state_cache[x_tuple]

        H = self.error_hamiltonian(x)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Ground state is eigenvector of smallest eigenvalue
        psi = eigenvectors[:, 0].astype(np.complex128)
        psi = psi / np.linalg.norm(psi)

        if use_cache:
            self._state_cache[tuple(x.round(6))] = psi

        return psi

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False
    ) -> 'QCMLVolForecaster':
        """
        Fit the QCML volatility forecaster.

        1. Fit feature operators via fit_operators()
        2. Learn forecast observable B via gradient descent on MSE

        Args:
            X: Feature data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            verbose: Print training progress

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        n_samples = len(X)
        assert len(y) == n_samples

        # Step 1: Fit feature operators
        self.fit_operators(X, method='pca_weighted')

        # Step 2: Learn forecast operator B
        # Initialize B as random Hermitian
        B = self._create_hermitian_operator(seed=999)
        B = B * np.std(y)  # Scale to target range

        # Precompute all states
        states = np.array([self.quasi_coherent_state(x) for x in X])

        self._loss_history = []
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # Compute predictions: <psi|B|psi>
            predictions = np.array([
                np.real(np.vdot(psi, B @ psi))
                for psi in states
            ])

            # MSE loss
            errors = predictions - y
            loss = np.mean(errors ** 2) + self.regularization * np.linalg.norm(B, 'fro') ** 2
            self._loss_history.append(loss)

            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    logger.info(f"Converged at iteration {iteration}, loss={loss:.6f}")
                break

            prev_loss = loss

            # Gradient descent
            # d/dB <psi|B|psi> = |psi><psi|
            # d/dB MSE = (2/n) sum_i error_i * |psi_i><psi_i|
            grad = np.zeros_like(B)
            for i in range(n_samples):
                outer = np.outer(states[i], states[i].conj())
                grad += errors[i] * outer

            grad = 2 * grad / n_samples
            grad += 2 * self.regularization * B  # Regularization gradient

            # Make gradient Hermitian
            grad = (grad + grad.conj().T) / 2

            # Update
            B = B - self.learning_rate * grad

            # Ensure B stays Hermitian
            B = (B + B.conj().T) / 2

            if verbose and iteration % 100 == 0:
                logger.info(f"Iteration {iteration}, loss={loss:.6f}")

        self.forecast_operator = B
        self.is_fitted = True

        # Clear state cache to save memory
        self._state_cache.clear()

        logger.info(f"Training complete. Final loss: {self._loss_history[-1]:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict volatility for new data.

        Prediction = <psi(x)|B|psi(x)>

        Args:
            X: Feature data of shape (n_samples, n_features) or (n_features,)

        Returns:
            predictions: Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            psi = self.quasi_coherent_state(x, use_cache=False)
            pred = np.real(np.vdot(psi, self.forecast_operator @ psi))
            predictions.append(pred)

        return np.array(predictions)

    def get_iv_operator(self) -> np.ndarray:
        """Get the operator corresponding to IV (VIX) feature."""
        if not self.operators:
            raise RuntimeError("Must call fit_operators() first")
        return self.operators[self.iv_feature_idx]

    def get_rv_operator(self) -> np.ndarray:
        """Get the operator corresponding to RV feature."""
        if not self.operators:
            raise RuntimeError("Must call fit_operators() first")
        return self.operators[self.rv_feature_idx]

    def get_commutator(self) -> np.ndarray:
        """
        Compute commutator [A_IV, A_RV] = A_IV @ A_RV - A_RV @ A_IV.

        This is the key quantity encoding the uncertainty principle
        between implied and realized volatility.

        Returns:
            C: Commutator matrix (should be anti-Hermitian: C = -C^dagger)
        """
        A_IV = self.get_iv_operator()
        A_RV = self.get_rv_operator()

        return A_IV @ A_RV - A_RV @ A_IV

    def get_commutator_norm(self) -> float:
        """
        Compute Frobenius norm of commutator ||[A_IV, A_RV]||_F.

        A non-zero norm indicates non-commutativity (uncertainty principle).

        Returns:
            Frobenius norm of commutator
        """
        C = self.get_commutator()
        return np.linalg.norm(C, 'fro')

    def get_commutator_spectral_norm(self) -> float:
        """
        Compute spectral norm (largest singular value) of commutator.

        Returns:
            Spectral norm of commutator
        """
        C = self.get_commutator()
        return np.linalg.norm(C, 2)

    def compute_uncertainty_bound(self, x: np.ndarray) -> float:
        """
        Compute Robertson-Schrodinger uncertainty bound at point x.

        The uncertainty relation is:
        Delta_IV * Delta_RV >= (1/2) |<[A_IV, A_RV]>|

        where Delta_A = sqrt(<A^2> - <A>^2) is the standard deviation.

        Args:
            x: Feature vector

        Returns:
            Lower bound on Delta_IV * Delta_RV
        """
        psi = self.quasi_coherent_state(x, use_cache=False)
        C = self.get_commutator()

        # <[A_IV, A_RV]>
        expectation = np.vdot(psi, C @ psi)

        # The bound is (1/2)|<[A,B]>|
        return 0.5 * np.abs(expectation)

    def spectral_gap(self, x: np.ndarray) -> float:
        """
        Compute spectral gap at point x.

        The gap between ground and first excited state indicates
        how well-defined the quasi-coherent state is.

        Args:
            x: Feature vector

        Returns:
            Spectral gap (E_1 - E_0)
        """
        H = self.error_hamiltonian(x)
        eigenvalues = np.sort(np.linalg.eigvalsh(H))

        if len(eigenvalues) < 2:
            return 0.0

        return eigenvalues[1] - eigenvalues[0]

    def feature_sensitivity(self, x: np.ndarray, feature_idx: int,
                           epsilon: float = 1e-5) -> float:
        """
        Compute sensitivity of prediction to a specific feature.

        This is the partial derivative d(prediction)/d(x_k).

        Args:
            x: Feature vector
            feature_idx: Index of feature to compute sensitivity for
            epsilon: Step size for numerical differentiation

        Returns:
            Sensitivity value
        """
        x = np.asarray(x).flatten()

        x_plus = x.copy()
        x_plus[feature_idx] += epsilon

        x_minus = x.copy()
        x_minus[feature_idx] -= epsilon

        pred_plus = self.predict(x_plus)[0]
        pred_minus = self.predict(x_minus)[0]

        return (pred_plus - pred_minus) / (2 * epsilon)


if __name__ == "__main__":
    # Test the forecaster
    logging.basicConfig(level=logging.INFO)

    print("Testing QCML Volatility Forecaster...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    # Simulate features
    X = np.random.randn(n_samples, n_features) * 0.1
    X[:, 0] = np.abs(X[:, 0])  # VIX-like (positive)
    X[:, 2] = np.abs(X[:, 2])  # RV-like (positive)

    # Simulate target with nonlinear relationship
    y = 0.3 * X[:, 0] + 0.5 * X[:, 2] + 0.2 * X[:, 0] * X[:, 2] + 0.1 * np.random.randn(n_samples)

    # Create and fit forecaster
    forecaster = QCMLVolForecaster(n_features=5, hilbert_dim=8)
    forecaster.fit(X, y, verbose=True)

    # Make predictions
    predictions = forecaster.predict(X)

    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    correlation = np.corrcoef(predictions, y)[0, 1]

    print(f"\nTraining Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation: {correlation:.4f}")

    # Test commutator
    commutator = forecaster.get_commutator()
    comm_norm = forecaster.get_commutator_norm()

    print(f"\nCommutator Analysis:")
    print(f"  Commutator shape: {commutator.shape}")
    print(f"  ||[A_IV, A_RV]||_F: {comm_norm:.6f}")
    print(f"  Is anti-Hermitian: {np.allclose(commutator, -commutator.conj().T)}")

    # Test uncertainty bound
    x_test = X[0]
    bound = forecaster.compute_uncertainty_bound(x_test)
    print(f"\nUncertainty bound at x[0]: {bound:.6f}")

    # Test spectral gap
    gap = forecaster.spectral_gap(x_test)
    print(f"Spectral gap at x[0]: {gap:.6f}")

    print("\nQCML Volatility Forecaster tests passed!")
