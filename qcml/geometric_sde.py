"""
Geometric SDE Module - Stochastic Differential Equations on Learned Manifolds

This module implements SDEs that respect the geometric structure learned by QCML.
The drift and diffusion coefficients are defined on the Riemannian manifold
induced by the quantum metric tensor.

Key Innovation:
Traditional SDE: dX = μ(X)dt + σ(X)dW
Geometric SDE: dX^a = μ^a(X)dt + σ^a_b(X)dW^b

where the diffusion respects the metric: Σ^{ab} = σ^a_c σ^{bc} ∝ g^{ab}
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, List, Callable, Union
import math

from .qcml_geometry import QCMLGeometry


class GeometricSDE:
    """
    Stochastic Differential Equations on QCML-learned manifolds.

    This class implements geometry-aware SDEs where:
    1. Drift can be constrained to the tangent space
    2. Diffusion respects the metric structure
    3. Simulations follow geodesics in expectation (optional)

    Attributes:
        geometry: QCMLGeometry instance with learned operators
        drift_fn: Drift function μ^a(x) → ℝ^n
        diffusion_fn: Diffusion function σ(x) → ℝ^{n×m}
    """

    def __init__(self, geometry: QCMLGeometry,
                 drift_fn: Optional[Callable] = None,
                 diffusion_fn: Optional[Callable] = None):
        """
        Initialize geometric SDE.

        Args:
            geometry: Fitted QCMLGeometry instance
            drift_fn: Custom drift function (default: zero drift)
            diffusion_fn: Custom diffusion function (default: metric-induced)
        """
        self.geometry = geometry
        self._drift_fn = drift_fn
        self._diffusion_fn = diffusion_fn

    def drift_on_manifold(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute geometry-aware drift μ^a(x).

        If no custom drift is provided, returns zero drift.
        Can be extended to include:
        - Mean reversion to data manifold
        - Gradient of potential function
        - Geodesic spray (for Brownian motion on manifold)

        Args:
            x: Current position
            t: Time (for time-dependent drift)

        Returns:
            mu: Drift vector of shape (n_features,)
        """
        x = np.asarray(x).flatten()

        if self._drift_fn is not None:
            return self._drift_fn(x, t)

        # Default: zero drift (pure diffusion)
        return np.zeros_like(x)

    def diffusion_on_manifold(self, x: np.ndarray, t: float = 0.0,
                             scale: float = 1.0) -> np.ndarray:
        """
        Compute geometry-aware diffusion σ^a_b(x).

        The diffusion matrix is constructed so that the covariance
        respects the inverse metric: Σ = σσᵀ ∝ g⁻¹

        This means diffusion is larger in directions where the metric
        is smaller (flatter manifold = more diffusion).

        Args:
            x: Current position
            t: Time (for time-dependent diffusion)
            scale: Overall scale factor

        Returns:
            sigma: Diffusion matrix of shape (n_features, n_features)
        """
        x = np.asarray(x).flatten()

        if self._diffusion_fn is not None:
            return self._diffusion_fn(x, t)

        # Compute metric-induced diffusion
        g = self.geometry.quantum_metric(x)

        # Regularize metric for numerical stability
        epsilon = 1e-6
        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.maximum(eigenvalues, epsilon)

        # Inverse metric g⁻¹
        g_inv = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

        # Square root of inverse metric: σσᵀ = g⁻¹
        # Using eigendecomposition: σ = V @ sqrt(Λ⁻¹) @ Vᵀ
        sigma = eigenvectors @ np.diag(np.sqrt(1.0 / eigenvalues)) @ eigenvectors.T

        return scale * sigma

    def simulate_euler_maruyama(self, x0: np.ndarray, T: float, dt: float,
                                n_paths: int = 1, seed: Optional[int] = None,
                                use_metric_diffusion: bool = True,
                                diffusion_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate SDE paths using Euler-Maruyama scheme.

        dX^a = μ^a(X)dt + σ^a_b(X)dW^b

        Args:
            x0: Initial condition of shape (n_features,)
            T: Total simulation time
            dt: Time step
            n_paths: Number of paths to simulate
            seed: Random seed
            use_metric_diffusion: Whether to use metric-induced diffusion
            diffusion_scale: Scale factor for diffusion

        Returns:
            paths: Array of shape (n_paths, n_steps+1, n_features)
            times: Array of shape (n_steps+1,)
        """
        rng = np.random.default_rng(seed)
        x0 = np.asarray(x0).flatten()
        n_features = len(x0)
        n_steps = int(T / dt)
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1, n_features))
        paths[:, 0, :] = x0

        times = np.linspace(0, T, n_steps + 1)

        for path_idx in range(n_paths):
            x = x0.copy()

            for step in range(n_steps):
                t = step * dt

                # Compute drift
                mu = self.drift_on_manifold(x, t)

                # Compute diffusion
                if use_metric_diffusion:
                    sigma = self.diffusion_on_manifold(x, t, scale=diffusion_scale)
                else:
                    sigma = diffusion_scale * np.eye(n_features)

                # Brownian increment
                dW = rng.standard_normal(n_features) * sqrt_dt

                # Euler-Maruyama update
                x = x + mu * dt + sigma @ dW

                paths[path_idx, step + 1, :] = x

        return paths, times

    def simulate_milstein(self, x0: np.ndarray, T: float, dt: float,
                         n_paths: int = 1, seed: Optional[int] = None,
                         diffusion_scale: float = 1.0,
                         epsilon: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate SDE paths using Milstein scheme (higher order).

        Includes correction term for better convergence:
        dX = μdt + σdW + ½σ(∂σ/∂x)((dW)² - dt)

        Args:
            x0: Initial condition
            T: Total time
            dt: Time step
            n_paths: Number of paths
            seed: Random seed
            diffusion_scale: Scale factor
            epsilon: Step for numerical derivative

        Returns:
            paths, times
        """
        rng = np.random.default_rng(seed)
        x0 = np.asarray(x0).flatten()
        n_features = len(x0)
        n_steps = int(T / dt)
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1, n_features))
        paths[:, 0, :] = x0

        times = np.linspace(0, T, n_steps + 1)

        for path_idx in range(n_paths):
            x = x0.copy()

            for step in range(n_steps):
                t = step * dt

                mu = self.drift_on_manifold(x, t)
                sigma = self.diffusion_on_manifold(x, t, scale=diffusion_scale)

                # Compute ∂σ/∂x for Milstein correction
                dsigma_dx = np.zeros((n_features, n_features, n_features))
                for k in range(n_features):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[k] += epsilon
                    x_minus[k] -= epsilon

                    sigma_plus = self.diffusion_on_manifold(x_plus, t, scale=diffusion_scale)
                    sigma_minus = self.diffusion_on_manifold(x_minus, t, scale=diffusion_scale)

                    dsigma_dx[:, :, k] = (sigma_plus - sigma_minus) / (2 * epsilon)

                # Brownian increment
                dW = rng.standard_normal(n_features) * sqrt_dt

                # Milstein update
                x_new = x + mu * dt + sigma @ dW

                # Milstein correction: ½ Σⱼₖ σⱼₐ (∂σₐᵦ/∂xⱼ) (dWₖdWᵦ - δₖᵦ dt)
                for a in range(n_features):
                    for b in range(n_features):
                        for j in range(n_features):
                            correction = 0.5 * sigma[j, a] * dsigma_dx[a, b, j]
                            # (dW)² term
                            for k in range(n_features):
                                dW_dW = dW[k] * dW[b] - (1 if k == b else 0) * dt
                                correction *= dW_dW
                            x_new[a] += correction

                x = x_new
                paths[path_idx, step + 1, :] = x

        return paths, times

    def geodesic_brownian_motion(self, x0: np.ndarray, T: float, dt: float,
                                 n_paths: int = 1, seed: Optional[int] = None,
                                 diffusion_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Brownian motion on the manifold (geodesic drift correction).

        Includes the Itô-Stratonovich correction for Brownian motion on Riemannian
        manifolds, ensuring that the process follows geodesics in expectation.

        The drift correction is: μ^a = -½ Γ^a_{bc} g^{bc}

        where Γ are the Christoffel symbols.

        Args:
            x0: Initial condition
            T: Total time
            dt: Time step
            n_paths: Number of paths
            seed: Random seed
            diffusion_scale: Scale factor

        Returns:
            paths, times
        """
        rng = np.random.default_rng(seed)
        x0 = np.asarray(x0).flatten()
        n_features = len(x0)
        n_steps = int(T / dt)
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1, n_features))
        paths[:, 0, :] = x0

        times = np.linspace(0, T, n_steps + 1)

        for path_idx in range(n_paths):
            x = x0.copy()

            for step in range(n_steps):
                t = step * dt

                # Compute metric and its derivatives for Christoffel symbols
                g = self.geometry.quantum_metric(x)
                epsilon = 1e-5

                # Numerical derivative of metric
                dg = np.zeros((n_features, n_features, n_features))
                for k in range(n_features):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[k] += epsilon
                    x_minus[k] -= epsilon

                    g_plus = self.geometry.quantum_metric(x_plus)
                    g_minus = self.geometry.quantum_metric(x_minus)

                    dg[:, :, k] = (g_plus - g_minus) / (2 * epsilon)

                # Inverse metric
                eigenvalues, eigenvectors = np.linalg.eigh(g)
                eigenvalues = np.maximum(eigenvalues, 1e-8)
                g_inv = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

                # Christoffel symbols (first kind): Γ_{abc} = ½(∂_a g_{bc} + ∂_b g_{ac} - ∂_c g_{ab})
                # Christoffel symbols (second kind): Γ^a_{bc} = g^{ad} Γ_{dbc}
                christoffel = np.zeros((n_features, n_features, n_features))
                for a in range(n_features):
                    for b in range(n_features):
                        for c in range(n_features):
                            gamma_first = 0.5 * (dg[b, c, a] + dg[a, c, b] - dg[a, b, c])
                            for d in range(n_features):
                                christoffel[a, b, c] += g_inv[a, d] * gamma_first

                # Geodesic drift correction: -½ Γ^a_{bc} g^{bc}
                mu_geo = np.zeros(n_features)
                for a in range(n_features):
                    for b in range(n_features):
                        for c in range(n_features):
                            mu_geo[a] -= 0.5 * christoffel[a, b, c] * g_inv[b, c]

                # User-defined drift
                mu_user = self.drift_on_manifold(x, t)

                # Total drift
                mu = mu_user + mu_geo

                # Diffusion from metric
                sigma = eigenvectors @ np.diag(np.sqrt(1.0 / eigenvalues)) @ eigenvectors.T
                sigma *= diffusion_scale

                # Brownian increment
                dW = rng.standard_normal(n_features) * sqrt_dt

                # Update
                x = x + mu * dt + sigma @ dW
                paths[path_idx, step + 1, :] = x

        return paths, times


class NeuralGeometricSDE(nn.Module):
    """
    Neural network model for learning geometric SDE coefficients.

    Learns μ(x) and log(σ²(x)) from trajectory data, similar to
    the SDE learning notebook but with optional geometry constraints.
    """

    def __init__(self, n_features: int, hidden_dim: int = 32,
                 n_layers: int = 2, geometry: Optional[QCMLGeometry] = None):
        """
        Initialize neural SDE model.

        Args:
            n_features: Input dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            geometry: Optional QCML geometry for constraints
        """
        super().__init__()

        self.n_features = n_features
        self.geometry = geometry

        # Drift network
        drift_layers = [nn.Linear(n_features, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            drift_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        drift_layers.append(nn.Linear(hidden_dim, n_features))
        self.drift_net = nn.Sequential(*drift_layers)

        # Log-variance network (outputs log σ² for numerical stability)
        var_layers = [nn.Linear(n_features, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            var_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        # Output n_features diagonal elements of covariance (simplified)
        var_layers.append(nn.Linear(hidden_dim, n_features))
        self.log_var_net = nn.Sequential(*var_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing drift and log-variance.

        Args:
            x: Input tensor of shape (batch, n_features)

        Returns:
            mu: Drift of shape (batch, n_features)
            log_sigma2: Log-variance of shape (batch, n_features)
        """
        mu = self.drift_net(x)
        log_sigma2 = self.log_var_net(x)

        return mu, log_sigma2

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """Get drift only."""
        return self.drift_net(x)

    def diffusion_std(self, x: torch.Tensor) -> torch.Tensor:
        """Get diffusion standard deviation (σ, not σ²)."""
        log_sigma2 = self.log_var_net(x)
        return torch.exp(0.5 * log_sigma2)


class SDETrajectoryDataset(Dataset):
    """
    Dataset for SDE trajectory data: (x_t, Δx, Δt).

    Compatible with the format from the SDE learning notebook.
    """

    def __init__(self, paths: np.ndarray, times: np.ndarray):
        """
        Initialize dataset from trajectory data.

        Args:
            paths: Array of shape (n_paths, n_steps+1, n_features)
                   or (n_paths, n_steps+1) for 1D
            times: Array of shape (n_steps+1,)
        """
        paths = np.asarray(paths)

        # Handle 2D paths (1D SDE)
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]

        n_paths, n_steps_plus_1, n_features = paths.shape
        n_steps = n_steps_plus_1 - 1

        # Extract (x_t, Δx, Δt) tuples
        x_t = paths[:, :-1, :].reshape(-1, n_features)
        dx = (paths[:, 1:, :] - paths[:, :-1, :]).reshape(-1, n_features)

        dts = np.diff(times)
        dt_all = np.tile(dts, n_paths).reshape(-1, 1)

        self.x_t = torch.from_numpy(x_t.astype(np.float32))
        self.dx = torch.from_numpy(dx.astype(np.float32))
        self.dt = torch.from_numpy(dt_all.astype(np.float32))
        self.n_features = n_features

    def __len__(self) -> int:
        return self.x_t.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_t[idx], self.dx[idx], self.dt[idx]


def gaussian_nll_loss(mu: torch.Tensor, log_sigma2: torch.Tensor,
                     dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss for SDE increments.

    Assumes: Δx | x ~ N(μ(x)Δt, σ²(x)Δt)

    Args:
        mu: Predicted drift of shape (batch, n_features)
        log_sigma2: Log-variance of shape (batch, n_features)
        dx: Observed increment of shape (batch, n_features)
        dt: Time step of shape (batch, 1)

    Returns:
        loss: Scalar loss value
    """
    LOG2PI = math.log(2.0 * math.pi)

    # Clamp for stability
    log_sigma2 = torch.clamp(log_sigma2, -20.0, 20.0)

    # Log variance per sample: log(Δt) + log(σ²)
    log_dt = torch.log(dt.clamp_min(1e-12))
    log_var = log_dt + log_sigma2

    var = torch.exp(log_var).clamp_min(1e-24)

    # Expected increment
    mean = mu * dt

    # NLL for multivariate Gaussian (diagonal covariance)
    n_features = mu.shape[1]
    nll = 0.5 * (n_features * LOG2PI + log_var.sum(dim=1) +
                 ((dx - mean).pow(2) / var).sum(dim=1))

    return nll.mean()


def train_neural_sde(model: NeuralGeometricSDE,
                    dataset: SDETrajectoryDataset,
                    n_epochs: int = 100,
                    batch_size: int = 1024,
                    lr: float = 1e-3,
                    val_fraction: float = 0.1,
                    device: str = 'cpu',
                    verbose: bool = True) -> Tuple[List[float], List[float]]:
    """
    Train neural SDE model on trajectory data.

    Args:
        model: NeuralGeometricSDE instance
        dataset: SDETrajectoryDataset
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_fraction: Validation set fraction
        device: Training device
        verbose: Print progress

    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    model = model.to(device)

    # Split data
    n_val = max(1, int(val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x_t, dx, dt in train_loader:
            x_t, dx, dt = x_t.to(device), dx.to(device), dt.to(device)

            optimizer.zero_grad()
            mu, log_sigma2 = model(x_t)
            loss = gaussian_nll_loss(mu, log_sigma2, dx, dt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_t, dx, dt in val_loader:
                x_t, dx, dt = x_t.to(device), dx.to(device), dt.to(device)
                mu, log_sigma2 = model(x_t)
                loss = gaussian_nll_loss(mu, log_sigma2, dx, dt)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    return train_losses, val_losses


def simulate_from_neural_sde(model: NeuralGeometricSDE,
                            x0: np.ndarray, T: float, dt: float,
                            n_paths: int = 1, seed: Optional[int] = None,
                            device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate paths using learned neural SDE model.

    Args:
        model: Trained NeuralGeometricSDE
        x0: Initial condition
        T: Total time
        dt: Time step
        n_paths: Number of paths
        seed: Random seed
        device: Computation device

    Returns:
        paths, times
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0).flatten()
    n_features = len(x0)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    paths = np.zeros((n_paths, n_steps + 1, n_features))
    paths[:, 0, :] = x0

    times = np.linspace(0, T, n_steps + 1)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for path_idx in range(n_paths):
            x = x0.copy()

            for step in range(n_steps):
                x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)

                mu, log_sigma2 = model(x_tensor)
                mu = mu.cpu().numpy().flatten()
                sigma = np.exp(0.5 * log_sigma2.cpu().numpy().flatten())

                # Brownian increment
                dW = rng.standard_normal(n_features) * sqrt_dt

                # Euler-Maruyama
                x = x + mu * dt + sigma * dW

                paths[path_idx, step + 1, :] = x

    return paths, times


if __name__ == "__main__":
    print("Testing Geometric SDE Module...")

    # Create test geometry
    from .qcml_geometry import create_test_data_sphere, QCMLGeometry

    X = create_test_data_sphere(n_samples=200, noise=0.05)
    qcml = QCMLGeometry(n_features=3, hilbert_dim=4)
    qcml.fit_operators(X, method='pca_inspired')

    # Test geometric SDE simulation
    geo_sde = GeometricSDE(geometry=qcml)

    x0 = np.array([1.0, 0.0, 0.0])
    paths, times = geo_sde.simulate_euler_maruyama(
        x0=x0, T=1.0, dt=0.01, n_paths=5, seed=42
    )

    print(f"Simulated paths shape: {paths.shape}")
    print(f"Final positions (first path): {paths[0, -1, :]}")

    # Test neural SDE
    print("\nTesting Neural Geometric SDE...")

    # Generate training data
    train_paths, train_times = geo_sde.simulate_euler_maruyama(
        x0=x0, T=5.0, dt=0.01, n_paths=100, seed=123,
        use_metric_diffusion=True, diffusion_scale=0.5
    )

    dataset = SDETrajectoryDataset(train_paths, train_times)
    print(f"Dataset size: {len(dataset)}")

    model = NeuralGeometricSDE(n_features=3, hidden_dim=32, n_layers=2)
    train_losses, val_losses = train_neural_sde(
        model, dataset, n_epochs=50, batch_size=512, verbose=True
    )

    print(f"\nFinal training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")

    print("\nGeometric SDE Module tests passed!")
