"""
Learned Operators for QCML Geometry

End-to-end differentiable operator learning via parameterized
Hermitian matrices. Instead of using PCA-based heuristic operators,
we learn A_k = V @ diag(λ) @ V† to maximize regime detection
effect size on training crises.

Hermiticity is enforced by construction:
  A = V @ diag(λ) @ V†

where V is unitary (parameterized via Cayley map) and λ is real.

Author: QCML Research
"""

import logging
from typing import Optional, Tuple, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


if HAS_TORCH:

    class HermitianOperatorModule(nn.Module):
        """Differentiable Hermitian operator parameterization.

        Parameterizes d Hermitian operators A_k of dimension n x n:
          A_k = V_k @ diag(λ_k) @ V_k†

        where V_k is unitary and λ_k are real eigenvalues.

        Args:
            n_operators: Number of operators d.
            hilbert_dim: Operator dimension n.
        """

        def __init__(self, n_operators: int, hilbert_dim: int):
            super().__init__()
            self.n_operators = n_operators
            self.hilbert_dim = hilbert_dim

            # Real eigenvalues for each operator
            self.eigenvalues = nn.Parameter(
                torch.randn(n_operators, hilbert_dim) * 0.1
            )

            # Skew-Hermitian generators for unitary matrices (Cayley map)
            # W_k is a real anti-symmetric matrix → exp(W) is orthogonal
            self.skew_params = nn.Parameter(
                torch.randn(n_operators, hilbert_dim, hilbert_dim) * 0.01
            )

        def get_operators(self) -> torch.Tensor:
            """Compute Hermitian operators from parameters.

            Returns:
                ops: (n_operators, hilbert_dim, hilbert_dim) real Hermitian matrices.
            """
            # Make skew-symmetric: W = (P - P^T) / 2
            skew = (self.skew_params - self.skew_params.transpose(-1, -2)) / 2

            # Cayley map: V = (I + W)(I - W)^{-1} — orthogonal by construction
            eye = torch.eye(self.hilbert_dim, device=skew.device).unsqueeze(0)
            V = torch.linalg.solve(eye - skew, eye + skew)

            # A_k = V @ diag(λ) @ V^T (real Hermitian)
            diag_lambda = torch.diag_embed(self.eigenvalues)
            ops = V @ diag_lambda @ V.transpose(-1, -2)

            return ops

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Compute error Hamiltonian ground state energy for batch of points.

            H(x) = (1/2) Σ_k (A_k - x_k I)²

            Args:
                x: (batch, d) feature vectors.

            Returns:
                energies: (batch,) ground state energies.
            """
            ops = self.get_operators()  # (d, n, n)
            batch_size = x.shape[0]
            n = self.hilbert_dim
            d = self.n_operators

            eye = torch.eye(n, device=x.device)

            # Build Hamiltonian for each batch element
            H = torch.zeros(batch_size, n, n, device=x.device)
            for k in range(d):
                diff = ops[k].unsqueeze(0) - x[:, k:k+1, None] * eye.unsqueeze(0)
                H = H + 0.5 * (diff @ diff)

            # Ground state energy via eigendecomposition
            eigenvalues = torch.linalg.eigvalsh(H)
            return eigenvalues[:, 0]  # smallest eigenvalue

        def compute_berry_curvature(
            self, x: torch.Tensor, indices: Tuple[int, int] = (0, 1),
            epsilon: float = 1e-4,
        ) -> torch.Tensor:
            """Compute Berry curvature via plaquette method (differentiable).

            Args:
                x: (batch, d) feature vectors.
                indices: Pair of parameter indices for 2D curvature.
                epsilon: Grid spacing.

            Returns:
                curvature: (batch,) Berry curvature values.
            """
            a, b = indices
            batch_size = x.shape[0]

            # Four corners of plaquette
            x00 = x.clone()
            x10 = x.clone(); x10[:, a] += epsilon
            x11 = x.clone(); x11[:, a] += epsilon; x11[:, b] += epsilon
            x01 = x.clone(); x01[:, b] += epsilon

            # Get ground states at each corner
            psi00 = self._ground_state(x00)
            psi10 = self._ground_state(x10)
            psi11 = self._ground_state(x11)
            psi01 = self._ground_state(x01)

            # Wilson loop: product of overlaps around plaquette
            overlap = (
                torch.sum(psi00.conj() * psi10, dim=-1) *
                torch.sum(psi10.conj() * psi11, dim=-1) *
                torch.sum(psi11.conj() * psi01, dim=-1) *
                torch.sum(psi01.conj() * psi00, dim=-1)
            )

            # Berry phase = Im(log(overlap))
            phase = torch.angle(overlap)
            curvature = phase / (epsilon ** 2)

            return curvature.real

        def _ground_state(self, x: torch.Tensor) -> torch.Tensor:
            """Compute ground state of H(x) for a batch.

            Args:
                x: (batch, d).

            Returns:
                psi: (batch, n) ground state vectors.
            """
            ops = self.get_operators()
            batch_size = x.shape[0]
            n = self.hilbert_dim
            d = self.n_operators
            eye = torch.eye(n, device=x.device)

            H = torch.zeros(batch_size, n, n, device=x.device)
            for k in range(d):
                diff = ops[k].unsqueeze(0) - x[:, k:k+1, None] * eye.unsqueeze(0)
                H = H + 0.5 * (diff @ diff)

            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            return eigenvectors[:, :, 0]  # ground state

    class LearnedOperatorQCML(nn.Module):
        """End-to-end learned QCML operators for regime detection.

        Learns Hermitian operators A_k to maximize Cohen's d (or a
        differentiable proxy) for crisis vs non-crisis discrimination.

        Args:
            n_features: Number of input features (= number of operators).
            hilbert_dim: Hilbert space dimension.
        """

        def __init__(self, n_features: int, hilbert_dim: int = 8):
            super().__init__()
            self.operators = HermitianOperatorModule(n_features, hilbert_dim)

        def forward(
            self, X: torch.Tensor, indices: Tuple[int, int] = (0, 1),
        ) -> torch.Tensor:
            """Compute Berry curvature scores for a batch of data points.

            Args:
                X: (batch, d) feature matrix.
                indices: Parameter indices for Berry curvature.

            Returns:
                scores: (batch,) Berry curvature values.
            """
            return self.operators.compute_berry_curvature(X, indices)

        def get_numpy_operators(self) -> List[np.ndarray]:
            """Export learned operators as numpy arrays.

            Returns:
                List of d Hermitian matrices, each (n, n).
            """
            with torch.no_grad():
                ops = self.operators.get_operators().cpu().numpy()
            return [ops[k] for k in range(ops.shape[0])]


def train_learned_operators(
    model: 'LearnedOperatorQCML',
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
) -> dict:
    """Train learned operators to maximize crisis discrimination.

    Uses a differentiable proxy for Cohen's d: the standardized mean
    difference of Berry curvature scores between crisis and non-crisis
    windows.

    Args:
        model: LearnedOperatorQCML module.
        X_train: (T, d) feature matrix.
        y_train: (T,) binary labels (0=normal, 1=crisis).
        n_epochs: Training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        Training history dict.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    torch.manual_seed(seed)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "proxy_d": []}

    crisis_mask = y_t > 0.5
    normal_mask = ~crisis_mask

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample a batch balanced between crisis/normal
        n_crisis = crisis_mask.sum().item()
        n_normal = normal_mask.sum().item()

        if n_crisis < 5 or n_normal < 5:
            logger.warning("Too few crisis/normal samples for training")
            break

        half_batch = min(batch_size // 2, n_crisis, n_normal)
        rng = torch.Generator().manual_seed(seed + epoch)

        crisis_idx = torch.where(crisis_mask)[0]
        normal_idx = torch.where(normal_mask)[0]

        crisis_sample = crisis_idx[torch.randperm(len(crisis_idx), generator=rng)[:half_batch]]
        normal_sample = normal_idx[torch.randperm(len(normal_idx), generator=rng)[:half_batch]]

        batch_idx = torch.cat([crisis_sample, normal_sample])
        X_batch = X_t[batch_idx]
        y_batch = y_t[batch_idx]

        # Forward pass: Berry curvature scores
        scores = model(X_batch)

        # Differentiable Cohen's d proxy
        crisis_scores = scores[y_batch > 0.5]
        normal_scores = scores[y_batch < 0.5]

        mean_crisis = crisis_scores.mean()
        mean_normal = normal_scores.mean()
        std_pooled = torch.sqrt(
            (crisis_scores.var() + normal_scores.var()) / 2 + 1e-8
        )
        proxy_d = (mean_crisis - mean_normal) / std_pooled

        # Loss: negative d (maximize effect size)
        loss = -proxy_d

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Project operators to ensure Hermiticity (already guaranteed by construction)
        history["loss"].append(loss.item())
        history["proxy_d"].append(proxy_d.item())

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, "
                f"proxy_d={proxy_d.item():.4f}"
            )

    return history
