"""
Q17: Learned Embedding via Neural Network — Theoretical Analysis + Minimal Empirical Test

Research question: Can we learn the optimal embedding mapping features to density
matrices via a neural network, instead of the hand-crafted error Hamiltonian
H(x) = sum_k (A_k - x_k * I)^2?

This module provides:

1. THEORETICAL ANALYSIS
   - Information captured vs lost by PCA + random operators
   - Effective rank of the feature-to-state mapping
   - Comparison: linear map (W*x -> |psi>) vs nonlinear H(x) ground state
   - Degrees of freedom comparison: current vs neural net embedding

2. MINIMAL EMPIRICAL TEST
   - Parametric Gibbs-state embedding: rho(x) = exp(sum_k x_k * sigma_k) / Tr(...)
   - Ground-state observable of the log-density-matrix
   - Tested on 4 crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
   - Cohen's d vs normal periods (60-day pre-crisis baseline)

Output: research/ideation/learned_embedding/smoke_results.json
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ["SPY", "DIA"]
START_DATE = "2005-01-01"
END_DATE = "2025-12-31"
TEST_CRISES = ["2008_gfc", "2020_covid", "2022_rates", "2023_svb"]
N_BOOTSTRAP = 1000  # Smoke-test speed; use 10000 for publication
CONTEXT_DAYS = 60   # Normal baseline window
HILBERT_DIM = 4     # 2-qubit Hilbert space


# ===========================================================================
# PART 1: THEORETICAL ANALYSIS
# ===========================================================================

def analyze_pca_operator_information(features: np.ndarray) -> dict:
    """Analyze what information the PCA + random operator embedding captures.

    The current QCML embedding works as follows:
      1. Standardize features -> x in R^d
      2. Fit PCA, keep top-k components
      3. Build Hermitian operators A_k (random or Pauli-based), scale by sqrt(eigenvalue_k)
      4. Compute H(x) = sum_k (A_k - x_k * I)^2
      5. Take ground state |psi(x)> as the state representation

    Information captured:
      - The dominant variance directions (PCA components)
      - The relative importance of each direction (via eigenvalue scaling)
      - Nonlinear structure via the quadratic Hamiltonian and ground-state selection

    Information lost:
      - Higher PCA components (rotational modes not in top-k)
      - Absolute scale (only relative ratios survive normalization)
      - Distributional shape beyond second moments (PCA only uses covariance)
      - Temporal ordering (the embedding is i.i.d. per time step)

    Args:
        features: Raw feature matrix, shape (T, d).

    Returns:
        dict with analysis results.
    """
    T, d = features.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA()
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # How many components to reach 90%, 95%, 99% variance explained?
    k90 = int(np.searchsorted(cumulative, 0.90)) + 1
    k95 = int(np.searchsorted(cumulative, 0.95)) + 1
    k99 = int(np.searchsorted(cumulative, 0.99)) + 1

    # Eigenvalue decay: how quickly does information fall off?
    eigenvalues = pca.explained_variance_
    # Effective rank: exp(entropy of normalized eigenvalue distribution)
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    effective_rank_pca = float(np.exp(entropy))

    # Information loss when using only hilbert_dim - 1 operators
    # (the Hilbert space of dim d_H can represent d_H^2 - 1 independent directions)
    max_operators_hilbert = HILBERT_DIM ** 2 - 1  # 15 for 4x4

    # How much variance is captured by top max_operators_hilbert components?
    k_used = min(max_operators_hilbert, d)
    variance_captured_by_operators = float(cumulative[k_used - 1])
    variance_lost = float(1.0 - variance_captured_by_operators)

    # Nonlinearity analysis: compare linear vs nonlinear projection variance
    # Project to PCA subspace (linear) and check reconstruction error
    k_recon = min(8, d)
    X_recon = pca.inverse_transform(pca.transform(X)[:, :k_recon] @ np.diag(
        np.concatenate([np.ones(k_recon), np.zeros(d - k_recon)])
    )[:k_recon])
    recon_error = float(np.mean((X - X_recon.reshape(T, d)) ** 2))

    return {
        "n_samples": T,
        "n_features": d,
        "explained_variance_ratio_top5": explained[:5].tolist(),
        "cumulative_variance_90": float(cumulative[k90 - 1]),
        "cumulative_variance_95": float(cumulative[k95 - 1]),
        "components_for_90pct": k90,
        "components_for_95pct": k95,
        "components_for_99pct": k99,
        "effective_rank_pca": float(effective_rank_pca),
        "max_operators_in_hilbert_dim4": max_operators_hilbert,
        "variance_captured_by_hilbert_operators": variance_captured_by_operators,
        "variance_lost_by_hilbert_truncation": variance_lost,
        "reconstruction_error_top8_linear": recon_error,
        "information_captured": (
            "PCA + random operators capture the dominant variance directions, weighted by "
            f"sqrt(eigenvalue). Top {k_used} operators cover {variance_captured_by_operators:.1%} "
            f"of variance. The quadratic Hamiltonian H(x) = sum_k (A_k - x_k*I)^2 adds "
            "nonlinearity: ground state selection is a non-analytic function of x, so the "
            "mapping x -> |psi(x)> can represent complex surfaces in Hilbert space."
        ),
        "information_lost": (
            f"PCA discards {variance_lost:.1%} of variance when limited to {k_used} operators. "
            "Additionally: (1) distributional shape beyond second moments is ignored, "
            "(2) temporal ordering is discarded (i.i.d. per timestep), "
            "(3) the specific basis of random operators is arbitrary — a neural network "
            "could learn basis operators optimized for crisis discrimination."
        ),
    }


def analyze_effective_rank_of_mapping(features: np.ndarray, hilbert_dim: int = 4) -> dict:
    """Compute the effective rank of the feature-to-state mapping.

    The mapping phi: R^d -> CP^{d_H - 1} (projective Hilbert space) is described
    by its Jacobian J = d|psi>/dx at each point x. The effective rank measures
    how many independent directions in feature space actually influence the state.

    We compute this empirically: take a random sample of feature vectors,
    compute states |psi(x)> using random operators, and measure the effective
    dimension of the image in Hilbert space.

    Args:
        features: Feature matrix, shape (T, d).
        hilbert_dim: Hilbert space dimension.

    Returns:
        dict with effective rank analysis.
    """
    from qcml_geometry.core import QCMLGeometry

    T, d = features.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Use a subsample for speed
    n_sample = min(200, T)
    rng = np.random.default_rng(42)
    idx = rng.choice(T, n_sample, replace=False)
    X_sample = X[idx]

    # Reduce to PCA components that fit in Hilbert space operators
    n_components = min(d, hilbert_dim ** 2 - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_sample)

    # Fit QCML geometry with random operators
    geometry = QCMLGeometry(n_features=n_components, hilbert_dim=hilbert_dim)
    geometry.fit_operators(X_pca, method="random")

    # Compute states for each sample point
    states = np.zeros((n_sample, hilbert_dim), dtype=complex)
    for i, x in enumerate(X_pca):
        states[i] = geometry.quasi_coherent_state(x)

    # Compute the Gram matrix: G[i,j] = |<psi_i|psi_j>|^2
    gram = np.abs(states @ states.conj().T) ** 2  # (n_sample, n_sample)

    # Effective dimension via IPR: D_eff = n_sample^2 / sum(G)
    # This is the same formula used in effective_state_dimension
    d_eff_state_space = float(n_sample ** 2 / np.sum(gram))

    # SVD of the state matrix to measure effective rank of the image
    # States are rows in C^{hilbert_dim}; singular values tell us how many
    # dimensions of Hilbert space are actually used.
    U, s, Vh = np.linalg.svd(states, full_matrices=False)
    s_normalized = s / (s.sum() + 1e-15)
    entropy_svd = float(-np.sum(s_normalized * np.log(s_normalized + 1e-15)))
    effective_rank_image = float(np.exp(entropy_svd))

    # The maximum possible rank is min(n_sample, hilbert_dim)
    max_rank = min(n_sample, hilbert_dim)

    # Fraction of Hilbert space directions utilized
    utilization = effective_rank_image / hilbert_dim

    # Measure separability: how much do crisis vs normal states differ?
    # Check if states cluster in Hilbert space
    singular_values = s[:hilbert_dim].tolist()

    return {
        "n_sample": n_sample,
        "n_components_used": n_components,
        "hilbert_dim": hilbert_dim,
        "max_possible_rank": max_rank,
        "effective_rank_image_svd": float(effective_rank_image),
        "effective_dimension_ipr": float(d_eff_state_space),
        "hilbert_space_utilization_fraction": float(utilization),
        "singular_values_of_state_matrix": singular_values,
        "interpretation": (
            f"The mapping x -> |psi(x)> produces states with effective rank {effective_rank_image:.2f} "
            f"in a {hilbert_dim}-dimensional Hilbert space (utilization {utilization:.1%}). "
            f"This means the current embedding explores {effective_rank_image:.2f} / {hilbert_dim} "
            "effective directions. A neural network could potentially saturate all directions "
            "by optimizing operators for maximum discriminability."
        ),
    }


def analyze_linear_vs_nonlinear(features: np.ndarray, hilbert_dim: int = 4) -> dict:
    """Test whether a simple linear map W*x -> |psi> could match the nonlinear H(x) ground state.

    The current mapping is highly nonlinear: |psi(x)> = ground_state(H(x)) where
    H(x) = sum_k (A_k - x_k*I)^2. The ground state is an eigenvector — a decidedly
    nonlinear function of x.

    A linear map would be: |psi_linear(x)> = normalize(W * x) where W is a
    complex matrix of shape (hilbert_dim, n_features).

    We test linearity by:
    1. Computing nonlinear states for a sample of feature vectors
    2. Fitting a linear map (least squares) from x to psi
    3. Measuring how well the linear map reproduces the nonlinear states (via fidelity)

    Args:
        features: Feature matrix, shape (T, d).
        hilbert_dim: Hilbert space dimension.

    Returns:
        dict with linear vs nonlinear comparison.
    """
    from qcml_geometry.core import QCMLGeometry

    T, d = features.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_components = min(d, hilbert_dim ** 2 - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Use train/test split to avoid overfitting in linear map test
    n_train = min(300, int(T * 0.6))
    n_test = min(100, int(T * 0.2))

    X_train = X_pca[:n_train]
    X_test = X_pca[n_train:n_train + n_test]

    geometry = QCMLGeometry(n_features=n_components, hilbert_dim=hilbert_dim)
    geometry.fit_operators(X_train, method="random")

    # Compute nonlinear ground states
    psi_train = np.array([geometry.quasi_coherent_state(x) for x in X_train])
    psi_test = np.array([geometry.quasi_coherent_state(x) for x in X_test])

    # Fit linear map: psi ≈ X @ W^T, solve with least squares
    # Real and imaginary parts separately
    W_real, _, _, _ = np.linalg.lstsq(X_train, psi_train.real, rcond=None)
    W_imag, _, _, _ = np.linalg.lstsq(X_train, psi_train.imag, rcond=None)

    # Predict on test set
    psi_test_pred_raw = X_test @ W_real + 1j * (X_test @ W_imag)
    norms = np.linalg.norm(psi_test_pred_raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    psi_test_pred = psi_test_pred_raw / norms

    # Measure fidelity between linear prediction and nonlinear ground state
    fidelities = np.abs(np.einsum("ij,ij->i", psi_test_pred.conj(), psi_test)) ** 2

    mean_fidelity = float(np.mean(fidelities))
    median_fidelity = float(np.median(fidelities))
    std_fidelity = float(np.std(fidelities))

    # Compare: random (trivial) baseline fidelity
    # For dim-d_H system, random state fidelity expectation = 1/d_H
    random_baseline_fidelity = 1.0 / hilbert_dim

    # Fidelity gain of linear model over random baseline
    gain = (mean_fidelity - random_baseline_fidelity) / (1.0 - random_baseline_fidelity)

    # Check training set fit as upper bound
    psi_train_pred_raw = X_train @ W_real + 1j * (X_train @ W_imag)
    norms_tr = np.linalg.norm(psi_train_pred_raw, axis=1, keepdims=True)
    psi_train_pred = psi_train_pred_raw / np.maximum(norms_tr, 1e-15)
    train_fidelities = np.abs(np.einsum("ij,ij->i", psi_train_pred.conj(), psi_train)) ** 2
    train_mean_fidelity = float(np.mean(train_fidelities))

    # Interpretation: fidelity >> 1/d_H implies the nonlinear map has "linear character"
    # fidelity ≈ 1/d_H implies the mapping is genuinely nonlinear
    is_effectively_linear = mean_fidelity > 0.5 * (1 + random_baseline_fidelity)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "n_components": n_components,
        "hilbert_dim": hilbert_dim,
        "random_baseline_fidelity": float(random_baseline_fidelity),
        "linear_map_train_fidelity": float(train_mean_fidelity),
        "linear_map_test_mean_fidelity": float(mean_fidelity),
        "linear_map_test_median_fidelity": float(median_fidelity),
        "linear_map_test_std_fidelity": float(std_fidelity),
        "fidelity_gain_over_random": float(gain),
        "is_effectively_linear": bool(is_effectively_linear),
        "interpretation": (
            f"A linear map achieves test fidelity {mean_fidelity:.3f} vs random baseline "
            f"{random_baseline_fidelity:.3f}. "
            + (
                "The mapping has substantial linear character — a neural network may not need "
                "deep nonlinearity to outperform the current approach. A single linear layer "
                "with optimized weights could already improve discriminability."
                if is_effectively_linear else
                "The mapping is genuinely nonlinear — a linear model cannot reproduce the "
                "H(x) ground state. This validates that the nonlinear ground-state construction "
                "is important, and a neural network would need at least 1-2 hidden layers to "
                "approximate it well."
            )
        ),
    }


def analyze_degrees_of_freedom(n_features: int, hilbert_dim: int = 4) -> dict:
    """Estimate and compare degrees of freedom for current vs neural net embedding.

    Current QCML embedding parameters:
      - n_features Hermitian operators, each of size (hilbert_dim x hilbert_dim)
      - A Hermitian matrix has d_H^2 real parameters (d_H diagonal + 2*d_H*(d_H-1)/2 off-diagonal)
      - Total: n_features * d_H^2 parameters

    Neural net alternatives:
      - Linear: W in C^{d_H x n_features} -> 2 * d_H * n_features real params
      - Shallow (1 hidden layer, h units): W1 (h x n_features) + W2 (d_H x h)
        -> h * n_features + d_H * h real params + bias
      - Deep (2 hidden layers): W1 + W2 + W3 + biases

    The key trade-off: more parameters = more expressive but harder to optimize,
    more prone to overfitting on small financial datasets.

    Args:
        n_features: Number of input features.
        hilbert_dim: Hilbert space dimension.

    Returns:
        dict with degrees of freedom comparison.
    """
    d_H = hilbert_dim

    # Real parameters in a d_H x d_H Hermitian matrix
    hermitian_params = d_H ** 2  # d_H real diagonal + d_H*(d_H-1) real off-diagonal

    # Current QCML embedding: one operator per feature
    current_params = n_features * hermitian_params

    # However, only the operators matter modulo unitary equivalence:
    # The ground state is invariant to unitary transformations U A_k U^dagger
    # So the effective dof is less: SU(d_H) has d_H^2 - 1 generators
    # Quotient by U(1) phase per state: subtract 1
    unitary_redundancy = d_H ** 2 - 1  # real dim of SU(d_H)
    effective_current_params = current_params - unitary_redundancy

    # Linear neural net (no hidden layers): maps x -> psi directly
    # Complex weight matrix W in C^{d_H x n_features} = 2 real matrices
    linear_nn_params = 2 * d_H * n_features

    # Shallow net (1 hidden layer of width h):
    hidden_widths = [16, 32, 64, 128]
    shallow_params = {}
    for h in hidden_widths:
        # Input -> hidden: h * n_features + h (bias)
        # Hidden -> output (complex d_H): 2 * d_H * h + 2 * d_H (bias)
        p = h * n_features + h + 2 * d_H * h + 2 * d_H
        shallow_params[f"h={h}"] = p

    # Deep net (2 hidden layers, h x h):
    deep_params = {}
    for h in hidden_widths:
        # W1: h * n_features + h
        # W2: h * h + h
        # W3 (output): 2 * d_H * h + 2 * d_H
        p = h * n_features + h + h * h + h + 2 * d_H * h + 2 * d_H
        deep_params[f"h={h}"] = p

    # Gibbs state parameterization (this module's empirical test):
    # rho(x) = exp(sum_k x_k * sigma_k) / Tr(...) where sigma_k are n_features matrices
    # Number of parameters: n_features operators, each Hermitian d_H x d_H
    # But sigma_k can be fixed (Pauli basis), so 0 learnable params!
    # Or sigma_k can be learned: n_features * d_H^2 params (same as current)
    gibbs_fixed_basis_params = 0  # If sigma_k = Pauli basis (fixed)
    gibbs_learned_basis_params = n_features * hermitian_params  # If sigma_k learned

    # Context-aware: with ~4000 training timesteps per crisis, overfitting risk
    typical_training_samples = 4000  # rows in feature matrix

    return {
        "n_features": n_features,
        "hilbert_dim": d_H,
        "hermitian_matrix_params": hermitian_params,
        "current_qcml_params_total": current_params,
        "current_qcml_effective_params": effective_current_params,
        "linear_neural_net_params": linear_nn_params,
        "shallow_neural_net_params": shallow_params,
        "deep_neural_net_params": deep_params,
        "gibbs_fixed_basis_params": gibbs_fixed_basis_params,
        "gibbs_learned_basis_params": gibbs_learned_basis_params,
        "typical_training_samples": typical_training_samples,
        "overfit_risk_analysis": {
            "current_qcml_ratio": round(effective_current_params / typical_training_samples, 3),
            "linear_nn_ratio": round(linear_nn_params / typical_training_samples, 3),
            "shallow_h32_ratio": round(shallow_params["h=32"] / typical_training_samples, 3),
        },
        "recommendation": (
            f"Current embedding has {effective_current_params} effective parameters "
            f"({effective_current_params / typical_training_samples:.1%} of training samples). "
            f"A linear neural net has {linear_nn_params} params "
            f"({linear_nn_params / typical_training_samples:.1%}). "
            "Both are well-regularized for ~4K samples. A shallow network (h=32) is marginally "
            "larger but still within overfitting-safe range. The key advantage of a neural net "
            "is not expressivity but *optimization*: gradient descent on a discriminative objective "
            "(e.g., crisis vs normal fidelity contrast) vs the current fixed random operator design."
        ),
    }


# ===========================================================================
# PART 2: MINIMAL EMPIRICAL TEST — GIBBS STATE EMBEDDING
# ===========================================================================

# Pauli matrices for 2-qubit system (4 x 4)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Complete Pauli basis for 2-qubit system: {I, X, Y, Z}^(x2), excluding I(x)I
_PAULI_BASIS_4 = [
    np.kron(s1, s2)
    for s1 in [_I2, _X, _Y, _Z]
    for s2 in [_I2, _X, _Y, _Z]
][1:]  # 15 Hermitian traceless-or-I matrices


def _gibbs_density_matrix(x: np.ndarray, sigma_list: list) -> np.ndarray:
    """Compute Gibbs density matrix rho(x) = exp(M) / Tr(exp(M)).

    M = sum_k x_k * sigma_k where sigma_k are Hermitian matrices.
    This is a well-defined density matrix (positive semi-definite, trace 1).

    The observable is the ground state of -log(rho) = -M + log(Z)*I, which
    equals the ground state of M with reversed ordering (highest eigenstate of M
    becomes lowest eigenstate of -M). We take the ground state of -M for
    consistency with the QCML convention: "crisis = low energy state".

    Args:
        x: Feature vector, shape (n_features,). Only up to len(sigma_list) used.
        sigma_list: List of Hermitian matrices, each shape (d_H, d_H).

    Returns:
        rho: Density matrix, shape (d_H, d_H).
    """
    n = min(len(x), len(sigma_list))
    M = sum(float(x[k]) * sigma_list[k] for k in range(n))
    # Compute matrix exponential via eigendecomposition for stability
    eigvals, eigvecs = np.linalg.eigh(M)
    exp_eigvals = np.exp(eigvals - np.max(eigvals))  # numerically stable
    Z = np.sum(exp_eigvals)
    rho = eigvecs @ np.diag(exp_eigvals / Z) @ eigvecs.conj().T
    return rho


def _gibbs_ground_state_score(x: np.ndarray, sigma_list: list) -> float:
    """Compute the Gibbs-state score for a feature vector.

    Score = minimum eigenvalue of -log(rho(x)), which corresponds to
    the purity of the highest-eigenvalue direction of rho(x).

    In normal markets: rho(x) is approximately maximally mixed (all eigenvalues ~1/d_H),
    so the spread of log(rho) eigenvalues is small.

    In crises: rho(x) concentrates on one direction, so the smallest
    eigenvalue of -log(rho) is very negative (large |score|).

    Equivalently: score = max eigenvalue of log(rho) = log(max eigenvalue of rho).

    Args:
        x: Feature vector, shape (n_features,).
        sigma_list: Pauli-basis matrices.

    Returns:
        score: log(max_eigenvalue_of_rho), in (-inf, 0].
    """
    rho = _gibbs_density_matrix(x, sigma_list)
    eigvals = np.linalg.eigvalsh(rho)
    max_eig = np.max(eigvals)
    return float(np.log(max(max_eig, 1e-15)))


def _prepare_gibbs_sigmas(n_features: int, hilbert_dim: int = 4) -> list:
    """Prepare Pauli basis operators for Gibbs embedding.

    Selects the first n_features operators from the Pauli basis,
    or pads with random Hermitian matrices if n_features > len(basis).

    Args:
        n_features: Number of features (= number of sigma operators needed).
        hilbert_dim: Hilbert space dimension (must be 4 for 2-qubit Pauli basis).

    Returns:
        List of Hermitian matrices, length = n_features.
    """
    if hilbert_dim == 4:
        basis = _PAULI_BASIS_4
    else:
        # For other dims, use random Hermitian matrices
        rng = np.random.default_rng(42)
        basis = []
        for k in range(n_features):
            A = rng.standard_normal((hilbert_dim, hilbert_dim)) + \
                1j * rng.standard_normal((hilbert_dim, hilbert_dim))
            basis.append((A + A.conj().T) / 2)

    # If n_features > len(basis), pad with random Hermitian
    sigmas = list(basis[:n_features])
    if n_features > len(basis):
        rng = np.random.default_rng(123)
        for _ in range(n_features - len(basis)):
            A = rng.standard_normal((hilbert_dim, hilbert_dim)) + \
                1j * rng.standard_normal((hilbert_dim, hilbert_dim))
            sigmas.append((A + A.conj().T) / 2)
    return sigmas


def compute_gibbs_scores(features_pca: np.ndarray, sigma_list: list) -> np.ndarray:
    """Compute Gibbs-state regime scores for all timesteps.

    Args:
        features_pca: PCA-projected feature matrix, shape (T, n_features).
        sigma_list: Pauli/Hermitian basis operators, length = n_features.

    Returns:
        scores: 1-D array of shape (T,) with log(max_eigenvalue_of_rho).
    """
    T = len(features_pca)
    scores = np.zeros(T)
    for t, x in enumerate(features_pca):
        scores[t] = _gibbs_ground_state_score(x, sigma_list)
    return scores


def run_empirical_test(
    features: np.ndarray,
    dates,
    n_pca_components: int = 13,
) -> dict:
    """Run the Gibbs-state embedding empirical test on 4 crises.

    Pipeline:
      1. Standardize features
      2. Project to PCA subspace (n_pca_components)
      3. Scale by PCA coefficients so dominant directions dominate M
      4. Compute Gibbs density matrix rho(x) = exp(sum_k x_k sigma_k) / Z
      5. Score = log(max eigenvalue of rho) — large crisis, small normal
      6. Expanding-window z-score normalization
      7. Cohen's d per crisis (bootstrap CI, n=1000)

    Args:
        features: Feature matrix (T, d).
        dates: DatetimeIndex aligned with features.
        n_pca_components: Number of PCA components. Capped at 15 (Pauli basis size).

    Returns:
        dict with per-crisis Cohen's d results.
    """
    n_pca_components = min(n_pca_components, len(_PAULI_BASIS_4), features.shape[1])
    logger.info(f"Gibbs embedding: n_pca_components={n_pca_components}")

    # Standardize and project
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X)

    # Scale PCA components by sqrt(explained_variance) so dominant directions
    # contribute more to M = sum_k x_k sigma_k (analogous to QCML operator scaling)
    ev = pca.explained_variance_
    scale = np.sqrt(np.maximum(ev, 1e-10))
    X_scaled = X_pca * scale  # (T, n_pca_components)

    # Prepare Pauli basis operators
    sigma_list = _prepare_gibbs_sigmas(n_pca_components, hilbert_dim=HILBERT_DIM)

    # Compute raw Gibbs scores
    logger.info("Computing Gibbs density matrix scores...")
    T = len(features)
    raw_scores = compute_gibbs_scores(X_scaled, sigma_list)

    # Expanding-window z-score (match QCML observatory normalization)
    MIN_EXPANDING = 60
    z_scores = np.full(T, np.nan)
    for t in range(MIN_EXPANDING, T):
        past = raw_scores[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma_std = np.std(past_valid, ddof=1)
        if sigma_std > 1e-12:
            z_scores[t] = (raw_scores[t] - mu) / sigma_std

    n_valid = int(np.sum(~np.isnan(z_scores)))
    logger.info(f"Valid z-scores: {n_valid} / {T}")

    # Evaluate per crisis
    dates_idx = pd.DatetimeIndex(dates)
    results = {}

    for crisis_key in TEST_CRISES:
        crisis = ALL_CRISES[crisis_key]
        crisis_start = pd.Timestamp(crisis["start"])
        crisis_end = pd.Timestamp(crisis["end"])
        normal_start = crisis_start - pd.Timedelta(days=CONTEXT_DAYS)

        crisis_mask = (dates_idx >= crisis_start) & (dates_idx <= crisis_end)
        normal_mask = (dates_idx >= normal_start) & (dates_idx < crisis_start)

        crisis_scores = z_scores[np.asarray(crisis_mask)]
        normal_scores = z_scores[np.asarray(normal_mask)]

        crisis_valid = crisis_scores[~np.isnan(crisis_scores)]
        normal_valid = normal_scores[~np.isnan(normal_scores)]

        logger.info(
            f"{crisis_key}: n_crisis={len(crisis_valid)}, n_normal={len(normal_valid)}"
        )

        if len(crisis_valid) >= 2 and len(normal_valid) >= 2:
            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                crisis_valid, normal_valid, n_bootstrap=N_BOOTSTRAP, seed=42
            )
        else:
            d, ci_lo, ci_hi = np.nan, np.nan, np.nan

        results[crisis_key] = {
            "cohens_d": float(d) if not np.isnan(d) else None,
            "ci_lo": float(ci_lo) if not np.isnan(ci_lo) else None,
            "ci_hi": float(ci_hi) if not np.isnan(ci_hi) else None,
            "crisis_n": int(len(crisis_valid)),
            "normal_n": int(len(normal_valid)),
            "crisis_mean_zscore": float(np.nanmean(crisis_valid)) if len(crisis_valid) > 0 else None,
            "normal_mean_zscore": float(np.nanmean(normal_valid)) if len(normal_valid) > 0 else None,
        }
        if not np.isnan(d):
            logger.info(f"  d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    d_vals = [r["cohens_d"] for r in results.values() if r["cohens_d"] is not None]
    median_d = float(np.median(d_vals)) if d_vals else None

    return {
        "per_crisis": results,
        "d_values": d_vals,
        "median_d": median_d,
        "max_d": float(np.max(d_vals)) if d_vals else None,
        "n_valid_scores": n_valid,
        "n_pca_components": n_pca_components,
        "raw_scores_stats": {
            "mean": float(np.nanmean(raw_scores)),
            "std": float(np.nanstd(raw_scores)),
            "min": float(np.nanmin(raw_scores)),
            "max": float(np.nanmax(raw_scores)),
        },
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Fetch data
    # -------------------------------------------------------------------------
    logger.info(f"Fetching data for {SYMBOLS} {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(SYMBOLS, START_DATE, END_DATE)
    close_prices = raw_df["close"].unstack("symbol").dropna()
    logger.info(f"Close prices: {close_prices.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Create feature matrix
    # -------------------------------------------------------------------------
    logger.info("Creating feature matrix...")
    features, dates = create_feature_matrix(close_prices)
    T, d = features.shape
    logger.info(f"Features: {features.shape}  ({dates[0]} to {dates[-1]})")

    # -------------------------------------------------------------------------
    # Step 3: Theoretical analysis
    # -------------------------------------------------------------------------
    logger.info("=== THEORETICAL ANALYSIS ===")

    logger.info("1. Analyzing PCA + operator information capture/loss...")
    info_analysis = analyze_pca_operator_information(features)

    logger.info("2. Analyzing effective rank of feature-to-state mapping...")
    rank_analysis = analyze_effective_rank_of_mapping(features, HILBERT_DIM)

    logger.info("3. Linear vs nonlinear mapping comparison...")
    linear_analysis = analyze_linear_vs_nonlinear(features, HILBERT_DIM)

    logger.info("4. Degrees of freedom comparison...")
    dof_analysis = analyze_degrees_of_freedom(d, HILBERT_DIM)

    # -------------------------------------------------------------------------
    # Step 4: Empirical test — Gibbs state embedding
    # -------------------------------------------------------------------------
    logger.info("=== EMPIRICAL TEST: GIBBS STATE EMBEDDING ===")
    t_emp = time.time()
    empirical_results = run_empirical_test(features, dates)
    dt_emp = time.time() - t_emp
    logger.info(f"Empirical test completed in {dt_emp:.1f}s")

    # -------------------------------------------------------------------------
    # Step 5: Determine recommendation
    # -------------------------------------------------------------------------
    median_d = empirical_results["median_d"]
    passes_threshold = median_d is not None and median_d > 0.3
    is_effectively_linear = linear_analysis["is_effectively_linear"]
    hilbert_utilization = rank_analysis["hilbert_space_utilization_fraction"]

    if median_d is not None and median_d >= 0.5 and not is_effectively_linear:
        recommendation = "proceed"
    elif median_d is not None and median_d >= 0.2:
        recommendation = "future_work"
    else:
        recommendation = "too_complex"

    # -------------------------------------------------------------------------
    # Step 6: Compose analysis findings
    # -------------------------------------------------------------------------
    analysis_findings = (
        f"Current QCML uses H(x) = sum_k (A_k - x_k*I)^2 with random/PCA-scaled operators. "
        f"Analysis shows: (1) PCA captures {info_analysis['components_for_95pct']} components "
        f"for 95% variance; d_H=4 Hilbert space limits to {info_analysis['max_operators_in_hilbert_dim4']} "
        f"operators covering {info_analysis['variance_captured_by_hilbert_operators']:.1%} of variance. "
        f"(2) The state-space image has effective rank {rank_analysis['effective_rank_image_svd']:.2f} "
        f"out of {HILBERT_DIM} dimensions (utilization {hilbert_utilization:.1%}). "
        f"(3) Linear map achieves fidelity {linear_analysis['linear_map_test_mean_fidelity']:.3f} "
        f"vs random baseline {linear_analysis['random_baseline_fidelity']:.3f} — "
        f"{'mapping has linear character' if is_effectively_linear else 'mapping is genuinely nonlinear'}. "
        f"(4) Gibbs state (exp(sum x_k sigma_k)/Z) with PCA-scaled features achieves "
        f"median Cohen's d={median_d:.3f} on 4 crises. "
        f"(5) DOF comparison: current QCML ~{dof_analysis['current_qcml_effective_params']} effective "
        f"params vs linear NN ~{dof_analysis['linear_neural_net_params']} params — "
        f"comparable scale, favoring supervised optimization as the primary advantage of neural nets."
    )

    # -------------------------------------------------------------------------
    # Step 7: Assemble and save results
    # -------------------------------------------------------------------------
    total_time = time.time() - t0

    output = {
        "question": "Q17: Learned embedding via neural network mapping returns to density matrices",
        "embedding_tested": "Gibbs state rho(x) = exp(sum_k x_k * sigma_k) / Tr(exp(...)), "
                            "sigma_k = 2-qubit Pauli basis, scored by log(max_eigenvalue)",
        "theoretical_analysis": {
            "pca_operator_information": info_analysis,
            "effective_rank_of_mapping": rank_analysis,
            "linear_vs_nonlinear": linear_analysis,
            "degrees_of_freedom": dof_analysis,
        },
        "empirical_test": empirical_results,
        "analysis_findings": analysis_findings,
        "empirical_d": {
            "per_crisis": {
                k: v["cohens_d"] for k, v in empirical_results["per_crisis"].items()
            },
            "median": median_d,
        },
        "passes_threshold": passes_threshold,
        "recommendation": recommendation,
        "timing": {
            "total_seconds": round(total_time, 1),
            "empirical_test_seconds": round(dt_emp, 1),
        },
    }

    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "smoke_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Q17: LEARNED EMBEDDING — ANALYSIS RESULTS")
    print("=" * 70)

    print("\n--- THEORETICAL ANALYSIS ---")
    print(f"  PCA components for 95% variance: {info_analysis['components_for_95pct']}")
    print(f"  Variance captured by d_H=4 operators: {info_analysis['variance_captured_by_hilbert_operators']:.1%}")
    print(f"  Effective rank of state-space image: {rank_analysis['effective_rank_image_svd']:.2f} / {HILBERT_DIM}")
    print(f"  Hilbert space utilization: {hilbert_utilization:.1%}")
    print(f"  Linear map test fidelity: {linear_analysis['linear_map_test_mean_fidelity']:.3f} "
          f"(baseline: {linear_analysis['random_baseline_fidelity']:.3f})")
    print(f"  Mapping character: {'linear' if is_effectively_linear else 'nonlinear'}")
    print(f"  Current QCML effective params: {dof_analysis['current_qcml_effective_params']}")
    print(f"  Linear NN params: {dof_analysis['linear_neural_net_params']}")

    print("\n--- EMPIRICAL TEST (GIBBS STATE) ---")
    for crisis_key, r in empirical_results["per_crisis"].items():
        d = r["cohens_d"]
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r["ci_lo"] is not None else "N/A"
        d_str = f"{d:.3f}" if d is not None else "N/A"
        print(f"  {crisis_key:25s}  d={d_str}  CI={ci}  "
              f"(n_crisis={r['crisis_n']}, n_normal={r['normal_n']})")

    print(f"\n  Median d: {median_d:.3f}" if median_d is not None else "\n  Median d: N/A")
    print(f"  Passes threshold (d>0.3): {passes_threshold}")
    print(f"  Recommendation: {recommendation}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
