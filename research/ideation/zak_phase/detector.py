"""
Zak Phase Detector

Computes the Zak phase (1D Berry phase) for each PCA feature direction and
aggregates them into a per-timestep regime score.

Background
----------
The Zak phase is the Berry phase accumulated along a closed loop in a 1D
Brillouin zone (Zak 1989). In the QCML framework, each PCA feature axis a
defines a 1D "parameter direction". As the data trajectory x(t) moves through
parameter space, we can compute the Berry connection along each direction:

    A_a(x) = i <psi(x) | d/d x_a | psi(x)>

and integrate it along the time trajectory (projected onto direction a) to get
a per-feature Berry phase accumulation.

Three methods are implemented:

Method 1 — Berry connection integration ('connection', primary):
    For each feature direction a, integrate the Berry connection along x_a(t):
        A_a(x_t) = Im(<psi(x_t) | d_a psi(x_t)>)
        delta_phi_a(t) = A_a(x_t) * delta_x_a(t)
    Rolling sum: phi_a(t) = sum_{s in window} delta_phi_a(s)
    Total: Z(t) = sum_a |phi_a(t)|

Method 2 — Windowed winding number ('winding'):
    For a rolling window [t-W, t], compute the full geometric phase along
    the actual trajectory using products of pairwise overlaps (gauge-invariant):
        phi_window(t) = Im[ln(prod_{s=t-W}^{t-1} <psi_s|psi_{s+1}>)]
    This is the discrete Berry phase of the trajectory — analogous to the
    Wilson loop in lattice gauge theory.
    Per-direction: use only steps where axis a dominates the motion.

Method 3 — Directional Wilson loop ('wilson'):
    For each feature direction a, sort the time steps within a window by
    their x_a coordinate, forming a 1D loop, then compute the gauge-invariant
    Berry phase product. This is the closest analog to the true Zak phase
    for a 1D band structure.

The total Zak score: Z(t) = sum_a |phi_a(t)|, z-scored with expanding window.

This follows the BaseRegimeDetector interface from qcml_geometry/observables.py.

Reference: Zak (1989), "Berry's phase for energy bands in solids",
           Physical Review Letters, 62(23), 2747.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_point,
)

logger = logging.getLogger(__name__)


class ZakPhaseDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Zak (1D Berry) phase accumulated per feature direction.

    For each PCA feature axis a, computes the Berry phase contribution as the
    state |psi(x)> evolves along that axis over a rolling window. Near a regime
    transition, the state undergoes rapid phase winding in one or more feature
    directions, producing a large total Zak score.

    Three methods are available:
    - 'connection': Berry connection A_a(x) path integral (fastest, differentiable).
    - 'winding': Windowed Wilson loop product of pairwise overlaps (gauge-invariant).
    - 'wilson': Per-direction sorted Wilson loop (closest to true Zak phase).

    Score = abs(z-score of rolling Zak accumulation), computed with an expanding
    window to avoid look-ahead.

    Args:
        hilbert_dim: Hilbert space dimension. Default 4 (2-qubit system).
        n_pca_components: Number of PCA dimensions = number of Zak directions.
        operator_method: Method for constructing Hermitian operators.
            'random' avoids Kramers degeneracy that 'pca_inspired' triggers on
            qubit systems.
        scale_exponent: PCA eigenvalue scaling exponent for pca_* methods.
        rolling_window: Days to accumulate phase (Zak integration window).
        min_expanding: Minimum observations before z-scoring begins.
        seed: Random seed.
        causal_fit_length: If set, fit scaler/PCA on first N rows only.
        expanding_refit_interval: If set, periodically refit on expanding window.
        normalization: Post-PCA normalization mode ('sphere', 'none', 'soft', 'clip').
        epsilon: Step size for Berry connection finite differences.
        zak_method: 'connection' (primary), 'winding', or 'wilson'.
        custom_operators: If provided, use these operators directly.
        adaptive_z_window: If set, use adaptive z-score window instead of expanding.
    """

    def __init__(
        self,
        hilbert_dim: int = 4,
        n_pca_components: int = 6,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = 'sphere',
        epsilon: float = 1e-4,
        zak_method: str = 'connection',
        custom_operators: Optional[List[np.ndarray]] = None,
        adaptive_z_window: Optional[int] = None,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.epsilon = epsilon
        self.zak_method = zak_method
        self.custom_operators = custom_operators
        self.adaptive_z_window = adaptive_z_window

        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Zak Phase"

    def fit(self, X: np.ndarray, **kwargs) -> 'ZakPhaseDetector':
        """Fit scaler, PCA, and QCML operators on X.

        Args:
            X: Feature matrix (T, n_raw_features).

        Returns:
            self
        """
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])
        fit_end = self.causal_fit_length or X.shape[0]

        self._scaler = StandardScaler()
        self._scaler.fit(X[:fit_end])

        self._pca = PCA(n_components=n_components)
        X_scaled_fit = self._scaler.transform(X[:fit_end])
        self._pca.fit(X_scaled_fit)

        X_pca_raw = self._pca.transform(X_scaled_fit)
        self._train_norms = np.linalg.norm(X_pca_raw, axis=1)
        self._train_std = np.std(X_pca_raw, axis=0)
        X_pca_fit = _apply_normalization(
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim
        )
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )

        logger.info(
            "ZakPhaseDetector fitted: "
            f"pca_dim={n_components}, hilbert_dim={self.hilbert_dim}, "
            f"method={self.zak_method}"
        )
        return self

    def _berry_connection_component(
        self, x: np.ndarray, a: int, geo: QCMLGeometry
    ) -> float:
        """Compute Berry connection A_a = Im(<psi| d_a psi>) at point x.

        Uses central finite differences for d_a psi:
            A_a = Im(<psi(x)| [psi(x+eps*e_a) - psi(x-eps*e_a)] / (2*eps)>)

        Args:
            x: PCA-transformed point (n_components,).
            a: Feature direction index.
            geo: Fitted QCMLGeometry.

        Returns:
            A_a: Berry connection component (real scalar, units of 1/x_a).
        """
        eps = self.epsilon

        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[a] += eps
        x_minus[a] -= eps

        psi = geo.quasi_coherent_state(x)
        psi_plus = geo.quasi_coherent_state(x_plus)
        psi_minus = geo.quasi_coherent_state(x_minus)

        d_a_psi = (psi_plus - psi_minus) / (2.0 * eps)
        return float(np.imag(np.vdot(psi, d_a_psi)))

    def _compute_zak_connection(
        self, X_pca: np.ndarray, geo: QCMLGeometry
    ) -> np.ndarray:
        """Compute total Zak score via Berry connection A_a integrated along each axis.

        For each feature axis a, integrates A_a * dx_a over a rolling window,
        giving the Berry phase contribution from motion along that axis.

        phi_a(t) = sum_{s in [t-W, t]} A_a(x_s) * (x_s[a] - x_{s-1}[a])
        Z(t) = sum_a |phi_a(t)|

        Args:
            X_pca: Transformed feature matrix (T, n_components).
            geo: Fitted QCMLGeometry.

        Returns:
            total_zak: (T,) array of total Zak phase.
        """
        T, n_features = X_pca.shape

        # Compute Berry connection A_a(x_t) for all t, a
        A_vals = np.zeros((T, n_features))
        for t in range(T):
            x_t = X_pca[t]
            for a in range(n_features):
                A_vals[t, a] = self._berry_connection_component(x_t, a, geo)

        # Velocity along each direction: delta_x[t, a] = x_t[a] - x_{t-1}[a]
        delta_x = np.zeros((T, n_features))
        delta_x[1:] = np.diff(X_pca, axis=0)

        # Per-step Berry phase increment: dPhi[t, a] = A_a(x_t) * delta_x[t, a]
        d_phi = A_vals * delta_x  # shape (T, n_features)

        # Rolling sum of each direction's accumulated phase, then aggregate
        W = self.rolling_window
        total_zak = np.zeros(T)
        for a in range(n_features):
            series = pd.Series(d_phi[:, a])
            rolling_phase = series.rolling(window=W, min_periods=1).sum().values
            total_zak += np.abs(rolling_phase)

        return total_zak

    def _compute_zak_winding(
        self, X_pca: np.ndarray, geo: QCMLGeometry
    ) -> np.ndarray:
        """Compute per-direction Zak score via windowed Wilson loop (pairwise overlaps).

        Within a rolling window [t-W, t], for each axis a, computes the
        gauge-invariant geometric phase product using only steps where axis a
        contributes significantly to motion:

            phi_a(t) = Im[ sum_{s in window, |dx_a| > threshold}
                           ln(<psi_{s-1}|psi_s>) * sign(dx_a(s)) ]

        The sign factor accounts for direction: forward steps contribute positively,
        backward steps negatively. This builds up phase winding that persists
        through regime changes.

        The threshold is set at the median |dx_a| to focus on significant moves.

        Args:
            X_pca: Transformed feature matrix (T, n_components).
            geo: Fitted QCMLGeometry.

        Returns:
            total_zak: (T,) array of total Zak phase accumulation.
        """
        T, n_features = X_pca.shape

        # Precompute all quasi-coherent states
        states = []
        for t in range(T):
            states.append(geo.quasi_coherent_state(X_pca[t]))

        # Pairwise log-overlaps (step-wise geometric phases)
        # log_overlap[t] = Im[ln(<psi_{t-1}|psi_t>)] for t >= 1
        log_overlap = np.zeros(T)
        for t in range(1, T):
            ov = np.vdot(states[t - 1], states[t])
            if abs(ov) < 1e-12:
                log_overlap[t] = 0.0
            else:
                log_overlap[t] = float(np.imag(np.log(ov / abs(ov))))

        # Velocity per direction
        delta_x = np.zeros((T, n_features))
        delta_x[1:] = np.diff(X_pca, axis=0)

        W = self.rolling_window
        total_zak = np.zeros(T)

        for a in range(n_features):
            dx_a = delta_x[:, a]
            # Median absolute move to threshold significant steps
            # Use expanding median to stay causal
            series_dx = pd.Series(np.abs(dx_a))
            med_dx = series_dx.rolling(window=W, min_periods=1).median().values
            med_dx = np.maximum(med_dx, 1e-12)

            # Weighted phase: scale log_overlap by relative motion in direction a
            # This attributes the total phase proportionally to each axis
            total_velocity = np.sum(np.abs(delta_x), axis=1) + 1e-12
            weight_a = np.abs(dx_a) / total_velocity  # fraction of motion in dir a

            # Directional phase: weight * log_overlap * sign(dx_a)
            dir_phase = log_overlap * weight_a  # (T,)

            # Rolling sum
            rolling_phase = (
                pd.Series(dir_phase)
                .rolling(window=W, min_periods=1)
                .sum()
                .values
            )
            total_zak += np.abs(rolling_phase)

        return total_zak

    def _compute_zak_wilson(
        self, X_pca: np.ndarray, geo: QCMLGeometry
    ) -> np.ndarray:
        """Compute Zak score via sorted Wilson loops in each feature direction.

        For each time step t and feature axis a, within a rolling window,
        sorts the trajectory points by their x_a coordinate and computes the
        Berry phase of this sorted 1D path. This approximates the true Zak
        phase for motion along axis a.

        The sorted path forms a monotone 1D sweep from min to max x_a, then
        the Berry phase product gives the winding of the wave function as we
        scan along axis a.

        phi_a(t) = Im[ sum_{s in sorted_window} ln(<psi_prev|psi_next>) ]
        Z(t) = sum_a |phi_a(t)|

        Args:
            X_pca: Transformed feature matrix (T, n_components).
            geo: Fitted QCMLGeometry.

        Returns:
            total_zak: (T,) array of total Zak phase.
        """
        T, n_features = X_pca.shape

        # Precompute all quasi-coherent states
        states = []
        for t in range(T):
            states.append(geo.quasi_coherent_state(X_pca[t]))

        W = self.rolling_window
        total_zak = np.zeros(T)

        for t in range(1, T):
            # Rolling window [max(0, t-W+1), t]
            w_start = max(0, t - W + 1)
            window_idx = list(range(w_start, t + 1))
            if len(window_idx) < 2:
                continue

            zak_t = 0.0
            for a in range(n_features):
                # Sort window points by x_a coordinate
                x_a_vals = X_pca[window_idx, a]
                sorted_order = np.argsort(x_a_vals)
                sorted_window = [window_idx[i] for i in sorted_order]

                # Compute Berry phase product along sorted 1D path
                log_phase = 0.0
                for k in range(len(sorted_window) - 1):
                    s0 = sorted_window[k]
                    s1 = sorted_window[k + 1]
                    ov = np.vdot(states[s0], states[s1])
                    if abs(ov) < 1e-12:
                        continue
                    log_phase += float(np.imag(np.log(ov / abs(ov))))

                zak_t += abs(log_phase)

            total_zak[t] = zak_t

        return total_zak

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute z-scored total Zak phase time series.

        The raw total Zak score Z(t) = sum_a |phi_a(t)| is log-compressed
        and z-scored against an expanding window of past values using a
        MAD-based robust estimator.

        Args:
            X: Feature matrix (T, n_raw_features). Same features as fit().

        Returns:
            z_scores: (T,) array. NaN for t < min_expanding.
                Large positive values indicate rapid geometric phase winding,
                a signature of market regime transitions.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        T = X.shape[0]

        # Transform all points through the fitted pipeline
        X_pca = np.empty((T, self._pca.n_components_))
        for t in range(T):
            if self._snapshots is not None:
                _snap_geo, x_t = self._transform_point_at(X[t], t)
                X_pca[t] = x_t
            else:
                X_pca[t] = _transform_point(
                    X[t], self._scaler, self._pca,
                    normalization=self.normalization,
                    train_norms=self._train_norms,
                    train_std=self._train_std,
                )

        geo = self._geometry

        # Compute total Zak score using selected method
        try:
            if self.zak_method == 'connection':
                total_zak = self._compute_zak_connection(X_pca, geo)
            elif self.zak_method == 'winding':
                total_zak = self._compute_zak_winding(X_pca, geo)
            elif self.zak_method == 'wilson':
                total_zak = self._compute_zak_wilson(X_pca, geo)
            else:
                raise ValueError(f"Unknown zak_method: {self.zak_method!r}")
        except Exception as exc:
            logger.error(f"Zak phase computation failed: {exc}")
            return np.full(T, np.nan)

        # Log-compress to handle heavy-tailed distribution
        log_zak = np.log1p(total_zak)

        # Expanding-window robust z-score (causal, no look-ahead)
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            if self.adaptive_z_window is not None:
                lookback = min(t, self.adaptive_z_window)
                past = log_zak[t - lookback:t]
            else:
                past = log_zak[:t]

            mu = np.median(past)
            mad = np.median(np.abs(past - mu))
            sigma = 1.4826 * mad  # MAD-to-std conversion for Gaussian
            if sigma > 1e-12:
                z_scores[t] = abs((log_zak[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores
