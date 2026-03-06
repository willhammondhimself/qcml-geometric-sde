"""
Kibble-Zurek Mechanism Detector

The Kibble-Zurek (KZ) mechanism, originally derived for cosmological phase
transitions (Kibble 1976) and later extended to quantum systems (Zurek 1985),
predicts that the density of topological defects formed during a phase
transition scales as a power law of the quench rate:

    n_defect ~ tau_Q^{-d*nu / (1 + z*nu)}

where tau_Q is the "quench time" (inverse speed of approach to criticality),
d is spatial dimension, nu is the correlation length exponent, and z is the
dynamical critical exponent.

Financial Analogy
-----------------
- The "critical point" corresponds to a spectral gap closing in the error
  Hamiltonian H(x) — a near-degeneracy between the ground state and first
  excited state.
- The "quench rate" is how fast the market approaches that critical point:
  tau_Q(t) = gap(t) / |d(gap)/dt|
  Large tau_Q means slow approach (adiabatic); small tau_Q means rapid quench.
- The KZ signal = 1 / tau_Q^alpha captures "how far into the impulse regime"
  the system is — high signal = fast approach to criticality = regime transition.

The detector:
1. Fits PCA + StandardScaler on training data.
2. At each time step, computes spectral gap of H(x).
3. Estimates d(gap)/dt via finite differences over a short smoothing window.
4. Computes tau_Q = gap / |d(gap)/dt|  (clipped to avoid division by zero).
5. KZ signal = (1 / tau_Q)^alpha, but only when gap is closing (d(gap)/dt < 0).
   When gap is widening, KZ signal = 0 (system is moving away from criticality).
6. Applies an expanding-window Z-score for regime detection.

Reference
---------
Kibble, T.W.B. (1976). Topology of cosmic domains and strings. J. Phys. A, 9.
Zurek, W.H. (1985). Cosmological experiments in superfluid helium? Nature, 317.
Zurek, W.H., Dorner, U., & Zoller, P. (2005). Dynamics of a quantum phase
    transition. Phys. Rev. Lett., 95, 105701.
"""

import logging
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_point,
    _transform_array,
)

logger = logging.getLogger(__name__)


class KibbleZurekDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via the Kibble-Zurek quench rate from spectral gap dynamics.

    Tracks the rate at which the spectral gap of the QCML error Hamiltonian
    closes over time. When the gap closes rapidly (fast quench), the system is
    in the "impulse regime" of the KZ mechanism — the analogue of a market
    approaching a critical transition rapidly.

    KZ signal at time t:
        gap(t) = spectral gap of H(x_t)
        gap_rate(t) = (gap(t) - gap(t - gap_lookback)) / gap_lookback
        tau_Q(t) = gap(t) / |gap_rate(t)|     if gap_rate < 0 else infinity
        kz_raw(t) = (1 / tau_Q(t))^alpha       if gap closing
                  = 0                            if gap widening or stable
        score(t) = expanding-window Z-score of kz_raw(t)

    The alpha exponent corresponds to d*nu / (1 + z*nu) in the KZ formula.
    We treat it as a hyperparameter to be tuned (default 1.0 = linear scaling).

    Args:
        hilbert_dim: Hilbert space dimension (default 4 for 2-qubit system).
        n_pca_components: Number of PCA dimensions for feature reduction.
        operator_method: Method for constructing Hermitian operators.
            'random' avoids Kramers degeneracy on qubit systems.
        scale_exponent: PCA eigenvalue scaling exponent for pca_* methods.
        gap_lookback: Number of time steps over which to estimate d(gap)/dt.
            Larger values give smoother gradient estimates; default 5.
        alpha: KZ scaling exponent — maps tau_Q to defect density.
            Corresponds to d*nu/(1+z*nu); default 1.0 (linear).
        rolling_window: Days to smooth raw KZ values before z-scoring.
        min_expanding: Minimum observations before z-scoring begins.
        seed: Random seed for reproducibility.
        causal_fit_length: If set, fit scaler/PCA only on first N rows.
        expanding_refit_interval: If set, periodically refit on expanding window.
        normalization: Post-PCA normalization mode ('sphere', 'none', 'soft', 'clip').
        tau_clip_min: Minimum tau_Q to avoid log(0) instability (default 1e-8).
    """

    def __init__(
        self,
        hilbert_dim: int = 4,
        n_pca_components: int = 6,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        gap_lookback: int = 5,
        alpha: float = 1.0,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = 'sphere',
        tau_clip_min: float = 1e-8,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.gap_lookback = gap_lookback
        self.alpha = alpha
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.tau_clip_min = tau_clip_min
        self.custom_operators = custom_operators

        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Kibble-Zurek"

    def fit(self, X: np.ndarray, **kwargs) -> 'KibbleZurekDetector':
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
            "KibbleZurekDetector fitted: "
            f"pca_dim={n_components}, "
            f"gap_lookback={self.gap_lookback}, "
            f"alpha={self.alpha}"
        )
        return self

    def _compute_kz_signal(
        self, gap_series: np.ndarray, t: int
    ) -> float:
        """Compute KZ signal at time t from the spectral gap time series.

        Uses a backward finite difference over gap_lookback steps to estimate
        d(gap)/dt, then computes the quench time tau_Q and the KZ predictor.

        Args:
            gap_series: 1-D array of spectral gap values up to and including t.
            t: Current time index (into gap_series).

        Returns:
            kz: KZ signal (non-negative). Zero when gap is not closing.
        """
        if t < self.gap_lookback:
            return 0.0

        gap_now = gap_series[t]
        gap_past = gap_series[t - self.gap_lookback]

        # Finite difference estimate of d(gap)/dt
        gap_rate = (gap_now - gap_past) / self.gap_lookback

        # KZ mechanism only applies when gap is closing (moving toward critical point)
        if gap_rate >= 0.0:
            return 0.0

        # Quench time: how long until the gap closes at current rate
        # tau_Q = gap / |d(gap)/dt|
        tau_Q = gap_now / max(abs(gap_rate), self.tau_clip_min)
        tau_Q = max(tau_Q, self.tau_clip_min)

        # KZ predictor: 1 / tau_Q^alpha
        # Higher alpha emphasizes faster quenches more strongly
        kz = (1.0 / tau_Q) ** self.alpha
        return float(kz)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute expanding-window Z-scored KZ predictor time series.

        Process:
        1. Transform X through scaler + PCA.
        2. Compute spectral gap at every time step.
        3. Estimate quench rate and compute KZ signal.
        4. Apply log(1 + kz) for dynamic range compression.
        5. Smooth with a rolling window.
        6. Expanding-window Z-score (MAD-based, robust to outliers).

        Args:
            X: Feature matrix (T, n_raw_features). Same feature set as fit().

        Returns:
            z_scores: (T,) array. NaN for t < min_expanding.
                Large positive values indicate rapid approach to criticality.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        T = X.shape[0]

        # Transform all points through PCA pipeline
        if self._snapshots is not None:
            # Expanding refit mode: use causal snapshots
            X_pca = np.empty((T, self.n_pca_components))
            geometries = []
            for t in range(T):
                snap = self._get_snapshot_at(t)
                x_t = _transform_point(
                    X[t], snap['scaler'], snap['pca'], self.normalization,
                    snap['train_norms'], snap['train_std'],
                )
                X_pca[t] = x_t
                geometries.append(snap['geometry'])
        else:
            X_pca = _transform_array(
                X, self._scaler, self._pca, self.normalization,
                self._train_norms, self._train_std,
            )
            geometries = [self._geometry] * T

        # Step 1: Compute spectral gap at each time step
        logger.info("Computing spectral gaps over %d time steps...", T)
        gap_series = np.empty(T)
        for t in range(T):
            try:
                gap_series[t] = geometries[t].spectral_gap(X_pca[t])
            except Exception as exc:
                logger.warning("spectral_gap failed at t=%d: %s", t, exc)
                gap_series[t] = np.nan

        # Forward-fill NaN gaps (rare numerical failures)
        for t in range(1, T):
            if np.isnan(gap_series[t]):
                gap_series[t] = gap_series[t - 1]
        if np.isnan(gap_series[0]):
            gap_series[0] = 0.0

        # Step 2: Compute KZ signal at each time step
        kz_raw = np.empty(T)
        for t in range(T):
            kz_raw[t] = self._compute_kz_signal(gap_series, t)

        # Step 3: Log-compress to reduce heavy tails from very fast quenches
        kz_log = np.log1p(kz_raw)

        # Step 4: Rolling smooth
        if self.rolling_window > 1:
            kz_smooth = np.full(T, np.nan)
            for t in range(self.rolling_window - 1, T):
                window = kz_log[t - self.rolling_window + 1: t + 1]
                kz_smooth[t] = np.nanmean(window)
        else:
            kz_smooth = kz_log.copy()

        # Step 5: Expanding-window MAD Z-score
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            history = kz_smooth[:t + 1]
            history = history[~np.isnan(history)]
            if len(history) < self.min_expanding // 2:
                continue
            med = np.median(history)
            mad = np.median(np.abs(history - med))
            if mad < 1e-12:
                # Constant signal: z-score is zero (no deviation)
                z_scores[t] = 0.0
            else:
                z_scores[t] = (kz_smooth[t] - med) / (1.4826 * mad)

        return z_scores
