"""
Online Regime Detection Models

Streaming detection models that produce P(crisis) at each timestep using only
data available up to that point (strictly causal).

Models:
    OnlineGeometricFeatureComputer — streaming geometric feature extraction
    ExpandingPercentileDetector    — unsupervised percentile-rank baseline (RMS aggregation)
    OnlineBayesianDetector         — multivariate Bayesian filtering with sticky transitions
    OnlineHMMDetector              — periodic HMM refit + forward algorithm
    OnlineLogisticDetector         — expanding-window logistic regression
    OnlineEnsembleDetector         — weighted ensemble of multiple detectors

All models inherit OnlineDetectorBase and implement:
    update(features_dict) -> float  (returns P(crisis) at current step)
    reset()                         (reset internal state)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .core import QCMLGeometry

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Name Constants
# =============================================================================

FEATURE_NAMES = [
    'berry_rate', 'qfi_logdet', 'multilag_infid',
    'inv_spectral_gap', 'log_condition',
    'multi_berry_max', 'multi_berry_mean', 'ricci_scalar', 'geodesic_dist',
]

VELOCITY_NAMES = [f'{f}_velocity' for f in FEATURE_NAMES]

ALL_FEATURE_NAMES = FEATURE_NAMES + VELOCITY_NAMES


# =============================================================================
# Online Geometric Feature Computer
# =============================================================================

class OnlineGeometricFeatureComputer:
    """Streaming version of GeometricFeatureExtractor.

    At each timestep t, uses only data up to t (strictly causal).
    Periodically refits scaler/PCA/geometry on expanding window.
    Returns 18 features: 9 base geometric + 9 velocity (rate of change).

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: Number of PCA components.
        operator_method: Operator fitting method.
        refit_interval: Days between refits (default: 21).
        min_history: Minimum history before first fit (default: 126).
        rolling_window: EMA effective window for smoothing (default: 10).
        lags: Fidelity lags.
        lag_weights: Fidelity lag weights.
        seed: Random seed.
    """

    FEATURE_NAMES = FEATURE_NAMES
    VELOCITY_NAMES = VELOCITY_NAMES
    ALL_FEATURE_NAMES = ALL_FEATURE_NAMES

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=15,
        operator_method='pca_inspired',
        refit_interval=21,
        min_history=126,
        rolling_window=10,
        max_history=500,
        lags=None,
        lag_weights=None,
        seed=42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.refit_interval = refit_interval
        self.min_history = min_history
        self.rolling_window = rolling_window
        self.max_history = max_history
        self.lags = lags or [1, 3, 5, 10]
        self.lag_weights = lag_weights or [0.4, 0.3, 0.2, 0.1]
        self.seed = seed

        self._scaler = None
        self._pca = None
        self._geometry = None
        self._history = []
        self._states = []  # coherent states for fidelity
        self._raw_features = {name: [] for name in FEATURE_NAMES}
        self._last_berry = None
        self._last_multi_berry = None  # for multi-plane berry rate
        self._centroid_buffer = []  # rolling buffer for geodesic centroid
        self._last_refit = 0
        self._t = 0
        self._ema = {}
        self._ema_history = {}

    def _fit_pipeline(self, data):
        """Refit scaler/PCA/geometry on accumulated data."""
        np.random.seed(self.seed)
        X = np.array(data)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)

        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)

        X_pca = self._pca.transform(X_scaled)
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        X_pca = X_pca / (norms + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)

    def _transform_point(self, x_raw):
        """Transform a single raw point through current pipeline."""
        x_scaled = self._scaler.transform(x_raw.reshape(1, -1))
        x_pca = self._pca.transform(x_scaled).ravel()
        norm = np.linalg.norm(x_pca)
        return x_pca / (norm + 1e-8)

    def update(self, x_enriched):
        """Process one timestep and return geometric feature dict.

        Args:
            x_enriched: Enriched feature vector (d,) for timestep t.

        Returns:
            Dict of 10 smoothed geometric features (5 base + 5 velocity),
            or None if insufficient history.
        """
        self._history.append(x_enriched.copy())
        self._t += 1

        # Cap history to prevent O(T^2) scaling during refit
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        # Check if we need to refit
        if self._t >= self.min_history and (
            self._geometry is None or
            self._t - self._last_refit >= self.refit_interval
        ):
            self._fit_pipeline(self._history)
            self._last_refit = self._t

        if self._geometry is None:
            self._states.append(None)
            for name in FEATURE_NAMES:
                self._raw_features[name].append(np.nan)
            return None

        # Compute geometric features for this point
        xt = self._transform_point(x_enriched)
        eig_tol = 1e-10

        # Berry curvature
        berry = self._geometry.berry_curvature_2d(xt, indices=(0, 1))
        if self._last_berry is not None:
            berry_rate = abs(berry - self._last_berry)
        else:
            berry_rate = 0.0
        self._last_berry = berry

        # Metric tensor
        g_ij = self._geometry.quantum_metric(xt)
        eigenvalues = np.linalg.eigvalsh(g_ij)
        nonzero_eigs = eigenvalues[eigenvalues > eig_tol]

        if len(nonzero_eigs) > 0:
            logdet = np.sum(np.log(nonzero_eigs))
            log_cond = np.log(nonzero_eigs[-1] / nonzero_eigs[0] + 1e-12)
        else:
            logdet = np.log(eig_tol) * len(eigenvalues)
            log_cond = 0.0

        # Spectral gap
        gap = self._geometry.spectral_gap(xt)
        inv_gap = 1.0 / (gap + 1e-8)

        # Coherent state for fidelity (keep only last max_lag+1 entries)
        psi = self._geometry.quasi_coherent_state(xt)
        self._states.append(psi)
        max_lag = max(self.lags)
        keep = max_lag + 2
        if len(self._states) > keep:
            self._states = self._states[-keep:]

        # Multi-lag infidelity
        infidelity = 0.0
        max_lag = max(self.lags)
        if len(self._states) > max_lag:
            for lag, w in zip(self.lags, self.lag_weights):
                past_state = self._states[-1 - lag]
                if past_state is not None:
                    overlap = np.abs(np.vdot(psi, past_state))
                    infidelity += w * (1.0 - overlap ** 2)

        # Multi-plane Berry curvature (top-10 planes by PCA variance order)
        n_dims = len(xt)
        pairs = [(i, j) for i in range(min(n_dims, 5)) for j in range(i + 1, min(n_dims, 5))]
        if pairs:
            curvatures = [abs(self._geometry.berry_curvature_2d(xt, indices=(i, j)))
                          for i, j in pairs]
            multi_berry_max = max(curvatures)
            multi_berry_mean = float(np.mean(curvatures))
        else:
            multi_berry_max = 0.0
            multi_berry_mean = 0.0

        # Multi-plane berry rate of change
        if self._last_multi_berry is not None:
            multi_berry_max_rate = abs(multi_berry_max - self._last_multi_berry)
        else:
            multi_berry_max_rate = 0.0
        self._last_multi_berry = multi_berry_max

        # Ricci scalar (only compute if n_dims <= 6 to keep latency manageable)
        if n_dims <= 6:
            try:
                ricci_val = abs(self._geometry.ricci_scalar(xt))
            except Exception:
                ricci_val = 0.0
        else:
            ricci_val = 0.0

        # Geodesic distance from rolling centroid
        self._centroid_buffer.append(xt.copy())
        if len(self._centroid_buffer) > 252:
            self._centroid_buffer = self._centroid_buffer[-252:]

        if len(self._centroid_buffer) > 20:
            centroid = np.mean(self._centroid_buffer[:-1], axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c > 1e-8:
                centroid = centroid / norm_c
            geodesic_dist = self._geometry.quantum_distance(xt, centroid)
        else:
            geodesic_dist = 0.0

        # Store raw features (trim to last 20 — only used for EMA seeding)
        self._raw_features['berry_rate'].append(berry_rate)
        self._raw_features['qfi_logdet'].append(logdet)
        self._raw_features['multilag_infid'].append(infidelity)
        self._raw_features['inv_spectral_gap'].append(inv_gap)
        self._raw_features['log_condition'].append(log_cond)
        self._raw_features['multi_berry_max'].append(multi_berry_max_rate)
        self._raw_features['multi_berry_mean'].append(multi_berry_mean)
        self._raw_features['ricci_scalar'].append(ricci_val)
        self._raw_features['geodesic_dist'].append(geodesic_dist)
        for name in FEATURE_NAMES:
            if len(self._raw_features[name]) > 20:
                self._raw_features[name] = self._raw_features[name][-20:]

        # EMA smoothing + velocity computation
        alpha = 2.0 / (self.rolling_window + 1)
        result = {}

        for name in FEATURE_NAMES:
            val = self._raw_features[name][-1]

            if np.isnan(val):
                result[name] = 0.0
                result[f'{name}_velocity'] = 0.0
                continue

            # EMA update
            if name not in self._ema:
                self._ema[name] = val
                self._ema_history[name] = [val]
            else:
                self._ema[name] = alpha * val + (1 - alpha) * self._ema[name]
                self._ema_history[name].append(self._ema[name])

            result[name] = self._ema[name]

            # Velocity: 5-step change in EMA
            buf = self._ema_history[name]
            if len(buf) > 5:
                result[f'{name}_velocity'] = self._ema[name] - buf[-6]
            else:
                result[f'{name}_velocity'] = 0.0

            # Trim buffer to last 10 entries
            if len(buf) > 10:
                self._ema_history[name] = buf[-10:]

        return result

    def reset(self):
        """Reset all internal state."""
        self._scaler = None
        self._pca = None
        self._geometry = None
        self._history = []
        self._states = []
        self._raw_features = {name: [] for name in FEATURE_NAMES}
        self._last_berry = None
        self._last_multi_berry = None
        self._centroid_buffer = []
        self._last_refit = 0
        self._t = 0
        self._ema = {}
        self._ema_history = {}


# =============================================================================
# Online Detector Base
# =============================================================================

class OnlineDetectorBase(ABC):
    """Base class for online regime detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def update(self, features: Dict[str, float]) -> float:
        """Process one timestep and return P(crisis) in [0, 1]."""
        ...

    @abstractmethod
    def reset(self):
        """Reset internal state for a fresh run."""
        ...


# =============================================================================
# Model 1: Expanding Percentile Detector (unsupervised baseline)
# =============================================================================

class ExpandingPercentileDetector(OnlineDetectorBase):
    """P(crisis) = RMS percentile rank of features against expanding history.

    Uses root-mean-square of per-feature anomaly scores instead of max,
    so a single noisy feature cannot dominate the signal.

    Args:
        min_history: Minimum observations before producing scores.
    """

    def __init__(self, min_history=60):
        self.min_history = min_history
        self._history = {name: [] for name in ALL_FEATURE_NAMES}
        self._t = 0

    @property
    def name(self) -> str:
        return "Online Percentile"

    def update(self, features):
        if features is None:
            return np.nan
        self._t += 1

        anomaly_scores = []
        for fname in ALL_FEATURE_NAMES:
            val = features.get(fname, np.nan)
            self._history[fname].append(val)

            if self._t < self.min_history or np.isnan(val):
                continue

            hist = np.array(self._history[fname])
            hist = hist[~np.isnan(hist)]
            if len(hist) < 10:
                continue

            # Direction-agnostic: anomaly = deviation from median in either direction
            pct = np.mean(hist <= val)
            anomaly_scores.append(2 * abs(pct - 0.5))  # 0=median, 1=extreme

        if not anomaly_scores:
            return np.nan

        # RMS aggregation: penalizes broad-based anomalies, resists single-feature noise
        return float(np.sqrt(np.mean(np.array(anomaly_scores) ** 2)))

    def reset(self):
        self._history = {name: [] for name in ALL_FEATURE_NAMES}
        self._t = 0


# =============================================================================
# Welford Accumulators
# =============================================================================

class _WelfordAccumulator:
    """Online mean/variance via Welford's algorithm (scalar)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0

    @property
    def variance(self):
        if self.n < 2:
            return 1.0
        return self._M2 / (self.n - 1)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2


class _MultivariateWelfordAccumulator:
    """Online mean vector / covariance matrix via Welford's algorithm.

    Args:
        d: Dimensionality of the feature vector.
    """

    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = np.zeros(d)
        self._M2 = np.zeros((d, d))

    @property
    def covariance(self):
        if self.n < 2:
            return np.eye(self.d)
        return self._M2 / (self.n - 1)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self._M2 += np.outer(delta, delta2)


# =============================================================================
# Model 2: Online Bayesian Detector (Multivariate)
# =============================================================================

class OnlineBayesianDetector(OnlineDetectorBase):
    """Multivariate Bayesian filtering with sticky regime transitions.

    Maintains running statistics (multivariate Welford accumulators) for calm
    and crisis distributions. Uses Bayes rule with transition persistence and
    multivariate Gaussian likelihoods to produce P(crisis).

    Key improvement over scalar version: aggregates all feature z-scores into
    a d-dimensional vector instead of taking max(), which caused 5 independent
    chances per timestep to produce false alarms.

    Args:
        transition_prob: Base probability of switching regime per day.
        persistence: Factor making same-regime transitions more likely.
        min_history: Minimum observations before scoring.
        crisis_quantile: Kept for API compatibility (unused).
        regularization: Ridge regularization for covariance matrices.
    """

    def __init__(
        self,
        transition_prob=0.02,
        persistence=0.80,
        min_history=126,
        crisis_quantile=0.90,
        regularization=1e-4,
        forgetting_factor=0.995,
    ):
        self.transition_prob = transition_prob
        self.persistence = persistence
        self.min_history = min_history
        self.crisis_quantile = crisis_quantile
        self.regularization = regularization
        self.forgetting_factor = forgetting_factor

        self._p_crisis = 0.1  # prior P(crisis)
        self._calm_acc = None  # _MultivariateWelfordAccumulator, lazy init
        self._crisis_acc = None
        self._all_features = []
        self._t = 0
        self._d = None

    @property
    def name(self) -> str:
        return "Online Bayesian"

    def update(self, features):
        if features is None:
            return np.nan
        self._t += 1

        # Aggregate features into d-dimensional z-score vector
        signal = self._aggregate(features)
        if self._calm_acc is None:
            self._all_features.append(signal)

        if self._t < self.min_history:
            return np.nan

        # Initialize distributions on first scoring step
        if self._calm_acc is None:
            self._init_distributions()
            self._all_features = []  # free memory; no longer needed

        # Multivariate log-likelihoods under each regime
        ll_calm = self._multivariate_log_likelihood(signal, self._calm_acc)
        ll_crisis = self._multivariate_log_likelihood(signal, self._crisis_acc)

        # Transition matrix (sticky)
        p_stay = self.persistence
        p_switch = 1.0 - p_stay

        # Prior from previous step
        prior_crisis = self._p_crisis * p_stay + (1 - self._p_crisis) * p_switch
        prior_calm = 1.0 - prior_crisis

        # Bayes rule in log space for numerical stability
        log_num = ll_crisis + np.log(max(prior_crisis, 1e-300))
        log_den_calm = ll_calm + np.log(max(prior_calm, 1e-300))

        max_log = max(log_num, log_den_calm)
        log_den = max_log + np.log(
            np.exp(log_num - max_log) + np.exp(log_den_calm - max_log)
        )

        self._p_crisis = float(np.exp(log_num - log_den))

        # Clamp
        self._p_crisis = np.clip(self._p_crisis, 0.001, 0.999)

        # Update accumulators based on posterior
        if self._p_crisis > 0.5:
            self._crisis_acc.update(signal)
        else:
            self._calm_acc.update(signal)

        # Exponential forgetting: scale down effective sample sizes periodically
        # This makes recent observations more influential than old ones
        if self._t % 21 == 0 and self.forgetting_factor < 1.0:
            for acc in [self._calm_acc, self._crisis_acc]:
                if acc is not None and acc.n > 50:
                    acc.n = max(int(acc.n * self.forgetting_factor), 20)
                    acc._M2 *= self.forgetting_factor

        return float(self._p_crisis)

    def _aggregate(self, features):
        """Return full d-dimensional z-score vector.

        Computes per-feature absolute z-scores against expanding statistics,
        returning a vector that captures anomaly magnitude across all features
        simultaneously instead of collapsing to max (which inflates false alarms).
        """
        z_scores = []
        for fname in ALL_FEATURE_NAMES:
            v = features.get(fname, np.nan)
            if np.isnan(v):
                z_scores.append(0.0)
                continue

            key = f'_acc_{fname}'
            if not hasattr(self, key):
                setattr(self, key, _WelfordAccumulator())
            acc = getattr(self, key)

            if acc.n >= 10:
                sigma = max(np.sqrt(acc.variance), 1e-8)
                z = abs((v - acc.mean) / sigma)
                z_scores.append(z)
            else:
                z_scores.append(0.0)
            acc.update(v)

        return np.array(z_scores)

    def _init_distributions(self):
        """Initialize calm/crisis distributions using percentile-based splitting.

        Top 20% by L2 norm → crisis distribution, bottom 80% → calm.
        More robust than k-means when crisis periods are rare.
        """
        X = np.array(self._all_features)
        d = X.shape[1]
        self._d = d

        norms = np.linalg.norm(X, axis=1)
        threshold = np.percentile(norms, 80)

        self._calm_acc = _MultivariateWelfordAccumulator(d)
        self._crisis_acc = _MultivariateWelfordAccumulator(d)

        for i, x in enumerate(X):
            if norms[i] >= threshold:
                self._crisis_acc.update(x)
            else:
                self._calm_acc.update(x)

    def _multivariate_log_likelihood(self, x, acc):
        """Multivariate Gaussian log-likelihood with regularization.

        Args:
            x: Feature vector (d,).
            acc: _MultivariateWelfordAccumulator.

        Returns:
            Log-likelihood (float).
        """
        if acc.n < acc.d + 2:
            return -100.0  # uninformative prior

        mu = acc.mean
        cov = acc.covariance + self.regularization * np.eye(acc.d)

        diff = x - mu
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -100.0

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return -100.0

        mahal = float(diff @ cov_inv @ diff)
        log_lik = -0.5 * (acc.d * np.log(2 * np.pi) + logdet + mahal)
        return float(log_lik)

    def reset(self):
        self._p_crisis = 0.1
        self._calm_acc = None
        self._crisis_acc = None
        self._all_features = []
        self._t = 0
        self._d = None
        # Clear per-feature accumulators
        for fname in ALL_FEATURE_NAMES:
            key = f'_acc_{fname}'
            if hasattr(self, key):
                delattr(self, key)


# =============================================================================
# Model 3: Online HMM Detector
# =============================================================================

class OnlineHMMDetector(OnlineDetectorBase):
    """Periodic HMM refit + forward algorithm for causal inference.

    Refits a 2-state Gaussian HMM periodically on expanding history, then
    uses the forward algorithm to produce P(crisis) causally.

    Args:
        refit_interval: Days between HMM refits.
        min_history: Minimum observations before first fit.
        n_iter: EM iterations for HMM fitting.
    """

    def __init__(self, refit_interval=10, min_history=126, n_iter=100, seed=42,
                 vol_trigger_sigma=2.0):
        self.refit_interval = refit_interval
        self.min_history = min_history
        self.n_iter = n_iter
        self.seed = seed
        self.vol_trigger_sigma = vol_trigger_sigma

        self._model = None
        self._high_vol_state = None
        self._history = []
        self._t = 0
        self._last_refit = 0
        self._vol_acc = _WelfordAccumulator()  # track feature-norm volatility

    @property
    def name(self) -> str:
        return "Online HMM"

    def update(self, features):
        if features is None:
            return np.nan
        self._t += 1

        # Build feature vector from all 10 features
        fvec = np.array([
            features.get(fname, 0.0)
            for fname in ALL_FEATURE_NAMES
        ])
        self._history.append(fvec)

        # Cap history to prevent O(T^2) scaling
        if len(self._history) > 500:
            self._history = self._history[-500:]

        # Volatility-triggered refit: if recent feature norm > 2σ, force immediate refit
        fvec_norm = np.linalg.norm(fvec)
        vol_triggered = False
        if self._vol_acc.n >= 20:
            vol_sigma = max(np.sqrt(self._vol_acc.variance), 1e-8)
            if abs(fvec_norm - self._vol_acc.mean) > self.vol_trigger_sigma * vol_sigma:
                vol_triggered = True
        self._vol_acc.update(fvec_norm)

        # Check if we need to refit
        if self._t >= self.min_history and (
            self._model is None or
            self._t - self._last_refit >= self.refit_interval or
            vol_triggered
        ):
            self._fit_hmm()
            self._last_refit = self._t

        if self._model is None:
            return np.nan

        # Forward algorithm on recent window (causal)
        recent = np.array(self._history[-min(len(self._history), 500):])
        # Filter NaN rows for predict_proba
        valid_mask = ~np.any(np.isnan(recent), axis=1)
        recent_clean = recent[valid_mask]
        if len(recent_clean) < 2:
            return np.nan
        try:
            posteriors = self._model.predict_proba(recent_clean)
            return float(posteriors[-1, self._high_vol_state])
        except Exception:
            return np.nan

    def _fit_hmm(self):
        """Refit HMM on accumulated history."""
        from hmmlearn.hmm import GaussianHMM

        X = np.array(self._history)
        # Filter rows with any NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        if len(X) < 30:
            logger.warning("HMM fit skipped: insufficient valid rows after NaN filter")
            return
        try:
            model = GaussianHMM(
                n_components=2,
                covariance_type='full',
                n_iter=self.n_iter,
                random_state=self.seed,
            )
            model.fit(X)
            means_abs = np.abs(model.means_).sum(axis=1)
            self._high_vol_state = int(np.argmax(means_abs))
            self._model = model
        except Exception as e:
            logger.warning(f"HMM fit failed: {e}")

    def reset(self):
        self._model = None
        self._high_vol_state = None
        self._history = []
        self._t = 0
        self._last_refit = 0
        self._vol_acc = _WelfordAccumulator()


# =============================================================================
# Model 4: Online Logistic Detector (supervised)
# =============================================================================

class OnlineLogisticDetector(OnlineDetectorBase):
    """Expanding-window logistic regression on geometric features.

    Requires labels (crisis periods). Refits periodically on expanding window.

    Args:
        refit_interval: Days between refits.
        min_history: Minimum labeled observations before first fit.
    """

    def __init__(self, refit_interval=21, min_history=126):
        self.refit_interval = refit_interval
        self.min_history = min_history

        self._model = None
        self._X_history = []
        self._y_history = []
        self._t = 0
        self._last_refit = 0

    @property
    def name(self) -> str:
        return "Online Logistic"

    def add_label(self, y):
        """Add a binary label for the current timestep."""
        self._y_history.append(float(y))

    def update(self, features):
        if features is None:
            return np.nan
        self._t += 1

        fvec = np.array([
            features.get(fname, 0.0)
            for fname in ALL_FEATURE_NAMES
        ])
        self._X_history.append(fvec)

        # Cap history to prevent O(T^2) scaling
        if len(self._X_history) > 500:
            self._X_history = self._X_history[-500:]
        if len(self._y_history) > 500:
            self._y_history = self._y_history[-500:]

        # Check if we need to refit
        if (
            self._t >= self.min_history and
            len(self._y_history) >= self.min_history and
            (self._model is None or self._t - self._last_refit >= self.refit_interval)
        ):
            self._fit_logistic()
            self._last_refit = self._t

        if self._model is None:
            return np.nan

        try:
            proba = self._model.predict_proba(fvec.reshape(1, -1))
            return float(proba[0, 1])
        except Exception:
            return np.nan

    def _fit_logistic(self):
        """Refit logistic regression on accumulated labeled data."""
        from sklearn.linear_model import LogisticRegression

        n = min(len(self._X_history), len(self._y_history))
        X = np.array(self._X_history[:n])
        y = np.array(self._y_history[:n])

        # Filter rows with any NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Need both classes present
        if len(X) < 10 or len(np.unique(y)) < 2:
            return

        try:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
            self._model = model
        except Exception as e:
            logger.warning(f"Logistic fit failed: {e}")

    def reset(self):
        self._model = None
        self._X_history = []
        self._y_history = []
        self._t = 0
        self._last_refit = 0


# =============================================================================
# Model 5: Online Ensemble Detector
# =============================================================================

class OnlineEnsembleDetector(OnlineDetectorBase):
    """Weighted average of multiple online detectors.

    Produces output only when >= 50% of component detectors return non-NaN.

    Args:
        detectors: List of OnlineDetectorBase instances.
        weights: Weight per detector (default: equal weights).
    """

    def __init__(self, detectors, weights=None):
        self._detectors = detectors
        self._weights = weights or [1.0 / len(detectors)] * len(detectors)
        self._last_individual = {}  # {detector_name: last P(crisis)}

    @property
    def name(self) -> str:
        return "Online Ensemble"

    @property
    def last_individual(self) -> Dict[str, float]:
        """Last individual P(crisis) from each sub-detector."""
        return self._last_individual

    def update(self, features):
        self._last_individual = {}
        scores = []
        ws = []
        for det, w in zip(self._detectors, self._weights):
            p = det.update(features)
            self._last_individual[det.name] = p
            if not np.isnan(p):
                scores.append(p)
                ws.append(w)

        # Need >= 50% non-NaN
        if len(scores) < len(self._detectors) / 2:
            return np.nan

        ws = np.array(ws)
        ws = ws / ws.sum()
        return float(np.dot(scores, ws))

    def reset(self):
        for det in self._detectors:
            det.reset()


# =============================================================================
# Model 6: Stacking Meta-Learner Ensemble
# =============================================================================

class OnlineStackingEnsemble(OnlineDetectorBase):
    """Stacking ensemble: logistic meta-learner on component detector outputs.

    Instead of fixed-weight averaging, learns which detectors to trust via
    expanding-window logistic regression. Falls back to best single detector
    until enough labeled history is available.

    Args:
        detectors: List of OnlineDetectorBase instances.
        min_meta_history: Minimum labeled observations before meta-learner activates.
        refit_interval: Days between meta-learner refits.
    """

    def __init__(self, detectors, min_meta_history=252, refit_interval=21):
        self._detectors = detectors
        self.min_meta_history = min_meta_history
        self.refit_interval = refit_interval

        self._meta_model = None
        self._X_meta = []  # component detector outputs
        self._y_meta = []  # labels
        self._t = 0
        self._last_refit = 0
        self._last_individual = {}

    @property
    def name(self) -> str:
        return "Online Stacking"

    @property
    def last_individual(self) -> Dict[str, float]:
        return self._last_individual

    def add_label(self, y):
        """Add a binary label for the current timestep."""
        self._y_meta.append(float(y))

    def update(self, features):
        self._last_individual = {}
        self._t += 1

        # Collect component outputs
        component_scores = []
        for det in self._detectors:
            p = det.update(features)
            self._last_individual[det.name] = p
            component_scores.append(p if not np.isnan(p) else 0.5)

        meta_input = np.array(component_scores)
        self._X_meta.append(meta_input)

        # Cap history
        if len(self._X_meta) > 1000:
            self._X_meta = self._X_meta[-1000:]
        if len(self._y_meta) > 1000:
            self._y_meta = self._y_meta[-1000:]

        # Try to refit meta-learner
        if (
            len(self._y_meta) >= self.min_meta_history and
            (self._meta_model is None or self._t - self._last_refit >= self.refit_interval)
        ):
            self._fit_meta()
            self._last_refit = self._t

        # If meta-learner is ready, use it
        if self._meta_model is not None:
            try:
                proba = self._meta_model.predict_proba(meta_input.reshape(1, -1))
                return float(proba[0, 1])
            except Exception:
                pass

        # Fallback: use best single detector (highest non-NaN score)
        valid = [(p, name) for name, p in self._last_individual.items() if not np.isnan(p)]
        if valid:
            return max(valid, key=lambda x: x[0])[0]
        return np.nan

    def _fit_meta(self):
        """Refit logistic meta-learner on stacked component outputs."""
        from sklearn.linear_model import LogisticRegression

        n = min(len(self._X_meta), len(self._y_meta))
        X = np.array(self._X_meta[:n])
        y = np.array(self._y_meta[:n])

        if len(np.unique(y)) < 2 or len(X) < 20:
            return

        try:
            model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            model.fit(X, y)
            self._meta_model = model
        except Exception as e:
            logger.warning(f"Stacking meta-learner fit failed: {e}")

    def reset(self):
        self._meta_model = None
        self._X_meta = []
        self._y_meta = []
        self._t = 0
        self._last_refit = 0
        self._last_individual = {}
        for det in self._detectors:
            det.reset()
