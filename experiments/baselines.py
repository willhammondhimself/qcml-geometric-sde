"""
Classical baseline regime detectors for comparison with QCML observables.

All detectors implement the BaseRegimeDetector interface from
qcml_geometry.observables:
    - fit(X) -> self
    - compute_regime_scores(X) -> 1-D array of length T
    - name (property) -> str

Detectors:
    RollingVolatilityDetector    — 20-day rolling vol z-score
    CUSUMDetector                — Cumulative sum changepoint
    HMMRegimeDetector            — 2-state Gaussian HMM
    BOCPDDetector                — Bayesian Online Changepoint Detection
    IsolationForestDetector      — Isolation Forest anomaly scores
    RandomForestRegimeDetector   — Supervised RF with fit_with_labels()
    RollingWindowRFDetector      — RF trained on rolling VIX > threshold labels
    VIXThresholdDetector         — Expanding z-score of raw VIX close
    GARCHDetector                — GARCH(1,1) conditional volatility z-score
    HamiltonMSDetector           — Hamilton (1989) Markov-switching model
    EWMADetector                 — EWMA (RiskMetrics) volatility z-score
    MahalanobisDetector          — Mahalanobis distance anomaly score
    StructuralBreakDetector      — Bai-Perron / PELT structural break proximity
    TransferEntropyDetector      — Rolling transfer entropy between assets
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from qcml_geometry.observables import BaseRegimeDetector

logger = logging.getLogger(__name__)


class RollingVolatilityDetector(BaseRegimeDetector):
    """Rolling volatility z-score detector.

    Score = |z-score| of rolling standard deviation of first feature.
    """

    def __init__(self, vol_window: int = 20, min_expanding: int = 60):
        self.vol_window = vol_window
        self.min_expanding = min_expanding

    @property
    def name(self) -> str:
        return "Rolling Vol Z"

    def fit(self, X: np.ndarray, **kwargs) -> 'RollingVolatilityDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = X.shape[0]

        vol = pd.Series(X[:, 0]).rolling(self.vol_window, min_periods=1).std().values

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.nanmean(vol[:t])
            sigma = np.nanstd(vol[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((vol[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0
        return z_scores


class CUSUMDetector(BaseRegimeDetector):
    """Cumulative sum changepoint detector.

    Two-sided CUSUM on first feature: accumulates when |x - mu| > k*sigma.
    Score is the maximum of upper and lower CUSUM statistics.
    """

    def __init__(self, k: float = 0.5, burn_in: int = 60):
        self.k = k
        self.burn_in = burn_in
        self._mu = None
        self._sigma = None

    @property
    def name(self) -> str:
        return "CUSUM"

    def fit(self, X: np.ndarray, **kwargs) -> 'CUSUMDetector':
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        burn = min(self.burn_in, X.shape[0])
        self._mu = np.mean(X[:burn, 0])
        self._sigma = max(np.std(X[:burn, 0], ddof=1), 1e-12)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None:
            raise RuntimeError("Call fit() first")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T = X.shape[0]
        scores = np.zeros(T)
        s_high = 0.0
        s_low = 0.0

        for t in range(T):
            z = (X[t, 0] - self._mu) / self._sigma
            s_high = max(0.0, s_high + z - self.k)
            s_low = max(0.0, s_low - z - self.k)
            scores[t] = max(s_high, s_low)

        return scores


class HMMRegimeDetector(BaseRegimeDetector):
    """2-state Gaussian HMM regime detector.

    Score = P(high-volatility state | data). The high-vol state is identified
    as the one with higher mean absolute value of features.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100,
                 covariance_type: str = 'full', seed: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.seed = seed
        self._model = None
        self._high_vol_state = None

    @property
    def name(self) -> str:
        return "HMM"

    def fit(self, X: np.ndarray, **kwargs) -> 'HMMRegimeDetector':
        from hmmlearn.hmm import GaussianHMM

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self._model.fit(X)

        # Identify high-vol state as the one with larger mean absolute value
        means_abs = np.abs(self._model.means_).sum(axis=1)
        self._high_vol_state = int(np.argmax(means_abs))

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        posteriors = self._model.predict_proba(X)
        return posteriors[:, self._high_vol_state]


class BOCPDDetector(BaseRegimeDetector):
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Score = 1 - P(run_length > current_run), i.e. probability of
    recent changepoint.
    """

    def __init__(self, hazard_rate: float = 250.0, min_expanding: int = 30):
        self.hazard_rate = hazard_rate
        self.min_expanding = min_expanding

    @property
    def name(self) -> str:
        return "BOCPD"

    def fit(self, X: np.ndarray, **kwargs) -> 'BOCPDDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T = X.shape[0]
        x = X[:, 0]

        # Online Bayesian changepoint detection
        # R[t, r] = P(run_length = r at time t)
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0

        scores = np.full(T, np.nan)
        h = 1.0 / self.hazard_rate  # hazard function (constant)

        # Sufficient statistics for online mean/variance
        mu_0, kappa_0, alpha_0, beta_0 = 0.0, 1.0, 1.0, 1.0

        for t in range(T):
            # Predictive probability under each run length
            predprobs = np.zeros(t + 1)
            for r in range(t + 1):
                if r == 0:
                    # Prior predictive
                    predprobs[r] = _student_t_pdf(x[t], mu_0, beta_0 * (kappa_0 + 1) / (kappa_0 * alpha_0), 2.0 * alpha_0)
                else:
                    # Use running sufficient stats
                    start = t - r
                    seg = x[start:t + 1]
                    n = len(seg)
                    mean_seg = np.mean(seg)
                    var_seg = np.var(seg, ddof=0) if n > 1 else 0.0

                    kappa_n = kappa_0 + n
                    alpha_n = alpha_0 + n / 2.0
                    mu_n = (kappa_0 * mu_0 + n * mean_seg) / kappa_n
                    beta_n = beta_0 + 0.5 * n * var_seg + 0.5 * kappa_0 * n * (mean_seg - mu_0) ** 2 / kappa_n

                    predprobs[r] = _student_t_pdf(
                        x[t], mu_n, beta_n * (kappa_n + 1) / (kappa_n * alpha_n), 2.0 * alpha_n
                    )

            # Growth probabilities
            R[t + 1, 1:t + 2] = R[t, :t + 1] * predprobs * (1 - h)
            # Changepoint probability
            R[t + 1, 0] = np.sum(R[t, :t + 1] * predprobs * h)

            # Normalize
            evidence = np.sum(R[t + 1, :t + 2])
            if evidence > 0:
                R[t + 1, :t + 2] /= evidence

            # Score = P(changepoint in recent window)
            scores[t] = R[t + 1, 0] + np.sum(R[t + 1, 1:min(4, t + 2)])

        return scores


def _student_t_pdf(x, mu, scale, df):
    """Student-t PDF for BOCPD predictive distribution."""
    from scipy.special import gammaln

    scale = max(scale, 1e-12)
    z = (x - mu) / np.sqrt(scale)
    log_pdf = (
        gammaln((df + 1) / 2) - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi * scale)
        - (df + 1) / 2 * np.log(1 + z ** 2 / df)
    )
    return np.exp(log_pdf)


class IsolationForestDetector(BaseRegimeDetector):
    """Isolation Forest anomaly detector.

    Score = anomaly score from sklearn IsolationForest, rescaled to [0, 1].
    """

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05,
                 seed: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self._model = None

    @property
    def name(self) -> str:
        return "Isolation Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'IsolationForestDetector':
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self._model.fit(X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # score_samples returns negative anomaly scores (more negative = more anomalous)
        raw_scores = -self._model.score_samples(X)
        # Rescale to [0, 1]
        mn, mx = raw_scores.min(), raw_scores.max()
        if mx - mn > 1e-12:
            return (raw_scores - mn) / (mx - mn)
        return np.zeros(len(X))


class RandomForestRegimeDetector(BaseRegimeDetector):
    """Supervised Random Forest baseline.

    Requires fit_with_labels(X, y) where y is binary crisis labels.
    fit(X) is a no-op; compute_regime_scores() raises if no labels provided.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 seed: int = 42, lookback: int = 20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.lookback = lookback
        self._model = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'RandomForestRegimeDetector':
        # No-op: supervised baseline needs labels via fit_with_labels
        return self

    def fit_with_labels(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegimeDetector':
        """Fit supervised RF with binary crisis labels.

        Args:
            X: Feature matrix (T, d).
            y: Binary labels (T,) where 1 = crisis.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=self.lookback)
        y_trimmed = y[self.lookback - 1:]

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_enriched, y_trimmed)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Must call fit_with_labels(X, y) before compute_regime_scores()")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=self.lookback)

        # Handle single-class training (e.g., no crisis labels in causal window)
        if len(self._model.classes_) < 2:
            proba = np.zeros(len(X_enriched))
        else:
            proba = self._model.predict_proba(X_enriched)[:, 1]

        # Pad front with NaN to match original length
        pad = np.full(self.lookback - 1, np.nan)
        return np.concatenate([pad, proba])


class RollingWindowRFDetector(BaseRegimeDetector):
    """Random Forest trained on a rolling window with VIX > threshold labels.

    Unlike RandomForestRegimeDetector (leave-one-crisis-out with hand-labeled
    crises), this detector uses continuous VIX-based labels: any day where
    VIX > vix_threshold is labeled as crisis. Training uses a trailing window
    of `train_window` days before the evaluation point.

    VIX is used ONLY for labels, not as a feature.

    Args:
        n_estimators: Number of trees.
        max_depth: Max tree depth.
        seed: Random seed.
        lookback: Feature enrichment window (must match pipeline).
        train_window: Number of trailing days for training.
        vix_threshold: VIX level above which a day is labeled crisis.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 seed: int = 42, lookback: int = 20,
                 train_window: int = 250, vix_threshold: float = 25.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.lookback = lookback
        self.train_window = train_window
        self.vix_threshold = vix_threshold
        self._model = None

    @property
    def name(self) -> str:
        return "Rolling RF (VIX)"

    def fit(self, X: np.ndarray, **kwargs) -> 'RollingWindowRFDetector':
        return self

    def fit_rolling(self, X_train: np.ndarray, vix_train: np.ndarray) -> 'RollingWindowRFDetector':
        """Fit RF on trailing window with VIX-based labels.

        Args:
            X_train: Enriched feature matrix for the training window.
            vix_train: VIX close values aligned to X_train rows.
        """
        if len(X_train) != len(vix_train):
            raise ValueError(
                f"X_train ({len(X_train)}) and vix_train ({len(vix_train)}) length mismatch"
            )

        # Create binary labels from VIX threshold
        valid_mask = ~np.isnan(vix_train)
        if np.sum(valid_mask) < 10:
            logger.warning("Rolling RF: fewer than 10 valid VIX values in training window")
            return self

        y = (vix_train[valid_mask] > self.vix_threshold).astype(float)
        X_valid = X_train[valid_mask]

        # Handle single-class edge case
        if len(np.unique(y)) < 2:
            logger.warning(
                f"Rolling RF: single class in training window "
                f"(all {'crisis' if y[0] == 1 else 'normal'}), model will predict constant"
            )

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_valid, y)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Return P(crisis) from the trained RF.

        Args:
            X: Enriched feature matrix (T, d). Must already be enriched.

        Returns:
            1-D array of length T with P(crisis) scores.
        """
        if self._model is None:
            raise RuntimeError("Must call fit_rolling(X_train, vix_train) before compute_regime_scores()")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(self._model.classes_) < 2:
            return np.zeros(len(X))

        return self._model.predict_proba(X)[:, 1]


class VIXThresholdDetector(BaseRegimeDetector):
    """Expanding z-score of raw VIX close.

    Oracle-like upper bound since VIX directly measures implied volatility.
    Score = expanding z-score of VIX, floored at 0 (only elevated VIX
    contributes to regime score).

    The feature matrix X is ignored; scores come from stored VIX values.

    Args:
        min_expanding: Minimum history before computing z-score.
    """

    def __init__(self, min_expanding: int = 60):
        self.min_expanding = min_expanding
        self._vix = None

    @property
    def name(self) -> str:
        return "VIX Level"

    def set_vix(self, vix_aligned: np.ndarray) -> 'VIXThresholdDetector':
        """Store VIX values aligned to evaluation dates.

        Args:
            vix_aligned: 1-D array of VIX close values, same length as X
                         that will be passed to compute_regime_scores().
        """
        self._vix = np.asarray(vix_aligned, dtype=float)
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'VIXThresholdDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute expanding z-score of stored VIX values.

        Args:
            X: Feature matrix (ignored, scores come from VIX).

        Returns:
            1-D array of length T with z-scores (NaN before min_expanding).
        """
        if self._vix is None:
            raise RuntimeError("Must call set_vix(vix_aligned) before compute_regime_scores()")

        T = len(self._vix)
        scores = np.full(T, np.nan)

        for t in range(self.min_expanding, T):
            past = self._vix[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 1e-12:
                scores[t] = max(0.0, (self._vix[t] - mu) / sigma)
            else:
                scores[t] = 0.0

        return scores


class GARCHDetector(BaseRegimeDetector):
    """GARCH(1,1) conditional volatility z-score (Bollerslev 1986).

    Fits a GARCH(1,1) model to portfolio returns and uses the expanding
    z-score of the conditional variance as a regime score.  High
    conditional volatility relative to history signals a stress regime.

    Args:
        min_expanding: Minimum history before computing z-score.
        return_col: Column index for log returns (default: 0, first PC).
    """

    def __init__(self, min_expanding: int = 60, return_col: int = 0):
        self.min_expanding = min_expanding
        self.return_col = return_col
        self._omega = None
        self._alpha = None
        self._beta = None

    @property
    def name(self) -> str:
        return "GARCH(1,1)"

    def fit(self, X: np.ndarray, **kwargs) -> 'GARCHDetector':
        """Fit GARCH(1,1) to returns using the arch package.

        Args:
            X: Feature matrix of shape (T, d).  Column ``return_col``
               is treated as the return series.
        """
        try:
            from arch import arch_model
        except ImportError:
            logger.warning("arch package not installed; using fallback EWMA.")
            self._omega, self._alpha, self._beta = None, None, None
            return self

        returns = np.asarray(X[:, self.return_col], dtype=float) * 100
        valid = returns[~np.isnan(returns)]
        if len(valid) < 50:
            self._omega, self._alpha, self._beta = None, None, None
            return self

        am = arch_model(valid, vol='Garch', p=1, q=1, dist='normal',
                        rescale=False)
        res = am.fit(disp='off', show_warning=False)
        self._omega = res.params.get('omega', 0.01)
        self._alpha = res.params.get('alpha[1]', 0.05)
        self._beta = res.params.get('beta[1]', 0.90)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute expanding z-score of GARCH conditional variance.

        Returns:
            1-D array of length T with z-scores (NaN before min_expanding).
        """
        returns = np.asarray(X[:, self.return_col], dtype=float)
        T = len(returns)
        cond_var = np.full(T, np.nan)

        if self._omega is not None:
            omega, alpha, beta = self._omega, self._alpha, self._beta
        else:
            omega, alpha, beta = 0.0, 0.06, 0.94

        var_init = np.nanvar(returns[:max(20, self.min_expanding)])
        sigma2 = var_init
        for t in range(T):
            if t == 0:
                sigma2 = var_init
            else:
                r = returns[t - 1] if np.isfinite(returns[t - 1]) else 0.0
                sigma2 = omega + alpha * r**2 + beta * sigma2
            cond_var[t] = sigma2

        scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = cond_var[:t]
            valid = past[np.isfinite(past)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 1e-15:
                scores[t] = max(0.0, (cond_var[t] - mu) / sigma)
            else:
                scores[t] = 0.0

        return scores


class HamiltonMSDetector(BaseRegimeDetector):
    """Hamilton (1989) Markov-switching autoregression.

    Fits a 2-regime Markov-switching model to the first principal
    component of returns.  The smoothed probability of the high-variance
    regime is the regime score.

    Args:
        k_regimes: Number of regimes (default: 2).
        order: Autoregressive order (default: 1).
        min_history: Minimum observations before fitting.
    """

    def __init__(self, k_regimes: int = 2, order: int = 1,
                 min_history: int = 100):
        self.k_regimes = k_regimes
        self.order = order
        self.min_history = min_history
        self._params = None
        self._high_regime_idx = None

    @property
    def name(self) -> str:
        return "Hamilton MS"

    def fit(self, X: np.ndarray, **kwargs) -> 'HamiltonMSDetector':
        """Fit Markov-switching AR model via statsmodels.

        Args:
            X: Feature matrix of shape (T, d). First column is used.
        """
        try:
            from statsmodels.tsa.regime_switching.markov_autoregression import (
                MarkovAutoregression,
            )
        except ImportError:
            logger.warning("statsmodels MarkovAutoregression not available.")
            self._params = None
            return self

        series = np.asarray(X[:, 0], dtype=float)
        valid = series[np.isfinite(series)]
        if len(valid) < self.min_history:
            self._params = None
            return self

        try:
            mod = MarkovAutoregression(
                valid,
                k_regimes=self.k_regimes,
                order=self.order,
                switching_variance=True,
            )
            res = mod.fit(maxiter=200, disp=False)
            self._params = res.params
            param_names = mod.param_names
            sigma_indices = [i for i, n in enumerate(param_names)
                             if 'sigma2' in n]
            variances = [res.params[i] for i in sigma_indices]
            self._high_regime_idx = int(np.argmax(variances))
        except Exception as e:
            logger.warning("Hamilton MS fit failed: %s. Using fallback.", e)
            self._params = None

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute smoothed probability of high-variance regime.

        Returns:
            1-D array of length T with P(high-vol regime) or NaN if unfitted.
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        series = np.asarray(X[:, 0], dtype=float)
        T = len(series)

        if self._params is None:
            return np.full(T, np.nan)

        try:
            mod = MarkovAutoregression(
                series,
                k_regimes=self.k_regimes,
                order=self.order,
                switching_variance=True,
            )
            res = mod.smooth(self._params)
            probs = res.smoothed_marginal_probabilities[:, self._high_regime_idx]
            scores = np.asarray(probs, dtype=float)
            if len(scores) < T:
                scores = np.concatenate([np.full(T - len(scores), np.nan),
                                         scores])
            return scores[:T]
        except Exception as e:
            logger.warning("Hamilton MS scoring failed: %s", e)
            return np.full(T, np.nan)


class EWMADetector(BaseRegimeDetector):
    """EWMA (RiskMetrics) volatility z-score (J.P. Morgan 1996).

    Exponentially weighted moving average of squared returns with
    decay factor lambda.  Score = expanding z-score of EWMA variance.

    Args:
        decay: EWMA decay factor (lambda). Default: 0.94 (RiskMetrics daily).
        min_expanding: Minimum history before computing z-score.
        return_col: Column index for returns (default: 0).
    """

    def __init__(self, decay: float = 0.94, min_expanding: int = 60,
                 return_col: int = 0):
        self.decay = decay
        self.min_expanding = min_expanding
        self.return_col = return_col

    @property
    def name(self) -> str:
        return "EWMA Vol"

    def fit(self, X: np.ndarray, **kwargs) -> 'EWMADetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        returns = X[:, self.return_col]
        T = len(returns)

        ewma_var = np.full(T, np.nan)
        sigma2 = np.nanvar(returns[:max(20, self.min_expanding)])
        for t in range(T):
            r = returns[t] if np.isfinite(returns[t]) else 0.0
            sigma2 = self.decay * sigma2 + (1 - self.decay) * r ** 2
            ewma_var[t] = sigma2

        scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = ewma_var[:t]
            valid = past[np.isfinite(past)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 1e-15:
                scores[t] = max(0.0, (ewma_var[t] - mu) / sigma)
            else:
                scores[t] = 0.0
        return scores


class MahalanobisDetector(BaseRegimeDetector):
    """Mahalanobis distance anomaly detector (Mahalanobis 1936).

    Computes the Mahalanobis distance of each observation from the
    expanding-window mean, using the expanding covariance matrix.
    Direct comparator to QFI determinant.

    Ref: Kritzman et al. (2011) "Principal Components as a Measure of
    Systemic Risk", Journal of Portfolio Management.

    Args:
        min_expanding: Minimum history before computing distance.
        regularization: Ridge regularization for covariance inversion.
    """

    def __init__(self, min_expanding: int = 60, regularization: float = 1e-6):
        self.min_expanding = min_expanding
        self.regularization = regularization

    @property
    def name(self) -> str:
        return "Mahalanobis"

    def fit(self, X: np.ndarray, **kwargs) -> 'MahalanobisDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape
        scores = np.full(T, np.nan)

        # Use incremental mean and covariance for O(T*d^2) instead of O(T^2*d)
        # Running sums for Welford-like incremental covariance
        sum_x = np.zeros(d)
        sum_xx = np.zeros((d, d))
        n_valid = 0

        for t in range(T):
            x_t = X[t]
            if np.all(np.isfinite(x_t)):
                sum_x += x_t
                sum_xx += np.outer(x_t, x_t)
                n_valid += 1

            if t < self.min_expanding or n_valid < d + 2:
                continue

            mu = sum_x / n_valid
            cov = sum_xx / n_valid - np.outer(mu, mu)
            cov += self.regularization * np.eye(d)
            try:
                cov_inv = np.linalg.inv(cov)
                diff = x_t - mu
                scores[t] = np.sqrt(np.clip(diff @ cov_inv @ diff, 0, None))
            except np.linalg.LinAlgError:
                scores[t] = np.nan
        return scores


class StructuralBreakDetector(BaseRegimeDetector):
    """Structural break proximity detector (Bai & Perron 1998).

    Uses the PELT algorithm (via ``ruptures``) to detect changepoints.
    Score = inverse distance (in days) to the nearest detected changepoint,
    computed via expanding z-score for causal consistency.

    Args:
        model: Cost model for PELT ('rbf', 'l2', 'l1'). Default: 'rbf'.
        penalty: Penalty parameter for PELT. Default: 3.0.
        min_expanding: Minimum history before scoring.
    """

    def __init__(self, model: str = 'rbf', penalty: float = 3.0,
                 min_expanding: int = 60):
        self.model = model
        self.penalty = penalty
        self.min_expanding = min_expanding

    @property
    def name(self) -> str:
        return "Structural Break"

    def fit(self, X: np.ndarray, **kwargs) -> 'StructuralBreakDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        try:
            import ruptures as rpt
        except ImportError:
            logger.warning("ruptures package not installed; returning NaN.")
            return np.full(X.shape[0], np.nan)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = X.shape[0]

        # Detect changepoints on full series
        algo = rpt.Pelt(model=self.model).fit(X)
        try:
            changepoints = algo.predict(pen=self.penalty)
        except Exception:
            return np.full(T, np.nan)

        # Remove the last element (always T)
        changepoints = [cp for cp in changepoints if cp < T]

        if len(changepoints) == 0:
            return np.full(T, np.nan)

        # Compute inverse distance to nearest changepoint
        cp_arr = np.array(changepoints)
        inv_dist = np.zeros(T)
        for t in range(T):
            min_dist = np.min(np.abs(t - cp_arr))
            inv_dist[t] = 1.0 / (1.0 + min_dist)

        # Expanding z-score
        scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(inv_dist[:t])
            sigma = np.std(inv_dist[:t], ddof=1)
            if sigma > 1e-12:
                scores[t] = (inv_dist[t] - mu) / sigma
            else:
                scores[t] = 0.0
        return scores


class TransferEntropyDetector(BaseRegimeDetector):
    """Rolling transfer entropy between assets (Schreiber 2000).

    Computes binned transfer entropy between the first two features
    (e.g. SPY and DIA returns) using a rolling window.  Score = expanding
    z-score of rolling TE.  Elevated TE signals increased information
    flow during stress.

    Args:
        te_window: Rolling window for TE estimation. Default: 60.
        n_bins: Number of bins for discretization. Default: 5.
        lag: Transfer entropy lag. Default: 1.
        min_expanding: Minimum history before z-scoring.
    """

    def __init__(self, te_window: int = 60, n_bins: int = 5, lag: int = 1,
                 min_expanding: int = 60):
        self.te_window = te_window
        self.n_bins = n_bins
        self.lag = lag
        self.min_expanding = min_expanding

    @property
    def name(self) -> str:
        return "Transfer Entropy"

    def fit(self, X: np.ndarray, **kwargs) -> 'TransferEntropyDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1 or X.shape[1] < 2:
            logger.warning("TransferEntropyDetector requires >= 2 features.")
            return np.full(X.shape[0], np.nan)

        x = X[:, 0]
        y = X[:, 1]
        T = len(x)

        # Rolling transfer entropy (Y -> X)
        te_vals = np.full(T, np.nan)
        for t in range(self.te_window + self.lag, T):
            win_x = x[t - self.te_window:t]
            win_y = y[t - self.te_window:t]
            if np.any(~np.isfinite(win_x)) or np.any(~np.isfinite(win_y)):
                continue
            te_vals[t] = self._binned_transfer_entropy(win_x, win_y)

        # Expanding z-score
        scores = np.full(T, np.nan)
        start = max(self.min_expanding, self.te_window + self.lag)
        for t in range(start, T):
            past = te_vals[:t]
            valid = past[np.isfinite(past)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 1e-12 and np.isfinite(te_vals[t]):
                scores[t] = (te_vals[t] - mu) / sigma
            else:
                scores[t] = 0.0
        return scores

    def _binned_transfer_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute TE(Y -> X) using binned estimator.

        TE(Y -> X) = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, Y_{t-1})
        """
        n = len(x) - self.lag
        if n < 10:
            return 0.0

        # Discretize
        x_bins = np.clip(
            np.digitize(x, np.linspace(x.min(), x.max(), self.n_bins + 1)[1:-1]),
            0, self.n_bins - 1,
        )
        y_bins = np.clip(
            np.digitize(y, np.linspace(y.min(), y.max(), self.n_bins + 1)[1:-1]),
            0, self.n_bins - 1,
        )

        x_t = x_bins[self.lag:]
        x_past = x_bins[:-self.lag]
        y_past = y_bins[:-self.lag]

        # Joint counts with small pseudocount to avoid log(0)
        eps = 1e-10
        nb = self.n_bins

        # P(x_t, x_past)
        joint_xx = np.zeros((nb, nb))
        for i in range(n):
            joint_xx[x_t[i], x_past[i]] += 1
        joint_xx = joint_xx / n + eps

        # P(x_t, x_past, y_past)
        joint_xxy = np.zeros((nb, nb, nb))
        for i in range(n):
            joint_xxy[x_t[i], x_past[i], y_past[i]] += 1
        joint_xxy = joint_xxy / n + eps

        # Marginals
        p_xpast = joint_xx.sum(axis=0)
        p_xpast_ypast = joint_xxy.sum(axis=0)

        # TE = sum p(x_t, x_past, y_past) * log(p(x_t|x_past,y_past) / p(x_t|x_past))
        te = 0.0
        for xt in range(nb):
            for xp in range(nb):
                for yp in range(nb):
                    p_joint = joint_xxy[xt, xp, yp]
                    p_cond_xy = p_joint / p_xpast_ypast[xp, yp]
                    p_cond_x = joint_xx[xt, xp] / p_xpast[xp]
                    if p_cond_xy > eps and p_cond_x > eps:
                        te += p_joint * np.log(p_cond_xy / p_cond_x)
        return max(0.0, te)
