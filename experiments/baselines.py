"""
Classical Baseline Regime Detectors for Head-to-Head Comparison

Six classical/statistical baselines:
1. RollingVolatilityDetector — rolling sigma z-scored against expanding window
2. CUSUMDetector — cumulative sum of mean-adjusted absolute returns
3. HMMRegimeDetector — 2-state Gaussian HMM via hmmlearn
4. RandomForestRegimeDetector — P(crisis) from sklearn RF
5. BOCPDDetector — Bayesian Online Changepoint Detection (Adams & MacKay 2007)
6. IsolationForestDetector — unsupervised anomaly detection via sklearn
"""

import logging
from typing import Optional

import numpy as np
from scipy.special import gammaln

from qcml_geometry import BaseRegimeDetector

logger = logging.getLogger(__name__)


class RollingVolatilityDetector(BaseRegimeDetector):
    """Rolling volatility z-scored against an expanding window.

    score[t] = (sigma_window[t] - mu_expanding[t]) / sigma_expanding[t]

    Args:
        vol_window: Window for computing rolling standard deviation.
        min_expanding: Minimum expanding window before scoring starts.
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
        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        T = len(returns)
        scores = np.full(T + 1, np.nan)

        for t in range(self.vol_window, T):
            sigma_t = np.std(returns[t - self.vol_window:t], ddof=1)

            if t >= self.min_expanding:
                expanding_vols = []
                for j in range(self.vol_window, t + 1):
                    expanding_vols.append(
                        np.std(returns[j - self.vol_window:j], ddof=1)
                    )
                mu_exp = np.mean(expanding_vols)
                sigma_exp = np.std(expanding_vols, ddof=1)
                if sigma_exp > 1e-12:
                    scores[t + 1] = (sigma_t - mu_exp) / sigma_exp
                else:
                    scores[t + 1] = 0.0

        return scores


class CUSUMDetector(BaseRegimeDetector):
    """Cumulative sum (CUSUM) detector for mean shifts in absolute returns.

    S[t] = max(0, S[t-1] + |r[t]| - k)

    where k (the drift parameter) defaults to the mean of |returns| during
    a burn-in period.

    Args:
        burn_in: Number of observations to estimate k.
    """

    def __init__(self, burn_in: int = 60):
        self.burn_in = burn_in
        self._k: Optional[float] = None

    @property
    def name(self) -> str:
        return "CUSUM"

    def fit(self, X: np.ndarray, **kwargs) -> 'CUSUMDetector':
        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        n = min(self.burn_in, len(returns))
        self._k = float(np.mean(np.abs(returns[:n])))
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._k is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        T = len(returns)
        S = np.zeros(T + 1)
        for t in range(T):
            S[t + 1] = max(0.0, S[t] + np.abs(returns[t]) - self._k)

        return S


class HMMRegimeDetector(BaseRegimeDetector):
    """2-state Gaussian Hidden Markov Model for regime detection.

    Score = P(high-volatility state) at each time step.  The high-vol
    state is identified as the one with larger variance.

    Requires ``hmmlearn``.

    Args:
        n_iter: Maximum EM iterations.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_iter: int = 100, seed: int = 42):
        self.n_iter = n_iter
        self.seed = seed
        self._model = None
        self._high_vol_state: Optional[int] = None

    @property
    def name(self) -> str:
        return "HMM 2-state"

    def fit(self, X: np.ndarray, **kwargs) -> 'HMMRegimeDetector':
        from hmmlearn.hmm import GaussianHMM

        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0]).reshape(-1, 1)
        else:
            returns = np.diff(X.ravel()).reshape(-1, 1)

        sorted_rets = np.sort(returns.ravel())
        n = len(sorted_rets)
        low_half = sorted_rets[:n // 2]
        high_half = sorted_rets[n // 2:]

        self._model = GaussianHMM(
            n_components=2,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.seed,
            init_params='',
        )
        self._model.startprob_ = np.array([0.5, 0.5])
        self._model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        self._model.means_ = np.array([
            [np.mean(low_half)], [np.mean(high_half)]
        ])
        self._model.covars_ = np.array([
            [[np.var(low_half, ddof=1) + 1e-10]],
            [[np.var(high_half, ddof=1) + 1e-10]],
        ])
        self._model.fit(returns)

        variances = [self._model.covars_[i][0, 0] for i in range(2)]
        self._high_vol_state = int(np.argmax(variances))
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0]).reshape(-1, 1)
        else:
            returns = np.diff(X.ravel()).reshape(-1, 1)

        posteriors = self._model.predict_proba(returns)
        high_vol_prob = posteriors[:, self._high_vol_state]

        return np.concatenate([[np.nan], high_vol_prob])


class RandomForestRegimeDetector(BaseRegimeDetector):
    """Random Forest classifier for regime detection.

    Produces P(crisis) from a trained RF.  Training requires labeled data
    via ``fit_with_labels``.  The standard ``fit()`` is a no-op.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        seed: Random seed.
        lookback: Rolling feature window for constructing ML features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        seed: int = 42,
        lookback: int = 20,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.lookback = lookback
        self._model = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'RandomForestRegimeDetector':
        return self

    def fit_with_labels(
        self, X: np.ndarray, y: np.ndarray
    ) -> 'RandomForestRegimeDetector':
        """Train the RF on labeled feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (0 = normal, 1 = crisis).
        """
        from sklearn.ensemble import RandomForestClassifier

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            class_weight='balanced',
        )
        X_feat = self._build_ml_features(X)
        y_trimmed = y[self.lookback - 1:]
        self._model.fit(X_feat, y_trimmed)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "Call fit_with_labels() before compute_regime_scores()."
            )
        X_feat = self._build_ml_features(X)
        proba = self._model.predict_proba(X_feat)

        crisis_col = list(self._model.classes_).index(1)
        scores = proba[:, crisis_col]

        pad = np.full(len(X) - len(scores), np.nan)
        return np.concatenate([pad, scores])

    def _build_ml_features(self, X: np.ndarray) -> np.ndarray:
        """Build rolling features (mean, std, min, max) from raw X."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T, d = X.shape
        features = []

        for t in range(self.lookback - 1, T):
            window = X[t - self.lookback + 1:t + 1]
            row = np.concatenate([
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.min(window, axis=0),
                np.max(window, axis=0),
            ])
            features.append(row)

        return np.array(features)


class BOCPDDetector(BaseRegimeDetector):
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Maintains a distribution over run lengths (time since last changepoint)
    using a conjugate Normal-Inverse-Gamma model with Student-t predictive
    distribution.  The regime score at each time step is
    P(changepoint in last ``cp_threshold`` steps), i.e. the posterior mass
    on short run lengths.

    Higher scores indicate the model believes a recent changepoint occurred,
    which corresponds to a regime transition.

    Reference:
        Adams, R.P. & MacKay, D.J.C. (2007). Bayesian Online Changepoint
        Detection. arXiv:0710.3742.

    Args:
        hazard_rate: Expected number of observations between changepoints.
            The constant hazard function is H(tau) = 1/hazard_rate.
        cp_threshold: Run lengths below this value contribute to the
            "recent changepoint" score.
        mu0: Prior mean of the Normal-Inverse-Gamma conjugate.
        kappa0: Prior precision scaling (number of pseudo-observations
            for the mean).
        alpha0: Prior shape for the Inverse-Gamma variance.
        beta0: Prior rate for the Inverse-Gamma variance.
    """

    def __init__(
        self,
        hazard_rate: float = 200.0,
        cp_threshold: int = 20,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        self.hazard_rate = hazard_rate
        self.cp_threshold = cp_threshold
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self._fitted = False

    @property
    def name(self) -> str:
        return "BOCPD"

    def fit(self, X: np.ndarray, **kwargs) -> 'BOCPDDetector':
        """Fit is a no-op; BOCPD is fully online.

        The prior hyperparameters are set at construction time.
        Calling fit() simply marks the detector as ready.

        Args:
            X: Feature matrix (T, d). Not used for fitting.

        Returns:
            self
        """
        self._fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Run the online BOCPD algorithm and return changepoint scores.

        Processes the first column of X (or the 1-D input) as the
        observation stream.  Uses a Normal-Inverse-Gamma conjugate model
        so that the predictive distribution at each run length is a
        Student-t.

        Args:
            X: Feature matrix (T, d) or 1-D array (T,).

        Returns:
            scores: Array of shape (T,) where scores[t] =
                sum of posterior run-length probabilities for
                run_length < cp_threshold, i.e. P(recent changepoint).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        # Extract univariate observation stream
        if X.ndim == 2 and X.shape[1] > 1:
            obs = X[:, 0].ravel()
        else:
            obs = X.ravel()

        T = len(obs)
        scores = np.zeros(T)

        # Hazard function: constant prior P(changepoint) = 1/hazard_rate
        hazard = 1.0 / self.hazard_rate

        # Sufficient statistics arrays, indexed by run length r = 0..t
        # At time t we maintain vectors of length (t+1) for run lengths 0..t
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        # Run-length distribution: R[r] = P(run_length = r | x_{1:t})
        # Initialize with R[0] = 1 (before observing any data, run length is 0)
        R = np.array([1.0])

        for t in range(T):
            x = obs[t]

            # --- Predictive probability: Student-t for each run length ---
            # Parameters of the predictive Student-t:
            #   df = 2 * alpha
            #   loc = mu
            #   scale = sqrt(beta * (kappa + 1) / (alpha * kappa))
            df = 2.0 * alpha
            scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
            scale = np.sqrt(np.maximum(scale_sq, 1e-20))

            # Student-t log-pdf (vectorized across all active run lengths)
            z = (x - mu) / scale
            log_pred = (
                gammaln(0.5 * (df + 1.0))
                - gammaln(0.5 * df)
                - 0.5 * np.log(np.pi * df)
                - np.log(scale)
                - 0.5 * (df + 1.0) * np.log1p(z ** 2 / df)
            )
            pred = np.exp(log_pred)

            # --- Growth probabilities: extend each existing run length ---
            # P(r_{t+1} = r+1, x_{1:t+1}) = P(r_t = r, x_{1:t}) * pred * (1 - H)
            growth = R * pred * (1.0 - hazard)

            # --- Changepoint probability: all mass that resets to r=0 ---
            # P(r_{t+1} = 0, x_{1:t+1}) = sum_r P(r_t = r, x_{1:t}) * pred * H
            cp_mass = np.sum(R * pred * hazard)

            # --- Assemble new run-length distribution ---
            R_new = np.empty(len(R) + 1)
            R_new[0] = cp_mass
            R_new[1:] = growth

            # Normalize to avoid numerical underflow / overflow
            R_total = R_new.sum()
            if R_total > 0:
                R_new /= R_total

            R = R_new

            # --- Score: posterior mass on short run lengths ---
            # P(run_length < cp_threshold) signals a recent changepoint
            upper = min(self.cp_threshold, len(R))
            scores[t] = R[:upper].sum()

            # --- Update sufficient statistics for the next step ---
            # Conjugate update: incorporate observation x into each run length
            mu_new = np.empty(len(mu) + 1)
            kappa_new = np.empty(len(kappa) + 1)
            alpha_new = np.empty(len(alpha) + 1)
            beta_new = np.empty(len(beta) + 1)

            # Run length 0 resets to the prior
            mu_new[0] = self.mu0
            kappa_new[0] = self.kappa0
            alpha_new[0] = self.alpha0
            beta_new[0] = self.beta0

            # Run lengths 1..t+1: update from run lengths 0..t
            kappa_updated = kappa + 1.0
            mu_updated = (kappa * mu + x) / kappa_updated
            alpha_updated = alpha + 0.5
            beta_updated = (
                beta + 0.5 * kappa * (x - mu) ** 2 / kappa_updated
            )

            mu_new[1:] = mu_updated
            kappa_new[1:] = kappa_updated
            alpha_new[1:] = alpha_updated
            beta_new[1:] = beta_updated

            mu = mu_new
            kappa = kappa_new
            alpha = alpha_new
            beta = beta_new

        return scores


class IsolationForestDetector(BaseRegimeDetector):
    """Isolation Forest anomaly detector for regime detection.

    Trains sklearn's IsolationForest on the feature matrix and uses the
    negated ``decision_function`` as the regime score so that higher values
    correspond to more anomalous (crisis-like) observations.

    This is a standard unsupervised baseline from the anomaly detection
    literature (Liu et al., 2008).

    Args:
        n_estimators: Number of isolation trees.
        contamination: Expected fraction of anomalies in the training data.
            Controls the offset of the decision function.
        seed: Random seed for reproducibility.
        causal_fit_length: If provided, fit() uses only the first
            ``causal_fit_length`` rows to avoid temporal leakage.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self._model = None

    @property
    def name(self) -> str:
        return "Isolation Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model.

        Args:
            X: Feature matrix (T, d).  If ``causal_fit_length`` is set,
                only the first ``causal_fit_length`` rows are used for
                training to maintain temporal causality.

        Returns:
            self
        """
        from sklearn.ensemble import IsolationForest

        X_fit = X
        if self.causal_fit_length is not None and self.causal_fit_length < len(X):
            X_fit = X[:self.causal_fit_length]

        if X_fit.ndim == 1:
            X_fit = X_fit.reshape(-1, 1)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self._model.fit(X_fit)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for each observation.

        Uses the negated ``decision_function`` from sklearn so that
        higher values indicate greater anomalousness (more crisis-like).
        The raw decision_function returns large positive values for
        inliers and negative values for outliers; negating aligns with
        the convention that higher score = more crisis-like.

        Args:
            X: Feature matrix (T, d) or 1-D array (T,).

        Returns:
            scores: Array of shape (T,) where higher = more anomalous.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # decision_function: positive = inlier, negative = outlier
        # Negate so that higher = more anomalous / crisis-like
        return -self._model.decision_function(X)
