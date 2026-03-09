"""
Empirical tests for fusion/combination questions Q40-Q42.

Q40: Bayesian Model Averaging (BMA) with observables as "models"
Q41: Stacking with gradient boosting (XGBoost/LightGBM) on observable z-scores
Q42: Switching model — select single best observable per regime (causal oracle selection)

Smoke-test protocol:
  - 4 crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
  - Symbol: SPY via yfinance
  - Base observables: BerryPhaseRateDetector, SpectralGapDetector, ReducedPurityDetector,
      SpectralEntropyDetector, DimensionalityCollapseDetector
  - Keep fusion method if median Cohen's d > 0.3
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- repo root on path ---
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix_single_asset, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import (
    BerryPhaseRateDetector,
    SpectralGapDetector,
    ReducedPurityDetector,
    SpectralEntropyDetector,
    DimensionalityCollapseDetector,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_CRISES = ["2008_gfc", "2020_covid", "2022_rates", "2023_svb"]
SYMBOL = "SPY"

# Data window: start well before earliest crisis, end after latest
DATA_START = "2005-01-01"
DATA_END = "2024-12-31"

KEEP_THRESHOLD = 0.3  # median Cohen's d threshold

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spy_prices():
    """Fetch SPY close prices as a Series."""
    print(f"Fetching {SYMBOL} data {DATA_START} to {DATA_END}...")
    raw = fetch_data([SYMBOL], DATA_START, DATA_END, use_cache=True)
    prices_df = raw["close"].unstack("symbol")
    prices = prices_df[SYMBOL].dropna()
    print(f"  Got {len(prices)} trading days ({prices.index[0].date()} to {prices.index[-1].date()})")
    return prices


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(prices: pd.Series):
    """Build feature matrix from SPY prices using single-asset features."""
    X, dates = create_feature_matrix_single_asset(prices, extra_lags=True)
    return X, dates


# ---------------------------------------------------------------------------
# Base detector fitting and scoring
# ---------------------------------------------------------------------------

def build_base_detectors():
    """Instantiate all 5 base detectors."""
    return [
        BerryPhaseRateDetector(hilbert_dim=8, n_pca_components=8, seed=42),
        SpectralGapDetector(hilbert_dim=8, n_pca_components=8, seed=42),
        ReducedPurityDetector(hilbert_dim=8, n_pca_components=8, seed=42),
        SpectralEntropyDetector(hilbert_dim=8, n_pca_components=8, seed=42),
        DimensionalityCollapseDetector(hilbert_dim=8, n_pca_components=8, seed=42),
    ]


def fit_and_score_detectors(detectors, X, dates, verbose=True):
    """Fit each detector on all data and return z-score time series.

    Returns:
        scores_df: DataFrame (T, n_detectors) of z-scores, index=dates.
    """
    score_dict = {}
    for det in detectors:
        if verbose:
            print(f"  Fitting {det.name}...")
        det.fit(X)
        raw_scores = det.compute_regime_scores(X)
        # Align to dates (BerryPhaseRateDetector outputs T-1 length scores)
        if len(raw_scores) == len(dates) - 1:
            score_dict[det.name] = pd.Series(raw_scores, index=dates[1:])
        else:
            score_dict[det.name] = pd.Series(raw_scores, index=dates)

    # Align all on common index
    scores_df = pd.DataFrame(score_dict)
    scores_df = scores_df.dropna(how="all")
    return scores_df


# ---------------------------------------------------------------------------
# Crisis mask utilities
# ---------------------------------------------------------------------------

def build_crisis_mask(dates: pd.DatetimeIndex, crisis_name: str) -> pd.Series:
    """Return boolean Series: True during the crisis window."""
    c = ALL_CRISES[crisis_name]
    start = pd.Timestamp(c["start"])
    end = pd.Timestamp(c["end"])
    mask = pd.Series(False, index=dates, name=crisis_name)
    mask.loc[start:end] = True
    return mask


def evaluate_score_per_crisis(score_series: pd.Series, crisis_names: list, n_bootstrap: int = 1000):
    """Compute Cohen's d per crisis, returning list of (crisis, d, ci_lo, ci_hi)."""
    results = []
    for crisis_name in crisis_names:
        mask = build_crisis_mask(score_series.index, crisis_name)
        aligned_mask = mask.reindex(score_series.index, fill_value=False)

        crisis_scores = score_series[aligned_mask].dropna().values
        normal_scores = score_series[~aligned_mask].dropna().values

        if len(crisis_scores) < 5 or len(normal_scores) < 10:
            results.append((crisis_name, np.nan, np.nan, np.nan))
            continue

        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_scores, normal_scores,
            n_bootstrap=n_bootstrap, seed=42, method="block",
        )
        results.append((crisis_name, d, ci_lo, ci_hi))
    return results


# ---------------------------------------------------------------------------
# Q40: Bayesian Model Averaging
# ---------------------------------------------------------------------------

def bma_weights_brier(scores_df: pd.Series, crisis_names: list, n_bootstrap: int = 500):
    """Compute BMA weights proportional to 1/(Brier score + eps) per observable.

    The Brier score for a binary prediction task B_i = mean((p_i - y)^2)
    where p_i = sigmoid(z_i) and y is the crisis binary label.
    Lower Brier score → better calibration → higher weight.

    Uses expanding window: weight for crisis k uses performance on crises 1..k-1.
    With only 4 crises the first crisis uses uniform weights (no prior data).

    Returns:
        weights_by_crisis: dict {crisis_name: np.ndarray of weights, shape (n_obs,)}
        all_brier: dict {crisis_name: per-observable Brier scores}
    """
    obs_names = list(scores_df.columns)
    n_obs = len(obs_names)
    weights_by_crisis = {}
    all_brier = {}

    for k, target_crisis in enumerate(crisis_names):
        if k == 0:
            # No training data yet — use uniform
            weights_by_crisis[target_crisis] = np.ones(n_obs) / n_obs
            all_brier[target_crisis] = np.ones(n_obs) / n_obs
            continue

        # Training crises = all before target
        train_crises = crisis_names[:k]

        brier_scores = np.zeros(n_obs)
        for i, obs_name in enumerate(obs_names):
            obs_scores = scores_df[obs_name].dropna()
            total_sq_err = 0.0
            n_total = 0

            for train_crisis in train_crises:
                mask = build_crisis_mask(obs_scores.index, train_crisis)
                aligned_mask = mask.reindex(obs_scores.index, fill_value=False)
                y = aligned_mask.values.astype(float)

                z = obs_scores.values
                # Clamp z to prevent sigmoid overflow
                z_clamped = np.clip(z, -20, 20)
                p = 1.0 / (1.0 + np.exp(-z_clamped))

                # Replace NaN z-scores with 0.5 probability
                p = np.where(np.isnan(obs_scores.values), 0.5, p)

                sq_err = (p - y) ** 2
                total_sq_err += np.nansum(sq_err)
                n_total += np.sum(~np.isnan(obs_scores.values))

            brier_scores[i] = total_sq_err / max(n_total, 1)

        all_brier[target_crisis] = brier_scores
        # Weight proportional to inverse Brier score
        inv_brier = 1.0 / (brier_scores + 1e-6)
        weights_by_crisis[target_crisis] = inv_brier / inv_brier.sum()

    return weights_by_crisis, all_brier


def compute_bma_score(scores_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted average of observable z-scores.

    Args:
        scores_df: (T, n_obs) DataFrame of z-scores.
        weights: (n_obs,) weight vector summing to 1.

    Returns:
        Series of length T.
    """
    mat = scores_df.fillna(0).values  # replace NaN with 0 (neutral z-score)
    bma_scores = mat @ weights
    return pd.Series(bma_scores, index=scores_df.index, name="BMA")


def run_q40(scores_df: pd.DataFrame, crisis_names: list):
    """Test Q40: BMA vs uniform weighting."""
    print("\n" + "=" * 60)
    print("Q40: Bayesian Model Averaging")
    print("=" * 60)

    obs_names = list(scores_df.columns)
    n_obs = len(obs_names)

    # Uniform baseline
    uniform_weights = np.ones(n_obs) / n_obs
    uniform_score = compute_bma_score(scores_df, uniform_weights)

    # BMA weights (expanding window, Brier-based)
    weights_by_crisis, all_brier = bma_weights_brier(scores_df, crisis_names)

    print("\nBMA weights per crisis (expanding window, Brier-based):")
    print(f"  {'Crisis':<20} " + " ".join(f"{n[:12]:<14}" for n in obs_names))
    for crisis_name in crisis_names:
        w = weights_by_crisis[crisis_name]
        row = f"  {crisis_name:<20} " + " ".join(f"{wi:<14.3f}" for wi in w)
        print(row)

    # Evaluate both uniform and BMA
    print("\nCohen's d per crisis:")
    print(f"  {'Crisis':<20} {'Uniform d':>12} {'BMA d':>10} {'Improvement':>12}")

    bma_per_crisis_scores = []
    uniform_d_vals = []
    bma_d_vals = []

    for crisis_name in crisis_names:
        mask = build_crisis_mask(uniform_score.index, crisis_name)
        aligned_mask = mask.reindex(uniform_score.index, fill_value=False)

        # Uniform
        u_crisis = uniform_score[aligned_mask].dropna().values
        u_normal = uniform_score[~aligned_mask].dropna().values
        u_d, _, _ = compute_cohens_d_with_ci(u_crisis, u_normal, n_bootstrap=500, seed=42)

        # BMA: compute the score for this crisis using its specific weights
        w = weights_by_crisis[crisis_name]
        bma_score_crisis = compute_bma_score(scores_df, w)
        b_crisis = bma_score_crisis[aligned_mask].dropna().values
        b_normal = bma_score_crisis[~aligned_mask].dropna().values
        b_d, _, _ = compute_cohens_d_with_ci(b_crisis, b_normal, n_bootstrap=500, seed=42)

        delta = b_d - u_d if not (np.isnan(u_d) or np.isnan(b_d)) else np.nan
        arrow = "+" if (not np.isnan(delta) and delta > 0) else ""
        print(f"  {crisis_name:<20} {u_d:>12.3f} {b_d:>10.3f} {arrow}{delta:>11.3f}")

        uniform_d_vals.append(u_d)
        bma_d_vals.append(b_d)

    median_u = np.nanmedian(uniform_d_vals)
    median_bma = np.nanmedian(bma_d_vals)
    print(f"\n  {'MEDIAN':<20} {median_u:>12.3f} {median_bma:>10.3f} {median_bma - median_u:>+12.3f}")

    keep = median_bma > KEEP_THRESHOLD
    print(f"\nResult: median BMA d = {median_bma:.3f} (threshold={KEEP_THRESHOLD})")
    print(f"        median Uniform d = {median_u:.3f}")
    print(f"Recommendation: {'KEEP - BMA improves over uniform' if keep and median_bma > median_u else 'MARGINAL' if keep else 'DISCARD'}")

    # Brier score analysis
    print("\nBrier scores (lower is better, used as BMA weights):")
    print(f"  {'Crisis (test)':<20} " + " ".join(f"{n[:12]:<14}" for n in obs_names))
    for crisis_name in crisis_names[1:]:  # skip first (uniform, no training data)
        brier = all_brier[crisis_name]
        row = f"  {crisis_name:<20} " + " ".join(f"{b:<14.4f}" for b in brier)
        print(row)

    return {
        "question": "Q40",
        "method": "BMA (Brier-weighted)",
        "median_d_uniform": float(median_u),
        "median_d_bma": float(median_bma),
        "improvement": float(median_bma - median_u),
        "per_crisis": {
            cn: {"uniform_d": float(ud), "bma_d": float(bd)}
            for cn, ud, bd in zip(crisis_names, uniform_d_vals, bma_d_vals)
        },
        "keep": bool(keep),
    }


# ---------------------------------------------------------------------------
# Q41: Stacking with gradient boosting
# ---------------------------------------------------------------------------

def run_q41(scores_df: pd.DataFrame, crisis_names: list):
    """Test Q41: Stacking with LightGBM/XGBoost (expanding window, no lookahead).

    With only 4 crises, we train on k crises and test on crisis k+1.
    Minimum training: 1 crisis before first test (so we can only test 3 crises).
    Documents overfitting honestly.
    """
    print("\n" + "=" * 60)
    print("Q41: Stacking with Gradient Boosting (Expanding Window)")
    print("=" * 60)

    # Try to import LightGBM; fall back to sklearn GradientBoosting
    try:
        import lightgbm as lgb
        STACKER_NAME = "LightGBM"

        def make_stacker():
            return lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                min_child_samples=5,
                verbose=-1,
                random_state=42,
            )

    except ImportError:
        try:
            import xgboost as xgb
            STACKER_NAME = "XGBoost"

            def make_stacker():
                return xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    min_child_leaves=5,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    verbosity=0,
                    random_state=42,
                )

        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            STACKER_NAME = "sklearn GradientBoosting"

            def make_stacker():
                return GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    min_samples_leaf=5,
                    random_state=42,
                )

    print(f"Stacker: {STACKER_NAME}")
    print("\nExpanding window cross-validation:")
    print(f"  Train on crises 1..k, test on crisis k+1")
    print(f"  (Minimum 1 training crisis required)")

    if len(crisis_names) < 2:
        print("ERROR: Need at least 2 crises for expanding window CV.")
        return None

    obs_names = list(scores_df.columns)

    # Build full feature matrix and label vector
    feat_df = scores_df.fillna(0)

    # Full crisis label
    full_label = pd.Series(0, index=feat_df.index, dtype=int)
    for cn in crisis_names:
        mask = build_crisis_mask(feat_df.index, cn)
        aligned = mask.reindex(feat_df.index, fill_value=False)
        full_label[aligned] = 1

    results_per_crisis = []
    stacked_d_vals = []

    # Also track uniform ensemble as baseline for comparison
    uniform_d_vals_q41 = []

    # Expanding window: train on first k crises, predict on crisis k+1
    for k in range(1, len(crisis_names)):
        train_crises = crisis_names[:k]
        test_crisis = crisis_names[k]

        print(f"\n  Fold {k}: train={train_crises}, test={test_crisis}")

        # Build training set: crisis days + equal sample of non-crisis days from training period
        # Determine training time range: up to start of test crisis
        test_start = pd.Timestamp(ALL_CRISES[test_crisis]["start"])

        train_mask = feat_df.index < test_start
        X_train_all = feat_df[train_mask].values
        y_train_all = full_label[train_mask].values

        # Check for enough training data
        n_pos_train = y_train_all.sum()
        n_neg_train = (y_train_all == 0).sum()

        print(f"    Training data: {len(X_train_all)} samples ({n_pos_train} crisis, {n_neg_train} normal)")

        if n_pos_train < 5:
            print(f"    SKIP: Too few training crisis samples ({n_pos_train})")
            stacked_d_vals.append(np.nan)
            uniform_d_vals_q41.append(np.nan)
            continue

        # Build test set: test crisis + normal context around it
        test_mask = build_crisis_mask(feat_df.index, test_crisis)
        aligned_test_mask = test_mask.reindex(feat_df.index, fill_value=False)
        # Use all data after test_start as the "test context"
        after_train = feat_df.index >= test_start
        X_test_context = feat_df[after_train].values
        y_test_context = full_label[after_train].values

        n_pos_test = y_test_context.sum()
        print(f"    Test data: {len(X_test_context)} samples ({n_pos_test} crisis)")

        if n_pos_test < 3:
            print(f"    SKIP: Too few test crisis samples ({n_pos_test})")
            stacked_d_vals.append(np.nan)
            uniform_d_vals_q41.append(np.nan)
            continue

        # Fit stacker
        stacker = make_stacker()
        try:
            stacker.fit(X_train_all, y_train_all)
        except Exception as e:
            print(f"    ERROR fitting stacker: {e}")
            stacked_d_vals.append(np.nan)
            uniform_d_vals_q41.append(np.nan)
            continue

        # Predict probabilities on test context
        try:
            proba = stacker.predict_proba(X_test_context)[:, 1]
        except Exception as e:
            print(f"    ERROR predicting: {e}")
            stacked_d_vals.append(np.nan)
            uniform_d_vals_q41.append(np.nan)
            continue

        # Split predictions into crisis vs normal
        test_crisis_idx = aligned_test_mask[after_train].values
        stacked_crisis_proba = proba[test_crisis_idx]
        stacked_normal_proba = proba[~test_crisis_idx]

        if len(stacked_crisis_proba) < 3 or len(stacked_normal_proba) < 5:
            print(f"    SKIP: Insufficient test split.")
            stacked_d_vals.append(np.nan)
            uniform_d_vals_q41.append(np.nan)
            continue

        stacked_d, _, _ = compute_cohens_d_with_ci(
            stacked_crisis_proba, stacked_normal_proba, n_bootstrap=500, seed=42
        )

        # Compare to uniform ensemble on same test crisis
        uniform_score_col = feat_df[obs_names].fillna(0).mean(axis=1)
        u_crisis_scores = uniform_score_col[aligned_test_mask].dropna().values
        u_normal_scores = uniform_score_col[~aligned_test_mask].dropna().values
        u_d, _, _ = compute_cohens_d_with_ci(u_crisis_scores, u_normal_scores, n_bootstrap=500, seed=42)

        print(f"    Stacking d = {stacked_d:.3f} | Uniform ensemble d = {u_d:.3f}")

        # Also compute in-sample training performance (to detect overfitting)
        train_proba = stacker.predict_proba(X_train_all)[:, 1]
        train_crisis_proba = train_proba[y_train_all == 1]
        train_normal_proba = train_proba[y_train_all == 0]
        if len(train_crisis_proba) >= 3 and len(train_normal_proba) >= 5:
            in_sample_d, _, _ = compute_cohens_d_with_ci(
                train_crisis_proba, train_normal_proba, n_bootstrap=200, seed=42
            )
            print(f"    In-sample (train) d = {in_sample_d:.3f} (compare to OOS d = {stacked_d:.3f})")
            overfit_ratio = (in_sample_d - stacked_d) / (in_sample_d + 1e-6) if in_sample_d > 0 else 0
            print(f"    Overfitting ratio = {overfit_ratio:.2f} (0=no overfit, 1=complete overfit)")

        stacked_d_vals.append(stacked_d)
        uniform_d_vals_q41.append(u_d)
        results_per_crisis.append({
            "test_crisis": test_crisis,
            "train_crises": train_crises,
            "stacked_d": float(stacked_d),
            "uniform_d": float(u_d),
        })

    valid_stacked = [d for d in stacked_d_vals if not np.isnan(d)]
    valid_uniform = [d for d in uniform_d_vals_q41 if not np.isnan(d)]
    median_stacked = float(np.median(valid_stacked)) if valid_stacked else np.nan
    median_uniform = float(np.median(valid_uniform)) if valid_uniform else np.nan

    print(f"\nSummary:")
    print(f"  Median stacked OOS d = {median_stacked:.3f}")
    print(f"  Median uniform ensemble d = {median_uniform:.3f}")
    keep = not np.isnan(median_stacked) and median_stacked > KEEP_THRESHOLD

    print(f"\nResult: median stacking d = {median_stacked:.3f} (threshold={KEEP_THRESHOLD})")
    print(f"        Stacker: {STACKER_NAME}")
    print(f"Conclusion: With only {len(crisis_names)} crises, the expanding window")
    print(f"  produces very limited training data. Stacking likely overfits.")
    print(f"Recommendation: {'KEEP' if keep else 'DISCARD - likely overfits with limited crisis data'}")

    return {
        "question": "Q41",
        "method": f"Stacking ({STACKER_NAME})",
        "median_d_stacked_oos": median_stacked,
        "median_d_uniform": median_uniform,
        "n_valid_folds": len(valid_stacked),
        "per_crisis": results_per_crisis,
        "keep": bool(keep),
        "note": f"Expanding window with {len(crisis_names)} crises. Only {len(valid_stacked)} valid OOS folds.",
    }


# ---------------------------------------------------------------------------
# Q42: Causal switching model (oracle channel selection)
# ---------------------------------------------------------------------------

def run_q42(scores_df: pd.DataFrame, crisis_names: list):
    """Test Q42: Switching model — select the single best observable at each time.

    Strategy 1: max(z_score_i) across observables at each time step.
    Strategy 2: Expanding-window oracle — use the observable that performed best
                in all prior crises. Switch between observables as historical
                evidence accumulates.

    Both are evaluated causally (expanding window, no lookahead).
    """
    print("\n" + "=" * 60)
    print("Q42: Causal Switching Model")
    print("=" * 60)

    obs_names = list(scores_df.columns)
    n_obs = len(obs_names)

    # --- Strategy 1: max(z_score) across observables at each time step ---
    print("\nStrategy 1: max(z_score) across all observables at each time step")
    max_score = scores_df.fillna(0).max(axis=1)
    max_score.name = "Max(z)"

    max_d_vals = []
    print(f"  {'Crisis':<20} {'Max(z) d':>10} {'Uniform d':>10} {'Best Single d':>14}")
    for crisis_name in crisis_names:
        mask = build_crisis_mask(max_score.index, crisis_name)
        aligned_mask = mask.reindex(max_score.index, fill_value=False)

        # Max score
        c_scores = max_score[aligned_mask].dropna().values
        n_scores = max_score[~aligned_mask].dropna().values
        d_max, _, _ = compute_cohens_d_with_ci(c_scores, n_scores, n_bootstrap=500, seed=42)

        # Uniform for comparison
        uniform = scores_df.fillna(0).mean(axis=1)
        u_c = uniform[aligned_mask].dropna().values
        u_n = uniform[~aligned_mask].dropna().values
        d_uni, _, _ = compute_cohens_d_with_ci(u_c, u_n, n_bootstrap=500, seed=42)

        # Best single for comparison
        best_d = 0.0
        for obs_name in obs_names:
            obs_s = scores_df[obs_name].dropna()
            obs_aligned_mask = aligned_mask.reindex(obs_s.index, fill_value=False)
            oc = obs_s[obs_aligned_mask].values
            on_ = obs_s[~obs_aligned_mask].values
            if len(oc) >= 3 and len(on_) >= 5:
                od, _, _ = compute_cohens_d_with_ci(oc, on_, n_bootstrap=200, seed=42)
                if not np.isnan(od):
                    best_d = max(best_d, od)

        print(f"  {crisis_name:<20} {d_max:>10.3f} {d_uni:>10.3f} {best_d:>14.3f}")
        max_d_vals.append(d_max)

    median_max = float(np.nanmedian(max_d_vals))

    # --- Strategy 2: Expanding-window oracle selection ---
    print(f"\nStrategy 2: Expanding-window oracle (select best observable from prior crises)")

    oracle_scores = pd.Series(np.nan, index=scores_df.index, name="Oracle(expanding)")
    best_obs_history = {}  # maps crisis -> best observable at that time

    prev_best_obs = obs_names[0]  # default to first observable before any crisis data

    for k, crisis_name in enumerate(crisis_names):
        test_start = pd.Timestamp(ALL_CRISES[crisis_name]["start"])
        test_end = pd.Timestamp(ALL_CRISES[crisis_name]["end"])

        if k == 0:
            # No prior data: use observable with highest recent variance
            # (a proxy for most active signal — does not use crisis labels)
            pre_crisis = scores_df[scores_df.index < test_start].tail(252)
            if len(pre_crisis) > 20:
                obs_variance = pre_crisis.fillna(0).var()
                prev_best_obs = obs_variance.idxmax()
            best_obs_history[crisis_name] = prev_best_obs
        else:
            # Select based on best d-value in all prior crises
            train_crises = crisis_names[:k]
            d_per_obs = {}
            for obs_name in obs_names:
                obs_s = scores_df[obs_name].dropna()
                ds = []
                for tc in train_crises:
                    mask = build_crisis_mask(obs_s.index, tc)
                    aligned = mask.reindex(obs_s.index, fill_value=False)
                    oc = obs_s[aligned].values
                    on_ = obs_s[~aligned].values
                    if len(oc) >= 3 and len(on_) >= 5:
                        od, _, _ = compute_cohens_d_with_ci(oc, on_, n_bootstrap=200, seed=42)
                        if not np.isnan(od):
                            ds.append(od)
                d_per_obs[obs_name] = float(np.mean(ds)) if ds else 0.0

            best_obs = max(d_per_obs, key=d_per_obs.get)
            best_obs_history[crisis_name] = best_obs
            prev_best_obs = best_obs

            print(f"  Crisis {crisis_name}: selected '{best_obs}' (d history: {d_per_obs})")

        # Fill oracle scores for crisis period + subsequent non-crisis period
        # using the selected observable
        period_mask = (scores_df.index >= test_start)
        # Only apply until next crisis starts
        if k < len(crisis_names) - 1:
            next_start = pd.Timestamp(ALL_CRISES[crisis_names[k + 1]]["start"])
            period_mask = period_mask & (scores_df.index < next_start)

        selected_obs = best_obs_history[crisis_name]
        oracle_scores.loc[period_mask] = scores_df.loc[period_mask, selected_obs].values

    # For pre-crisis period 1, use first selected observable
    first_crisis_start = pd.Timestamp(ALL_CRISES[crisis_names[0]]["start"])
    first_best = best_obs_history[crisis_names[0]]
    pre_mask = oracle_scores.index < first_crisis_start
    oracle_scores.loc[pre_mask] = scores_df.loc[pre_mask, first_best].values

    oracle_scores = oracle_scores.fillna(
        scores_df.fillna(0).mean(axis=1)  # fallback to uniform for any gaps
    )

    oracle_d_vals = []
    print(f"\n  {'Crisis':<20} {'Oracle d':>10} {'Uniform d':>10} {'Selected obs'}")
    for crisis_name in crisis_names:
        mask = build_crisis_mask(oracle_scores.index, crisis_name)
        aligned_mask = mask.reindex(oracle_scores.index, fill_value=False)

        c_scores = oracle_scores[aligned_mask].dropna().values
        n_scores = oracle_scores[~aligned_mask].dropna().values
        d_oracle, _, _ = compute_cohens_d_with_ci(c_scores, n_scores, n_bootstrap=500, seed=42)

        uniform = scores_df.fillna(0).mean(axis=1)
        u_c = uniform[aligned_mask].dropna().values
        u_n = uniform[~aligned_mask].dropna().values
        d_uni, _, _ = compute_cohens_d_with_ci(u_c, u_n, n_bootstrap=500, seed=42)

        selected = best_obs_history.get(crisis_name, "N/A")
        print(f"  {crisis_name:<20} {d_oracle:>10.3f} {d_uni:>10.3f}  {selected}")
        oracle_d_vals.append(d_oracle)

    median_oracle = float(np.nanmedian(oracle_d_vals))

    print(f"\nSummary:")
    print(f"  Strategy 1 (max z-score):            median d = {median_max:.3f}")
    print(f"  Strategy 2 (expanding oracle):       median d = {median_oracle:.3f}")

    keep_max = median_max > KEEP_THRESHOLD
    keep_oracle = median_oracle > KEEP_THRESHOLD

    print(f"\nResult:")
    print(f"  Max z-score fusion:  d={median_max:.3f} → {'KEEP' if keep_max else 'DISCARD'}")
    print(f"  Expanding oracle:    d={median_oracle:.3f} → {'KEEP' if keep_oracle else 'DISCARD'}")

    # Note about max strategy bias
    print(f"\nNote: max(z_score) inflates scores due to order statistics —")
    print(f"  selecting the maximum of {n_obs} z-scores naturally produces")
    print(f"  higher values in both crisis and normal periods. Check whether")
    print(f"  the d improvement over uniform is genuine or order-statistics artifact.")

    uniform_d_vals_q42 = []
    for crisis_name in crisis_names:
        uniform = scores_df.fillna(0).mean(axis=1)
        mask = build_crisis_mask(uniform.index, crisis_name)
        aligned_mask = mask.reindex(uniform.index, fill_value=False)
        u_c = uniform[aligned_mask].dropna().values
        u_n = uniform[~aligned_mask].dropna().values
        d_uni, _, _ = compute_cohens_d_with_ci(u_c, u_n, n_bootstrap=500, seed=42)
        uniform_d_vals_q42.append(d_uni)
    median_uniform_q42 = float(np.nanmedian(uniform_d_vals_q42))

    return {
        "question": "Q42",
        "method": "Causal Switching (max z-score + expanding oracle)",
        "median_d_max_zscore": median_max,
        "median_d_oracle_expanding": median_oracle,
        "median_d_uniform_baseline": median_uniform_q42,
        "per_crisis_max": {cn: float(d) for cn, d in zip(crisis_names, max_d_vals)},
        "per_crisis_oracle": {cn: float(d) for cn, d in zip(crisis_names, oracle_d_vals)},
        "oracle_selection": best_obs_history,
        "keep_max": bool(keep_max),
        "keep_oracle": bool(keep_oracle),
    }


# ---------------------------------------------------------------------------
# Base observable benchmarks
# ---------------------------------------------------------------------------

def run_base_benchmarks(scores_df: pd.DataFrame, crisis_names: list):
    """Report individual observable performance for reference."""
    print("\n" + "=" * 60)
    print("Base Observable Benchmarks (reference)")
    print("=" * 60)

    obs_names = list(scores_df.columns)
    print(f"\n  {'Observable':<30} " + " ".join(f"{cn[:10]:<12}" for cn in crisis_names) + " Median")

    benchmark_results = {}
    for obs_name in obs_names:
        obs_s = scores_df[obs_name].dropna()
        d_vals = []
        for crisis_name in crisis_names:
            mask = build_crisis_mask(obs_s.index, crisis_name)
            aligned = mask.reindex(obs_s.index, fill_value=False)
            oc = obs_s[aligned].values
            on_ = obs_s[~aligned].values
            if len(oc) >= 3 and len(on_) >= 5:
                od, _, _ = compute_cohens_d_with_ci(oc, on_, n_bootstrap=500, seed=42)
            else:
                od = np.nan
            d_vals.append(od)

        median_d = float(np.nanmedian(d_vals))
        benchmark_results[obs_name] = {"per_crisis": dict(zip(crisis_names, d_vals)), "median": median_d}

        row = f"  {obs_name:<30} " + " ".join(f"{(d if not np.isnan(d) else 0.0):<12.3f}" for d in d_vals)
        row += f" {median_d:.3f}"
        print(row)

    return benchmark_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Fusion Methods Empirical Test: Q40-Q42")
    print(f"Crises: {SMOKE_CRISES}")
    print(f"Observable: {SYMBOL}")
    print("=" * 60)

    # 1. Load data
    prices = load_spy_prices()

    # 2. Build features
    print("\nBuilding feature matrix...")
    X, dates = build_features(prices)
    print(f"  Feature matrix: {X.shape} over {len(dates)} dates")

    # 3. Fit and score base detectors
    print("\nFitting base detectors...")
    detectors = build_base_detectors()
    scores_df = fit_and_score_detectors(detectors, X, dates, verbose=True)
    print(f"  Score matrix: {scores_df.shape}")

    # 4. Base benchmarks
    benchmark_results = run_base_benchmarks(scores_df, SMOKE_CRISES)

    # 5. Q40: BMA
    result_q40 = run_q40(scores_df, SMOKE_CRISES)

    # 6. Q41: Stacking
    result_q41 = run_q41(scores_df, SMOKE_CRISES)

    # 7. Q42: Switching
    result_q42 = run_q42(scores_df, SMOKE_CRISES)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY: Q40-Q42 Fusion Methods")
    print("=" * 60)
    print(f"\n{'Method':<40} {'Median d':>10} {'Threshold':>12} {'Keep?':>8}")
    print("-" * 75)
    print(f"{'Uniform ensemble baseline':<40} {result_q40['median_d_uniform']:>10.3f} {KEEP_THRESHOLD:>12.1f} {'n/a':>8}")
    print(f"{'Q40: BMA (Brier-weighted)':<40} {result_q40['median_d_bma']:>10.3f} {KEEP_THRESHOLD:>12.1f} {'YES' if result_q40['keep'] else 'NO':>8}")
    q41_d = result_q41["median_d_stacked_oos"] if result_q41 else np.nan
    q41_keep = result_q41["keep"] if result_q41 else False
    print(f"{'Q41: Stacking (GB, OOS)':<40} {q41_d:>10.3f} {KEEP_THRESHOLD:>12.1f} {'YES' if q41_keep else 'NO':>8}")
    print(f"{'Q42: Max z-score':<40} {result_q42['median_d_max_zscore']:>10.3f} {KEEP_THRESHOLD:>12.1f} {'YES' if result_q42['keep_max'] else 'NO':>8}")
    print(f"{'Q42: Expanding oracle':<40} {result_q42['median_d_oracle_expanding']:>10.3f} {KEEP_THRESHOLD:>12.1f} {'YES' if result_q42['keep_oracle'] else 'NO':>8}")

    print("\n--- Interpretation ---")
    print("Q40 BMA: Does weighting observables by historical Brier score")
    print("  improve over uniform average? Limited data makes BIC-based BMA noisy.")
    print("Q41 Stacking: With only 4 crises, training set has 1-3 crises.")
    print("  Gradient boosting will memorize training patterns, expect high overfit.")
    print("Q42 Max z-score: Order statistics inflate max(z) in ALL periods —")
    print("  any d improvement vs uniform is likely due to selection bias.")
    print("Q42 Expanding oracle: Selects the historically best observable going")
    print("  forward. Tests whether past performance predicts future regime detection.")

    return {
        "benchmarks": benchmark_results,
        "q40": result_q40,
        "q41": result_q41,
        "q42": result_q42,
    }


if __name__ == "__main__":
    results = main()
