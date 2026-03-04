"""
Walk-forward evaluation with nested Optuna HPO at each window.

At each expanding window (2005→eval_year-1), re-optimizes hyperparameters
using ONLY crises available up to that point, then evaluates on the next year.
This eliminates ALL hyperparameter look-ahead bias.

Supports both QCML geometric detectors and classical baselines through the
same pipeline for an apples-to-apples comparison.

Usage:
    # QCML detectors only (original)
    caffeinate -i python experiments/walk_forward_hpo.py --detectors qcml

    # Classical baselines only
    caffeinate -i python experiments/walk_forward_hpo.py --detectors classical

    # All detectors (apples-to-apples comparison)
    caffeinate -i python experiments/walk_forward_hpo.py --detectors all

    # Specific detector(s)
    caffeinate -i python experiments/walk_forward_hpo.py --detectors berry,cusum,rf

    # Multi-scale Berry
    caffeinate -i python experiments/walk_forward_hpo.py --detectors multiscale_berry

    # Quick smoke test
    caffeinate -i python experiments/walk_forward_hpo.py --detectors all --quick
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

import optuna
from optuna.samplers import TPESampler

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
    MultiScaleBerryDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    BOCPDDetector,
    IsolationForestDetector,
    RandomForestRegimeDetector,
)
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXTENSION_DAYS = 10
CONSISTENCY_PENALTY = 0.3

# ============================================================================
# QCML Detector Registry
# ============================================================================

QCML_DETECTORS = {
    'berry': ('Berry Phase Rate', BerryPhaseRateDetector),
    'qfi': ('QFI Determinant', QFIDeterminantDetector),
    'mlf': ('Multi-Lag Fidelity', MultiLagFidelityDetector),
    'multiscale_berry': ('Multi-Scale Berry', MultiScaleBerryDetector),
}

OPERATOR_METHODS = {
    'berry': 'random',
    'qfi': 'pca_inspired',
    'mlf': 'pca_inspired',
    'multiscale_berry': 'random',
}

# Structural params fixed from MATHEMATICAL first principles (no data peeking).
# These are not empirical choices — they follow from the geometry of the method.
#
#   Berry normalization='sphere': Berry phase is defined on CP^n (projective
#     Hilbert space), which requires unit-norm states. Sphere normalization
#     maps features to the unit sphere, satisfying this mathematical constraint.
#
#   QFI qfi_mode='logdet': The log-determinant of the QFI matrix is the
#     standard volume measure in quantum estimation theory (Holevo bound,
#     Braunstein & Caves 1994). It is the canonical scalar summary of QFI.
#
# All other params (berry_aggregation, QFI normalization, MLF normalization)
# remain in the search space — their choice is empirical, not mathematical.
MATH_JUSTIFIED_PARAMS = {
    'berry': dict(normalization='sphere'),
    'qfi': dict(qfi_mode='logdet'),
    'mlf': {},  # no math-forced structural choices
    'multiscale_berry': dict(normalization='sphere'),
}

# Effective search space sizes (for coverage logging)
# Constrained = math-justified fixes only
# New params: operator_method(×2), expanding_refit_interval(×3), adaptive_z_window(×3),
#             n_pca reduced to [3,5,8](×3)
SEARCH_SPACE_SIZES = {
    # Berry: hilbert(3)×pca(3)×op(2)×refit(3)×az(3)×norm(1)×rw(3)×agg(3) = 1458
    'berry': {'constrained': 1458, 'full': 5832},
    # QFI: hilbert(3)×pca(3)×op(2)×refit(3)×az(3)×norm(4)×rw(3)×qfi(1) = 1944
    'qfi': {'constrained': 1944, 'full': 9720},
    # MLF: hilbert(3)×pca(3)×op(2)×refit(3)×az(3)×norm(4)×rw(3) = 1944
    'mlf': {'constrained': 1944, 'full': 1944},
    # Multi-Scale Berry: hilbert(3)×pca(3)×op(2)×refit(3)×az(3)×norm(1)×wp(4)×agg(3)×bagg(3)
    'multiscale_berry': {'constrained': 17496, 'full': 69984},
    'vol_z': {'constrained': 5, 'full': 5},
    'cusum': {'constrained': 12, 'full': 12},
    'hmm': {'constrained': 4, 'full': 4},
    'bocpd': {'constrained': 12, 'full': 12},
    'iforest': {'constrained': 12, 'full': 12},
    'rf': {'constrained': 36, 'full': 36},
}

# ============================================================================
# Classical Detector Registry
# ============================================================================

CLASSICAL_DETECTORS = {
    'vol_z': ('Rolling Vol Z', RollingVolatilityDetector),
    'cusum': ('CUSUM', CUSUMDetector),
    'hmm': ('HMM', HMMRegimeDetector),
    'bocpd': ('BOCPD', BOCPDDetector),
    'iforest': ('Isolation Forest', IsolationForestDetector),
    'rf': ('Random Forest', RandomForestRegimeDetector),
}

# Combined registry for summary/save
ALL_DETECTORS = {**QCML_DETECTORS, **CLASSICAL_DETECTORS}

# Convenience groups
DETECTOR_GROUPS = {
    'qcml': list(QCML_DETECTORS.keys()),
    'classical': list(CLASSICAL_DETECTORS.keys()),
    'all': list(ALL_DETECTORS.keys()),
}


# ============================================================================
# Crisis label generation (for supervised RF)
# ============================================================================

def generate_crisis_labels(dates, crisis_keys):
    """Generate binary crisis labels for supervised baselines.

    Args:
        dates: DatetimeIndex aligned with the feature matrix.
        crisis_keys: List of crisis keys to mark as positive.

    Returns:
        y: np.ndarray of shape (len(dates),) with 1=crisis, 0=normal.
    """
    y = np.zeros(len(dates), dtype=int)
    for ck in crisis_keys:
        if ck not in ALL_CRISES:
            continue
        ci = ALL_CRISES[ck]
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (dates >= cs) & (dates <= ce)
        y[mask] = 1
    return y


# ============================================================================
# Shared evaluation logic
# ============================================================================

def get_crisis_scores(scores, dates, crisis_key):
    """Extract crisis and normal scores for a given crisis."""
    ci = ALL_CRISES[crisis_key]
    cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)
    crisis_mask = (dates >= cs) & (dates <= ce)
    return scores[crisis_mask], scores[~crisis_mask]


def evaluate_detector_scores(scores, dates, crisis_keys):
    """Compute mean Cohen's d from pre-computed scores across given crises."""
    per_crisis = {}
    ds = []
    for ck in crisis_keys:
        try:
            crisis_s, normal_s = get_crisis_scores(scores, dates, ck)
            d, _, _ = compute_cohens_d_with_ci(crisis_s, normal_s, n_bootstrap=500)
            if not np.isnan(d):
                ds.append(d)
                per_crisis[ck] = float(d)
        except Exception:
            pass

    mean_d = float(np.mean(ds)) if ds else 0.0
    return mean_d, per_crisis


def evaluate_detector(detector_class, params, X, dates,
                      crisis_keys, crisis_labels=None):
    """Fit detector and compute mean Cohen's d across given crises.

    Args:
        detector_class: Detector class to instantiate.
        params: Dict of constructor kwargs.
        X: Feature matrix (enriched for most detectors, raw for RF).
        dates: DatetimeIndex aligned with X.
        crisis_keys: Crisis keys to evaluate.
        crisis_labels: Binary labels for supervised detectors (RF). If provided
            and the detector has fit_with_labels, uses that instead of fit().

    Returns:
        (mean_d, per_crisis_dict)
    """
    try:
        det = detector_class(**params)
        if crisis_labels is not None and hasattr(det, 'fit_with_labels'):
            det.fit_with_labels(X, crisis_labels)
        else:
            det.fit(X)
        scores = det.compute_regime_scores(X)
    except Exception as e:
        logger.debug(f"  Detector failed: {e}")
        return 0.0, {}

    # Drop NaN scores (e.g., from RF's lookback padding)
    valid = ~np.isnan(scores)
    scores = scores[valid]
    dates_valid = dates[valid] if hasattr(dates, '__getitem__') else dates

    return evaluate_detector_scores(scores, dates_valid, crisis_keys)


# ============================================================================
# QCML HPO Objective (original)
# ============================================================================

HP_STABILITY_WEIGHT = 0.1  # penalty per changed categorical HP

def create_qcml_objective(detector_key, detector_class, X_enriched,
                          dates_enriched, train_crises, constrained=True,
                          prev_best_params=None, frozen_structural=None):
    """Create Optuna objective for a QCML detector.

    Args:
        constrained: If True, fix math-justified structural params only
            (Berry sphere normalization, QFI logdet mode). All empirically-
            chosen params remain in the search space.
        prev_best_params: Best trial params from previous window. If provided,
            a stability penalty discourages HP thrashing across windows.
        frozen_structural: Dict of structural HPs to freeze (e.g.,
            {'hilbert_dim': 6, 'n_pca_components': 5}). Used after sufficient
            training windows to reduce search space.
    """
    default_operator_method = OPERATOR_METHODS[detector_key]
    fixed = MATH_JUSTIFIED_PARAMS.get(detector_key, {}) if constrained else {}
    frozen = frozen_structural or {}

    def objective(trial):
        if 'hilbert_dim' in frozen:
            hilbert_dim = frozen['hilbert_dim']
        else:
            hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 6, 8])

        if 'n_pca_components' in frozen:
            n_pca = frozen['n_pca_components']
        else:
            n_pca = trial.suggest_categorical('n_pca_components', [3, 5, 8])

        # Let Optuna discover that 'random' operators generalize better OOS
        operator_method = trial.suggest_categorical(
            'operator_method', ['random', 'pca_inspired']
        )

        if 'normalization' in fixed:
            normalization = fixed['normalization']
        else:
            normalization = trial.suggest_categorical(
                'normalization', ['sphere', 'none', 'soft', 'clip']
            )
        adaptive_epsilon = (normalization != 'sphere')

        n_features = X_enriched.shape[1]
        if n_pca > n_features:
            n_pca = n_features

        # Expanding refit: periodically refit scaler/PCA/operators on expanding data
        expanding_refit = trial.suggest_categorical(
            'expanding_refit_interval', ['none', '50', '100']
        )
        expanding_refit_interval = None if expanding_refit == 'none' else int(expanding_refit)

        # Adaptive z-scoring: rolling median/MAD instead of full-history mean/std
        adaptive_z = trial.suggest_categorical(
            'adaptive_z_window', ['none', '126', '252']
        )
        adaptive_z_window = None if adaptive_z == 'none' else int(adaptive_z)

        params = dict(
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca,
            operator_method=operator_method,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            expanding_refit_interval=expanding_refit_interval,
            adaptive_z_window=adaptive_z_window,
            seed=42,
        )

        # Detector-specific search params
        if detector_key == 'berry':
            params['rolling_window'] = trial.suggest_categorical(
                'rolling_window', [10, 15, 20]
            )
            # berry_aggregation is empirical — always in search space
            params['berry_aggregation'] = trial.suggest_categorical(
                'berry_aggregation', ['f01', 'frobenius', 'max']
            )
        elif detector_key == 'qfi':
            params['rolling_window'] = trial.suggest_categorical(
                'rolling_window', [10, 15, 20]
            )
            # normalization is empirical for QFI — already in search above
            if 'qfi_mode' in fixed:
                params['qfi_mode'] = fixed['qfi_mode']
            else:
                params['qfi_mode'] = trial.suggest_categorical(
                    'qfi_mode', ['logdet', 'trace', 'max_eig', 'condition', 'entropy']
                )
        elif detector_key == 'mlf':
            params['rolling_window'] = trial.suggest_categorical(
                'rolling_window', [10, 15, 20]
            )
        elif detector_key == 'multiscale_berry':
            windows_choice = trial.suggest_categorical(
                'windows_preset', ['short', 'medium', 'long', 'all']
            )
            windows_map = {
                'short': [3, 5, 10],
                'medium': [5, 10, 20],
                'long': [10, 20, 40],
                'all': [5, 10, 20, 40],
            }
            params['windows'] = windows_map[windows_choice]
            params['aggregation'] = trial.suggest_categorical(
                'aggregation', ['rms', 'max', 'mean']
            )
            # berry_aggregation is empirical — always in search space
            params['berry_aggregation'] = trial.suggest_categorical(
                'berry_aggregation', ['f01', 'frobenius', 'max']
            )

        mean_d, per_crisis = evaluate_detector(
            detector_class, params, X_enriched, dates_enriched, train_crises,
        )

        # Adaptive consistency penalty: ramp from 0→full based on crisis count
        n_eval = len(per_crisis)
        obj = mean_d
        if n_eval >= 2:
            std_d = float(np.std(list(per_crisis.values())))
            ramp = min(1.0, (n_eval - 1) / 3.0)
            obj -= CONSISTENCY_PENALTY * ramp * std_d

        # HP stability penalty: discourage configs that differ from previous window
        if prev_best_params is not None:
            categorical_keys = [
                k for k in trial.params
                if k not in ('hilbert_dim', 'n_pca_components')  # skip if frozen
                or k not in frozen
            ]
            n_changed = sum(
                1 for k in categorical_keys
                if k in prev_best_params and str(trial.params[k]) != str(prev_best_params.get(k))
            )
            n_total = max(len(categorical_keys), 1)
            obj -= HP_STABILITY_WEIGHT * n_changed / n_total

        return obj

    return objective


def extract_qcml_best_params(detector_key, best_trial, constrained=True,
                             frozen_structural=None):
    """Extract full parameter dict from Optuna best trial for QCML detector.

    When constrained=True, math-justified params not in trial.params are
    taken from MATH_JUSTIFIED_PARAMS (they were fixed, not suggested).

    Args:
        detector_key: QCML detector key (e.g. 'berry', 'qfi', 'mlf').
        best_trial: Optuna FrozenTrial with best params.
        constrained: Whether math-justified params were fixed.
        frozen_structural: Dict of structural HPs frozen after window 3
            (e.g. {'hilbert_dim': 6, 'n_pca_components': 3}). When a key
            is frozen, it won't be in best_trial.params.
    """
    fixed = MATH_JUSTIFIED_PARAMS.get(detector_key, {}) if constrained else {}
    frozen = frozen_structural or {}

    normalization = best_trial.params.get(
        'normalization', fixed.get('normalization', 'sphere')
    )
    operator_method = best_trial.params.get(
        'operator_method', OPERATOR_METHODS[detector_key]
    )

    # Expanding refit interval (stored as string in trial)
    expanding_refit = best_trial.params.get('expanding_refit_interval', 'none')
    expanding_refit_interval = None if expanding_refit == 'none' else int(expanding_refit)

    # Adaptive z-scoring window (stored as string in trial)
    adaptive_z = best_trial.params.get('adaptive_z_window', 'none')
    adaptive_z_window = None if adaptive_z == 'none' else int(adaptive_z)

    # Structural HPs: use frozen values when not in trial params
    hilbert_dim = best_trial.params.get(
        'hilbert_dim', frozen.get('hilbert_dim', 6)
    )
    n_pca_components = best_trial.params.get(
        'n_pca_components', frozen.get('n_pca_components', 5)
    )

    params = dict(
        hilbert_dim=hilbert_dim,
        n_pca_components=n_pca_components,
        operator_method=operator_method,
        normalization=normalization,
        adaptive_epsilon=(normalization != 'sphere'),
        expanding_refit_interval=expanding_refit_interval,
        adaptive_z_window=adaptive_z_window,
        seed=42,
    )

    if detector_key == 'multiscale_berry':
        windows_map = {
            'short': [3, 5, 10],
            'medium': [5, 10, 20],
            'long': [10, 20, 40],
            'all': [5, 10, 20, 40],
        }
        params['windows'] = windows_map[best_trial.params['windows_preset']]
        params['aggregation'] = best_trial.params['aggregation']
    else:
        params['rolling_window'] = best_trial.params.get('rolling_window', 20)

    # These are always suggested (empirical choices stay in search space)
    if 'berry_aggregation' in best_trial.params:
        params['berry_aggregation'] = best_trial.params['berry_aggregation']

    # qfi_mode: from trial if searched, else from math-justified fixed
    if 'qfi_mode' in best_trial.params:
        params['qfi_mode'] = best_trial.params['qfi_mode']
    elif 'qfi_mode' in fixed:
        params['qfi_mode'] = fixed['qfi_mode']

    return params


# ============================================================================
# Classical HPO Objective
# ============================================================================

def create_classical_objective(detector_key, detector_class, X_enriched,
                               dates_enriched, train_crises,
                               crisis_labels=None):
    """Create Optuna objective for a classical baseline detector.

    Args:
        detector_key: One of 'vol_z', 'cusum', 'hmm', 'bocpd', 'iforest', 'rf'.
        detector_class: Detector class to instantiate.
        X_enriched: Enriched feature matrix (training period only).
        dates_enriched: DatetimeIndex for training period.
        train_crises: Crisis keys available for training.
        crisis_labels: Binary labels for RF (aligned with X_enriched).
    """

    def objective(trial):
        if detector_key == 'vol_z':
            params = dict(
                vol_window=trial.suggest_categorical('vol_window', [5, 10, 15, 20, 30]),
            )
        elif detector_key == 'cusum':
            params = dict(
                k=trial.suggest_categorical('k', [0.25, 0.5, 1.0, 2.0]),
                burn_in=trial.suggest_categorical('burn_in', [30, 60, 90]),
            )
        elif detector_key == 'hmm':
            params = dict(
                n_states=trial.suggest_categorical('n_states', [2, 3]),
                covariance_type=trial.suggest_categorical(
                    'covariance_type', ['full', 'diag']
                ),
                seed=42,
            )
        elif detector_key == 'bocpd':
            params = dict(
                hazard_rate=trial.suggest_categorical(
                    'hazard_rate', [50.0, 100.0, 200.0, 500.0]
                ),
                min_expanding=trial.suggest_categorical(
                    'min_expanding', [10, 30, 60]
                ),
            )
        elif detector_key == 'iforest':
            params = dict(
                n_estimators=trial.suggest_categorical('n_estimators', [50, 100, 200]),
                contamination=trial.suggest_categorical(
                    'contamination', [0.01, 0.05, 0.1, 0.2]
                ),
                seed=42,
            )
        elif detector_key == 'rf':
            params = dict(
                n_estimators=trial.suggest_categorical('n_estimators', [100, 200, 500]),
                max_depth=trial.suggest_categorical('max_depth', [4, 6, 8, 10]),
                lookback=trial.suggest_categorical('lookback', [10, 20, 30]),
                seed=42,
            )
        else:
            raise ValueError(f"Unknown classical detector: {detector_key}")

        mean_d, per_crisis = evaluate_detector(
            detector_class, params, X_enriched, dates_enriched, train_crises,
            crisis_labels=crisis_labels,
        )

        # Adaptive consistency penalty: ramp from 0→full based on crisis count
        n_eval = len(per_crisis)
        if n_eval >= 2:
            std_d = float(np.std(list(per_crisis.values())))
            ramp = min(1.0, (n_eval - 1) / 3.0)
            return mean_d - CONSISTENCY_PENALTY * ramp * std_d
        return mean_d

    return objective


def extract_classical_best_params(detector_key, best_trial):
    """Extract full parameter dict from Optuna best trial for classical detector."""
    params = dict(best_trial.params)
    if detector_key in ('hmm', 'iforest', 'rf'):
        params['seed'] = 42
    # Convert floats that should stay as floats (hazard_rate, contamination, k)
    return params


# ============================================================================
# Crisis identification
# ============================================================================

def find_training_crises(eval_year):
    """Return crisis keys whose windows END before eval_year starts.

    Only includes post-2005 crises (our data starts 2005).
    """
    cutoff = pd.Timestamp(f'{eval_year}-01-01')
    matching = []
    for ck, ci in ALL_CRISES.items():
        year = int(ck[:4])
        if year < 2005:
            continue
        ce = pd.Timestamp(ci['end'])
        if ce < cutoff:
            matching.append(ck)
    return matching


def find_crises_in_year(year):
    """Return crisis keys whose windows overlap the given year."""
    start = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp(f'{year}-12-31')
    matching = []
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        if cs <= end and ce >= start:
            matching.append(ck)
    return matching


# ============================================================================
# Resolve which detectors to run
# ============================================================================

def resolve_detector_keys(detectors_arg):
    """Parse --detectors argument into a list of detector keys.

    Args:
        detectors_arg: Comma-separated string or group name.

    Returns:
        List of detector keys to run.
    """
    if detectors_arg in DETECTOR_GROUPS:
        return DETECTOR_GROUPS[detectors_arg]

    keys = [k.strip() for k in detectors_arg.split(',')]
    # Expand groups within comma-separated list
    resolved = []
    for k in keys:
        if k in DETECTOR_GROUPS:
            resolved.extend(DETECTOR_GROUPS[k])
        elif k in ALL_DETECTORS:
            resolved.append(k)
        else:
            raise ValueError(
                f"Unknown detector '{k}'. Valid: {list(ALL_DETECTORS.keys())} "
                f"or groups: {list(DETECTOR_GROUPS.keys())}"
            )
    # Deduplicate preserving order
    seen = set()
    return [k for k in resolved if not (k in seen or seen.add(k))]


# ============================================================================
# Main walk-forward HPO loop
# ============================================================================

def run_walk_forward_hpo(n_trials=100, quick=False, detector_keys=None,
                         mode='constrained'):
    """Run walk-forward evaluation with nested HPO at each window.

    For each expanding window:
      1. Identify training crises (ended before eval year)
      2. Run Optuna HPO on training crises (n_trials per detector)
      3. Fit detector with best HP on training data
      4. Evaluate on eval year crises

    Args:
        n_trials: Optuna trials per detector per window.
        quick: Use 3 windows and 25 trials.
        detector_keys: List of detector keys to run. Default: QCML only.
        mode: 'constrained' fixes structural params from ablation study,
              'full_search' uses original unconstrained search space.
    """
    constrained = (mode == 'constrained')

    if detector_keys is None:
        detector_keys = list(QCML_DETECTORS.keys())

    if quick:
        n_trials = min(n_trials, 25)

    # Determine trial counts per detector (HMM is slow, reduce if needed)
    def trials_for(det_key):
        if det_key == 'hmm' and n_trials > 50:
            return 50
        return n_trials

    logger.info("=" * 70)
    logger.info("WALK-FORWARD HPO (Nested Optimization)")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Detectors: {detector_keys}")
    logger.info(f"  Trials per detector per window: {n_trials}")
    logger.info(f"  Consistency penalty: {CONSISTENCY_PENALTY} (adaptive ramp)")
    if constrained:
        logger.info("  Math-justified fixed params (no data peeking):")
        for dk in detector_keys:
            fixed = MATH_JUSTIFIED_PARAMS.get(dk, {})
            if fixed:
                logger.info(f"    {dk}: {fixed}")
    # Log search space coverage
    for dk in detector_keys:
        if dk in SEARCH_SPACE_SIZES:
            space = SEARCH_SPACE_SIZES[dk][mode if mode in ('constrained', 'full') else 'full']
            dk_trials = trials_for(dk)
            coverage = min(100.0, dk_trials / space * 100)
            logger.info(f"  {dk}: space={space} configs, {dk_trials} trials → {coverage:.0f}% coverage")
    logger.info("=" * 70)

    # Fetch full data range once
    logger.info("\n[1] Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_full, dates_full = create_feature_matrix(prices_df)
    logger.info(f"  Full feature matrix: {X_full.shape}")
    logger.info(f"  Date range: {dates_full[0]} to {dates_full[-1]}")

    # Walk-forward windows
    if quick:
        eval_years = [2010, 2015, 2020]
    else:
        eval_years = list(range(2010, 2024))

    # Pre-compute which crises are available at each window
    for yr in eval_years:
        tc = find_training_crises(yr)
        ec = find_crises_in_year(yr)
        logger.info(f"  {yr}: train_crises={len(tc)} {tc}, eval_crises={ec or 'none'}")

    all_window_results = []
    all_hp_choices = {}
    prev_best_trial_params = {}  # det_key -> previous window's best trial.params
    frozen_structural = {}  # det_key -> frozen structural HPs after window 3
    total_start = time.time()

    for window_idx, eval_year in enumerate(eval_years):
        window_start = time.time()
        logger.info(f"\n{'=' * 70}")
        logger.info(f"WINDOW {window_idx + 1}/{len(eval_years)}: "
                     f"Train 2005-{eval_year - 1}, Eval {eval_year}")
        logger.info(f"{'=' * 70}")

        # Slice data: training = 2005 to eval_year-1
        train_end = pd.Timestamp(f'{eval_year - 1}-12-31')
        eval_start = pd.Timestamp(f'{eval_year}-01-01')
        eval_end = pd.Timestamp(f'{eval_year}-12-31')

        train_mask = dates_full <= train_end
        window_mask = dates_full <= eval_end

        X_train = X_full[train_mask]
        X_window = X_full[window_mask]
        dates_train = dates_full[train_mask]
        dates_window = dates_full[window_mask]

        if len(X_train) < 100:
            logger.info(f"  Skipping (insufficient training data: {len(X_train)})")
            continue

        # Build enriched features for training and full window
        X_train_enriched = BaseRegimeDetector.build_enriched_features(X_train, lookback=20)
        dates_train_enriched = dates_train[19:]
        X_window_enriched = BaseRegimeDetector.build_enriched_features(X_window, lookback=20)
        dates_window_enriched = dates_window[19:]

        eval_start_idx = np.searchsorted(dates_window_enriched, eval_start)

        # Identify crises
        train_crises = find_training_crises(eval_year)
        eval_crises = find_crises_in_year(eval_year)

        logger.info(f"  Training crises ({len(train_crises)}): {train_crises}")
        logger.info(f"  Eval crises: {eval_crises or 'none'}")
        logger.info(f"  Training features: {X_train_enriched.shape}")

        if len(train_crises) < 2:
            logger.info(f"  Skipping HPO (need >=2 training crises, have {len(train_crises)})")
            continue

        # Pre-compute crisis labels for supervised baselines
        # Enriched-aligned labels (for non-RF classical detectors that receive enriched data)
        train_crisis_labels = generate_crisis_labels(
            dates_train_enriched, train_crises
        )
        # Raw-aligned labels (for RF, which enriches internally)
        train_crisis_labels_raw = generate_crisis_labels(
            dates_train, train_crises
        )

        window_result = {
            'eval_year': eval_year,
            'n_train_crises': len(train_crises),
            'train_crises': train_crises,
            'eval_crises': eval_crises,
            'detectors': {},
        }

        # Run HPO for each detector
        for det_key in detector_keys:
            is_qcml = det_key in QCML_DETECTORS
            det_name, det_class = ALL_DETECTORS[det_key]
            det_n_trials = trials_for(det_key)
            hpo_start = time.time()
            logger.info(f"\n  --- HPO: {det_name} ({det_n_trials} trials) ---")

            # Create objective
            if is_qcml:
                objective = create_qcml_objective(
                    det_key, det_class, X_train_enriched,
                    dates_train_enriched, train_crises,
                    constrained=constrained,
                    prev_best_params=prev_best_trial_params.get(det_key),
                    frozen_structural=frozen_structural.get(det_key),
                )
            elif det_key == 'rf':
                # RF enriches internally — pass RAW data to avoid double-enrichment
                objective = create_classical_objective(
                    det_key, det_class, X_train,
                    dates_train, train_crises,
                    crisis_labels=train_crisis_labels_raw,
                )
            else:
                # Other classical baselines receive pre-enriched data
                # (consistent with regime_comparison.py)
                objective = create_classical_objective(
                    det_key, det_class, X_train_enriched,
                    dates_train_enriched, train_crises,
                )

            # Reduced TPE startup so more budget goes to informed search
            n_startup = min(5, det_n_trials // 3)
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42 + window_idx, n_startup_trials=n_startup),
                study_name=f'wf_hpo_{det_key}_{eval_year}',
            )

            study.optimize(objective, n_trials=det_n_trials, show_progress_bar=True)

            best = study.best_trial
            if is_qcml:
                best_params = extract_qcml_best_params(
                    det_key, best, constrained=constrained,
                    frozen_structural=frozen_structural.get(det_key),
                )
            else:
                best_params = extract_classical_best_params(det_key, best)
            hpo_elapsed = time.time() - hpo_start

            logger.info(f"    Best trial #{best.number}: obj={best.value:.4f} "
                         f"({hpo_elapsed:.1f}s)")
            logger.info(f"    Params: {best.params}")

            # Track HP choices and update stability state
            if det_key not in all_hp_choices:
                all_hp_choices[det_key] = []
            all_hp_choices[det_key].append({
                'year': eval_year,
                'params': {k: _jsonable(v) for k, v in best.params.items()},
                'train_obj': float(best.value),
            })

            # Store for next window's stability penalty
            if is_qcml:
                prev_best_trial_params[det_key] = dict(best.params)

            # After window 3, freeze structural HPs to most-frequent values
            if is_qcml and window_idx == 2 and det_key not in frozen_structural:
                choices = all_hp_choices[det_key]
                for structural_key in ('hilbert_dim', 'n_pca_components'):
                    vals = [c['params'].get(structural_key) for c in choices
                            if structural_key in c['params']]
                    if vals:
                        from collections import Counter
                        most_common_val = Counter(vals).most_common(1)[0][0]
                        if det_key not in frozen_structural:
                            frozen_structural[det_key] = {}
                        frozen_structural[det_key][structural_key] = most_common_val
                if det_key in frozen_structural:
                    logger.info(f"    Freezing structural HPs: {frozen_structural[det_key]}")

            # Evaluate on the eval year using best HP
            eval_d_values = {}
            try:
                det = det_class(**best_params)

                # Fit on training data and score full window
                if det_key == 'rf':
                    # RF enriches internally — use RAW data
                    det.fit_with_labels(X_train, train_crisis_labels_raw)
                    scores = det.compute_regime_scores(X_window)
                    # RF returns len(X_window) with lookback-1 NaN padding at front
                    # Scores align with dates_window (raw)
                    eval_start_idx_rf = np.searchsorted(dates_window, eval_start)
                    eval_scores = scores[eval_start_idx_rf:]
                    eval_dates = dates_window[eval_start_idx_rf:]
                else:
                    det.fit(X_train_enriched)
                    scores = det.compute_regime_scores(X_window_enriched)
                    eval_scores = scores[eval_start_idx:]
                    eval_dates = dates_window_enriched[eval_start_idx:]

                # Drop NaN scores (from RF lookback padding)
                valid = ~np.isnan(eval_scores)
                eval_scores_valid = eval_scores[valid]
                eval_dates_valid = eval_dates[valid]

                for ck in eval_crises:
                    ci = ALL_CRISES[ck]
                    cs = pd.Timestamp(ci['start'])
                    ce = pd.Timestamp(ci['end'])

                    crisis_mask = (
                        (eval_dates_valid >= cs) & (eval_dates_valid <= ce)
                    )
                    normal_mask = ~crisis_mask

                    crisis_scores = eval_scores_valid[crisis_mask]
                    normal_scores = eval_scores_valid[normal_mask]

                    if len(crisis_scores) >= 5 and len(normal_scores) >= 10:
                        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                            crisis_scores, normal_scores, n_bootstrap=5000
                        )
                        if not np.isnan(d):
                            eval_d_values[ck] = {
                                'd': float(d),
                                'ci_lo': float(ci_lo),
                                'ci_hi': float(ci_hi),
                            }
                            logger.info(
                                f"    {ck}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
                            )

            except Exception as e:
                logger.warning(f"    Eval failed: {e}")

            # Training performance with best HP for reference
            if det_key == 'rf':
                # RF enriches internally — use RAW data
                train_d, train_per_crisis = evaluate_detector(
                    det_class, best_params, X_train,
                    dates_train, train_crises,
                    crisis_labels=train_crisis_labels_raw,
                )
            else:
                train_labels = train_crisis_labels if det_key in CLASSICAL_DETECTORS else None
                train_d, train_per_crisis = evaluate_detector(
                    det_class, best_params, X_train_enriched,
                    dates_train_enriched, train_crises,
                    crisis_labels=train_labels,
                )

            window_result['detectors'][det_key] = {
                'name': det_name,
                'type': 'qcml' if is_qcml else 'classical',
                'best_params': {k: _jsonable(v) for k, v in best_params.items()},
                'train_objective': float(best.value),
                'train_mean_d': float(train_d),
                'train_per_crisis': train_per_crisis,
                'eval_per_crisis': eval_d_values,
                'eval_mean_d': (
                    float(np.mean([v['d'] for v in eval_d_values.values()]))
                    if eval_d_values else None
                ),
                'hpo_seconds': round(hpo_elapsed, 1),
                'n_trials': det_n_trials,
            }

        window_elapsed = time.time() - window_start
        logger.info(f"\n  Window {eval_year} complete in {window_elapsed / 60:.1f} min")
        all_window_results.append(window_result)

        # Save intermediate results after each window (crash recovery)
        _save_results(all_window_results, all_hp_choices, n_trials, symbols,
                      eval_years, detector_keys, mode=mode, intermediate=True)

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TOTAL TIME: {total_elapsed / 3600:.1f} hours")
    logger.info(f"{'=' * 70}")

    # Final summary
    _print_summary(all_window_results, all_hp_choices, detector_keys)

    # Save final results
    out_path = _save_results(all_window_results, all_hp_choices, n_trials,
                             symbols, eval_years, detector_keys,
                             mode=mode, intermediate=False)
    return out_path


def _jsonable(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def _print_summary(all_window_results, all_hp_choices, detector_keys):
    """Print summary of walk-forward HPO results."""
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD HPO SUMMARY")
    logger.info("=" * 70)

    # Collect and display results sorted by median d
    summary_rows = []

    for det_key in detector_keys:
        det_name = ALL_DETECTORS[det_key][0]
        all_eval_ds = []
        per_crisis_ds = {}

        for wr in all_window_results:
            if det_key in wr.get('detectors', {}):
                dr = wr['detectors'][det_key]
                for ck, v in dr.get('eval_per_crisis', {}).items():
                    all_eval_ds.append(v['d'])
                    if ck not in per_crisis_ds:
                        per_crisis_ds[ck] = []
                    per_crisis_ds[ck].append(v['d'])

        if all_eval_ds:
            median_d = np.median(all_eval_ds)
            mean_d = np.mean(all_eval_ds)
            summary_rows.append((median_d, det_key, det_name, all_eval_ds, per_crisis_ds))
        else:
            logger.info(f"\n  {det_name}: No eval crises encountered")

    # Sort by median d descending
    summary_rows.sort(key=lambda x: -x[0])

    logger.info(f"\n  {'Rank':<5} {'Method':<25} {'Median d':>10} {'Mean d':>10} {'N':>5}")
    logger.info(f"  {'-'*55}")
    for rank, (median_d, det_key, det_name, all_eval_ds, per_crisis_ds) in enumerate(summary_rows, 1):
        mean_d = np.mean(all_eval_ds)
        det_type = 'Q' if det_key in QCML_DETECTORS else 'C'
        logger.info(
            f"  {rank:<5} [{det_type}] {det_name:<21} {median_d:>10.3f} {mean_d:>10.3f} {len(all_eval_ds):>5}"
        )

    # Detailed per-crisis breakdown
    logger.info(f"\n  Per-crisis details:")
    for _, det_key, det_name, all_eval_ds, per_crisis_ds in summary_rows:
        logger.info(f"\n    {det_name}:")
        for ck in sorted(per_crisis_ds.keys()):
            ds = per_crisis_ds[ck]
            logger.info(f"      {ck}: d={np.mean(ds):.3f} (n={len(ds)})")

    # HP stability analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("HP STABILITY ACROSS WINDOWS")
    logger.info("=" * 70)
    for det_key, choices in all_hp_choices.items():
        det_name = ALL_DETECTORS[det_key][0]
        logger.info(f"\n  {det_name}:")
        if not choices:
            continue

        param_counts = {}
        for c in choices:
            for k, v in c['params'].items():
                if k not in param_counts:
                    param_counts[k] = {}
                v_str = str(v)
                param_counts[k][v_str] = param_counts[k].get(v_str, 0) + 1

        for param_name, counts in param_counts.items():
            total = sum(counts.values())
            most_common = max(counts, key=counts.get)
            pct = counts[most_common] / total * 100
            logger.info(
                f"    {param_name}: most_common={most_common} ({pct:.0f}%), "
                f"dist={dict(sorted(counts.items(), key=lambda x: -x[1]))}"
            )


def _save_results(all_window_results, all_hp_choices, n_trials, symbols,
                  eval_years, detector_keys, mode='constrained', intermediate=False):
    """Save results to JSON."""
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'walk_forward_hpo'
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = '_intermediate' if intermediate else ''
    out_path = out_dir / f'wf_hpo_{ts}{suffix}.json'

    # Compute aggregate stats
    aggregate = {}
    for det_key in detector_keys:
        det_name = ALL_DETECTORS[det_key][0]
        all_eval_ds = []
        all_train_ds = []
        for wr in all_window_results:
            if det_key in wr.get('detectors', {}):
                dr = wr['detectors'][det_key]
                for v in dr.get('eval_per_crisis', {}).values():
                    all_eval_ds.append(v['d'])
                if dr.get('train_mean_d') is not None:
                    all_train_ds.append(dr['train_mean_d'])

        aggregate[det_key] = {
            'name': det_name,
            'type': 'qcml' if det_key in QCML_DETECTORS else 'classical',
            'n_eval_observations': len(all_eval_ds),
            'eval_median_d': float(np.median(all_eval_ds)) if all_eval_ds else None,
            'eval_mean_d': float(np.mean(all_eval_ds)) if all_eval_ds else None,
            'eval_std_d': float(np.std(all_eval_ds)) if all_eval_ds else None,
            'train_median_d': float(np.median(all_train_ds)) if all_train_ds else None,
            'train_mean_d': float(np.mean(all_train_ds)) if all_train_ds else None,
            'overfitting_gap': (
                float(np.median(all_train_ds)) - float(np.median(all_eval_ds))
                if all_train_ds and all_eval_ds else None
            ),
        }

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials_per_detector_per_window': n_trials,
            'consistency_penalty': CONSISTENCY_PENALTY,
            'consistency_penalty_mode': 'adaptive_ramp',
            'mode': mode,
            'symbols': symbols,
            'eval_years': eval_years,
            'detector_keys': detector_keys,
        },
        'aggregate': aggregate,
        'hp_choices': all_hp_choices,
        'windows': all_window_results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_jsonable)

    if not intermediate:
        logger.info(f"\nResults saved to {out_path}")
    return out_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward evaluation with nested Optuna HPO',
    )
    parser.add_argument(
        '--n-trials', type=int, default=100,
        help='Optuna trials per detector per window (default: 100)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run: 3 windows, 25 trials',
    )
    parser.add_argument(
        '--detectors', type=str, default='qcml',
        help=(
            'Detectors to run. Groups: qcml, classical, all. '
            'Or comma-separated keys: berry,cusum,rf,multiscale_berry. '
            'Default: qcml'
        ),
    )
    parser.add_argument(
        '--mode', type=str, default='constrained',
        choices=['constrained', 'full_search'],
        help=(
            'Search mode. "constrained" fixes structural params (normalization, '
            'aggregation) to validated values, reducing search space ~10x. '
            '"full_search" uses original unconstrained space. Default: constrained'
        ),
    )
    # Legacy flags for backward compatibility
    parser.add_argument('--classical-only', action='store_true',
                        help='Shortcut for --detectors classical')
    parser.add_argument('--all', action='store_true',
                        help='Shortcut for --detectors all')
    args = parser.parse_args()

    # Resolve detector selection
    if args.all:
        detectors_arg = 'all'
    elif args.classical_only:
        detectors_arg = 'classical'
    else:
        detectors_arg = args.detectors

    detector_keys = resolve_detector_keys(detectors_arg)

    run_walk_forward_hpo(
        n_trials=args.n_trials,
        quick=args.quick,
        detector_keys=detector_keys,
        mode=args.mode,
    )


if __name__ == '__main__':
    main()
