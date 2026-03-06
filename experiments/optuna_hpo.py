"""
Optuna-based hyperparameter optimization for QCML regime detectors.

Replaces grid search with Bayesian optimization via Optuna. Each trial
evaluates a detector configuration across all crises using median Cohen's d
as the objective.

Usage:
    python experiments/optuna_hpo.py                     # Default: 50 trials
    python experiments/optuna_hpo.py --n-trials 100      # More trials
    python experiments/optuna_hpo.py --method "Berry Phase Rate"  # Single method
    python experiments/optuna_hpo.py --resume             # Resume previous study

Outputs:
    experiments/outputs/hpo/optuna_{method}_{timestamp}.json
    optuna-storage.db (SQLite, for resumable studies)
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
    SpectralGapDetector,
    SpeedLimitRatioDetector,
    DimensionalityCollapseDetector,
    SectionalCurvatureDetector,
    GeodesicVelocityDetector,
    SpectralEntropyDetector,
    HamiltonianSensitivityDetector,
    ReducedPurityDetector,
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

STORAGE_PATH = ROOT / 'optuna-storage.db'

# Crises to use for optimization (post-2005 only, skip pre-SPY)
OPT_CRISES = {
    k: v for k, v in ALL_CRISES.items()
    if int(k.split('_')[0]) >= 2005
}


# =============================================================================
# Search Space Definitions
# =============================================================================

SEARCH_SPACES = {
    'Berry Phase Rate': {
        'class': BerryPhaseRateDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='random',
            seed=42,
            normalization=trial.suggest_categorical('normalization', ['sphere', 'soft']),
            berry_aggregation=trial.suggest_categorical(
                'berry_aggregation', ['f01', 'trace']
            ),
        ),
    },
    'QFI Determinant': {
        'class': QFIDeterminantDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 12),
            n_pca_components=trial.suggest_int('n_pca_components', 8, 20),
            rolling_window=trial.suggest_int('rolling_window', 15, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization=trial.suggest_categorical('normalization', ['soft', 'sphere']),
            qfi_mode=trial.suggest_categorical('qfi_mode', ['logdet', 'trace', 'det']),
            adaptive_epsilon=True,
        ),
    },
    'Multi-Lag Fidelity': {
        'class': MultiLagFidelityDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 3, 8),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization='sphere',
        ),
    },
    'Spectral Gap': {
        'class': SpectralGapDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization='soft',
            adaptive_epsilon=True,
        ),
    },
    'Speed Limit Ratio': {
        'class': SpeedLimitRatioDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='random',
            seed=42,
            normalization=trial.suggest_categorical('normalization', ['sphere', 'soft']),
        ),
    },
    'Dim. Collapse': {
        'class': DimensionalityCollapseDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization='soft',
        ),
    },
    'Spectral Entropy': {
        'class': SpectralEntropyDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 12),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization='soft',
        ),
    },
    'Hamiltonian Sensitivity': {
        'class': HamiltonianSensitivityDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='random',
            seed=42,
            normalization='sphere',
        ),
    },
    'Reduced Purity': {
        'class': ReducedPurityDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 12),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization='soft',
        ),
    },
    'Geodesic Velocity': {
        'class': GeodesicVelocityDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='random',
            seed=42,
            normalization='sphere',
        ),
    },
    'Sect. Curv. Sign': {
        'class': SectionalCurvatureDetector,
        'params': lambda trial: dict(
            hilbert_dim=trial.suggest_int('hilbert_dim', 4, 10),
            n_pca_components=trial.suggest_int('n_pca_components', 6, 15),
            rolling_window=trial.suggest_int('rolling_window', 10, 30),
            operator_method='pca_inspired',
            seed=42,
            normalization='soft',
            scoring_mode='neg_fraction',
        ),
    },
}


# =============================================================================
# Objective Function
# =============================================================================

def create_objective(method_name, X, dates, crises, n_bootstrap=1000):
    """Create an Optuna objective function for a given method.

    Args:
        method_name: Key into SEARCH_SPACES.
        X: Feature matrix (T, d).
        dates: DatetimeIndex aligned with X.
        crises: Dict of crisis definitions.
        n_bootstrap: Bootstrap iterations for Cohen's d CI.

    Returns:
        Callable objective for Optuna.
    """
    config = SEARCH_SPACES[method_name]
    detector_class = config['class']
    param_fn = config['params']

    def objective(trial):
        params = param_fn(trial)

        d_values = []
        for ck, crisis in crises.items():
            c_start = pd.Timestamp(crisis['start'])
            c_end = pd.Timestamp(crisis['end'])

            crisis_mask = (dates >= c_start) & (dates <= c_end)
            if crisis_mask.sum() < 5:
                continue

            # Causal fit: only data before crisis
            fit_end_idx = np.searchsorted(dates, c_start)
            if fit_end_idx < 60:
                continue

            normal_mask = ~crisis_mask & (np.arange(len(dates)) < fit_end_idx)

            try:
                det = detector_class(**params, causal_fit_length=fit_end_idx)
                det.fit(X)
                scores = det.compute_regime_scores(X)

                d, _, _ = compute_cohens_d_with_ci(
                    scores[crisis_mask], scores[normal_mask],
                    n_bootstrap=n_bootstrap,
                )

                if np.isfinite(d):
                    d_values.append(d)
            except Exception:
                continue

            # Report intermediate value for pruning
            if d_values:
                trial.report(np.median(d_values), len(d_values))
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not d_values:
            return 0.0

        return float(np.median(d_values))

    return objective


# =============================================================================
# Main
# =============================================================================

def run_hpo(method_name, n_trials=50, resume=False, n_bootstrap=1000):
    """Run Optuna HPO for a single method.

    Args:
        method_name: Key into SEARCH_SPACES.
        n_trials: Number of Optuna trials.
        resume: If True, resume from existing study in SQLite.
        n_bootstrap: Bootstrap iterations for Cohen's d.

    Returns:
        Dict with best params and trial history.
    """
    import pandas as pd

    logger.info(f"Loading data for HPO...")
    raw = fetch_data(['SPY', 'DIA'], '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)

    study_name = f"hpo_{method_name.replace(' ', '_').lower()}"
    storage = f"sqlite:///{STORAGE_PATH}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=resume,
    )

    objective = create_objective(method_name, X, dates, OPT_CRISES, n_bootstrap)

    logger.info(f"Starting HPO for '{method_name}' ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"  Value (median d): {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_params}")

    # Save results
    out_dir = ROOT / 'experiments' / 'outputs' / 'hpo'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'optuna_{study_name}_{ts}.json'

    output = {
        'method': method_name,
        'n_trials': n_trials,
        'n_bootstrap': n_bootstrap,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ],
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Optuna HPO for QCML detectors')
    parser.add_argument(
        '--method', type=str, default=None,
        help=f'Method to optimize. Options: {list(SEARCH_SPACES.keys())}. '
             'Default: optimize all methods.',
    )
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials (default: 50)')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Bootstrap iterations')
    parser.add_argument('--resume', action='store_true', help='Resume from existing study')
    args = parser.parse_args()

    methods = [args.method] if args.method else list(SEARCH_SPACES.keys())

    for method in methods:
        if method not in SEARCH_SPACES:
            logger.error(f"Unknown method: {method}")
            logger.info(f"Available: {list(SEARCH_SPACES.keys())}")
            sys.exit(1)

        run_hpo(method, n_trials=args.n_trials, resume=args.resume,
                n_bootstrap=args.n_bootstrap)


if __name__ == '__main__':
    main()
