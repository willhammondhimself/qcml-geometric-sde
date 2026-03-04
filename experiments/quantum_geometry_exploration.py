"""
Quantum Geometry Exploration Sprint — Novel Research Directions

Prototypes 11 research directions using the QCML geometric framework.
Each exploration computes a novel geometric observable and evaluates it
against known crises using Cohen's d (crisis vs normal periods).

Usage:
    python experiments/quantum_geometry_exploration.py          # Full run (~45 min)
    python experiments/quantum_geometry_exploration.py --quick  # Phase 1 only (~5 min)

Output:
    experiments/outputs/quantum_geometry_exploration/
        summary_table.txt
        exploration_*.png
        results.json
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci, welch_t_test
from experiments.baselines import CUSUMDetector, RandomForestRegimeDetector
from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

OUTPUT_DIR = ROOT / 'experiments' / 'outputs' / 'quantum_geometry_exploration'

# Publication-quality figure defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Major crises for focused analysis (enough data, clear signal)
FOCUS_CRISES = [
    '2008_gfc', '2011_euro', '2015_china', '2018_volmageddon',
    '2020_covid', '2022_rates', '2023_svb', '2024_carry',
]

N_BOOTSTRAP = 2000  # Reduced for exploration speed; production uses 10000
MIN_EXPANDING = 60  # Minimum samples before z-scoring starts


# =============================================================================
# Shared Infrastructure
# =============================================================================

def expanding_zscore(signal, min_expanding=MIN_EXPANDING, signed=False):
    """Convert raw signal to expanding-window z-scores (causal, no lookahead).

    Args:
        signal: 1-D array (may contain NaN).
        min_expanding: Minimum samples before z-scoring starts.
        signed: If False (default), return absolute z-scores. If True, return
            signed z-scores (positive = above historical mean).

    Returns:
        z: 1-D array of z-scores, NaN before min_expanding.
    """
    T = len(signal)
    z = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = signal[:t]
        past = past[~np.isnan(past)]
        if len(past) < 10:
            continue
        mu = np.mean(past)
        sigma = np.std(past, ddof=1)
        if sigma > 1e-12 and not np.isnan(signal[t]):
            raw = (signal[t] - mu) / sigma
            z[t] = raw if signed else abs(raw)
    return z

def build_crisis_masks(dates, crises=None, pad_days=10):
    """Build per-crisis and aggregate crisis/normal masks.

    Args:
        dates: DatetimeIndex aligned with score arrays.
        crises: Dict of crisis definitions (default ALL_CRISES).
        pad_days: Days to extend each crisis window.

    Returns:
        crisis_masks: Dict[str, bool array] per crisis.
        any_crisis: Bool array, True if any crisis active.
    """
    if crises is None:
        crises = ALL_CRISES
    crisis_masks = {}
    any_crisis = np.zeros(len(dates), dtype=bool)
    for ck, ci in crises.items():
        cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=pad_days)
        ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=pad_days)
        mask = (dates >= cs) & (dates <= ce)
        crisis_masks[ck] = mask
        any_crisis |= mask
    return crisis_masks, any_crisis


def evaluate_signal(scores, dates, crisis_masks, any_crisis):
    """Compute aggregate and per-crisis Cohen's d for a signal.

    Returns:
        dict with 'aggregate_d', 'aggregate_ci', 'per_crisis', 'p_value'.
    """
    valid = ~np.isnan(scores)
    crisis_scores = scores[valid & any_crisis]
    normal_scores = scores[valid & ~any_crisis]

    if len(crisis_scores) < 5 or len(normal_scores) < 10:
        return {'aggregate_d': np.nan, 'aggregate_ci': (np.nan, np.nan),
                'per_crisis': {}, 'p_value': np.nan}

    d, ci_lo, ci_hi = compute_cohens_d_with_ci(
        crisis_scores, normal_scores, n_bootstrap=N_BOOTSTRAP
    )
    _, p_val = welch_t_test(crisis_scores, normal_scores)

    per_crisis = {}
    for ck, mask in crisis_masks.items():
        c = scores[valid & mask]
        n = scores[valid & ~any_crisis]
        if len(c) >= 3 and len(n) >= 10:
            dk, _, _ = compute_cohens_d_with_ci(c, n, n_bootstrap=500)
            per_crisis[ck] = round(float(dk), 3) if not np.isnan(dk) else None
        else:
            per_crisis[ck] = None

    return {
        'aggregate_d': round(float(d), 3),
        'aggregate_ci': (round(float(ci_lo), 3), round(float(ci_hi), 3)),
        'per_crisis': per_crisis,
        'p_value': float(p_val),
    }


def plot_signal_with_crises(signal, dates, crisis_masks, title, ylabel, filepath,
                            signal2=None, label1='Signal', label2=None):
    """Standard 2-panel figure: time series + crisis vs normal distribution."""
    valid = ~np.isnan(signal)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: Time series with crisis shading
    ax = axes[0]
    ax.plot(dates[valid], signal[valid], linewidth=0.5, color='#2c3e50', label=label1)
    if signal2 is not None:
        valid2 = ~np.isnan(signal2)
        ax.plot(dates[valid2], signal2[valid2], linewidth=0.5, color='#e74c3c',
                alpha=0.7, label=label2)
        ax.legend(fontsize=8)
    for ck, mask in crisis_masks.items():
        if mask.any():
            crisis_dates = dates[mask]
            ax.axvspan(crisis_dates.min(), crisis_dates.max(),
                       alpha=0.15, color='red', zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Distribution comparison
    ax2 = axes[1]
    any_crisis = np.zeros(len(dates), dtype=bool)
    for mask in crisis_masks.values():
        any_crisis |= mask
    crisis_vals = signal[valid & any_crisis]
    normal_vals = signal[valid & ~any_crisis]
    if len(crisis_vals) > 0 and len(normal_vals) > 0:
        bins = np.linspace(
            np.percentile(signal[valid], 1),
            np.percentile(signal[valid], 99),
            50
        )
        ax2.hist(normal_vals, bins=bins, alpha=0.6, density=True,
                 color='#3498db', label='Normal')
        ax2.hist(crisis_vals, bins=bins, alpha=0.6, density=True,
                 color='#e74c3c', label='Crisis')
        ax2.legend(fontsize=8)
    ax2.set_xlabel(ylabel)
    ax2.set_ylabel('Density')

    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    logger.info(f"  Saved {filepath.name}")


# =============================================================================
# Exploration A: Fubini-Study Velocity
# =============================================================================

def exploration_a_fs_velocity(geom, X_norm, dates, crisis_masks, any_crisis):
    """Fubini-Study velocity: v(t) = d_FS(psi(t), psi(t+1)), smoothed and z-scored."""
    logger.info("Exploration A: Fubini-Study Velocity")
    T = len(X_norm)
    velocity_raw = np.full(T, np.nan)
    for t in range(T - 1):
        velocity_raw[t + 1] = geom.quantum_distance(X_norm[t], X_norm[t + 1])

    # Smooth with 20-day rolling mean, then z-score
    velocity_smooth = pd.Series(velocity_raw).rolling(20, min_periods=1).mean().values
    velocity_z = expanding_zscore(velocity_smooth)

    result = evaluate_signal(velocity_z, dates, crisis_masks, any_crisis)
    result['name'] = 'A: FS Velocity'
    result['description'] = 'Fubini-Study velocity z-score (smoothed 20d)'

    plot_signal_with_crises(
        velocity_z, dates, crisis_masks,
        'Exploration A: Fubini-Study Velocity (z-scored)',
        r'$|z|$ of $v(t) = d_{FS}(\psi_t, \psi_{t+1})$',
        OUTPUT_DIR / 'exploration_A_fs_velocity.png',
    )
    return result, velocity_raw, velocity_smooth  # Return raw + smoothed for composite


# =============================================================================
# Exploration B: Quantum Speed Limit Ratio
# =============================================================================

def exploration_b_speed_limit(geom, X_norm, dates, crisis_masks, any_crisis,
                              velocity=None):
    """Quantum speed limit ratio: r(t) = v(t) / Delta(t).

    The spectral gap Delta = E_1 - E_0 sets the maximum evolution speed
    (Landau-Zener framework). When r -> 1, the system is evolving at its
    maximum possible speed — a hallmark of diabatic (sudden) transitions.

    r(t) = d_FS(psi_t, psi_{t+1}) / (E_1 - E_0)(t)
    """
    logger.info("Exploration B: Quantum Speed Limit Ratio (spectral gap)")
    T = len(X_norm)

    if velocity is None:
        velocity = np.full(T, np.nan)
        for t in range(T - 1):
            velocity[t + 1] = geom.quantum_distance(X_norm[t], X_norm[t + 1])

    # Compute spectral gap at each point
    gaps = np.full(T, np.nan)
    for t in range(T):
        gaps[t] = geom.spectral_gap(X_norm[t])

    # Speed limit ratio
    ratio = np.full(T, np.nan)
    valid = (~np.isnan(velocity)) & (gaps > 1e-6)
    ratio[valid] = velocity[valid] / gaps[valid]

    # Smooth and z-score
    ratio_smooth = pd.Series(ratio).rolling(20, min_periods=1).mean().values
    ratio_z = expanding_zscore(ratio_smooth)

    result = evaluate_signal(ratio_z, dates, crisis_masks, any_crisis)
    result['name'] = 'B: Speed Limit Ratio'
    result['description'] = 'v(t)/Delta(t) z-scored (Landau-Zener speed limit)'

    # Extra: fraction of crisis days in top quintile of raw ratio
    valid_r = ratio[~np.isnan(ratio)]
    if len(valid_r) > 0:
        p80 = np.percentile(valid_r, 80)
        crisis_r = ratio[(~np.isnan(ratio)) & any_crisis]
        if len(crisis_r) > 0:
            result['crisis_high_speed_frac'] = round(float(np.mean(crisis_r > p80)), 3)

    plot_signal_with_crises(
        ratio_z, dates, crisis_masks,
        'Exploration B: Speed Limit Ratio v/Delta (z-scored)',
        r'$|z|$ of $r(t) = v / \Delta$',
        OUTPUT_DIR / 'exploration_B_speed_limit.png',
    )
    return result, ratio_smooth


# =============================================================================
# Exploration C: Dimensionality Collapse
# =============================================================================

def exploration_c_dim_collapse(geom, X_norm, dates, crisis_masks, any_crisis,
                                subsample=5):
    """Inverse participation ratio of quantum metric eigenvalues.

    IPR = lambda_max / tr(g) — measures dimensionality collapse.
    During crises, one eigenvalue dominates (IPR -> 1 = everyone running same direction).
    Z-scored for detection.
    """
    logger.info("Exploration C: Dimensionality Collapse (metric eigenvalue spectrum)")
    T = len(X_norm)
    ipr_raw = np.full(T, np.nan)

    for t in range(0, T, subsample):
        g = geom.quantum_metric(X_norm[t])
        eigvals = np.linalg.eigvalsh(g)
        eigvals = np.maximum(eigvals, 0)
        total = np.sum(eigvals)
        if total > 1e-15:
            # Inverse participation ratio: lambda_max / trace
            ipr_raw[t] = eigvals[-1] / total

    # Interpolate for missing points
    known = ~np.isnan(ipr_raw)
    if known.sum() > 2:
        ipr_raw = pd.Series(ipr_raw).interpolate(method='linear').values

    # Smooth and z-score (higher IPR = more collapsed = more anomalous)
    ipr_smooth = pd.Series(ipr_raw).rolling(20, min_periods=1).mean().values
    ipr_z = expanding_zscore(ipr_smooth)

    result = evaluate_signal(ipr_z, dates, crisis_masks, any_crisis)
    result['name'] = 'C: Dimensionality Collapse'
    result['description'] = 'Metric tensor IPR (lambda_max/trace) z-scored'

    plot_signal_with_crises(
        ipr_z, dates, crisis_masks,
        'Exploration C: Dimensionality Collapse (IPR z-scored)',
        r'$|z|$ of $\lambda_{max} / \mathrm{tr}(g)$',
        OUTPUT_DIR / 'exploration_C_dim_collapse.png',
    )
    return result


# =============================================================================
# Exploration D: Entanglement Entropy
# =============================================================================

def exploration_d_entanglement(X_enriched, dates, crisis_masks, any_crisis,
                                scaler, pca_full, subsample=3):
    """Von Neumann entropy of reduced density matrix (bipartite entanglement)."""
    logger.info("Exploration D: Entanglement Entropy")

    # Use hilbert_dim=4 for clean 2x2 bipartition
    n_pca = min(4, X_enriched.shape[1])
    pca4 = PCA(n_components=n_pca)
    X_scaled = scaler.transform(X_enriched)
    pca4.fit(X_scaled)
    X_pca4 = pca4.transform(X_scaled)
    norms = np.linalg.norm(X_pca4, axis=1, keepdims=True)
    X_norm4 = X_pca4 / (norms + 1e-8)

    geom4 = QCMLGeometry(n_features=n_pca, hilbert_dim=4)
    np.random.seed(42)
    geom4.fit_operators(X_norm4, method='pca_inspired')

    T = len(X_norm4)
    entropy = np.full(T, np.nan)

    for t in range(0, T, subsample):
        psi = geom4.quasi_coherent_state(X_norm4[t])
        # Reshape to 2x2 (2-qubit bipartition)
        psi_mat = psi.reshape(2, 2)
        # Reduced density matrix: rho_A = Tr_B(|psi><psi|)
        rho_a = psi_mat @ psi_mat.conj().T
        # Von Neumann entropy: S = -Tr(rho log rho)
        eigvals = np.linalg.eigvalsh(rho_a)
        eigvals = eigvals[eigvals > 1e-15]
        entropy[t] = -np.sum(eigvals * np.log(eigvals))

    known = ~np.isnan(entropy)
    if known.sum() > 2:
        entropy = pd.Series(entropy).interpolate(method='linear').values

    result = evaluate_signal(entropy, dates, crisis_masks, any_crisis)
    result['name'] = 'D: Entanglement Entropy'
    result['description'] = 'Von Neumann entropy of reduced density matrix (2-qubit)'

    plot_signal_with_crises(
        entropy, dates, crisis_masks,
        'Exploration D: Entanglement Entropy (Von Neumann)',
        r'$S(\rho_A) = -\mathrm{Tr}(\rho_A \log \rho_A)$',
        OUTPUT_DIR / 'exploration_D_entanglement.png',
    )
    return result


# =============================================================================
# Exploration E: Lead Time Analysis
# =============================================================================

def exploration_e_lead_time(X_enriched, dates, crisis_masks, any_crisis):
    """Lead time: how many days before crisis start does each signal first alarm?"""
    logger.info("Exploration E: Lead Time Analysis")

    # Fit detectors
    detectors = {}

    berry = BerryPhaseRateDetector(hilbert_dim=8, n_pca_components=15, seed=42)
    berry.fit(X_enriched)
    detectors['Berry Phase Rate'] = berry.compute_regime_scores(X_enriched)

    qfi = QFIDeterminantDetector(hilbert_dim=8, n_pca_components=15, seed=42)
    qfi.fit(X_enriched)
    detectors['QFI Determinant'] = qfi.compute_regime_scores(X_enriched)

    cusum = CUSUMDetector()
    cusum.fit(X_enriched)
    detectors['CUSUM'] = cusum.compute_regime_scores(X_enriched)

    # RF baseline (supervised, needs labels)
    y_labels = any_crisis.astype(int)
    rf = RandomForestRegimeDetector(seed=42)
    rf.fit_with_labels(X_enriched[:len(dates)], y_labels)
    detectors['Random Forest'] = rf.compute_regime_scores(X_enriched)

    lead_times = {name: {} for name in detectors}

    for ck in FOCUS_CRISES:
        if ck not in ALL_CRISES:
            continue
        crisis_start = pd.Timestamp(ALL_CRISES[ck]['start'])

        for det_name, scores in detectors.items():
            valid = ~np.isnan(scores)
            if not valid.any():
                continue
            # Z-score the signal
            s = scores[valid]
            mu, sigma = np.mean(s), np.std(s, ddof=1)
            if sigma < 1e-12:
                continue
            z = (scores - mu) / sigma

            # Look in 90-day window before crisis start
            pre_window = (dates >= crisis_start - pd.Timedelta(days=90)) & \
                         (dates < crisis_start) & (~np.isnan(z))
            if not pre_window.any():
                continue

            alarm_indices = np.where(pre_window & (z > 2))[0]
            if len(alarm_indices) > 0:
                first_alarm_date = dates[alarm_indices[0]]
                lead_days = (crisis_start - first_alarm_date).days
                lead_times[det_name][ck] = lead_days
            else:
                lead_times[det_name][ck] = None

    result = {
        'name': 'E: Lead Time Analysis',
        'description': 'Days before crisis start that signal first exceeds z=2',
        'lead_times': lead_times,
    }

    # Compute median lead times
    medians = {}
    for det_name, lt in lead_times.items():
        vals = [v for v in lt.values() if v is not None]
        medians[det_name] = float(np.median(vals)) if vals else None
    result['median_lead_times'] = medians

    # Figure: bar chart of median lead times
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(medians.keys())
    vals = [medians[n] if medians[n] is not None else 0 for n in names]
    colors = ['#e74c3c' if 'Berry' in n or 'QFI' in n else '#3498db' for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel('Median Lead Time (days)')
    ax.set_title('Exploration E: Crisis Lead Time by Method')
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + 0.5, i, f'{v:.0f}d', va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_E_lead_time.png')
    plt.close(fig)
    logger.info("  Saved exploration_E_lead_time.png")

    return result


# =============================================================================
# Exploration F: Orthogonality Analysis
# =============================================================================

def exploration_f_orthogonality(X_enriched, dates, crisis_masks, any_crisis):
    """Spearman correlation between QCML and baseline scores."""
    logger.info("Exploration F: Orthogonality Analysis")

    # Compute scores from multiple methods
    methods = {}

    berry = BerryPhaseRateDetector(hilbert_dim=8, n_pca_components=15, seed=42)
    berry.fit(X_enriched)
    methods['Berry Phase'] = berry.compute_regime_scores(X_enriched)

    qfi = QFIDeterminantDetector(hilbert_dim=8, n_pca_components=15, seed=42)
    qfi.fit(X_enriched)
    methods['QFI Det'] = qfi.compute_regime_scores(X_enriched)

    cusum = CUSUMDetector()
    cusum.fit(X_enriched)
    methods['CUSUM'] = cusum.compute_regime_scores(X_enriched)

    y_labels = any_crisis.astype(int)
    rf = RandomForestRegimeDetector(seed=42)
    rf.fit_with_labels(X_enriched[:len(dates)], y_labels)
    methods['RF'] = rf.compute_regime_scores(X_enriched)

    # Align lengths and remove NaN
    min_len = min(len(v) for v in methods.values())
    score_df = pd.DataFrame({k: v[:min_len] for k, v in methods.items()})
    score_df = score_df.dropna()

    # Spearman correlation
    corr_matrix = score_df.corr(method='spearman')

    # Compute QCML vs baseline orthogonality
    qcml_names = ['Berry Phase', 'QFI Det']
    baseline_names = ['CUSUM', 'RF']
    cross_corrs = []
    for q in qcml_names:
        for b in baseline_names:
            if q in corr_matrix.columns and b in corr_matrix.columns:
                cross_corrs.append(abs(corr_matrix.loc[q, b]))

    result = {
        'name': 'F: Orthogonality Analysis',
        'description': 'Spearman rank correlation between QCML and baselines',
        'correlation_matrix': corr_matrix.round(3).to_dict(),
        'mean_qcml_baseline_corr': round(float(np.mean(cross_corrs)), 3) if cross_corrs else None,
    }

    # Separate crisis/normal correlations
    dates_aligned = dates[:min_len]
    score_df_indexed = score_df.copy()
    score_df_indexed.index = dates_aligned[:len(score_df)]

    for period, mask in [('crisis', any_crisis[:min_len]),
                         ('normal', ~any_crisis[:min_len])]:
        subset = score_df.loc[mask[:len(score_df)]]
        if len(subset) > 10:
            c = subset.corr(method='spearman')
            cross = []
            for q in qcml_names:
                for b in baseline_names:
                    if q in c.columns and b in c.columns:
                        cross.append(abs(c.loc[q, b]))
            result[f'mean_corr_{period}'] = round(float(np.mean(cross)), 3) if cross else None

    # Figure: heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='Spearman rho')
    ax.set_title('Exploration F: Method Orthogonality (Spearman)')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_F_orthogonality.png')
    plt.close(fig)
    logger.info("  Saved exploration_F_orthogonality.png")

    return result


# =============================================================================
# Exploration G: Scalar Curvature Rate
# =============================================================================

def exploration_g_curvature_rate(geom, X_norm, dates, crisis_masks, any_crisis,
                                  subsample=20):
    """Time derivative of Ricci scalar curvature as a leading indicator."""
    logger.info(f"Exploration G: Scalar Curvature Rate (subsample={subsample})")
    T = len(X_norm)
    ricci = np.full(T, np.nan)

    t0 = time.time()
    computed = 0
    for t in range(0, T, subsample):
        try:
            ricci[t] = geom.ricci_scalar(X_norm[t])
            computed += 1
            if computed % 50 == 0:
                elapsed = time.time() - t0
                rate = elapsed / computed
                remaining = (T // subsample - computed) * rate
                logger.info(f"    Ricci: {computed}/{T // subsample} "
                            f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        except Exception:
            pass

    # Interpolate
    known = ~np.isnan(ricci)
    if known.sum() > 2:
        ricci = pd.Series(ricci).interpolate(method='linear').values

    # Compute rate (finite difference), smooth and z-score
    ricci_rate = np.full(T, np.nan)
    ricci_rate[1:] = np.abs(np.diff(ricci))
    ricci_smooth = pd.Series(ricci_rate).rolling(20, min_periods=1).mean().values
    ricci_z = expanding_zscore(ricci_smooth)

    result = evaluate_signal(ricci_z, dates, crisis_masks, any_crisis)
    result['name'] = 'G: Curvature Rate'
    result['description'] = '|dR/dt| z-scored — Ricci scalar curvature rate'

    plot_signal_with_crises(
        ricci_z, dates, crisis_masks,
        'Exploration G: Ricci Scalar Curvature Rate |dR/dt| (z-scored)',
        '|z| of |dR/dt|',
        OUTPUT_DIR / 'exploration_G_curvature_rate.png',
    )
    return result, ricci_smooth


# =============================================================================
# Exploration H: Sectional Curvature Sign
# =============================================================================

def exploration_h_sectional_sign(geom, X_norm, dates, crisis_masks, any_crisis,
                                  subsample=10):
    """Sectional curvature sign: negative = diverging trajectories (chaotic)."""
    logger.info(f"Exploration H: Sectional Curvature Sign (subsample={subsample})")
    T = len(X_norm)
    sec_curv = np.full(T, np.nan)

    for t in range(0, T, subsample):
        try:
            sec_curv[t] = geom.sectional_curvature(X_norm[t], i=0, j=1)
        except Exception:
            pass

    known = ~np.isnan(sec_curv)
    if known.sum() > 2:
        sec_curv = pd.Series(sec_curv).interpolate(method='linear').values

    # Binary: is curvature negative?
    is_negative = (sec_curv < 0).astype(float)
    is_negative[np.isnan(sec_curv)] = np.nan

    # Rolling fraction of negative curvature (20-day window)
    neg_frac = pd.Series(is_negative).rolling(20, min_periods=5).mean().values

    result = evaluate_signal(neg_frac, dates, crisis_masks, any_crisis)
    result['name'] = 'H: Sectional Curvature Sign'
    result['description'] = 'Rolling fraction of negative sectional curvature K(e0,e1)<0'

    # Extra: crisis vs normal negative fraction
    valid = ~np.isnan(neg_frac)
    crisis_neg = neg_frac[valid & any_crisis]
    normal_neg = neg_frac[valid & ~any_crisis]
    if len(crisis_neg) > 0 and len(normal_neg) > 0:
        result['crisis_mean_neg_frac'] = round(float(np.mean(crisis_neg)), 3)
        result['normal_mean_neg_frac'] = round(float(np.mean(normal_neg)), 3)

    plot_signal_with_crises(
        neg_frac, dates, crisis_masks,
        'Exploration H: Negative Sectional Curvature Fraction',
        'Frac(K < 0)',
        OUTPUT_DIR / 'exploration_H_sectional_sign.png',
    )
    return result, neg_frac


# =============================================================================
# Exploration I: Crisis Lifecycle Phases
# =============================================================================

def exploration_i_lifecycle(geom, X_norm, dates, crisis_masks, any_crisis):
    """Track geometric score trajectory through crisis phases."""
    logger.info("Exploration I: Crisis Lifecycle Phases")

    # Compute FS velocity as the primary geometric signal
    T = len(X_norm)
    velocity = np.full(T, np.nan)
    for t in range(T - 1):
        velocity[t + 1] = geom.quantum_distance(X_norm[t], X_norm[t + 1])

    phases = ['pre_30d', 'pre_10d', 'onset_10d', 'peak', 'recovery', 'post_30d']
    phase_profiles = {ck: {} for ck in FOCUS_CRISES}

    for ck in FOCUS_CRISES:
        if ck not in ALL_CRISES:
            continue
        cs = pd.Timestamp(ALL_CRISES[ck]['start'])
        ce = pd.Timestamp(ALL_CRISES[ck]['end'])
        mid = cs + (ce - cs) / 2

        windows = {
            'pre_30d': (cs - pd.Timedelta(days=30), cs),
            'pre_10d': (cs - pd.Timedelta(days=10), cs),
            'onset_10d': (cs, cs + pd.Timedelta(days=10)),
            'peak': (mid - pd.Timedelta(days=5), mid + pd.Timedelta(days=5)),
            'recovery': (ce - pd.Timedelta(days=10), ce),
            'post_30d': (ce, ce + pd.Timedelta(days=30)),
        }

        for phase_name, (ws, we) in windows.items():
            mask = (dates >= ws) & (dates <= we) & (~np.isnan(velocity))
            if mask.any():
                phase_profiles[ck][phase_name] = round(float(np.mean(velocity[mask])), 5)
            else:
                phase_profiles[ck][phase_name] = None

    result = {
        'name': 'I: Crisis Lifecycle',
        'description': 'FS velocity profile through crisis phases',
        'phase_profiles': phase_profiles,
    }

    # Figure: phase profiles for top crises
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(phases))
    width = 0.12
    plotted = 0
    colors = plt.cm.Set2(np.linspace(0, 1, len(FOCUS_CRISES)))

    for i, ck in enumerate(FOCUS_CRISES):
        if ck not in phase_profiles or not phase_profiles[ck]:
            continue
        vals = [phase_profiles[ck].get(p) for p in phases]
        vals = [v if v is not None else 0 for v in vals]
        offset = (i - len(FOCUS_CRISES) / 2) * width
        ax.bar(x_positions + offset, vals, width, label=ALL_CRISES.get(ck, {}).get('label', ck),
               color=colors[i], alpha=0.8)
        plotted += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Pre 30d', 'Pre 10d', 'Onset 10d', 'Peak', 'Recovery', 'Post 30d'],
                       fontsize=8)
    ax.set_ylabel('Mean FS Velocity')
    ax.set_title('Exploration I: Crisis Lifecycle — FS Velocity by Phase')
    if plotted <= 8:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_I_lifecycle.png')
    plt.close(fig)
    logger.info("  Saved exploration_I_lifecycle.png")

    return result


# =============================================================================
# Exploration J: Adiabatic/Diabatic Classification
# =============================================================================

def exploration_j_adiabatic(geom, X_norm, dates, velocity=None):
    """Classify crises as adiabatic (slow, r<<1) or diabatic (sudden, r->1)."""
    logger.info("Exploration J: Adiabatic/Diabatic Classification")
    T = len(X_norm)

    if velocity is None:
        velocity = np.full(T, np.nan)
        for t in range(T - 1):
            velocity[t + 1] = geom.quantum_distance(X_norm[t], X_norm[t + 1])

    gaps = np.full(T, np.nan)
    for t in range(T):
        gaps[t] = geom.spectral_gap(X_norm[t])

    ratio = np.full(T, np.nan)
    valid = (~np.isnan(velocity)) & (~np.isnan(gaps)) & (gaps > 1e-6)
    ratio[valid] = velocity[valid] / gaps[valid]

    classifications = {}
    for ck, ci in ALL_CRISES.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (dates >= cs) & (dates <= ce) & (~np.isnan(ratio))
        if mask.sum() < 3:
            continue
        crisis_ratio = ratio[mask]
        max_r = float(np.max(crisis_ratio))
        mean_r = float(np.mean(crisis_ratio))
        p90_r = float(np.percentile(crisis_ratio, 90))

        classifications[ck] = {
            'label': ci['label'],
            'max_speed_ratio': round(max_r, 4),
            'mean_speed_ratio': round(mean_r, 4),
            'p90_speed_ratio': round(p90_r, 4),
            'classification': 'diabatic' if p90_r > np.nanmedian(ratio[~np.isnan(ratio)]) * 2 else 'adiabatic',
        }

    result = {
        'name': 'J: Adiabatic/Diabatic',
        'description': 'Crisis transition speed classification via Mandelstam-Tamm',
        'classifications': classifications,
    }

    # Figure: table-style visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    sorted_crises = sorted(classifications.items(),
                           key=lambda x: x[1]['max_speed_ratio'], reverse=True)
    table_data = []
    for ck, info in sorted_crises:
        table_data.append([
            info['label'],
            f"{info['max_speed_ratio']:.3f}",
            f"{info['mean_speed_ratio']:.3f}",
            info['classification'].upper(),
        ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Crisis', 'Max r', 'Mean r', 'Type'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color diabatic rows red, adiabatic blue
        for i, (_, info) in enumerate(sorted_crises):
            color = '#ffcccc' if info['classification'] == 'diabatic' else '#cce5ff'
            for j in range(4):
                table[i + 1, j].set_facecolor(color)

    ax.set_title('Exploration J: Adiabatic vs Diabatic Crisis Classification', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_J_adiabatic.png')
    plt.close(fig)
    logger.info("  Saved exploration_J_adiabatic.png")

    return result


# =============================================================================
# Exploration K: Correlation Regime Forecasting
# =============================================================================

def exploration_k_correlation_forecast(geom, X_norm, dates, prices_df):
    """Predict next-period realized correlation using geometric features."""
    logger.info("Exploration K: Correlation Regime Forecasting")

    if prices_df.shape[1] < 2:
        return {
            'name': 'K: Correlation Forecast',
            'description': 'Skipped (need >=2 assets)',
            'oos_r2': None,
        }

    # Compute realized correlation (20-day rolling)
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    realized_corr = log_ret.iloc[:, 0].rolling(20).corr(log_ret.iloc[:, 1])
    realized_corr = realized_corr.reindex(dates).dropna()

    # Align with geometric features
    common_dates = dates.intersection(realized_corr.index)
    if len(common_dates) < 100:
        return {
            'name': 'K: Correlation Forecast',
            'description': 'Insufficient overlapping data',
            'oos_r2': None,
        }

    # Build target: next-20d realized correlation (shift forward)
    target = realized_corr.shift(-20).reindex(common_dates).dropna()
    valid_dates = target.index

    # Build geometric feature set at each point
    date_to_idx = {d: i for i, d in enumerate(dates)}
    geo_features = []
    valid_targets = []

    for d in valid_dates:
        idx = date_to_idx.get(d)
        if idx is None or idx >= len(X_norm):
            continue

        x = X_norm[idx]
        gap = geom.spectral_gap(x)
        v = geom.quantum_distance(X_norm[max(0, idx - 1)], x) if idx > 0 else 0.0

        geo_features.append([gap, v])
        valid_targets.append(target.loc[d])

    if len(geo_features) < 100:
        return {
            'name': 'K: Correlation Forecast',
            'description': 'Too few valid samples',
            'oos_r2': None,
        }

    geo_features = np.array(geo_features)
    valid_targets = np.array(valid_targets)

    # Walk-forward regression: 60% train, 40% test
    split = int(0.6 * len(geo_features))
    X_train, X_test = geo_features[:split], geo_features[split:]
    y_train, y_test = valid_targets[:split], valid_targets[split:]

    # Ridge regression
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baseline: naive rolling mean
    y_naive = np.full_like(y_test, np.mean(y_train))

    # R² vs naive
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_naive = np.sum((y_test - y_naive) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    oos_r2_vs_naive = 1 - ss_res / ss_naive if ss_naive > 0 else 0.0

    result = {
        'name': 'K: Correlation Forecast',
        'description': 'OOS R² of geometric features predicting 20d realized correlation',
        'oos_r2': round(float(oos_r2), 4),
        'oos_r2_vs_naive': round(float(oos_r2_vs_naive), 4),
        'n_train': split,
        'n_test': len(y_test),
    }

    # Figure: predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#2c3e50')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('Actual Correlation')
    ax.set_ylabel('Predicted Correlation')
    ax.set_title(f'OOS R² = {oos_r2:.3f}')

    ax2 = axes[1]
    ax2.plot(y_test, label='Actual', linewidth=0.8, color='#2c3e50')
    ax2.plot(y_pred, label='Predicted', linewidth=0.8, color='#e74c3c', alpha=0.8)
    ax2.legend(fontsize=8)
    ax2.set_xlabel('Test Sample')
    ax2.set_ylabel('20d Realized Correlation')
    ax2.set_title('Exploration K: Correlation Forecast')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_K_correlation.png')
    plt.close(fig)
    logger.info("  Saved exploration_K_correlation.png")

    return result


# =============================================================================
# Exploration L: Composite Geometric Signal
# =============================================================================

def exploration_l_composite(dates, crisis_masks, any_crisis,
                            smooth_a, smooth_b, smooth_g, h_frac):
    """Composite signal fusing 4 complementary geometric detectors.

    Uses signed z-scores (positive = above historical mean) so that the
    composite preserves crisis/normal separation. Absolute z-scores inflate
    the noise floor when combined, destroying separation.

    Combines FS velocity (A), speed limit ratio (B), curvature rate (G),
    and sectional curvature sign (H) using 4 fusion strategies:
    - Max z-score: alarm if ANY signal fires
    - Mean z-score: equal-weight average
    - PCA composite: expanding-window first principal component (causal)
    - Rank fusion: expanding percentile per component, then average

    Args:
        dates: DatetimeIndex aligned with score arrays.
        crisis_masks: Dict of per-crisis boolean masks.
        any_crisis: Boolean array, True if any crisis active.
        smooth_a: 1-D array, smoothed FS velocity (pre-z-score).
        smooth_b: 1-D array, smoothed speed limit ratio (pre-z-score).
        smooth_g: 1-D array, smoothed curvature rate (pre-z-score).
        h_frac: 1-D array, raw negative curvature fraction.

    Returns:
        dict with per-strategy evaluation results and best strategy summary.
    """
    logger.info("Exploration L: Composite Geometric Signal")

    # Compute signed z-scores: positive = elevated above historical mean
    z_a = expanding_zscore(smooth_a, signed=True)
    z_b = expanding_zscore(smooth_b, signed=True)
    z_g = expanding_zscore(smooth_g, signed=True)
    z_h = expanding_zscore(h_frac, signed=True)

    T = len(z_a)
    Z = np.column_stack([z_a, z_b, z_g, z_h])  # (T, 4)

    # ------------------------------------------------------------------
    # Strategy 1: Max z-score (alarm if ANY signal fires)
    # ------------------------------------------------------------------
    max_z = np.full(T, np.nan)
    for t in range(T):
        row = Z[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            max_z[t] = np.max(valid)

    # ------------------------------------------------------------------
    # Strategy 2: Mean z-score (equal-weight average)
    # ------------------------------------------------------------------
    mean_z = np.full(T, np.nan)
    for t in range(T):
        row = Z[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            mean_z[t] = np.mean(valid)

    # ------------------------------------------------------------------
    # Strategy 3: Expanding-window PCA (causal, no lookahead)
    # ------------------------------------------------------------------
    MIN_PCA_WINDOW = 252  # 1 year minimum
    pca_z = np.full(T, np.nan)
    for t in range(T):
        row = Z[t]
        if np.any(np.isnan(row)):
            # Fall back to mean when any component is NaN
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                pca_z[t] = np.mean(valid)
            continue
        if t < MIN_PCA_WINDOW:
            # Not enough history for PCA — use mean
            pca_z[t] = np.mean(row)
            continue
        # Fit PCA on data up to time t (expanding window)
        history = Z[:t]
        # Remove rows with any NaN
        valid_rows = ~np.any(np.isnan(history), axis=1)
        history_clean = history[valid_rows]
        if len(history_clean) < MIN_PCA_WINDOW:
            pca_z[t] = np.mean(row)
            continue
        pca_model = PCA(n_components=1)
        pca_model.fit(history_clean)
        # Project current row onto first PC
        pc1 = pca_model.transform(row.reshape(1, -1))[0, 0]
        pca_z[t] = abs(pc1)

    # ------------------------------------------------------------------
    # Strategy 4: Rank fusion (nonparametric)
    # Rank each component across time (expanding percentile), then average.
    # ------------------------------------------------------------------
    n_components = Z.shape[1]
    percentiles = np.full_like(Z, np.nan)
    MIN_RANK_WINDOW = 60
    for k in range(n_components):
        for t in range(MIN_RANK_WINDOW, T):
            if np.isnan(Z[t, k]):
                continue
            past = Z[:t, k]
            past = past[~np.isnan(past)]
            if len(past) < 10:
                continue
            percentiles[t, k] = np.mean(past <= Z[t, k])

    # Average percentile ranks across components (higher = more anomalous)
    rank_raw = np.full(T, np.nan)
    for t in range(T):
        row = percentiles[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            rank_raw[t] = np.mean(valid)

    # Z-score the rank fusion output so evaluate_signal works on it
    rank_z_scored = expanding_zscore(rank_raw)

    # ------------------------------------------------------------------
    # Evaluate all strategies
    # ------------------------------------------------------------------
    strategies = {
        'max_zscore': max_z,
        'mean_zscore': mean_z,
        'pca_composite': pca_z,
        'rank_fusion': rank_z_scored,
    }

    strategy_results = {}
    for name, signal in strategies.items():
        r = evaluate_signal(signal, dates, crisis_masks, any_crisis)
        r['name'] = f'L: Composite ({name})'
        strategy_results[name] = r
        logger.info(f"  L/{name}: d = {r['aggregate_d']}")

    # Find best strategy
    best_name = max(strategy_results,
                    key=lambda k: strategy_results[k].get('aggregate_d', 0) or 0)
    best_d = strategy_results[best_name]['aggregate_d']
    logger.info(f"  Best composite: {best_name} (d = {best_d})")

    # ------------------------------------------------------------------
    # Figure: all 4 composites + crisis shading
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad']
    labels = ['Max z-score', 'Mean z-score', 'PCA composite', 'Rank fusion']

    for ax, (name, signal), color, label in zip(axes, strategies.items(), colors, labels):
        valid = ~np.isnan(signal)
        ax.plot(dates[valid], signal[valid], linewidth=0.5, color=color)
        for ck, mask in crisis_masks.items():
            if mask.any():
                crisis_dates = dates[mask]
                ax.axvspan(crisis_dates.min(), crisis_dates.max(),
                           alpha=0.15, color='red', zorder=0)
        d_val = strategy_results[name].get('aggregate_d', 0) or 0
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f'{label} (d = {d_val:.3f})', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('Exploration L: Composite Geometric Signal — 4 Fusion Strategies',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_L_composite.png')
    plt.close(fig)
    logger.info("  Saved exploration_L_composite.png")

    return {
        'name': 'L: Composite Signal',
        'description': f'Best strategy: {best_name} (d={best_d})',
        'aggregate_d': best_d,
        'aggregate_ci': strategy_results[best_name].get('aggregate_ci', (None, None)),
        'p_value': strategy_results[best_name].get('p_value'),
        'per_crisis': strategy_results[best_name].get('per_crisis', {}),
        'strategies': {k: {'aggregate_d': v['aggregate_d'],
                           'aggregate_ci': v['aggregate_ci'],
                           'p_value': v['p_value']}
                       for k, v in strategy_results.items()},
        'best_strategy': best_name,
    }


# =============================================================================
# Main Driver
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Quantum Geometry Exploration Sprint')
    parser.add_argument('--quick', action='store_true',
                        help='Run Phase 1 only (explorations A, B, C, E, F)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # Data loading (shared across all explorations)
    # -------------------------------------------------------------------------
    logger.info("Loading data...")
    raw = fetch_data(['SPY', 'DIA'], '1999-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates_raw = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates = dates_raw[19:]  # Trim to match enriched features
    logger.info(f"Data: {len(X_enriched)} samples, {X_enriched.shape[1]} features, "
                f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # -------------------------------------------------------------------------
    # Preprocessing: Scale -> PCA -> Sphere normalize -> Fit geometry
    # -------------------------------------------------------------------------
    logger.info("Fitting shared QCML geometry...")
    n_pca = min(15, X_enriched.shape[1])

    scaler = StandardScaler()
    scaler.fit(X_enriched)
    X_scaled = scaler.transform(X_enriched)

    pca = PCA(n_components=n_pca)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)

    np.random.seed(42)
    geom = QCMLGeometry(n_features=n_pca, hilbert_dim=8)
    geom.fit_operators(X_norm, method='random')
    logger.info(f"Geometry fitted: hilbert_dim=8, n_pca={n_pca}, method=random")

    # -------------------------------------------------------------------------
    # Crisis masks
    # -------------------------------------------------------------------------
    crisis_masks, any_crisis = build_crisis_masks(dates)
    n_crisis_days = any_crisis.sum()
    n_normal_days = (~any_crisis).sum()
    logger.info(f"Crisis days: {n_crisis_days}, Normal days: {n_normal_days}")

    results = {}

    # =========================================================================
    # Phase 1: Quick wins (A, B, C, E, F)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Quick wins")
    logger.info("=" * 60)

    # A: FS Velocity (also cached for B, I, J)
    result_a, velocity, velocity_smooth = exploration_a_fs_velocity(
        geom, X_norm, dates, crisis_masks, any_crisis
    )
    results['A'] = result_a
    logger.info(f"  A: d = {results['A']['aggregate_d']}")

    # B: Speed Limit Ratio
    results['B'], speed_smooth = exploration_b_speed_limit(
        geom, X_norm, dates, crisis_masks, any_crisis, velocity=velocity
    )
    logger.info(f"  B: d = {results['B']['aggregate_d']}")

    # C: Dimensionality Collapse
    results['C'] = exploration_c_dim_collapse(
        geom, X_norm, dates, crisis_masks, any_crisis, subsample=5
    )
    logger.info(f"  C: d = {results['C']['aggregate_d']}")

    # E: Lead Time Analysis
    results['E'] = exploration_e_lead_time(X_enriched, dates, crisis_masks, any_crisis)
    logger.info(f"  E: median lead times = {results['E'].get('median_lead_times', {})}")

    # F: Orthogonality
    results['F'] = exploration_f_orthogonality(X_enriched, dates, crisis_masks, any_crisis)
    logger.info(f"  F: mean QCML-baseline |rho| = {results['F'].get('mean_qcml_baseline_corr')}")

    phase1_time = time.time() - t_start
    logger.info(f"Phase 1 complete in {phase1_time:.0f}s")

    if not args.quick:
        # =====================================================================
        # Phase 2: Medium effort (D, G, H, I, J)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("PHASE 2: Medium effort")
        logger.info("=" * 60)

        # D: Entanglement Entropy
        results['D'] = exploration_d_entanglement(
            X_enriched, dates, crisis_masks, any_crisis, scaler, pca, subsample=3
        )
        logger.info(f"  D: d = {results['D']['aggregate_d']}")

        # G: Curvature Rate (expensive — subsample every 50 points, ~3s/point)
        results['G'], curv_smooth = exploration_g_curvature_rate(
            geom, X_norm, dates, crisis_masks, any_crisis, subsample=50
        )
        logger.info(f"  G: d = {results['G']['aggregate_d']}")

        # H: Sectional Curvature Sign
        results['H'], sect_frac = exploration_h_sectional_sign(
            geom, X_norm, dates, crisis_masks, any_crisis, subsample=10
        )
        logger.info(f"  H: d = {results['H']['aggregate_d']}")

        # I: Crisis Lifecycle
        results['I'] = exploration_i_lifecycle(geom, X_norm, dates, crisis_masks, any_crisis)
        logger.info(f"  I: lifecycle profiles computed")

        # J: Adiabatic/Diabatic
        results['J'] = exploration_j_adiabatic(geom, X_norm, dates, velocity=velocity)
        logger.info(f"  J: classifications = "
                    f"{sum(1 for v in results['J']['classifications'].values() if v.get('classification') == 'diabatic')} diabatic, "
                    f"{sum(1 for v in results['J']['classifications'].values() if v.get('classification') == 'adiabatic')} adiabatic")

        phase2_time = time.time() - t_start - phase1_time
        logger.info(f"Phase 2 complete in {phase2_time:.0f}s")

        # =====================================================================
        # Phase 2.5: Composite signal (needs A, B from Phase 1 + G, H from Phase 2)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("PHASE 2.5: Composite geometric signal")
        logger.info("=" * 60)

        results['L'] = exploration_l_composite(
            dates, crisis_masks, any_crisis,
            smooth_a=velocity_smooth, smooth_b=speed_smooth,
            smooth_g=curv_smooth, h_frac=sect_frac,
        )
        logger.info(f"  L: best d = {results['L']['aggregate_d']} "
                    f"({results['L']['best_strategy']})")

        # =====================================================================
        # Phase 3: Full pipeline (K)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("PHASE 3: Full pipeline")
        logger.info("=" * 60)

        results['K'] = exploration_k_correlation_forecast(
            geom, X_norm, dates, prices_df
        )
        logger.info(f"  K: OOS R² = {results['K'].get('oos_r2')}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"ALL EXPLORATIONS COMPLETE ({total_time:.0f}s)")
    logger.info("=" * 60)

    # Build summary table
    summary_lines = [
        "=" * 80,
        "QUANTUM GEOMETRY EXPLORATION — SUMMARY",
        "=" * 80,
        f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
        f"Samples: {len(X_enriched)}, Features: {X_enriched.shape[1]}",
        f"Crisis days: {n_crisis_days}, Normal days: {n_normal_days}",
        f"Total runtime: {total_time:.0f}s",
        "",
        f"{'ID':<4} {'Name':<30} {'d':>8} {'95% CI':>20} {'p-value':>10} {'Key Finding'}",
        "-" * 100,
    ]

    for key in sorted(results.keys()):
        r = results[key]
        name = r.get('name', key)
        d = r.get('aggregate_d', None)
        ci = r.get('aggregate_ci', (None, None))
        p = r.get('p_value', None)

        d_str = f"{d:.3f}" if d is not None and not (isinstance(d, float) and np.isnan(d)) else "N/A"
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None and not np.isnan(ci[0]) else "N/A"
        p_str = f"{p:.2e}" if p is not None and not np.isnan(p) else "N/A"

        # Key finding per exploration
        finding = ""
        if key == 'E':
            med = r.get('median_lead_times', {})
            parts = [f"{k}: {v:.0f}d" for k, v in med.items() if v is not None]
            finding = "; ".join(parts[:3])
        elif key == 'F':
            finding = f"|rho| = {r.get('mean_qcml_baseline_corr', 'N/A')}"
        elif key == 'K':
            finding = f"OOS R² = {r.get('oos_r2', 'N/A')}"
        elif key == 'J':
            n_dia = sum(1 for v in r.get('classifications', {}).values()
                        if v.get('classification') == 'diabatic')
            finding = f"{n_dia} diabatic crises"
        elif key == 'L':
            strats = r.get('strategies', {})
            parts = [f"{k}: {v['aggregate_d']:.3f}" for k, v in strats.items()]
            finding = f"best={r.get('best_strategy')}; " + "; ".join(parts)

        summary_lines.append(f"{key:<4} {name:<30} {d_str:>8} {ci_str:>20} {p_str:>10}  {finding}")

    summary_lines.extend([
        "",
        "=" * 80,
        "Interpretation guide:",
        "  d > 0.8: large effect (strong crisis/normal separation)",
        "  d > 0.5: medium effect (meaningful signal)",
        "  d > 0.2: small effect (weak but detectable)",
        "  p < 0.05: statistically significant",
        "=" * 80,
    ])

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save outputs
    (OUTPUT_DIR / 'summary_table.txt').write_text(summary_text)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'summary_table.txt'}")

    # JSON results (convert numpy types)
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    results_json = json.loads(json.dumps(results, default=_convert))
    (OUTPUT_DIR / 'results.json').write_text(
        json.dumps(results_json, indent=2, default=str)
    )
    logger.info(f"Results saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
