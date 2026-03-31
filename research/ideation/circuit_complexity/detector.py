"""
Circuit Complexity Proxy for Financial Regime Detection
========================================================

Research Question Q0126: Can quantum circuit complexity (minimum gates to
prepare state from reference) serve as a crisis proximity measure?

Implements the **Nielsen complexity proxy**: geodesic distance on SU(d)
via the unitary log-norm. For a target state |psi> and reference |ref>,
we construct the minimal unitary U such that U|ref> = |psi>, then compute:

    C(psi) = || log(U) ||_F

This is a proxy for the true circuit complexity (exact gate count is
intractable). The key insight is that states far from the reference
in the unitary manifold require more "gates" to prepare — during crises,
the market state |psi(t)> should drift far from the calm-market reference.

Additionally computes:
    - Complexity rate: |C(t) - C(t-1)| (speed of complexity change)
    - Infidelity proxy: 1 - |<psi|ref>|^2 (simplest proxy, for comparison)

Smoke test: 4 crises, SPY+DIA, 2006-2024.

Usage:
    python research/ideation/circuit_complexity/detector.py
"""

import json
import logging
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
from scipy.linalg import logm, null_space
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci, welch_t_test
from qcml_geometry.core import QCMLGeometry

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

OUTPUT_DIR = ROOT / 'research' / 'ideation' / 'circuit_complexity' / 'outputs'

# Smoke test configuration
SMOKE_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2006-01-01'
END_DATE = '2024-12-31'
HILBERT_DIM = 8  # 3 qubits
N_PCA = 3  # Match hilbert_dim constraints (need n_features for operators)
N_BOOTSTRAP = 2000
MIN_EXPANDING = 60
ROLLING_WINDOW = 20
SEED = 42

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# Core: Circuit Complexity Proxy
# =============================================================================

def construct_unitary(psi, ref=None):
    """Construct minimal unitary U such that U|ref> = |psi>.

    Uses Householder-like construction: finds the unitary that rotates
    |ref> to |psi> via the shortest path on SU(d).

    If |psi> = |ref>, returns identity.
    Otherwise, constructs a rotation in the 2D subspace spanned by
    {|ref>, |psi>} and acts as identity on the orthogonal complement.

    Args:
        psi: Target state, shape (d,), must be normalized.
        ref: Reference state, shape (d,). Default: |0...0>.

    Returns:
        U: Unitary matrix of shape (d, d).
    """
    d = len(psi)
    if ref is None:
        ref = np.zeros(d, dtype=complex)
        ref[0] = 1.0

    psi = np.asarray(psi, dtype=complex)
    ref = np.asarray(ref, dtype=complex)

    # Normalize
    psi = psi / np.linalg.norm(psi)
    ref = ref / np.linalg.norm(ref)

    # Overlap
    overlap = np.vdot(ref, psi)
    overlap_abs = np.abs(overlap)

    # If already aligned (up to phase), return identity * phase
    if overlap_abs > 1.0 - 1e-12:
        phase = overlap / overlap_abs if overlap_abs > 1e-15 else 1.0
        return phase * np.eye(d, dtype=complex)

    # Gram-Schmidt: construct orthonormal basis in the 2D subspace {ref, psi}
    # e1 = ref, e2 = (psi - <ref|psi> ref) / ||...||
    e1 = ref.copy()
    e2 = psi - overlap * ref
    e2 = e2 / np.linalg.norm(e2)

    # In this 2D subspace, the rotation that maps e1 -> psi is:
    #   R = |psi><e1| + |e2_perp><e2| + I_perp
    # where psi = cos(theta) e1 + sin(theta) e2 (theta = arccos(|overlap|))
    # and the rotation angle is theta in the {e1, e2} plane.
    #
    # More precisely: U = I + (psi - ref) @ ref^dag + (e2_rot - e2) @ e2^dag
    # But simplest: U = I + (cos(theta)-1)(|e1><e1| + |e2><e2|)
    #                  + sin(theta)(|e2><e1| - |e1><e2|)
    # adjusted for the phase of the overlap.

    # Angle between ref and psi (on the Bloch sphere of the 2D subspace)
    # psi = overlap * e1 + sqrt(1-|overlap|^2) * e2
    cos_theta = overlap  # complex!
    sin_theta = np.sqrt(1.0 - overlap_abs**2)

    # The 2x2 rotation matrix in the {e1, e2} basis:
    # R_2d = [[cos_theta, -sin_theta_conj], [sin_theta, cos_theta_conj]]
    # such that R_2d @ [1, 0] = [cos_theta, sin_theta] = psi in this basis.

    # Build full U = I + (R_2d - I_2d) projected into full space
    # U = I + (cos_theta - 1)|e1><e1| + sin_theta|e2><e1|
    #       + (-sin_theta*)|e1><e2| + (cos_theta* - 1)|e2><e2|
    # But we need U|ref> = psi, and ref = e1, so:
    # U e1 = cos_theta * e1 + sin_theta * e2 = psi  ✓

    U = np.eye(d, dtype=complex)
    U += (cos_theta - 1.0) * np.outer(e1, e1.conj())
    U += sin_theta * np.outer(e2, e1.conj())
    U += (-np.conj(sin_theta)) * np.outer(e1, e2.conj())
    U += (np.conj(cos_theta) - 1.0) * np.outer(e2, e2.conj())

    return U


def circuit_complexity_proxy(psi, ref=None):
    """Compute circuit complexity proxy via geodesic distance on SU(d).

    C(psi) = || log(U) ||_F

    where U is the minimal unitary rotating |ref> to |psi>.

    For the Householder-style U that acts nontrivially only in the
    2D subspace {|ref>, |psi>}, the complexity reduces to:

        C = sqrt(2) * arccos(|<ref|psi>|)

    which is proportional to the Fubini-Study distance. However, we
    compute it via the full matrix log for generality and to verify
    the analytical formula.

    Args:
        psi: Target state, shape (d,).
        ref: Reference state, shape (d,). Default: |0...0>.

    Returns:
        complexity: ||log(U)||_F (non-negative).
    """
    d = len(psi)
    if ref is None:
        ref = np.zeros(d, dtype=complex)
        ref[0] = 1.0

    psi = np.asarray(psi, dtype=complex)
    ref = np.asarray(ref, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    ref = ref / np.linalg.norm(ref)

    U = construct_unitary(psi, ref)

    # Matrix logarithm
    log_U = logm(U)

    # Frobenius norm
    complexity = np.linalg.norm(log_U, 'fro')

    # Should be real and non-negative
    return float(np.real(complexity))


def infidelity_proxy(psi, ref=None):
    """Simplest complexity proxy: 1 - |<psi|ref>|^2.

    This is the infidelity with the reference state. Zero when psi = ref,
    one when orthogonal. Serves as a baseline comparison for the full
    unitary log-norm proxy.

    Args:
        psi: Target state, shape (d,).
        ref: Reference state, shape (d,). Default: |0...0>.

    Returns:
        infidelity: 1 - |<psi|ref>|^2 in [0, 1].
    """
    d = len(psi)
    if ref is None:
        ref = np.zeros(d, dtype=complex)
        ref[0] = 1.0

    psi = np.asarray(psi, dtype=complex)
    ref = np.asarray(ref, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    ref = ref / np.linalg.norm(ref)

    fidelity = np.abs(np.vdot(ref, psi)) ** 2
    return float(1.0 - fidelity)


# =============================================================================
# Signal Construction
# =============================================================================

def expanding_zscore(signal, min_expanding=MIN_EXPANDING, signed=False):
    """Convert raw signal to expanding-window z-scores (causal, no lookahead).

    Args:
        signal: 1-D array (may contain NaN).
        min_expanding: Minimum samples before z-scoring starts.
        signed: If False, return absolute z-scores.

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
        crises: Dict of crisis definitions.
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


def evaluate_signal(scores, dates, crisis_masks, any_crisis, label=''):
    """Compute aggregate and per-crisis Cohen's d for a signal."""
    valid = ~np.isnan(scores)
    crisis_scores = scores[valid & any_crisis]
    normal_scores = scores[valid & ~any_crisis]

    if len(crisis_scores) < 5 or len(normal_scores) < 10:
        return {
            'name': label,
            'aggregate_d': float('nan'),
            'aggregate_ci': (float('nan'), float('nan')),
            'per_crisis': {},
            'p_value': float('nan'),
        }

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
        'name': label,
        'aggregate_d': round(float(d), 3),
        'aggregate_ci': (round(float(ci_lo), 3), round(float(ci_hi), 3)),
        'per_crisis': per_crisis,
        'p_value': float(p_val),
    }


# =============================================================================
# Reference State Strategies
# =============================================================================

def compute_calm_reference(geom, X_norm, calm_end=252):
    """Compute reference state as average ground state over calm period.

    Uses the first `calm_end` trading days (approximately 1 year) as the
    "calm" reference period. Averages the density matrices and takes the
    leading eigenvector.

    Args:
        geom: Fitted QCMLGeometry.
        X_norm: Normalized feature matrix (T, d).
        calm_end: Number of initial days to use as calm reference.

    Returns:
        ref: Reference state vector, shape (hilbert_dim,).
    """
    d = geom.hilbert_dim
    rho_avg = np.zeros((d, d), dtype=complex)
    count = 0

    for t in range(min(calm_end, len(X_norm))):
        psi = geom.quasi_coherent_state(X_norm[t])
        rho_avg += np.outer(psi, psi.conj())
        count += 1

    rho_avg /= count
    eigenvalues, eigenvectors = np.linalg.eigh(rho_avg)
    ref = eigenvectors[:, -1]  # Leading eigenvector
    ref = ref / np.linalg.norm(ref)
    return ref


# =============================================================================
# Main Smoke Test
# =============================================================================

def run_smoke_test():
    """Run circuit complexity proxy smoke test on 4 crises."""
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Data loading ----
    logger.info("Fetching market data...")
    raw = fetch_data(SYMBOLS, START_DATE, END_DATE)
    prices_df = raw['close'].unstack('symbol').dropna()
    X_raw, dates = create_feature_matrix(prices_df)
    logger.info(f"Feature matrix: {X_raw.shape}, dates: {dates[0]} to {dates[-1]}")

    T = X_raw.shape[0]

    # ---- Preprocessing: StandardScaler + PCA + sphere normalization ----
    np.random.seed(SEED)
    scaler = StandardScaler()
    scaler.fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    pca = PCA(n_components=N_PCA)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # Sphere normalization (project onto unit sphere)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)

    logger.info(f"PCA components: {N_PCA}, explained variance: "
                f"{pca.explained_variance_ratio_.sum():.3f}")

    # ---- Build QCML geometry ----
    geom = QCMLGeometry(n_features=N_PCA, hilbert_dim=HILBERT_DIM)
    geom.fit_operators(X_norm, method='random')
    logger.info(f"QCML geometry fitted: d={HILBERT_DIM}, {len(geom.operators)} operators")

    # ---- Compute reference states ----
    # Strategy 1: |0...0> (default)
    ref_zero = np.zeros(HILBERT_DIM, dtype=complex)
    ref_zero[0] = 1.0

    # Strategy 2: Average calm-period state
    ref_calm = compute_calm_reference(geom, X_norm, calm_end=252)
    logger.info(f"Calm reference overlap with |0>: {np.abs(np.vdot(ref_zero, ref_calm)):.4f}")

    # ---- Compute ground states for all time steps ----
    logger.info("Computing ground states...")
    states = []
    for t in range(T):
        psi = geom.quasi_coherent_state(X_norm[t])
        states.append(psi)
    states = np.array(states)
    logger.info(f"Ground states computed: {states.shape}")

    # ---- Compute complexity proxies ----
    logger.info("Computing circuit complexity proxies...")

    # 1. Nielsen complexity (unitary log-norm) with |0> reference
    complexity_zero = np.array([
        circuit_complexity_proxy(states[t], ref_zero) for t in range(T)
    ])

    # 2. Nielsen complexity with calm-period reference
    complexity_calm = np.array([
        circuit_complexity_proxy(states[t], ref_calm) for t in range(T)
    ])

    # 3. Complexity rate: |C(t) - C(t-1)|
    complexity_rate = np.full(T, np.nan)
    for t in range(1, T):
        complexity_rate[t] = abs(complexity_calm[t] - complexity_calm[t - 1])

    # 4. Infidelity proxy (simplest: 1 - |<psi|ref>|^2)
    infidelity = np.array([
        infidelity_proxy(states[t], ref_calm) for t in range(T)
    ])

    # 5. Multi-lag fidelity (for correlation comparison)
    logger.info("Computing multi-lag fidelity...")
    lags = [1, 3, 5, 10]
    lag_weights = [0.4, 0.3, 0.2, 0.1]
    fidelity_raw = np.full(T, np.nan)
    for t in range(max(lags), T):
        wf = 0.0
        for lag, w in zip(lags, lag_weights):
            overlap = np.abs(np.vdot(states[t], states[t - lag]))
            wf += w * (1.0 - overlap**2)
        fidelity_raw[t] = wf

    logger.info("Complexity proxies computed. Generating z-scores...")

    # ---- Smooth and z-score ----
    signals = {
        'complexity_zero': complexity_zero,
        'complexity_calm': complexity_calm,
        'complexity_rate': complexity_rate,
        'infidelity': infidelity,
        'fidelity_multilag': fidelity_raw,
    }

    smoothed = {}
    zscored = {}
    for name, sig in signals.items():
        sm = pd.Series(sig).rolling(ROLLING_WINDOW, min_periods=1).mean().values
        smoothed[name] = sm
        zscored[name] = expanding_zscore(sm)

    # ---- Build crisis masks (focus on 4 smoke-test crises) ----
    smoke_crises = {k: ALL_CRISES[k] for k in SMOKE_CRISES if k in ALL_CRISES}
    crisis_masks, any_crisis = build_crisis_masks(dates, smoke_crises)

    # Also build full crisis masks for correlation
    all_crisis_masks, all_any_crisis = build_crisis_masks(dates)

    # ---- Evaluate signals ----
    logger.info("Evaluating signals...")
    results = {}
    for name, z in zscored.items():
        res = evaluate_signal(z, dates, crisis_masks, any_crisis, label=name)
        results[name] = res
        logger.info(
            f"  {name}: aggregate d={res['aggregate_d']:.3f}, "
            f"CI=({res['aggregate_ci'][0]:.3f}, {res['aggregate_ci'][1]:.3f}), "
            f"p={res['p_value']:.2e}"
        )
        for ck in SMOKE_CRISES:
            d_val = res['per_crisis'].get(ck)
            if d_val is not None:
                logger.info(f"    {ck}: d={d_val:.3f}")

    # ---- Correlation analysis: complexity vs multi-lag fidelity ----
    logger.info("\nCorrelation analysis: Circuit Complexity vs Multi-Lag Fidelity")

    # Raw signal correlation
    valid_both = ~np.isnan(smoothed['complexity_calm']) & ~np.isnan(smoothed['fidelity_multilag'])
    if valid_both.sum() > 30:
        corr_raw = np.corrcoef(
            smoothed['complexity_calm'][valid_both],
            smoothed['fidelity_multilag'][valid_both]
        )[0, 1]
        logger.info(f"  Raw signal Pearson r = {corr_raw:.4f}")

        # Rank correlation (more robust)
        from scipy.stats import spearmanr
        rho, p_rho = spearmanr(
            smoothed['complexity_calm'][valid_both],
            smoothed['fidelity_multilag'][valid_both]
        )
        logger.info(f"  Spearman rho = {rho:.4f}, p = {p_rho:.2e}")
    else:
        corr_raw = float('nan')
        rho = float('nan')
        p_rho = float('nan')

    # Z-score correlation
    valid_z = ~np.isnan(zscored['complexity_calm']) & ~np.isnan(zscored['fidelity_multilag'])
    if valid_z.sum() > 30:
        corr_z = np.corrcoef(
            zscored['complexity_calm'][valid_z],
            zscored['fidelity_multilag'][valid_z]
        )[0, 1]
        logger.info(f"  Z-score Pearson r = {corr_z:.4f}")
    else:
        corr_z = float('nan')

    # ---- Analytical verification ----
    # For our Householder construction, C = sqrt(2) * arccos(|<ref|psi>|)
    # Check this relationship holds
    logger.info("\nAnalytical verification: C vs sqrt(2)*arccos(|<ref|psi>|)")
    sample_indices = np.random.choice(T, min(100, T), replace=False)
    c_numerical = np.array([complexity_calm[i] for i in sample_indices])
    c_analytical = np.array([
        np.sqrt(2) * np.arccos(np.clip(np.abs(np.vdot(ref_calm, states[i])), 0, 1))
        for i in sample_indices
    ])
    max_diff = np.max(np.abs(c_numerical - c_analytical))
    logger.info(f"  Max |C_numerical - C_analytical| = {max_diff:.2e}")
    if max_diff < 1e-6:
        logger.info("  CONFIRMED: Nielsen proxy = sqrt(2) * Fubini-Study distance")
    else:
        logger.info(f"  WARNING: Discrepancy detected (max_diff={max_diff:.2e})")

    # ---- Plotting ----
    logger.info("Generating plots...")

    # Plot 1: Complexity time series with crisis shading
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    valid = ~np.isnan(zscored['complexity_calm'])
    ax.plot(dates[valid], zscored['complexity_calm'][valid],
            linewidth=0.5, color='#2c3e50', label='Nielsen (calm ref)')
    valid2 = ~np.isnan(zscored['complexity_zero'])
    ax.plot(dates[valid2], zscored['complexity_zero'][valid2],
            linewidth=0.5, color='#3498db', alpha=0.6, label='Nielsen (|0> ref)')
    for ck, mask in crisis_masks.items():
        if mask.any():
            crisis_dates = dates[mask]
            ax.axvspan(crisis_dates.min(), crisis_dates.max(),
                       alpha=0.15, color='red', zorder=0)
    ax.set_ylabel('|z-score|')
    ax.set_title('Circuit Complexity Proxy (Nielsen log-norm)', fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    valid = ~np.isnan(zscored['complexity_rate'])
    ax.plot(dates[valid], zscored['complexity_rate'][valid],
            linewidth=0.5, color='#e74c3c')
    for ck, mask in crisis_masks.items():
        if mask.any():
            crisis_dates = dates[mask]
            ax.axvspan(crisis_dates.min(), crisis_dates.max(),
                       alpha=0.15, color='red', zorder=0)
    ax.set_ylabel('|z-score|')
    ax.set_title('Complexity Rate |dC/dt|', fontsize=11)

    ax = axes[2]
    valid = ~np.isnan(zscored['infidelity'])
    ax.plot(dates[valid], zscored['infidelity'][valid],
            linewidth=0.5, color='#27ae60', label='Infidelity proxy')
    valid_f = ~np.isnan(zscored['fidelity_multilag'])
    ax.plot(dates[valid_f], zscored['fidelity_multilag'][valid_f],
            linewidth=0.5, color='#8e44ad', alpha=0.7, label='Multi-Lag Fidelity')
    for ck, mask in crisis_masks.items():
        if mask.any():
            crisis_dates = dates[mask]
            ax.axvspan(crisis_dates.min(), crisis_dates.max(),
                       alpha=0.15, color='red', zorder=0)
    ax.set_ylabel('|z-score|')
    ax.set_title('Infidelity Proxy vs Multi-Lag Fidelity', fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'circuit_complexity_timeseries.png')
    plt.close(fig)
    logger.info("  Saved circuit_complexity_timeseries.png")

    # Plot 2: Scatter — complexity vs infidelity (analytical relationship)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    valid = ~np.isnan(smoothed['complexity_calm']) & ~np.isnan(smoothed['infidelity'])
    ax.scatter(smoothed['infidelity'][valid], smoothed['complexity_calm'][valid],
               s=1, alpha=0.3, color='#2c3e50')
    ax.set_xlabel('Infidelity 1-|<psi|ref>|^2')
    ax.set_ylabel('Nielsen Complexity ||log(U)||_F')
    ax.set_title('Complexity vs Infidelity (raw)')

    ax = axes[1]
    valid = ~np.isnan(smoothed['complexity_calm']) & ~np.isnan(smoothed['fidelity_multilag'])
    ax.scatter(smoothed['fidelity_multilag'][valid], smoothed['complexity_calm'][valid],
               s=1, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Multi-Lag Infidelity (weighted)')
    ax.set_ylabel('Nielsen Complexity ||log(U)||_F')
    ax.set_title(f'Complexity vs Multi-Lag Fidelity (r={corr_raw:.3f})')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'circuit_complexity_scatter.png')
    plt.close(fig)
    logger.info("  Saved circuit_complexity_scatter.png")

    # Plot 3: Per-crisis bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['complexity_calm', 'complexity_rate', 'infidelity', 'fidelity_multilag']
    method_labels = ['Nielsen (calm)', 'Complexity Rate', 'Infidelity', 'Multi-Lag Fidelity']
    x = np.arange(len(SMOKE_CRISES))
    width = 0.2

    for i, (method, label) in enumerate(zip(methods, method_labels)):
        d_vals = [results[method]['per_crisis'].get(ck, 0) or 0 for ck in SMOKE_CRISES]
        ax.bar(x + i * width, d_vals, width, label=label, alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([ALL_CRISES[ck]['label'] for ck in SMOKE_CRISES],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Cohen's d")
    ax.set_title('Per-Crisis Effect Sizes')
    ax.legend(fontsize=8)
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.5, label='d=0.8 (large)')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'circuit_complexity_per_crisis.png')
    plt.close(fig)
    logger.info("  Saved circuit_complexity_per_crisis.png")

    # ---- Save results JSON ----
    elapsed = time.time() - t_start
    output = {
        'knight': 'Knight 2 (Empirical Implementation)',
        'research_question': 'Q0126',
        'question': 'Can circuit complexity proxy serve as crisis proximity measure?',
        'method': 'Nielsen complexity proxy: ||log(U)||_F on SU(d)',
        'config': {
            'hilbert_dim': HILBERT_DIM,
            'n_pca': N_PCA,
            'symbols': SYMBOLS,
            'date_range': f'{START_DATE} to {END_DATE}',
            'smoke_crises': SMOKE_CRISES,
            'rolling_window': ROLLING_WINDOW,
            'n_bootstrap': N_BOOTSTRAP,
            'operator_method': 'random',
            'reference_strategies': ['|0...0>', 'calm-period average'],
        },
        'results': {},
        'correlation_analysis': {
            'complexity_vs_multilag_fidelity': {
                'pearson_r_raw': round(float(corr_raw), 4) if not np.isnan(corr_raw) else None,
                'spearman_rho': round(float(rho), 4) if not np.isnan(rho) else None,
                'spearman_p': float(p_rho) if not np.isnan(p_rho) else None,
                'pearson_r_zscore': round(float(corr_z), 4) if not np.isnan(corr_z) else None,
            }
        },
        'analytical_verification': {
            'nielsen_equals_sqrt2_fs_distance': max_diff < 1e-6,
            'max_discrepancy': float(max_diff),
        },
        'runtime_seconds': round(elapsed, 1),
    }

    # Add per-signal results
    for name, res in results.items():
        output['results'][name] = {
            'aggregate_d': res['aggregate_d'],
            'aggregate_ci': list(res['aggregate_ci']),
            'p_value': res['p_value'],
            'per_crisis': res['per_crisis'],
        }

    output_path = OUTPUT_DIR / 'smoke_test_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # ---- Print summary ----
    logger.info("\n" + "=" * 70)
    logger.info("CIRCUIT COMPLEXITY PROXY — SMOKE TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Signal':<25} {'Agg d':>8} {'CI':>18} {'p-value':>12}")
    logger.info("-" * 70)
    for name, res in results.items():
        ci = f"({res['aggregate_ci'][0]:.3f}, {res['aggregate_ci'][1]:.3f})"
        logger.info(
            f"{name:<25} {res['aggregate_d']:>8.3f} {ci:>18} {res['p_value']:>12.2e}"
        )
    logger.info("-" * 70)
    logger.info(f"\nCorrelation with Multi-Lag Fidelity:")
    logger.info(f"  Pearson r (raw):    {corr_raw:.4f}")
    logger.info(f"  Spearman rho:       {rho:.4f}")
    logger.info(f"  Pearson r (z-score): {corr_z:.4f}")
    logger.info(f"\nAnalytical: Nielsen = sqrt(2) * FS distance: {max_diff < 1e-6}")
    logger.info(f"Runtime: {elapsed:.1f}s")

    return output


if __name__ == '__main__':
    results = run_smoke_test()
