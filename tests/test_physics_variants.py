"""Invariants for the opt-in physics variants (gauge fixing, ...).

Contract: every variant defaults OFF and leaves committed behavior byte-identical
(the headline golden snapshot enforces that separately); when ON it is
deterministic and has a measurable, physically-motivated effect.
"""

import numpy as np

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import BaseRegimeDetector, GeometricPhaseRateDetector


def _fitted(gauge_fix, seed=0):
    rng = np.random.default_rng(seed)
    Xfit = rng.normal(0, 1, (60, 3))
    Xfit /= np.linalg.norm(Xfit, axis=1, keepdims=True) + 1e-9
    g = QCMLGeometry(n_features=3, hilbert_dim=4, gauge_fix=gauge_fix)
    g.fit_operators(Xfit, method="random")
    return g, Xfit


def test_gauge_fix_defaults_off():
    assert QCMLGeometry(n_features=3, hilbert_dim=4).gauge_fix is False


def test_gauge_off_matches_plain_eig():
    """With gauge_fix off, the ground state is exactly eigh's raw eigenvector."""
    g, X = _fitted(gauge_fix=False)
    H = g.error_hamiltonian(X[0])
    evals, evecs = np.linalg.eigh(H)
    raw = evecs[:, int(np.argmin(evals))]
    raw = raw / np.linalg.norm(raw)
    psi = g.quasi_coherent_state(X[0])
    # identical up to the implicit no-op (same code path); compare directly
    np.testing.assert_allclose(psi, raw.astype(complex))


def test_gauge_fix_pins_phase_and_is_deterministic():
    g, X = _fitted(gauge_fix=True)
    psi1 = g.quasi_coherent_state(X[0])
    g2, _ = _fitted(gauge_fix=True)  # same seed/ops
    psi2 = g2.quasi_coherent_state(X[0])
    np.testing.assert_allclose(psi1, psi2)  # deterministic
    j = int(np.argmax(np.abs(psi1)))
    assert abs(psi1[j].imag) < 1e-12 and psi1[j].real > 0  # pinned real-positive


def test_gauge_fix_changes_geometric_phase_rate():
    """The fix targets the gauge-dependent np.angle(overlap) — it must move it."""
    raw_g, X = _fitted(gauge_fix=False)
    fix_g, _ = _fitted(gauge_fix=True)
    raw = np.array([raw_g.geometric_phase_rate(X[t], X[t + 1]) for t in range(20)])
    fix = np.array([fix_g.geometric_phase_rate(X[t], X[t + 1]) for t in range(20)])
    assert not np.allclose(raw, fix)


def test_detector_accepts_gauge_fix_and_runs():
    rng = np.random.default_rng(2)
    raw = rng.normal(0, 0.01, (120, 4))
    raw[60:70] *= 6.0
    Xe = BaseRegimeDetector.build_enriched_features(raw, lookback=20)
    fit_end = Xe.shape[0] // 2

    off = GeometricPhaseRateDetector(hilbert_dim=4, n_pca_components=6, causal_fit_length=fit_end)
    on = GeometricPhaseRateDetector(
        hilbert_dim=4, n_pca_components=6, causal_fit_length=fit_end, gauge_fix=True
    )
    s_off = np.asarray(off.fit(Xe).compute_regime_scores(Xe), dtype=float)
    s_on = np.asarray(on.fit(Xe).compute_regime_scores(Xe), dtype=float)
    assert s_off.shape == s_on.shape == (Xe.shape[0],)
    assert np.isfinite(s_on).sum() > 10
    assert not np.allclose(np.nan_to_num(s_off), np.nan_to_num(s_on))
