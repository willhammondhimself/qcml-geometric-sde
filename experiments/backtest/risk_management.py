"""
Risk management: drawdown circuit breaker and position limits.
"""

import numpy as np


def drawdown_circuit_breaker(
    equity_curve, weights, max_drawdown=0.15, cooldown_days=20,
):
    """Cut exposure when drawdown exceeds threshold.

    Args:
        equity_curve: Cumulative equity (T,).
        weights: Target weights (T,).
        max_drawdown: Drawdown threshold to trigger circuit breaker.
        cooldown_days: Days to stay flat after trigger.

    Returns:
        adjusted_weights: Weights with circuit breaker applied.
    """
    adjusted = weights.copy()
    T = len(adjusted)
    peak = equity_curve[0]
    cooldown_until = -1

    for t in range(T):
        peak = max(peak, equity_curve[t])
        drawdown = (peak - equity_curve[t]) / peak if peak > 0 else 0.0

        if t < cooldown_until:
            adjusted[t] = 0.0
        elif drawdown > max_drawdown:
            adjusted[t] = 0.0
            cooldown_until = t + cooldown_days

    return adjusted


def apply_position_limits(weights, max_long=1.5, max_short=-0.5):
    """Clip weights to position limits.

    Args:
        weights: Target weights (T,) or (T, n_assets).
        max_long: Maximum long weight per asset.
        max_short: Maximum short weight per asset.

    Returns:
        Clipped weights.
    """
    return np.clip(weights, max_short, max_long)


def dynamic_vol_target(
    p_crisis, vol_calm=0.12, vol_crisis=0.05,
):
    """Compute regime-dependent volatility target.

    Interpolates between calm and crisis vol targets using P(crisis).
    Lower target during crises = faster de-risking and lower drawdowns.

    Args:
        p_crisis: P(crisis) signal (T,).
        vol_calm: Vol target during calm periods (annualized).
        vol_crisis: Vol target during crisis periods (annualized).

    Returns:
        target_vol: Dynamic vol target (T,).
    """
    p = np.where(np.isnan(p_crisis), 0.0, p_crisis)
    return vol_calm * (1.0 - p) + vol_crisis * p
