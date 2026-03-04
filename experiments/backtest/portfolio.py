"""
Portfolio construction: vol targeting and regime-adjusted weights.

All functions are pure (no side effects, no state). Takes signals and returns
target weights.
"""

import numpy as np
import pandas as pd


def vol_target_weight(returns, target_vol=0.10, lookback=20, floor=0.0, cap=2.0):
    """Compute vol-targeting weight for a return series.

    Weight = target_vol / realized_vol. Capped and floored.

    Args:
        returns: Daily return series (T,).
        target_vol: Annualized target volatility (default: 10%).
        lookback: Lookback window for realized vol estimation.
        floor: Minimum weight (default: 0.0 = can go to cash).
        cap: Maximum weight (default: 2.0 = max 2x leverage).

    Returns:
        weights: Array (T,) of target weights.
    """
    returns = pd.Series(returns)
    realized_vol = returns.rolling(lookback, min_periods=5).std() * np.sqrt(252)
    realized_vol = realized_vol.clip(lower=0.01)  # avoid division by tiny vol

    raw_weight = target_vol / realized_vol
    weights = raw_weight.clip(lower=floor, upper=cap)
    weights = weights.fillna(floor)

    return weights.values


def regime_adjusted_weight(
    base_weight, p_crisis, crisis_threshold=0.5,
    crisis_weight_multiplier=0.0, ramp_width=0.3,
    ramp_type='sigmoid',
):
    """Adjust portfolio weight based on regime probability.

    Smoothly reduces weight as P(crisis) increases above threshold.
    Supports both sigmoid and linear ramp types.

    Args:
        base_weight: Base weight from vol targeting (T,).
        p_crisis: P(crisis) signal (T,). NaN treated as 0.
        crisis_threshold: P(crisis) center point for sigmoid / start for linear.
        crisis_weight_multiplier: Weight multiplier at max P(crisis).
            0.0 = go flat, -0.5 = modest short.
        ramp_width: Width of smooth transition (P(crisis) units).
            For sigmoid: controls steepness (k = 6/ramp_width).
            For linear: width of linear ramp zone.
        ramp_type: 'sigmoid' (default) or 'linear'.

    Returns:
        adjusted_weight: Array (T,).
    """
    p = np.where(np.isnan(p_crisis), 0.0, p_crisis)

    if ramp_type == 'sigmoid':
        # Sigmoid: continuous S-curve centered at crisis_threshold
        # k = 6/ramp_width gives ~5% at threshold-ramp_width/2, ~95% at threshold+ramp_width/2
        k = 6.0 / max(ramp_width, 1e-6)
        alpha = 1.0 / (1.0 + np.exp(-k * (p - crisis_threshold)))
    else:
        # Linear ramp (original behavior)
        alpha = np.clip((p - crisis_threshold) / max(ramp_width, 1e-6), 0.0, 1.0)

    multiplier = 1.0 - alpha * (1.0 - crisis_weight_multiplier)

    return base_weight * multiplier


def equal_weight_allocation(n_assets, total_weight=1.0):
    """Equal-weight allocation across n assets.

    Args:
        n_assets: Number of assets.
        total_weight: Total portfolio weight.

    Returns:
        Array (n_assets,) of weights.
    """
    return np.full(n_assets, total_weight / n_assets)
