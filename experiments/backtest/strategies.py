"""
Trading strategies that convert regime signals to portfolio positions.

All strategies take P(crisis) signal and asset returns as input.
Output: daily portfolio weights (T,) or (T, n_assets).
"""

import numpy as np
import pandas as pd
from .portfolio import vol_target_weight, regime_adjusted_weight
from .execution import apply_min_holding_period
from .risk_management import drawdown_circuit_breaker, apply_position_limits


def geometric_long_flat(
    spy_returns, p_crisis,
    target_vol=0.10, crisis_threshold=0.5,
    min_holding=3, max_drawdown=0.20,
):
    """Long SPY, go flat when P(crisis) high. Vol-targeted.

    Args:
        spy_returns: Daily SPY returns (T,).
        p_crisis: P(crisis) signal (T,).
        target_vol: Annualized target vol.
        crisis_threshold: P(crisis) above which to reduce.
        min_holding: Minimum holding period in days.
        max_drawdown: Drawdown circuit breaker threshold.

    Returns:
        weights: Daily portfolio weights (T,).
        metadata: Dict of intermediate signals.
    """
    T = len(spy_returns)

    # Vol-targeted base weight
    base_w = vol_target_weight(spy_returns, target_vol=target_vol)

    # Regime adjustment (go flat in crisis) — sigmoid ramp
    adjusted_w = regime_adjusted_weight(
        base_w, p_crisis,
        crisis_threshold=crisis_threshold,
        crisis_weight_multiplier=0.0,
        ramp_width=0.3,
        ramp_type='sigmoid',
    )

    # Minimum holding period
    adjusted_w = apply_min_holding_period(adjusted_w, min_days=min_holding)

    # Position limits
    adjusted_w = apply_position_limits(adjusted_w, max_long=1.5, max_short=0.0)

    # Shift by 1 day (signal at t affects weight at t+1)
    weights = np.zeros(T)
    weights[1:] = adjusted_w[:-1]

    # Drawdown circuit breaker
    equity = np.cumprod(1 + spy_returns * weights)
    weights = drawdown_circuit_breaker(equity, weights, max_drawdown=max_drawdown)

    metadata = {
        'base_weight': base_w,
        'regime_adjusted_weight': adjusted_w,
    }
    return weights, metadata


def geometric_multi_asset(
    asset_returns, p_crisis, symbols,
    target_vol=0.10, crisis_threshold=0.5,
    min_holding=3,
):
    """Equal-weight multi-asset, reduce all on crisis signal. Vol-targeted.

    Args:
        asset_returns: Daily returns (T, n_assets).
        p_crisis: P(crisis) signal (T,).
        symbols: Asset ticker symbols.
        target_vol: Annualized target vol.
        crisis_threshold: P(crisis) above which to reduce.
        min_holding: Minimum holding period.

    Returns:
        weights: Daily weights (T, n_assets).
        metadata: Dict.
    """
    T, n_assets = asset_returns.shape

    # Equal-weight portfolio return for vol targeting
    ew_returns = np.mean(asset_returns, axis=1)
    base_w = vol_target_weight(ew_returns, target_vol=target_vol)

    # Regime adjustment — sigmoid ramp
    adjusted_w = regime_adjusted_weight(
        base_w, p_crisis,
        crisis_threshold=crisis_threshold,
        crisis_weight_multiplier=0.0,
        ramp_width=0.3,
        ramp_type='sigmoid',
    )
    adjusted_w = apply_min_holding_period(adjusted_w, min_days=min_holding)

    # Distribute equally across assets, shifted by 1 day
    weights = np.zeros((T, n_assets))
    for i in range(n_assets):
        weights[1:, i] = adjusted_w[:-1] / n_assets

    metadata = {'base_weight': base_w}
    return weights, metadata


def geometric_long_short(
    spy_returns, p_crisis,
    target_vol=0.10, crisis_threshold=0.5,
    crisis_short_size=-0.3, min_holding=3,
    max_drawdown=0.20,
):
    """Long in calm, modest short in crisis. Vol-targeted.

    Args:
        spy_returns: Daily SPY returns (T,).
        p_crisis: P(crisis) signal (T,).
        target_vol: Annualized target vol.
        crisis_threshold: P(crisis) above which to go short.
        crisis_short_size: Short weight during crisis (negative).
        min_holding: Minimum holding period.
        max_drawdown: Drawdown circuit breaker.

    Returns:
        weights: Daily weights (T,).
        metadata: Dict.
    """
    T = len(spy_returns)

    base_w = vol_target_weight(spy_returns, target_vol=target_vol)

    # Regime adjustment (go short in crisis) — sigmoid ramp
    adjusted_w = regime_adjusted_weight(
        base_w, p_crisis,
        crisis_threshold=crisis_threshold,
        crisis_weight_multiplier=crisis_short_size,
        ramp_width=0.3,
        ramp_type='sigmoid',
    )
    adjusted_w = apply_min_holding_period(adjusted_w, min_days=min_holding)
    adjusted_w = apply_position_limits(adjusted_w, max_long=1.5, max_short=-0.5)

    # Shift by 1 day
    weights = np.zeros(T)
    weights[1:] = adjusted_w[:-1]

    # Drawdown circuit breaker
    equity = np.cumprod(1 + spy_returns * weights)
    weights = drawdown_circuit_breaker(equity, weights, max_drawdown=max_drawdown)

    metadata = {'base_weight': base_w}
    return weights, metadata


def geometric_continuous(
    spy_returns, p_crisis,
    target_vol=0.10, crisis_threshold=0.5,
    sigmoid_k=10.0, min_holding=3, max_drawdown=0.20,
):
    """Continuous position sizing via sigmoid function of P(crisis).

    Instead of binary {0, 1} position, uses smooth sigmoid:
    position = 1 / (1 + exp(k * (P_crisis - threshold)))

    Args:
        spy_returns: Daily SPY returns (T,).
        p_crisis: P(crisis) signal (T,).
        target_vol: Annualized target vol.
        crisis_threshold: Sigmoid center point.
        sigmoid_k: Steepness of sigmoid (higher = sharper transition).
        min_holding: Minimum holding period in days.
        max_drawdown: Drawdown circuit breaker threshold.

    Returns:
        weights: Daily portfolio weights (T,).
        metadata: Dict of intermediate signals.
    """
    T = len(spy_returns)

    base_w = vol_target_weight(spy_returns, target_vol=target_vol)

    p = np.where(np.isnan(p_crisis), 0.0, p_crisis)
    sigmoid_multiplier = 1.0 / (1.0 + np.exp(sigmoid_k * (p - crisis_threshold)))

    adjusted_w = base_w * sigmoid_multiplier
    adjusted_w = apply_min_holding_period(adjusted_w, min_days=min_holding)
    adjusted_w = apply_position_limits(adjusted_w, max_long=1.5, max_short=0.0)

    weights = np.zeros(T)
    weights[1:] = adjusted_w[:-1]

    equity = np.cumprod(1 + spy_returns * weights)
    weights = drawdown_circuit_breaker(equity, weights, max_drawdown=max_drawdown)

    metadata = {
        'base_weight': base_w,
        'sigmoid_multiplier': sigmoid_multiplier,
    }
    return weights, metadata


def multi_signal_strategy(
    spy_returns, p_crisis,
    target_vol=0.10, crisis_threshold=0.5,
    geometric_weight=0.50, min_holding=3, max_drawdown=0.20,
    lookback_mom=200, lookback_vol=20,
):
    """Combine geometric P(crisis) with classical signals (momentum + vol).

    Geometric signal gets geometric_weight, classical signals share the rest.
    Classical signals: realized vol percentile and momentum (price vs 200d MA).

    Args:
        spy_returns: Daily SPY returns (T,).
        p_crisis: P(crisis) signal (T,).
        target_vol: Annualized target vol.
        crisis_threshold: P(crisis) center point.
        geometric_weight: Weight for geometric signal (0-1).
        min_holding: Minimum holding period.
        max_drawdown: Drawdown circuit breaker.
        lookback_mom: Momentum lookback (days).
        lookback_vol: Vol lookback (days).

    Returns:
        weights: Daily portfolio weights (T,).
        metadata: Dict.
    """
    T = len(spy_returns)
    prices = np.cumprod(1 + spy_returns)

    # Classical signal 1: realized vol percentile -> P(crisis)
    returns_s = pd.Series(spy_returns)
    realized_vol = returns_s.rolling(lookback_vol, min_periods=5).std() * np.sqrt(252)
    vol_pctile = realized_vol.expanding(min_periods=60).rank(pct=True).fillna(0.5).values

    # Classical signal 2: momentum -> below MA = bearish
    ma = pd.Series(prices).rolling(lookback_mom, min_periods=60).mean().ffill().values
    mom_signal = np.where(prices < ma, 0.7, 0.3)

    # Blend: geometric + vol_pctile + momentum
    classical_weight = 1.0 - geometric_weight
    p = np.where(np.isnan(p_crisis), 0.5, p_crisis)
    blended_signal = (
        geometric_weight * p +
        classical_weight * 0.5 * vol_pctile +
        classical_weight * 0.5 * mom_signal
    )

    base_w = vol_target_weight(spy_returns, target_vol=target_vol)
    adjusted_w = regime_adjusted_weight(
        base_w, blended_signal,
        crisis_threshold=crisis_threshold,
        crisis_weight_multiplier=0.0,
        ramp_width=0.3,
        ramp_type='sigmoid',
    )
    adjusted_w = apply_min_holding_period(adjusted_w, min_days=min_holding)
    adjusted_w = apply_position_limits(adjusted_w, max_long=1.5, max_short=0.0)

    weights = np.zeros(T)
    weights[1:] = adjusted_w[:-1]

    equity = np.cumprod(1 + spy_returns * weights)
    weights = drawdown_circuit_breaker(equity, weights, max_drawdown=max_drawdown)

    metadata = {
        'base_weight': base_w,
        'blended_signal': blended_signal,
        'vol_pctile': vol_pctile,
        'mom_signal': mom_signal,
    }
    return weights, metadata
