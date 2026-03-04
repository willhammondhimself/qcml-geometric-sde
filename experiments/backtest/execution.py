"""
Execution model: transaction costs, turnover, and holding period constraints.
"""

import numpy as np


# Default transaction costs (in basis points)
DEFAULT_COSTS = {
    'SPY': 0.5,
    'DIA': 0.5,
    'QQQ': 0.5,
    'IWM': 1.0,
    'EFA': 1.5,
}
DEFAULT_COMMISSION = 0.5  # bps per trade


def compute_turnover(weights):
    """Compute daily turnover from weight time series.

    Args:
        weights: Weight array (T,) or (T, n_assets).

    Returns:
        turnover: Daily absolute weight change (T,).
    """
    if weights.ndim == 1:
        return np.abs(np.diff(weights, prepend=weights[0]))
    else:
        return np.sum(np.abs(np.diff(weights, axis=0, prepend=weights[0:1])), axis=1)


def apply_transaction_costs(
    gross_returns, weights, symbols=None,
    cost_bps=None, commission_bps=DEFAULT_COMMISSION,
):
    """Apply transaction costs to gross returns.

    Args:
        gross_returns: Daily gross returns (T,) or (T, n_assets).
        weights: Target weights (T,) or (T, n_assets).
        symbols: List of ticker symbols (for per-asset costs).
        cost_bps: Override cost in bps (if None, use per-asset defaults).
        commission_bps: Commission in bps per trade.

    Returns:
        net_returns: Daily returns after transaction costs (T,).
        cost_series: Daily cost drag (T,).
    """
    turnover = compute_turnover(weights)

    if cost_bps is not None:
        total_cost_bps = cost_bps + commission_bps
    elif symbols is not None and weights.ndim == 2:
        # Per-asset costs
        asset_costs = np.array([
            DEFAULT_COSTS.get(s, 1.0) + commission_bps
            for s in symbols
        ])
        total_cost_bps = None  # handled per-asset below
    else:
        total_cost_bps = 1.0 + commission_bps  # default 1.5 bps

    if total_cost_bps is not None:
        cost_series = turnover * total_cost_bps / 10000.0
    else:
        # Per-asset turnover * per-asset cost
        asset_turnover = np.abs(np.diff(weights, axis=0, prepend=weights[0:1]))
        cost_series = np.sum(asset_turnover * asset_costs[None, :] / 10000.0, axis=1)

    # Compute portfolio gross return
    if gross_returns.ndim == 2 and weights.ndim == 2:
        portfolio_gross = np.sum(gross_returns * weights, axis=1)
    elif gross_returns.ndim == 1:
        portfolio_gross = gross_returns * (weights if weights.ndim == 1 else weights[:, 0])
    else:
        portfolio_gross = gross_returns

    net_returns = portfolio_gross - cost_series
    return net_returns, cost_series


def apply_min_holding_period(weights, min_days=5):
    """Enforce minimum holding period by smoothing rapid weight changes.

    Args:
        weights: Weight array (T,).
        min_days: Minimum number of days between weight changes.

    Returns:
        smoothed_weights: Weight array with holding constraint.
    """
    smoothed = weights.copy()
    last_change = 0

    for t in range(1, len(smoothed)):
        if abs(smoothed[t] - smoothed[t - 1]) > 0.01:
            if t - last_change < min_days:
                smoothed[t] = smoothed[t - 1]
            else:
                last_change = t

    return smoothed
