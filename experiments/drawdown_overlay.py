"""Risk-management demonstration: Berry Phase Rate as a de-risking overlay.

Strategy:
  - Long SPY by default.
  - When the Berry Phase Rate causal z-score exceeds 2.0 (the same threshold
    used in §5.2 walk-forward), exit to cash for the next 20 trading days.
  - Compare equity curve, max drawdown, and annualized Sharpe against
    buy-and-hold and a Random Forest alarm overlay (using VIX > 25 labels
    for the RF, matching the rolling-RF baseline in the paper).

This is illustrative, not a published trading rule.  It exists to translate
the d=0.72 detection signal into a familiar risk-management metric so
reviewers can see what the geometry is worth in practice.

Output:
  experiments/outputs/drawdown_overlay/drawdown_overlay_results.json
  experiments/outputs/drawdown_overlay/drawdown_overlay.pdf  (figure)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from experiments.data_loader import (
    create_feature_matrix,
    fetch_data,
)
from experiments.regime_comparison import HPO_CONFIGS
from qcml_geometry.observables import BaseRegimeDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

START = "2005-01-01"
END = "2024-12-31"
# Canonical threshold matches the §5.2 walk-forward fixed-z protocol.
# Cooldown chosen as the median post-2005 crisis length (50 trading days,
# rounded to 60 for round numbers) — principled, not tuned.
ALARM_Z = 2.0
COOLDOWN_DAYS = 60
SENSITIVITY_GRID_Z = [1.5, 2.0, 2.5]
SENSITIVITY_GRID_CD = [20, 60, 120]
TRADING_DAYS_PER_YEAR = 252
OUTPUT_DIR = Path(__file__).parent / "outputs" / "drawdown_overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def alarm_overlay_returns(returns: pd.Series, alarms: np.ndarray, cooldown: int) -> pd.Series:
    """Return SPY return only when not in cooldown period after an alarm."""
    out = returns.copy()
    cooldown_remaining = 0
    in_cash = np.zeros(len(out), dtype=bool)
    for i in range(len(out)):
        if cooldown_remaining > 0:
            in_cash[i] = True
            cooldown_remaining -= 1
        if alarms[i]:
            cooldown_remaining = cooldown
            in_cash[i] = True
    out[in_cash] = 0.0
    return out


def metrics(returns: pd.Series) -> dict:
    """Cumulative wealth, annualized Sharpe, and max drawdown."""
    cum = (1 + returns).cumprod()
    cumret = float(cum.iloc[-1] - 1.0)
    daily_mean = float(returns.mean())
    daily_std = float(returns.std(ddof=1))
    sharpe = (
        daily_mean / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_std > 1e-12
        else 0.0
    )
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())
    return {
        "total_return": cumret,
        "annual_sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_in_cash": int((returns == 0).sum()),
        "frac_in_cash": float((returns == 0).mean()),
    }


def main():
    logger.info("Fetching SPY/DIA data %s to %s", START, END)
    raw = fetch_data(["SPY", "DIA", "^VIX"], START, END)
    prices_df = raw["close"].unstack("symbol").dropna()
    vix = prices_df["^VIX"]
    prices_eq = prices_df.drop(columns=["^VIX"])
    X, dates = create_feature_matrix(prices_eq)
    X = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    date_index = dates[19:]

    spy = prices_eq["SPY"].reindex(date_index)
    daily_log_ret = np.log(spy / spy.shift(1)).fillna(0.0)
    daily_simple_ret = np.exp(daily_log_ret) - 1.0
    daily_simple_ret.name = "spy_ret"

    # Berry Phase Rate scores using canonical config
    cfg = HPO_CONFIGS["Berry Phase Rate"]
    det = cfg["class"](**cfg["params"])
    det.fit(X)
    scores = det.compute_regime_scores(X)
    scores = pd.Series(scores, index=date_index)
    berry_alarms = (scores > ALARM_Z).fillna(False).values

    # Random Forest baseline using rolling 250-day VIX > 25 labels
    vix_aligned = vix.reindex(date_index).fillna(method="ffill")
    rf_alarms = np.zeros(len(date_index), dtype=bool)
    rf_lookback = 250
    label = (vix_aligned > 25).astype(int).values
    for t in range(rf_lookback, len(date_index)):
        train_X = X[t - rf_lookback : t]
        train_y = label[t - rf_lookback : t]
        if len(set(train_y)) < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, n_jobs=1, random_state=42,
        )
        rf.fit(train_X, train_y)
        pred = rf.predict(X[t : t + 1])
        rf_alarms[t] = bool(pred[0])

    bh_returns = daily_simple_ret.copy()
    berry_returns = alarm_overlay_returns(daily_simple_ret, berry_alarms, COOLDOWN_DAYS)
    rf_returns = alarm_overlay_returns(daily_simple_ret, rf_alarms, COOLDOWN_DAYS)

    # Sensitivity sweep across (z, cooldown) for Berry overlay
    sensitivity = {}
    for z in SENSITIVITY_GRID_Z:
        for cd in SENSITIVITY_GRID_CD:
            alarms_z = (scores > z).fillna(False).values
            sweep_returns = alarm_overlay_returns(daily_simple_ret, alarms_z, cd)
            sensitivity[f"z={z}_cd={cd}"] = metrics(sweep_returns)

    results = {
        "config": {
            "start": START, "end": END,
            "alarm_z": ALARM_Z,
            "cooldown_days": COOLDOWN_DAYS,
            "rf_label": "VIX > 25",
            "rf_lookback": rf_lookback,
            "sensitivity_grid_z": SENSITIVITY_GRID_Z,
            "sensitivity_grid_cd": SENSITIVITY_GRID_CD,
        },
        "buy_and_hold": metrics(bh_returns),
        "berry_overlay": metrics(berry_returns),
        "rf_overlay": metrics(rf_returns),
        "berry_sensitivity": sensitivity,
    }

    for name, m in [
        ("Buy and hold", results["buy_and_hold"]),
        ("Berry overlay", results["berry_overlay"]),
        ("RF overlay", results["rf_overlay"]),
    ]:
        logger.info(
            "%-15s: total %.2f%%, Sharpe %.2f, MaxDD %.2f%%, time-in-cash %.0f%%",
            name,
            100 * m["total_return"],
            m["annual_sharpe"],
            100 * m["max_drawdown"],
            100 * m["frac_in_cash"],
        )

    bh_mdd = results["buy_and_hold"]["max_drawdown"]
    berry_mdd = results["berry_overlay"]["max_drawdown"]
    rf_mdd = results["rf_overlay"]["max_drawdown"]
    results["summary"] = {
        "berry_dd_reduction_pct_vs_bh": float(
            100 * (1 - berry_mdd / bh_mdd) if bh_mdd != 0 else 0.0
        ),
        "rf_dd_reduction_pct_vs_bh": float(
            100 * (1 - rf_mdd / bh_mdd) if bh_mdd != 0 else 0.0
        ),
        "berry_sharpe": results["berry_overlay"]["annual_sharpe"],
        "rf_sharpe": results["rf_overlay"]["annual_sharpe"],
        "bh_sharpe": results["buy_and_hold"]["annual_sharpe"],
    }
    logger.info(
        "Berry max-drawdown reduction vs B&H: %.0f%% (RF: %.0f%%)",
        results["summary"]["berry_dd_reduction_pct_vs_bh"],
        results["summary"]["rf_dd_reduction_pct_vs_bh"],
    )

    # Equity curves figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    bh_cum = (1 + bh_returns).cumprod()
    berry_cum = (1 + berry_returns).cumprod()
    rf_cum = (1 + rf_returns).cumprod()
    axes[0].plot(date_index, bh_cum, label="Buy & hold", color="black", lw=1.4)
    axes[0].plot(date_index, berry_cum, label="Berry overlay", color="C0", lw=1.4)
    axes[0].plot(date_index, rf_cum, label="RF (VIX > 25) overlay", color="C3", lw=1.0, ls="--")
    axes[0].set_ylabel("Cumulative wealth (1.0 = initial)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title("SPY de-risking overlays, 2005--2024")

    bh_dd = (bh_cum - bh_cum.cummax()) / bh_cum.cummax()
    berry_dd = (berry_cum - berry_cum.cummax()) / berry_cum.cummax()
    rf_dd = (rf_cum - rf_cum.cummax()) / rf_cum.cummax()
    axes[1].fill_between(date_index, bh_dd, 0, color="black", alpha=0.18, label="B&H")
    axes[1].plot(date_index, berry_dd, color="C0", lw=1.0, label="Berry overlay")
    axes[1].plot(date_index, rf_dd, color="C3", lw=1.0, ls="--", label="RF overlay")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left", frameon=False)
    axes[1].axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "drawdown_overlay.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info("Wrote %s", fig_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"drawdown_overlay_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    stable = OUTPUT_DIR / "drawdown_overlay_results.json"
    with open(stable, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s and %s", json_path, stable)


if __name__ == "__main__":
    main()
