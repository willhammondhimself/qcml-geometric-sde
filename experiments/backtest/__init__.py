"""
PnL Backtest Framework for Geometric Regime Detection

Converts regime detection signals to portfolio positions and measures
economic value. Designed to withstand scrutiny from quantitative
researchers (SIG, Jane Street).

Key design:
- Signal at close(t) affects weight at open(t+1) — strictly causal
- Transaction costs: 0.5-1.5 bps + 0.5 bps commission
- In-sample/OOS split: parameters on 2005-2019, frozen for 2020-2024
- Ledoit-Wolf (2008) test for Sharpe ratio comparison
"""
