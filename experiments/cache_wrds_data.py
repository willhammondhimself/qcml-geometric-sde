#!/usr/bin/env python3
"""Cache all WRDS data needed by experiment scripts.

Run this ONCE from an interactive terminal (Duo push required):

    python experiments/cache_wrds_data.py

After caching, all experiment scripts will read from local parquet
files and won't need a WRDS connection.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.wrds_data_loader import fetch_wrds_equities


def main():
    print("=== WRDS Data Caching ===")
    print("Accept the Duo push when prompted.\n")

    # SPY + DIA: covers regime_comparison, enhanced_comparison, walk_forward,
    # honest_hpo_sweep, online_regime_evaluation, etc.
    print("[1/3] SPY + DIA (2005-2024)...")
    df1 = fetch_wrds_equities(['SPY', 'DIA'], '2005-01-01', '2024-12-31')
    print(f"  Cached: {df1.shape}")

    # Multi-asset: covers multi_asset_revalidation, poster_evaluation
    print("[2/3] SPY + DIA + QQQ + IWM + EFA (1995-2024)...")
    df2 = fetch_wrds_equities(
        ['SPY', 'DIA', 'QQQ', 'IWM', 'EFA'], '1995-01-01', '2024-12-31'
    )
    print(f"  Cached: {df2.shape}")

    # Cross-asset: covers cross_asset_generalization
    print("[3/3] Bond/commodity/FX ETFs (2003-2024)...")
    df3 = fetch_wrds_equities(
        ['AGG', 'TLT', 'HYG', 'LQD', 'GLD', 'USO', 'FXE', 'UUP'],
        '2003-01-01', '2024-12-31',
    )
    print(f"  Cached: {df3.shape}")

    print("\n=== Done! All data cached to data/wrds_cache/ ===")
    print("Experiment scripts will now use cached data automatically.")


if __name__ == "__main__":
    main()
