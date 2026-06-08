"""Stage 2 — market-value valuation (log_market_value prediction).

Per-position models trained on ACTUAL observed stats + context (D-02 — never on Stage-1
projections). Temporal split: train 2021-22/22-23/23-24, validate 2024-25. Bonus §11.4
validation against real 2024-25 transfer fees.
"""
