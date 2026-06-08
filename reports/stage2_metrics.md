# Stage 2 — Validation Metrics

Target `log_market_value`; train 2021-22/22-23/23-24, validate 2024-25 (D-01). Trained on ACTUAL observed stats (D-02). Baseline = train-median predictor. `log_market_value_lag1` EXCLUDED from features (avoids 'copy last year's value'; ablation later). `MAE_eur` = mean |€| error via expm1.

| pos | model | n_train | n_val | MAE_log | MAE_€M | medAE_€M | R² | Δ_base% |
|---|---|---|---|---|---|---|---|---|
| DEF | Stacked ★ | 2805 | 735 | 0.467 | 4.08 | 1.14 | 0.846 | 62.9 |
| DEF | Ridge | 2805 | 735 | 0.537 | 16.76 | 1.36 | 0.762 | 57.3 |
| DEF | XGBoost | 2805 | 735 | 0.556 | 4.94 | 1.27 | 0.792 | 55.8 |
| DEF | MLP | 2805 | 735 | 0.560 | 78284.99 | 1.50 | 0.663 | 55.5 |
| FWD | Stacked ★ | 1277 | 293 | 0.448 | 4.56 | 1.70 | 0.875 | 65.9 |
| FWD | Ridge | 1277 | 293 | 0.489 | 5.59 | 1.81 | 0.836 | 62.8 |
| FWD | XGBoost | 1277 | 293 | 0.526 | 6.82 | 1.54 | 0.829 | 60.0 |
| FWD | MLP | 1277 | 293 | 0.746 | 24.40 | 3.19 | 0.654 | 43.3 |
| GK | Ridge ★ | 637 | 184 | 0.490 | 2.56 | 1.05 | 0.814 | 59.9 |
| GK | Stacked | 637 | 184 | 0.612 | 3.70 | 0.92 | 0.708 | 49.9 |
| GK | XGBoost | 637 | 184 | 0.684 | 4.08 | 1.00 | 0.631 | 44.0 |
| GK | MLP | 637 | 184 | 1.138 | 3790.13 | 2.90 | -0.524 | 6.8 |
| MID | MLP ★ | 3515 | 1016 | 0.441 | 4.23 | 1.39 | 0.850 | 64.9 |
| MID | Stacked | 3515 | 1016 | 0.458 | 4.70 | 1.30 | 0.852 | 63.5 |
| MID | XGBoost | 3515 | 1016 | 0.538 | 5.93 | 1.40 | 0.801 | 57.2 |
| MID | Ridge | 3515 | 1016 | 0.542 | 7.30 | 1.62 | 0.790 | 56.9 |

★ = best (lowest log-MAE) per position. R²>0.90 would warrant a leakage recheck.
