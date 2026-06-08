# Stage 1 — Validation Metrics

Temporal split (D-01): train = (2021-22→22-23)+(2022-23→23-24), validation = (2023-24→24-25). Baseline = naive 'next == current'. `Δ%` = improvement over baseline MAE (positive = beats naive).

## Best model per target

| pos | target | model | n_train | n_val | MAE | baseline | Δ% | R² |
|---|---|---|---|---|---|---|---|---|
| DEF | goals_per_90 | XGBoost | 1510 | 717 | 0.0421 | 0.0499 | 15.6 | 0.02 |
| DEF | interceptions_per_90 | XGBoost | 1509 | 716 | 0.3129 | 0.3387 | 7.6 | 0.16 |
| DEF | tackles_won_per_90 | XGBoost | 1509 | 716 | 0.2750 | 0.3281 | 16.2 | 0.31 |
| FWD | assists_per_90 | Stacked | 621 | 258 | 0.0792 | 0.0953 | 16.9 | 0.12 |
| FWD | goals_per_90 | Stacked | 621 | 258 | 0.1602 | 0.1852 | 13.5 | 0.21 |
| FWD | npxg_per_90 | Stacked | 388 | 148 | 0.1177 | 0.1409 | 16.5 | 0.21 |
| FWD | shots_per_90 | Stacked | 621 | 257 | 0.5795 | 0.6625 | 12.5 | 0.21 |
| FWD | understat_xa_per_90 | XGBoost | 395 | 150 | 0.0669 | 0.0684 | 2.1 | 0.28 |
| FWD | xg_per_90 | Stacked | 388 | 148 | 0.1287 | 0.1547 | 16.8 | 0.27 |
| GK ⚠️ | clean_sheets_per_match | Stacked | 325 | 159 | 0.1010 | 0.1288 | 21.6 | 0.07 |
| GK ⚠️ | goals_against_per_90 | Stacked | 325 | 159 | 0.2850 | 0.3650 | 21.9 | 0.14 |
| GK ⚠️ | save_pct | Stacked | 325 | 159 | 4.9083 | 6.5457 | 25.0 | -0.03 |
| GK ⚠️ | saves_per_90 | Stacked | 325 | 159 | 0.4855 | 0.5898 | 17.7 | 0.04 |
| MID | assists_per_90 | Stacked | 1785 | 936 | 0.0725 | 0.0944 | 23.3 | 0.23 |
| MID | goals_per_90 | XGBoost | 1785 | 936 | 0.0807 | 0.1013 | 20.3 | 0.29 |
| MID | interceptions_per_90 | Stacked | 1784 | 936 | 0.2356 | 0.2648 | 11.0 | 0.48 |
| MID | tackles_won_per_90 | Stacked | 1784 | 936 | 0.2943 | 0.3369 | 12.6 | 0.38 |
| MID | understat_xa_per_90 | XGBoost | 1088 | 522 | 0.0490 | 0.0573 | 14.4 | 0.48 |
| MID | xg_per_90 | XGBoost | 1068 | 516 | 0.0535 | 0.0587 | 8.7 | 0.55 |

⚠️ GK targets train on only n=325 pairs — metrics are higher-variance.

## All models

| pos | target | model | n_train | n_val | MAE | RMSE | R² | MAPE% | Δ% |
|---|---|---|---|---|---|---|---|---|---|
| DEF | goals_per_90 | XGBoost ★ | 1510 | 717 | 0.0421 | 0.0542 | 0.02 | 65.0 | 15.6 |
| DEF | goals_per_90 | Stacked | 1510 | 717 | 0.0432 | 0.0542 | 0.02 | 67.8 | 13.5 |
| DEF | goals_per_90 | Ridge | 1510 | 717 | 0.0502 | 0.0599 | -0.19 | 51.2 | -0.6 |
| DEF | goals_per_90 | MLP | 1510 | 717 | 0.0972 | 0.1242 | -4.11 | 67.2 | -94.8 |
| DEF | interceptions_per_90 | XGBoost ★ | 1509 | 716 | 0.3129 | 0.3991 | 0.16 | 33.6 | 7.6 |
| DEF | interceptions_per_90 | Stacked | 1509 | 716 | 0.3317 | 0.4354 | -0.00 | 29.9 | 2.1 |
| DEF | interceptions_per_90 | MLP | 1509 | 716 | 0.4117 | 0.5261 | -0.46 | 41.8 | -21.6 |
| DEF | interceptions_per_90 | Ridge | 1509 | 716 | 0.4656 | 0.5799 | -0.78 | 38.5 | -37.5 |
| DEF | tackles_won_per_90 | XGBoost ★ | 1509 | 716 | 0.2750 | 0.3564 | 0.31 | 31.1 | 16.2 |
| DEF | tackles_won_per_90 | Stacked | 1509 | 716 | 0.2802 | 0.3590 | 0.30 | 31.5 | 14.6 |
| DEF | tackles_won_per_90 | Ridge | 1509 | 716 | 0.2986 | 0.3789 | 0.22 | 33.4 | 9.0 |
| DEF | tackles_won_per_90 | MLP | 1509 | 716 | 0.3920 | 0.5228 | -0.49 | 44.8 | -19.5 |
| FWD | assists_per_90 | Stacked ★ | 621 | 258 | 0.0792 | 0.1012 | 0.12 | 29.4 | 16.9 |
| FWD | assists_per_90 | Ridge | 621 | 258 | 0.0836 | 0.1103 | -0.05 | 38.9 | 12.4 |
| FWD | assists_per_90 | XGBoost | 621 | 258 | 0.0862 | 0.1106 | -0.05 | 37.2 | 9.6 |
| FWD | assists_per_90 | MLP | 621 | 258 | 0.2169 | 0.2707 | -5.30 | 109.8 | -127.6 |
| FWD | goals_per_90 | Stacked ★ | 621 | 258 | 0.1602 | 0.2079 | 0.21 | 43.9 | 13.5 |
| FWD | goals_per_90 | XGBoost | 621 | 258 | 0.1644 | 0.2119 | 0.18 | 44.0 | 11.3 |
| FWD | goals_per_90 | Ridge | 621 | 258 | 0.2020 | 0.2466 | -0.12 | 60.5 | -9.1 |
| FWD | goals_per_90 | MLP | 621 | 258 | 0.3036 | 0.3975 | -1.90 | 83.5 | -63.9 |
| FWD | npxg_per_90 | Stacked ★ | 388 | 148 | 0.1177 | 0.1464 | 0.21 | 35.5 | 16.5 |
| FWD | npxg_per_90 | XGBoost | 388 | 148 | 0.1254 | 0.1542 | 0.12 | 37.8 | 11.0 |
| FWD | npxg_per_90 | Ridge | 388 | 148 | 0.1288 | 0.1660 | -0.01 | 37.2 | 8.5 |
| FWD | npxg_per_90 | MLP | 388 | 148 | 0.2952 | 0.3779 | -4.26 | 90.7 | -109.6 |
| FWD | shots_per_90 | Stacked ★ | 621 | 257 | 0.5795 | 0.7258 | 0.21 | 24.7 | 12.5 |
| FWD | shots_per_90 | XGBoost | 621 | 257 | 0.5844 | 0.7404 | 0.18 | 24.4 | 11.8 |
| FWD | shots_per_90 | Ridge | 621 | 257 | 0.6398 | 0.8096 | 0.02 | 27.6 | 3.4 |
| FWD | shots_per_90 | MLP | 621 | 257 | 0.7441 | 0.9571 | -0.37 | 29.9 | -12.3 |
| FWD | understat_xa_per_90 | XGBoost ★ | 395 | 150 | 0.0669 | 0.0863 | 0.28 | 30.7 | 2.1 |
| FWD | understat_xa_per_90 | Stacked | 395 | 150 | 0.0677 | 0.0874 | 0.26 | 27.3 | 1.0 |
| FWD | understat_xa_per_90 | Ridge | 395 | 150 | 0.0768 | 0.0962 | 0.11 | 35.4 | -12.4 |
| FWD | understat_xa_per_90 | MLP | 395 | 150 | 0.2547 | 0.3129 | -8.44 | 147.6 | -272.5 |
| FWD | xg_per_90 | Stacked ★ | 388 | 148 | 0.1287 | 0.1598 | 0.27 | 34.6 | 16.8 |
| FWD | xg_per_90 | XGBoost | 388 | 148 | 0.1326 | 0.1662 | 0.21 | 35.8 | 14.3 |
| FWD | xg_per_90 | Ridge | 388 | 148 | 0.1392 | 0.1790 | 0.08 | 36.6 | 10.0 |
| FWD | xg_per_90 | MLP | 388 | 148 | 0.3062 | 0.3943 | -3.45 | 86.0 | -98.0 |
| GK | clean_sheets_per_match | Stacked ★ | 325 | 159 | 0.1010 | 0.1285 | 0.07 | 31.0 | 21.6 |
| GK | clean_sheets_per_match | XGBoost | 325 | 159 | 0.1027 | 0.1315 | 0.03 | 35.0 | 20.2 |
| GK | clean_sheets_per_match | Ridge | 325 | 159 | 0.1133 | 0.1536 | -0.33 | 36.7 | 12.0 |
| GK | clean_sheets_per_match | MLP | 325 | 159 | 0.2676 | 0.3530 | -6.01 | 104.8 | -107.8 |
| GK | goals_against_per_90 | Stacked ★ | 325 | 159 | 0.2850 | 0.3972 | 0.14 | 23.0 | 21.9 |
| GK | goals_against_per_90 | XGBoost | 325 | 159 | 0.3017 | 0.4213 | 0.04 | 23.9 | 17.3 |
| GK | goals_against_per_90 | Ridge | 325 | 159 | 0.3715 | 0.5154 | -0.44 | 29.5 | -1.8 |
| GK | goals_against_per_90 | MLP | 325 | 159 | 0.5103 | 0.6952 | -1.62 | 42.5 | -39.8 |
| GK | save_pct | Stacked ★ | 325 | 159 | 4.9083 | 6.2536 | -0.03 | 7.5 | 25.0 |
| GK | save_pct | XGBoost | 325 | 159 | 5.4075 | 6.8931 | -0.26 | 8.2 | 17.4 |
| GK | save_pct | Ridge | 325 | 159 | 6.5525 | 9.1192 | -1.20 | 9.9 | -0.1 |
| GK | save_pct | MLP | 325 | 159 | 17.2791 | 31.7732 | -25.67 | 25.8 | -164.0 |
| GK | saves_per_90 | Stacked ★ | 325 | 159 | 0.4855 | 0.6258 | 0.04 | 20.3 | 17.7 |
| GK | saves_per_90 | XGBoost | 325 | 159 | 0.5058 | 0.6536 | -0.04 | 20.7 | 14.2 |
| GK | saves_per_90 | Ridge | 325 | 159 | 0.6081 | 0.8393 | -0.72 | 24.9 | -3.1 |
| GK | saves_per_90 | MLP | 325 | 159 | 0.8313 | 1.3395 | -3.39 | 31.7 | -40.9 |
| MID | assists_per_90 | Stacked ★ | 1785 | 936 | 0.0725 | 0.0956 | 0.23 | 30.2 | 23.3 |
| MID | assists_per_90 | XGBoost | 1785 | 936 | 0.0739 | 0.0980 | 0.19 | 34.0 | 21.7 |
| MID | assists_per_90 | Ridge | 1785 | 936 | 0.0758 | 0.0996 | 0.16 | 32.6 | 19.7 |
| MID | assists_per_90 | MLP | 1785 | 936 | 0.1198 | 0.1579 | -1.11 | 62.9 | -26.9 |
| MID | goals_per_90 | XGBoost ★ | 1785 | 936 | 0.0807 | 0.1146 | 0.29 | 38.6 | 20.3 |
| MID | goals_per_90 | Stacked | 1785 | 936 | 0.0815 | 0.1147 | 0.29 | 36.2 | 19.5 |
| MID | goals_per_90 | Ridge | 1785 | 936 | 0.0832 | 0.1161 | 0.27 | 40.1 | 17.9 |
| MID | goals_per_90 | MLP | 1785 | 936 | 0.1909 | 0.2515 | -2.41 | 102.3 | -88.4 |
| MID | interceptions_per_90 | Stacked ★ | 1784 | 936 | 0.2356 | 0.3115 | 0.48 | 38.4 | 11.0 |
| MID | interceptions_per_90 | XGBoost | 1784 | 936 | 0.2383 | 0.3126 | 0.48 | 42.8 | 10.0 |
| MID | interceptions_per_90 | Ridge | 1784 | 936 | 0.2697 | 0.3489 | 0.35 | 39.1 | -1.9 |
| MID | interceptions_per_90 | MLP | 1784 | 936 | 0.3217 | 0.4148 | 0.08 | 59.0 | -21.5 |
| MID | tackles_won_per_90 | Stacked ★ | 1784 | 936 | 0.2943 | 0.3772 | 0.38 | 34.9 | 12.6 |
| MID | tackles_won_per_90 | XGBoost | 1784 | 936 | 0.2948 | 0.3801 | 0.37 | 34.2 | 12.5 |
| MID | tackles_won_per_90 | Ridge | 1784 | 936 | 0.3143 | 0.3953 | 0.32 | 38.0 | 6.7 |
| MID | tackles_won_per_90 | MLP | 1784 | 936 | 0.4088 | 0.5190 | -0.17 | 52.4 | -21.4 |
| MID | understat_xa_per_90 | XGBoost ★ | 1088 | 522 | 0.0490 | 0.0672 | 0.48 | 27.8 | 14.4 |
| MID | understat_xa_per_90 | Stacked | 1088 | 522 | 0.0510 | 0.0680 | 0.47 | 25.6 | 10.9 |
| MID | understat_xa_per_90 | Ridge | 1088 | 522 | 0.0573 | 0.0758 | 0.34 | 30.6 | -0.0 |
| MID | understat_xa_per_90 | MLP | 1088 | 522 | 0.2052 | 0.2670 | -7.22 | 120.5 | -258.2 |
| MID | xg_per_90 | XGBoost ★ | 1068 | 516 | 0.0535 | 0.0739 | 0.55 | 30.2 | 8.7 |
| MID | xg_per_90 | Stacked | 1068 | 516 | 0.0547 | 0.0737 | 0.55 | 28.0 | 6.8 |
| MID | xg_per_90 | Ridge | 1068 | 516 | 0.0581 | 0.0785 | 0.49 | 32.6 | 0.9 |
| MID | xg_per_90 | MLP | 1068 | 516 | 0.1771 | 0.2334 | -3.49 | 100.2 | -201.9 |