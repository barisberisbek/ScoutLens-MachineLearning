# Features Summary (Phase 4)

`unified_panel.parquet` (101 cols) → `features.parquet` (195 cols): **+94 engineered features**, 19,356 rows.

## per-90 / per-match (M1) (10)

| column | % null |
|---|---|
| xg_per_90 | 45.5 |
| xag_per_90 | 86.4 |
| npxg_per_90 | 45.5 |
| understat_xa_per_90 | 44.9 |
| understat_xg_per_90 | 44.9 |
| saves_per_90 | 92.3 |
| tackles_won_per_90 | 0.1 |
| interceptions_per_90 | 0.1 |
| shots_on_target_against_per_90 | 92.3 |
| clean_sheets_per_match | 92.3 |

## composites (M2) (6)

| column | % null |
|---|---|
| goal_threat | 45.5 |
| creative_threat | 44.6 |
| defensive_actions | 0.1 |
| shooting_efficiency | 34.7 |
| attacking_output | 0.0 |
| composite_completeness | 0.0 |

## z-scores (M3) (30)

| column | % null |
|---|---|
| goals_per_90_z_pos | 0.0 |
| goals_per_90_z_league | 0.0 |
| assists_per_90_z_pos | 0.0 |
| assists_per_90_z_league | 0.0 |
| shots_per_90_z_pos | 0.0 |
| shots_per_90_z_league | 0.0 |
| xg_per_90_z_pos | 45.5 |
| xg_per_90_z_league | 45.5 |
| xag_per_90_z_pos | 86.4 |
| xag_per_90_z_league | 86.4 |
| npxg_per_90_z_pos | 45.5 |
| npxg_per_90_z_league | 45.5 |
| tackles_won_per_90_z_pos | 0.1 |
| tackles_won_per_90_z_league | 0.1 |
| interceptions_per_90_z_pos | 0.1 |
| interceptions_per_90_z_league | 0.1 |
| saves_per_90_z_pos | 92.3 |
| saves_per_90_z_league | 92.3 |
| clean_sheets_per_match_z_pos | 92.3 |
| clean_sheets_per_match_z_league | 92.3 |
| minutes_played_z_pos | 0.0 |
| minutes_played_z_league | 0.0 |
| matches_played_z_pos | 0.0 |
| matches_played_z_league | 0.0 |
| goal_threat_z_pos | 45.5 |
| goal_threat_z_league | 45.5 |
| defensive_actions_z_pos | 0.1 |
| defensive_actions_z_league | 0.1 |
| creative_threat_z_pos | 44.6 |
| creative_threat_z_league | 45.1 |

## age curve (M4) (6)

| column | % null |
|---|---|
| age_squared | 0.0 |
| age_cubed | 0.0 |
| distance_from_peak | 0.0 |
| is_young | 0.0 |
| is_peak | 0.0 |
| is_declining | 0.0 |

## contract (M5) (3)

| column | % null |
|---|---|
| contract_years_until_expiry | 27.3 |
| is_expiring | 0.0 |
| is_long_contract | 0.0 |

## lag / trajectory (M6) (26)

| column | % null |
|---|---|
| goals_per_90_lag1 | 46.9 |
| delta_goals_per_90 | 46.9 |
| assists_per_90_lag1 | 46.9 |
| delta_assists_per_90 | 46.9 |
| shots_per_90_lag1 | 46.9 |
| delta_shots_per_90 | 46.9 |
| xg_per_90_lag1 | 69.1 |
| delta_xg_per_90 | 71.1 |
| xag_per_90_lag1 | 100.0 |
| delta_xag_per_90 | 100.0 |
| minutes_played_lag1 | 46.9 |
| delta_minutes_played | 46.9 |
| matches_played_lag1 | 46.9 |
| delta_matches_played | 46.9 |
| tackles_won_per_90_lag1 | 46.9 |
| delta_tackles_won_per_90 | 47.0 |
| interceptions_per_90_lag1 | 46.9 |
| delta_interceptions_per_90 | 47.0 |
| log_market_value_lag1 | 58.1 |
| delta_log_market_value | 63.1 |
| fifa_rating_lag1 | 55.0 |
| delta_fifa_rating | 56.7 |
| fifa_potential_lag1 | 55.0 |
| delta_fifa_potential | 71.2 |
| has_lag1 | 0.0 |
| consecutive_seasons | 0.0 |

## multipliers (M7) (1)

| column | % null |
|---|---|
| year_inflation_multiplier | 0.0 |

## categorical (M8) (11)

| column | % null |
|---|---|
| is_GK | 0.0 |
| is_DEF | 0.0 |
| is_MID | 0.0 |
| is_FWD | 0.0 |
| is_continent_Europe | 0.0 |
| is_continent_Africa | 0.0 |
| is_continent_SouthAmerica | 0.0 |
| is_continent_NorthAmerica | 0.0 |
| is_continent_Asia | 0.0 |
| is_continent_Oceania | 0.0 |
| is_continent_Other | 0.0 |

## fifa (M9) (1)

| column | % null |
|---|---|
| fifa_potential_gap | 38.5 |

## % null by position (new features, mean over group)

| position | mean % null (new cols) |
|---|---|
| DEF | 32.2 |
| FWD | 33.5 |
| GK | 25.8 |
| MID | 32.0 |

## Top-10 highest-null new columns (debug)

| column | % null |
|---|---|
| xag_per_90_lag1 | 100.0 |
| delta_xag_per_90 | 100.0 |
| clean_sheets_per_match_z_pos | 92.3 |
| saves_per_90_z_pos | 92.3 |
| saves_per_90 | 92.3 |
| clean_sheets_per_match | 92.3 |
| shots_on_target_against_per_90 | 92.3 |
| clean_sheets_per_match_z_league | 92.3 |
| saves_per_90_z_league | 92.3 |
| xag_per_90 | 86.4 |

Note: high null is expected for xG-based features (top-5 only), GK stats (null for outfield), lag features in 2021-22 (no prior season), and 2024-25 `fifa_potential_gap` (FC25 has no potential). No imputation is done here — Phase 5 handles NaN (tree models natively, linear via a pipeline imputer).
