# External lookup tables

Small, hand-maintained reference CSVs. **Committed to git** (unlike `data/raw/`).

| File | Columns | Notes |
|---|---|---|
| `league_value_multipliers.csv` | `league, multiplier` | **Heuristic starting points only.** Re-derived empirically in Phase 4 via a position-controlled regression of `log_market_value ~ league` (PROJECT_ROADMAP.md §10.5). Do not treat as final. |
| `uefa_coefficients.csv` | `league, uefa_coefficient` | Snapshot of UEFA association coefficients; league-prestige weighting. |
| `league_tier_map.csv` | `league, tier` | `1` = top-5, `2` = lower European (the 4 discovery leagues). |
| `continent_map.csv` | `nationality, continent_group` | Nationality → 5-way continent group (D-06). Mapped by FIFA confederation (UEFA→Europe, CONMEBOL→SouthAmerica, CAF→Africa, AFC→Asia, CONCACAF→NorthAmerica, OFC→Oceania). Australia is kept in Oceania despite AFC membership. Unmapped nationalities fall back to `Other` at runtime. Includes common spelling variants (e.g. Ireland/Republic of Ireland, South Korea/Korea Republic). |

When a source emits a league name that isn't a canonical key here, normalize it first via `LEAGUE_NAME_PREFIX_MAP` in `src/utils/constants.py`.
