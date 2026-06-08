# EDA Findings — Phase 3

Exploratory analysis of `data/processed/unified_panel.parquet` (19,356 player-seasons ×
101 cols, 9 leagues × 4 seasons, 9,072 players). Full analysis + figures in
`notebooks/01_data_exploration.ipynb` and `reports/figures/`. Source code: `src/eda/`.

## Key Empirical Findings (headline)

1. **Log MV transform empirically justified** — raw market value is extreme right-skew
   (skew **4.2**, kurtosis **27**); `log1p` is ≈normal (skew **−0.03**, Shapiro W **0.99**).
   Confirms Stage-2's `log1p` target (D-05).
2. **League premium spans 3.8×–18.8×** — PL median €15M vs Süper Lig €0.8M (18.8×); La Liga
   vs Süper Lig 3.8×. Comparable players are far cheaper in lower tiers (the discovery thesis, D-08).
3. **MV median MID > DEF ≈ FWD > GK** (MID €3.0M, DEF €2.5M, FWD €2.5M, GK €1.2M). FWD's
   superstar tail lifts its **mean** (€9.6M) above MID (€8.8M) but **not** its median —
   position stratification (D-03) is justified.
4. **YoY MV inflation +10% → +20%** (2021-22 €2.0M → 2024-25 €3.0M) — empirical basis for
   the D-07 year-inflation multiplier feature.
5. **Synthetic cohort (18%)** — players with no Transfermarkt match: no market value but full
   FBref stats. Concentrated in lower-4 + older seasons = the lower-league discovery target.

## Observations

1. **Coverage is balanced** — ~480–620 player-rows per league × season; no league/season
   under-represented → a single global model + league features (D-04) keeps statistical power.
2. **Two distinct missingness regimes.** Structural: `saves` 92% null overall but **0.1%
   within GK** (GK stats simply don't apply to outfielders). Source-gap (by design): xG ~45%
   null (lower-4 has none; pre-2024-25 top-5 from Understat), xag ~86% (Kaggle 2024-25 top-5
   only), FC25 `potential` 100% null in 2024-25, MV ~31% null overall (58% non-null in 2024-25).
3. **No accidental gaps.** The 106 soccerdata-empty FBref extended columns were dropped in
   Phase 2 (P2-D6); every remaining null is explained (structural or source-gap).
4. **Position taxonomy is sound** — GK ≈ 8% of rows everywhere; 11.2% of players change
   primary position across seasons (genuine winger MID↔FWD reclassifications, not errors).
5. **Age is well-formed** — median 26.0, range 14.8–42.8, only 2 implausible youth rows.
6. **Value curve peaks ~25–26 for outfielders, ~30 for GKs** (LOWESS) — age-curve features
   must be position-specific.
7. **xG calibration r = 0.93** — Understat runs slightly conservative (mean resid −0.14) vs
   Kaggle (−0.02): comparable on rank, not absolute scale (confirms the Phase-1B note).
8. **Lower-4 forwards score at comparable rates** to mid-table top-5 forwards, yet cost
   3.8–18.8× less — the core discovery signal made visual.
9. **82% TM-matched** (71% exact); the 18% synthetic are the discovery cohort, auto-excluded
   from Stage-2 training by `market_value.notna()`.
10. **No outlier data errors** — top-10 MVs are all genuine elite players; the 19
    low-minute/high-MV rows are injuries/loans/new signings; max minutes 3,724 < the 3,800 cap.

## ⚠️ Structural data finding (drove the §6.2 revision)

**Soccerdata never delivered the FBref *extended* stats** — total tackles, blocks, clearances,
passes-completed, key passes, touches, carries, take-ons, SCA/GCA, PSxG, progressive
passes/carries, and aerials (which don't exist at all). Verified at source: the cached HTML
has `data-stat="tackles"` empty while `tackles_won`/`interceptions` are populated in the same
rows. Not a parse bug; unrecoverable without an alternative source. **61 of 167 performance
columns survived.** Decision (P2-D6): drop the 106 all-null columns; revise §6.2 Stage-1
targets to the available stat universe.

## Phase-4 action items (feature engineering)

- **Per-90s** from survivors: goals, assists, shots, shots_on_target, `tackles_won`,
  `interceptions`, saves, goals_against (the §10.1 mandatory per-90s, restricted to what exists).
- **Position-conditional handling**: encode an `is_gk` mask / separate GK feature block;
  don't impute structural nulls. Leave lower-4 xG NaN (tree models tolerate it).
- **Age-curve features per position** (distance-from-peak, age²) — peaks differ (GK ~30 vs ~25).
- **Trajectory features**: prioritise lag-1 and a 2-season slope (Stage-1 inputs, §10.9).
- **xG**: keep `xg` and `understat_xa` separate; defer any `understat_xa→xag` factor (P2-D3).
- **Year-inflation multiplier** from the +10–20% YoY medians (D-07).

## Phase-5/6 modeling cautions

- **Sample sizes**: Stage-2 train (2021-23, MV non-null) ≈ 13.4k, validation (2024-25) ≈ 1.6k;
  per-position validation ≈ 250–550 rows (GK smallest) — adequate but watch GK variance.
- **Reduced Stage-1 targets (revised §6.2, finalize in Phase 5):**
  - GK: `saves_per_90`, `save_pct`, `clean_sheets_per_90`, `goals_against_per_90`.
  - DEF: `tackles_won_per_90`, `interceptions_per_90`, `goals_per_90`.
  - MID: `goals_per_90`, `assists_per_90`, `xg_per_90`, `xag_per_90`, `tackles_won_per_90`, `interceptions_per_90`.
  - FWD: `xg_per_90`, `goals_per_90`, `assists_per_90`, `shots_per_90`, `npxg`/`understat_xa`.
- **GK target alternatives (psxg unavailable):** `clean_sheets_per_90` (primary), `save_pct`
  (secondary), `goals_against_per_90` (negative direction); custom composite deferred (no
  shots-faced/box data). Decide in Phase 5.
- **Stratify by `primary_position`** using the season-specific position (it legitimately shifts).
- **Temporal split only** (D-01); lower-4 xG stays NaN — Stage-1 for the discovery cohort
  relies on the surviving non-xG stats.
