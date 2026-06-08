# Project Memory — AML Final Project

## Current State
- **Current phase:** **PHASE 3 (EDA) COMPLETE.** Unified panel explored; soccerdata extended-stat gap caught + handled (panel 207→101 cols, §6.2 revised). Phase 1-2 complete. **Next: Phase 4 — Feature Engineering** (`src/features/`, one transform per file; `feature_forwarder.py` tests MANDATORY; per-90s/z-scores/age-curve/trajectory; honor reduced §6.2 targets + position-conditional missingness). Optional: populate `manual_id_overrides.csv` from top-500-MV synthetics (deferred).
- **Data inventory:** `data/processed/` → **`unified_panel.parquet` (19,356 rows × 207 cols, one per (player_id, season), 9 leagues × 4 seasons)**. `data/interim/` → kaggle/tm_*/fifa_ratings. `data/raw/` → fbref(396)/understat(20)/transfer_fees(548). **Committed:** `data/external/nationality_map.csv`, `data/manual/{match_log.csv, manual_id_overrides.csv (empty scaffold)}`, `reports/{decisions_log.md, name_resolution_audit.md, coverage_matrix.md}`.
- **Last updated:** 2026-06-08
- **Last session summary:** Phase 2 Sessions 1-3 done (one sitting). Full pipeline in `src/integration/unified_panel_builder.py`: `load_fbref_stats` (11-table collision-safe merge, curated clean names + `<table>__` namespaced tail, hard-error on unexpected dup) → `resolve_backbone` → `split_id_collisions`+`split_minutes_overflow` (namesake guards) → `collapse_split_season` (sum counting / minutes-weighted pct / per-90 recomputed from totals / max-minutes club) → attach xG (Kaggle 24-25 > Understat hist > NaN), MV, contract, FIFA, league-meta → `finalize_panel`. `scripts/build_panel.py` orchestrates; `src/integration/panel_reports.py` writes the 2 reports. **21 tests green** (13 name-resolution + 8 panel-builder). Run scripts via `PYTHONPATH=. .venv/Scripts/python.exe scripts/x.py` (bash env-var syntax; `set PYTHONPATH=` is a no-op in the Bash tool).

## Phase Completion Log
- [x] Phase 0 — Foundation (repo scaffolding)
- [x] Phase 1A — Kaggle loader
- [x] Phase 1B — FBref scraper
- [x] Phase 1C — Transfermarkt-datasets ingest
- [x] Phase 1D — 2025 transfer fees scraper
- [x] Phase 1E — FIFA ratings loader
- [x] Phase 1F — External lookup CSVs (UEFA, continent, etc.)  ← done early in Phase 0
- [x] Phase 2 — Data Integration
- [x] Phase 3 — EDA
- [ ] Phase 4 — Feature Engineering
- [ ] Phase 5 — Stage 1 Modeling (×4 positions)
- [ ] Phase 6 — Stage 2 Modeling (×4 positions)
- [ ] Phase 7 — Pipeline Assembly + Validation
- [ ] Phase 8 — Discovery Layer
- [ ] Phase 9 — Auxiliary Modules (clustering, patterns, SHAP)
- [ ] Phase 10 — Streamlit Demo
- [ ] Phase 11 — Report & Presentation

## Decisions Log (beyond PROJECT_ROADMAP.md)
Full entries in `reports/decisions_log.md`. Phase 2 (all user-approved 2026-06-08):
- **P2-D1** Min-minutes = KEEP + flag (`meets_min_minutes`/`min_minutes_threshold`), never drop in the panel; physical filter at Stage-1/2 entry (§6.3 authoritative on *where*).
- **P2-D2** Unmatched players = KEEP with deterministic `synthetic_<sha256[:12]>` id + `data_richness` col. They're the discovery cohort; Stage 2 auto-filters via `market_value.notna()`.
- **P2-D3** `xag` (FBref/Kaggle) and `understat_xa` (Understat) = SEPARATE columns; coalesce deferred to Phase 4 (may build position-based `understat_xa→xag` factor).

## Open Questions / Issues

### ✅ RESOLVED via pivot (found in Phase 3 EDA, 2026-06-08) — FBref EXTENDED stats empty
**Decision: pivot (no re-scrape).** Diagnostic confirmed soccerdata limitation (cached HTML
source-empty for these cols; `data-stat="tackles"`=0 numeric while `tackles_won`/`interceptions`
=580, both in HTML & parquet → our parser is correct, the data was never delivered).
**Action taken:** `finalize_panel` now drops all-null FBref stat columns → panel **207→101 cols**
(106 dropped); §6.2 targets revised (below + roadmap); EDA resumes on the clean panel.
EDA on `unified_panel.parquet` revealed that
~80 of the 146 clean stat columns are **100% null across all 36 league-seasons** — and
this is a **Phase 1B scrape defect, not a Phase 2 bug**. The defect:
- **Native** FBref tables (standard, shooting, keeper, misc, playing_time) = fully populated ✓.
- **Extended** tables (defense, passing, passing_types, possession, gca, keeper_adv) = only
  2-3 columns each survived; the rest are null. Survivors are erratic: defense kept only
  `90s, Tackles_TklW, Int`; passing kept only `90s, Ast`; possession 1/21; gca 1/17.
- **Root cause:** the cached raw HTML (`~/soccerdata/data/FBref/*.html`, 408 files) itself
  lacks the values — `data-stat="tackles"/"blocks"/"clearances"/"challenge_tackles"` are
  EMPTY in the player table, while `tackles_won`/`interceptions` (same rows) are full. So
  **NOT recoverable by re-parsing cache.** This is the same soccerdata FBref limitation that
  already dropped xG/npxG/xAG/progression-counts — but WIDER than realized (also tackles,
  passes_completed, key_passes, touches, carries, SCA/GCA, PSxG). Phase 1B's coverage
  validator only checked file/row presence; the cross-val only checked native stats
  (goals/assists/minutes/shots Pearson 1.000) → the gap went undetected.
- **Impact on §6.2 Stage-1 targets (SEVERE):** DEAD = tackles_per_90 (DEF/MID), blocks_per_90
  (DEF), key_passes_per_90 (MID), progressive_carries_per_90 (MID/FWD), sca_per_90 (FWD),
  psxg_per_90 (GK). SURVIVING = goals, assists, shots, interceptions, saves, save_pct,
  clean_sheets, goals_against, xg/xag(top-5 from Kaggle/Understat).
- **Decisive next step (proposed):** a 1-combo diagnostic spike — fetch ONE FBref defense
  page FRESH (live, not cache) and check whether `data-stat="tackles"` is populated. If YES
  → cache is stale/buggy, re-scrape the 6 extended types (cache miss → network, but no
  parser issue). If NO → genuine soccerdata/FBref-view limitation → fall back to Kaggle
  (full FBref stats but only 2024-25 top-5) + reduce §6.2 target lists to available stats.
- **Mitigation anchor:** Kaggle 2024-25 top-5 (158 cols) DOES carry the full stat set
  (confirmed Phase 1A) — usable if a single-season full-stat feature is ever needed.
- **61 of 167 perf cols survived.** Key survivors: goals, assists, shots, shots_on_target,
  `tackles_won`, `interceptions`, saves, save_pct, clean_sheets, goals_against, fouls,
  crosses, cards, team-context (on_off/plus_minus/ppm), + xg/xag/npxg/understat_xa (top-5).
  Dead: total tackles, blocks, clearances, aerials (none exist), passes_*, key_passes,
  touches, carries, take_ons, SCA/GCA, PSxG, prog-passes/carries.

### Phase 5 Stage-1 REDUCED targets (revised §6.2, finalize in Phase 5)
Soccerdata gap forced a §6.2 revision (roadmap updated, P2-D6). Candidate targets:
- **GK:** saves_per_90, save_pct, clean_sheets_per_90, goals_against_per_90.
- **DEF:** tackles_won_per_90, interceptions_per_90, goals_per_90.
- **MID:** goals_per_90, assists_per_90, xg_per_90, xag_per_90, tackles_won_per_90, interceptions_per_90.
- **FWD:** xg_per_90, goals_per_90, assists_per_90, shots_per_90, npxg/understat_xa.
NOTE: aerials do NOT exist in our data (user's draft `aerial_won_per_90` for DEF dropped →
replaced by `tackles_won`+`interceptions`, which survived).

### Phase 5 GK target alternatives (psxg unavailable)
Decide in Phase 5: `clean_sheets_per_90` (primary), `save_pct` (secondary),
`goals_against_per_90` (negative direction); custom composite deferred (no shots-faced/box
data). psxg_per_90 from §6.2 is permanently unavailable (soccerdata gap).

### Phase 2 carry-forward limitations (for Phase 3/4)
- **Known soccerdata data gaps (NaN in panel, fill in Phase 4):** no xG/npxG/xAG, no progression COUNTS (PrgP/PrgC), no aerial-duel cols, no PSxG (keeper_adv Expected-family null). xG filled from Kaggle (2024-25 top5) + Understat (hist top5) only → **lower-4 + many historical rows have NaN xG**; **psxg/aerials/PrgP only exist for Kaggle 2024-25 top5**. §6.2 GK target `psxg_per_90` and DEF target `aerial_won_pct`/`progressive_passes_per_90` are largely unavailable from FBref — Phase 4 must proxy (the `understat_xa→xag` position factor is one such planned proxy, P2-D3).
- **`is_loan` is a `False` placeholder** (D-15 loan detection deferred — TM snapshot lacks clean loan status).
- **`contract_remaining_months` uses the CURRENT TM contract date for ALL seasons** (TM only stores current), so historical seasons' values are approximate; 36% null where no contract date.
- **Same-name/same-year/same-nat namesakes** (e.g. two PT "Vitinha" born 2000) can't be disambiguated without birth-DATE → fall to `synthetic_split` ids (62 rows). These + the 17.7% `synthetic` are the **manual-override worklist** for the optional Session-3 pass.
- **MV 2024-25 top-5 = 1604** vs TM's ~1880 valued (recent MV sparse, known); panel MV non-null overall 69%.

### Name-resolution match rate (acceptable, improve via overrides)
Full FBref backbone (20,557 rows) → **82.0% TM-matched** (71.7% exact / 4.7% name+year / 5.6% fuzzy_nat / 0.3% fuzzy), **17.7% synthetic**. By league: top-5 + Eredivisie 83-87%, **Liga Portugal 73% (lowest)**, Belgian 78%, Süper Lig 81%. By season: 2024-25 87% → 2021-22 79%. **Why the gap:** `tm_players` only holds players whose *current* club is in the 9 leagues, so anyone who has since left → no TM metadata → synthetic. Older seasons + lower leagues have more such departures. These can't be discovery candidates anyway (no current MV), so synthetic-keep loses nothing for the pipeline. Manual top-500 overrides (Session 2-3) will lift high-MV matches. 3 rows have null FBref position (kept + warned).

### Lower-league xG gap (decide in Phase 4/5 — do NOT decide now)
soccerdata's FBref tables carry NO xG/npxG/xAG/progression-counts (any view, verified
2026-06-07). xG is sourced from **Kaggle** (2024-25 top-5) and **Understat** (top-5, all
seasons). **Lower-4 leagues have NO xG from any source** (Understat covers only top-5).
This affects Stage-1 inputs for the lower-league discovery cohort. Pragmatic resolution
order, to test empirically in Phase 4 (don't pre-decide):
1. Leave lower-league xG as NaN — tree models (XGBoost/LightGBM) handle NaN natively.
2. If Stage-2 residuals show systematic under-prediction for tier-2 leagues, compute a
   position-based xG proxy (`shots × position_avg_xg_per_shot`).
3. Only if (1)+(2) are insufficient: separate Stage-1 models for top-5 vs lower leagues.

### xG model difference (note for Phase 2/4)
Understat xG and FBref/Kaggle xG come from different models — Understat ~12% higher on
average (2024-25 PL: Pearson 0.993, Spearman 0.99). Rank is reliable; absolute values are
not directly comparable across sources. Pick ONE xG source per row consistently (prefer
Kaggle for 2024-25 top-5; Understat for historical top-5).

### Understat ↔ FBref/Kaggle player matching
No shared player_id across sources. Name-based match: ~92% exact (normalized) on 2024-25
PL; remainder are accent/short-name variants → fuzzy + (team, season) disambiguation in
Phase 2 `name_resolver`.

### Transfermarkt MV/transfer coverage limits (note for Phase 2 / Phase 7)
- **MV snapshots thin for recent seasons** (not a window issue — ±45d≈±180d): top-5
  MV-aligned players 21-22=5852, 22-23=6273, 23-24=2800, 24-25=1880. The davidcariboo
  dataset's 2024-25 valuation batches are sparser (~2000/batch) than 2021-22's. Effective
  Stage-2 train/val coverage is determined by the Phase-2 join (stats ∩ MV).
- **Transfer fees sparse** (D-09/§11.4 "absolute truth" will be SMALL): of 481 inbound
  top-5 transfers in Jun2024–Sep2025, only **121** have fee>0 (340 fee=0, 75 null), and
  the max recorded fee is only €23.7M — the marquee €60M+ moves have no recorded fee in
  this dataset. **RESOLVED by Phase 1D**: the custom TM scraper recovered 548 fee>0 inbound
  top-5 transfers (max €80M, 9 ≥€60M). Use Phase 1D's
  `transfer_fees_2024_25.csv` as the primary §11.4 reference; davidcariboo's `tm_transfers`
  is a secondary cross-check.

## Performance / Results Log
Phase 5 ve 6 başlayınca model performans metrikleri burada raporlanacak.

## Bug Log / Learnings
- **2026-06-08: Parallel scrape canceled** — FBref throttle risk too high (b0w7cqcmc already showing intermittent CAPTCHA / IP-block, save rate slowed to ~15 min/combo, 1 combo hard-failed `ligue_1/2324/playing_time`). A 2nd concurrent Chrome from the same IP would likely hard-block both. Lesson: one polite soccerdata scraper at a time.
- **2026-06-08: ROOT CAUSE of the "throttle" = Chrome process LEAK (not an FBref IP block).** When killing the scrape we found **646 chrome.exe + 13 uc_driver** processes leaked. `_fetch_one` creates a NEW `sd.FBref(...)` (→ new seleniumbase browser) PER COMBO and never closes it → ~1 browser leaked per combo → after ~200 combos the machine ran out of RAM/handles → new Chrome launches `Read timed out (localhost)` → soccerdata reports its GENERIC "failed CAPTCHA / IP block" message. The real errors were LOCAL timeouts, not FBref blocking. **FIX before resume:** make the scraper reuse ONE `sd.FBref` reader (or explicitly close the driver per combo), OR resume per-league in separate process runs (each exits and frees its browsers, bounding the leak to ~44). Cleanup: killed the 591 seleniumbase-leak Chrome by CommandLine signature (temp profile / `--window-position=-2400`), preserved the user's ~18 real-browser processes (default `User Data` profile). **RESOLVED:** added `reader._driver.quit()` in `_fetch_one`'s `finally` (`_close_reader`); verified chrome 15→15 after a combo. Resume then completed **396/396 cleanly** — chrome stayed ~15-30 (occasional transient spikes to ~80 that drop back, NOT accumulation), no throttle, combos ~20-45 s. Confirms a local leak, never an FBref IP block.

## Phase Output Summaries

### Phase 3 — EDA (2026-06-08)
`notebooks/01_data_exploration.ipynb` (10 sections, executed, outputs embedded, 3.6 MB) +
`reports/figures/*.png` (16) + `reports/eda_findings.md`. Logic in **`src/eda/`** (`style.py`,
`summary.py` pure-pandas, `plots.py` 16 figure fns); notebook is thin (path-bootstrap +
`src.eda` calls + rich closing notes). Generator: `scripts/build_eda_notebook.py` (nbformat).
**Biggest outcome = caught the soccerdata extended-stat gap** (see RESOLVED note above): panel
207→101 cols, §6.2 revised. Other confirmations: log-target justified (skew 4.2→−0.03, Shapiro
0.99); YoY inflation +10→+20%; league premium 3.8×–18.8×; MV median MID>DEF≈FWD>GK (FWD mean
highest via tail); GK peak-age ~30 vs outfield ~25; xG r=0.93 (Understat resid −0.14 vs Kaggle
−0.02); 11.2% position-changers; 82% TM-matched. Added `scipy`+`statsmodels` to requirements.
Tests still 21 green (src/eda has no behavior tests; pure viz). Run notebook:
`python -m nbconvert --execute --to notebook --inplace notebooks/01_data_exploration.ipynb`.

### Phase 2 Sessions 2-3 — Unified panel build (2026-06-08)
Delivered `data/processed/unified_panel.parquet` (**19,356 rows × 207 cols**, 1 per (player_id, season), 9 leagues × 4 seasons, 9,072 players). Pipeline in `unified_panel_builder.py`: **(1)** `load_fbref_stats` merges the 11 FBref stat tables per league-season — strips `<Section>_<Short>` prefixes, applies a curated per-table clean-name map (`FBREF_TABLE_STAT_RENAME` in constants, 186 clean cols) and namespaces the long tail as `<table>__<col>` (21 cols); NaN-safe composite merge key incl. `born` (disambiguates same-name-same-club, e.g. two "João Mendes"); drops expected dups (`FBREF_DROP_DUP_COLS`); **hard-errors on any unexpected dup column** (no silent overwrite). **(2)** resolve → player_id. **(3)** namesake guards: `split_id_collisions` (mononym magnet, e.g. TM "Gabriel"=Magalhães scored 100 vs every "Gabriel X" under token_set_ratio → 31 ids/65 rows re-iD'd) + `split_minutes_overflow` (>3,800 min/season = 2 same-name players, 2 rows split). **(4)** `collapse_split_season` (2,401 stints→1,200 players): sum counting, minutes-weighted pct, per-90 **recomputed from summed totals**, max-minutes club (D-15). **(5)** attach xG (Kaggle 2024-25 top5 `xg/xag/npxg` matched via name+year incl. initials for short FIFA-style names; Understat hist top5 matched by season+club then club-independent season+name fallback → **orphans 20.6%→2.4%**), MV (`tm_player_seasons`), contract (`contract_remaining_months` clip0 + `has_contract_date`), FIFA (overall/potential, FC25 potential null), league meta via `competition_id`. **(6)** `finalize_panel`: `nationality`(full)+`continent_group`, `age_at_season_end`+`age_precision` (exact 15.9k / year_only 3.5k), `log_market_value`, §5.3 schema order. Coverage: MV 69%, xG 55% (top5), FIFA 80%. Reports `name_resolution_audit.md` + `coverage_matrix.md`. **+8 panel-builder tests** (collapse sum/wavg/per90/max-club, both namesake guards) → 21 total green.

### Phase 2 Session 1 — Name-resolution spine (2026-06-08)
Built and proved the identity backbone that Sessions 2-3 attach all sources to. **`src/data/name_resolver.py`**: `PlayerIDResolver` resolves a source row → Transfermarkt `player_id` via a 5-step cascade (manual override → exact `(norm_name, birth_year, nat)` → exact `(norm_name, birth_year)` nat-agnostic safety net → fuzzy `token_set_ratio≥85` within (year,nat) pool w/ initials pre-check for "L. Messi" → name+team disambiguation for Understat → unmatched, candidates logged, never a silent wrong match). Pure helpers `normalize_name` (NFKD + ø/ß premap + initials), `normalize_nationality` (IOC code OR full-name variant → canonical via code-map/alias/normalized-index), `normalize_club` (alias+stopword fold, ratio≥90), `synthetic_player_id` (deterministic sha256). Audit buffer → `data/manual/match_log.csv`. **Crosswalk:** committed `data/external/nationality_map.csv` (140 IOC codes ⊆ continent_map keys, build-time asserted); +17 countries added to `continent_map.csv`; TM spelling variants (Bosnia-Herzegovina, Korea South, The Gambia, Türkiye…) handled via `NATIONALITY_NAME_ALIAS` + accent-insensitive normalized index. **`tests/test_name_resolution.py`: 13 tests green** (exact, accent fuzzy, short-name+team, namesake birth-year disambiguation, override precedence+source-scoping, unmatched, crosswalk + integrity-assert, synthetic determinism, club/name folding). **`scripts/build_panel.py`** smoke build resolved the full 20,557-row backbone in ~1s. Key facts confirmed: FBref `standard` cols are `<Section>_<Short>` prefixed + lowercase id cols (`player,nation,born,team,pos`), birth-YEAR only (no DOB); `tm_player_seasons` unique on (id,season); FIFA 24-25 potential 100% null; TM contract date 36% null. Output: `data/interim/fbref_backbone_resolved.parquet`.

### Phase 0 — Foundation (2026-06-07)
Scaffolded the full repository per PROJECT_ROADMAP.md §15: 42 directories, 52 files. Created the `src/` package skeleton (14 `__init__.py`) plus three infrastructure utilities — `src/utils/constants.py` (league/position maps, PER_90_STATS (25), LAG_STATS, min-minutes filters, season splits + a pure `normalize_position()` helper), `src/utils/io.py` (project_root, lookup-CSV/parquet I/O), and `src/utils/logging.py` (console+file logger). Populated 4 committed lookup CSVs in `data/external/`: `continent_map.csv` (146 rows, by FIFA confederation), `uefa_coefficients.csv`, `league_tier_map.csv`, `league_value_multipliers.csv` (9 rows each; multipliers heuristic, to be re-derived in Phase 4). Wrote pinned `requirements.txt`, README.md, and a `.gitignore` using a `.gitkeep`-negation pattern so empty skeleton dirs are tracked while data/model artifacts are ignored. Relocated the raw Kaggle dataset to `data/raw/kaggle/` (gitignored). No domain/ML logic written — pure scaffolding.

### Phase 1A — Kaggle loader (2026-06-07)
`src/data/kaggle_loader.py` + `scripts/load_kaggle.py` clean the Kaggle "Hubert" top-5 2024-25 CSV: **267 → 158 columns**, **2854 rows** preserved. De-dup dropped **117** duplicate FBref stat-table columns, renamed 0, and **preserved 1** distinct suffixed column (`Lost_stats_misc` → `aerials_lost`, so `aerial_lost` survives alongside defense `tackles_lost`). Added `FBREF_COLUMN_RENAME` + `STATS_PRESERVE_SUFFIXED` to `constants.py`. Two spec fixes verified necessary: (1) nationality parsed via whitespace split — `str[:3]` slicing broke for 2-char FBref codes ("dz ALG"); (2) `_auto_normalize` now encodes `%`/`/90`/`+/-` so percentage/per-90 variants don't collide (caught by a post-rename duplicate-column guard: `Succ`/`Succ%`, `xG`/`xG+/-`, etc.). League dist: Serie A 634, La Liga 601, PL 574, Ligue 1 553, Bundesliga 492. Position dist: DEF 1022, MID 900, FWD 720, GK 212. Split-season (mid-transfer) players: **147** (294 records) — flagged via `is_split_season`, NOT merged (Phase 2's job). Output: `data/interim/kaggle_2024_25_clean.parquet` (gitignored).

### Phase 1C — Transfermarkt (davidcariboo) ingest (2026-06-07)
`src/data/transfermarkt_loader.py` + `scripts/load_transfermarkt.py` ingest the 12 davidcariboo CSVs → 5 interim parquets: `tm_competitions` (9 leagues, all present incl. TR1 Süper Lig), `tm_players` (**21,454** in our 9 leagues, position mapped via new `TM_POSITION_MAP`, WC proxy = `has_international_caps` from players.csv), `tm_player_seasons` (**28,587** season-end-aligned MV snapshots via `merge_asof` ±45d), `tm_transfers` (**121** fee>0 inbound top-5), `tm_national_teams` (team-level, fifa_ranking). MV-snapshot orphans (no current metadata) = 1.3%. Top-5 MV coverage per season: 21-22=5852, 22-23=6273, 23-24=2800, 24-25=1880.

### Phase 1D — Custom 2024-25 transfer-fee scraper (2026-06-07)
`src/data/transfer_fee_scraper.py` + `scripts/scrape_transfer_fees.py` scrape Transfermarkt's public per-league transfer pages (requests+BS4, browser UA, **no Cloudflare**, HTML cache, retry/backoff). 2024-25 top-5 inbound: **1757 arrivals, 548 fee>0** (vs davidcariboo's 121), median €6.75M, max **€80M**, **9 deals ≥€60M** (Kvaratskhelia→PSG €80M, J.Álvarez→Atléti €75M, Marmoush→City €75M, …). Name-overlap with davidcariboo = 63 → ~485 fee>0 transfers (incl. all marquee deals) are NEW. Output `data/raw/transfer_fees_2025/transfer_fees_2024_25.csv` (gitignored) + `reports/transfer_fees_cross_validation.md` (committed). Pre-research note: TM transfer page lacks per-transfer dates → `transfer_date`/`transfer_window`/`from_league` = None (Phase 2 resolves). This is the primary §11.4 "absolute-truth" reference (D-09).

### Phase 1E — FIFA / EA FC ratings loader (2026-06-08)
`src/data/fifa_loader.py` + `scripts/load_fifa.py` → `data/interim/fifa_ratings.parquet` (**72,283 rows**, unified 14-col schema). **Source consolidation (key finding):** `EA FC 24/male_players.csv` holds one clean snapshot per fifa_version 15-24, so FIFA **22 (19,239) / 23 (18,533) / 24 (18,350)** all come from that one file — the 5.3 GB weekly-update `male_players.csv` and the legacy file are redundant and SKIPPED (IO-light). FC **25 (16,161)** from nyagami `EA FC 25/male_players.csv`. Added `FIFA_STEFANO_COLUMN_RENAME`, `FIFA_NYAGAMI_COLUMN_RENAME`, `FIFA_POSITION_MAP` (FIFA codes ST/CB/CM/GK… → primary; `normalize_position` does NOT apply), `FIFA_YEAR_TO_SEASON` to constants. **nyagami FC25 has NO `potential`/`dob`/sofifa_id** → null for 2024-25 (note for Phase 4: young-player potential feature missing for 24-25). stefano `player_id`=sofifa_id. 0 'Other' positions; Mbappé spot-check OVR 91, POT 95→94 (22-24), null (25). Phase-2 joins by name+age+nationality.

### Phase 1B — FBref hybrid scrape (complete 2026-06-08)
**396/396 combos** = 9 leagues × 4 seasons × 11 stat types (native standard/shooting/playing_time/keeper/misc + extended passing/passing_types/gca/defense/possession/keeper_adv via the monkeypatched `read_player_season_stats`). Output `data/raw/fbref/<slug>/<season>/<stat_type>.parquet` (gitignored). Plus **Understat 20/20** (top-5 × 4 seasons, xG source). Coverage validator: **416/416 present, 0 missing**; player counts ~490-660/league-season (Süper Lig ~660 slightly over the conservative range, benign). Cross-val vs Kaggle 2024-25 (PL+LaLiga): goals/assists/minutes **Pearson 1.000, MAE 0.00**, shots ~1.000 → FBref-soccerdata ≡ Kaggle for shared stats. The big saga: a Chrome-process leak (see Bug Log) masqueraded as throttling; fixed with per-combo `driver.quit()`, after which the scrape finished cleanly with no FBref block. Reports committed: `reports/fbref_coverage_summary.csv`, `reports/fbref_kaggle_cross_validation.md`.
