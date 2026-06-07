# Project Memory — AML Final Project

## Current State
- **Current phase:** Phase 1E complete → **all Phase-1 sub-phases done EXCEPT 1B finalization.** **Phase 1B FBref scrape RUNNING in background** (`b0w7cqcmc`; was 128/396 last checked) — coverage/cross-val/[x]/commit pending its completion. After 1B → Phase 2 (Integration).
- **Last updated:** 2026-06-08
- **Last session summary:** Phase 1E done — `src/data/fifa_loader.py` → `data/interim/fifa_ratings.parquet` (72,283 rows, 4 seasons). Env: `.venv` Python 3.13.1, pandas 3.0.3; run via `.venv\Scripts\python.exe` + `PYTHONPATH=<root>`.

## Phase Completion Log
- [x] Phase 0 — Foundation (repo scaffolding)
- [x] Phase 1A — Kaggle loader
- [ ] Phase 1B — FBref scraper
- [x] Phase 1C — Transfermarkt-datasets ingest
- [x] Phase 1D — 2025 transfer fees scraper
- [x] Phase 1E — FIFA ratings loader
- [x] Phase 1F — External lookup CSVs (UEFA, continent, etc.)  ← done early in Phase 0
- [ ] Phase 2 — Data Integration
- [ ] Phase 3 — EDA
- [ ] Phase 4 — Feature Engineering
- [ ] Phase 5 — Stage 1 Modeling (×4 positions)
- [ ] Phase 6 — Stage 2 Modeling (×4 positions)
- [ ] Phase 7 — Pipeline Assembly + Validation
- [ ] Phase 8 — Discovery Layer
- [ ] Phase 9 — Auxiliary Modules (clustering, patterns, SHAP)
- [ ] Phase 10 — Streamlit Demo
- [ ] Phase 11 — Report & Presentation

## Decisions Log (beyond PROJECT_ROADMAP.md)
Henüz yok. Implementation sırasında implicit kararlar burada loglanacak.

## Open Questions / Issues

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
Henüz yok.

## Phase Output Summaries

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
