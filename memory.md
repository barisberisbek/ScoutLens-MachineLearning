# Project Memory — AML Final Project

## Current State
- **Current phase:** Phase 1C complete; **Phase 1B FBref scrape still RUNNING in background** (`boo3tlz7e`, ~3-5h) — its coverage/cross-val/commit are pending its completion.
- **Last updated:** 2026-06-07
- **Last session summary:** Phase 1C done — `src/data/transfermarkt_loader.py` ingests davidcariboo CSVs → 5 `data/interim/tm_*.parquet`. Understat (Phase 1B xG source) scraped 20/20. Env note: `.venv` Python 3.13.1, pandas 3.0.3, pyarrow 24; run scripts via `.venv\Scripts\python.exe` with `PYTHONPATH=<root>`.

## Phase Completion Log
- [x] Phase 0 — Foundation (repo scaffolding)
- [x] Phase 1A — Kaggle loader
- [ ] Phase 1B — FBref scraper
- [x] Phase 1C — Transfermarkt-datasets ingest
- [ ] Phase 1D — 2025 transfer fees scraper
- [ ] Phase 1E — FIFA ratings loader
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
  this dataset. The transfer-fee validation will be limited to these mostly mid-size deals;
  Phase 1D's dedicated 2025-fee scraper is the intended remedy.

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
