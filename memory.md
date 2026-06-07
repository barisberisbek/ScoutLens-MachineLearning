# Project Memory — AML Final Project

## Current State
- **Current phase:** Phase 1A complete — ready for Phase 1B (FBref scraper)
- **Last updated:** 2026-06-07
- **Last session summary:** Phase 1A done — `src/data/kaggle_loader.py` cleans the Kaggle top-5 2024-25 CSV (267→158 cols, 2854 rows) to `data/interim/kaggle_2024_25_clean.parquet`. Env note: `.venv` is Python 3.13.1 with pandas 3.0.3 / pyarrow 24; run scripts via `.venv\Scripts\python.exe` with `PYTHONPATH=<root>`.

## Phase Completion Log
- [x] Phase 0 — Foundation (repo scaffolding)
- [x] Phase 1A — Kaggle loader
- [ ] Phase 1B — FBref scraper
- [ ] Phase 1C — Transfermarkt-datasets ingest
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
Henüz yok.

## Performance / Results Log
Phase 5 ve 6 başlayınca model performans metrikleri burada raporlanacak.

## Bug Log / Learnings
Henüz yok.

## Phase Output Summaries

### Phase 0 — Foundation (2026-06-07)
Scaffolded the full repository per PROJECT_ROADMAP.md §15: 42 directories, 52 files. Created the `src/` package skeleton (14 `__init__.py`) plus three infrastructure utilities — `src/utils/constants.py` (league/position maps, PER_90_STATS (25), LAG_STATS, min-minutes filters, season splits + a pure `normalize_position()` helper), `src/utils/io.py` (project_root, lookup-CSV/parquet I/O), and `src/utils/logging.py` (console+file logger). Populated 4 committed lookup CSVs in `data/external/`: `continent_map.csv` (146 rows, by FIFA confederation), `uefa_coefficients.csv`, `league_tier_map.csv`, `league_value_multipliers.csv` (9 rows each; multipliers heuristic, to be re-derived in Phase 4). Wrote pinned `requirements.txt`, README.md, and a `.gitignore` using a `.gitkeep`-negation pattern so empty skeleton dirs are tracked while data/model artifacts are ignored. Relocated the raw Kaggle dataset to `data/raw/kaggle/` (gitignored). No domain/ML logic written — pure scaffolding.

### Phase 1A — Kaggle loader (2026-06-07)
`src/data/kaggle_loader.py` + `scripts/load_kaggle.py` clean the Kaggle "Hubert" top-5 2024-25 CSV: **267 → 158 columns**, **2854 rows** preserved. De-dup dropped **117** duplicate FBref stat-table columns, renamed 0, and **preserved 1** distinct suffixed column (`Lost_stats_misc` → `aerials_lost`, so `aerial_lost` survives alongside defense `tackles_lost`). Added `FBREF_COLUMN_RENAME` + `STATS_PRESERVE_SUFFIXED` to `constants.py`. Two spec fixes verified necessary: (1) nationality parsed via whitespace split — `str[:3]` slicing broke for 2-char FBref codes ("dz ALG"); (2) `_auto_normalize` now encodes `%`/`/90`/`+/-` so percentage/per-90 variants don't collide (caught by a post-rename duplicate-column guard: `Succ`/`Succ%`, `xG`/`xG+/-`, etc.). League dist: Serie A 634, La Liga 601, PL 574, Ligue 1 553, Bundesliga 492. Position dist: DEF 1022, MID 900, FWD 720, GK 212. Split-season (mid-transfer) players: **147** (294 records) — flagged via `is_split_season`, NOT merged (Phase 2's job). Output: `data/interim/kaggle_2024_25_clean.parquet` (gitignored).
