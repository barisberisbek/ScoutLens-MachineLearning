# Project Memory — AML Final Project

## Current State
- **Current phase:** Phase 0 complete — ready for Phase 1 (Data Acquisition)
- **Last updated:** 2026-06-07
- **Last session summary:** Phase 0 scaffolding done — full §15 directory tree, src package skeleton, .gitignore (with .gitkeep negation), pinned requirements.txt, constants.py, io/logging utils, 4 external lookup CSVs, README; first commit landed. Kaggle CSV relocated to data/raw/kaggle/.

## Phase Completion Log
- [x] Phase 0 — Foundation (repo scaffolding)
- [ ] Phase 1A — Kaggle loader
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
