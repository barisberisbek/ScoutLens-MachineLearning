# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

- **Two-stage ML pipeline:** Performance Projection (Stage 1) → Valuation (Stage 2).
- **Goal:** discover undervalued football players, focused on lower-tier European leagues.
- **Team (3):** Abdullah Özdemir, Barış Berişbek, Serhat Tay. Course: CENG 3522 Applied ML.
- **Authority:** this repo works to `PROJECT_ROADMAP.md` (v2.0) — the single source of
  truth for schemas, formulas, and locked decisions. Read the relevant § before implementing.

## What This Project Is NOT

- It is **NOT** a naive model that predicts *current* market value from *current* stats.
- That approach produces **circular reasoning** (it just mimics Transfermarkt) and
  overfitting (diagonal predicted-vs-actual line, zero discovery signal). It was
  **explicitly rejected** by the instructor and by §2 of the roadmap.
- The correct approach: **project next-season stats → derive a forward-looking value**
  from those projections, then compare to current MV to find the gap.

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run tests
```bash
pytest tests/ -v
pytest tests/test_feature_forwarder.py -v   # single test file
```

### Data pipeline (run in order)
```bash
python scripts/scrape_fbref.py
python scripts/scrape_transfer_fees.py
python scripts/build_panel.py
python scripts/train_stage1.py
python scripts/train_stage2.py
python scripts/precompute_app_data.py
```

### Demo app
```bash
streamlit run app/streamlit_app.py
```

## Architecture

The core problem: predicting current value from current stats is circular (the model
just mimics Transfermarkt). Solution: predict *future* value from *projected future* stats.

```
Player (season s) ──→ Stage 1 (Projection) ──→ projected stats_{s+1}
                              │
                              ▼
                     Feature Forwarder (deterministic)
                       - age += 1
                       - contract_remaining_months -= 12 (clip 0)
                       - swap in projected stats
                       - recompute z-scores, per-90s, composites
                              │
                              ▼
                     Stage 2 (Valuation) ──→ log1p(market_value_{s+1})
                              │
                              ▼
                     value_gap = predicted − current_mv
```

**Stage 1** — per-position ensemble (GK/DEF/MID/FWD) trained on `(season_s features) →
(season_s+1 stat vector)` pairs. Per-position target lists differ (§6.2). Stacked
Ridge + XGBoost + per-stat XGBoost + Keras MLP.

**Feature Forwarder** (`src/features/feature_forwarder.py`) — deterministic only, no
learning. Has mandatory unit tests; a bug here silently corrupts every prediction.

**Stage 2** — per-position ensemble trained on `(actual observed stats + context) →
log1p(market_value)`. **Never trained on Stage 1 projections.**

**Discovery** — train Stage 2 on top-5 leagues (efficient pricing), apply to Eredivisie /
Liga Portugal / Belgian Pro / Süper Lig (inefficient pricing); rank by value gap ×
confidence.

### Module Map

| Directory | Responsibility |
|---|---|
| `src/data/` | Loaders & scrapers per source + `name_resolver.py` |
| `src/integration/` | `unified_panel_builder.py` |
| `src/features/` | One transform per file + `build_features.py` + `feature_forwarder.py` |
| `src/models/` | `stage1_projection/`, `stage2_valuation/`, `ensemble.py` |
| `src/pipeline/` | `inference.py`, `confidence.py` |
| `src/discovery/` | `runner.py`, `cross_league_calibration.py` |
| `src/clustering/` | `trainer.py`, `similarity.py` |
| `src/patterns/` | `apriori_analyzer.py` |
| `src/interpretability/` | `shap_explainer.py` |
| `src/validation/` | `metrics.py`, `temporal_split.py`, `stage_validators.py`, `transfer_fee_validator.py` |
| `src/utils/` | `constants.py` (all maps), `io.py`, `logging.py` |
| `scripts/` | CLI entry points — standalone, loggable, resumable |
| `app/` | Streamlit demo (local only) |
| `notebooks/` | Reports 01–10 — narrative + viz only, no logic |
| `tests/` | `test_name_resolution.py`, `test_feature_forwarder.py`, `test_pipeline.py` |

### Data Sources (§5.1)

Kaggle (top-5 2024-25 + FIFA) · FBref via `soccerdata` (all leagues, 2021-22→2024-25,
all stat types) · `transfermarkt-datasets` (dcaribou — MV/contracts/transfers) · custom
2025 transfer-fee scraper · static UEFA coefficients / continent / league-tier CSVs.

Name resolution: composite key `(normalized_name, birth_date, nationality)` → `rapidfuzz
≥ 85` fallback → manual override CSV for top-500 → audit log.

### Temporal Split (D-01 — locked)

Train: 2021-22, 2022-23, 2023-24. Validate: 2024-25. Bonus: actual summer-2025 transfer
fees ("absolute truth"). **Never use random train/test splits anywhere.**

## Locked Architectural Decisions (§3)

Settled — any deviation needs team consensus + a `reports/decisions_log.md` entry.

| ID | Decision | Rationale |
|---|---|---|
| D-01 | Train 21-22/22-23/23-24; validate 24-25 → summer-25 MV | Eliminates random-split leakage |
| D-02 | Two-stage: projection → forwarder → valuation | Resolves circular reasoning |
| D-03 | Position-stratified GK/DEF/MID/FWD at both stages | Different value drivers per position |
| D-04 | Single global model + league features (NOT per-league) | Maintains statistical power |
| D-05 | Target `log1p(mv)`; inverse `expm1` at inference | Right-skewed target |
| D-06 | Nationality → `continent_group` (5-way) + `worldcup_25_squad` | Avoid sparse high-cardinality |
| D-07 | Empirical year-inflation multiplier feature | Counter ~8–12% YoY market inflation |
| D-08 | Lower-league discovery applied post-training | Project differentiator |
| D-09 | Bonus validation vs actual 24-25/25 transfer fees | "Absolute truth" reference |
| D-10 | Hybrid data: Kaggle + FBref + TM repo + scraping | No single source is complete |
| D-11 | Local Streamlit demo only; cloud deferred | Scope decision |
| D-12 | SHAP on all Stage 2 outputs | Scout-facing UX |
| D-13 | Apriori on top-undervalued cohort | Week-8 curriculum coverage |
| D-14 | K-Means + Hierarchical clustering per position | Curriculum + similar-player engine |
| D-15 | Loans: include w/ `is_loan`; stats from playing club, contract from parent | Standard practice |
| D-16 | MV snapshot = closest TM snapshot to season end (±45 days) | Operational convention |
| D-17 | FIFA/SoFIFA ratings as features | Low cost, high signal for young players |
| D-18 | Capology wages deferred to Phase 6 revisit | Skip until evidence needed |
| D-19 | Compute local-first; Colab only for heavy sweeps | Data fits laptop |

## Critical Anti-Patterns (REFUSE if asked to do these)

- **Random `train_test_split`** → temporal split ONLY; `TimeSeriesSplit` for CV.
- **`market_value` as a feature** → it is ONLY the target.
- **Training Stage 2 on Stage 1 projections** → Stage 2 sees actual observed stats.
- **Stage 2 inference without the Feature Forwarder** → age+1, contract−12mo are MANDATORY.
- **Target encoding without out-of-fold** → leakage; OOF only, fit on train.
- **Composite position strings ("CM,DM")** → normalize to primary GK/DEF/MID/FWD first.
- **Magic numbers outside `src/`** → everything in `src/utils/constants.py`.
- **Silent row drops** → every filter logs `(input_count, output_count, reason)`.
- **Re-scraping cached data** → cache is authoritative; never re-fetch unchanged pages.
- **Heavy logic in notebooks** → logic lives in `src/`; notebooks orchestrate & visualize.

## Code Conventions

- Python 3.10+, type hints + docstrings on every public function.
- `random_state=42` everywhere; save splits to disk.
- One transform / one model / one metric → one file.
- Parquet for every intermediate state; save at every phase boundary.
- Defensive joins: log `(left, right, result, matched, unmatched)` on every merge.
- Minimum tests: `name_resolver`, `feature_forwarder`, `pipeline`.

## File Layout (§15, condensed)

```
data/{raw,interim,processed,external,manual}/   raw+interim+models gitignored
src/{data,integration,features,models,pipeline,discovery,clustering,patterns,
     interpretability,validation,utils}/
scripts/        scrape_fbref, scrape_transfer_fees, build_panel,
                train_stage1, train_stage2, precompute_app_data
models/{stage1,stage2}/{gk,def,mid,fwd}.pkl     (gitignored)
app/{streamlit_app.py, pages/, components/, cached_data/}
notebooks/01..10   reports/{final_report,feature_dictionary,decisions_log,figures,tables}
tests/{test_name_resolution,test_feature_forwarder,test_pipeline}.py
```
External lookup CSVs (`data/external/`) and manual overrides (`data/manual/`) ARE committed.

## Phase-Specific Rules

### Phase 1 — Data Acquisition
- All scraping runs LOCAL, never Colab. Cache aggressively (`data/raw/<source>/.cache/`).
- 2–4 s random delay between requests. Scrapers as `scripts/scrape_*.py`, never notebooks.

### Phase 2 — Integration
- Name resolution: composite key first, then `rapidfuzz ≥ 85`. Manual override CSV for top-500.
- Report match rate after every join. Min-minutes filter: 450 (top-5) / 300 (lower).

### Phase 3 — EDA
- Notebooks = narrative + visualization ONLY, no logic. Save plots to `reports/figures/`
  programmatically.

### Phase 4 — Feature Engineering
- One transform per file under `src/features/`; `build_features.py` orchestrates all.
- `feature_forwarder.py` unit tests are MANDATORY. Update `reports/feature_dictionary.md`
  whenever a feature is added.

### Phase 5 — Stage 1 (Projection)
- 4 models GK/DEF/MID/FWD; per-position output target list (§6.2).
- `TimeSeriesSplit`, never random KFold. Training pairs `(s, s+1)`; 2024-25 reserved for test.

### Phase 6 — Stage 2 (Valuation)
- 4 models GK/DEF/MID/FWD. Target `log1p(mv)`, `expm1` at inference.
- Training data `(actual_stats[s], mv[s])` — NOT Stage 1 projections.
- Resolve linear-baseline multicollinearity via VIF analysis.

### Phase 7 — Pipeline + Validation
- Report THREE validations: Stage 1 standalone, Stage 2 standalone, full pipeline.
- Feature Forwarder MANDATORY at inference. Bonus validation vs actual 24-25/25 transfer fees.

### Phase 8 — Discovery
- Train Stage 2 on top-5 ONLY, apply to lower leagues.
- Cross-league calibration validation (transferred players) is required.

### Phase 9 — Auxiliary
- Clustering: do NOT use `market_value` as a feature (style, not value).
- Apriori: on the top-10% undervalued cohort.
- SHAP: pre-compute per player, save to `app/cached_data/`.

### Phase 10 — Demo
- Local Streamlit only, no cloud. Pre-computed predictions; no live inference in-app.

## Working With the Architect

### When to update memory.md
- Every time a phase completes.
- When an important decision changes.
- On any unexpected finding (e.g. a data-quality issue).
- When an open question arises.

### When to STOP and ask (don't implement — note in memory.md, flag the user)
- A deviation from a `PROJECT_ROADMAP.md` decision is needed.
- There is a real tradeoff between two approaches.
- Validation results are unexpectedly poor.
- A blocker requires an architectural change.

## Session Protocol (§19.1)

Start each session by stating the current phase and the specific deliverable. Reference
roadmap sections explicitly (e.g. "Read §5.4 for the name-resolution spec"). The roadmap
is the single source of truth.
