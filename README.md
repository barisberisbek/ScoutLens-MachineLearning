# Football Player Valuation & Hidden-Gem Discovery

A two-stage ML system that produces **forward-looking** football player valuations
(project next-season performance → derive future market value) and uses them to discover
**undervalued players in under-scouted lower-tier European leagues**.

## Team

- Abdullah Özdemir
- Barış Berişbek
- Serhat Tay

Course: CENG 3522 — Applied Machine Learning

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Run the pipeline scripts in order (implemented across later phases):

```bash
python scripts/scrape_fbref.py
python scripts/scrape_transfer_fees.py
python scripts/build_panel.py
python scripts/train_stage1.py
python scripts/train_stage2.py
python scripts/precompute_app_data.py
```

Launch the local demo:

```bash
streamlit run app/streamlit_app.py
```

## Project structure

```
data/
  raw/        kaggle, fbref, transfermarkt, transfer_fees_2025, fifa   (gitignored)
  interim/    intermediate artifacts                                   (gitignored)
  processed/  unified_panel.parquet, features.parquet                  (parquet gitignored)
  external/   committed lookup CSVs (UEFA, continent, league maps)
  manual/     committed manual overrides + audit logs
src/
  data/ integration/ features/ models/ pipeline/ discovery/
  clustering/ patterns/ interpretability/ validation/ utils/
scripts/      standalone CLI entry points (scraping, training, precompute)
notebooks/    01–10 analysis reports (narrative + viz only)
models/       trained model pickles                                    (gitignored)
app/          Streamlit demo (pages/, components/, cached_data/)
reports/      final report, feature dictionary, decisions log, figures, tables
tests/        name resolver, feature forwarder, pipeline
```

## Documentation

- **`PROJECT_ROADMAP.md`** — full technical specification: architecture, locked
  decisions, data schema, feature catalog, validation framework, phase plan.
- **`CLAUDE.md`** — working memo for Claude Code sessions (conventions, anti-patterns,
  phase-specific rules).
- **`memory.md`** — running project state and phase completion log.
