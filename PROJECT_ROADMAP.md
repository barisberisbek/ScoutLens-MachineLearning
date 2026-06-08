# AML Final Project — Technical Specification & Implementation Roadmap

**Project:** Performance Projection & Player Valuation Discovery System
**Course:** CENG 3522 — Applied Machine Learning
**Track:** Alternative 2 (Data to Problem) + Alternative 3 (Literature-informed)
**Team:** 3 members (Abdullah Özdemir, Barış Berişbek, Serhat Tay)
**Document type:** Operational specification — single source of truth for implementation

---

## Contents

1. [Project Summary](#1-project-summary)
2. [The Architectural Pivot](#2-the-architectural-pivot)
3. [Locked Decisions](#3-locked-decisions)
4. [System Architecture](#4-system-architecture)
5. [Data Strategy](#5-data-strategy)
6. [Stage 1: Performance Projection](#6-stage-1-performance-projection)
7. [Stage 2: Valuation Model](#7-stage-2-valuation-model)
8. [Feature Forwarder (Inference Bridge)](#8-feature-forwarder-inference-bridge)
9. [Discovery Layer](#9-discovery-layer)
10. [Feature Engineering Catalog](#10-feature-engineering-catalog)
11. [Validation Framework](#11-validation-framework)
12. [Auxiliary Modules](#12-auxiliary-modules)
13. [Demo Application (Local)](#13-demo-application-local)
14. [Technical Stack & Compute Strategy](#14-technical-stack--compute-strategy)
15. [Repository Structure](#15-repository-structure)
16. [Implementation Phases](#16-implementation-phases)
17. [Risk Register](#17-risk-register)
18. [Curriculum Alignment](#18-curriculum-alignment)
19. [Claude Code Workflow](#19-claude-code-workflow)

---

## 1. Project Summary

We build a two-stage ML system that produces **forward-looking** football player valuations and uses them to discover **undervalued players in under-scouted leagues**.

The system consists of:

- **Stage 1 — Performance Projection.** Position-stratified ensemble predicting a player's next-season statistical profile from their historical performance, age, and context.
- **Feature Forwarder.** Deterministic component that transforms current player features (age, contract, league) into their next-season equivalents.
- **Stage 2 — Valuation Model.** Position-stratified ensemble mapping a statistical+contextual profile to a fair market value.
- **Discovery Layer.** Application of the pipeline to selected lower-tier leagues (Eredivisie, Liga Portugal, Belgian Pro League, Süper Lig) to surface hidden gems.
- **Auxiliary modules.** Clustering, frequent pattern mining, SHAP interpretability.
- **Local Streamlit demo.** Interactive exploration of the system. No cloud deployment in this phase.

**Primary user persona:** A sporting director seeking a CB under €20M whose projected next-season performance suggests current undervaluation, ideally in markets less competitive than top-5.

---

## 2. The Architectural Pivot

### 2.1 Why the naive approach must be avoided

A direct "predict current market value from current stats" model fails for three reasons:

1. **Circular reasoning.** Transfermarkt's market value is already a function of the same features being used as inputs. The model converges to mimicking the existing valuation, producing zero discovery signal.
2. **Diagonal-line predictions.** Predicted-vs-actual scatter collapses to the identity line. High R² is illusory — the model is reproducing its inputs.
3. **No forward value to a scout.** Current market value is public information. What is needed is the value a player *will* have given their projected trajectory.

This is the exact problem flagged by the instructor in the midterm review.

### 2.2 The two-stage solution

We decompose the problem into two distinct learning tasks:

**Stage 1 (the hard problem):** Learn how players develop, peak, and decline across seasons.
```
Past 1–3 seasons of stats + age + context → next-season per-90 stat vector
```

**Stage 2 (the calibration problem):** Learn the market's pricing function.
```
A stat profile + age + league + contract + nationality → market value
```

**Inference flow:**
```
1. Get player P with current features X_s
2. Stage 1: predict next-season stat vector → S_{s+1}
3. Feature Forwarder: transform X_s into X_{s+1}
   (age+1, contract−12mo, projected_stats replace current_stats)
4. Stage 2: predict log market value on X_{s+1}
5. Inverse-transform to euros, attach confidence score
6. Compare to current market value → value_gap
```

This architecture is **structurally immune** to the circular reasoning problem: Stage 2 sees projected future inputs at inference, not the same inputs it was trained on.

### 2.3 The discovery angle

Top-5 European leagues are heavily scouted; pricing is efficient. The opportunity is in **lower-tier leagues** where:
- Analyst coverage is thinner
- Market values lag behind performance
- Genuine pricing inefficiencies persist

Strategy: train on top-5 (high signal-to-noise), apply to lower leagues (via league multipliers and UEFA coefficients), surface undervalued candidates.

---

## 3. Locked Decisions

These are settled; any deviation requires team consensus and a log entry in `reports/decisions_log.md`.

| ID | Decision | Rationale |
|---|---|---|
| **D-01** | Train: 2021-22 / 22-23 / 23-24. Validate: 2024-25 → 2025 summer MV. | Eliminates random-split leakage |
| **D-02** | Two-stage pipeline: projection → forwarder → valuation | Resolves circular reasoning |
| **D-03** | Position-stratified: GK / DEF / MID / FWD at both stages | Different value drivers per position |
| **D-04** | Single global model + explicit league features (NOT per-league models) | Maintains statistical power |
| **D-05** | Target: `log1p(market_value_eur)`; inverse `expm1` at inference | Right-skewed target |
| **D-06** | Nationality: `continent_group` (5-way) + `worldcup_25_squad` (binary) | Avoid sparse high-cardinality features |
| **D-07** | Year inflation multiplier feature, derived empirically from training data | Counter ~8–12% YoY market inflation |
| **D-08** | Lower-league discovery applied post-training | Project differentiator |
| **D-09** | Bonus validation against 2024-25 winter + 2025 summer actual transfer fees (top-5 inbound) | "Absolute truth" reference |
| **D-10** | Hybrid data: Kaggle base + FBref multi-season + transfermarkt-datasets repo + targeted scraping | No single source is complete |
| **D-11** | Local Streamlit demo only; cloud deployment deferred | User scope decision |
| **D-12** | SHAP interpretability on all Stage 2 outputs | Required for scout-facing UX |
| **D-13** | Apriori on top-undervalued cohort (Week 8 curriculum coverage) | Curriculum completeness |
| **D-14** | K-Means + Hierarchical clustering per position (Weeks 10–12) | Curriculum + similar-player engine |
| **D-15** | Loan players: include with `is_loan` flag; stats from playing club, contract from parent | Standard football data practice |
| **D-16** | MV snapshot per season: closest Transfermarkt snapshot to season end date (±45 days) | Operational convention |
| **D-17** | FIFA / SoFIFA ratings included as features | Low cost, high signal for young players |
| **D-18** | Capology wage data deferred to Phase 6 revisit (not Phase 1) | Skip until evidence it's needed |
| **D-19** | Compute: local-first, Colab only for heavy hyperparameter sweeps | Data size fits laptop comfortably |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│  Kaggle CSV    FBref (soccerdata)    TM repo    Targeted scraping   │
│       └────────────┬────────┴───────────┴────────────┘              │
│                    │ Fuzzy name + birth-date matching               │
│                    ▼                                                │
│            data/processed/unified_panel.parquet                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                                │
│  Per-90 │ Z-scores │ Age curves │ Contract │ League × Position      │
│  Continent │ FIFA │ Composite │ Trajectory │ Year inflation         │
│                       ↓                                              │
│            data/processed/features.parquet                          │
└───────────────────────┬─────────────────────────────────────────────┘
                        ▼
┌───────────────────────┼─────────────────────────────────────────────┐
│   STAGE 1 (×4 pos)    │              STAGE 2 (×4 pos)                │
│   Projection           │              Valuation                       │
│                        │                                              │
│   X_s → S_{s+1}        │   X_{s+1} → log_value                       │
│                        │                                              │
│   Trained on:          │   Trained on:                                │
│   (stats_s, stats_{s+1})│   (full_features_s, market_value_s)         │
└───────────────┬────────┴──────────┬──────────────────────────────────┘
                │                   │
                └────────┬──────────┘
                         ▼
        ┌────────────────────────────────────┐
        │       FEATURE FORWARDER             │
        │  X_s + S_{s+1} → X_{s+1}            │
        │  (age+1, contract−12mo, ...)       │
        └────────────────┬────────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │        PIPELINE INFERENCE           │
        │   Stage1 → Forwarder → Stage2       │
        │   + confidence score + SHAP         │
        └────────────────┬────────────────────┘
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
┌──────────────┐  ┌─────────────┐  ┌──────────────┐
│ Value Gap    │  │ Clustering  │  │ Apriori on   │
│ Analysis     │  │ + Similar   │  │ undervalued  │
│              │  │ Player Eng. │  │ cohort       │
└──────┬───────┘  └──────┬──────┘  └──────┬───────┘
       └─────────────────┼─────────────────┘
                         ▼
       ┌────────────────────────────────────┐
       │       DECISION SYSTEM               │
       │  Ranked transfer recommendations    │
       └────────────────┬────────────────────┘
                        ▼
       ┌────────────────────────────────────┐
       │      LOCAL STREAMLIT DEMO           │
       └─────────────────────────────────────┘
```

---

## 5. Data Strategy

### 5.1 Sources

| Source | Type | Coverage | Access | Phase |
|---|---|---|---|---|
| Kaggle hubertsidorowicz Football Players Stats 2024-25 | CSV | Top-5, 2024-25 | Direct download | 1A |
| FBref via `soccerdata` | Python library | All target leagues, all seasons | API-wrapped scrape | 1B |
| `transfermarkt-datasets` (dcaribou GitHub) | CSV (auto-updated) | Global MV + contracts + transfers | git clone | 1C |
| Transfermarkt 2025 transfer fees | Web | Top-5 inbound transfers | Custom scraper | 1D |
| FIFA / SoFIFA ratings | Kaggle | Game ratings, potential | Direct download | 1E |
| UEFA Country Coefficients | Static | League prestige | Manual CSV | 1F |
| Capology wages | Web | Top-5 only | (Deferred to Phase 6 if needed) | — |

### 5.2 Seasons and leagues

| Season | Top-5 | Lower (4) | Purpose |
|---|---|---|---|
| 2021-22 | ✅ | ✅ | Train |
| 2022-23 | ✅ | ✅ | Train |
| 2023-24 | ✅ | ✅ | Train + Stage 1 trajectory |
| 2024-25 | ✅ | ✅ | Test + Discovery input |
| 2025 summer | MV + transfers | MV + transfers | Validation ground truth |

**Lower leagues (locked first iteration):**
- Eredivisie (Netherlands)
- Liga Portugal (Portugal)
- Belgian Pro League / Jupiler Pro League
- Süper Lig (Turkey)

All four have strong FBref coverage from 2021-22 onward. Expansion to MLS / Brazilian Série A / Argentine Liga Profesional is deferred to a possible Phase 8 stretch if time allows.

### 5.3 Data schema (unified panel)

`data/processed/unified_panel.parquet`, one row = one (player, season):

```
Identity:
  player_id (str, internal)
  player_name, birth_date, nationality, continent_group

Season context:
  season ("2023-24"), season_end_year (int), season_end_date (date)
  league, league_tier (1=top5, 2=lower-european)
  uefa_coefficient (float)
  club, is_loan (bool)
  age_at_season_end (float)

Position:
  primary_position (GK/DEF/MID/FWD)
  detailed_position (CB/LB/CM/CAM/ST/...)

Performance (100+ stat columns from FBref):
  minutes_played, goals, assists, xg, xag, shots, ...
  progressive_passes, key_passes, sca, gca, ...
  tackles, interceptions, blocks, aerials, ...
  saves, save_pct, clean_sheets, psxg, ...

Market value:
  market_value_eur (float), market_value_date (date)
  log_market_value (float)

Contract:
  contract_end_date (date)
  contract_remaining_months (float)

External enrichment:
  fifa_rating (int, nullable)
  fifa_potential (int, nullable)
  worldcup_25_squad (bool)
```

### 5.4 Name resolution

The hardest data engineering challenge. Strategy:

1. **Primary key:** `(name_lowercase_normalized, birth_date, nationality)` — birth date disambiguates.
2. **Fuzzy fallback:** `rapidfuzz.fuzz.ratio ≥ 85` within same nationality.
3. **Manual override table:** `data/manual/manual_id_overrides.csv` for top-500 by market value where automated matching fails.
4. **Audit log:** every match decision logged to `data/manual/match_log.csv`.

```python
def resolve_player_id(name, birth_date, nationality, master_db):
    # Exact composite match
    exact = master_db[
        (master_db['name_normalized'] == normalize(name)) &
        (master_db['birth_date'] == birth_date)
    ]
    if len(exact) == 1:
        return exact.iloc[0]['player_id'], 'exact'

    # Fuzzy fallback within nationality
    candidates = master_db[master_db['nationality'] == nationality]
    if len(candidates) == 0:
        return None, 'no_candidates'
    scores = candidates['name_normalized'].apply(
        lambda x: fuzz.ratio(normalize(name), x)
    )
    best_idx = scores.idxmax()
    if scores[best_idx] >= 85:
        return candidates.loc[best_idx, 'player_id'], 'fuzzy'

    return None, 'unmatched'
```

### 5.5 Scraping principles

- All scraping runs **locally**, not in Colab.
- Respect 2–4 second random delays between requests.
- Cache aggressively (`data/raw/<source>/.cache/`); never re-scrape unchanged pages.
- Save raw HTML alongside parsed CSV for re-processing.
- Run scrapers as standalone scripts (`scripts/scrape_*.py`), never inside notebooks.
- Output: append-only CSVs in `data/raw/`.

---

## 6. Stage 1: Performance Projection

### 6.1 Formulation

For each player with at least 2 consecutive seasons:
```
X = features(seasons up to s) + static context (age, league, position, ...)
y = stats vector(season s+1)
```

Multi-output regression on a per-position-tuned vector of stats.

### 6.2 Position-specific targets

> **REVISED 2026-06-08 (Phase 3 EDA finding).** The original target lists below assumed
> FBref extended stats (tackles total, blocks, aerials, progressive passes/carries, key
> passes, SCA/GCA, PSxG). EDA proved **soccerdata never delivered these columns** — they
> are empty at source in the cached HTML across all 36 league-seasons (not a parse bug,
> unrecoverable without an alternative source). Only `tackles_won`, `interceptions`, and
> the native scoring/shooting/keeper stats survived. Targets are therefore reduced to the
> available stat universe + xG/xAG (Kaggle 2024-25 top-5, Understat historical top-5).
> Final selection is locked in Phase 5. See `reports/decisions_log.md` P2-D6.

**Revised realistic targets (Phase-5 to finalize):**

- **GK:** `saves_per_90`, `save_pct`, `clean_sheets_per_90`, `goals_against_per_90` *(psxg unavailable)*
- **DEF:** `tackles_won_per_90`, `interceptions_per_90`, `goals_per_90` *(total tackles/blocks/aerials/prog-passes unavailable)*
- **MID:** `goals_per_90`, `assists_per_90`, `xg_per_90`, `xag_per_90`, `tackles_won_per_90`, `interceptions_per_90` *(key passes/prog-carries/SCA unavailable)*
- **FWD:** `xg_per_90`, `goals_per_90`, `assists_per_90`, `shots_per_90`, `npxg_per_90` / `understat_xa_per_90` *(SCA/prog-carries unavailable)*

**Original targets (superseded — kept for record):**

- ~~GK: `saves_per_90`, `save_pct`, `clean_sheets_per_90`, `goals_against_per_90`, `psxg_per_90`~~
- ~~DEF: `tackles_per_90`, `interceptions_per_90`, `blocks_per_90`, `aerial_won_pct`, `progressive_passes_per_90`, `goals_per_90`~~
- ~~MID: `xg_per_90`, `xag_per_90`, `progressive_passes_per_90`, `progressive_carries_per_90`, `key_passes_per_90`, `tackles_per_90`~~
- ~~FWD: `xg_per_90`, `goals_per_90`, `xag_per_90`, `shots_per_90`, `progressive_carries_per_90`, `sca_per_90`~~

### 6.3 Training data construction

```python
def build_stage1_training_pairs(panel: pd.DataFrame) -> list[tuple]:
    pairs = []
    for player_id, group in panel.groupby('player_id'):
        seasons = group.sort_values('season_end_year')
        for i in range(len(seasons) - 1):
            s_row = seasons.iloc[i]
            s_plus_1_row = seasons.iloc[i + 1]
            # Filters
            if s_row['minutes_played'] < 450: continue
            if s_plus_1_row['minutes_played'] < 450: continue
            if s_row['primary_position'] != s_plus_1_row['primary_position']: continue
            # Only training-window seasons go in training set
            if s_row['season_end_year'] >= 2024: continue  # 2024-25 is test
            pairs.append((build_features(s_row, history=seasons.iloc[:i+1]),
                          extract_targets(s_plus_1_row)))
    return pairs
```

Expected sample sizes after filtering (rough estimate):
- Top-5 only: ~6,000 pairs
- + Lower 4 leagues: ~9,500 pairs

### 6.4 Models per position

For each of GK/DEF/MID/FWD:

1. **Baseline:** Multi-output Ridge regression
2. **Primary candidate A:** Multi-output XGBoost
3. **Primary candidate B:** Per-stat XGBoost ensemble (one model per output stat) — often outperforms multi-output
4. **NN variant:** 3-layer MLP regressor (Keras)
5. **Final:** Stacked ensemble of top-2 from above

### 6.5 Key predictors

- Lagged per-90 stats (current season + prior 1–2 if available)
- Year-over-year deltas
- Age, age², peak distance, age bucket
- Minutes trajectory (proxy for trust/availability)
- League tier change indicator
- Position-conditional z-scores of past stats
- `seasons_in_data` count

---

## 7. Stage 2: Valuation Model

### 7.1 Formulation

```
X = full feature vector at end of season s
    (stats_s + age_s + contract_remaining_s + league_s
     + nationality_s + composite_s + ...)
y = log1p(market_value_eur at end of season s)
```

### 7.2 Training data

**Critical:** Stage 2 is trained on `(actual_observed_stats[s], market_value[s])` pairs — never on Stage 1 projections.

```python
def build_stage2_training_rows(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in panel.iterrows():
        if row['season_end_year'] >= 2024: continue  # test reserve
        if row['minutes_played'] < 450: continue
        if pd.isna(row['market_value_eur']): continue
        rows.append({
            **build_full_features(row),
            'log_market_value': np.log1p(row['market_value_eur']),
        })
    return pd.DataFrame(rows)
```

### 7.3 Models per position

For each of GK/DEF/MID/FWD:

1. **Baseline:** Linear regression on engineered features (interpretable benchmark)
2. **Tree ensemble:** XGBoost, LightGBM, CatBoost — benchmark all three
3. **NN variant:** 3-4 layer dense network (Keras), `huber_loss` for outlier robustness
4. **Final:** Stacked ensemble with linear meta-learner

### 7.4 Multicollinearity handling

Raw stats and composite scores are correlated. Strategy:

- **Linear baseline:** raw stats OR composites, not both
- **Tree-based & NN:** both (no harm)
- **Use VIF analysis** to drop redundant features for linear baseline

### 7.5 Year inflation feature (empirical derivation)

```python
def derive_year_inflation(panel: pd.DataFrame) -> dict:
    """
    For each year, compute the median log_market_value drift relative
    to a reference cohort of players present in both year y and y-1
    with stable minutes.
    """
    inflation = {2022: 1.00}  # reference
    for year in [2023, 2024, 2025]:
        cohort = find_stable_players(panel, year, year - 1)
        mv_ratio = (
            cohort[cohort['season_end_year'] == year]['market_value_eur'].median() /
            cohort[cohort['season_end_year'] == year - 1]['market_value_eur'].median()
        )
        inflation[year] = inflation[year - 1] * mv_ratio
    return inflation
```

Cross-check with Transfermarkt published "total league value" YoY changes if available.

---

## 8. Feature Forwarder (Inference Bridge)

A deterministic component that transforms `X_s` into `X_{s+1}` for Stage 2 inference.

```python
def forward_features(
    X_s: pd.DataFrame,
    projected_stats_s_plus_1: dict,
    inflation_lookup: dict
) -> pd.DataFrame:
    """
    Transform current-season features to next-season equivalents.

    Updates:
      - age + 1
      - contract_remaining_months − 12 (clipped at 0)
      - contract_expires_within_X flags re-derived
      - peak_distance, age buckets re-derived
      - season_end_year + 1
      - year_inflation_multiplier updated
      - stats replaced by Stage 1's projections
      - z-scores, per-90s, composites re-computed from projected stats

    Assumed UNCHANGED:
      - league (no transfer modeling)
      - nationality, continent_group
      - worldcup_25_squad
      - league_value_multiplier, uefa_coefficient
      - fifa_potential (long-horizon stable)
    """
    X_next = X_s.copy()
    X_next['age_at_season_end'] = X_s['age_at_season_end'] + 1
    X_next['contract_remaining_months'] = max(0, X_s['contract_remaining_months'] - 12)
    X_next['season_end_year'] = X_s['season_end_year'] + 1
    X_next['year_inflation'] = inflation_lookup[X_next['season_end_year']]

    # Replace stat columns with projections
    for stat, value in projected_stats_s_plus_1.items():
        X_next[stat] = value

    # Re-derive all computed features
    X_next = recompute_age_features(X_next)
    X_next = recompute_contract_flags(X_next)
    X_next = recompute_per_90_features(X_next)
    X_next = recompute_z_scores(X_next, reference_distribution=training_distribution)
    X_next = recompute_composites(X_next)

    return X_next
```

This component is **deterministic and transparent** — no learning. It's tested as a standalone unit (`tests/test_feature_forwarder.py`).

---

## 9. Discovery Layer

### 9.1 Strategy

After Stage 1 + Stage 2 are trained, apply the full pipeline to lower-league players in 2024-25, compute value gaps, surface candidates.

```python
def discover_hidden_gems(
    lower_league_panel: pd.DataFrame,
    pipeline: Pipeline,
    top_k_per_position: int = 25
) -> pd.DataFrame:
    results = []
    for _, player_row in lower_league_panel.iterrows():
        prediction = pipeline.predict_with_confidence(player_row)
        value_gap = prediction['predicted_value'] - player_row['market_value_eur']
        value_gap_pct = value_gap / player_row['market_value_eur']
        results.append({
            'player_id': player_row['player_id'],
            'predicted_value': prediction['predicted_value'],
            'actual_value': player_row['market_value_eur'],
            'value_gap_eur': value_gap,
            'value_gap_pct': value_gap_pct,
            'confidence': prediction['confidence'],
            'position': player_row['primary_position'],
        })
    df = pd.DataFrame(results)
    df['combined_score'] = df['value_gap_pct'] * df['confidence']
    return df.groupby('position').apply(
        lambda x: x.nlargest(top_k_per_position, 'combined_score')
    )
```

### 9.2 Distribution-shift mitigation

Lower-league players have stats and market values from a distribution different from training (top-5 dominated). Mitigations:

- League multipliers + UEFA coefficients as explicit features
- Lower league players included in Stage 1 training (multi-league projection learning)
- Stage 2 training restricted to top-5 (preserves valuation calibration) — and we **rely on league features** to extrapolate
- Distribution-distance penalty in confidence score (§9.4)

### 9.3 Cross-league calibration validation

Identify players who transferred from lower-4 leagues to top-5 in 2024-25 or summer 2025. For each:

- Pre-transfer prediction (from lower-league stats + applied league multiplier)
- Post-transfer actual market value or transfer fee

This subset (~30–80 players historically) gives empirical calibration evidence. Reported in `notebooks/07_discovery_analysis.ipynb`.

### 9.4 Confidence score

```python
def confidence_score(features: pd.Series, training_centroid: np.ndarray) -> float:
    minutes_score = min(features['minutes_played'] / 2700, 1.0)  # ~30 matches
    history_score = min(features['seasons_in_data'] / 3, 1.0)
    age_score = max(0, 1.0 - abs(features['age_at_season_end'] - 26) / 15)
    league_score = 1.0 if features['league_tier'] == 1 else 0.7
    dist = euclidean_distance(features.to_numpy(), training_centroid)
    dist_score = max(0, 1.0 - dist / dist_normalizer)

    return (0.30 * minutes_score +
            0.20 * history_score +
            0.20 * age_score +
            0.15 * league_score +
            0.15 * dist_score)
```

Weights are starting points; re-tuned in Phase 7 against actual residual magnitudes.

---

## 10. Feature Engineering Catalog

All categories below are implemented in `src/features/`. Feature dictionary lives in `reports/feature_dictionary.md`.

### 10.1 Per-90 metrics (mandatory)

```python
PER_90_STATS = [
    'goals', 'assists', 'shots', 'shots_on_target',
    'xg', 'xag', 'npxg', 'sca', 'gca',
    'progressive_passes', 'progressive_carries', 'key_passes',
    'tackles', 'interceptions', 'blocks', 'clearances',
    'aerial_won', 'aerial_lost',
    'touches', 'progressive_passes_received',
    'fouls_committed', 'fouls_drawn',
    'saves', 'goals_against', 'psxg',
]

for stat in PER_90_STATS:
    df[f'{stat}_per_90'] = df[stat] / (df['minutes_played'] / 90)
```

### 10.2 Position-conditional z-scores (high impact)

```python
for stat in PER_90_STATS:
    col = f'{stat}_per_90'
    df[f'{col}_z_pos'] = df.groupby(['season', 'primary_position'])[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
```

Per-season grouping neutralizes year-wide drift.

### 10.3 Age curve features

```python
df['age_squared'] = df['age_at_season_end'] ** 2
df['peak_distance'] = abs(df['age_at_season_end'] - 27)
df['is_emerging'] = df['age_at_season_end'].between(19, 22).astype(int)
df['is_young_talent'] = (df['age_at_season_end'] < 23).astype(int)
df['is_in_prime'] = df['age_at_season_end'].between(25, 30).astype(int)
df['is_veteran'] = (df['age_at_season_end'] > 32).astype(int)
```

### 10.4 Contract features (highest single-feature ROI)

```python
df['contract_remaining_months'] = (
    (df['contract_end_date'] - df['season_end_date']).dt.days / 30
).clip(lower=0)
df['contract_expires_within_6m'] = (df['contract_remaining_months'] < 6).astype(int)
df['contract_expires_within_12m'] = (df['contract_remaining_months'] < 12).astype(int)
df['contract_long_term'] = (df['contract_remaining_months'] > 36).astype(int)
```

### 10.5 League features

```python
# Starting heuristics; re-derive empirically from training data once available
LEAGUE_VALUE_MULTIPLIER = {
    'Premier League': 1.30, 'La Liga': 1.15, 'Bundesliga': 1.10,
    'Serie A': 1.05, 'Ligue 1': 0.95,
    'Eredivisie': 0.55, 'Liga Portugal': 0.50,
    'Belgian Pro League': 0.45, 'Süper Lig': 0.40,
}
LEAGUE_TIER = {  # 1 = top-5, 2 = secondary European
    'Premier League': 1, 'La Liga': 1, 'Bundesliga': 1, 'Serie A': 1, 'Ligue 1': 1,
    'Eredivisie': 2, 'Liga Portugal': 2, 'Belgian Pro League': 2, 'Süper Lig': 2,
}
UEFA_COEFFICIENTS = {  # Snapshot from current UEFA rankings
    'Premier League': 99.0, 'La Liga': 88.0, 'Serie A': 87.0,
    'Bundesliga': 79.0, 'Ligue 1': 67.0,
    'Eredivisie': 58.0, 'Liga Portugal': 52.0,
    'Belgian Pro League': 49.0, 'Süper Lig': 38.0,
}

df['league_value_multiplier'] = df['league'].map(LEAGUE_VALUE_MULTIPLIER)
df['league_tier'] = df['league'].map(LEAGUE_TIER)
df['uefa_coefficient'] = df['league'].map(UEFA_COEFFICIENTS)
```

**Re-derivation in Phase 4:** After unified panel is built, fit a position-controlled regression of `log_market_value ~ league` and use coefficients as data-driven `LEAGUE_VALUE_MULTIPLIER`.

### 10.6 League × position interaction

```python
df['league_position'] = df['league'].astype(str) + '_' + df['primary_position'].astype(str)
# Then out-of-fold target-encode this column
```

### 10.7 Nationality & continent

```python
CONTINENT_MAP = load_csv('data/external/continent_map.csv')  # Maintained externally
df['continent_group'] = df['nationality'].map(CONTINENT_MAP).fillna('Other')
df['worldcup_25_squad'] = df['player_id'].isin(WC25_ROSTER_SET).astype(int)

# Out-of-fold target encoding for nationality (training only — never on test)
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['nationality'], smoothing=10)
df['nationality_te'] = te.fit_transform(df['nationality'], df['log_market_value'])
```

### 10.8 Composite domain features

```python
df['goal_threat'] = df['xg_per_90'] + 0.5 * df['sca_per_90']
df['creative_threat'] = df['xag_per_90'] + 0.3 * df['key_passes_per_90']
df['defensive_actions_per_90'] = (
    df['tackles_per_90'] + df['interceptions_per_90'] + df['blocks_per_90']
)
df['progressive_actions_per_90'] = (
    df['progressive_passes_per_90'] + df['progressive_carries_per_90']
)
df['aerial_dominance'] = df['aerial_won_pct'] * (df['aerial_won_per_90'] / 5).clip(0, 1)
```

### 10.9 Trajectory features (multi-season — critical for Stage 1)

```python
df = df.sort_values(['player_id', 'season_end_year'])

LAG_STATS = ['xg_per_90', 'minutes_played', 'progressive_passes_per_90',
             'xag_per_90', 'tackles_per_90', 'saves_per_90']
for stat in LAG_STATS:
    df[f'{stat}_lag1'] = df.groupby('player_id')[stat].shift(1)
    df[f'{stat}_lag2'] = df.groupby('player_id')[stat].shift(2)
    df[f'{stat}_yoy_change'] = df[stat] - df[f'{stat}_lag1']
    df[f'{stat}_3yr_mean'] = df.groupby('player_id')[stat].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

df['seasons_in_data'] = df.groupby('player_id').cumcount() + 1
df['cumulative_minutes'] = df.groupby('player_id')['minutes_played'].cumsum()
```

### 10.10 Year inflation multiplier

See §7.5 for empirical derivation.

### 10.11 FIFA enrichment

```python
df = df.merge(
    fifa_ratings_df[['player_id', 'fifa_rating', 'fifa_potential']],
    on='player_id', how='left'
)
# Missing FIFA values → impute with position median (rare, generally non-top-tier players)
```

---

## 11. Validation Framework

Three layered validations, all separately reported.

### 11.1 Stage 1 standalone

**Setup:** train on (2021-22 → 2022-23) and (2022-23 → 2023-24) pairs. Validate on (2023-24 → 2024-25) pairs.

**Metrics per output stat:**
- MAE on each projected stat
- R² per stat
- Forecast skill vs naive baseline ("predict same as last year"): `(MAE_naive − MAE_model) / MAE_naive`
- Per-position breakdown

### 11.2 Stage 2 standalone

**Setup:** train on `(features_s, log_mv_s)` for s ∈ {2021-22, 2022-23, 2023-24}. Validate on 2024-25 hold-out.

**Metrics:**
- MAE (€), MAPE (%)
- R² on log target
- Spearman rank correlation
- Per-quartile MAE (does it work for cheap players AND expensive ones?)
- Per-league residual analysis
- Per-position breakdown

### 11.3 Full pipeline

**Setup:** for each player with 2023-24 stats, run full pipeline (Stage 1 → Forwarder → Stage 2). Compare prediction to 2025 summer market value.

**Metrics:**
- MAE (€), MAPE (%)
- Spearman rank correlation against actual 2025 MV
- Top-K discovery hit rate (§11.5)

### 11.4 Bonus validation: actual transfer fees

For players who actually transferred into top-5 leagues during 2024-25 winter or 2025 summer windows, compare our predicted value to the actual fee paid. Subset is ~50–150 transfers depending on availability.

This is the "absolute truth" reference requested by the instructor. Reported as a dedicated section in the final report.

### 11.5 Top-K discovery metric

The single most domain-relevant number:

```
For top K=20 undervalued players flagged in 2024-25:
  - How many had market value increase by >25% by summer 2025?
  - How many were actually transferred?
  - Average upside captured per flagged player?
```

### 11.6 Cross-validation strategy

For internal hyperparameter tuning within the 2021-22 / 22-23 / 23-24 window:

- `TimeSeriesSplit(n_splits=3)` respecting season order
- Never use random KFold on this dataset

---

## 12. Auxiliary Modules

### 12.1 Clustering module (Weeks 10–12)

**Goal:** group players by **playing style**, not by value.

```python
# Per position: K-Means on z-scored stat profile (market value EXCLUDED)
CLUSTERING_FEATURES = [
    'xg_per_90_z_pos', 'xag_per_90_z_pos',
    'progressive_passes_per_90_z_pos', 'progressive_carries_per_90_z_pos',
    'tackles_per_90_z_pos', 'interceptions_per_90_z_pos',
    'aerial_won_pct', 'pass_completion_pct', ...
]

for pos in ['GK', 'DEF', 'MID', 'FWD']:
    X_pos = df[df['primary_position'] == pos][CLUSTERING_FEATURES]
    # Silhouette-based k selection (k=3..9)
    # K-Means + Hierarchical (sklearn AgglomerativeClustering)
    # 2D visualization via UMAP
```

**Outputs:** cluster assignments, archetypal players (closest to centroid), similarity engine for `find_similar(player_id, n=5)`.

### 12.2 Frequent pattern mining (Week 8)

**Goal:** discover stat patterns among the most undervalued cohort.

```python
top_undervalued = df.nlargest(int(len(df) * 0.10), 'value_gap_pct')

# Binarize each stat against its position-conditional 75th percentile
binary_df = binarize_position_conditional(top_undervalued, KEY_STATS, percentile=75)

from mlxtend.frequent_patterns import apriori, association_rules
frequent = apriori(binary_df, min_support=0.15, use_colnames=True)
rules = association_rules(frequent, metric='lift', min_threshold=1.5)
top_rules = rules.sort_values('lift', ascending=False).head(20)
```

**Output:** report section "Patterns of Undervaluation."

### 12.3 SHAP interpretability

```python
import shap

# Per-position Stage 2 SHAP explainer
for pos in ['GK', 'DEF', 'MID', 'FWD']:
    explainer = shap.TreeExplainer(stage2_models[pos])
    shap_values = explainer.shap_values(X_test_pos)
    # Save: per-player SHAP values for use in demo
    save_shap_artifacts(pos, shap_values, X_test_pos)
```

Every demo prediction shows a SHAP waterfall plot of the top 8 contributing features.

---

## 13. Demo Application (Local)

Local Streamlit app, not deployed. Runs via `streamlit run app/streamlit_app.py`.

### 13.1 Structure

```
app/
├── streamlit_app.py              # Entry; landing page
├── pages/
│   ├── 1_Player_Explorer.py
│   ├── 2_Similar_Players.py
│   ├── 3_Hidden_Gems.py          # Flagship — discovery page
│   ├── 4_Transfer_Recommender.py
│   └── 5_Insights.py
├── components/
│   ├── shap_widget.py
│   ├── player_card.py
│   ├── radar_chart.py
│   └── cluster_viz.py
└── cached_data/                  # Pre-computed predictions, SHAP values
```

### 13.2 Pages

| Page | Function |
|---|---|
| Player Explorer | Player dropdown → radar chart, projected stats, predicted value vs actual, SHAP waterfall |
| Similar Players | Player + N → ranked similar-style players via cluster cosine similarity |
| Hidden Gems | Filters (position, age, max budget, league) → ranked undervalued lower-league players with confidence badges |
| Transfer Recommender | Budget + role + style profile → ranked candidates from full dataset |
| Insights | League premium plots, age curves, cluster UMAP, top patterns, model performance dashboards |

### 13.3 Performance

- Pre-compute all predictions and SHAP values offline
- Stream from `app/cached_data/*.parquet` at runtime
- No model inference on the live app (fast page loads)

---

## 14. Technical Stack & Compute Strategy

### 14.1 Stack

```
Python 3.10+
Data: pandas, polars, pyarrow
Football: soccerdata
Scraping: requests, beautifulsoup4, lxml, rapidfuzz
Modeling: scikit-learn, xgboost, lightgbm, catboost
Tuning: optuna
Encoding: category_encoders
NN: keras (tensorflow) — pick this over torch for compactness
Interpretability: shap
Clustering & viz: umap-learn, plotly, matplotlib, seaborn
Patterns: mlxtend
Demo: streamlit
```

`requirements.txt` is committed in repo root and locked.

### 14.2 Compute strategy: local-first

| Operation | Where | Why |
|---|---|---|
| Scraping (FBref, TM, transfer fees) | Local | Long-running, needs control & logging |
| Data integration & cleaning | Local | Small data, fits in memory easily |
| EDA notebooks | Local | Iterative |
| Feature engineering | Local | Seconds to minutes |
| Baseline & tree models | Local | XGBoost/LightGBM/CatBoost are fast |
| Optuna sweeps (≤100 trials) | Local | Manageable in 1–2 hours |
| Optuna sweeps (≥200 trials) | Colab | Optional, only if local proves slow |
| Neural network training | Local | Tabular NNs are small |
| SHAP computation | Local | Standard |
| Clustering, pattern mining | Local | Fast |
| Streamlit demo | Local | Required (no deployment) |

### 14.3 Colab usage (only when needed)

If a hyperparameter sweep exceeds 2 hours locally:
1. Push code to GitHub
2. In Colab: `!git clone <repo>` + `!pip install -r requirements.txt`
3. Mount Drive to access raw data
4. Run sweep
5. Save best params to repo (commit) → train final model locally on saved params

### 14.4 Storage

| Asset | Location |
|---|---|
| Source code | Git (GitHub private repo) |
| Raw data | `data/raw/` (gitignored — too large) |
| Processed parquet files | `data/processed/` (gitignored if >50MB; else committed) |
| External lookup CSVs | `data/external/` (committed — small, important) |
| Manual override tables | `data/manual/` (committed — important) |
| Trained model pickles | `models/` (gitignored — regenerable) |
| Cached app data | `app/cached_data/` (gitignored — regenerable) |
| Reports & figures | `reports/` (committed) |
| Backups of large raw data | Optional: a shared Drive folder |

---

## 15. Repository Structure

```
AML Project/
│
├── README.md
├── PROJECT_ROADMAP.md             # This document
├── CLAUDE.md                      # Claude Code session reference
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                       # Gitignored
│   │   ├── kaggle/
│   │   ├── fbref/
│   │   ├── transfermarkt/
│   │   └── transfer_fees_2025/
│   ├── interim/                   # Gitignored
│   ├── processed/                 # Gitignored if large
│   │   ├── unified_panel.parquet
│   │   └── features.parquet
│   ├── external/                  # Committed
│   │   ├── uefa_coefficients.csv
│   │   ├── continent_map.csv
│   │   ├── league_tier_map.csv
│   │   └── league_value_multipliers.csv
│   └── manual/                    # Committed
│       ├── manual_id_overrides.csv
│       └── match_log.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_integration.ipynb
│   ├── 03_feature_engineering_check.ipynb
│   ├── 04_stage1_projection_modeling.ipynb
│   ├── 05_stage2_valuation_modeling.ipynb
│   ├── 06_pipeline_validation.ipynb
│   ├── 07_discovery_analysis.ipynb
│   ├── 08_clustering.ipynb
│   ├── 09_pattern_mining.ipynb
│   └── 10_shap_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── kaggle_loader.py
│   │   ├── fbref_scraper.py
│   │   ├── transfermarkt_loader.py
│   │   ├── transfer_fee_scraper.py
│   │   ├── fifa_loader.py
│   │   └── name_resolver.py
│   │
│   ├── integration/
│   │   └── unified_panel_builder.py
│   │
│   ├── features/
│   │   ├── per_90.py
│   │   ├── z_scores.py
│   │   ├── age_curves.py
│   │   ├── contract.py
│   │   ├── league.py
│   │   ├── nationality.py
│   │   ├── composite.py
│   │   ├── trajectory.py
│   │   ├── year_inflation.py
│   │   ├── feature_forwarder.py
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── stage1_projection/
│   │   │   ├── trainer.py
│   │   │   ├── predictor.py
│   │   │   └── targets.py
│   │   ├── stage2_valuation/
│   │   │   ├── trainer.py
│   │   │   └── predictor.py
│   │   └── ensemble.py
│   │
│   ├── pipeline/
│   │   ├── inference.py
│   │   └── confidence.py
│   │
│   ├── discovery/
│   │   ├── runner.py
│   │   └── cross_league_calibration.py
│   │
│   ├── clustering/
│   │   ├── trainer.py
│   │   └── similarity.py
│   │
│   ├── patterns/
│   │   └── apriori_analyzer.py
│   │
│   ├── interpretability/
│   │   └── shap_explainer.py
│   │
│   ├── validation/
│   │   ├── metrics.py
│   │   ├── temporal_split.py
│   │   ├── stage_validators.py
│   │   └── transfer_fee_validator.py
│   │
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── constants.py            # All league maps, coefficients
│
├── scripts/                       # Standalone runnable scripts
│   ├── scrape_fbref.py
│   ├── scrape_transfer_fees.py
│   ├── build_panel.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── precompute_app_data.py
│
├── models/                        # Gitignored
│   ├── stage1/{gk,def,mid,fwd}.pkl
│   └── stage2/{gk,def,mid,fwd}.pkl
│
├── app/                           # Streamlit demo
│   ├── streamlit_app.py
│   ├── pages/
│   ├── components/
│   └── cached_data/
│
├── reports/
│   ├── final_report.md
│   ├── feature_dictionary.md
│   ├── decisions_log.md
│   ├── figures/
│   └── tables/
│
└── tests/
    ├── test_name_resolution.py
    ├── test_feature_forwarder.py
    └── test_pipeline.py
```

---

## 16. Implementation Phases

Sequential by dependency. Independent tracks can parallelize within a phase.

### Phase 0 — Foundation

- Initialize repo per §15
- Configure environment from `requirements.txt`
- Write `src/utils/constants.py` with all maps
- Write `CLAUDE.md` (Claude Code reference)
- Set up logging infrastructure
- README with team info

**Deliverable:** Runnable skeleton, all directories scaffolded.

### Phase 1 — Data acquisition

| Sub-phase | Task | Output |
|---|---|---|
| 1A | Load Kaggle dataset | `data/raw/kaggle/` |
| 1B | `soccerdata` FBref pull: top-5 + lower-4, seasons 2021-22 → 2024-25, all stat types | `data/raw/fbref/<season>/<league>/*.csv` |
| 1C | Clone `transfermarkt-datasets`, extract relevant CSVs | `data/raw/transfermarkt/` |
| 1D | Custom scraper: 2025 transfer fees into top-5 | `data/raw/transfer_fees_2025/fees.csv` |
| 1E | FIFA ratings (Kaggle SoFIFA) | `data/raw/fifa/` |
| 1F | UEFA coefficients, continent map, league tier map (manual) | `data/external/*.csv` |

### Phase 2 — Integration

- 2A: Implement `name_resolver.py` with composite + fuzzy
- 2B: Build master player ID table; create manual override CSV for top-500
- 2C: Join all sources into `unified_panel.parquet`
- 2D: Apply minimum-minutes filter (450 top-5, 300 lower)
- 2E: Normalize position taxonomy
- 2F: Align MV snapshots to season ends (±45 days closest)
- 2G: Generate data quality report (`notebooks/02_data_integration.ipynb`)

### Phase 3 — EDA

Comprehensive `notebooks/01_data_exploration.ipynb`:
- Distribution analysis (log transform justification)
- Missing data patterns by column
- Correlation matrix on stat features
- League-wide summary stats + MV distribution per league (the "league premium" visualization)
- Age × MV scatter by position
- Position taxonomy validation

### Phase 4 — Feature engineering

Implement each module in `src/features/` (per §10):
- `per_90.py`, `z_scores.py`, `age_curves.py`, `contract.py`
- `league.py` (including empirical re-derivation of multipliers)
- `nationality.py`, `composite.py`, `trajectory.py`
- `year_inflation.py` (empirical derivation per §7.5)
- `build_features.py` orchestrator → `features.parquet`
- `feature_forwarder.py` + unit tests

### Phase 5 — Stage 1 (projection)

For each position GK/DEF/MID/FWD:
- 5A: Construct (s, s+1) pairs (§6.3)
- 5B: Ridge multi-output baseline
- 5C: Multi-output XGBoost (Optuna-tuned)
- 5D: Per-stat XGBoost ensemble (Optuna-tuned)
- 5E: MLP regressor (Keras)
- 5F: Pick winner per position via TimeSeriesSplit CV
- 5G: Save to `models/stage1/<pos>.pkl`
- 5H: Stage 1 standalone validation (§11.1) report

### Phase 6 — Stage 2 (valuation)

For each position:
- 6A: Construct training rows (§7.2)
- 6B: Linear regression baseline (with VIF-pruned features)
- 6C: XGBoost / LightGBM / CatBoost benchmark
- 6D: MLP regressor
- 6E: Stacked ensemble (winner)
- 6F: Save to `models/stage2/<pos>.pkl`
- 6G: Stage 2 standalone validation (§11.2) report

### Phase 7 — Pipeline assembly + full validation

- 7A: Implement `src/pipeline/inference.py` (Stage1 → Forwarder → Stage2)
- 7B: Implement `src/pipeline/confidence.py`
- 7C: Run full pipeline on 2024-25 hold-out
- 7D: Full pipeline validation (§11.3)
- 7E: Bonus validation against actual transfer fees (§11.4)
- 7F: Top-K discovery hit rate (§11.5)
- 7G: Re-tune confidence score weights based on observed residuals

### Phase 8 — Discovery layer

- 8A: Apply pipeline to all lower-4 league players in 2024-25
- 8B: Compute value gaps, confidence scores
- 8C: Cross-league calibration validation (§9.3)
- 8D: Save ranked discovery list per position → `data/processed/hidden_gems.csv`
- 8E: `notebooks/07_discovery_analysis.ipynb`

### Phase 9 — Auxiliary modules

- 9A: Clustering (per position, K-Means + Hierarchical) → `notebooks/08_clustering.ipynb`
- 9B: Similar player retrieval engine
- 9C: Apriori on top-undervalued cohort → `notebooks/09_pattern_mining.ipynb`
- 9D: SHAP per-player explanations → `notebooks/10_shap_analysis.ipynb`

### Phase 10 — Demo

- 10A: Scaffold Streamlit app
- 10B: Pre-compute app data (predictions, SHAP, clusters) → `app/cached_data/`
- 10C: Implement all 5 pages
- 10D: Internal usability testing
- 10E: Polish UI

### Phase 11 — Report & presentation

- 11A: Final report draft (`reports/final_report.md`)
- 11B: All figures generated programmatically
- 11C: Final presentation deck
- 11D: Dry-run rehearsals

---

## 17. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-01 | Name matching failure rate >20% | M | H | Manual override for top-500; composite-key matching |
| R-02 | Lower-league FBref data sparser than expected in older seasons | M | M | Tolerate; minimum-minutes filter excludes the worst-affected |
| R-03 | Stage 1 projections poor for young players | H | M | Confidence score flags this; report it transparently |
| R-04 | Distribution shift top-5 → lower league breaks Stage 2 | M | H | League multipliers + cross-league calibration validation (§9.3) |
| R-05 | 2025 transfer fee data sparse | M | M | Bonus validation, not primary; report what we have |
| R-06 | Hyperparameter tuning consumes excessive compute | M | L | Optuna early stopping; bounded search spaces |
| R-07 | Year inflation feature misspecified | M | M | Empirical derivation + Transfermarkt cross-check |
| R-08 | Team member loss | M | H | All code in shared repo, modular ownership |
| R-09 | Position taxonomy mixing (e.g., "CM,DM") | H | M | Explicit normalization function with unit tests |
| R-10 | Feature forwarder bugs propagate to predictions | M | H | Standalone unit tests; deterministic = testable |

---

## 18. Curriculum Alignment

| Week | Topic | Project coverage |
|---|---|---|
| W1 | Intro | Problem framing, generative/discriminative |
| W2-3 | Data Munging | §5 — multi-source integration, cleaning, name resolution |
| W4-5 | Classification | Player tier classification + value gap flagging |
| W6 | Transformation | log target, encoding, scaling, VIF |
| W7 | Regression | Stage 1 + Stage 2 both regression |
| **W8** | **Frequent Patterns** | **§12.2 Apriori — full coverage** |
| W10-11 | Project review | Midterm presentation already delivered |
| W12 | Clustering | §12.1 K-Means + Hierarchical |
| W13 | Deep Learning (CNN) | Tabular NNs in both stages |
| W14 | Streaming | Mentioned in future work; not implemented |

**Differentiator:** most student projects skip Week 8 entirely. Our Apriori module ensures full curriculum coverage.

---

## 19. Claude Code Workflow

### 19.1 Session opening protocol

Every Claude Code session begins with:
```
1. "Read CLAUDE.md first."
2. "Then read PROJECT_ROADMAP.md sections [relevant §§]."
3. "We are implementing Phase X."
4. "Goal of this session: [specific deliverable]."
```

### 19.2 Phase-by-phase prompt templates

**Phase 0 — Foundation**
```
Read CLAUDE.md and PROJECT_ROADMAP.md §15 fully.
Scaffold the repository structure exactly as specified.
- Create all directories listed.
- Initialize git with a sensible .gitignore (cover data/raw/, data/interim/,
  data/processed/*.parquet, models/, app/cached_data/, *.pyc, __pycache__/,
  .ipynb_checkpoints/, .env).
- Create requirements.txt per §14.1.
- Create src/utils/constants.py with all maps from §10.5 and §10.7.
- Write a README.md with team info and setup instructions.
After completion, run `tree -L 3 -I '__pycache__'` and show me the result.
```

**Phase 1A — Kaggle loader**
```
Read PROJECT_ROADMAP.md §5.1 and §5.3.
Implement src/data/kaggle_loader.py to load the Hubert Sidorowicz
Football Players Stats 2024-2025 dataset.
The user will manually download the dataset to data/raw/kaggle/.
The loader should:
- Read all CSVs from data/raw/kaggle/
- Apply initial schema normalization (column name standardization)
- Output a single DataFrame matching the §5.3 schema, partial coverage acceptable
- Log row counts and column inventory
Test it on the dataset and report what columns are present vs missing
compared to the target schema.
```

**Phase 1B — FBref scraper**
```
Read PROJECT_ROADMAP.md §5.1, §5.2, §5.5.
Implement src/data/fbref_scraper.py using the soccerdata library.
Scrape:
- Top-5 leagues + Eredivisie + Liga Portugal + Belgian Pro League + Süper Lig
- Seasons: 2021-22, 2022-23, 2023-24, 2024-25
- All stat types (standard, shooting, passing, defense, possession, misc, goalkeeping)
Use soccerdata's caching. Save outputs to data/raw/fbref/<season>/<league>/<stat_type>.csv.
Write a corresponding script in scripts/scrape_fbref.py that can be run
from the command line. Include progress logging and graceful resume after interruption.
```

**Phase 2 — Integration**
```
Read PROJECT_ROADMAP.md §5.3, §5.4.
Implement src/data/name_resolver.py per §5.4.
Then implement src/integration/unified_panel_builder.py that:
- Loads all raw sources
- Resolves player IDs via the name resolver
- Joins everything per the schema in §5.3
- Applies the season MV snapshot alignment per D-16
- Writes data/processed/unified_panel.parquet
- Logs comprehensive match/unmatch counts per source
Also produce notebooks/02_data_integration.ipynb showing
quality metrics: match rates, coverage gaps, sample resolved names.
```

(Templates for Phases 3–11 follow the same pattern. See CLAUDE.md for the canonical short version.)

### 19.3 Coding principles (enforced)

1. **Modularity.** Each feature transform, each model, each metric in its own file. Notebooks are reports, not engines.
2. **Determinism.** Set seeds everywhere. Save splits to disk.
3. **No silent drops.** Every row-removing filter logs `(input_count, output_count, reason)`.
4. **No magic numbers.** All constants in `src/utils/constants.py`.
5. **Type hints + docstrings** on every public function.
6. **Defensive joins.** Every merge logs `(left, right, result, matched, unmatched)`.
7. **Save intermediate state.** Parquet at every phase boundary.
8. **Tests for critical paths.** `test_name_resolution.py`, `test_feature_forwarder.py`, `test_pipeline.py`.
9. **Document as you go.** Update `feature_dictionary.md` and `decisions_log.md` with every change.

### 19.4 Anti-patterns to flag explicitly to Claude Code

- **Data leakage via target encoding** — must be out-of-fold
- **Random train/test split** — always temporal
- **Training Stage 2 on Stage 1's projections** — Stage 2 sees actual stats
- **Using market value as a feature** — only as target
- **Position string concatenation issues** — "CM,DM" must be normalized to primary "MID"
- **Inferring without the Feature Forwarder** — must transform features to s+1

### 19.5 Iteration loop

```
1. User gives Claude Code a phase prompt
2. Claude Code implements, produces output
3. User shares output back to Claude (architect)
4. Architect reviews, identifies issues / next prompt
5. Loop until phase complete
6. Move to next phase
```

When Claude Code's output is fine, the architect (this assistant) gives the next phase prompt. When the output needs correction, the architect drafts a targeted fix prompt. When the work requires the user directly (e.g., manual override table, dataset download), the architect routes to the user.

---

*End of document.*
