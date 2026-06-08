# Decisions Log

Implementation decisions that resolve ambiguity in `PROJECT_ROADMAP.md` (not deviations
from locked D-xx decisions — those would need team consensus). Each entry: date, context,
decision, rationale.

## Phase 2 — Data Integration

### P2-D1 (2026-06-08) — Min-minutes filter: keep + flag, never drop
**Context:** §2D lists the 450/300 min-minutes filter as a Phase-2 step, but §6.3 applies
it at Stage-1 training-pair construction.
**Decision:** The unified panel KEEPS all rows and carries `meets_min_minutes` (bool) +
`min_minutes_threshold` (450 top-5 / 300 lower). Physical filtering happens at Stage-1/2
modeling entry, not in the panel.
**Rationale:** Honors the "no silent row drops" anti-pattern; preserves multi-season
trajectory history (§10.9) and discovery candidates; §6.3 is authoritative on *where* the
filter lives. (User-approved, 2026-06-08.)

### P2-D2 (2026-06-08) — Unmatched players: keep with deterministic synthetic id
**Context:** ~18% of FBref backbone rows have no Transfermarkt match (mostly players who
have since left the 9-league set, so no current TM metadata).
**Decision:** Keep them with `player_id = "synthetic_" + sha256(f"{norm_name}|{birth_year}|
{nationality_code}")[:12]` (stable across runs), `resolve_method="synthetic"`, and a
`data_richness` column (`"full"` vs `"synthetic_basic"`). MV/contract/FIFA stay null.
**Rationale:** These are the lower-league discovery cohort; dropping violates "no silent
row drops". Stage 2 (needs MV) auto-filters them via `market_value.notna()`. (User-approved.)

### P2-D4 (2026-06-08) — Same-id namesake guards (impl. detail)
**Context:** Transfermarkt lists some players as mononyms (e.g. "Gabriel" = Gabriel
Magalhães). Under `token_set_ratio` a single-token name scores 100 against every
"Gabriel ⟨surname⟩", so distinct players collapsed onto one id and were wrongly summed.
FBref has no birth-DATE to disambiguate same-name/same-year/same-nationality namesakes.
**Decision:** Two post-resolution guards (`split_id_collisions`, `split_minutes_overflow`):
(1) a player_id mapping to >1 distinct normalized name keeps the id for the
dominant-minutes player, others → synthetic; (2) a (player_id, season) summing past 3,800
minutes (physically impossible for one player) is split, max-minutes stint keeps the id,
others → club-salted synthetic. Re-assigned players are kept + flagged `synthetic_split`
(prime manual-override candidates for Session 3), never dropped or wrongly summed.

### P2-D5 (2026-06-08) — Known soccerdata gaps left as NaN (no fabrication)
**Context:** soccerdata's FBref tables carry no xG/npxG/xAG, no progression COUNTS
(PrgP/PrgC), no aerial-duel columns, and `keeper_adv` Expected-family (PSxG) is null.
**Decision:** Leave these NaN in the panel (xG filled from Kaggle/Understat for top-5
only; psxg/aerials/PrgP unavailable except Kaggle 2024-25 top-5). Proxy/imputation is a
Phase-4 decision. `is_loan` is a `False` placeholder — D-15 loan detection deferred (the
current TM snapshot does not cleanly expose loan status).

### P2-D3 (2026-06-08) — `xag` and `understat_xa` stay as separate columns
**Context:** FBref/Kaggle ship `xag` (expected assisted goals); Understat ships `xa`
(expected assists) — related but not the same metric. `xag` is otherwise only available
for 2024-25 top-5 (Kaggle).
**Decision:** Store `xag` (FBref/Kaggle only) and `understat_xa` (Understat) as distinct
columns. The coalesce decision is deferred to Phase 4, which may derive a position-based
`understat_xa → xag` conversion factor.
**Rationale:** Avoids mixing two metrics in one column now; consistent with deferring the
xG-gap resolution to Phase 4 (per memory.md). (User-approved.)
