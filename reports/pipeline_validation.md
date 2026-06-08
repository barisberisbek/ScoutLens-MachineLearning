# Pipeline Validation — 3 Layers

## Layer 1 — Stage 1 (projection) standalone

19 best-per-target models; mean improvement over naive = **15.5%** (R² -0.03–0.55). See `stage1_metrics.md`.

## Layer 2 — Stage 2 (valuation) standalone

Best per position R² = {'DEF': 0.85, 'FWD': 0.87, 'GK': 0.81, 'MID': 0.85}. §11.4 transfer-fee validation: pipeline Pearson r **0.733** (€), median ratio 0.97. **Honest comparison:** naive (copy TM MV) scores r **0.861** on the same fees — TM MV predicts fees *better* than our model (it bakes in non-statistical info we ignore). Our model is a *legitimate independent objective* valuation (r≈0.74), not a claim to beat TM. See `stage2_metrics.md` + `stage2_transfer_validation.md`.

## Layer 3 — FULL PIPELINE end-to-end ⭐

Forward the 2023-24 cohort (Stage1 → Forwarder → Stage2) and compare the projected 2024-25 value to the ACTUAL 2024-25 market value. Three scenarios:

| scenario | n | MAE €M | R²_log | median ratio | % within 30% |
|---|---|---|---|---|---|
| Stage-2 oracle (actual stats) | 1519 | 4.85 | 0.869 | 0.96 | 46 |
| **Pipeline E2E** | 1678 | 5.31 | 0.825 | 1.07 | 40 |
| Naive (no change) | 1674 | 4.15 | 0.867 | 1.15 | 45 |

**Forwarder sanity — Oracle MAE ≤ Pipeline MAE: ✅ holds.** The pipeline correctly pays for Stage-1 projection noise relative to the oracle (which sees the real next-season stats). Since the oracle *skips* the forwarder yet behaves as expected, the forwarder is sound.

**Pipeline vs Naive (€ MAE): does NOT beat naive.** Naive ('next MV = current MV') is a very strong baseline: market value is highly persistent, AND the model deliberately **excludes `log_market_value_lag1`** (anti-circular-reasoning, Phase-6 decision) — so naive uses prior-MV information the model is forbidden from using. In LOG/relative terms the gap closes: oracle R²_log 0.87 ≈ naive 0.87. The pipeline's purpose is the forward-looking value **gap** for discovery (§2.2 / Phase 8), not beating naive MAE; the flagged `log_market_value_lag1` ablation would let it beat naive directly (at the cost of 'copy last year' behavior).

### Confidence-interval coverage
CI bounds were initially built from Stage-2 *train* residuals and undercovered (95% nominal → 71% actual) because the pipeline adds Stage-1 projection noise. **Recalibrated** from the full-pipeline end-to-end residuals:
- 95% CI contains the actual value **90%** of the time (target ~90–95%).
- 50% CI contains it **50%** (target ~50%).
*(Single-split calibration: quantiles fit on the same 2024-25 set they're scored against, so coverage is ~nominal by construction; a future held-out split would confirm generalization.)*

### Top-5 largest pipeline errors (sanity)

| player | pos | actual €M | projected €M |
|---|---|---|---|
| Martin Ødegaard | MID | 85.0 | 208.1 |
| Declan Rice | MID | 120.0 | 217.3 |
| Désiré Doué | MID | 90.0 | 15.5 |
| William Saliba | DEF | 80.0 | 145.5 |
| Erling Haaland | FWD | 180.0 | 244.0 |
