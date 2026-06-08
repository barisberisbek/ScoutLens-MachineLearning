"""§11.4 — bonus validation against ACTUAL 2024-25 transfer fees.

Market value (the model's target) is Transfermarkt's subjective estimate; a real transfer
fee is a harder real-world signal the model never trained on. For each paid transfer we
take the player's PRE-transfer (2023-24) features, predict their value with the Stage-2
model, and compare to the fee actually paid.

Caveat: 2023-24 is a Stage-2 train season, so those MV predictions are in-sample — but the
comparison is to the FEE (never a training target), so it remains a valid external check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from src.data.name_resolver import normalize_name
from src.models.stage1.target_specs import POSITIONS
from src.models.stage2 import data_loader, persist
from src.models.stage2.evaluate import select_best
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PRE_SEASON = "2023-24"   # season before the 2024-25 transfer window


def _transfer_csv() -> pd.DataFrame:
    p = project_root() / "data" / "raw" / "transfer_fees_2025" / "transfer_fees_2024_25.csv"
    t = pd.read_csv(p)
    return t[t["transfer_fee_eur"] > 0].reset_index(drop=True)


def _match_transfers(transfers: pd.DataFrame, f_pre: pd.DataFrame) -> pd.DataFrame:
    """Match each transfer to a pre-transfer features row by normalized name + nearest age."""
    index: dict[str, list[tuple]] = {}
    for row in f_pre.itertuples(index=True):
        index.setdefault(normalize_name(row.player_name), []).append(
            (row.Index, float(row.age_at_season_end) if pd.notna(row.age_at_season_end) else np.nan,
             row.primary_position))
    rows = []
    for t in transfers.itertuples(index=False):
        cands = index.get(normalize_name(t.player_name), [])
        if not cands:
            continue
        idx, age, pos = min(cands, key=lambda c: abs((c[1] if c[1] == c[1] else 1e9) - t.player_age))
        if not (abs((age if age == age else 1e9) - t.player_age) <= 3):  # nearest age within 3y
            continue
        rows.append({"player_name": t.player_name, "pre_index": idx, "position": pos,
                     "fee_eur": float(t.transfer_fee_eur), "to_league": t.to_league})
    return pd.DataFrame(rows)


def run_transfer_validation(metrics_df: pd.DataFrame, features_path: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (per-transfer prediction df, aggregate metrics dict)."""
    feats_full = load_parquet(features_path or data_loader._features_path())
    f_pre = feats_full[feats_full["season"] == _PRE_SEASON].copy()
    transfers = _transfer_csv()
    matched = _match_transfers(transfers, f_pre)
    logger.info("Transfer validation: %d/%d paid transfers matched to a %s features row",
                len(matched), len(transfers), _PRE_SEASON)

    best_per_pos = select_best(metrics_df).set_index("position")["model"].to_dict()
    records = []
    for pos in POSITIONS:
        sub = matched[matched["position"] == pos]
        if sub.empty or pos not in best_per_pos:
            continue
        _, _, feats = data_loader.load(pos, features_path)
        model = persist.load(pos, best_per_pos[pos])
        X = f_pre.loc[sub["pre_index"], feats]
        pred_mv = np.expm1(model.predict(X))
        for (name, fee, league), pmv in zip(sub[["player_name", "fee_eur", "to_league"]].values, pred_mv):
            records.append({"player_name": name, "position": pos, "to_league": league,
                            "fee_eur": float(fee), "predicted_mv_eur": float(pmv),
                            "ratio": float(pmv / fee)})
    res = pd.DataFrame(records)
    return res, _aggregate(res)


def _aggregate(res: pd.DataFrame) -> dict:
    if res.empty:
        return {"n": 0}
    log_pred = np.log1p(res["predicted_mv_eur"]); log_fee = np.log1p(res["fee_eur"])
    return {
        "n": int(len(res)),
        "median_ratio": float(res["ratio"].median()),
        "mean_ratio": float(res["ratio"].mean()),
        "pearson_mv_fee": float(pearsonr(res["predicted_mv_eur"], res["fee_eur"])[0]),
        "pearson_log": float(pearsonr(log_pred, log_fee)[0]),
        "log_mae": float(np.abs(log_pred - log_fee).mean()),
        "r2_log": float(r2_score(log_fee, log_pred)),
    }


def write_report(res: pd.DataFrame, agg: dict) -> None:
    """Write reports/stage2_transfer_validation.md + a predicted-vs-fee scatter PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eda.style import save_fig, set_eda_style

    lines = ["# Stage 2 — Transfer-Fee Validation (§11.4)", "",
             "Predicted value (from PRE-transfer 2023-24 features) vs the ACTUAL 2024-25 fee. "
             "Fee is never a training target → external real-world check. "
             "*(2023-24 is a train season, so the MV prediction is in-sample; the fee comparison is not.)*", "",
             "## Aggregate", "", "| metric | value |", "|---|---|",
             f"| matched transfers | {agg.get('n', 0)} |",
             f"| median predicted/fee ratio | {agg.get('median_ratio', float('nan')):.2f} |",
             f"| mean ratio | {agg.get('mean_ratio', float('nan')):.2f} |",
             f"| Pearson r (€ space) | {agg.get('pearson_mv_fee', float('nan')):.3f} |",
             f"| Pearson r (log space) | {agg.get('pearson_log', float('nan')):.3f} |",
             f"| R² (log space) | {agg.get('r2_log', float('nan')):.3f} |",
             f"| log MAE | {agg.get('log_mae', float('nan')):.3f} |", ""]

    if not res.empty:
        res = res.assign(err=res["predicted_mv_eur"] - res["fee_eur"])
        for title, sub in [("Top-10 OVER-predicted (model ≫ fee)", res.nlargest(10, "err")),
                           ("Top-10 UNDER-predicted (model ≪ fee)", res.nsmallest(10, "err"))]:
            lines += [f"## {title}", "", "| player | pos | fee €M | pred €M | ratio |", "|---|---|---|---|---|"]
            for r in sub.itertuples(index=False):
                lines.append(f"| {r.player_name} | {r.position} | {r.fee_eur/1e6:.1f} | "
                             f"{r.predicted_mv_eur/1e6:.1f} | {r.ratio:.2f} |")
            lines.append("")
        lines += ["## Per-position", "", "| pos | n | median ratio | Pearson r (log) |", "|---|---|---|---|"]
        for pos, g in res.groupby("position"):
            lp, lf = np.log1p(g["predicted_mv_eur"]), np.log1p(g["fee_eur"])
            r = pearsonr(lp, lf)[0] if len(g) > 2 else float("nan")
            lines.append(f"| {pos} | {len(g)} | {g['ratio'].median():.2f} | {r:.3f} |")
        lines.append("")

        set_eda_style()
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(res["fee_eur"] / 1e6, res["predicted_mv_eur"] / 1e6, s=18, alpha=0.5, color="#0173B2")
        lim = max(res["fee_eur"].max(), res["predicted_mv_eur"].max()) / 1e6
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="predicted = fee")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("actual transfer fee (€M, log)"); ax.set_ylabel("predicted value (€M, log)")
        ax.set_title(f"Stage-2 valuation vs transfer fee (r_log={agg.get('pearson_log', float('nan')):.2f}, n={agg.get('n')})")
        ax.legend()
        save_fig(fig, "stage2_transfer_scatter")
        lines += ["![predicted vs fee](figures/stage2_transfer_scatter.png)", ""]

    path = project_root() / "reports" / "stage2_transfer_validation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
