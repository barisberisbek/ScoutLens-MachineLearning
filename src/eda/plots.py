"""EDA figure functions — one per figure, each returns a Figure and saves a PNG.

Called from notebooks/01_data_exploration.ipynb. All data crunching is delegated to
``src.eda.summary``; this module only renders.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda import summary as S
from src.eda.style import (
    LEAGUE_ORDER,
    POSITION_ORDER,
    SEASON_ORDER,
    TOP5_LEAGUES,
    euro_millions,
    save_fig,
)


def _caption(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, style="italic", color="#555")


# ── S1 ───────────────────────────────────────────────────────────────────────
def fig01_rows_by_league_season(panel: pd.DataFrame):
    mat = (panel.groupby(["league", "season"]).size().unstack()
           .reindex(LEAGUE_ORDER)[SEASON_ORDER])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "player-rows"}, ax=ax)
    ax.set_title("Panel coverage: player-rows per league × season")
    ax.set_xlabel(""); ax.set_ylabel("")
    _caption(fig, "Balanced ~480–620 rows per league-season; the modeling substrate is even across the 9×4 grid.")
    save_fig(fig, "fig01_rows_by_league_season")
    return fig


# ── S2 target ─────────────────────────────────────────────────────────────────
def fig02_mv_raw_vs_log_hist(panel: pd.DataFrame):
    t = S.target_distribution(panel)
    raw = panel["market_value_eur"].dropna() / 1e6
    logv = np.log1p(panel["market_value_eur"].dropna())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(raw, bins=60, ax=axes[0], color="#DE8F05")
    axes[0].set_title(f"Raw market value (skew={t['skew_raw']:.1f}, kurt={t['kurtosis_raw']:.0f})")
    axes[0].set_xlabel("market value (€M)"); axes[0].set_ylabel("players")
    sns.histplot(logv, bins=60, ax=axes[1], color="#0173B2", kde=True)
    axes[1].set_title(f"log1p(MV) (skew={t['skew_log']:.2f}, Shapiro W={t['shapiro_log_stat']:.3f})")
    axes[1].set_xlabel("log1p(market value)"); axes[1].set_ylabel("players")
    fig.suptitle("Target transform: raw MV is extreme right-skew; log1p is ≈normal")
    _caption(fig, "Empirically justifies Stage-2's log1p target (D-05): a raw target would inflate quadratic error on top-tier values.")
    save_fig(fig, "fig02_mv_raw_vs_log_hist")
    return fig


def fig03_mv_median_by_season(panel: pd.DataFrame):
    m = S.mv_medians(panel)
    s = m["by_season"] / 1e6
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(s.index, s.values, color="#0173B2")
    for i, (yoy) in enumerate(m["yoy_pct"]):
        ax.annotate(f"+{yoy:.0f}%", (i + 1, s.values[i + 1]), ha="center", va="bottom", fontsize=9, color="#C44")
    ax.set_title("Median market value by season — YoY inflation")
    ax.set_ylabel("median MV (€M)"); ax.set_xlabel("")
    _caption(fig, "Median MV rises +10%→+20% YoY — the empirical basis for the D-07 year-inflation multiplier feature.")
    save_fig(fig, "fig03_mv_median_by_season")
    return fig


def fig04_mv_median_by_league(panel: pd.DataFrame):
    m = S.mv_medians(panel)
    s = (m["by_league"] / 1e6).reindex(LEAGUE_ORDER)
    colors = ["#0173B2" if lg in TOP5_LEAGUES else "#DE8F05" for lg in s.index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(s.index[::-1], s.values[::-1], color=colors[::-1])
    ax.set_title("Median market value by league (blue = top-5, orange = lower-4)")
    ax.set_xlabel("median MV (€M)")
    ax.annotate(f"PL / Süper Lig = {m['premium_pl_vs_superlig']:.1f}×\nLa Liga / Süper Lig = {m['premium_laliga_vs_superlig']:.1f}×",
                xy=(0.55, 0.15), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    _caption(fig, "League premium spans 3.8×–18.8× — the discovery thesis (D-08): comparable players are far cheaper in lower-4 leagues.")
    save_fig(fig, "fig04_mv_median_by_league")
    return fig


def fig05_mv_by_position_box(panel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = panel.dropna(subset=["market_value_eur"]).copy()
    df["mv_m"] = df["market_value_eur"] / 1e6
    sns.boxplot(data=df, x="primary_position", y="mv_m", order=POSITION_ORDER, ax=ax, showfliers=False)
    ax.set_yscale("log")
    ax.set_title("Market value by position (log scale, outliers hidden)")
    ax.set_xlabel(""); ax.set_ylabel("market value (€M, log)")
    _caption(fig, "Median ordering MID>DEF≈FWD>GK; FWD's long upper tail (superstars) lifts its MEAN above MID but not its median.")
    save_fig(fig, "fig05_mv_by_position_box")
    return fig


# ── S3 missingness ───────────────────────────────────────────────────────────
def fig06_null_rate_top(panel: pd.DataFrame):
    miss = S.missingness(panel)
    nr = miss["null_rate"]
    nr = nr[nr > 0.02].head(25)
    palette = {"source_gap": "#C44E52", "structural_gk": "#0173B2",
               "structural_outfield": "#56B4E9", "target_sparse": "#DE8F05",
               "contract_sparse": "#CC78BC", "other": "#999999"}
    colors = [palette.get(miss["kinds"].get(c, "other"), "#999") for c in nr.index]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(nr.index[::-1], (nr.values * 100)[::-1], color=colors[::-1])
    ax.set_title("Top null-rate columns, colored by cause")
    ax.set_xlabel("% null")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in palette.values()]
    ax.legend(handles, palette.keys(), fontsize=8, loc="lower right")
    _caption(fig, "Two distinct causes: structural (GK stats null for outfield) vs source-gap (xG/xAG/FIFA — by design). No accidental gaps.")
    save_fig(fig, "fig06_null_rate_top")
    return fig


def fig07_missing_by_source_season(panel: pd.DataFrame):
    cols = {"xg": "xg", "xag": "xag", "understat_xa": "understat_xa",
            "fifa_potential": "fifa_potential", "fifa_rating": "fifa_rating",
            "contract": "contract_remaining_months", "market_value": "market_value_eur"}
    mat = pd.DataFrame({
        lbl: panel.groupby("season")[col].apply(lambda x: x.isna().mean() * 100).reindex(SEASON_ORDER)
        for lbl, col in cols.items()
    }).T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100,
                cbar_kws={"label": "% null"}, ax=ax)
    ax.set_title("Source-gap missingness: % null by source × season")
    ax.set_xlabel(""); ax.set_ylabel("")
    _caption(fig, "FC25 potential 100% null in 2024-25; xG sparse (top-5 only); MV sparsest in 2024-25 (recent snapshots thin). All expected.")
    save_fig(fig, "fig07_missing_by_source_season")
    return fig


# ── S4 position ──────────────────────────────────────────────────────────────
def fig08_position_by_league_heatmap(panel: pd.DataFrame):
    ps = S.position_stats(panel)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(ps["by_league"], annot=True, fmt="d", cmap="Greens",
                cbar_kws={"label": "players"}, ax=ax)
    ax.set_title(f"Position counts by league (GK = {ps['gk_pct']:.1f}% of panel)")
    ax.set_xlabel(""); ax.set_ylabel("")
    _caption(fig, f"GK ≈ 8% everywhere (realistic squad mix); {ps['pct_position_changers']:.0f}% of players change primary position across seasons (winger MID↔FWD reclass).")
    save_fig(fig, "fig08_position_by_league_heatmap")
    return fig


# ── S5 age ───────────────────────────────────────────────────────────────────
def fig09_age_hist(panel: pd.DataFrame):
    a = S.age_stats(panel)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.histplot(panel["age_at_season_end"].dropna(), bins=40, ax=ax, color="#0173B2")
    ax.axvline(a["median"], color="#C44", ls="--", label=f"median {a['median']:.0f}")
    ax.set_title("Age distribution at season end")
    ax.set_xlabel("age (years)"); ax.set_ylabel("player-seasons"); ax.legend()
    _caption(fig, f"Median {a['median']:.0f}, range {a['min']:.0f}–{a['max']:.0f}; {a['n_implausible']} implausible (<15/>43). Age is a top Stage-2 driver (age curve, Phase 4).")
    save_fig(fig, "fig09_age_hist")
    return fig


def fig10_age_vs_mv_lowess(panel: pd.DataFrame):
    a = S.age_stats(panel)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, pos in zip(axes.flat, POSITION_ORDER):
        sub = panel[(panel.primary_position == pos)].dropna(subset=["age_at_season_end", "market_value_eur"])
        ax.scatter(sub["age_at_season_end"], sub["market_value_eur"] / 1e6, s=6, alpha=0.15, color="#0173B2")
        sns.regplot(x=sub["age_at_season_end"], y=sub["market_value_eur"] / 1e6, lowess=True,
                    scatter=False, ax=ax, line_kws={"color": "#C44", "lw": 2})
        ax.set_yscale("log")
        peak = a["peak_age_by_position"][pos]
        if peak:
            ax.axvline(peak, color="#444", ls=":", lw=1)
            ax.set_title(f"{pos} — peak MV age ≈ {peak:.0f}")
        ax.set_xlabel("age"); ax.set_ylabel("MV (€M, log)")
    fig.suptitle("Age × market value by position (LOWESS) — the value curve")
    _caption(fig, "GK peaks latest (~30); outfield peaks ~25–26. Confirms age-curve features matter and should be position-specific (Phase 4/5).")
    save_fig(fig, "fig10_age_vs_mv_lowess")
    return fig


# ── S6 trajectories ──────────────────────────────────────────────────────────
def fig11_case_study_trajectories(panel: pd.DataFrame):
    cs = S.select_case_studies(panel)
    items = list(cs.items())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (label, info) in zip(axes.flat, items):
        rows = panel[panel.player_id == info["player_id"]].sort_values("season")
        ax.plot(rows["season"], rows["market_value_eur"] / 1e6, "o-", color="#0173B2", label="MV (€M)")
        ax.set_ylabel("MV (€M)", color="#0173B2")
        ax2 = ax.twinx()
        ax2.plot(rows["season"], rows["goals"], "s--", color="#DE8F05", label="goals")
        ax2.set_ylabel("goals", color="#DE8F05")
        ax.set_title(f"{info['name']}\n{label.split('(')[0].strip()}", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Career trajectories — value vs production across seasons")
    _caption(fig, "Distinct archetypes (young riser / peak / late-career / cross-tier mover) motivate trajectory features (lag-1, momentum) for Stage 1.")
    save_fig(fig, "fig11_case_study_trajectories")
    return fig


# ── S7 league comparison ─────────────────────────────────────────────────────
def fig12_fwd_goals_per90_by_league(panel: pd.DataFrame):
    fwd = panel[(panel.primary_position == "FWD") & (panel.minutes_played >= 900)].copy()
    med = fwd.groupby("league")["goals_per_90"].median().reindex(LEAGUE_ORDER)
    colors = ["#0173B2" if lg in TOP5_LEAGUES else "#DE8F05" for lg in med.index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(med.index[::-1], med.values[::-1], color=colors[::-1])
    ax.set_title("Median FWD goals per 90 by league (≥900 min)")
    ax.set_xlabel("goals per 90")
    _caption(fig, "Scoring rates are broadly similar across tiers — lower-4 forwards aren't less productive, just cheaper (discovery signal).")
    save_fig(fig, "fig12_fwd_goals_per90_by_league")
    return fig


# ── S8 xG calibration ────────────────────────────────────────────────────────
def fig13_xg_vs_goals_scatter(panel: pd.DataFrame):
    x = S.xg_calibration(panel)
    df = panel.dropna(subset=["xg", "goals"])
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for src, color in [("understat", "#0173B2"), ("kaggle", "#DE8F05")]:
        s = df[df.xg_source == src]
        ax.scatter(s["xg"], s["goals"], s=8, alpha=0.25, color=color, label=f"{src} (r={x['by_source'].get(src,{}).get('pearson',float('nan')):.3f})")
    lim = max(df["xg"].max(), df["goals"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="goals = xG")
    ax.set_title(f"xG vs goals (overall r = {x['pearson_overall']:.3f}, n={x['n']})")
    ax.set_xlabel("expected goals (xG)"); ax.set_ylabel("actual goals"); ax.legend()
    _caption(fig, "Strong calibration; Understat runs slightly conservative (resid −0.14) vs Kaggle (−0.02) — pick one source per row (done in Phase 2).")
    save_fig(fig, "fig13_xg_vs_goals_scatter")
    return fig


# ── S9 provenance ────────────────────────────────────────────────────────────
def fig14_provenance(panel: pd.DataFrame):
    pr = S.provenance_stats(panel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rm = pr["resolve_method"]
    axes[0].barh(rm.index[::-1], rm.values[::-1], color="#0173B2")
    axes[0].set_title(f"Name-resolution method ({pr['tm_matched_pct']:.0f}% TM-matched)")
    axes[0].set_xlabel("player-rows")
    sns.heatmap(pr["synthetic_pct_by_league_season"], annot=True, fmt=".0f", cmap="Oranges",
                cbar_kws={"label": "% synthetic"}, ax=axes[1])
    axes[1].set_title("Synthetic (no-TM-match) % by league × season")
    axes[1].set_xlabel(""); axes[1].set_ylabel("")
    fig.suptitle("Provenance — Phase-2 resolution quality")
    _caption(fig, "82% TM-matched. Synthetic rows (no MV, full stats) concentrate in lower-4 + older seasons = the discovery cohort.")
    save_fig(fig, "fig14_provenance")
    return fig


# ── S10 outliers ─────────────────────────────────────────────────────────────
def fig15_top10_mv(panel: pd.DataFrame):
    top = pd.DataFrame(S.outliers(panel)["top10_mv"])
    top["label"] = top["player_name"] + " (" + top["season"] + ")"
    top = top.drop_duplicates("label").head(10)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(top["label"][::-1], (top["market_value_eur"] / 1e6)[::-1], color="#0173B2")
    ax.set_title("Top-10 market values (sanity check)")
    ax.set_xlabel("market value (€M)")
    _caption(fig, "All genuine elite players (Haaland, Bellingham, Vinícius, Mbappé…) — no spurious high-MV rows.")
    save_fig(fig, "fig15_top10_mv")
    return fig


def fig16_minutes_vs_mv(panel: pd.DataFrame):
    df = panel.dropna(subset=["market_value_eur", "minutes_played"])
    susp = df[(df.minutes_played < 100) & (df.market_value_eur > 10e6)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["minutes_played"], df["market_value_eur"] / 1e6, s=6, alpha=0.15, color="#999")
    ax.scatter(susp["minutes_played"], susp["market_value_eur"] / 1e6, s=30, color="#C44",
               label=f"<100 min & >€10M ({len(susp)})")
    ax.set_yscale("log")
    ax.set_title("Minutes vs market value (flagged: low-minute high-value)")
    ax.set_xlabel("minutes played"); ax.set_ylabel("market value (€M, log)"); ax.legend()
    _caption(fig, "The flagged points are explicable (injuries / loans / new signings), not data errors. Max minutes 3,724 < the 3,800 namesake cap.")
    save_fig(fig, "fig16_minutes_vs_mv")
    return fig
