"""Cross-validate FBref-via-soccerdata vs the Kaggle dataset for 2024-25 top-5.

Sanity check that the two FBref-derived sources agree on the stats they share (goals,
assists, minutes, shots) — xG is excluded (soccerdata's FBref tables don't carry it).
Matches players by normalized name within a league (aggregating mid-season transfers).
Writes reports/fbref_kaggle_cross_validation.md.

    python scripts/validate_fbref_kaggle_cross.py
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.data.understat_loader import _league_slug
from src.utils.io import load_parquet, project_root

_LEAGUES = ["Premier League", "La Liga"]
_SEASON = "2425"
# FBref flattened column -> (kaggle column, friendly name)
_STAT_MAP = {
    "Performance_Gls": ("goals", "goals"),
    "Performance_Ast": ("assists", "assists"),
    "Playing Time_Min": ("minutes_played", "minutes"),
}
_SHOOTING_MAP = {"Standard_Sh": ("shots", "shots")}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def _agg_by_name(df: pd.DataFrame, name_col: str, num_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["nkey"] = df[name_col].map(_norm)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby("nkey", as_index=False)[num_cols].sum()


def main() -> None:
    root = project_root()
    kaggle = load_parquet(root / "data" / "interim" / "kaggle_2024_25_clean.parquet")

    lines = ["# FBref (soccerdata) vs Kaggle — 2024-25 cross-validation", ""]
    lines.append("xG is excluded (soccerdata's FBref tables don't provide it). "
                 "Players matched by normalized name within league.\n")

    for league in _LEAGUES:
        slug = _league_slug(league)
        std_path = root / "data" / "raw" / "fbref" / slug / _SEASON / "standard.parquet"
        sh_path = root / "data" / "raw" / "fbref" / slug / _SEASON / "shooting.parquet"
        if not std_path.exists():
            lines.append(f"## {league}\n\n_standard.parquet missing — skipped._\n")
            continue

        fb = load_parquet(std_path)
        fb_cols = [c for c in _STAT_MAP if c in fb.columns]
        fb_agg = _agg_by_name(fb, "player", fb_cols).rename(
            columns={c: _STAT_MAP[c][1] for c in fb_cols}
        )
        if sh_path.exists():
            sh = load_parquet(sh_path)
            sh_cols = [c for c in _SHOOTING_MAP if c in sh.columns]
            if sh_cols:
                sh_agg = _agg_by_name(sh, "player", sh_cols).rename(
                    columns={c: _SHOOTING_MAP[c][1] for c in sh_cols}
                )
                fb_agg = fb_agg.merge(sh_agg, on="nkey", how="left")

        kl = kaggle[kaggle["league"] == league]
        k_cols = [_STAT_MAP[c][0] for c in fb_cols] + (
            ["shots"] if "shots" in fb_agg.columns else []
        )
        k_agg = _agg_by_name(kl, "player_name", k_cols)

        merged = fb_agg.merge(k_agg, on="nkey", how="inner", suffixes=("_fb", "_kg"))
        n_fb, n_kg, n_m = len(fb_agg), len(k_agg), len(merged)

        lines.append(f"## {league}")
        lines.append("")
        lines.append(f"- FBref players: {n_fb} | Kaggle players: {n_kg} | "
                     f"matched: {n_m} ({100 * n_m / max(n_fb, 1):.1f}% of FBref)")
        lines.append("")
        lines.append("| stat | Pearson r | MAE | mean FBref | mean Kaggle |")
        lines.append("|---|---|---|---|---|")
        for friendly, kcol in [("goals", "goals"), ("assists", "assists"),
                               ("minutes", "minutes_played"), ("shots", "shots")]:
            fb_c = friendly
            if fb_c in merged.columns and kcol in merged.columns:
                a, b = merged[fb_c], merged[kcol]
                mask = a.notna() & b.notna()
                if mask.sum() > 5:
                    r = a[mask].corr(b[mask])
                    mae = (a[mask] - b[mask]).abs().mean()
                    lines.append(f"| {friendly} | {r:.3f} | {mae:.2f} | "
                                 f"{a[mask].mean():.2f} | {b[mask].mean():.2f} |")
        lines.append("")

    out_path = root / "reports" / "fbref_kaggle_cross_validation.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
