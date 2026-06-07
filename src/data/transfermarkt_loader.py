"""Ingest the davidcariboo (Transfermarkt) player-scores CSVs.

Filters to our 9 leagues + 2021-2025, extracts season-end-aligned market-value snapshots
(D-16: nearest within ±45 days, vectorized via ``pd.merge_asof``), and the real inbound
top-5 transfer fees (§11.4). RAW-ish: keeps ``tm_player_id`` only — name resolution to
FBref/Kaggle is Phase 2. Loans (D-15) are NOT split here (no loan flag in the data).

Reads are sequential with ``low_memory=True`` to stay gentle on resources while the FBref
scrape runs in parallel. Does not touch soccerdata.
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import (
    MV_SNAPSHOT_WINDOW_DAYS,
    SEASON_END_DATES,
    TM_LEAGUE_COMPETITION_IDS,
    TM_POSITION_MAP,
    TOP5_LEAGUES,
)
from src.utils.io import load_lookup_csv, project_root
from src.utils.logging import get_logger

_TM_DIR = project_root() / "data" / "raw" / "transfermarkt"
_COMP_TO_LEAGUE = {cid: league for league, cid in TM_LEAGUE_COMPETITION_IDS.items()}
_OUR_COMP_IDS = set(TM_LEAGUE_COMPETITION_IDS.values())
_TOP5_COMP_IDS = {TM_LEAGUE_COMPETITION_IDS[league] for league in TOP5_LEAGUES}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(_TM_DIR / name, low_memory=True)


def load_competitions() -> pd.DataFrame:
    """Our 9 leagues with tier / UEFA coefficient / value multiplier (from data/external)."""
    comp = _read("competitions.csv")
    comp = comp[comp["competition_id"].isin(_OUR_COMP_IDS)].copy()
    comp["league_name"] = comp["competition_id"].map(_COMP_TO_LEAGUE)

    tier = load_lookup_csv("league_tier_map.csv")
    uefa = load_lookup_csv("uefa_coefficients.csv")
    mult = load_lookup_csv("league_value_multipliers.csv")
    comp["league_tier"] = comp["league_name"].map(tier)
    comp["uefa_coefficient"] = comp["league_name"].map(uefa)
    comp["league_value_multiplier"] = comp["league_name"].map(mult)

    out = comp[
        ["competition_id", "league_name", "country_name", "league_tier",
         "uefa_coefficient", "league_value_multiplier"]
    ].rename(columns={"country_name": "country"})
    return out.sort_values("league_name").reset_index(drop=True)


def load_players() -> pd.DataFrame:
    """Players whose current club is in our 9 leagues, with normalized position + WC proxy."""
    logger = get_logger(__name__)
    pl = _read("players.csv")
    n_in = len(pl)
    pl = pl[pl["current_club_domestic_competition_id"].isin(_OUR_COMP_IDS)].copy()
    pl["position_primary"] = pl["position"].map(TM_POSITION_MAP).fillna("Other")
    n_other = int((pl["position_primary"] == "Other").sum())
    logger.info(
        f"players: {n_in} -> {len(pl)} in 9 leagues; "
        f"{n_other} positions mapped to 'Other' (unmapped/Missing)"
    )
    pl["current_league_name"] = pl["current_club_domestic_competition_id"].map(_COMP_TO_LEAGUE)
    caps = pd.to_numeric(pl["international_caps"], errors="coerce").fillna(0)
    pl["has_international_caps"] = caps > 0

    keep = [
        "player_id", "name", "date_of_birth", "country_of_citizenship",
        "position", "sub_position", "position_primary", "foot", "height_in_cm",
        "market_value_in_eur", "highest_market_value_in_eur",
        "current_club_id", "current_club_name",
        "current_club_domestic_competition_id", "current_league_name",
        "contract_expiration_date", "international_caps", "international_goals",
        "current_national_team_id", "has_international_caps", "agent_name", "image_url",
    ]
    keep = [c for c in keep if c in pl.columns]
    return pl[keep].rename(columns={"player_id": "tm_player_id", "name": "player_name"})


def load_player_seasons_mv() -> pd.DataFrame:
    """Season-end-aligned MV snapshots (nearest within ±45d), vectorized with merge_asof."""
    logger = get_logger(__name__)
    pv = _read("player_valuations.csv")
    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    pv = pv[
        pv["player_club_domestic_competition_id"].isin(_OUR_COMP_IDS) & pv["date"].notna()
    ].copy()

    right = pv[
        ["player_id", "date", "market_value_in_eur", "player_club_domestic_competition_id"]
    ].sort_values("date")

    tol = pd.Timedelta(days=MV_SNAPSHOT_WINDOW_DAYS)
    frames = []
    for season, end_str in SEASON_END_DATES.items():
        end = pd.Timestamp(end_str)
        left = pd.DataFrame({"player_id": pv["player_id"].unique()})
        left["target"] = end
        left = left.sort_values("target")
        m = pd.merge_asof(
            left, right, left_on="target", right_on="date", by="player_id",
            direction="nearest", tolerance=tol,
        ).dropna(subset=["date"])
        m["season"] = season
        m["season_end_date"] = end
        m["days_from_season_end"] = (m["date"] - end).dt.days
        frames.append(m)
        logger.info(f"MV {season}: {len(m)} players with aligned snapshot (±{MV_SNAPSHOT_WINDOW_DAYS}d)")

    out = pd.concat(frames, ignore_index=True).rename(
        columns={
            "player_id": "tm_player_id",
            "date": "snapshot_date",
            "market_value_in_eur": "market_value_eur",
            "player_club_domestic_competition_id": "domestic_competition_id",
        }
    )
    return out[
        ["tm_player_id", "season", "season_end_date", "market_value_eur",
         "snapshot_date", "days_from_season_end", "domestic_competition_id"]
    ]


def load_transfers() -> pd.DataFrame:
    """Real (fee>0) inbound transfers to the top-5 in the 2024-25 / summer-2025 window."""
    tr = _read("transfers.csv")
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    # Jun-2024 .. Sep-2025 fully covers summer-2024, winter-2024/25, and summer-2025
    # windows. (Wider than the Aug-1 spec start so July — the peak transfer month — is
    # captured: this ~doubles the fee>0 count, 66 -> 121.)
    tr = tr[
        (tr["transfer_date"] >= pd.Timestamp("2024-06-01"))
        & (tr["transfer_date"] <= pd.Timestamp("2025-09-30"))
    ].copy()

    clubs = _read("clubs.csv")[["club_id", "domestic_competition_id"]]
    tr = tr.merge(
        clubs.rename(columns={"club_id": "to_club_id", "domestic_competition_id": "to_comp"}),
        on="to_club_id", how="left",
    )
    tr = tr.merge(
        clubs.rename(columns={"club_id": "from_club_id", "domestic_competition_id": "from_comp"}),
        on="from_club_id", how="left",
    )
    tr = tr[tr["to_comp"].isin(_TOP5_COMP_IDS)]
    tr["transfer_fee"] = pd.to_numeric(tr["transfer_fee"], errors="coerce")
    tr = tr[tr["transfer_fee"] > 0].copy()

    tr["from_league_name"] = tr["from_comp"].map(_COMP_TO_LEAGUE)
    tr["to_league_name"] = tr["to_comp"].map(_COMP_TO_LEAGUE)
    tr["window"] = tr["transfer_date"].dt.month.map(
        lambda mo: "winter" if mo in (1, 2) else "summer"
    )

    out = tr.rename(
        columns={
            "player_id": "tm_player_id",
            "transfer_fee": "transfer_fee_eur",
            "market_value_in_eur": "market_value_at_transfer",
        }
    )
    cols = [
        "tm_player_id", "transfer_date", "transfer_season", "from_club_name",
        "to_club_name", "from_league_name", "to_league_name", "transfer_fee_eur",
        "market_value_at_transfer", "window",
    ]
    return out[[c for c in cols if c in out.columns]]


def load_national_teams() -> pd.DataFrame:
    """Team-level national-team metadata (fifa_ranking etc.). Per-player WC proxy is in load_players."""
    nt = _read("national_teams.csv")
    keep = [
        c for c in [
            "national_team_id", "name", "country_name", "confederation",
            "fifa_ranking", "squad_size", "average_age", "total_market_value", "coach_name",
        ] if c in nt.columns
    ]
    return nt[keep]
