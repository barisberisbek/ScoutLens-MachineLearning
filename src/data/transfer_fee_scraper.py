"""Custom Transfermarkt scraper for disclosed 2024-25 transfer fees (top-5 inbound).

davidcariboo's transfer fees are sparse (marquee deals undisclosed). This scrapes TM's
public per-league transfer-summary pages — which list every club's Arrivals with a Fee
column — to recover the big disclosed fees for the §11.4 "absolute-truth" validation (D-09).

Polite: browser UA, 3-5s random delay, retry-with-backoff, and an on-disk HTML cache so
re-parsing never re-hits the network. Targets transfermarkt.com only (NOT soccerdata).
Name resolution to other sources is Phase 2 — we keep player_name + clubs only.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.constants import TM_LEAGUE_COMPETITION_IDS
from src.utils.io import project_root
from src.utils.logging import get_logger

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
}
_TM_SLUGS = {
    "GB1": "premier-league", "ES1": "laliga", "L1": "bundesliga",
    "IT1": "serie-a", "FR1": "ligue-1",
}
_COMP_TO_LEAGUE = {cid: league for league, cid in TM_LEAGUE_COMPETITION_IDS.items()}
_CACHE_DIR = project_root() / "data" / "raw" / "transfer_fees_2025" / ".html_cache"


def _fetch_page(league_id: str, season_id: int, force: bool, logger) -> str:
    """Return page HTML, using the on-disk cache unless ``force``. Retries with backoff."""
    cache_path = _CACHE_DIR / f"{league_id}_{season_id}.html"
    if cache_path.exists() and not force:
        logger.info(f"  cache hit: {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")

    slug = _TM_SLUGS[league_id]
    url = f"https://www.transfermarkt.com/{slug}/transfers/wettbewerb/{league_id}?saison_id={season_id}"
    last_err = None
    for attempt in range(1, 4):
        time.sleep(random.uniform(3.0, 5.0))
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code == 200 and "just a moment" not in resp.text.lower():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(resp.text, encoding="utf-8")
                return resp.text
            last_err = f"status={resp.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        logger.warning(f"  attempt {attempt}/3 failed ({last_err}); backing off")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _parse_fee(text: str) -> tuple[float | None, str]:
    """Parse a TM fee cell → (fee_eur or None, transfer_type)."""
    t = (text or "").replace("\xa0", " ").strip().lower().replace("€", "").strip()
    if not t or t in ("?", "-"):
        return None, "undisclosed"
    if "free" in t:
        return None, "free"
    if "end of loan" in t:
        return None, "end_of_loan"
    if "loan" in t:
        return None, "loan"
    m = re.match(r"([\d.,]+)\s*(bn|m|k)?", t)
    if m:
        num = float(m.group(1).replace(",", ""))
        mult = {"k": 1e3, "m": 1e6, "bn": 1e9}.get(m.group(2), 1.0)
        return num * mult, "paid"
    return None, "undisclosed"


def _cell(tr, css: str) -> str:
    c = tr.select_one(css)
    return c.get_text(" ", strip=True) if c else ""


def _parse_arrivals(html: str, to_league: str) -> list[dict]:
    """Parse every club's Arrivals table (inbound transfers) from a league page."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for box in soup.select("div.box"):
        h2 = box.select_one("h2")
        if not h2:
            continue
        club = h2.get_text(strip=True)
        if not club or club == "Transfer record" or "season" in club.lower():
            continue
        for tbl in box.select("div.responsive-table table"):
            hdr = [th.get_text(strip=True) for th in tbl.select("thead th")]
            if not hdr or hdr[0] != "In":  # arrivals tables only
                continue
            for tr in tbl.select("tbody tr"):
                player_a = tr.select_one("td a[href*='/profil/spieler/']")
                if player_a is None:  # skip "no arrivals" placeholder rows
                    continue
                fee_cell = tr.find_all("td", recursive=False)[-1]
                fee_eur, ttype = _parse_fee(fee_cell.get_text(" ", strip=True))
                mv_eur, _ = _parse_fee(_cell(tr, "td.mw-transfer-cell"))
                rows.append({
                    "player_name": player_a.get_text(strip=True),
                    "player_age": _cell(tr, "td.alter-transfer-cell"),
                    "player_position": _cell(tr, "td.pos-transfer-cell"),
                    "from_club": _cell(tr, "td.verein-flagge-transfer-cell"),
                    "from_league": None,  # only club name on page; Phase 2 resolves
                    "to_club": club,
                    "to_league": to_league,
                    "transfer_fee_eur": fee_eur,
                    "transfer_type": ttype,
                    "market_value_eur": mv_eur,
                    "transfer_date": None,   # not present on this view
                    "transfer_window": None,
                })
    return rows


def scrape_league_transfers(league_id: str, season_id: int = 2024, force: bool = False) -> pd.DataFrame:
    """Scrape all inbound transfers for one top-5 league × season."""
    logger = get_logger(__name__)
    to_league = _COMP_TO_LEAGUE[league_id]
    logger.info(f"FETCH transfers: {to_league} ({league_id}) saison {season_id}")
    html = _fetch_page(league_id, season_id, force, logger)
    rows = _parse_arrivals(html, to_league)
    logger.info(f"  parsed {len(rows)} arrivals for {to_league}")
    return pd.DataFrame(rows)


def scrape_all_top5_transfers(season_id: int = 2024, force: bool = False) -> pd.DataFrame:
    """Scrape inbound transfers for all 5 top leagues and concatenate."""
    frames = [
        scrape_league_transfers(lid, season_id, force)
        for lid in ["GB1", "ES1", "L1", "IT1", "FR1"]
    ]
    return pd.concat(frames, ignore_index=True)
