"""Mandatory tests for player identity resolution (CLAUDE.md / roadmap §5.4).

A bug in the resolver silently corrupts every downstream join, so the cascade is
pinned here: exact, accent-folded fuzzy, short-name + team disambiguation,
birth-year disambiguation of namesakes, manual-override precedence, below-threshold
unmatched, the nationality crosswalk, deterministic synthetic ids, and club folding.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.name_resolver import (
    PlayerIDResolver,
    normalize_club,
    normalize_name,
    synthetic_player_id,
)


@pytest.fixture(scope="module")
def tm_master() -> pd.DataFrame:
    """A tiny Transfermarkt master covering every cascade branch under test."""
    rows = [
        # id, name, dob, country, current club
        ("1", "Erling Haaland", "2000-07-21", "Norway", "Manchester City"),
        ("2", "Lionel Messi", "1987-06-24", "Argentina", "Inter Miami"),
        ("3", "Lautaro Martínez", "1997-08-22", "Argentina", "Inter"),
        ("6", "Lisandro Martínez", "1998-01-18", "Argentina", "Manchester United"),
        ("4", "Danilo", "1991-07-15", "Brazil", "Juventus"),          # namesake A
        ("5", "Danilo", "2001-04-29", "Brazil", "Nottingham Forest"),  # namesake B
        ("8", "Bruno Fernandes", "1994-09-08", "Portugal", "Manchester United"),
        ("9", "Sergi Roberto", "1992-02-07", "Spain", "Barcelona"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "tm_player_id",
            "player_name",
            "date_of_birth",
            "country_of_citizenship",
            "current_club_name",
        ],
    )


def make_resolver(tm_master, overrides: pd.DataFrame | None = None) -> PlayerIDResolver:
    """Build a resolver against the fixture master + the real committed crosswalks."""
    return PlayerIDResolver(tm_master, manual_overrides=overrides)


@pytest.fixture(scope="module")
def resolver(tm_master) -> PlayerIDResolver:
    return make_resolver(tm_master)


# ── 1. exact composite ───────────────────────────────────────────────────────
def test_exact_composite_match(resolver):
    res = resolver.resolve("Erling Haaland", birth_year=2000, nationality="NOR", source="fbref")
    assert res.player_id == "1"
    assert res.method == "exact"
    assert res.score == 100.0


# ── 2. accent-folded fuzzy (dropped middle name) ─────────────────────────────
def test_accent_and_middle_name_fuzzy(resolver):
    res = resolver.resolve("Lionel Andrés Messi", birth_year=1987, nationality="ARG", source="fbref")
    assert res.player_id == "2"
    assert res.method == "fuzzy_nat"
    assert res.score >= 85.0


# ── 3. short-name + team disambiguation (Understat-style, no birth/nat) ───────
def test_short_name_team_disambiguation(resolver):
    # Two Argentine "L. Martínez" namesakes; team picks Lautaro (Inter), not Lisandro.
    res = resolver.resolve("L. Martínez", team="Inter", season="2023-24", source="understat")
    assert res.player_id == "3"
    assert res.method == "fuzzy_team"


# ── 4. birth-year disambiguation of namesakes ────────────────────────────────
def test_birth_year_disambiguates_namesakes(resolver):
    young = resolver.resolve("Danilo", birth_year=2001, nationality="BRA", source="fbref")
    old = resolver.resolve("Danilo", birth_year=1991, nationality="BRA", source="fbref")
    assert young.player_id == "5"
    assert old.player_id == "4"
    assert young.method == "exact" and old.method == "exact"


def test_ambiguous_namesake_without_disambiguator_is_unmatched(resolver):
    # Wrong birth year + no team → two same-name Brazilians remain; do NOT guess.
    res = resolver.resolve("Danilo", birth_year=1985, nationality="BRA", source="fbref")
    assert res.player_id is None
    assert res.method == "unmatched"


# ── 5. manual-override precedence ────────────────────────────────────────────
def test_manual_override_beats_exact(tm_master):
    overrides = pd.DataFrame(
        [{"variant_name": "Bruno Fernandes", "source": "fbref", "target_player_id": "99999"}]
    )
    r = make_resolver(tm_master, overrides=overrides)
    res = r.resolve("Bruno Fernandes", birth_year=1994, nationality="POR", source="fbref")
    assert res.player_id == "99999"  # not "8"
    assert res.method == "manual"


def test_override_is_source_scoped(tm_master):
    overrides = pd.DataFrame(
        [{"variant_name": "Bruno Fernandes", "source": "fbref", "target_player_id": "99999"}]
    )
    r = make_resolver(tm_master, overrides=overrides)
    # Same name from a DIFFERENT source must not pick up the fbref override.
    res = r.resolve("Bruno Fernandes", birth_year=1994, nationality="POR", source="kaggle")
    assert res.player_id == "8"
    assert res.method == "exact"


# ── 6. below-threshold / no-candidate unmatched ──────────────────────────────
def test_unmatched_when_no_candidate(resolver):
    res = resolver.resolve("Random Unknown Person", birth_year=1995, nationality="FRA", source="fbref")
    assert res.player_id is None
    assert res.method == "unmatched"


# ── 7. nationality crosswalk ─────────────────────────────────────────────────
def test_nationality_crosswalk(resolver):
    assert resolver.normalize_nationality("ENG") == "England"
    assert resolver.normalize_nationality("ALG") == "Algeria"
    assert resolver.normalize_nationality("England") == "England"        # idempotent
    assert resolver.normalize_nationality("Cote d'Ivoire") == "Côte d'Ivoire"  # accent variant
    assert resolver.normalize_nationality("Bosnia-Herzegovina") == "Bosnia and Herzegovina"
    assert resolver.normalize_nationality("ZZZ") is None


def test_crosswalk_integrity_enforced_at_build(tm_master):
    # Construction asserts every nationality_map value is a continent_map key.
    bad_nat = {"ENG": "Nowhereland"}
    with pytest.raises(ValueError):
        PlayerIDResolver(tm_master, nationality_map=bad_nat)


# ── 8. deterministic synthetic id ────────────────────────────────────────────
def test_synthetic_id_is_deterministic():
    a = synthetic_player_id("joao felix", 1999, "POR")
    b = synthetic_player_id("joao felix", 1999, "POR")
    c = synthetic_player_id("joao felix", 2000, "POR")
    assert a == b
    assert a.startswith("synthetic_")
    assert a != c


# ── 9. club + name normalization ─────────────────────────────────────────────
def test_club_normalization():
    assert normalize_club("Manchester Utd") == normalize_club("Manchester United")
    assert normalize_club("Inter") == normalize_club("Internazionale")
    assert normalize_club("AFC Bournemouth") == normalize_club("Bournemouth")


def test_name_normalization():
    assert normalize_name("Lionel Andrés Messi") == "lionel andres messi"
    assert normalize_name("L. Messi") == "l messi"
    assert normalize_name("O'Riley") == "oriley"
    assert normalize_name("Ødegaard") == "odegaard"
    assert normalize_name("Müller") == "muller"
