"""Player identity resolution across data sources (Phase 2, roadmap §5.4).

The same player is spelled differently in every source and there is no shared id,
so this module resolves an incoming ``(name, birth_year, nationality, team)`` tuple
to a stable Transfermarkt ``player_id`` (the identity anchor). The cascade is:

    1. manual override (data/manual/manual_id_overrides.csv) — always wins
    2. exact composite (normalized_name, birth_year, nationality)
    3. exact (normalized_name, birth_year)  — nationality-agnostic safety net,
       resilient to nationality spelling drift between sources
    4. fuzzy within (birth_year, nationality) pool, rapidfuzz token_set_ratio >= 85
       (with an initials pre-check for "L. Messi"-style abbreviations)
    5. team/club disambiguation among same-name candidates (for sources lacking a
       birth year / nationality, e.g. Understat)
    6. unmatched  → caller mints a deterministic synthetic id

Every decision is appended to an audit buffer (-> data/manual/match_log.csv). No
silent drops: an unmatched row returns ``player_id=None`` with its candidates, never
a silent wrong match.

Public API: ``normalize_name``, ``normalize_nationality`` (method), ``normalize_club``,
``synthetic_player_id``, ``ResolutionResult``, ``PlayerIDResolver``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz, process

from src.utils.constants import (
    CLUB_ALIAS_MAP,
    CLUB_STOPWORDS,
    NATIONALITY_NAME_ALIAS,
)
from src.utils.io import load_lookup_csv
from src.utils.logging import get_logger

# Characters NFKD does not decompose but which must fold for cross-source matching.
_ACCENT_PREMAP = str.maketrans(
    {
        "ø": "o", "Ø": "o", "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
        "đ": "d", "Đ": "d", "ð": "d", "Ð": "d", "ł": "l", "Ł": "l", "ı": "i",
        "þ": "th", "Þ": "th", "ħ": "h",
    }
)


def normalize_name(raw: str | None) -> str:
    """Fold a player name to a comparison key.

    lowercase → premap special letters → NFKD accent-strip → drop punctuation
    (``.`` becomes a space so an initial survives as its own token; apostrophes are
    deleted so ``O'Riley`` → ``oriley``) → collapse whitespace.

    >>> normalize_name("Lionel Andrés Messi")
    'lionel andres messi'
    >>> normalize_name("L. Messi")
    'l messi'
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    s = s.translate(_ACCENT_PREMAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = s.replace("'", "").replace("’", "")  # apostrophes deleted, not spaced
    s = s.replace(".", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_tokens(norm: str) -> frozenset[str]:
    """Token set of an already-normalized name."""
    return frozenset(norm.split())


def initials_form(norm: str) -> str:
    """Collapse the first token to its initial: ``'lionel messi'`` → ``'l messi'``.

    Lets ``"L. Messi"`` (normalizes to ``"l messi"``) match ``"Lionel Messi"``
    deterministically before falling back to fuzzy scoring.
    """
    toks = norm.split()
    if not toks:
        return ""
    if len(toks) == 1:
        return toks[0][0]
    return toks[0][0] + " " + " ".join(toks[1:])


def _norm_country(s: str) -> str:
    """Aggressively fold a country string for accent/punctuation-insensitive lookup."""
    s = str(s).translate(_ACCENT_PREMAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    return re.sub(r"[^a-z0-9]", "", s)


def normalize_club(raw: str | None) -> str:
    """Fold a club name for the team-disambiguation path.

    Same letter pipeline as :func:`normalize_name`, then token aliases
    (``utd`` → ``united``) and structural stopwords (``fc``/``afc``/…) are applied.

    >>> normalize_club("Manchester Utd") == normalize_club("Manchester United")
    True
    """
    base = normalize_name(raw)
    if not base:
        return ""
    toks = [CLUB_ALIAS_MAP.get(t, t) for t in base.split()]
    toks = [t for t in toks if t not in CLUB_STOPWORDS]
    return " ".join(toks).strip()


def clubs_match(a: str | None, b: str | None, threshold: float = 90.0) -> bool:
    """True if two club names refer to the same club (normalized-equal or ratio>=90)."""
    na, nb = normalize_club(a), normalize_club(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return fuzz.ratio(na, nb) >= threshold


def synthetic_player_id(norm_name: str, birth_year, nationality_code: str | None) -> str:
    """Deterministic id for players with no Transfermarkt match.

    Stable across runs (sha256 of the identity tuple), so the same player gets the
    same id every build. Uses the raw nationality CODE (always present for FBref),
    not the resolved full name, to avoid coupling to the crosswalk.
    """
    by = "" if birth_year is None or pd.isna(birth_year) else int(birth_year)
    code = "" if nationality_code is None or pd.isna(nationality_code) else str(nationality_code)
    key = f"{norm_name}|{by}|{code}"
    return "synthetic_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of a single resolution attempt."""

    player_id: str | None
    method: str  # manual | exact | exact_name_year | fuzzy_nat | fuzzy_team | unmatched
    score: float
    candidates: list[tuple[str, float]] = field(default_factory=list)


class PlayerIDResolver:
    """Resolve source rows to Transfermarkt ``player_id`` via the §5.4 cascade.

    Built once from ``tm_players``; ``resolve``/``resolve_frame`` then map incoming
    rows. Every call appends to an in-memory audit buffer exposed via
    :meth:`audit_frame`.
    """

    def __init__(
        self,
        tm_players: pd.DataFrame,
        *,
        nationality_map: dict | None = None,
        continent_map: dict | None = None,
        name_alias: dict | None = None,
        manual_overrides: pd.DataFrame | None = None,
        fuzzy_threshold: float = 85.0,
        club_threshold: float = 90.0,
        logger=None,
    ) -> None:
        self.log = logger or get_logger(__name__)
        self.fuzzy_threshold = fuzzy_threshold
        self.club_threshold = club_threshold

        # ── Nationality crosswalk ──────────────────────────────────────────
        self._code_map = nationality_map or load_lookup_csv("nationality_map")
        cont = continent_map or load_lookup_csv("continent_map")
        self._country_set = set(cont.keys())
        self._norm_country_index = {_norm_country(c): c for c in self._country_set}
        self._alias = name_alias if name_alias is not None else dict(NATIONALITY_NAME_ALIAS)
        # Integrity guard: every crosswalk target must be a continent_map key.
        orphans = sorted(set(self._code_map.values()) - self._country_set)
        if orphans:
            raise ValueError(
                f"nationality_map has {len(orphans)} target(s) absent from continent_map: "
                f"{orphans[:10]}"
            )

        # ── Manual overrides keyed (source, variant) on raw AND normalized name ─
        self._override: dict[tuple[str, str], str] = {}
        if manual_overrides is not None and len(manual_overrides):
            self._load_overrides(manual_overrides)

        # ── Master index from tm_players ───────────────────────────────────
        self._ids: list[str] = []
        self._names: list[str] = []
        self._years: list[int | None] = []
        self._nats: list[str | None] = []
        self._clubs: list[str] = []
        self._initials: list[str] = []
        self._exact_full: dict[tuple[str, int, str], list[int]] = {}
        self._exact_name_year: dict[tuple[str, int], list[int]] = {}
        self._pool_year_nat: dict[tuple[int, str], list[int]] = {}
        self._pool_nat: dict[str, list[int]] = {}
        self._pool_year: dict[int, list[int]] = {}
        self._build_master(tm_players)

        self._audit: list[dict] = []

    # ── construction helpers ───────────────────────────────────────────────
    def _load_overrides(self, df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            src = str(getattr(row, "source", "") or "")
            variant = str(getattr(row, "variant_name", "") or "")
            target = getattr(row, "target_player_id", None)
            if not variant or target is None or pd.isna(target):
                continue
            target = str(target)
            self._override[(src, variant)] = target
            self._override[(src, normalize_name(variant))] = target
        self.log.info("Loaded %d manual override key(s)", len(self._override))

    def _build_master(self, tm: pd.DataFrame) -> None:
        years = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.year
        for pos, row in enumerate(tm.itertuples(index=False)):
            pid = str(getattr(row, "tm_player_id"))
            norm = normalize_name(getattr(row, "player_name", None))
            if not norm:
                continue
            by = years.iloc[pos]
            by = None if pd.isna(by) else int(by)
            nat = self.normalize_nationality(getattr(row, "country_of_citizenship", None))
            club = normalize_club(getattr(row, "current_club_name", None))

            idx = len(self._ids)
            self._ids.append(pid)
            self._names.append(norm)
            self._years.append(by)
            self._nats.append(nat)
            self._clubs.append(club)
            self._initials.append(initials_form(norm))

            if by is not None and nat is not None:
                self._exact_full.setdefault((norm, by, nat), []).append(idx)
            if by is not None:
                self._exact_name_year.setdefault((norm, by), []).append(idx)
                self._pool_year.setdefault(by, []).append(idx)
                if nat is not None:
                    self._pool_year_nat.setdefault((by, nat), []).append(idx)
            if nat is not None:
                self._pool_nat.setdefault(nat, []).append(idx)
        self.log.info("Master index built: %d Transfermarkt players", len(self._ids))

    # ── nationality ─────────────────────────────────────────────────────────
    def normalize_nationality(self, value: str | None) -> str | None:
        """IOC code OR full name → canonical continent_map spelling, else ``None``."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        s = str(value).strip()
        if not s:
            return None
        if s in self._code_map:           # IOC/FBref code (e.g. "ENG")
            return self._code_map[s]
        if s in self._country_set:        # already canonical
            return s
        if s in self._alias:              # known reordering / extra-word variant
            return self._alias[s]
        canon = self._norm_country_index.get(_norm_country(s))  # accent/punct variant
        return canon  # None if unknown (logged by caller via audit)

    # ── core resolution ──────────────────────────────────────────────────────
    def resolve(
        self,
        name: str,
        *,
        birth_year=None,
        nationality: str | None = None,
        team: str | None = None,
        season: str | None = None,
        source: str = "",
    ) -> ResolutionResult:
        """Resolve one row to a ``player_id`` via the §5.4 cascade (audited)."""
        result = self._resolve_uncached(
            name, birth_year=birth_year, nationality=nationality, team=team, source=source
        )
        self._audit.append(
            {
                "source": source,
                "input_name": name,
                "input_norm_name": normalize_name(name),
                "input_birth_year": None if birth_year is None or pd.isna(birth_year) else int(birth_year),
                "input_nationality": nationality,
                "input_team": team,
                "input_season": season,
                "resolved_player_id": result.player_id,
                "method": result.method,
                "score": round(result.score, 2),
                "candidate_ids": ";".join(c[0] for c in result.candidates),
                "candidate_scores": ";".join(f"{c[1]:.1f}" for c in result.candidates),
                "n_candidates": len(result.candidates),
            }
        )
        return result

    def _resolve_uncached(self, name, *, birth_year, nationality, team, source) -> ResolutionResult:
        # 1) manual override (raw or normalized variant), highest precedence
        norm = normalize_name(name)
        for key in ((source, str(name)), (source, norm)):
            if key in self._override:
                return ResolutionResult(self._override[key], "manual", 100.0)

        if not norm:
            return ResolutionResult(None, "unmatched", 0.0)

        by = None if birth_year is None or pd.isna(birth_year) else int(birth_year)
        nat = self.normalize_nationality(nationality)
        norm_team = normalize_club(team) if team else ""

        # 2) exact composite (name, birth_year, nationality)
        if by is not None and nat is not None:
            ids = self._exact_full.get((norm, by, nat))
            if ids and len(ids) == 1:
                return ResolutionResult(self._ids[ids[0]], "exact", 100.0)
            if ids and len(ids) > 1:
                hit = self._disambiguate(ids, norm_team)
                if hit is not None:
                    return ResolutionResult(self._ids[hit], "exact", 100.0)

        # 3) exact (name, birth_year) — nationality-agnostic safety net
        if by is not None:
            ids = self._exact_name_year.get((norm, by))
            if ids and len(ids) == 1:
                return ResolutionResult(self._ids[ids[0]], "exact_name_year", 99.0)
            if ids and len(ids) > 1:
                narrowed = [i for i in ids if nat is None or self._nats[i] == nat]
                if len(narrowed) == 1:
                    return ResolutionResult(self._ids[narrowed[0]], "exact_name_year", 99.0)
                hit = self._disambiguate(narrowed or ids, norm_team)
                if hit is not None:
                    return ResolutionResult(self._ids[hit], "exact_name_year", 99.0)

        # 4) fuzzy within the tightest available pool
        pool = self._fuzzy_pool(by, nat)
        if not pool:
            return ResolutionResult(None, "unmatched", 0.0)

        # 4a) initials pre-check (handles "L. Messi")
        q_initials = initials_form(norm)
        init_hits = [i for i in pool if self._initials[i] == q_initials]
        if len(init_hits) == 1:
            return ResolutionResult(self._ids[init_hits[0]], self._fuzzy_method(nat), 90.0)
        if len(init_hits) > 1 and norm_team:
            hit = self._disambiguate(init_hits, norm_team)
            if hit is not None:
                return ResolutionResult(self._ids[hit], "fuzzy_team", 90.0)

        # 4b) token_set_ratio over the pool
        pool_names = [self._names[i] for i in pool]
        matches = process.extract(
            norm, pool_names, scorer=fuzz.token_set_ratio, limit=8,
            score_cutoff=self.fuzzy_threshold,
        )
        if not matches:
            return ResolutionResult(None, "unmatched", 0.0)

        best_name, best_score, _ = matches[0]
        # positions in the pool sharing the best matched name (namesakes)
        same = [i for i in pool if self._names[i] == best_name]
        candidates = [(self._ids[s], best_score) for s in same][:8]
        if len(same) == 1:
            return ResolutionResult(self._ids[same[0]], self._fuzzy_method(nat), best_score, candidates)
        # multiple namesakes → need team to disambiguate
        if norm_team:
            hit = self._disambiguate(same, norm_team)
            if hit is not None:
                return ResolutionResult(self._ids[hit], "fuzzy_team", best_score, candidates)
        # ambiguous: do not guess
        return ResolutionResult(None, "unmatched", best_score, candidates)

    # ── small helpers ────────────────────────────────────────────────────────
    def _fuzzy_method(self, nat: str | None) -> str:
        return "fuzzy_nat" if nat is not None else "fuzzy"

    def _fuzzy_pool(self, by: int | None, nat: str | None) -> list[int]:
        """Tightest non-empty candidate pool: (year,nat) → nat → year → all."""
        if by is not None and nat is not None:
            pool = self._pool_year_nat.get((by, nat))
            if pool:
                return pool
        if nat is not None:
            pool = self._pool_nat.get(nat)
            if pool:
                return pool
        if by is not None:
            pool = self._pool_year.get(by)
            if pool:
                return pool
        # last resort only when nothing narrows (avoids 21k-wide false positives)
        if by is None and nat is None:
            return list(range(len(self._ids)))
        return []

    def _disambiguate(self, idxs: list[int], norm_team: str) -> int | None:
        """Pick the single candidate whose club matches ``norm_team``; else None."""
        if not norm_team:
            return None
        hits = [i for i in idxs if self._clubs[i] and clubs_match(self._clubs[i], norm_team, self.club_threshold)]
        return hits[0] if len(hits) == 1 else None

    # ── batch + audit ────────────────────────────────────────────────────────
    def resolve_frame(
        self,
        df: pd.DataFrame,
        *,
        name_col: str,
        birth_year_col: str | None = None,
        nationality_col: str | None = None,
        team_col: str | None = None,
        season_col: str | None = None,
        source: str,
    ) -> pd.DataFrame:
        """Resolve every row of ``df``; return a copy with id/method/score columns."""
        out = df.copy()
        ids: list[str | None] = []
        methods: list[str] = []
        scores: list[float] = []
        for row in df.itertuples(index=False):
            res = self.resolve(
                getattr(row, name_col),
                birth_year=getattr(row, birth_year_col) if birth_year_col else None,
                nationality=getattr(row, nationality_col) if nationality_col else None,
                team=getattr(row, team_col) if team_col else None,
                season=getattr(row, season_col) if season_col else None,
                source=source,
            )
            ids.append(res.player_id)
            methods.append(res.method)
            scores.append(res.score)
        out["player_id"] = ids
        out["resolve_method"] = methods
        out["resolve_score"] = scores
        return out

    def audit_frame(self) -> pd.DataFrame:
        """All resolution decisions so far → DataFrame for data/manual/match_log.csv."""
        return pd.DataFrame(self._audit)
