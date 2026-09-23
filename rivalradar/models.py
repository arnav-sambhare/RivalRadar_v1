"""Core data models shared across signal sources, storage, and reporting.

These are plain dataclasses — deliberately light. Each signal source produces
these; ``db`` persists them; ``diff`` compares them week over week.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class Rival:
    """A competitor organisation being tracked."""

    name: str
    # Stable external identifiers, when known. Any may be empty.
    openalex_institution_id: str = ""
    homepage: str = ""
    # How this rival entered the set: "seed" or "discovery".
    source: str = "seed"
    # Patent count observed for the small-startup filter (None = unknown yet).
    patent_count: int | None = None
    # OpenAlex works count, a second size signal (None = unknown / no id).
    works_count: int | None = None

    @property
    def key(self) -> str:
        """Stable identity for dedup/diffing."""
        return self.openalex_institution_id or self.name.strip().lower()


@dataclass
class Publication:
    """A paper attributed to a rival (arXiv / OpenAlex)."""

    title: str
    rival_key: str
    source: str  # "arxiv" | "openalex"
    external_id: str  # arXiv id or OpenAlex work id
    url: str = ""
    published: date | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}:{self.external_id}"


@dataclass
class Patent:
    """A patent/application attributed to a rival (EPO OPS)."""

    title: str
    rival_key: str
    external_id: str  # publication number
    url: str = ""
    published: date | None = None
    cpc_codes: list[str] = field(default_factory=list)
    applicants: list[str] = field(default_factory=list)
    # Inventor names, used to tell an inventor filing in person from a company.
    inventors: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"epo:{self.external_id}"


@dataclass
class Affiliation:
    """A researcher's affiliation with one organisation (OpenAlex).

    ``org_key`` is the author's real institution (OpenAlex id), not the rival
    being queried. An author tied to a rival now, with other institutions in
    earlier years, is the hiring signal — a researcher joining a rival.
    ``first_year``/``last_year`` come from OpenAlex's affiliation history.
    """

    author_id: str
    author_name: str
    org_key: str
    org_name: str
    observed: date | None = None
    first_year: int | None = None
    last_year: int | None = None

    @property
    def key(self) -> str:
        return f"{self.author_id}@{self.org_key}"


@dataclass
class SignalBundle:
    """Everything one run collected, before diffing and ranking."""

    rivals: list[Rival] = field(default_factory=list)
    publications: list[Publication] = field(default_factory=list)
    patents: list[Patent] = field(default_factory=list)
    affiliations: list[Affiliation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for JSON snapshots."""
        return {
            "rivals": [_asdict(r) for r in self.rivals],
            "publications": [_asdict(p) for p in self.publications],
            "patents": [_asdict(p) for p in self.patents],
            "affiliations": [_asdict(a) for a in self.affiliations],
        }


def parse_iso_date(value: str | None) -> date | None:
    """ISO date string -> date; None for blank or malformed input."""
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _asdict(obj: Any) -> dict[str, Any]:
    """dataclass -> dict with dates rendered ISO for JSON."""
    from dataclasses import asdict as _dc_asdict

    out = _dc_asdict(obj)
    for k, v in list(out.items()):
        if isinstance(v, date):
            out[k] = v.isoformat()
    return out
