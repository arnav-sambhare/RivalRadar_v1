"""Raw weekly diff: what changed since last week's snapshot.

Compares this run's ``SignalBundle`` with the previous snapshot and returns
only what is new, flattened into ``ChangeItem``s for the relevance pass:

  * new_rival    — a rival that joined the tracked set (e.g. via discovery)
  * publication  — a paper not seen last week (arXiv/OpenAlex duplicates of
                   the same paper are merged by title)
  * patent       — a patent not seen last week
  * hire         — a researcher who recently joined a rival from another org
  * role         — an open role posted on a rival's careers page (optional)

Hires are inferred from OpenAlex affiliation histories, which are noisy (it
sometimes merges different people who share a name). They are candidates for
the relevance pass to judge, not facts; prior orgs are included so it can.

With no previous snapshot (first run) everything counts as new — except the
seed rivals, which the founder supplied — and ``WeeklyDiff.baseline`` is False,
so the brief can say so. A rival's first week of data is its baseline, not
news: only rivals tracked last week (on a first run, the seeds) contribute
changes; a newly tracked rival is reported once, as ``new_rival``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .models import Affiliation, Patent, Publication, Rival, SignalBundle

ROLE_PREFIX = "role:"


@dataclass
class Hire:
    """A researcher who recently joined a rival from another organisation."""

    author_id: str
    author_name: str
    rival_key: str
    rival_name: str
    joined_year: int
    # (org name, first year, last year), most recent first.
    prior_orgs: list[tuple[str, int | None, int | None]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.author_id}@{self.rival_key}"


@dataclass
class ChangeItem:
    """One thing that changed, in a uniform shape for ranking and reporting."""

    id: str
    kind: str  # new_rival | publication | patent | hire | role
    rival_key: str
    rival_name: str
    title: str
    detail: str = ""
    when: date | None = None
    url: str = ""


@dataclass
class WeeklyDiff:
    """Everything new this run, plus whether a baseline existed."""

    baseline: bool
    new_rivals: list[Rival] = field(default_factory=list)
    publications: list[Publication] = field(default_factory=list)
    patents: list[Patent] = field(default_factory=list)
    hires: list[Hire] = field(default_factory=list)
    roles: list[Affiliation] = field(default_factory=list)
    items: list[ChangeItem] = field(default_factory=list)


# --- helpers ---------------------------------------------------------------


def _title_key(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def _rival_names(bundle: SignalBundle) -> dict[str, str]:
    return {r.key: r.name for r in bundle.rivals}


def dedup_publications(pubs: list[Publication]) -> list[Publication]:
    """Merge the same paper seen via several sources (matched by title).

    Keeps the first copy and fills gaps (date, URL, authors) from later ones.
    """
    by_title: dict[str, Publication] = {}
    for pub in pubs:
        key = _title_key(pub.title)
        if not key:
            continue
        kept = by_title.get(key)
        if kept is None:
            by_title[key] = pub
            continue
        kept.published = kept.published or pub.published
        kept.url = kept.url or pub.url
        kept.authors = kept.authors or pub.authors
        kept.abstract = kept.abstract or pub.abstract
    return list(by_title.values())


def find_hires(bundle: SignalBundle, today: date, max_join_age_years: int = 1) -> list[Hire]:
    """Researchers who joined a rival recently, having been elsewhere before.

    An author counts if their first year at a rival is within
    ``max_join_age_years`` of ``today`` and they have an affiliation with
    another org that started before that year.
    """
    rival_by_org = {r.openalex_institution_id: r for r in bundle.rivals if r.openalex_institution_id}
    by_author: dict[str, list[Affiliation]] = {}
    for aff in bundle.affiliations:
        if aff.author_id.startswith(ROLE_PREFIX):
            continue
        by_author.setdefault(aff.author_id, []).append(aff)

    hires: list[Hire] = []
    for author_id, rows in by_author.items():
        for rival_org in {a.org_key for a in rows if a.org_key in rival_by_org}:
            at_rival = [a for a in rows if a.org_key == rival_org]
            years = [a.first_year or (a.observed.year if a.observed else None) for a in at_rival]
            years = [y for y in years if y is not None]
            if not years:
                continue
            joined = min(years)
            if joined < today.year - max_join_age_years:
                continue  # long-standing member, not a recent move
            prior = [
                a for a in rows
                if a.org_key not in rival_by_org and a.first_year is not None and a.first_year < joined
            ]
            if not prior:
                continue
            prior.sort(key=lambda a: a.last_year or 0, reverse=True)
            rival = rival_by_org[rival_org]
            hires.append(Hire(
                author_id=author_id,
                author_name=at_rival[0].author_name,
                rival_key=rival.key,
                rival_name=rival.name,
                joined_year=joined,
                prior_orgs=[(a.org_name, a.first_year, a.last_year) for a in prior],
            ))
    return hires


# --- diff ------------------------------------------------------------------


def compute_diff(
    current: SignalBundle,
    previous: SignalBundle | None,
    *,
    today: date | None = None,
    max_join_age_years: int = 1,
) -> WeeklyDiff:
    """What is in ``current`` but was not in ``previous``."""
    today = today or date.today()
    prev = previous or SignalBundle()
    names = _rival_names(current)

    prev_rivals = {r.key for r in prev.rivals}
    prev_pubs = {p.key for p in prev.publications} | {_title_key(p.title) for p in prev.publications}
    prev_patents = {p.key for p in prev.patents}
    prev_hires = {h.key for h in find_hires(prev, today, max_join_age_years)}
    prev_roles = {a.key for a in prev.affiliations if a.author_id.startswith(ROLE_PREFIX)}

    diff = WeeklyDiff(baseline=previous is not None)
    diff.new_rivals = [
        r for r in current.rivals
        if r.key not in prev_rivals and (previous is not None or r.source != "seed")
    ]
    seeds = {r.key for r in current.rivals if r.source == "seed"}

    def had_baseline(rival_key: str) -> bool:
        return rival_key in (prev_rivals if previous is not None else seeds)

    diff.publications = [
        p for p in dedup_publications(current.publications)
        if p.key not in prev_pubs and _title_key(p.title) not in prev_pubs and had_baseline(p.rival_key)
    ]
    diff.patents = [
        p for p in current.patents if p.key not in prev_patents and had_baseline(p.rival_key)
    ]
    diff.hires = [
        h for h in find_hires(current, today, max_join_age_years)
        if h.key not in prev_hires and had_baseline(h.rival_key)
    ]
    diff.roles = [
        a for a in current.affiliations
        if a.author_id.startswith(ROLE_PREFIX) and a.key not in prev_roles and had_baseline(a.org_key)
    ]
    diff.items = to_items(diff, names)
    return diff


def to_items(diff: WeeklyDiff, names: dict[str, str]) -> list[ChangeItem]:
    """Flatten a diff into uniform ChangeItems (ids are stable across runs)."""
    items: list[ChangeItem] = []
    for r in diff.new_rivals:
        items.append(ChangeItem(
            id=f"new_rival:{r.key}", kind="new_rival", rival_key=r.key, rival_name=r.name,
            title=f"{r.name} added to tracked rivals",
            detail=f"source: {r.source}" + (f"; patents: {r.patent_count}" if r.patent_count is not None else ""),
        ))
    for p in diff.publications:
        items.append(ChangeItem(
            id=f"publication:{p.key}", kind="publication", rival_key=p.rival_key,
            rival_name=names.get(p.rival_key, p.rival_key), title=p.title,
            detail=("authors: " + ", ".join(p.authors[:5])) if p.authors else "",
            when=p.published, url=p.url,
        ))
    for p in diff.patents:
        items.append(ChangeItem(
            id=f"patent:{p.key}", kind="patent", rival_key=p.rival_key,
            rival_name=names.get(p.rival_key, p.rival_key), title=p.title,
            detail=("CPC: " + ", ".join(p.cpc_codes[:5])) if p.cpc_codes else "",
            when=p.published, url=p.url,
        ))
    for h in diff.hires:
        prior = "; ".join(
            f"{org} ({first}-{last})" if first else org for org, first, last in h.prior_orgs[:3]
        )
        items.append(ChangeItem(
            id=f"hire:{h.key}", kind="hire", rival_key=h.rival_key, rival_name=h.rival_name,
            title=f"{h.author_name} joined {h.rival_name} ({h.joined_year})",
            detail=f"previously: {prior}",
        ))
    for a in diff.roles:
        items.append(ChangeItem(
            id=f"role:{a.key}", kind="role", rival_key=a.org_key,
            rival_name=a.org_name, title=f"Open role: {a.author_name}", when=a.observed,
        ))
    return items
