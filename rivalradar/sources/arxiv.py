"""arXiv publications source.

arXiv has no reliable affiliation field, so searching a company name matches
any paper that merely mentions it. Instead we query by the rival's known
researchers — authors OpenAlex placed at the rival this run (read from the
registry's ``context``). Common names collide across fields, so a paper counts
for a rival only when ``min_author_overlap`` of its researchers co-author it.

Runs after OpenAlex; with OpenAlex disabled or empty it contributes nothing.
Free, no key. Supplementary to OpenAlex for publications.
"""

from __future__ import annotations

import logging
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

from ..config import Config
from ..models import Publication, Rival, SignalBundle
from .base import http_get

log = logging.getLogger("rivalradar.sources.arxiv")

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
MAX_RESULTS = 50
REQUEST_SPACING = 3  # seconds between calls, per arXiv API etiquette


def _norm_person(name: str) -> str:
    """Case- and accent-insensitive form of a person's name."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(ascii_only.lower().split())


def _parse_entry(entry: ET.Element, rival_key: str) -> Publication | None:
    title_el = entry.find(f"{ATOM}title")
    id_el = entry.find(f"{ATOM}id")
    if title_el is None or id_el is None or not id_el.text:
        return None

    published: date | None = None
    pub_el = entry.find(f"{ATOM}published")
    if pub_el is not None and pub_el.text:
        try:
            published = datetime.fromisoformat(
                pub_el.text.replace("Z", "+00:00")
            ).date()
        except ValueError:
            published = None

    authors = [
        a.findtext(f"{ATOM}name", default="").strip()
        for a in entry.findall(f"{ATOM}author")
    ]
    authors = [a for a in authors if a]

    abstract = (entry.findtext(f"{ATOM}summary", default="") or "").strip()
    arxiv_id = id_el.text.rsplit("/", 1)[-1]

    return Publication(
        title=" ".join((title_el.text or "").split()),
        rival_key=rival_key,
        source="arxiv",
        external_id=arxiv_id,
        url=id_el.text.strip(),
        published=published,
        authors=authors,
        abstract=abstract,
    )


def rival_researchers(rival: Rival, context: SignalBundle, limit: int) -> list[str]:
    """Names of authors OpenAlex placed at this rival, most recent first."""
    if not rival.openalex_institution_id:
        return []
    at_rival = [
        a for a in context.affiliations
        if a.org_key == rival.openalex_institution_id and a.author_name
    ]
    at_rival.sort(key=lambda a: a.last_year or 0, reverse=True)
    names = list(dict.fromkeys(a.author_name for a in at_rival))
    return names[:limit]


def select_papers(
    entries: list[Publication],
    researchers: list[str],
    since: date,
    min_overlap: int,
) -> list[Publication]:
    """Keep recent papers co-authored by at least ``min_overlap`` researchers."""
    known = {_norm_person(n) for n in researchers}
    kept: list[Publication] = []
    for pub in entries:
        if pub.published is not None and pub.published < since:
            continue
        overlap = sum(1 for a in pub.authors if _norm_person(a) in known)
        if overlap >= min_overlap:
            kept.append(pub)
    return kept


def _query(researchers: list[str], rival_key: str) -> list[Publication]:
    query = " OR ".join(f'au:"{name}"' for name in researchers)
    resp = http_get(
        ARXIV_API,
        params={
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": MAX_RESULTS,
        },
    )
    root = ET.fromstring(resp.text)
    pubs = [_parse_entry(e, rival_key) for e in root.findall(f"{ATOM}entry")]
    return [p for p in pubs if p is not None]


def collect(config: Config, rivals: list[Rival], context: SignalBundle) -> SignalBundle:
    """Collect recent arXiv papers by each rival's known researchers."""
    opts = config.arxiv
    min_overlap = int(opts.get("min_author_overlap", 2))
    limit = int(opts.get("max_authors_per_rival", 10))
    since = (datetime.now(timezone.utc) - timedelta(days=config.lookback_days)).date()

    pubs: dict[str, Publication] = {}
    first = True
    for rival in rivals:
        researchers = rival_researchers(rival, context, limit)
        if not researchers:
            continue
        if not first:
            time.sleep(REQUEST_SPACING)
        first = False
        entries = _query(researchers, rival.key)
        for pub in select_papers(entries, researchers, since, min_overlap):
            pubs.setdefault(pub.key, pub)
    if first:
        log.info("arxiv: no rival researchers known yet (needs OpenAlex); skipping")
    return SignalBundle(publications=list(pubs.values()))
