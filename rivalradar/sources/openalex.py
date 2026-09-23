"""OpenAlex source: publications and hiring signals.

Rivals are first resolved to an OpenAlex institution id (``resolve_institutions``,
run by the registry before any source) — name-based affiliation search proved
too noisy, returning software releases and unrelated authors. Rivals that don't
resolve are skipped here; set ``openalex_institution_id`` in config to fix one.

Two contributions per resolved rival:
  * Publications — recent articles/preprints with an author at the rival.
  * Affiliations — for each of those authors, their real institution history
    (from the author record, with years). An author at a rival now with other
    institutions in earlier years is the "hire" signal; the diff step reads it.

Free API. Optional ``OPENALEX_API_KEY`` avoids anonymous rate limits; the
config's ``contact_email`` is sent as ``mailto`` (OpenAlex's polite pool).
Core source.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from ..config import Config
from ..models import Affiliation, Publication, Rival, SignalBundle, parse_iso_date
from ..names import org_matches
from .base import http_get

log = logging.getLogger("rivalradar.sources.openalex")

OPENALEX = "https://api.openalex.org"
OPENALEX_WORKS = f"{OPENALEX}/works"
OPENALEX_AUTHORS = f"{OPENALEX}/authors"
OPENALEX_INSTITUTIONS = f"{OPENALEX}/institutions"
PER_PAGE = 50
AUTHOR_BATCH = 50  # ids per author lookup (OpenAlex allows up to 100 OR values)


def _short_id(url_or_id: str | None) -> str:
    return (url_or_id or "").rsplit("/", 1)[-1]


def base_params(config: Config) -> dict:
    """Auth/etiquette params shared by every OpenAlex request."""
    params: dict = {}
    if "@" in config.contact_email:
        params["mailto"] = config.contact_email
    if config.secrets.openalex_api_key:
        params["api_key"] = config.secrets.openalex_api_key
    return params


def works_filter(rival: Rival) -> str | None:
    """Filter selecting works with an author at this rival, or None if unresolved."""
    if not rival.openalex_institution_id:
        return None
    return f"authorships.institutions.id:{rival.openalex_institution_id}"


# --- institution resolution ----------------------------------------------


def resolve_institutions(config: Config, rivals: list[Rival]) -> None:
    """Fill ``openalex_institution_id`` in place for rivals that lack one.

    Takes the first search hit whose name matches the rival (see
    ``names.org_matches``). Fail-soft per rival: unresolved rivals keep an empty
    id and are skipped by the OpenAlex source.
    """
    params = base_params(config)
    for rival in rivals:
        if rival.openalex_institution_id:
            continue
        try:
            resp = http_get(
                OPENALEX_INSTITUTIONS,
                params={**params, "search": rival.name, "per-page": 5},
            )
        except Exception as exc:  # noqa: BLE001 - per-rival soft fail
            log.info("openalex: could not resolve %s (%s)", rival.name, exc)
            continue
        for inst in resp.json().get("results", []):
            if org_matches(rival.name, inst.get("display_name", "")):
                rival.openalex_institution_id = _short_id(inst.get("id"))
                log.info(
                    "openalex: resolved %s -> %s (%s)",
                    rival.name, rival.openalex_institution_id, inst.get("display_name"),
                )
                break
        else:
            log.info("openalex: no matching institution for %s; skipping it", rival.name)


# --- collection ------------------------------------------------------------


def _merge(affs: dict[str, Affiliation], aff: Affiliation) -> None:
    """Add an affiliation, widening the year range if already present."""
    existing = affs.get(aff.key)
    if existing is None:
        affs[aff.key] = aff
        return
    years = [y for y in (existing.first_year, existing.last_year,
                         aff.first_year, aff.last_year) if y is not None]
    if years:
        existing.first_year, existing.last_year = min(years), max(years)
    existing.observed = existing.observed or aff.observed


def _rival_works(
    config: Config, rival: Rival, since: date
) -> tuple[list[Publication], dict[str, Affiliation]]:
    """Recent works by the rival, plus the rival-affiliated authors on them."""
    resp = http_get(
        OPENALEX_WORKS,
        params={
            **base_params(config),
            "filter": (
                f"{works_filter(rival)},from_publication_date:{since.isoformat()},"
                "type:article|preprint"
            ),
            "per-page": PER_PAGE,
            "sort": "publication_date:desc",
        },
    )

    pubs: list[Publication] = []
    affs: dict[str, Affiliation] = {}
    for work in resp.json().get("results", []):
        work_id = _short_id(work.get("id"))
        if not work_id:
            continue
        published = parse_iso_date(work.get("publication_date"))
        authorships = work.get("authorships", []) or []
        pubs.append(
            Publication(
                title=work.get("display_name") or "(untitled)",
                rival_key=rival.key,
                source="openalex",
                external_id=work_id,
                url=work.get("id", ""),
                published=published,
                authors=[
                    (a.get("author") or {}).get("display_name", "")
                    for a in authorships
                    if (a.get("author") or {}).get("display_name")
                ],
            )
        )
        # Only authors actually listed at the rival on this work.
        for authorship in authorships:
            inst_ids = {_short_id(i.get("id")) for i in authorship.get("institutions", []) or []}
            if rival.openalex_institution_id not in inst_ids:
                continue
            author = authorship.get("author") or {}
            author_id = _short_id(author.get("id"))
            if not author_id:
                continue
            year = published.year if published else None
            _merge(affs, Affiliation(
                author_id=author_id,
                author_name=author.get("display_name", ""),
                org_key=rival.openalex_institution_id,
                org_name=rival.name,
                observed=published,
                first_year=year,
                last_year=year,
            ))
    return pubs, affs


def _author_histories(config: Config, author_ids: list[str]) -> list[Affiliation]:
    """Each author's full institution history, with year ranges."""
    out: list[Affiliation] = []
    for i in range(0, len(author_ids), AUTHOR_BATCH):
        batch = author_ids[i : i + AUTHOR_BATCH]
        resp = http_get(
            OPENALEX_AUTHORS,
            params={
                **base_params(config),
                "filter": "openalex:" + "|".join(batch),
                "per-page": AUTHOR_BATCH,
                "select": "id,display_name,affiliations",
            },
        )
        for author in resp.json().get("results", []):
            author_id = _short_id(author.get("id"))
            for entry in author.get("affiliations", []) or []:
                inst = entry.get("institution") or {}
                inst_id = _short_id(inst.get("id"))
                years = [y for y in entry.get("years", []) or [] if isinstance(y, int)]
                if not inst_id:
                    continue
                out.append(Affiliation(
                    author_id=author_id,
                    author_name=author.get("display_name", ""),
                    org_key=inst_id,
                    org_name=inst.get("display_name", ""),
                    first_year=min(years) if years else None,
                    last_year=max(years) if years else None,
                ))
    return out


def collect(config: Config, rivals: list[Rival], context: SignalBundle) -> SignalBundle:
    """Collect recent OpenAlex publications and author affiliation histories."""
    since = (datetime.now(timezone.utc) - timedelta(days=config.lookback_days)).date()

    pubs: dict[str, Publication] = {}
    affs: dict[str, Affiliation] = {}
    for rival in rivals:
        if works_filter(rival) is None:
            continue  # unresolved; logged during resolution
        rival_pubs, rival_affs = _rival_works(config, rival, since)
        for pub in rival_pubs:
            pubs.setdefault(pub.key, pub)
        for aff in rival_affs.values():
            _merge(affs, aff)

    # History lookup for everyone seen at a rival this run. Fail-soft: without
    # it we still have the current rival affiliations, just no prior orgs.
    author_ids = sorted({a.author_id for a in affs.values()})
    if author_ids:
        try:
            for aff in _author_histories(config, author_ids):
                _merge(affs, aff)
        except Exception as exc:  # noqa: BLE001
            log.warning("openalex: author history lookup failed (%s)", exc)

    return SignalBundle(publications=list(pubs.values()), affiliations=list(affs.values()))
