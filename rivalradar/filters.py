"""Small-startup filter.

Excludes any organisation with more than ``max_patents`` patents — a proxy for
"too big" (incumbents, universities). The patent count is looked up from EPO
OPS. The partition logic is pure and testable; the count lookup is network and
fails soft (an org whose count can't be determined is kept, not dropped, so we
never silently discard a genuine small startup).

A patent count can be wrongly low: a company named one way in OpenAlex
("Hyundai Motor Group") files under another at EPO ("HYUNDAI MOTOR CO LTD"),
so its lookup finds nothing. Organisations with an OpenAlex id therefore also
get a second size check: more than ``max_works`` OpenAlex works is too big.
"""

from __future__ import annotations

import logging

from .config import Config
from .models import Patent, Rival
from .sources.base import http_get
from .sources.epo import OPSClient, applicant_query, filed_by, get_client, search
from .sources.openalex import OPENALEX_INSTITUTIONS, base_params

log = logging.getLogger("rivalradar.filters")

COUNT_SAMPLE = 100  # OPS maximum per request


# --- pure cores ------------------------------------------------------------


def too_big(rival: Rival, max_patents: int, max_works: int | None = None) -> bool:
    """Over the patent limit, or (when known) over the OpenAlex works limit."""
    if rival.patent_count is not None and rival.patent_count > max_patents:
        return True
    return max_works is not None and rival.works_count is not None and rival.works_count > max_works


def partition(
    rivals: list[Rival], max_patents: int, max_works: int | None = None
) -> tuple[list[Rival], list[Rival]]:
    """Split rivals into (kept, excluded) by known size.

    Unknown sizes are kept. A rival over ``max_patents`` patents, or over
    ``max_works`` OpenAlex works when that is known, is excluded.
    """
    kept: list[Rival] = []
    excluded: list[Rival] = []
    for rival in rivals:
        if too_big(rival, max_patents, max_works):
            excluded.append(rival)
        else:
            kept.append(rival)
    return kept, excluded


def estimate_count(rival_name: str, total: int | None, sample: list[Patent]) -> int | None:
    """Patents actually filed by the rival, from a broad query's sample.

    The ``pa all`` query also returns other applicants sharing the words, so
    only name-matched records count. If the sample covers every hit the count is
    exact; otherwise the matched share of the sample is scaled to the total.
    """
    if total is None:
        return None
    if not sample:
        return 0
    matched = sum(1 for p in sample if filed_by(p, rival_name))
    if total <= len(sample):
        return matched
    return round(matched / len(sample) * total)


# --- network count lookup (fail-soft) ------------------------------------


def _patent_count(rival_name: str, client: OPSClient) -> int | None:
    """EPO patent count for a rival, or None if unavailable."""
    try:
        total, sample = search(client, applicant_query(rival_name), COUNT_SAMPLE)
    except Exception as exc:  # noqa: BLE001
        log.info("filter: patent-count lookup failed for %s (%s)", rival_name, exc)
        return None
    return estimate_count(rival_name, total, sample)


def annotate_works_counts(config: Config, rivals: list[Rival]) -> list[Rival]:
    """Fill ``works_count`` from OpenAlex for rivals with an institution id. Fail-soft."""
    for rival in rivals:
        if rival.works_count is not None or not rival.openalex_institution_id:
            continue
        try:
            resp = http_get(f"{OPENALEX_INSTITUTIONS}/{rival.openalex_institution_id}",
                            params={**base_params(config), "select": "works_count"})
            rival.works_count = int(resp.json().get("works_count") or 0)
        except Exception as exc:  # noqa: BLE001
            log.info("filter: works-count lookup failed for %s (%s)", rival.name, exc)
    return rivals


def annotate_patent_counts(config: Config, rivals: list[Rival]) -> list[Rival]:
    """Fill in ``patent_count`` for rivals that lack it, via EPO. Fail-soft.

    If EPO credentials are missing or auth fails, counts stay None and every
    rival is kept downstream.
    """
    try:
        client = get_client(config)
    except Exception as exc:  # noqa: BLE001
        log.warning("filter: EPO unavailable, skipping patent-count lookup (%s)", exc)
        return rivals

    for rival in rivals:
        if rival.patent_count is None:
            rival.patent_count = _patent_count(rival.name, client)
    return rivals


def apply_small_startup_filter(
    config: Config, rivals: list[Rival], *, keep_at_most: int | None = None
) -> tuple[list[Rival], list[Rival]]:
    """Check patent counts in order and split into (kept, excluded).

    With ``keep_at_most``, checking stops once that many have passed: each
    check is one paced EPO request, and candidates come best first. Rivals left
    unchecked are in neither list.
    """
    max_works = int(config.raw.get("small_startup_filter", {}).get("max_works", 500))
    if keep_at_most is None:
        annotate_patent_counts(config, rivals)
        annotate_works_counts(config, rivals)
        kept, excluded = partition(rivals, config.max_patents, max_works)
    else:
        kept, excluded = [], []
        for rival in rivals:
            if len(kept) >= keep_at_most:
                break
            annotate_works_counts(config, [rival])
            if not too_big(rival, config.max_patents, max_works):
                annotate_patent_counts(config, [rival])  # EPO call only if still in
            k, e = partition([rival], config.max_patents, max_works)
            kept += k
            excluded += e
    if excluded:
        log.info(
            "filter: excluded %d org(s) over %d patents: %s",
            len(excluded),
            config.max_patents,
            ", ".join(r.name for r in excluded),
        )
    return kept, excluded
