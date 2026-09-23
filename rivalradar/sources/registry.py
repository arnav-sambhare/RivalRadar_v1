"""Source registry: run the enabled sources and merge their contributions.

Sources are looked up by the same names used in ``config.yaml`` under
``sources:``. Each runs through ``safe_collect`` so one failure never aborts the
weekly run.

Order matters: rivals are resolved to OpenAlex institution ids first (so every
source sees stable rival keys), and OpenAlex runs before arXiv because arXiv
searches by the researchers OpenAlex found. Each source receives the merged
signals collected so far as ``context``.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config
from ..models import Affiliation, Patent, Publication, Rival, SignalBundle
from . import arxiv, careers, epo, openalex
from .base import log, safe_collect

# name -> collect(config, rivals, context) -> SignalBundle, in run order.
SOURCES: dict[str, Callable[[Config, list[Rival], SignalBundle], SignalBundle]] = {
    "openalex": openalex.collect,
    "epo": epo.collect,
    "arxiv": arxiv.collect,
    "careers": careers.collect,
}


def collect_all(config: Config, rivals: list[Rival]) -> SignalBundle:
    """Collect from every enabled source and merge into one bundle."""
    if config.source_enabled("openalex"):
        try:
            openalex.resolve_institutions(config, rivals)
        except Exception as exc:  # noqa: BLE001 - resolution is best-effort
            log.warning("institution resolution failed: %s", exc)

    pubs: dict[str, Publication] = {}
    patents: dict[str, Patent] = {}
    affs: dict[str, Affiliation] = {}

    def merged() -> SignalBundle:
        return SignalBundle(
            rivals=list(rivals),
            publications=list(pubs.values()),
            patents=list(patents.values()),
            affiliations=list(affs.values()),
        )

    for name, fn in SOURCES.items():
        if not config.source_enabled(name):
            continue
        context = merged()
        bundle = safe_collect(name, lambda fn=fn: fn(config, rivals, context))
        for pub in bundle.publications:
            pubs.setdefault(pub.key, pub)
        for patent in bundle.patents:
            patents.setdefault(patent.key, patent)
        for aff in bundle.affiliations:
            affs[aff.key] = aff

    return merged()
