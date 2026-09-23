"""Careers-page hiring source — OPTIONAL and best-effort.

Fetches a rival's careers/jobs page (or RSS) and extracts posting titles as a
weak hiring signal. This source is deliberately degrade-graceful: pages block
bots, change markup, or 404 constantly, and none of that may break a run. It is
off by default in config.

Open roles are recorded as ``Affiliation`` observations with a synthetic
``author_id`` prefixed ``role:`` so downstream code can tell a posted opening
apart from a real researcher move (OpenAlex). No LinkedIn scraping.
"""

from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urljoin

from ..config import Config
from ..models import Affiliation, Rival, SignalBundle
from .base import SourceError, http_get, log

CANDIDATE_PATHS = ["/careers", "/jobs", "/careers/", "/company/careers", "/about/careers"]

# Very rough role-title heuristics; careers markup is wildly inconsistent.
ROLE_HINT = re.compile(
    r"\b(engineer|scientist|researcher|developer|lead|manager|designer|"
    r"intern|analyst|director|founder|head of)\b",
    re.IGNORECASE,
)


class _LinkTextParser(HTMLParser):
    """Collect anchor text — job boards usually render roles as links."""

    def __init__(self) -> None:
        super().__init__()
        self._in_a = False
        self._buf: list[str] = []
        self.texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._in_a = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_a:
            text = " ".join("".join(self._buf).split())
            if text:
                self.texts.append(text)
            self._in_a = False

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._buf.append(data)


def _synthetic_id(rival: Rival, role: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", role.lower()).strip("-")
    return f"role:{rival.key}:{slug}"[:120]


def _fetch_first_careers_page(rival: Rival) -> str:
    homepage = (rival.homepage or "").strip()
    if not homepage:
        raise SourceError(f"no homepage for rival {rival.name!r}")
    if not homepage.startswith("http"):
        homepage = "https://" + homepage

    last_err: Exception | None = None
    for path in CANDIDATE_PATHS:
        url = urljoin(homepage, path)
        try:
            return http_get(url, retries=1).text
        except Exception as exc:  # noqa: BLE001 - best-effort probing
            last_err = exc
            continue
    raise SourceError(f"no careers page reachable for {rival.name!r}: {last_err}")


def _roles_from_html(html: str) -> list[str]:
    parser = _LinkTextParser()
    parser.feed(html)
    roles = [t for t in parser.texts if ROLE_HINT.search(t) and len(t) <= 120]
    return list(dict.fromkeys(roles))  # dedup, preserve order


def collect(config: Config, rivals: list[Rival], context: SignalBundle) -> SignalBundle:
    """Best-effort open-role extraction. Never raises past a single rival."""
    affs: dict[str, Affiliation] = {}
    for rival in rivals:
        try:
            html = _fetch_first_careers_page(rival)
            for role in _roles_from_html(html):
                aff = Affiliation(
                    author_id=_synthetic_id(rival, role),
                    author_name=role,
                    org_key=rival.key,
                    org_name=rival.name,
                    observed=date.today(),
                )
                affs[aff.key] = aff
        except Exception as exc:  # noqa: BLE001 - degrade per rival
            log.info("careers: skipping %s (%s)", rival.name, exc)
            continue
    return SignalBundle(affiliations=list(affs.values()))
