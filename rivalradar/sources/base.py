"""Shared helpers for signal sources: HTTP with retries and soft failure.

Every source runs through ``safe_collect``, which guarantees a run is never
broken by a single failing source — errors are logged and the source
contributes an empty bundle instead.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import requests

from ..models import SignalBundle

log = logging.getLogger("rivalradar.sources")

DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 3
MAX_RETRY_AFTER = 60  # cap on a server-requested wait, in seconds
USER_AGENT = "RivalRadar/0.1 (competitive-landscape brief; contact via config)"


class SourceError(RuntimeError):
    """A recoverable error from a signal source (``status``: HTTP code, if any)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def http_get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> requests.Response:
    """GET with a shared UA, timeout, and backoff on 429/5xx.

    Honours a server's ``Retry-After`` (capped at ``MAX_RETRY_AFTER``) — OpenAlex
    in particular asks for ~30s waits under load. Other 4xx responses are final
    and raised at once, with the status on the ``SourceError`` (EPO, for one,
    answers an empty search with 404). Raises ``SourceError`` if all attempts
    fail so the caller (usually ``safe_collect``) can degrade gracefully.
    """
    merged_headers = {"User-Agent": USER_AGENT}
    if headers:
        merged_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        retry_after: int | None = None
        try:
            resp = requests.get(
                url, params=params, headers=merged_headers, timeout=timeout
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                header = resp.headers.get("Retry-After", "")
                retry_after = int(header) if header.isdigit() else None
                raise SourceError(f"HTTP {resp.status_code} from {url}", resp.status_code)
            if 400 <= resp.status_code < 500:
                raise _FinalError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:200]}", resp.status_code)
            resp.raise_for_status()
            return resp
        except _FinalError:
            raise
        except (requests.RequestException, SourceError) as exc:
            last_exc = exc
            if attempt < retries:
                backoff = (
                    min(retry_after + 1, MAX_RETRY_AFTER)
                    if retry_after is not None
                    else min(2 ** attempt, 10)
                )
                log.warning(
                    "GET %s failed (attempt %d/%d): %s — retrying in %ds",
                    url, attempt, retries, exc, backoff,
                )
                time.sleep(backoff)
    status = getattr(last_exc, "status", None)
    raise SourceError(f"GET {url} failed after {retries} attempts: {last_exc}", status)


class _FinalError(SourceError):
    """A response retrying won't fix (4xx other than 429)."""


def safe_collect(
    name: str,
    collect: Callable[[], SignalBundle],
) -> SignalBundle:
    """Run a source's collect fn, degrading to an empty bundle on any error.

    This is the graceful-degradation boundary: a broken source logs and returns
    nothing rather than aborting the weekly run.
    """
    try:
        bundle = collect()
        log.info(
            "source %s: %d pubs, %d patents, %d affiliations",
            name,
            len(bundle.publications),
            len(bundle.patents),
            len(bundle.affiliations),
        )
        return bundle
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for soft fail
        log.warning("source %s failed, skipping: %s", name, exc)
        return SignalBundle()
