"""EPO OPS patents source.

Uses the European Patent Office Open Patent Services (OPS) REST API. Auth is
OAuth2 client-credentials from EPO_OPS_KEY / EPO_OPS_SECRET (env only). A CQL
search by applicant name returns bibliographic records, from which we read the
publication number, title, CPC codes, and applicants.

Applicant names vary ("RIVAL ONE INC [US]", "Rival One GmbH"), so the query
uses ``pa all`` (every word present, any order/extra words) and results are
then kept only if an applicant matches the rival after normalisation (see
``names.org_matches``) — broad query, strict filter.

Free with a registered key. Core source, but requires credentials — if they are
missing this raises and ``safe_collect`` degrades the run gracefully.

OPS enforces fair use and blocks clients that look like robots
("CLIENT.RobotDetected"). Every EPO call in a run goes through one ``OPSClient``
(see ``get_client``), which behaves like a polite client:
  * one HTTP session and one access token, reused until it expires;
  * a User-Agent carrying the config's ``contact_email``;
  * ``epo.min_interval`` seconds (default 10) before every request, stretched
    further if the server's ``X-Throttling-Control`` asks for it;
  * 403/429/5xx retried with backoff, but after a RobotDetected fault no more
    EPO calls are made for the rest of the run.
"""

from __future__ import annotations

import base64
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import requests

from ..config import Config
from ..models import Patent, Rival, SignalBundle
from ..names import normalize_org, org_matches
from .base import USER_AGENT, SourceError

OPS_ROOT = "https://ops.epo.org/3.2"
AUTH_URL = f"{OPS_ROOT}/auth/accesstoken"
SEARCH_URL = f"{OPS_ROOT}/rest-services/published-data/search/biblio"
MAX_RESULTS = 25
DEFAULT_INTERVAL = 10.0  # seconds before each request; known not to trip OPS
TOKEN_TTL_MARGIN = 60  # refresh the token this many seconds before it expires
TIMEOUT = 30
RETRYABLE = (403, 429, 500, 502, 503, 504)
_THROTTLE = re.compile(r"search=(\w+):(\d+)")

log = logging.getLogger("rivalradar.sources.epo")

NS = {
    "ops": "http://ops.epo.org",
    "ex": "http://www.epo.org/exchange",
}


class OPSClient:
    """One polite OPS session: cached token, paced requests, backoff."""

    def __init__(self, key: str, secret: str, *, contact_email: str = "",
                 interval: float = DEFAULT_INTERVAL, retries: int = 3):
        self.key, self.secret = key, secret
        self.interval = interval  # configured floor: never go faster
        self.pace = interval  # current wait; grows if the server asks
        self.retries = retries
        self.disabled: str | None = None  # why EPO calls stopped for this run
        self._token: str | None = None
        self._expires_at = 0.0
        self._last = 0.0
        self.session = requests.Session()
        agent = USER_AGENT.split(" (")[0]
        self.session.headers["User-Agent"] = f"{agent} ({contact_email})" if contact_email else USER_AGENT

    # --- auth ---

    def _auth_header(self) -> dict:
        if self._token is None or time.time() >= self._expires_at:
            creds = base64.b64encode(f"{self.key}:{self.secret}".encode()).decode()
            resp = self.session.post(
                AUTH_URL,
                headers={"Authorization": f"Basic {creds}",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200 or not resp.json().get("access_token"):
                # Bad credentials won't fix themselves: stop, don't retry per rival.
                self.disabled = f"EPO OPS auth failed (HTTP {resp.status_code}); check EPO_OPS_KEY/SECRET"
                raise SourceError(self.disabled, resp.status_code)
            payload = resp.json()
            self._token = payload["access_token"]
            self._expires_at = time.time() + float(payload.get("expires_in", 1200)) - TOKEN_TTL_MARGIN
        return {"Authorization": f"Bearer {self._token}"}

    # --- requests ---

    def _wait_turn(self) -> None:
        wait = self._last + self.pace - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, params: dict) -> requests.Response:
        """GET with pacing, token refresh on 401 and backoff on 403/429/5xx.

        A 404 is returned to the caller: OPS uses it for "no results".
        """
        if self.disabled:
            raise SourceError(f"{self.disabled}; skipping further EPO calls this run", 403)
        resp = None
        for attempt in range(self.retries):
            self._wait_turn()
            headers = {"Accept": "application/xml", **self._auth_header()}
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=TIMEOUT)
            except requests.RequestException as exc:  # DNS blips, resets, timeouts
                self._last = time.monotonic()
                if attempt == self.retries - 1:
                    raise SourceError(f"EPO request failed: {exc}") from exc
                delay = 5 * 2 ** attempt
                log.info("epo: network error (%s), retrying in %ds", type(exc).__name__, delay)
                time.sleep(delay)
                continue
            self._last = time.monotonic()
            self._adapt_pace(resp.headers.get("X-Throttling-Control", ""))

            if resp.status_code in (200, 404):
                return resp
            if resp.status_code == 401 and attempt < self.retries - 1:
                self._token = None  # rejected mid-run: re-authenticate
                continue
            if resp.status_code == 403 and "RobotDetected" in resp.text:
                self.disabled = "EPO OPS flagged this client as a robot"
                log.warning("epo: OPS robot detection triggered; no more EPO calls this run")
                raise SourceError(f"HTTP 403 from OPS: {resp.text[:200]}", 403)
            if resp.status_code in RETRYABLE and attempt < self.retries - 1:
                self._backoff(resp, attempt)
                continue
            raise SourceError(f"HTTP {resp.status_code} from OPS: {resp.text[:200]}", resp.status_code)
        raise SourceError(f"OPS request failed after {self.retries} attempts",
                          getattr(resp, "status_code", None))

    def _backoff(self, resp: requests.Response, attempt: int) -> None:
        retry_after = resp.headers.get("Retry-After", "")
        try:
            delay = float(retry_after)
        except ValueError:
            delay = min(5 * 2 ** attempt, 60)
        log.info("epo: HTTP %s, retrying in %.0fs", resp.status_code, delay)
        time.sleep(delay)

    def _adapt_pace(self, header: str) -> None:
        """Slow down if OPS's advertised search quota (per minute) demands it.

        The header looks like "busy (..., search=green:30)"; red/black mean back
        off. Never goes faster than the configured interval.
        """
        match = _THROTTLE.search(header or "")
        if not match:
            return
        colour, per_minute = match.group(1), int(match.group(2))
        needed = 60 / per_minute if per_minute else 60.0
        if colour in ("red", "black"):
            needed = max(needed, 30.0)
        self.pace = max(self.interval, needed)


_clients: dict[str, OPSClient] = {}


def get_client(config: Config) -> OPSClient:
    """The shared OPS client for these credentials: one session per process."""
    key = config.secrets.require("epo_ops_key")
    secret = config.secrets.require("epo_ops_secret")
    client = _clients.get(key)
    if client is None:
        opts = config.raw.get("epo", {}) or {}
        client = OPSClient(
            key, secret,
            contact_email=config.contact_email,
            interval=float(opts.get("min_interval", DEFAULT_INTERVAL)),
            retries=int(opts.get("retries", 3)),
        )
        _clients[key] = client
    return client


def _text(el: ET.Element | None) -> str:
    return " ".join((el.text or "").split()) if el is not None and el.text else ""


def _parse_document(doc: ET.Element) -> Patent | None:
    # Publication number: prefer docdb format.
    number = ""
    for pub_ref in doc.findall(".//ex:publication-reference/ex:document-id", NS):
        country = _text(pub_ref.find("ex:country", NS))
        docnum = _text(pub_ref.find("ex:doc-number", NS))
        kind = _text(pub_ref.find("ex:kind", NS))
        if country and docnum:
            number = f"{country}{docnum}{kind}"
            break
    if not number:
        return None

    # Title: first English invention-title, else first available.
    title = ""
    titles = doc.findall(".//ex:invention-title", NS)
    for t in titles:
        if t.get("lang") == "en":
            title = _text(t)
            break
    if not title and titles:
        title = _text(titles[0])

    # CPC classification symbols.
    cpc_codes: list[str] = []
    for cls in doc.findall(".//ex:patent-classifications/ex:patent-classification", NS):
        scheme = cls.find("ex:classification-scheme", NS)
        if scheme is not None and scheme.get("scheme") not in (None, "CPC", "CPCI"):
            continue
        section = _text(cls.find("ex:section", NS))
        cls_ = _text(cls.find("ex:class", NS))
        subclass = _text(cls.find("ex:subclass", NS))
        main_group = _text(cls.find("ex:main-group", NS))
        subgroup = _text(cls.find("ex:subgroup", NS))
        symbol = f"{section}{cls_}{subclass}".strip()
        if symbol and main_group and subgroup:
            symbol = f"{symbol}{main_group}/{subgroup}"  # e.g. G06T7/0004
        if symbol:
            cpc_codes.append(symbol)

    # Applicants.
    applicants = [
        _text(name)
        for name in doc.findall(".//ex:applicants/ex:applicant//ex:name", NS)
    ]
    applicants = list(dict.fromkeys(a for a in applicants if a))
    inventors = [
        _text(name)
        for name in doc.findall(".//ex:inventors/ex:inventor//ex:name", NS)
    ]
    inventors = list(dict.fromkeys(i for i in inventors if i))

    # Publication date (docdb).
    published: date | None = None
    date_el = doc.find(".//ex:publication-reference/ex:document-id/ex:date", NS)
    if date_el is not None and date_el.text:
        try:
            published = datetime.strptime(date_el.text.strip(), "%Y%m%d").date()
        except ValueError:
            published = None

    return Patent(
        title=title or "(untitled)",
        rival_key="",  # set by caller
        external_id=number,
        url=f"https://worldwide.espacenet.com/patent/search?q=pn%3D{number}",
        published=published,
        cpc_codes=list(dict.fromkeys(cpc_codes)),
        applicants=applicants,
        inventors=inventors,
    )


def applicant_query(rival_name: str) -> str:
    """CQL clause matching every word of the rival's normalised name."""
    return f'pa all "{normalize_org(rival_name)}"'


def search(client: OPSClient, cql: str, max_results: int = MAX_RESULTS) -> tuple[int | None, list[Patent]]:
    """Run a biblio search; return (total hit count, parsed patents).

    OPS reports a search with no hits as HTTP 404; that is (0, []), not an error.
    """
    resp = client.get(SEARCH_URL, {"q": cql, "Range": f"1-{max_results}"})
    if resp.status_code == 404:
        return 0, []
    root = ET.fromstring(resp.text)
    total: int | None = None
    for el in root.iter():
        value = el.get("total-result-count")
        if value is not None:
            total = int(value) if value.isdigit() else None
            break
    patents = [p for p in map(_parse_document, root.findall(".//ex:exchange-document", NS)) if p]
    return total, patents


def filed_by(patent: Patent, rival_name: str) -> bool:
    """True if any applicant on the patent is the rival (name-normalised)."""
    return any(org_matches(rival_name, a) for a in patent.applicants)


def _search_rival(rival: Rival, client: OPSClient, since: date) -> list[Patent]:
    # Broad applicant query restricted to the lookback window, then strict filter.
    cql = f'{applicant_query(rival.name)} and pd within "{since:%Y%m%d} {date.today():%Y%m%d}"'
    _, patents = search(client, cql)
    kept = [p for p in patents if filed_by(p, rival.name)]
    for patent in kept:
        patent.rival_key = rival.key
    return kept


def collect(config: Config, rivals: list[Rival], context: SignalBundle) -> SignalBundle:
    """Collect recent patents for the given rivals from EPO OPS."""
    client = get_client(config)

    since = (datetime.now(timezone.utc) - timedelta(days=config.lookback_days)).date()
    patents: list[Patent] = []
    seen: set[str] = set()
    for rival in rivals:
        try:
            found = _search_rival(rival, client, since)
        except SourceError as exc:  # one rival's failure must not drop the others
            log.warning("epo: skipping %s (%s)", rival.name, exc)
            continue
        for patent in found:
            if patent.key in seen:
                continue
            seen.add(patent.key)
            patents.append(patent)
    return SignalBundle(patents=patents)
