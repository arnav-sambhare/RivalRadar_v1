"""Source parsers and fail-soft behaviour, against captured response shapes."""

import xml.etree.ElementTree as ET
from datetime import date

from rivalradar.config import Config, Secrets
from rivalradar.models import Affiliation, Patent, Publication, Rival, SignalBundle
from rivalradar.sources import arxiv, base, careers, epo, openalex, registry

ATOM_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2409.12345v1</id><title>Cool  Paper</title><summary>abs</summary>
<published>2026-09-20T00:00:00Z</published><author><name>Jane Doe</name></author>
</entry></feed>"""

EPO_DOC = """<world xmlns:ex="http://www.epo.org/exchange"><ex:exchange-document><ex:bibliographic-data>
<ex:publication-reference><ex:document-id document-id-type="docdb">
<ex:country>EP</ex:country><ex:doc-number>1234567</ex:doc-number><ex:kind>A1</ex:kind><ex:date>20260918</ex:date>
</ex:document-id></ex:publication-reference>
<ex:invention-title lang="en">A Widget</ex:invention-title>
<ex:patent-classifications><ex:patent-classification>
<ex:classification-scheme scheme="CPC"/><ex:section>G</ex:section><ex:class>06</ex:class><ex:subclass>N</ex:subclass>
</ex:patent-classification></ex:patent-classifications>
<ex:applicants><ex:applicant><ex:applicant-name><ex:name>RIVAL ONE INC [US]</ex:name></ex:applicant-name></ex:applicant></ex:applicants>
</ex:bibliographic-data></ex:exchange-document></world>"""


def test_safe_collect_degrades_to_empty_bundle():
    def boom():
        raise RuntimeError("nope")
    assert base.safe_collect("x", boom).publications == []


def test_arxiv_entry_parse():
    entry = ET.fromstring(ATOM_FEED).find("{http://www.w3.org/2005/Atom}entry")
    pub = arxiv._parse_entry(entry, "rk")
    assert (pub.external_id, pub.title, pub.authors, pub.published) == (
        "2409.12345v1", "Cool Paper", ["Jane Doe"], date(2026, 9, 20))


def test_arxiv_requires_author_overlap_and_recency():
    researchers = ["Leandro von Werra", "Thomas Wolf", "Jérôme Dockès"]
    def pub(title, authors, day=date(2026, 9, 20)):
        return Publication(title, "r", "arxiv", title, published=day, authors=authors)
    entries = [
        pub("A", ["Leandro Von Werra", "Thomas Wolf"]),     # 2 known (case differs)
        pub("B", ["Thomas Wolf", "Someone Else"]),           # only 1: possible namesake
        pub("C", ["Jerome Dockes", "Thomas Wolf"]),          # accents differ
        pub("D", ["Leandro von Werra", "Thomas Wolf"], date(2026, 1, 1)),  # too old
    ]
    kept = arxiv.select_papers(entries, researchers, date(2026, 9, 1), 2)
    assert [p.title for p in kept] == ["A", "C"]


def test_arxiv_researchers_come_from_openalex_context():
    rival = Rival("HF", openalex_institution_id="I1")
    ctx = SignalBundle(affiliations=[
        Affiliation("a1", "Old Timer", "I1", "HF", last_year=2019),
        Affiliation("a2", "Recent Person", "I1", "HF", last_year=2026),
        Affiliation("a3", "Elsewhere", "I9", "Oxford", last_year=2026),
    ])
    assert arxiv.rival_researchers(rival, ctx, 10) == ["Recent Person", "Old Timer"]


def test_epo_document_parse_and_applicant_filter():
    doc = ET.fromstring(EPO_DOC).find(".//{http://www.epo.org/exchange}exchange-document")
    pat = epo._parse_document(doc)
    assert (pat.external_id, pat.title, pat.cpc_codes, pat.published) == (
        "EP1234567A1", "A Widget", ["G06N"], date(2026, 9, 18))
    assert epo.filed_by(pat, "Rival One")
    assert not epo.filed_by(pat, "Rival Two")
    assert epo.applicant_query("Rival One, Inc.") == 'pa all "rival one"'


def test_epo_parses_inventors():
    doc = ET.fromstring(EPO_DOC.replace(
        "</ex:applicants>",
        "</ex:applicants><ex:inventors><ex:inventor><ex:inventor-name><ex:name>DOE JANE [US]</ex:name>"
        "</ex:inventor-name></ex:inventor></ex:inventors>",
    )).find(".//{http://www.epo.org/exchange}exchange-document")
    assert epo._parse_document(doc).inventors == ["DOE JANE [US]"]


def test_epo_cpc_symbol_includes_group_when_present():
    doc = ET.fromstring(EPO_DOC.replace(
        "<ex:subclass>N</ex:subclass>",
        "<ex:subclass>N</ex:subclass><ex:main-group>3</ex:main-group><ex:subgroup>08</ex:subgroup>",
    )).find(".//{http://www.epo.org/exchange}exchange-document")
    assert epo._parse_document(doc).cpc_codes == ["G06N3/08"]


def test_careers_role_extraction():
    html = '<a href="1">Senior ML Engineer</a><a href="2">Our mission</a><a href="3">Research Scientist</a>'
    assert careers._roles_from_html(html) == ["Senior ML Engineer", "Research Scientist"]


def test_openalex_merge_widens_year_range():
    affs = {}
    openalex._merge(affs, Affiliation("A1", "X", "I1", "R", observed=date(2026, 9, 1), first_year=2026, last_year=2026))
    openalex._merge(affs, Affiliation("A1", "X", "I1", "R", first_year=2024, last_year=2026))
    a = affs["A1@I1"]
    assert (a.first_year, a.last_year, a.observed) == (2024, 2026, date(2026, 9, 1))


def test_registry_degrades_when_epo_credentials_missing():
    cfg = Config(raw={"sources": {"epo": True}}, path="x", secrets=Secrets())
    out = registry.collect_all(cfg, [Rival("Rival One")])
    assert out.patents == [] and len(out.rivals) == 1


def test_registry_passes_earlier_results_as_context(monkeypatch):
    seen = {}
    def first(config, rivals, context):
        return SignalBundle(patents=[Patent("P", "k", "EP9")])
    def second(config, rivals, context):
        seen["patents"] = [p.external_id for p in context.patents]
        return SignalBundle()
    monkeypatch.setattr(registry, "SOURCES", {"first": first, "second": second})
    cfg = Config(raw={"sources": {"first": True, "second": True}}, path="x", secrets=Secrets())
    registry.collect_all(cfg, [Rival("R")])
    assert seen["patents"] == ["EP9"]


class _Resp:
    def __init__(self, status, text="", headers=None, payload=None):
        self.status_code, self.text, self.headers = status, text, headers or {}
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for requests.Session: scripted GET replies, counted auth posts."""

    def __init__(self, replies):
        self.replies, self.gets, self.posts = list(replies), 0, 0
        self.headers = {}

    def post(self, *a, **k):
        self.posts += 1
        return _Resp(200, payload={"access_token": f"tok{self.posts}", "expires_in": 1200})

    def get(self, *a, **k):
        self.gets += 1
        return self.replies.pop(0)


def _client(monkeypatch, replies, interval=10.0):
    sleeps = []
    monkeypatch.setattr(epo.time, "sleep", sleeps.append)
    client = epo.OPSClient("k", "s", contact_email="me@example.com", interval=interval)
    client.session = _FakeSession(replies)
    return client, sleeps


def test_epo_empty_search_404_is_zero_results(monkeypatch):
    client, _ = _client(monkeypatch, [_Resp(404, "CLIENT.NoResultsFound")])
    assert epo.search(client, 'pa all "nobody"') == (0, [])


def test_epo_client_reuses_token_and_paces_every_request(monkeypatch):
    ok = _Resp(200, "<world/>")
    client, sleeps = _client(monkeypatch, [ok, ok, ok])
    for _ in range(3):
        epo.search(client, "q")
    assert client.session.posts == 1  # one token for the whole run
    assert len(sleeps) == 2 and all(s > 9 for s in sleeps)  # ~10s before each later request


def test_epo_client_refreshes_token_on_401(monkeypatch):
    client, _ = _client(monkeypatch, [_Resp(401), _Resp(200, "<world/>")])
    epo.search(client, "q")
    assert client.session.posts == 2


def test_epo_client_retries_transient_errors(monkeypatch):
    client, sleeps = _client(monkeypatch, [_Resp(503), _Resp(403, "busy"), _Resp(200, "<world/>")])
    assert epo.search(client, "q") == (None, [])
    assert client.session.gets == 3


def test_epo_client_stops_after_robot_detection(monkeypatch):
    client, _ = _client(monkeypatch, [_Resp(403, "<code>CLIENT.RobotDetected</code>")])
    for _ in range(3):
        try:
            epo.search(client, "q")
        except base.SourceError:
            pass
    assert client.session.gets == 1  # later searches are refused locally


def test_epo_client_stops_after_bad_credentials(monkeypatch):
    client, _ = _client(monkeypatch, [])
    client.session.post = lambda *a, **k: _Resp(401, payload={})
    for _ in range(2):
        try:
            epo.search(client, "q")
        except base.SourceError as exc:
            assert "auth failed" in str(exc)
    assert client.session.gets == 0


def test_epo_pace_follows_throttling_header(monkeypatch):
    client, _ = _client(monkeypatch, [], interval=2.0)
    client._adapt_pace("busy (images=green:200, search=green:30)")
    assert client.pace == 2.0
    client._adapt_pace("overloaded (search=yellow:10)")
    assert client.pace == 6.0
    client._adapt_pace("overloaded (search=black:0)")
    assert client.pace == 60.0
    client._adapt_pace("idle (search=green:200)")
    assert client.pace == 2.0  # never faster than configured


def test_epo_user_agent_carries_contact_email(monkeypatch):
    client = epo.OPSClient("k", "s", contact_email="me@example.com")
    assert "me@example.com" in client.session.headers["User-Agent"]


def test_epo_one_failing_rival_does_not_drop_the_others(monkeypatch):
    monkeypatch.setattr(epo, "get_client", lambda config: object())
    def search_rival(rival, client, since):
        if rival.name == "Bad":
            raise base.SourceError("HTTP 403", 403)
        return [Patent("P", rival.key, "EP1")]
    monkeypatch.setattr(epo, "_search_rival", search_rival)
    cfg = Config(raw={}, path="x", secrets=Secrets(epo_ops_key="k", epo_ops_secret="s"))
    out = epo.collect(cfg, [Rival("Bad"), Rival("Good")], SignalBundle())
    assert [p.external_id for p in out.patents] == ["EP1"]


def test_http_get_does_not_retry_client_errors(monkeypatch):
    calls = []
    class Resp:
        status_code = 404
        text = "no results"
        headers = {}
    monkeypatch.setattr(base.requests, "get", lambda *a, **k: calls.append(1) or Resp())
    try:
        base.http_get("https://example.test")
    except base.SourceError as exc:
        assert exc.status == 404
    assert len(calls) == 1


def test_epo_client_backs_off_on_network_errors(monkeypatch):
    client, sleeps = _client(monkeypatch, [])
    replies = iter([epo.requests.ConnectionError("dns"), _Resp(200, "<world/>")])
    def get(*a, **k):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r
    client.session.get = get
    assert epo.search(client, "q") == (None, [])
    assert 5 in sleeps  # waited before retrying
