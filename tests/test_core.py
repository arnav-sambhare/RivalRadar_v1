"""Pure logic: name matching, discovery ranking, filter, diff, relevance."""

import json
from collections import Counter
from datetime import date

from rivalradar import discovery, filters, relevance
from rivalradar.config import Config, Secrets
from rivalradar.diff import compute_diff
from rivalradar.models import Affiliation, Patent, Publication, Rival, SignalBundle
from rivalradar.names import normalize_org, org_matches

TODAY = date(2026, 9, 23)


# --- names -----------------------------------------------------------------


def test_normalize_org_strips_suffixes_and_country_tags():
    assert normalize_org("RIVAL ONE INC [US]") == "rival one"
    assert normalize_org("Rival One, GmbH") == "rival one"


def test_org_matches_is_prefix_based():
    assert org_matches("Mistral AI", "MISTRAL AI FRANCE SAS")
    assert org_matches("Mistral AI", "Mistral AI (France)")
    assert not org_matches("Mistral AI", "Mistral Aerospace")
    assert not org_matches("Rival One", "One Rival Corp")


# --- discovery & filter ----------------------------------------------------


def test_rank_candidates_drops_known_and_weak():
    counts = Counter({("I1", "Big Co"): 5, ("I2", "Small Co"): 3, ("I3", "Rare"): 1, ("", "Known"): 9})
    out = discovery.rank_candidates(counts, {"known"}, min_cooccurrence=2, max_new=2)
    # Fewer hits first: in a broad pool, many patents usually means an incumbent.
    assert [(r.name, r.openalex_institution_id, r.source) for r in out] == [
        ("Small Co", "I2", "discovery"), ("Big Co", "I1", "discovery")]


def test_rank_candidates_excludes_known_names_and_target():
    counts = Counter({("", "ACME ROBOTICS LTD"): 5, ("", "RIVAL ONE INC"): 4, ("", "NEWCO GMBH"): 3})
    out = discovery.rank_candidates(counts, set(), min_cooccurrence=2, max_new=10,
                                    known_names=["Rival One", "Acme Robotics"])
    assert [r.name for r in out] == ["NEWCO GMBH"]


def test_clean_phrases_sanitises_and_dedups():
    raw = ['weld "seam" inspection', "Weld seam inspection", 42, "", "a b c d e f g", "x-ray CT"]
    assert discovery.clean_phrases(raw) == ["weld seam inspection", "x-ray CT"]


def test_company_applicants_keeps_companies_in_epodoc_form_only():
    p = Patent("t", "", "CN1", applicants=[
        "HUNAN AOCHUANGPU TECH CO LTD [CN]",
        "湖南奥创普科技有限公司",  # same company, original format
        "ANQING NORMAL UNIV [CN]",   # university
        "DOE JOHN [US]",             # individual inventor
        "Siemens AG",                # original format: ignored
        "SIEMENS AG [DE]"])
    assert discovery.company_applicants(p) == {"HUNAN AOCHUANGPU TECH CO LTD", "SIEMENS AG"}


def test_cpc_profile_counts_subgroups_once_per_patent_and_skips_y_tags():
    pats = [Patent("a", "", "1", cpc_codes=["G06T7/0004", "G06T7/0004", "Y02P90/30", "G06T"]),
            Patent("b", "", "2", cpc_codes=["G06T7/0004", "G01N21/88"])]
    assert discovery.cpc_profile(pats) == Counter({"G06T7/0004": 2, "G01N21/88": 1})


def test_merge_candidates_joins_paper_and_patent_names():
    papers = Counter({("I7", "Mistral AI (France)"): 2})
    patents = Counter({"MISTRAL AI": 3, "NEWCO GMBH": 2})
    assert discovery.merge_candidates(papers, patents) == Counter(
        {("I7", "Mistral AI (France)"): 5, ("", "NEWCO GMBH"): 2})


def test_search_phrases_prefers_keywords_then_llm_then_short_description(monkeypatch):
    def cfg(**target):
        return Config(raw={"target": target}, path="x", secrets=Secrets())
    assert discovery.search_phrases(cfg(description="x", keywords=["weld inspection"])) == (
        ["weld inspection"], "keywords")
    monkeypatch.setattr(discovery, "complete_json",
                        lambda c, s, u: '{"phrases": ["weld seam inspection", "x-ray CT"]}')
    assert discovery.search_phrases(cfg(description="We inspect welds with X-ray CT for aerospace.")) == (
        ["weld seam inspection", "x-ray CT"], "llm")

    def no_llm(c, s, u):
        raise RuntimeError("no key")
    monkeypatch.setattr(discovery, "complete_json", no_llm)
    assert discovery.search_phrases(cfg(description="Weld inspection robots")) == (
        ["Weld inspection robots"], "description")
    assert discovery.search_phrases(cfg(description="We build robots that inspect welds on pipelines.")) == ([], "")


def test_discover_anchors_on_target_and_merges_sources(monkeypatch):
    def fake_llm(c, system, user):
        if system == discovery.SCREEN_PROMPT:  # relevance check: keep all but candidate 3
            n = user.count("-- matched:")
            return json.dumps({"keep": [i for i in range(1, n + 1) if i != 3]})
        return '{"technology_phrases": ["weld inspection"]}'
    monkeypatch.setattr(discovery, "complete_json", fake_llm)
    monkeypatch.setattr(discovery, "_resolve_target",
                        lambda c: Rival("Acme", openalex_institution_id="I100", source="target"))
    anchors_seen = []

    def cooccur(c, rivals, evidence=None):
        anchors_seen.append([r.name for r in rivals])
        return Counter({("I100", "Acme"): 9, ("I5", "Paper Co"): 2})  # the target itself must not come back
    monkeypatch.setattr(discovery, "_openalex_cooccurrence", cooccur)
    monkeypatch.setattr(discovery, "_topic_papers",
                        lambda c, p, s, evidence=None: (Counter({("I5", "Paper Co"): 1}),
                                                        {"weld inspection": 40}))
    # SOLO GMBH has 1 patent from broad searches only, so it needs a 2nd: out.
    # NICHE CO has 1 patent, but from a niche search: in.
    monkeypatch.setattr(discovery, "_topic_patents", lambda c, t, s, a, evidence=None: (
        Counter({"PATENT CO LTD": 2, "ACME LTD": 4, "SOLO GMBH": 1, "NICHE CO": 1}),
        Counter({"NICHE CO": 1}), {"weld inspection": 80}, {"C12P+C09B": 19}))
    cfg = Config(raw={"target": {"name": "Acme", "description": "Weld inspection."},
                      "sources": {"openalex": True, "epo": True}},
                 path="x", secrets=Secrets(epo_ops_key="k", epo_ops_secret="s"))
    res = discovery.discover(cfg, [], SignalBundle())
    assert anchors_seen == [["Acme"]]  # no seeds: the target is the anchor
    assert [r.name for r in res.rivals][0] == "NICHE CO"  # niche matches rank first
    # Ranked: NICHE CO (niche), PATENT CO LTD (2 hits), Paper Co (3); the check dropped #3.
    assert res.screened and res.screened_out == ["Paper Co"]
    assert [r.name for r in res.rivals] == ["NICHE CO", "PATENT CO LTD"]
    assert res.cpc_pair_hits == {"C12P+C09B": 19}
    assert res.phrases == ["weld inspection"] and res.phrase_source == "llm"
    assert res.phrase_hits == {"weld inspection": {"papers": 40, "patents": 80}}


def test_partition_keeps_unknown_counts():
    kept, excluded = filters.partition(
        [Rival("A", patent_count=10), Rival("B", patent_count=200), Rival("C")], 50)
    assert [r.name for r in kept] == ["A", "C"]
    assert [r.name for r in excluded] == ["B"]


def test_estimate_count_exact_and_scaled():
    mine = Patent("t", "", "EP1", applicants=["RIVAL ONE INC [US]"])
    other = Patent("x", "", "EP2", applicants=["OTHER CO"])
    assert filters.estimate_count("Rival One", 2, [mine, other]) == 1
    assert filters.estimate_count("Rival One", 1000, [mine] * 50 + [other] * 50) == 500
    assert filters.estimate_count("Rival One", None, []) is None


def test_small_startup_filter_keeps_all_without_epo_credentials():
    cfg = Config(raw={}, path="x", secrets=Secrets())
    kept, excluded = filters.apply_small_startup_filter(cfg, [Rival("A"), Rival("B")])
    assert len(kept) == 2 and not excluded


# --- diff ------------------------------------------------------------------


def _aff(author, org, name, first, last):
    return Affiliation(author, author.upper(), org, name, first_year=first, last_year=last)


def _bundles():
    hf = Rival("Hugging Face", openalex_institution_id="I1")
    prev = SignalBundle(
        rivals=[hf],
        publications=[Publication("Old Paper", "I1", "openalex", "W1")],
        patents=[Patent("Old Patent", "I1", "EP1")],
        affiliations=[_aff("a1", "I1", "HF", 2026, 2026), _aff("a1", "I9", "Oxford", 2022, 2025)],
    )
    cur = SignalBundle(
        rivals=[hf, Rival("NewCo", openalex_institution_id="I2", source="discovery")],
        publications=[
            Publication("Old Paper", "I1", "openalex", "W1"),
            Publication("New  Paper!", "I1", "openalex", "W2", published=date(2026, 9, 20)),
            Publication("new paper", "I1", "arxiv", "2609.1", url="https://arxiv.org/abs/2609.1"),
            Publication("Old paper", "I1", "arxiv", "2609.0"),
        ],
        patents=[Patent("Old Patent", "I1", "EP1"), Patent("New Patent", "I1", "EP2", cpc_codes=["G06N"])],
        affiliations=[
            _aff("a1", "I1", "HF", 2026, 2026), _aff("a1", "I9", "Oxford", 2022, 2025),  # reported last week
            _aff("a2", "I1", "HF", 2026, 2026), _aff("a2", "I8", "KIT", 2019, 2025),     # new hire
            _aff("a3", "I1", "HF", 2018, 2026), _aff("a3", "I8", "KIT", 2015, 2017),     # long-tenured
            _aff("a4", "I1", "HF", 2026, 2026),                                          # no prior org
            Affiliation("role:I1:ml-engineer", "ML Engineer", "I1", "Hugging Face"),
        ],
    )
    return prev, cur


def test_diff_reports_only_new_things():
    prev, cur = _bundles()
    d = compute_diff(cur, prev, today=TODAY)
    assert d.baseline
    assert [r.name for r in d.new_rivals] == ["NewCo"]
    assert [p.title for p in d.publications] == ["New  Paper!"]  # arXiv copy merged, old ones seen
    assert d.publications[0].url == "https://arxiv.org/abs/2609.1"
    assert [p.external_id for p in d.patents] == ["EP2"]
    assert [h.author_id for h in d.hires] == ["a2"]
    assert d.hires[0].prior_orgs == [("KIT", 2019, 2025)]
    assert len(d.roles) == 1
    assert sorted(i.kind for i in d.items) == ["hire", "new_rival", "patent", "publication", "role"]


def test_newly_tracked_rival_history_is_baseline_not_news():
    prev, cur = _bundles()
    cur.publications.append(Publication("NewCo back catalogue", "I2", "openalex", "W99"))
    cur.affiliations += [_aff("n1", "I2", "NewCo", 2026, 2026), _aff("n1", "I8", "KIT", 2019, 2025)]
    d = compute_diff(cur, prev, today=TODAY)
    assert "NewCo back catalogue" not in [p.title for p in d.publications]
    assert "n1" not in [h.author_id for h in d.hires]
    assert [r.name for r in d.new_rivals] == ["NewCo"]  # reported once, as a new rival


def test_first_run_counts_everything_as_new():
    _, cur = _bundles()
    cur.publications.append(Publication("NewCo back catalogue", "I2", "openalex", "W99"))
    d = compute_diff(cur, None, today=TODAY)
    assert not d.baseline
    assert len(d.hires) == 2 and len(d.publications) == 2  # seed data only
    assert [r.name for r in d.new_rivals] == ["NewCo"]  # seeds are not news


# --- relevance -------------------------------------------------------------


def test_parse_ranking_validates_llm_output():
    prev, cur = _bundles()
    items = compute_diff(cur, prev, today=TODAY).items
    ids = [i.id for i in items]
    text = json.dumps({"ranked": [
        {"id": ids[1], "score": 8, "why": "IP overlap"},
        {"id": "made-up", "score": 10},
        {"id": ids[1], "score": 9},  # duplicate
        {"id": ids[3], "score": 42},  # clamped to 10
        {"id": ids[0], "score": "n/a"},  # unusable score
    ]})
    ranked = relevance.parse_ranking(text, items, max_items=12)
    assert [(r.item.id, r.score) for r in ranked] == [(ids[3], 10.0), (ids[1], 8.0)]


def test_preselect_puts_hires_and_patents_first():
    prev, cur = _bundles()
    items = compute_diff(cur, prev, today=TODAY).items
    assert [i.kind for i in relevance.preselect(items, 10)][:2] == ["hire", "patent"]


def test_rank_falls_back_without_api_key():
    prev, cur = _bundles()
    items = compute_diff(cur, prev, today=TODAY).items
    cfg = Config(raw={"llm": {"max_items": 3}}, path="x", secrets=Secrets())
    res = relevance.rank(cfg, items)
    assert not res.llm_used and len(res.ranked) == 3
    assert res.total == len([i for i in items if i.kind != "new_rival"])
    assert all(r.item.kind != "new_rival" for r in res.ranked)  # listed in the summary instead


def test_cooccurrence_counts_only_company_institutions(monkeypatch):
    class Resp:
        def json(self):
            inst = lambda i, t: {"id": f"https://openalex.org/{i}", "display_name": i, "type": t}
            work = {"authorships": [{"institutions": [inst("I1", "company"), inst("I5", "company"),
                                                      inst("I6", "education"), inst("I7", "nonprofit")]}]}
            return {"results": [work, work]}
    monkeypatch.setattr(discovery, "http_get", lambda url, params=None: Resp())
    cfg = Config(raw={}, path="x", secrets=Secrets())
    counts = discovery._openalex_cooccurrence(cfg, [Rival("R", openalex_institution_id="I1")])
    assert counts == Counter({("I5", "I5"): 2})  # the rival itself, universities and nonprofits excluded


def test_normalize_org_drops_openalex_parenthetical_tags():
    assert normalize_org("Amgen (United States)") == "amgen"
    assert org_matches("Mistral AI", "Mistral AI (France)")


def test_office_clause():
    assert discovery.office_clause(["EP", "us", " WO "]) == " and (pn=EP or pn=US or pn=WO)"
    assert discovery.office_clause([]) == ""


def test_size_filter_stops_once_enough_have_passed(monkeypatch):
    checked = []
    counts = {"Big1": 900, "Small1": 3, "Big2": 500, "Small2": 1, "Small3": 0}
    def fake_annotate(config, rivals):
        for r in rivals:
            checked.append(r.name)
            r.patent_count = counts[r.name]
        return rivals
    monkeypatch.setattr(filters, "annotate_patent_counts", fake_annotate)
    cfg = Config(raw={"small_startup_filter": {"max_patents": 20}}, path="x", secrets=Secrets())
    kept, excluded = filters.apply_small_startup_filter(
        cfg, [Rival(n) for n in counts], keep_at_most=2)
    assert [r.name for r in kept] == ["Small1", "Small2"]   # big firms didn't use up the slots
    assert [r.name for r in excluded] == ["Big1", "Big2"]
    assert checked == ["Big1", "Small1", "Big2", "Small2"]  # stopped before Small3


def test_company_applicants_uses_inventors_to_spot_people():
    p = Patent("t", "", "WO1",
               applicants=["PILI [FR]", "KEMIWATT [FR]", "BENJAMIN E DROGUET [GB]",
                           "INSTITUT NAT DES SCIENCES APPLIQUEES DE TOULOUSE INSA TOULOUSE [FR]",
                           "CENTRE NAT RECH SCIENT [FR]", "SPARXELL UK LTD [GB]"],
               inventors=["DROGUET BENJAMIN [GB]", "LITTLEWOOD FLORA [GB]"])
    # Startups without a legal form count; the inventor filing in person and research bodies don't.
    assert discovery.company_applicants(p) == {"PILI", "KEMIWATT", "SPARXELL UK LTD"}


def test_one_word_company_is_not_mistaken_for_an_inventor():
    p = Patent("t", "", "IT1", applicants=["PILI [FR]"], inventors=["PILI GIORGIO [IT]"])
    assert discovery.company_applicants(p) == {"PILI"}


def test_qualifying_takes_one_niche_patent_or_enough_hits():
    papers = Counter({("I1", "Paper Only Co"): 2, ("I2", "Weak Co"): 1, ("I3", "Mixed Co"): 1})
    patents = Counter({"MIXED CO": 1, "NICHE LTD": 1, "BROAD LTD": 1})
    niche = Counter({"NICHE LTD": 1})
    kept, niche_by_key = discovery.qualifying(papers, patents, niche, min_cooccurrence=2)
    assert set(kept) == {("I1", "Paper Only Co"), ("I3", "Mixed Co"), ("", "NICHE LTD")}
    assert niche_by_key == {("", "NICHE LTD"): 1}


def test_rank_candidates_puts_niche_matches_first():
    counts = Counter({("", "BIG CORP"): 9, ("", "NICHE LTD"): 1})
    out = discovery.rank_candidates(counts, set(), min_cooccurrence=1, max_new=5,
                                    priority={("", "NICHE LTD"): 1})
    assert [r.name for r in out] == ["NICHE LTD", "BIG CORP"]


def test_clean_cpc_and_pairs():
    assert discovery.clean_cpc(["h01m", "B60L", "bogus", "B60L", "H02J", "C01B"]) == ["H01M", "B60L", "H02J"]
    assert discovery.clean_cpc(["H01M", "B60L", "H02J", "C01B"], 4) == ["H01M", "B60L", "H02J", "C01B"]
    assert discovery.clean_cpc("H01M") == []
    assert discovery.cpc_pairs(["H01M", "C01B"], ["B60L", "C01B"]) == [
        ("H01M", "B60L"), ("H01M", "C01B"), ("C01B", "B60L")]


def test_search_terms_takes_cpc_from_llm_even_with_keyword_override(monkeypatch):
    monkeypatch.setattr(discovery, "complete_json", lambda c, s, u: (
        '{"technology_phrases": ["solid electrolyte"], "technology_cpc": ["H01M"], '
        '"market_cpc": ["B60L", "H02J", "bad", "H01M", "Y02E", "B60K"]}'))
    cfg = Config(raw={"target": {"description": "Solid-state batteries.", "keywords": ["sulfide electrolyte"]}},
                 path="x", secrets=Secrets())
    t = discovery.search_terms(cfg)
    assert (t.phrases, t.source, t.technology_phrases, t.technology_cpc) == (
        ["sulfide electrolyte"], "keywords", ["sulfide electrolyte"], ["H01M"])
    assert t.market_cpc == ["B60L", "H02J", "H01M", "B60K"]  # up to 4; invalid and Y-tags dropped


def test_search_terms_splits_technology_and_market_phrases(monkeypatch):
    monkeypatch.setattr(discovery, "complete_json", lambda c, s, u: json.dumps({
        "technology_phrases": ["solid electrolyte", "lithium anode"],
        "market_phrases": ["electric vehicle battery"], "technology_cpc": [], "market_cpc": []}))
    t = discovery.search_terms(Config(raw={"target": {"description": "Solid-state batteries."}},
                                      path="x", secrets=Secrets()))
    assert t.phrases == ["solid electrolyte", "lithium anode", "electric vehicle battery"]
    assert t.technology_phrases == ["solid electrolyte", "lithium anode"]


def test_screen_batches_respect_token_budget():
    lines = ["x" * 300] * 10  # ~100 tokens each
    batches = discovery.screen_batches(lines, max_tokens=350, overhead="y" * 30)
    assert all(len(b) <= 3 for b in batches) and sum(len(b) for b in batches) == 10


def test_screen_keeps_everyone_when_llm_unavailable():
    cfg = Config(raw={"target": {"name": "Acme"}}, path="x", secrets=Secrets())
    rivals = [Rival("A Co"), Rival("B Co")]
    kept, dropped, ran = discovery.screen_candidates(cfg, rivals, {})
    assert kept == rivals and dropped == [] and not ran


def test_inventor_match_folds_transliterations():
    p = Patent("t", "", "TR1", applicants=["OEZTUERK ONUR [TR]"], inventors=["OZTURK ONUR [TR]"])
    assert discovery.company_applicants(p) == set()


def test_dead_phrases_are_those_matching_nothing_searched():
    res = discovery.DiscoveryResult(phrase_hits={
        "indigo biosynthesis": {"papers": 12, "patents": 3},
        "precision fermentation dyes": {"papers": 0, "patents": 0},
        "not searched": {"papers": None, "patents": None},
    })
    assert res.dead_phrases() == ["precision fermentation dyes"]



# --- crowded fields ---------------------------------------------------------


def test_clean_cpc_groups():
    assert discovery.clean_cpc_groups(["h01m10/0562", "H01M 4/134", "H01M", "bad/1", "H01M10/0562"]) == [
        "H01M10/0562", "H01M4/134"]


def test_rank_puts_niche_first_then_fewer_patents():
    counts = Counter({("", "INCUMBENT CO LTD"): 11, ("", "STARTUP INC"): 1,
                      ("", "NICHE BIG CORP"): 7, ("", "NICHE SMALL LTD"): 1})
    niche = {("", "NICHE BIG CORP"): 7, ("", "NICHE SMALL LTD"): 1}
    out = discovery.rank_candidates(counts, set(), min_cooccurrence=1, max_new=10, priority=niche)
    assert [r.name for r in out] == ["NICHE SMALL LTD", "NICHE BIG CORP", "STARTUP INC", "INCUMBENT CO LTD"]


def test_openalex_research_institutes_are_not_candidates():
    works = [{"display_name": "w", "authorships": [{"institutions": [
        {"id": "I1", "display_name": "Tianmu Lake Institute of Advanced Energy Storage", "type": "company"},
        {"id": "I2", "display_name": "Acme Batteries", "type": "company"}]}]}]
    assert discovery._count_institutions(works, {"company"}, set()) == Counter({("I2", "Acme Batteries"): 1})


def test_crowded_field_query_plan(monkeypatch):
    """Broad technology searches trigger narrower combinations; usable ones don't."""
    queries = []
    totals = {'ta all "solid electrolyte"': 3665, 'ta all "ionic liquid"': 12,
              "cpc=H01M and cpc=B60L": 5393, "cpc=H01M and cpc=B64C": 900}

    def fake_search(client, cql, n):
        queries.append(cql.split(" and pd>=")[0])
        base = cql.split(" and pd>=")[0]
        return totals.get(base, 5), []
    monkeypatch.setattr(discovery.epo, "get_client", lambda c: object())
    monkeypatch.setattr(discovery.epo, "search", fake_search)
    terms = discovery.SearchTerms(
        phrases=["solid electrolyte", "ionic liquid", "vehicle battery"],
        technology_phrases=["solid electrolyte", "ionic liquid"],
        technology_cpc=["H01M"], market_cpc=["B60L", "B64C"], technology_groups=["H01M10/0562"])
    cfg = Config(raw={"target": {}}, path="x", secrets=Secrets())
    discovery._topic_patents(cfg, terms, date(2024, 9, 23), [])
    # broad technology phrase -> combined with the market phrase; narrow one isn't
    assert 'ta all "solid electrolyte" and ta all "vehicle battery"' in queries
    assert not any(q.startswith('ta all "ionic liquid" and') for q in queries)
    # both subclass pairs too broad -> detailed group paired with each market subclass
    assert 'cpc="H01M10/0562" and cpc=B60L' in queries and 'cpc="H01M10/0562" and cpc=B64C' in queries


def test_crowded_field_skips_groups_when_subclass_pair_is_usable(monkeypatch):
    queries = []
    monkeypatch.setattr(discovery.epo, "get_client", lambda c: object())
    monkeypatch.setattr(discovery.epo, "search",
                        lambda client, cql, n: (queries.append(cql) or 20, []))
    terms = discovery.SearchTerms(technology_cpc=["C12P"], market_cpc=["D06P"],
                                  technology_groups=["C12P17/10"])
    discovery._topic_patents(Config(raw={"target": {}}, path="x", secrets=Secrets()),
                             terms, date(2024, 9, 23), [])
    assert not any('cpc="C12P17/10"' in q for q in queries)


def test_size_filter_uses_openalex_works_when_patents_look_small(monkeypatch):
    monkeypatch.setattr(filters, "annotate_patent_counts",
                        lambda c, rs: [setattr(r, "patent_count", r.patent_count or 0) for r in rs] and rs)
    monkeypatch.setattr(filters, "annotate_works_counts",
                        lambda c, rs: [setattr(r, "works_count", {"I1": 9000, "I2": 40}.get(r.openalex_institution_id))
                                       for r in rs] and rs)
    cfg = Config(raw={"small_startup_filter": {"max_patents": 20, "max_works": 500}}, path="x", secrets=Secrets())
    kept, excluded = filters.apply_small_startup_filter(
        cfg, [Rival("Hyundai Motor Group", openalex_institution_id="I1"),
              Rival("Tiny Cells", openalex_institution_id="I2"), Rival("EPO ONLY LTD")], keep_at_most=5)
    assert [r.name for r in excluded] == ["Hyundai Motor Group"]  # 0 EPO patents, but 9000 works
    assert [r.name for r in kept] == ["Tiny Cells", "EPO ONLY LTD"]
