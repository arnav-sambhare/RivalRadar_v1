"""Rival auto-discovery.

Expands the tracked set from public signals, anchored on the seed rivals AND
the target startup itself — so it works with no seeds at all:

  1. Co-occurrence in OpenAlex — companies that repeatedly co-author with the
     seeds, or with the target if OpenAlex knows it.
  2. Topic search — the LLM turns the target's description into short search
     phrases in patent vocabulary: some for its technology, some for its
     product/market (or ``--target-keywords`` sets them, as technology
     phrases). Companies on recent papers (OpenAlex, exact phrase in
     title/abstract) and patents (EPO, title/abstract) on those topics are
     candidates. Each phrase's hit counts are reported, so dead phrases show.
  3. Shared EPO CPC classifications:
     - the LLM also infers CPC subclasses for the target's technology (how
       the product is made or works) and its market (what the product itself
       is, and its main uses). Patents classed in both a technology and a
       market subclass are the target's niche, so each pair is searched
       (pairs over ``discovery.cpc_max_hits`` patents are too broad: skipped);
     - the most common CPC subgroups among the topic, seed and target patents
       are searched too. Groups too broad to be meaningful (over
       ``discovery.cpc_max_hits`` patents) are skipped.

Patent searches here are limited to ``discovery.patent_offices`` (default EP,
US, WO and the GB/FR/DE national offices, where European startups often file
first). Unrestricted, over 90% of topical hits are Chinese national filings,
which crowd out the companies competing in the markets a typical startup
targets, and each company shows up only once in the sample.

Only companies count: OpenAlex institutions of ``discovery.institution_types``
(default: company), and patent applicants that are neither research bodies
(universities, institutes...) nor inventors filing in their own name. Many
startups file without a legal form in the name (e.g. "PILI [FR]"), so an
applicant without one still counts unless it matches an inventor on the patent.

Thresholds: a candidate needs ``min_cooccurrence`` hits, except that one
patent is enough when it came from a niche query — a technology phrase or a
technology+market CPC pair matching at most ``niche_query_max_hits`` patents.
Market-only searches never count as niche: they find the market's incumbents
and suppliers as much as competitors. Niche matches are ranked first.

Crowded fields: when a technology phrase or a technology+market subclass pair
is too broad, narrower searches take over — the technology phrase combined
with each market phrase, and the LLM's detailed technology CPC groups (e.g. a
main group or subgroup rather than the whole subclass) paired with the market
subclasses. Within a group of equal standing, candidates with FEWER patents
in the pool rank first: in a broad search, many patents means incumbent.

Finally, an LLM relevance check reads each candidate with the titles of the
documents that found it and drops those that are clearly not competitors
(equipment suppliers, incidental filers, people). It is batched to fit the
Groq token budget (see ``llm.py``) and falls back to keeping everyone.
Up to ``discovery.max_candidates`` candidates are returned, best first; the
small-startup filter (``filters.py``) then keeps the first ``max_new_rivals``
that pass, so large firms near the top don't use up the slots.

Network probing is isolated and fails soft; ranking and merging are pure.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import Config
from .llm import complete_json, estimate_tokens
from .models import Patent, Rival, SignalBundle
from .names import LEGAL_SUFFIXES, normalize_org, org_matches
from .sources import epo
from .sources.base import http_get
from .sources.openalex import OPENALEX_WORKS, base_params, resolve_institutions, works_filter

log = logging.getLogger("rivalradar.discovery")

PHRASE_PROMPT = """\
You help find a startup's competitors by searching patent and paper titles and
abstracts. Competitors are companies using the same technology for the same
kind of product, and companies selling a substitute product into the same
market with a different technology.

From the startup's description, give:
- "technology_phrases": 2 or 3 search phrases for its core technology (how the
  product is made or how it works);
- "market_phrases": 2 or 3 search phrases for its product and market (what it
  sells, and what it is used for);
- "technology_cpc": 1 to 3 CPC patent subclasses for that technology;
- "technology_cpc_groups": 1 to 3 CPC main groups or subgroups that pin the
  technology down more precisely than its subclass (format like "H01M10/0562");
- "market_cpc": 1 to 4 CPC patent subclasses for the product itself (the class
  of the substance, material, device or article it is) and its main uses.

Phrases use the plain technical nouns found in patent claims and paper
abstracts, not startup or marketing language, and are 1 to 3 words long
(every word must appear in a matching document). No company names.
CPC subclasses are 4 characters: section letter, two digits, letter (e.g.
"H01M", "A61K"). Choose the most specific subclasses that apply.
Reply with JSON only:
{"technology_phrases": [], "market_phrases": [], "technology_cpc": [],
 "technology_cpc_groups": [], "market_cpc": []}
"""
SCREEN_PROMPT = """\
You check candidate competitors for a startup. Each numbered candidate is an
organisation found in patents or papers on the startup's topic, followed by
titles of the documents that matched it.

Keep a candidate if it plausibly competes with the startup: it develops or
sells a product that does the same job for the same customers, whether by the
same technology or a different one. Drop generic suppliers (equipment,
chemicals, services) whose matching documents are incidental to their
business; companies that would buy, integrate or distribute the startup's
product rather than make a competing one (its likely customers or partners);
and anything that is not a company (e.g. a person or a research institute).
When unsure, keep it.
Reply with JSON only: {"keep": [numbers of the candidates to keep]}
"""
CPC_SUBCLASS = re.compile(r"^[A-H]\d{2}[A-Z]$")
CPC_GROUP = re.compile(r"^[A-H]\d{2}[A-Z]\d{1,4}/\d{2,6}$")
MAX_CPC_PAIRS = 8
MAX_PHRASE_COMBOS = 4
MAX_PHRASES = 6
MAX_EVIDENCE = 3  # document titles kept per candidate for the relevance check
DEFAULT_OFFICES = ["EP", "US", "WO", "GB", "FR", "DE"]
# Research bodies and other non-company applicants (EPO abbreviates, e.g. UNIV,
# INST, CENTRE NAT RECH SCIENT).
NON_COMPANY = re.compile(
    r"\b(univ\w*|inst|institut\w*|college|school|hospital|hopita\w*|clinic|academ\w*|"
    r"foundation|council|ministry|government|govt|commissariat|cnrs|inserm|"
    r"fraunhofer|forschung\w*|centre nat|center nat|trust)\b"
)
# Applicants in EPO's romanized "epodoc" format end with a country tag, e.g.
# "HUNAN AOCHUANGPU TECH CO LTD [CN]". The other ("original") format repeats the
# same applicant, sometimes in non-Latin script, so only epodoc names count.
EPODOC_TAG = re.compile(r"\[[A-Z]{2}\]\s*$")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class DiscoveryResult:
    rivals: list[Rival] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    phrase_source: str = ""  # "keywords" | "llm" | "description" | ""
    # Hits per phrase: {"phrase": {"papers": n, "patents": n}}; None = not searched.
    phrase_hits: dict[str, dict[str, int | None]] = field(default_factory=dict)
    # CPC subclass pairs searched, "C12P+C09B" -> patents matched (None = failed).
    cpc_pair_hits: dict[str, int | None] = field(default_factory=dict)
    # Relevance check: whether the LLM ran, and the candidates it dropped.
    screened: bool = False
    screened_out: list[str] = field(default_factory=list)

    def dead_phrases(self) -> list[str]:
        """Phrases that were searched and matched nothing anywhere."""
        return [p for p, h in self.phrase_hits.items()
                if any(v is not None for v in h.values()) and not any(h.values())]


# --- pure helpers ----------------------------------------------------------


def clean_phrases(raw: list, limit: int = MAX_PHRASES) -> list[str]:
    """Keep short, distinct phrases, safe to embed in API query syntax."""
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        phrase = " ".join(re.sub(r"[^\w\s-]", " ", item).split())
        if 1 <= len(phrase.split()) <= 6 and phrase.lower() not in {p.lower() for p in out}:
            out.append(phrase)
    return out[:limit]


def _name_tokens(name: str) -> set[str]:
    """Lowercase words, with German-style transliterations folded
    ("OEZTUERK" -> "ozturk") so the same person matches across spellings."""
    words = _WORD.findall(EPODOC_TAG.sub("", name).lower())
    return {w.replace("ae", "a").replace("oe", "o").replace("ue", "u") for w in words}


def _is_inventor(tokens: set[str], inventors: list[set[str]]) -> bool:
    """Same person, allowing for word order and middle initials.

    "BENJAMIN E DROGUET" matches inventor "DROGUET BENJAMIN". Needs two words
    in common, so a one-word company ("PILI") never matches an inventor.
    """
    return any(len(tokens & inv) >= 2 and (tokens <= inv or inv <= tokens) for inv in inventors)


def company_applicants(patent: Patent) -> set[str]:
    """Applicants on a patent that are companies, in epodoc form, tag removed.

    Research bodies are dropped. A name with a legal form (LTD, INC...) is a
    company; one without is a company too unless it is an inventor filing in
    person — which needs inventor data, so without it only legal forms count.
    """
    inventors = [t for t in (_name_tokens(i) for i in patent.inventors) if t]
    out: set[str] = set()
    for name in patent.applicants:
        if not EPODOC_TAG.search(name):
            continue
        bare = EPODOC_TAG.sub("", name).strip()
        tokens = _name_tokens(bare)
        if not tokens or NON_COMPANY.search(bare.lower()):
            continue
        if tokens & LEGAL_SUFFIXES:
            out.add(bare)
        elif inventors and not _is_inventor(tokens, inventors):
            out.add(bare)
    return out


def cpc_profile(patents: list[Patent]) -> Counter[str]:
    """How often each full CPC subgroup occurs, one count per patent.

    Y-section codes are cross-cutting tags (e.g. climate), not technology
    classes, so they are left out.
    """
    counts: Counter[str] = Counter()
    for patent in patents:
        for code in set(patent.cpc_codes):
            if "/" in code and not code.startswith("Y"):
                counts[code] += 1
    return counts


def _same_org(a: str, b: str) -> bool:
    return org_matches(a, b) or org_matches(b, a)


def merge_candidates(
    paper_counts: Counter[tuple[str, str]], patent_counts: Counter[str]
) -> Counter[tuple[str, str]]:
    """Combine OpenAlex (id, name) and patent-applicant name counts.

    A patent applicant naming the same company as an OpenAlex institution adds
    to that institution's weight; otherwise it becomes its own candidate.
    """
    merged: Counter[tuple[str, str]] = Counter(paper_counts)
    for name, weight in patent_counts.items():
        match = next((k for k in merged if k[1] and _same_org(k[1], name)), None)
        merged[match or ("", name)] += weight
    return merged


def qualifying(
    paper_counts: Counter[tuple[str, str]],
    patent_counts: Counter[str],
    niche_counts: Counter[str],
    *,
    min_cooccurrence: int,
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], int]]:
    """Merged weights of qualifying candidates, and their niche-patent counts.

    A candidate qualifies with ``min_cooccurrence`` hits in total, or with a
    single patent from a niche query.
    """
    merged = merge_candidates(paper_counts, patent_counts)
    niche = merge_candidates(Counter(), niche_counts)
    niche_by_key: dict[tuple[str, str], int] = {}
    for (_, name), n in niche.items():
        key = next((k for k in merged if k[1] and _same_org(k[1], name)), ("", name))
        niche_by_key[key] = niche_by_key.get(key, 0) + n
    kept = Counter({
        key: weight for key, weight in merged.items()
        if weight >= min_cooccurrence or niche_by_key.get(key, 0) >= 1
    })
    return kept, niche_by_key


def rank_candidates(
    counts: Counter[tuple[str, str]],
    known_keys: set[str],
    *,
    min_cooccurrence: int,
    max_new: int,
    known_names: list[str] | tuple[str, ...] = (),
    priority: dict[tuple[str, str], int] | None = None,
) -> list[Rival]:
    """Turn (org_key, org_name) -> weight counts into ranked new Rivals.

    Drops anything already known (by key, or by name for candidates without an
    OpenAlex id) or below ``min_cooccurrence``; at most ``max_new``, best first:
    niche matches (``priority``) before the rest, and within each, fewer hits
    first — a company with many patents in a broad pool is an incumbent.
    """
    priority = priority or {}
    ordered = sorted(counts.items(), key=lambda kv: (-(priority.get(kv[0], 0) > 0), kv[1]))
    ranked: list[Rival] = []
    for (org_key, org_name), weight in ordered:
        if weight < min_cooccurrence:
            continue
        key = org_key or org_name.strip().lower()
        if key in known_keys or any(_same_org(org_name, n) for n in known_names if n):
            continue
        ranked.append(Rival(
            name=org_name,
            openalex_institution_id=org_key if org_key.startswith("I") else "",
            source="discovery",
        ))
        if len(ranked) >= max_new:
            break
    return ranked


# --- search phrases ----------------------------------------------------------


def clean_cpc(raw, limit: int = 3) -> list[str]:
    """Valid, distinct CPC subclasses (e.g. "H01M"), at most ``limit``."""
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        code = str(item).strip().upper()[:4]
        if CPC_SUBCLASS.match(code) and code not in out:
            out.append(code)
    return out[:limit]


def clean_cpc_groups(raw, limit: int = 3) -> list[str]:
    """Valid, distinct CPC main groups/subgroups (e.g. "H01M10/0562")."""
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        code = "".join(str(item).split()).upper()
        if CPC_GROUP.match(code) and code not in out:
            out.append(code)
    return out[:limit]


def cpc_pairs(technology: list[str], market: list[str], limit: int = MAX_CPC_PAIRS) -> list[tuple[str, str]]:
    """Technology x market subclass pairs to search, skipping same-class pairs."""
    return [(t, m) for t in technology for m in market if t != m][:limit]


@dataclass
class SearchTerms:
    phrases: list[str] = field(default_factory=list)  # all phrases, technology first
    source: str = ""  # where the phrases came from
    technology_phrases: list[str] = field(default_factory=list)
    technology_cpc: list[str] = field(default_factory=list)
    market_cpc: list[str] = field(default_factory=list)
    technology_groups: list[str] = field(default_factory=list)

    @property
    def market_phrases(self) -> list[str]:
        return [p for p in self.phrases if p not in self.technology_phrases]


def search_terms(config: Config) -> SearchTerms:
    """Phrases and CPC subclasses for topic discovery.

    ``--target-keywords`` set the phrases; otherwise the LLM derives them from
    the description; without the LLM a short description is used as-is. The
    CPC subclasses come from the LLM only (none without it).
    """
    target = config.target
    keywords = clean_phrases(list(target.get("keywords") or []))
    terms = SearchTerms(phrases=keywords, technology_phrases=list(keywords))
    if terms.phrases:
        terms.source = "keywords"
    description = (target.get("description") or "").strip()
    if not description:
        return terms
    try:
        data = json.loads(complete_json(config, PHRASE_PROMPT, f"Startup description: {description}"))
        terms.technology_cpc = clean_cpc(data.get("technology_cpc"), 3)
        terms.market_cpc = clean_cpc(data.get("market_cpc"), 4)
        terms.technology_groups = clean_cpc_groups(data.get("technology_cpc_groups"))
        if not terms.phrases:
            tech = clean_phrases(data.get("technology_phrases") or data.get("phrases") or [], 3)
            market = clean_phrases(data.get("market_phrases") or [], 3)
            terms.phrases = clean_phrases(tech + market)
            terms.technology_phrases = [p for p in tech if p in terms.phrases]
            terms.source = "llm" if terms.phrases else ""
    except Exception as exc:  # noqa: BLE001 - LLM is optional
        log.info("discovery: LLM search terms unavailable (%s)", exc)
    if terms.phrases:
        return terms
    short = clean_phrases([description])
    if short and len(short[0].split()) <= 6:
        terms.phrases, terms.technology_phrases, terms.source = short, list(short), "description"
        return terms
    log.warning("discovery: no search phrases (LLM unavailable, description too long); "
                "pass --target-keywords to enable topic discovery")
    return terms


def search_phrases(config: Config) -> tuple[list[str], str]:
    """Just the phrases of ``search_terms`` and where they came from."""
    terms = search_terms(config)
    return terms.phrases, terms.source


# --- network probing (fail-soft) -----------------------------------------


def _add_evidence(evidence: dict[str, list[str]] | None, name: str, title: str) -> None:
    """Remember up to MAX_EVIDENCE document titles per organisation."""
    if evidence is None or not title:
        return
    titles = evidence.setdefault(normalize_org(name), [])
    if title not in titles and len(titles) < MAX_EVIDENCE:
        titles.append(title)


def _evidence_for(name: str, evidence: dict[str, list[str]]) -> list[str]:
    exact = evidence.get(normalize_org(name))
    if exact:
        return exact
    return next((t for k, t in evidence.items() if k and _same_org(k, normalize_org(name))), [])


def _openalex_cooccurrence(
    config: Config, rivals: list[Rival], evidence: dict[str, list[str]] | None = None
) -> Counter[tuple[str, str]]:
    """Count company institutions co-occurring with the given anchors' works."""
    counts: Counter[tuple[str, str]] = Counter()
    base = base_params(config)
    types = set(config.discovery.get("institution_types", ["company"]))
    known_ids = {r.openalex_institution_id for r in rivals if r.openalex_institution_id}

    for rival in rivals:
        rival_filter = works_filter(rival)
        if rival_filter is None:
            continue  # unresolved; nothing reliable to co-occur with
        try:
            resp = http_get(OPENALEX_WORKS, params={
                **base, "filter": rival_filter, "per-page": 50, "sort": "publication_date:desc"})
        except Exception as exc:  # noqa: BLE001 - per-anchor soft fail
            log.info("discovery: OpenAlex probe failed for %s (%s)", rival.name, exc)
            continue
        counts += _count_institutions(resp.json().get("results", []), types, known_ids, evidence)
    return counts


def _count_institutions(
    works: list[dict], types: set[str], skip_ids: set[str],
    evidence: dict[str, list[str]] | None = None,
) -> Counter[tuple[str, str]]:
    """Each eligible institution once per work (and the work's title as evidence)."""
    counts: Counter[tuple[str, str]] = Counter()
    for work in works:
        seen: dict[str, str] = {}
        for authorship in work.get("authorships", []) or []:
            for inst in authorship.get("institutions", []) or []:
                inst_id = (inst.get("id") or "").rsplit("/", 1)[-1]
                name = inst.get("display_name") or ""
                if (inst.get("type") in types and inst_id and name and inst_id not in skip_ids
                        and not NON_COMPANY.search(name.lower())):
                    seen[inst_id] = name
        for inst_id, name in seen.items():
            counts[(inst_id, name)] += 1
            _add_evidence(evidence, name, work.get("display_name") or "")
    return counts


def _resolve_target(config: Config) -> Rival | None:
    """The target as an OpenAlex anchor, if OpenAlex knows it."""
    name = (config.target.get("name") or "").strip()
    if not name:
        return None
    target = Rival(name=name, source="target")
    resolve_institutions(config, [target])
    return target if target.openalex_institution_id else None


def _topic_papers(
    config: Config, phrases: list[str], since: date, evidence: dict[str, list[str]] | None = None
) -> tuple[Counter[tuple[str, str]], dict[str, int]]:
    """Companies on recent papers with a phrase in the title or abstract, and
    how many papers each phrase matched."""
    types = set(config.discovery.get("institution_types", ["company"]))
    counts: Counter[tuple[str, str]] = Counter()
    hits: dict[str, int] = {}
    for phrase in phrases:
        try:
            resp = http_get(OPENALEX_WORKS, params={
                **base_params(config),
                "filter": f'title_and_abstract.search:"{phrase}",'
                          f"from_publication_date:{since.isoformat()},type:article|preprint",
                "per-page": 200,
            })
        except Exception as exc:  # noqa: BLE001
            log.info("discovery: OpenAlex topic search failed for %r (%s)", phrase, exc)
            continue
        data = resp.json()
        hits[phrase] = int(data.get("meta", {}).get("count") or 0)
        counts += _count_institutions(data.get("results", []), types, set(), evidence)
    return counts, hits


def office_clause(offices: list[str]) -> str:
    """CQL restricting publications to the given offices, e.g. EP/US/WO."""
    codes = [o.strip().upper() for o in offices if str(o).strip()]
    if not codes:
        return ""
    return " and (" + " or ".join(f"pn={c}" for c in codes) + ")"


def _topic_patents(
    config: Config, terms: SearchTerms, since: date, anchor_patents: list[Patent],
    evidence: dict[str, list[str]] | None = None,
) -> tuple[Counter[str], Counter[str], dict[str, int], dict[str, int | None]]:
    """Company applicants on topical, CPC-pair and shared-CPC patents.

    Returns (patents per applicant, niche patents per applicant, patents per
    phrase, patents per CPC pair). A patent is niche if a technology phrase or
    technology+market CPC pair that matched at most ``niche_query_max_hits``
    patents found it.
    """
    client = epo.get_client(config)
    disc = config.discovery
    niche_max = int(disc.get("niche_query_max_hits", 50))
    pool: dict[str, Patent] = {}
    niche_keys: set[str] = set()
    hits: dict[str, int] = {}
    pair_hits: dict[str, int | None] = {}

    def add(cql: str, label: str, *, max_hits: int | None = None, niche_ok: bool = False) -> int | None:
        try:
            total, patents = epo.search(client, cql, 100)
        except Exception as exc:  # noqa: BLE001
            log.info("discovery: EPO search failed for %s (%s)", label, exc)
            return None
        if max_hits is not None and total is not None and total > max_hits:
            log.info("discovery: CPC %s too broad (%d patents), skipped", label, total)
            return total
        for p in patents:
            pool.setdefault(p.key, p)
            if niche_ok and total is not None and total <= niche_max:
                niche_keys.add(p.key)
        return total

    offices = office_clause(disc.get("patent_offices", DEFAULT_OFFICES))
    max_hits = int(disc.get("cpc_max_hits", 500))
    for phrase in terms.phrases:
        total = add(f'ta all "{phrase}" and pd>={since:%Y%m%d}{offices}', repr(phrase),
                    niche_ok=phrase in terms.technology_phrases)
        if total is not None:
            hits[phrase] = total
    window = f" and pd>={since:%Y%m%d}{offices}"

    # Technology phrases too broad on their own: require a market phrase too.
    combos = [(t, m) for t in terms.technology_phrases if (hits.get(t) or 0) > niche_max
              for m in terms.market_phrases][:MAX_PHRASE_COMBOS]
    for tech, market in combos:
        pair_hits[f'"{tech}"+"{market}"'] = add(
            f'ta all "{tech}" and ta all "{market}"{window}', f"{tech!r}+{market!r}",
            max_hits=max_hits, niche_ok=True)

    usable: set[str] = set()  # technology subclasses with a usable (not too broad) pair
    for tech, market in cpc_pairs(terms.technology_cpc, terms.market_cpc):
        label = f"{tech}+{market}"
        n = add(f"cpc={tech} and cpc={market}{window}", label, max_hits=max_hits, niche_ok=True)
        pair_hits[label] = n
        if n is not None and n <= max_hits:
            usable.add(tech)

    # Subclass too broad (or none given): use the detailed technology groups.
    group_pairs = [(g, m) for g in terms.technology_groups if g[:4] not in usable
                   for m in terms.market_cpc if m != g[:4]][:MAX_CPC_PAIRS]
    for group, market in group_pairs:
        label = f"{group}+{market}"
        pair_hits[label] = add(f'cpc="{group}" and cpc={market}{window}', label,
                               max_hits=max_hits, niche_ok=True)

    target = (config.target.get("name") or "").strip()
    target_patents: list[Patent] = []
    if target:
        try:
            _, found = epo.search(client, epo.applicant_query(target), 100)
            target_patents = [p for p in found if epo.filed_by(p, target)]
        except Exception as exc:  # noqa: BLE001
            log.info("discovery: EPO lookup of target failed (%s)", exc)

    profile = cpc_profile(list(pool.values()) + anchor_patents + target_patents)
    for code, n in profile.most_common(int(disc.get("cpc_groups", 2))):
        if n < 2:
            break
        add(f"cpc={code} and pd>={since:%Y%m%d}{offices}", code, max_hits=max_hits)

    counts: Counter[str] = Counter()
    niche: Counter[str] = Counter()
    for patent in pool.values():
        for name in company_applicants(patent):
            counts[name] += 1
            _add_evidence(evidence, name, patent.title)
            if patent.key in niche_keys:
                niche[name] += 1
    return counts, niche, hits, pair_hits


# --- relevance check (LLM) -----------------------------------------------------


def screen_batches(lines: list[str], max_tokens: int, overhead: str) -> list[list[int]]:
    """Group candidate lines (by index) so each request stays within max_tokens."""
    batches: list[list[int]] = []
    current: list[int] = []
    size = estimate_tokens(overhead)
    for i, line in enumerate(lines):
        cost = estimate_tokens(line)
        if current and size + cost > max_tokens:
            batches.append(current)
            current, size = [], estimate_tokens(overhead)
        current.append(i)
        size += cost
    if current:
        batches.append(current)
    return batches


def screen_candidates(
    config: Config, rivals: list[Rival], evidence: dict[str, list[str]]
) -> tuple[list[Rival], list[Rival], bool]:
    """(kept, dropped, whether the LLM ran). Never raises; keeps all on failure."""
    if not rivals or not config.discovery.get("relevance_check", True):
        return rivals, [], False
    target = config.target
    header = (f"Startup: {target.get('name') or '(unnamed)'}\n"
              f"What it does: {target.get('description') or '(not given)'}\n\nCandidates:\n")
    lines = []
    for n, rival in enumerate(rivals, 1):
        titles = "; ".join(t[:120] for t in _evidence_for(rival.name, evidence)) or "(no titles)"
        lines.append(f"{n}. {rival.name} -- matched: {titles}")

    budget = int(config.llm.get("screen_batch_tokens", 2500))
    keep_idx: set[int] = set()
    ran = False
    for batch in screen_batches(lines, budget, SCREEN_PROMPT + header):
        user = header + "\n".join(lines[i] for i in batch)
        try:
            data = json.loads(complete_json(config, SCREEN_PROMPT, user))
            wanted = {int(n) for n in data.get("keep", []) if str(n).strip().isdigit()}
            keep_idx.update(i for i in batch if i + 1 in wanted)
            ran = True
        except Exception as exc:  # noqa: BLE001 - LLM optional: keep this batch
            log.warning("discovery: relevance check unavailable, keeping %d candidates (%s)",
                        len(batch), exc)
            keep_idx.update(batch)
    kept = [r for i, r in enumerate(rivals) if i in keep_idx]
    dropped = [r for i, r in enumerate(rivals) if i not in keep_idx]
    return kept, dropped, ran


# --- entry point ---------------------------------------------------------------


def discover(
    config: Config,
    rivals: list[Rival],
    signals: SignalBundle,
    *,
    known: list[Rival] | None = None,
) -> DiscoveryResult:
    """Propose new rivals around the seeds and the target. Never raises.

    ``rivals`` are the seed anchors; ``known`` (default: ``rivals``) is
    everything already tracked, which is never proposed again.
    """
    disc = config.discovery
    known = known if known is not None else rivals
    since = date.today() - timedelta(days=365 * int(disc.get("topic_years", 2)))
    target_name = (config.target.get("name") or "").strip()

    terms = search_terms(config)
    phrases, phrase_source = terms.phrases, terms.source
    if phrases:
        log.info("discovery: search phrases (%s): %s", phrase_source, "; ".join(phrases))
    if terms.technology_cpc or terms.market_cpc:
        log.info("discovery: CPC technology %s x market %s", terms.technology_cpc, terms.market_cpc)

    anchors = list(rivals)
    exclude_keys = {r.key for r in known}
    paper_counts: Counter[tuple[str, str]] = Counter()
    paper_hits: dict[str, int] = {}
    patent_hits: dict[str, int] = {}
    pair_hits: dict[str, int | None] = {}
    niche_counts: Counter[str] = Counter()
    evidence: dict[str, list[str]] = {}
    if config.source_enabled("openalex"):
        try:
            target = _resolve_target(config)
            if target:
                anchors.append(target)
                exclude_keys.add(target.key)
        except Exception as exc:  # noqa: BLE001
            log.info("discovery: target resolution failed (%s)", exc)
        if phrases:
            found, paper_hits = _topic_papers(config, phrases, since, evidence=evidence)
            paper_counts += found
    try:
        paper_counts += _openalex_cooccurrence(config, anchors, evidence=evidence)
    except Exception as exc:  # noqa: BLE001
        log.warning("discovery: co-occurrence stage failed: %s", exc)

    patent_counts: Counter[str] = Counter()
    if config.source_enabled("epo") and config.secrets.epo_ops_key and config.secrets.epo_ops_secret:
        try:
            patent_counts, niche_counts, patent_hits, pair_hits = _topic_patents(
                config, terms, since, list(signals.patents), evidence=evidence)
        except Exception as exc:  # noqa: BLE001
            log.warning("discovery: patent stage failed: %s", exc)

    kept, niche_by_key = qualifying(paper_counts, patent_counts, niche_counts,
                                    min_cooccurrence=int(disc.get("min_cooccurrence", 2)))
    ranked = rank_candidates(
        kept,
        exclude_keys,
        min_cooccurrence=1,
        max_new=int(disc.get("max_candidates", 40)),
        known_names=[r.name for r in known] + [target_name],
        priority=niche_by_key,
    )
    hits = {p: {"papers": paper_hits.get(p), "patents": patent_hits.get(p)} for p in phrases}
    for phrase, h in hits.items():
        log.info("discovery: %r matched %s papers, %s patents", phrase, h["papers"], h["patents"])
    for pair, n in pair_hits.items():
        log.info("discovery: CPC %s matched %s patents", pair, n)
    kept_rivals, dropped, screened = screen_candidates(config, ranked, evidence)
    if screened:
        log.info("discovery: relevance check dropped %d of %d: %s", len(dropped), len(ranked),
                 ", ".join(r.name for r in dropped))
    log.info("discovery: %d niche candidates; proposing %d new rivals",
             sum(1 for n in niche_by_key.values() if n), len(kept_rivals))
    return DiscoveryResult(rivals=kept_rivals, phrases=phrases, phrase_source=phrase_source,
                           phrase_hits=hits, cpc_pair_hits=pair_hits,
                           screened=screened, screened_out=[r.name for r in dropped])
