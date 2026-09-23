"""LLM relevance pass: rank what changed by how much it matters to the founder.

The raw diff can hold dozens of items (hundreds on a first run). Groq ranks
them against the target startup and keeps the top ``llm.max_items``, each with a
one-line reason — this drives what makes the 2-page brief. Newly tracked
rivals are not ranked: the brief lists all of them in its summary.

Fail-soft: with no ``GROQ_API_KEY``, or if the call or its output fails, a
deterministic fallback ranks by kind and recency so the run still produces a
brief. ``RelevanceResult.llm_used`` tells the report which happened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from .config import Config
from .diff import ChangeItem
from .llm import complete_json

log = logging.getLogger("rivalradar.relevance")

# Fallback ordering when the LLM is unavailable: moves and IP before papers.
KIND_PRIORITY = {"hire": 0, "patent": 1, "publication": 2, "role": 3}

SYSTEM_PROMPT = """\
You are a competitive-intelligence analyst briefing a startup founder.
You receive a list of changes observed this week among the founder's rivals:
new papers, patents, researcher hires and open roles.
Rank the changes by how much each matters to THIS founder's competitive
position. Favour signals of strategic direction (new capability, new product
area, IP in the founder's space, senior or specialist hires) over routine or
off-topic output. Hire items come from noisy author-disambiguation data: if a
hire looks implausible (e.g. unrelated prior field), score it low.

Reply with JSON only, in exactly this shape:
{"ranked": [{"id": "<item id>", "score": <1-10>, "why": "<one short sentence>"}]}
Include only items scoring 5 or more, highest first. Use ids exactly as given.
"""


@dataclass
class RankedItem:
    item: ChangeItem
    score: float
    why: str = ""


@dataclass
class RelevanceResult:
    ranked: list[RankedItem] = field(default_factory=list)
    llm_used: bool = False
    considered: int = 0  # items sent for ranking
    total: int = 0  # rankable items in the diff


# --- pure helpers ----------------------------------------------------------


def preselect(items: list[ChangeItem], limit: int) -> list[ChangeItem]:
    """Cap what is sent to the LLM: by kind priority, then most recent first."""
    ordered = sorted(
        items,
        key=lambda i: (KIND_PRIORITY.get(i.kind, 9), -(i.when or date.min).toordinal()),
    )
    return ordered[:limit]


def fallback_rank(items: list[ChangeItem], max_items: int) -> list[RankedItem]:
    """Deterministic ranking used when the LLM is unavailable."""
    return [RankedItem(item=i, score=0.0) for i in preselect(items, max_items)]


def build_prompt(target: dict, items: list[ChangeItem]) -> str:
    lines = [
        f"Founder's startup: {target.get('name') or '(unnamed)'}",
        f"What it does: {target.get('description') or '(not given)'}",
        "",
        "Changes this week:",
    ]
    for i in items:
        entry = {"id": i.id, "kind": i.kind, "rival": i.rival_name, "title": i.title}
        if i.detail:
            entry["detail"] = i.detail
        if i.when:
            entry["date"] = i.when.isoformat()
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)


def parse_ranking(text: str, items: list[ChangeItem], max_items: int) -> list[RankedItem]:
    """Validate the LLM's JSON: known ids only, no duplicates, clamped scores."""
    by_id = {i.id: i for i in items}
    data = json.loads(text)
    ranked: list[RankedItem] = []
    seen: set[str] = set()
    for row in data.get("ranked", []):
        item_id = row.get("id") if isinstance(row, dict) else None
        if item_id not in by_id or item_id in seen:
            continue
        seen.add(item_id)
        try:
            score = max(1.0, min(10.0, float(row.get("score", 0))))
        except (TypeError, ValueError):
            continue
        ranked.append(RankedItem(item=by_id[item_id], score=score, why=str(row.get("why", "")).strip()))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:max_items]


# --- LLM call --------------------------------------------------------------


def _ask_groq(config: Config, prompt: str) -> str:
    return complete_json(config, SYSTEM_PROMPT, prompt)


def rank(config: Config, items: list[ChangeItem]) -> RelevanceResult:
    """Rank changes by relevance to the target. Never raises."""
    llm = config.llm
    max_items = int(llm.get("max_items", 12))
    rankable = [i for i in items if i.kind in KIND_PRIORITY]
    candidates = preselect(rankable, int(llm.get("max_input_items", 60)))
    result = RelevanceResult(considered=len(candidates), total=len(rankable))
    if not candidates:
        return result

    try:
        text = _ask_groq(config, build_prompt(config.target, candidates))
        result.ranked = parse_ranking(text, candidates, max_items)
        result.llm_used = True
    except Exception as exc:  # noqa: BLE001 - any failure falls back
        log.warning("relevance: LLM pass unavailable, using fallback order (%s)", exc)
        result.ranked = fallback_rank(candidates, max_items)
    return result
