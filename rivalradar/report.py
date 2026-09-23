"""The weekly "What Moved" brief: a PDF of at most ``output.max_pages`` pages.

Sections follow the spec: rivals moved, new publications, new patents, hiring
signals. "Rivals moved" summarises the whole diff per rival; the other sections
show only what the relevance pass kept, in its order. If the result runs past
the page limit, the lowest-ranked item is dropped and the PDF rebuilt until it
fits — ruthless prioritisation, as the spec asks.
"""

from __future__ import annotations

import io
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import Config
from .diff import WeeklyDiff
from .relevance import RankedItem, RelevanceResult

log = logging.getLogger("rivalradar.report")

# Report sections, in spec order: (heading, item kinds shown).
SECTIONS = [
    ("New publications", {"publication"}),
    ("New patents", {"patent"}),
    ("Hiring signals", {"hire", "role"}),
]
KIND_LABEL = {"publication": "paper", "patent": "patent", "hire": "hire", "role": "open role"}
MAX_RIVAL_ROWS = 10  # the summary table must not crowd out the ranked items

INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#5b636e")
RULE = colors.HexColor("#d0d7de")

# Unicode TTFs (author names, paper titles) — Helvetica only covers Latin-1.
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf"),
]


@dataclass
class ReportInfo:
    path: Path
    pages: int
    shown: int  # ranked items that made it into the PDF
    dropped: int  # ranked items cut to fit the page limit


# --- fonts & styles ------------------------------------------------------


def _fonts() -> tuple[str, str, str]:
    """Register a Unicode font family if one is installed; else Helvetica."""
    for regular, bold, italic in FONT_CANDIDATES:
        if all(Path(p).exists() for p in (regular, bold, italic)):
            try:
                if "RR" not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont("RR", regular))
                    pdfmetrics.registerFont(TTFont("RR-Bold", bold))
                    pdfmetrics.registerFont(TTFont("RR-Italic", italic))
                    pdfmetrics.registerFontFamily("RR", normal="RR", bold="RR-Bold", italic="RR-Italic")
                return "RR", "RR-Bold", "RR-Italic"
            except Exception as exc:  # noqa: BLE001 - fall back to built-ins
                log.info("report: could not load %s (%s)", regular, exc)
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold, italic = _fonts()
    return {
        "title": ParagraphStyle("title", fontName=bold, fontSize=16, leading=20, textColor=INK),
        "meta": ParagraphStyle("meta", fontName=regular, fontSize=8.5, leading=11, textColor=MUTED),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=11, leading=14, textColor=INK,
                             spaceBefore=8, spaceAfter=3),
        "item": ParagraphStyle("item", fontName=regular, fontSize=9, leading=11.5, textColor=INK),
        "why": ParagraphStyle("why", fontName=italic, fontSize=8.5, leading=10.5, textColor=INK,
                              leftIndent=8),
        "detail": ParagraphStyle("detail", fontName=regular, fontSize=7.5, leading=9.5,
                                 textColor=MUTED, leftIndent=8, spaceAfter=4),
        "cell": ParagraphStyle("cell", fontName=regular, fontSize=8.5, leading=10.5, textColor=INK),
        "note": ParagraphStyle("note", fontName=italic, fontSize=7.5, leading=9.5, textColor=MUTED,
                               spaceBefore=8),
    }


# --- content ---------------------------------------------------------------


def _esc(text: str) -> str:
    return escape(text or "")


def _link(text: str, url: str) -> str:
    if not url:
        return _esc(text)
    return f'<a href="{escape(url, {chr(34): "&quot;"})}" color="#1a4f8b">{_esc(text)}</a>'


def _summary_line(diff: WeeklyDiff, result: RelevanceResult, shown: int) -> str:
    rivals = len({i.rival_key for i in diff.items if i.kind != "new_rival"})
    new = sum(1 for i in diff.items if i.kind == "new_rival")
    parts = [f"{result.total} change{'s' if result.total != 1 else ''} across "
             f"{rivals} rival{'s' if rivals != 1 else ''}; top {shown} shown."]
    if new:
        parts.append(f"{new} rival{'s' if new != 1 else ''} newly tracked.")
    if result.llm_used:
        parts.append("Ranked by LLM relevance to your startup.")
    else:
        parts.append("LLM ranking unavailable this run: ordered by type and recency.")
    if not diff.baseline:
        parts.append("First run: no previous snapshot, so everything counts as new.")
    return " ".join(parts)


def _rivals_moved(diff: WeeklyDiff, ranked: list[RankedItem], s: dict) -> list:
    """Per-rival activity table over the whole diff, plus newly tracked rivals."""
    counts: dict[str, Counter] = defaultdict(Counter)
    names: dict[str, str] = {}
    for item in diff.items:
        if item.kind == "new_rival":
            continue
        counts[item.rival_key][item.kind] += 1
        names[item.rival_key] = item.rival_name
    top: dict[str, str] = {}
    for r in ranked:  # first ranked item per rival = its headline
        if r.item.kind != "new_rival":
            top.setdefault(r.item.rival_key, r.item.title)

    flow: list = [Paragraph("Rivals moved", s["h2"])]
    if counts:
        order = sorted(counts, key=lambda k: (k not in top, -sum(counts[k].values())))
        rows = [[Paragraph("<b>Rival</b>", s["cell"]), Paragraph("<b>Activity</b>", s["cell"]),
                 Paragraph("<b>Headline</b>", s["cell"])]]
        for key in order[:MAX_RIVAL_ROWS]:
            activity = ", ".join(
                f"{n} {KIND_LABEL.get(kind, kind)}{'s' if n != 1 else ''}"
                for kind, n in counts[key].most_common()
            )
            rows.append([Paragraph(_esc(names[key]), s["cell"]), Paragraph(activity, s["cell"]),
                         Paragraph(_esc(top.get(key, "—")), s["cell"])])
        table = Table(rows, colWidths=[38 * mm, 42 * mm, None], repeatRows=1)
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        flow.append(table)
        if len(order) > MAX_RIVAL_ROWS:
            flow.append(Paragraph(f"+ {len(order) - MAX_RIVAL_ROWS} more rivals with minor activity.", s["detail"]))
    else:
        flow.append(Paragraph("No rival activity this week.", s["item"]))

    new = [i.rival_name for i in diff.items if i.kind == "new_rival"]
    if new:
        flow.append(Spacer(1, 3))
        flow.append(Paragraph("<b>Newly tracked:</b> " + _esc(", ".join(new)), s["item"]))
    return flow


def _item_flow(r: RankedItem, s: dict, llm_used: bool) -> list:
    item = r.item
    meta = [_esc(item.rival_name)]
    if item.when:
        meta.append(item.when.isoformat())
    if llm_used:
        meta.append(f"relevance {r.score:.0f}/10")
    flow = [Paragraph(f"<b>{_link(item.title, item.url)}</b> — {' · '.join(meta)}", s["item"])]
    if r.why:
        flow.append(Paragraph(_esc(r.why), s["why"]))
    flow.append(Paragraph(_esc(item.detail), s["detail"]) if item.detail else Spacer(1, 4))
    return flow


def _story(config: Config, diff: WeeklyDiff, result: RelevanceResult,
           ranked: list[RankedItem], today: date) -> list:
    s = _styles()
    target = config.target.get("name") or "your startup"
    story: list = [
        Paragraph(f"What Moved — {_esc(target)}", s["title"]),
        Paragraph(f"Week of {today:%d %b %Y}. {_esc(_summary_line(diff, result, len(ranked)))}", s["meta"]),
        Spacer(1, 4),
    ]
    story += _rivals_moved(diff, ranked, s)

    for heading, kinds in SECTIONS:
        items = [r for r in ranked if r.item.kind in kinds]
        if not items:
            continue
        story.append(Paragraph(heading, s["h2"]))
        for r in items:
            story += _item_flow(r, s, result.llm_used)

    if not ranked:
        story.append(Paragraph("Quiet week: nothing new cleared the bar.", s["item"]))
    if any(r.item.kind == "hire" for r in ranked):
        story.append(Paragraph(
            "Hiring signals are inferred from OpenAlex affiliation histories, which sometimes "
            "merge different people with the same name. Verify before acting on one.", s["note"]))
    return story


# --- rendering -------------------------------------------------------------


def _render(config: Config, diff: WeeklyDiff, result: RelevanceResult,
            ranked: list[RankedItem], today: date) -> tuple[bytes, int]:
    """Render to bytes and return (pdf, page count)."""
    buf = io.BytesIO()
    regular = _fonts()[0]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(15 * mm, 9 * mm, f"Rival Radar · generated {today.isoformat()}")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=15 * mm,
        title=f"What Moved — {config.target.get('name') or ''}", author="Rival Radar",
    )
    doc.build(_story(config, diff, result, ranked, today), onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue(), doc.page


def build_report(
    config: Config,
    diff: WeeklyDiff,
    result: RelevanceResult,
    path: str | Path,
    *,
    today: date | None = None,
) -> ReportInfo:
    """Write the brief to ``path``, cutting lowest-ranked items to fit the limit."""
    today = today or date.today()
    max_pages = int(config.output.get("max_pages", 2))
    ranked = list(result.ranked)
    dropped = 0

    pdf, pages = _render(config, diff, result, ranked, today)
    while pages > max_pages and ranked:
        ranked.pop()  # lowest-ranked goes first
        dropped += 1
        pdf, pages = _render(config, diff, result, ranked, today)
    if pages > max_pages:
        log.warning("report: %d pages even with no ranked items (limit %d)", pages, max_pages)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pdf)
    return ReportInfo(path=out, pages=pages, shown=len(ranked), dropped=dropped)
