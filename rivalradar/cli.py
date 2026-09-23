"""Command line entry point: ``python -m rivalradar run``.

One run = collect signals for the tracked rivals, expand the set via discovery,
write the snapshot and database, diff against last week, rank with the LLM,
and write the PDF brief. Scheduling is left to the OS (cron / Task Scheduler);
see README.

The target startup is given per run, not stored in config: pass
``--target-name`` / ``--target-description``, or the command asks for them.
Scheduled runs have no terminal to ask in, so they must pass the flags.
Discovery searches around the target too, so seed rivals are optional.

Relative paths in config (database, snapshots, output) resolve against the
config file's folder, so the command works from any working directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from .config import Config, ConfigError, load_config
from .db import Database
from .diff import WeeklyDiff, compute_diff
from .discovery import DiscoveryResult, discover
from .filters import apply_small_startup_filter
from .models import Rival, SignalBundle
from .names import normalize_org
from .relevance import RelevanceResult, rank
from .report import ReportInfo, build_report
from .snapshot import latest_snapshot, load_snapshot, write_snapshot
from .sources.registry import collect_all

log = logging.getLogger("rivalradar")


@dataclass
class RunSummary:
    rivals: list[Rival]
    discovered: list[Rival]
    excluded: list[Rival]
    discovery: DiscoveryResult
    snapshot: Path
    diff: WeeklyDiff
    relevance: RelevanceResult
    report: ReportInfo


def ask_target(
    name: str | None,
    description: str | None,
    *,
    interactive: bool,
    keywords: str | None = None,
    ask: Callable[[str], str] | None = None,
) -> dict:
    """The target startup for this run, from flags or typed at the prompt.

    Whatever the flags don't supply is asked for when there is a terminal.
    The name is required; an empty description is allowed but makes the
    relevance ranking and discovery generic. ``keywords`` (comma-separated,
    flag only) override the discovery search phrases derived from the
    description.
    """
    ask = ask or input
    name = (name or "").strip()
    description = (description or "").strip()
    if not name and not interactive:
        raise ConfigError(
            "No target company given. Pass --target-name and --target-description "
            "(needed when running without a terminal, e.g. on a schedule)."
        )
    while not name:
        name = ask("Target company name: ").strip()
    if not description and interactive:
        description = ask("What does it do? (1-2 sentences, used to rank relevance): ").strip()
    if not description:
        print("warning: no target description; relevance ranking will be generic.", file=sys.stderr)
    return {
        "name": name,
        "description": description,
        "keywords": [k.strip() for k in (keywords or "").split(",") if k.strip()],
    }


def _resolve(config: Config, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else config.path.parent / p


def load_rivals(config: Config, tracked: list[Rival]) -> list[Rival]:
    """Seeds from config plus rivals discovered on earlier runs.

    A seed reuses the institution id resolved on a previous run when config
    leaves it blank. Seeds removed from config are dropped; discovered rivals
    stay tracked.
    """
    by_name = {normalize_org(t.name): t for t in tracked}
    rivals: dict[str, Rival] = {}
    seed_names: set[str] = set()
    for seed in config.seed_rivals:
        name = (seed.get("name") or "").strip()
        if not name:
            continue
        seed_names.add(normalize_org(name))
        known = by_name.get(normalize_org(name))
        rival = Rival(
            name=name,
            openalex_institution_id=(seed.get("openalex_institution_id") or "")
            or (known.openalex_institution_id if known else ""),
            homepage=(seed.get("homepage") or "") or (known.homepage if known else ""),
            source="seed",
            patent_count=known.patent_count if known else None,
        )
        rivals.setdefault(rival.key, rival)
    for t in tracked:
        if t.source == "discovery" and normalize_org(t.name) not in seed_names:
            rivals.setdefault(t.key, t)
    return list(rivals.values())


def _merge_into(bundle: SignalBundle, extra: SignalBundle) -> None:
    """Add ``extra``'s records to ``bundle``, skipping keys already present."""
    for attr in ("publications", "patents", "affiliations"):
        have = {x.key for x in getattr(bundle, attr)}
        getattr(bundle, attr).extend(x for x in getattr(extra, attr) if x.key not in have)


def run(config: Config, *, do_discovery: bool = True, today: date | None = None) -> RunSummary:
    """Execute one weekly run end to end."""
    today = today or date.today()
    snapshot_dir = _resolve(config, config.snapshot_dir)
    out_dir = _resolve(config, config.output.get("dir", "output"))

    with Database(_resolve(config, config.db_path)) as db:
        rivals = load_rivals(config, db.get_rivals())
        if not rivals and not do_discovery:
            raise ConfigError("No rivals to track: add seed_rivals to the config, or allow discovery.")

        prev_path = latest_snapshot(snapshot_dir)
        previous = load_snapshot(prev_path) if prev_path else None

        bundle = collect_all(config, rivals)

        discovered: list[Rival] = []
        excluded: list[Rival] = []
        found = DiscoveryResult()
        if do_discovery:
            # Expand from the founder's seeds and the target only: discovering
            # from discovered rivals snowballs toward big, well-connected orgs.
            seeds = [r for r in rivals if r.source == "seed"]
            found = discover(config, seeds, bundle, known=rivals)
            discovered, excluded = apply_small_startup_filter(
                config, found.rivals,
                keep_at_most=int(config.discovery.get("max_new_rivals", 25)))
            if discovered:
                # Collect now so this run is their baseline; the diff reports
                # them once as new rivals rather than their history as news.
                _merge_into(bundle, collect_all(config, discovered))
                bundle.rivals.extend(discovered)

        # Baseline for next week, written before anything below can fail.
        snap = write_snapshot(bundle, snapshot_dir)
        db.save_bundle(bundle, now=today.isoformat())

    diff = compute_diff(bundle, previous, today=today, max_join_age_years=config.max_join_age_years)
    relevance = rank(config, diff.items)
    report = build_report(config, diff, relevance, out_dir / f"what-moved-{today.isoformat()}.pdf", today=today)
    return RunSummary(bundle.rivals, discovered, excluded, found, snap, diff, relevance, report)


def _print_summary(s: RunSummary) -> None:
    kinds: dict[str, int] = {}
    for item in s.diff.items:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    changes = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "none"
    print(f"Rivals tracked: {len(s.rivals)} (discovered {len(s.discovered)}, "
          f"excluded as too big {len(s.excluded)})")
    if s.discovery.phrases:
        origin = {"keywords": "from --target-keywords", "llm": "from the description, by the LLM",
                  "description": "the description as-is"}.get(s.discovery.phrase_source, "")
        print(f"  discovery search terms ({origin}): " + "; ".join(s.discovery.phrases))
        if s.discovery.cpc_pair_hits:
            print("  technology+market searches: " + "; ".join(
                f"{pair} ({'failed' if n is None else n})" for pair, n in s.discovery.cpc_pair_hits.items()))
        if s.discovery.screened:
            out = s.discovery.screened_out
            print(f"  relevance check (LLM) dropped {len(out)} candidate(s)"
                  + (": " + ", ".join(out[:8]) + (" ..." if len(out) > 8 else "") if out else ""))
        dead = s.discovery.dead_phrases()
        if dead:
            print("  matched nothing: " + "; ".join(dead)
                  + " (set better terms with --target-keywords)")
    unsized = [r.name for r in s.discovered if r.patent_count is None]
    if unsized:
        print(f"  note: {len(unsized)} discovered rival(s) kept without a patent count "
              "(size filter needs EPO credentials): " + ", ".join(unsized[:8])
              + (" ..." if len(unsized) > 8 else ""))
    print(f"Changes since last run: {changes}" + ("" if s.diff.baseline else " [first run]"))
    print(f"Ranking: {'LLM' if s.relevance.llm_used else 'fallback (no LLM)'}; "
          f"{s.report.shown} item(s) in brief" + (f", {s.report.dropped} cut to fit" if s.report.dropped else ""))
    print(f"Brief: {s.report.path} ({s.report.pages} page{'s' if s.report.pages != 1 else ''})")
    print(f"Snapshot: {s.snapshot}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rivalradar", description="Weekly 'What Moved' competitor brief.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="collect, diff, rank and write this week's brief")
    run_p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    run_p.add_argument("--target-name", help="target startup's name (asked for if omitted)")
    run_p.add_argument("--target-description",
                       help="what the target does, in 1-2 sentences (asked for if omitted)")
    run_p.add_argument("--target-keywords",
                       help="comma-separated search phrases for rival discovery "
                            "(default: derived from the description by the LLM)")
    run_p.add_argument("--no-discover", action="store_true", help="skip rival auto-discovery")
    run_p.add_argument("-v", "--verbose", action="store_true", help="log progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        config = load_config(args.config)
        try:
            config.raw["target"] = ask_target(
                args.target_name, args.target_description,
                interactive=sys.stdin.isatty(), keywords=args.target_keywords,
            )
        except (EOFError, KeyboardInterrupt):
            # EOF also covers unattended runs on Windows, where a detached
            # stdin can still report itself as a terminal.
            raise ConfigError(
                "no target company entered. When running unattended (e.g. on a "
                "schedule), pass --target-name and --target-description."
            ) from None
        summary = run(config, do_discovery=not args.no_discover)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_summary(summary)
    return 0
