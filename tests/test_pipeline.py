"""Report rendering and the end-to-end weekly run, offline.

Sources and discovery are replaced with fakes; everything else (snapshot,
database, diff, fallback ranking, PDF) runs for real.
"""

import json
from collections import Counter
from datetime import date

import pytest

from rivalradar import cli, discovery, relevance
from rivalradar.config import Config, Secrets
from rivalradar.diff import ChangeItem, WeeklyDiff
from rivalradar.models import Affiliation, Patent, Publication, SignalBundle
from rivalradar.relevance import RankedItem, RelevanceResult
from rivalradar.report import build_report
from rivalradar.sources import registry

TODAY = date(2026, 9, 23)


def _config(tmp_path, **raw):
    base = {
        "target": {"name": "Example AI", "description": "Small on-device LLMs."},
        "seed_rivals": [{"name": "Rival One", "openalex_institution_id": "I1"}],
        "sources": {"fake": True},
        "output": {"dir": "out", "max_pages": 2},
        "storage": {"db_path": "rr.sqlite", "snapshot_dir": "snaps"},
    }
    base.update(raw)
    return Config(raw=base, path=tmp_path / "config.yaml", secrets=Secrets())


# --- report ----------------------------------------------------------------


def _item(n, kind="publication", title=None):
    return ChangeItem(id=f"{kind}:{n}", kind=kind, rival_key="I1", rival_name="Rival One",
                      title=title or f"Item {n}", detail="x " * 80, when=TODAY, url=f"https://ex.com/{n}")


def test_report_escapes_markup_in_titles(tmp_path):
    item = _item(1, title='Attention <is> all you need & "more"')
    diff = WeeklyDiff(baseline=True, items=[item])
    res = RelevanceResult(ranked=[RankedItem(item, 9, "why <b>")], llm_used=True, considered=1, total=1)
    info = build_report(_config(tmp_path), diff, res, tmp_path / "r.pdf", today=TODAY)
    assert info.path.read_bytes().startswith(b"%PDF") and info.pages == 1


def test_report_cuts_lowest_ranked_items_to_fit(tmp_path):
    items = [_item(n) for n in range(60)]
    diff = WeeklyDiff(baseline=True, items=items)
    res = RelevanceResult(ranked=[RankedItem(i, 10 - n / 10, "reason " * 20) for n, i in enumerate(items)],
                          llm_used=True, considered=60, total=60)
    cfg = _config(tmp_path, output={"dir": "out", "max_pages": 1})
    info = build_report(cfg, diff, res, tmp_path / "r.pdf", today=TODAY)
    assert info.pages == 1 and info.dropped > 0 and info.shown + info.dropped == 60


def test_report_quiet_week(tmp_path):
    info = build_report(_config(tmp_path), WeeklyDiff(baseline=True), RelevanceResult(), tmp_path / "r.pdf", today=TODAY)
    assert info.pages == 1 and info.shown == 0


# --- end-to-end ------------------------------------------------------------


@pytest.fixture
def fake_world(monkeypatch):
    """A controllable rival world: tests mutate ``world`` between runs."""
    world = {
        "pubs": [Publication("Paper A", "I1", "openalex", "W1", published=TODAY)],
        "patents": [Patent("Widget", "I1", "EP1", published=TODAY)],
        "affs": [Affiliation("A1", "Jane", "I1", "Rival One", first_year=2026, last_year=2026),
                 Affiliation("A1", "Jane", "I9", "Oxford", first_year=2020, last_year=2025)],
        "cooccur": Counter({("I77", "TinyCo"): 3}),
        "discovery_bases": [],
    }
    def fake_source(config, rivals, context):
        return SignalBundle(publications=list(world["pubs"]), patents=list(world["patents"]),
                            affiliations=list(world["affs"]))
    monkeypatch.setattr(registry, "SOURCES", {"fake": fake_source})
    def fake_cooccurrence(config, rivals, evidence=None):
        world["discovery_bases"].append(sorted(r.name for r in rivals))
        return world["cooccur"]
    monkeypatch.setattr(discovery, "_openalex_cooccurrence", fake_cooccurrence)
    return world


def test_weekly_runs_end_to_end(tmp_path, fake_world):
    cfg = _config(tmp_path)

    # Week 1: no baseline, everything new, TinyCo discovered.
    s1 = cli.run(cfg, today=TODAY)
    assert not s1.diff.baseline
    assert [r.name for r in s1.discovered] == ["TinyCo"]
    assert sorted(i.kind for i in s1.diff.items) == ["hire", "new_rival", "patent", "publication"]
    assert [i.title for i in s1.diff.items if i.kind == "new_rival"] == ["TinyCo added to tracked rivals"]
    assert s1.report.shown == 3  # every rankable item fits; new rivals are listed, not ranked
    assert s1.report.path.exists() and 1 <= s1.report.pages <= 2
    assert (tmp_path / "rr.sqlite").exists() and s1.snapshot.parent == tmp_path / "snaps"
    assert not s1.relevance.llm_used  # no key in tests -> fallback

    # Week 2: nothing changed -> quiet; TinyCo is now tracked from the database.
    s2 = cli.run(cfg, today=TODAY)
    assert s2.diff.baseline and s2.diff.items == []
    assert sorted(r.name for r in s2.rivals) == ["Rival One", "TinyCo"]  # tracked once
    assert s2.discovered == []  # already tracked, not rediscovered
    # Discovery expands from seeds only, never from rivals it discovered.
    assert fake_world["discovery_bases"] == [["Rival One"], ["Rival One"]]

    # Week 3: one new paper.
    fake_world["pubs"].append(Publication("Paper B", "I1", "openalex", "W2", published=TODAY))
    s3 = cli.run(cfg, today=TODAY)
    assert [i.title for i in s3.diff.items] == ["Paper B"]


def test_run_uses_llm_ranking_when_available(tmp_path, fake_world, monkeypatch):
    def fake_groq(config, prompt):
        assert "Small on-device LLMs." in prompt
        item_id = json.loads(prompt.splitlines()[-1])["id"]
        return json.dumps({"ranked": [{"id": item_id, "score": 9, "why": "Directly competes."}]})
    monkeypatch.setattr(relevance, "_ask_groq", fake_groq)
    s = cli.run(_config(tmp_path), today=TODAY, do_discovery=False)
    assert s.relevance.llm_used and len(s.relevance.ranked) == 1
    assert s.relevance.ranked[0].why == "Directly competes."


def test_discovery_works_with_no_seed_rivals(tmp_path, fake_world):
    s = cli.run(_config(tmp_path, seed_rivals=[]), today=TODAY)
    assert [r.name for r in s.discovered] == ["TinyCo"]
    assert [r.name for r in s.rivals] == ["TinyCo"]


def test_no_seeds_and_no_discovery_is_a_config_error(tmp_path, fake_world):
    with pytest.raises(cli.ConfigError):
        cli.run(_config(tmp_path, seed_rivals=[]), today=TODAY, do_discovery=False)


def test_seed_reuses_resolved_id_and_drops_removed_seeds(tmp_path, fake_world):
    cfg = _config(tmp_path, seed_rivals=[{"name": "Rival One", "openalex_institution_id": "I1"},
                                         {"name": "Gone Co", "openalex_institution_id": "I5"}])
    cli.run(cfg, today=TODAY, do_discovery=False)
    cfg2 = _config(tmp_path, seed_rivals=[{"name": "Rival One"}])  # id left blank now
    s = cli.run(cfg2, today=TODAY, do_discovery=False)
    assert [(r.name, r.openalex_institution_id) for r in s.rivals] == [("Rival One", "I1")]


def test_cli_main_reports_config_errors(tmp_path, capsys):
    assert cli.main(["run", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "Config file not found" in capsys.readouterr().err


# --- target entered per run -----------------------------------------------


def _asker(answers):
    """Stand-in for input(): returns scripted answers, records prompts."""
    asked = []
    def ask(prompt):
        asked.append(prompt)
        return answers.pop(0)
    return ask, asked


def test_target_from_flags_needs_no_prompt():
    ask, asked = _asker([])
    assert cli.ask_target("Acme", "Robots.", interactive=True, ask=ask) == {
        "name": "Acme", "description": "Robots.", "keywords": []}
    assert asked == []


def test_target_prompts_for_missing_fields_and_rejects_blank_name():
    ask, asked = _asker(["", "  Acme  ", "Edge LLMs."])
    assert cli.ask_target(None, None, interactive=True, ask=ask) == {
        "name": "Acme", "description": "Edge LLMs.", "keywords": []}
    assert len(asked) == 3  # blank name asked again


def test_target_prompts_for_description_when_only_name_given():
    ask, asked = _asker(["Edge LLMs."])
    assert cli.ask_target("Acme", None, interactive=True, ask=ask)["description"] == "Edge LLMs."


def test_target_required_without_a_terminal():
    with pytest.raises(cli.ConfigError, match="--target-name"):
        cli.ask_target(None, None, interactive=False)
    # Description is optional: a scheduled run with just a name still goes ahead.
    assert cli.ask_target("Acme", None, interactive=False) == {"name": "Acme", "description": "", "keywords": []}


def test_target_keywords_flag_is_split_on_commas():
    t = cli.ask_target("Acme", "Robots.", interactive=False, keywords=" weld inspection, ,defect detection ")
    assert t["keywords"] == ["weld inspection", "defect detection"]


def test_cli_main_passes_target_into_the_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.yaml").write_text("seed_rivals: [{name: Rival One}]", encoding="utf-8")
    seen = {}
    def fake_run(config, do_discovery):
        seen["target"] = config.target
        raise cli.ConfigError("stop here")
    monkeypatch.setattr(cli, "run", fake_run)
    code = cli.main(["run", "--config", str(tmp_path / "config.yaml"),
                     "--target-name", "Acme", "--target-description", "Edge LLMs."])
    assert code == 2 and seen["target"] == {"name": "Acme", "description": "Edge LLMs.", "keywords": []}


def test_cli_main_without_target_and_terminal_fails_clearly(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("seed_rivals: [{name: Rival One}]", encoding="utf-8")
    assert cli.main(["run", "--config", str(tmp_path / "config.yaml")]) == 2  # pytest stdin is not a tty
    assert "--target-name" in capsys.readouterr().err


def test_cli_main_explains_flags_when_input_runs_out(tmp_path, monkeypatch, capsys):
    """Windows can report a detached stdin as a terminal; the prompt then hits EOF."""
    (tmp_path / "config.yaml").write_text("seed_rivals: [{name: Rival One}]", encoding="utf-8")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert cli.main(["run", "--config", str(tmp_path / "config.yaml")]) == 2
    assert "--target-name" in capsys.readouterr().err
