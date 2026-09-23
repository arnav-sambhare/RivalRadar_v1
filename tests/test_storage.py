"""SQLite and snapshot round-trips."""

from datetime import date, datetime

from rivalradar import snapshot
from rivalradar.db import Database
from rivalradar.models import Affiliation, Patent, Publication, Rival, SignalBundle


def _bundle():
    return SignalBundle(
        rivals=[Rival("Rival One", patent_count=3)],
        publications=[Publication("A paper", "rival one", "arxiv", "2401.00001",
                                  published=date(2026, 9, 15), authors=["X", "Y"])],
        patents=[Patent("A patent", "rival one", "EP123", cpc_codes=["G06N"], applicants=["Rival One"])],
        affiliations=[Affiliation("A1", "X", "I1", "Rival One", observed=date(2026, 9, 10),
                                  first_year=2024, last_year=2026)],
    )


def test_db_upserts_are_idempotent(tmp_path):
    with Database(tmp_path / "t.sqlite") as db:
        db.save_bundle(_bundle())
        db.save_bundle(_bundle())
        assert {t: db.count(t) for t in ["rivals", "publications", "patents", "affiliations"]} == {
            "rivals": 1, "publications": 1, "patents": 1, "affiliations": 1}
        assert (db.get_rivals()[0].name, db.get_rivals()[0].patent_count) == ("Rival One", 3)


def test_db_save_bundle_is_atomic(tmp_path):
    bad = _bundle()
    bad.patents.append(Patent(None, "rival one", "EP999"))  # NOT NULL title -> fails mid-run
    with Database(tmp_path / "t.sqlite") as db:
        try:
            db.save_bundle(bad)
        except Exception:
            pass
        assert db.count("rivals") == 0  # nothing from the failed run was kept


def test_snapshot_round_trip_and_baseline(tmp_path):
    first = snapshot.write_snapshot(_bundle(), tmp_path, when=datetime(2026, 9, 16, 9))
    assert snapshot.latest_snapshot(tmp_path, before=first) is None
    second = snapshot.write_snapshot(SignalBundle(), tmp_path, when=datetime(2026, 9, 23, 9))
    assert snapshot.latest_snapshot(tmp_path) == second
    assert snapshot.latest_snapshot(tmp_path, before=second) == first

    loaded = snapshot.load_snapshot(first)
    assert loaded.publications[0].published == date(2026, 9, 15)
    assert loaded.publications[0].authors == ["X", "Y"]
    a = loaded.affiliations[0]
    assert (a.observed, a.first_year, a.last_year) == (date(2026, 9, 10), 2024, 2026)
    assert loaded.rivals[0].patent_count == 3
