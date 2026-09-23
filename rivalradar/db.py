"""SQLite storage for structured records.

Holds the durable, deduplicated view of rivals, publications, patents, and
affiliations. Diffing week-over-week is done from JSON snapshots (see
``snapshot.py``); this database is the accumulating source of truth and backs
things like the small-startup patent filter and rival persistence.

Records are upserted on their stable ``key``. Dates are stored as ISO strings.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

from .models import Affiliation, Patent, Publication, Rival, SignalBundle

SCHEMA = """
CREATE TABLE IF NOT EXISTS rivals (
    key                     TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    openalex_institution_id TEXT DEFAULT '',
    homepage                TEXT DEFAULT '',
    source                  TEXT DEFAULT 'seed',
    patent_count            INTEGER,
    first_seen              TEXT NOT NULL,
    last_seen               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS publications (
    key         TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    rival_key   TEXT NOT NULL,
    source      TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url         TEXT DEFAULT '',
    published   TEXT,
    authors     TEXT DEFAULT '',       -- newline-joined
    abstract    TEXT DEFAULT '',
    first_seen  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patents (
    key         TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    rival_key   TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url         TEXT DEFAULT '',
    published   TEXT,
    cpc_codes   TEXT DEFAULT '',       -- newline-joined
    applicants  TEXT DEFAULT '',       -- newline-joined
    first_seen  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS affiliations (
    key         TEXT PRIMARY KEY,
    author_id   TEXT NOT NULL,
    author_name TEXT NOT NULL,
    org_key     TEXT NOT NULL,
    org_name    TEXT NOT NULL,
    observed    TEXT,
    first_year  INTEGER,
    last_year   INTEGER,
    first_seen  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pub_rival ON publications(rival_key);
CREATE INDEX IF NOT EXISTS idx_pat_rival ON patents(rival_key);
CREATE INDEX IF NOT EXISTS idx_aff_author ON affiliations(author_id);
"""


def _join(values: Iterable[str]) -> str:
    return "\n".join(v for v in values if v)


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


class Database:
    """Thin wrapper over a SQLite connection with typed upserts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._depth = 0
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error. Nested use joins the outer
        transaction, so ``save_bundle`` writes a whole run atomically."""
        if self._depth:
            yield self.conn
            return
        self._depth += 1
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self._depth -= 1

    # --- upserts ---------------------------------------------------------

    def upsert_rival(self, rival: Rival, *, now: str | None = None) -> None:
        now = now or date.today().isoformat()
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO rivals
                    (key, name, openalex_institution_id, homepage, source,
                     patent_count, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    name = excluded.name,
                    openalex_institution_id =
                        COALESCE(NULLIF(excluded.openalex_institution_id, ''),
                                 rivals.openalex_institution_id),
                    homepage = COALESCE(NULLIF(excluded.homepage, ''),
                                        rivals.homepage),
                    patent_count = COALESCE(excluded.patent_count,
                                            rivals.patent_count),
                    last_seen = excluded.last_seen
                """,
                (
                    rival.key,
                    rival.name,
                    rival.openalex_institution_id,
                    rival.homepage,
                    rival.source,
                    rival.patent_count,
                    now,
                    now,
                ),
            )

    def upsert_publication(self, pub: Publication, *, now: str | None = None) -> None:
        now = now or date.today().isoformat()
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO publications
                    (key, title, rival_key, source, external_id, url,
                     published, authors, abstract, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    title = excluded.title,
                    url = COALESCE(NULLIF(excluded.url, ''), publications.url)
                """,
                (
                    pub.key,
                    pub.title,
                    pub.rival_key,
                    pub.source,
                    pub.external_id,
                    pub.url,
                    _iso(pub.published),
                    _join(pub.authors),
                    pub.abstract,
                    now,
                ),
            )

    def upsert_patent(self, patent: Patent, *, now: str | None = None) -> None:
        now = now or date.today().isoformat()
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO patents
                    (key, title, rival_key, external_id, url, published,
                     cpc_codes, applicants, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    title = excluded.title,
                    url = COALESCE(NULLIF(excluded.url, ''), patents.url)
                """,
                (
                    patent.key,
                    patent.title,
                    patent.rival_key,
                    patent.external_id,
                    patent.url,
                    _iso(patent.published),
                    _join(patent.cpc_codes),
                    _join(patent.applicants),
                    now,
                ),
            )

    def upsert_affiliation(self, aff: Affiliation, *, now: str | None = None) -> None:
        now = now or date.today().isoformat()
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO affiliations
                    (key, author_id, author_name, org_key, org_name,
                     observed, first_year, last_year, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    author_name = excluded.author_name,
                    org_name = excluded.org_name,
                    observed = excluded.observed,
                    first_year = COALESCE(excluded.first_year, affiliations.first_year),
                    last_year = COALESCE(excluded.last_year, affiliations.last_year)
                """,
                (
                    aff.key,
                    aff.author_id,
                    aff.author_name,
                    aff.org_key,
                    aff.org_name,
                    _iso(aff.observed),
                    aff.first_year,
                    aff.last_year,
                    now,
                ),
            )

    def save_bundle(self, bundle: SignalBundle, *, now: str | None = None) -> None:
        """Persist an entire run's collected signals in one transaction."""
        now = now or date.today().isoformat()
        with self._tx():
            for rival in bundle.rivals:
                self.upsert_rival(rival, now=now)
            for pub in bundle.publications:
                self.upsert_publication(pub, now=now)
            for patent in bundle.patents:
                self.upsert_patent(patent, now=now)
            for aff in bundle.affiliations:
                self.upsert_affiliation(aff, now=now)

    # --- reads -----------------------------------------------------------

    def get_rivals(self) -> list[Rival]:
        rows = self.conn.execute("SELECT * FROM rivals ORDER BY name").fetchall()
        return [
            Rival(
                name=r["name"],
                openalex_institution_id=r["openalex_institution_id"] or "",
                homepage=r["homepage"] or "",
                source=r["source"] or "seed",
                patent_count=r["patent_count"],
            )
            for r in rows
        ]

    def count(self, table: str) -> int:
        if table not in {"rivals", "publications", "patents", "affiliations"}:
            raise ValueError(f"Unknown table: {table}")
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
