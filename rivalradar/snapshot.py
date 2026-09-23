"""JSON snapshots for week-over-week diffing.

Every run writes one timestamped snapshot of the signals it collected. The next
run reads the most recent prior snapshot as its baseline. Writing a snapshot on
every run is mandatory — without it the following run has nothing to diff
against (per project spec).
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from .models import Affiliation, Patent, Publication, Rival, SignalBundle, parse_iso_date

SNAPSHOT_GLOB = "snapshot-*.json"


def _snapshot_name(when: datetime) -> str:
    return f"snapshot-{when.strftime('%Y%m%dT%H%M%S')}.json"


def write_snapshot(
    bundle: SignalBundle,
    snapshot_dir: str | Path,
    *,
    when: datetime | None = None,
) -> Path:
    """Write ``bundle`` as a timestamped JSON snapshot and return its path."""
    when = when or datetime.now()
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / _snapshot_name(when)
    payload = {
        "created": when.isoformat(),
        "signals": bundle.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def list_snapshots(snapshot_dir: str | Path) -> list[Path]:
    """Return snapshot paths, oldest first (sorted by filename timestamp)."""
    directory = Path(snapshot_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob(SNAPSHOT_GLOB))


def latest_snapshot(
    snapshot_dir: str | Path, *, before: Path | None = None
) -> Path | None:
    """Return the most recent snapshot, optionally excluding ``before``.

    Pass the just-written snapshot as ``before`` to get the prior baseline.
    """
    snapshots = list_snapshots(snapshot_dir)
    if before is not None:
        snapshots = [s for s in snapshots if s != before]
    return snapshots[-1] if snapshots else None


# Fields stored as ISO strings in snapshots and parsed back to dates.
DATE_FIELDS = {"published", "observed"}


def _from_dict(cls: type, data: dict) -> object:
    """Rebuild a model from its snapshot dict, ignoring unknown keys."""
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            value = data[f.name]
            kwargs[f.name] = parse_iso_date(value) if f.name in DATE_FIELDS else value
    return cls(**kwargs)


def load_snapshot(path: str | Path) -> SignalBundle:
    """Load a snapshot file back into a ``SignalBundle``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    signals = data.get("signals", {})
    return SignalBundle(
        rivals=[_from_dict(Rival, r) for r in signals.get("rivals", [])],
        publications=[_from_dict(Publication, p) for p in signals.get("publications", [])],
        patents=[_from_dict(Patent, p) for p in signals.get("patents", [])],
        affiliations=[_from_dict(Affiliation, a) for a in signals.get("affiliations", [])],
    )
