"""Lineage-DB snapshots -- the state half of an Eve checkpoint.

Each iteration boundary freezes the solver/optimizer lineage DBs into
``.snapshots/<db>_iter_<N>.db``; ``resume.py`` rolls these back when resuming and
``checkpoint.json`` (see ``resume.py``) is the manifest that indexes them.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

SNAPSHOTS_SUBDIR = ".snapshots"
SOLVER_LINEAGE_DB = "solver_lineage.db"
OPTIMIZER_LINEAGE_DB = "optimizer_lineage.db"


@dataclass(frozen=True)
class SnapshotFamily:
    """One lineage DB snapshot family."""

    live_name: str
    prefix: str

    def path(self, snapshot_root: Path, iteration: int) -> Path:
        return snapshot_root / f"{self.prefix}{iteration}.db"

    def parse_iteration(self, path: Path) -> int | None:
        stem = path.stem
        if not stem.startswith(self.prefix):
            return None
        try:
            return int(stem.removeprefix(self.prefix))
        except ValueError:
            return None


SNAPSHOT_FAMILIES = (
    SnapshotFamily(SOLVER_LINEAGE_DB, "solver_lineage_iter_"),
    SnapshotFamily(OPTIMIZER_LINEAGE_DB, "optimizer_lineage_iter_"),
)


def snapshot_root(run_root: str | Path) -> Path:
    """Return the Eve snapshot root for one run."""

    return Path(run_root) / SNAPSHOTS_SUBDIR


def write_lineage_snapshots(
    *,
    run_root: str | Path,
    solver_db_path: str | Path,
    optimizer_db_path: str | Path,
    anchor_iteration: int,
) -> None:
    """Write both lineage DB snapshots for an iteration boundary."""

    root = snapshot_root(run_root)
    root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        SOLVER_LINEAGE_DB: Path(solver_db_path),
        OPTIMIZER_LINEAGE_DB: Path(optimizer_db_path),
    }
    for family in SNAPSHOT_FAMILIES:
        snapshot_path = family.path(root, anchor_iteration)
        if snapshot_path.exists():
            continue
        _backup_sqlite_db(source_paths[family.live_name], snapshot_path)


def rollback_lineage_dbs(run_root: str | Path, *, anchor_iteration: int) -> None:
    """Overwrite live lineage DBs with the snapshots for ``anchor_iteration``."""

    root = Path(run_root)
    snapshots = snapshot_root(root)
    missing = [
        family.path(snapshots, anchor_iteration)
        for family in SNAPSHOT_FAMILIES
        if not family.path(snapshots, anchor_iteration).is_file()
    ]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing snapshot(s) for resume rollback: {names}")
    for family in SNAPSHOT_FAMILIES:
        snapshot_path = family.path(snapshots, anchor_iteration)
        live_path = root / family.live_name
        temp_path = live_path.with_suffix(f"{live_path.suffix}.resume_tmp")
        shutil.copy2(snapshot_path, temp_path)
        temp_path.replace(live_path)
        _LOGGER.info("Rolled %s back to %s.", family.live_name, snapshot_path.name)


def _sorted_family_snapshots(
    root: Path,
    family: SnapshotFamily,
) -> list[tuple[int, Path]]:
    snapshots: list[tuple[int, Path]] = []
    for path in root.glob(f"{family.prefix}*.db"):
        iteration = family.parse_iteration(path)
        if iteration is None:
            continue
        snapshots.append((iteration, path))
    return sorted(snapshots, key=lambda item: item[0])


def _backup_sqlite_db(source_path: Path, dest_path: Path) -> None:
    if not source_path.exists():
        return
    temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp")
    source_conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    dest_conn = sqlite3.connect(temp_path)
    try:
        source_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        source_conn.close()
    temp_path.replace(dest_path)
