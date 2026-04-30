"""Restore Eve populations from previous run directories.

`restore_from` config accepts a list. Each item may be either:

1. a plain path string, example:
    restore_from:
        - /path/to/old/run

2. a mapping with optional id filters, example:
    restore_from:
        - path: /path/to/old/run
        solver_ids:
            - solver_a1b2c3d4e5f6
            - solver_112233445566
        optimizer_ids:
            - optimizer_abcdef123456

   If `solver_ids` is omitted, restore every solver from that source.
   If `optimizer_ids` is omitted, restore every optimizer from that source.

3. a mixed list of both forms, example:
    restore_from:
        - /path/to/old/run_a
        - path: /path/to/old/run_b
        solver_ids:
            - solver_deadbeefcafe
        - path: /path/to/old/run_c/solver_workspaces/20260406T120000_seed
        optimizer_ids:
            - optimizer_1234abcd5678

Rules:
- `path` may point to the run root or any nested directory inside that run.
- `solver_ids` omitted means restore every solver from that source.
- `optimizer_ids` omitted means restore every optimizer from that source.
- multiple `restore_from` entries are additive and are imported in order.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from scaling_evolve.algorithms.eve.populations.base import Population
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.core.storage.models import ArtifactRef
from scaling_evolve.storage.sqlite import SQLiteLineageStore

_REQUIRED_RUN_FILES = ("solver_lineage.db", "optimizer_lineage.db")
_ENTRY_TABLE = "eve_population_entries"


@dataclass(frozen=True)
class PopulationRestoreResult:
    """Summary for one restored Eve population."""

    entries_restored: int


@dataclass(frozen=True)
class RestoreResult:
    """Summary for one restore operation."""

    source_run_root: Path
    solver: PopulationRestoreResult
    optimizer: PopulationRestoreResult


@dataclass(frozen=True)
class RestoreSpec:
    """One restore source plus optional entry-id filters.

    Fields:
    - `path`: run root or any nested directory inside a previous Eve run
    - `solver_ids`: optional solver ids to import from that source
    - `optimizer_ids`: optional optimizer ids to import from that source

    Omitted or empty `solver_ids` / `optimizer_ids` means "restore all entries
    of that kind from this source".
    """

    path: Path
    solver_ids: tuple[str, ...] = ()
    optimizer_ids: tuple[str, ...] = ()


def restore_populations_from_run(
    source_path: str | Path,
    *,
    solver_population: Population,
    optimizer_population: Population,
    solver_ids: tuple[str, ...] = (),
    optimizer_ids: tuple[str, ...] = (),
) -> RestoreResult:
    """Restore solver and optimizer populations from a previous Eve run.

    `source_path` may point at the run root itself or any nested directory inside it.
    """

    source_run_root = resolve_restore_run_root(source_path)
    solver_entries = _load_population_entries(
        source_run_root / "solver_lineage.db",
        allowed_ids=set(solver_ids),
    )
    optimizer_entries = _load_population_entries(
        source_run_root / "optimizer_lineage.db",
        allowed_ids=set(optimizer_ids),
    )

    for entry in solver_entries:
        solver_population.add(entry)
    for entry in optimizer_entries:
        optimizer_population.add(entry)

    return RestoreResult(
        source_run_root=source_run_root,
        solver=PopulationRestoreResult(entries_restored=len(solver_entries)),
        optimizer=PopulationRestoreResult(entries_restored=len(optimizer_entries)),
    )


def resolve_restore_run_root(source_path: str | Path) -> Path:
    """Resolve an Eve run root from a path inside or at the run directory."""

    candidate = Path(source_path).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for root in (candidate, *candidate.parents):
        if all((root / filename).exists() for filename in _REQUIRED_RUN_FILES):
            return root
    expected = ", ".join(_REQUIRED_RUN_FILES)
    raise FileNotFoundError(
        f"Could not find an Eve run root above {candidate}. Expected: {expected}"
    )


def parse_restore_spec(raw_value: object) -> RestoreSpec:
    """Parse one `restore_from` item from config.

    Accepted forms:

    - `"/path/to/old/run"`
    - `{"path": "/path/to/old/run"}`
    - `{"path": "/path/to/old/run", "solver_ids": ["solver_a"]}`
    - `{"path": "/path/to/old/run", "optimizer_ids": "optimizer_x"}`
    - `{"path": "/path/to/old/run", "solver_ids": [...], "optimizer_ids": [...]}`

    The returned `RestoreSpec` is the normalized internal representation used by
    the runner/factory restore flow. When `solver_ids` or `optimizer_ids` is
    omitted, that side restores the full source population.
    """

    if isinstance(raw_value, str) and raw_value:
        return RestoreSpec(path=Path(raw_value))
    if not isinstance(raw_value, dict):
        raise ValueError("Each `restore_from` entry must be a path string or a mapping.")
    path = raw_value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("Each restore mapping must contain a non-empty `path`.")
    solver_ids = _parse_restore_ids(raw_value.get("solver_ids"), key="solver_ids")
    optimizer_ids = _parse_restore_ids(raw_value.get("optimizer_ids"), key="optimizer_ids")
    return RestoreSpec(
        path=Path(path),
        solver_ids=solver_ids,
        optimizer_ids=optimizer_ids,
    )


def _parse_restore_ids(raw_value: object, *, key: str) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str) and raw_value:
        return (raw_value,)
    if isinstance(raw_value, list):
        parsed: list[str] = []
        for item in raw_value:
            if not isinstance(item, str) or not item:
                raise ValueError(f"`{key}` entries must be non-empty strings.")
            parsed.append(item)
        return tuple(parsed)
    raise ValueError(f"`{key}` must be a string or a list of strings.")


def _load_population_entries(db_path: Path, *, allowed_ids: set[str]) -> list[PopulationEntry]:
    store = SQLiteLineageStore(db_path)
    try:
        run_ids = [
            str(row["run_id"])
            for row in store.connection.execute(
                f"SELECT DISTINCT run_id FROM {_ENTRY_TABLE} ORDER BY run_id ASC"
            ).fetchall()
        ]
        if not run_ids:
            return []
        if len(run_ids) > 1:
            raise ValueError(f"Expected exactly one run in {db_path}, found {len(run_ids)}.")
        rows = store.connection.execute(
            f"""
            SELECT entry_id, files_ref_json, logs_ref_json, score_ref_json
            FROM {_ENTRY_TABLE}
            WHERE run_id = ?
            ORDER BY created_at ASC, entry_id ASC
            """,
            (run_ids[0],),
        ).fetchall()
        return [
            _entry_from_row(row, run_root=db_path.parent)
            for row in rows
            if not allowed_ids or str(row["entry_id"]) in allowed_ids
        ]
    finally:
        store.close()


def _entry_from_row(row: sqlite3.Row, *, run_root: Path) -> PopulationEntry:
    files_ref = ArtifactRef.model_validate_json(str(row["files_ref_json"]))
    logs_ref = ArtifactRef.model_validate_json(str(row["logs_ref_json"]))
    score_ref = ArtifactRef.model_validate_json(str(row["score_ref_json"]))
    return PopulationEntry(
        id=str(row["entry_id"]),
        files=_load_json_tree(files_ref, run_root=run_root),
        score=_load_yaml_value(score_ref, run_root=run_root),
        logs=_load_json_tree(logs_ref, run_root=run_root),
    )


def _load_json_tree(ref: ArtifactRef, *, run_root: Path) -> dict[str, str]:
    path = _resolve_ref_path(ref, run_root=run_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return {str(key): str(value) for key, value in payload.items()}


def _load_yaml_value(ref: ArtifactRef, *, run_root: Path) -> object:
    path = _resolve_ref_path(ref, run_root=run_root)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_ref_path(ref: ArtifactRef, *, run_root: Path) -> Path:
    candidate = ref.uri or ref.location or ref.relpath
    if not isinstance(candidate, str) or not candidate:
        raise ValueError(f"Artifact ref {ref.artifact_id!r} is missing a readable path.")
    path = Path(candidate)
    if not path.exists():
        artifact_root = (
            run_root / "artifacts" if ref.metadata.get("storage_root") != "snapshots" else run_root
        )
        path = artifact_root / candidate
    if not path.exists():
        raise FileNotFoundError(f"Artifact path not found: {path}")
    return path
