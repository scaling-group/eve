"""Same-run resume for interrupted Eve runs.

A *checkpoint* is the resumable savepoint written at each iteration boundary. It
has two parts:

- ``checkpoint.json`` (this module): the manifest -- run id, last completed
  iteration, and max iterations.
- the iteration's lineage *snapshots* (see ``snapshots.py``): frozen copies of
  the solver/optimizer lineage DBs to roll back to.

``prepare_resume`` reads the manifest, rolls the lineage DBs back to the selected
completed iteration's snapshot, archives later active workspaces, and lets the
run continue under its original run id.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from scaling_evolve.algorithms.eve.runtime.snapshots import rollback_lineage_dbs

_LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"
RESUME_ARCHIVE_SUBDIR = ".resume_archive"
_WORKSPACE_SUBDIRS = ("solver_workspaces", "evaluation_workspaces")
_ArchiveItem = tuple[str, int, Path, Path]


class ResumeError(RuntimeError):
    """Raised when a run directory cannot be resumed."""


@dataclass(frozen=True)
class EveCheckpoint:
    """Minimal Eve state persisted at an iteration boundary."""

    run_id: str
    last_completed_iteration: int
    max_iterations: int
    schema_version: int = 1


@dataclass(frozen=True)
class ResumePlan:
    """How the runner should continue an interrupted Eve run."""

    run_root: Path
    run_id: str
    start_iteration: int
    checkpoint: EveCheckpoint


def checkpoint_path(run_root: str | Path) -> Path:
    """Return the checkpoint path for one Eve run root."""

    return Path(run_root) / CHECKPOINT_FILE


def write_checkpoint(run_root: str | Path, checkpoint: EveCheckpoint) -> None:
    """Write ``checkpoint.json`` atomically."""

    target = checkpoint_path(run_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(f"{target.suffix}.tmp")
    temp_path.write_text(
        f"{json.dumps(asdict(checkpoint), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temp_path.replace(target)


def read_checkpoint(run_root: str | Path) -> EveCheckpoint | None:
    """Read ``checkpoint.json``; return ``None`` when absent or malformed."""

    path = checkpoint_path(run_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        checkpoint = EveCheckpoint(
            run_id=str(data["run_id"]),
            last_completed_iteration=int(data["last_completed_iteration"]),
            max_iterations=int(data["max_iterations"]),
            schema_version=int(data.get("schema_version", 1)),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if checkpoint.schema_version != 1:
        return None
    return checkpoint


def prepare_resume(run_root: str | Path, resume_iteration: int | None = None) -> ResumePlan:
    """Rollback databases and archive partial workspaces for an interrupted run."""

    root = Path(run_root).resolve()
    checkpoint = read_checkpoint(root)
    if checkpoint is None:
        raise ResumeError(f"{root} has no {CHECKPOINT_FILE}; this run is not resumable.")
    last_completed = checkpoint.last_completed_iteration
    if last_completed < 0:
        raise ResumeError(f"{root}: invalid last_completed_iteration {last_completed}.")
    if resume_iteration is None and last_completed >= checkpoint.max_iterations:
        raise ResumeError(f"{root} already completed all {checkpoint.max_iterations} iterations.")
    anchor_iteration = last_completed if resume_iteration is None else resume_iteration
    if anchor_iteration < 0:
        raise ResumeError(f"{root}: resume_iteration must be non-negative.")
    if anchor_iteration > last_completed:
        raise ResumeError(
            f"{root}: resume_iteration {anchor_iteration} is after the last completed "
            f"iteration {last_completed}."
        )
    if anchor_iteration >= checkpoint.max_iterations:
        raise ResumeError(
            f"{root}: resume_iteration {anchor_iteration} must be less than "
            f"max_iterations {checkpoint.max_iterations}."
        )

    try:
        rollback_lineage_dbs(root, anchor_iteration=anchor_iteration)
    except FileNotFoundError as exc:
        raise ResumeError(str(exc)) from exc
    _archive_iterations_after_anchor(
        root,
        checkpoint=checkpoint,
        anchor_iteration=anchor_iteration,
    )

    _LOGGER.info(
        "Prepared Eve resume of %s: run_id=%s start_iteration=%d",
        root,
        checkpoint.run_id,
        anchor_iteration,
    )
    return ResumePlan(
        run_root=root,
        run_id=checkpoint.run_id,
        start_iteration=anchor_iteration,
        checkpoint=checkpoint,
    )


def _archive_iterations_after_anchor(
    run_root: Path,
    *,
    checkpoint: EveCheckpoint,
    anchor_iteration: int,
) -> None:
    archive_items = list(_collect_post_anchor_paths(run_root, anchor_iteration=anchor_iteration))
    if not archive_items:
        return

    archive_root = _next_resume_archive_root(run_root, anchor_iteration=anchor_iteration)
    moved_paths: list[dict[str, object]] = []
    for kind, step, source, archive_relative in archive_items:
        destination = archive_root / archive_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved_paths.append(
            {
                "kind": kind,
                "iteration": step,
                "source": str(source.relative_to(run_root)),
                "archive": str(archive_relative),
            }
        )
        _LOGGER.info("Archived post-anchor %s %s -> %s.", kind, source.name, destination)

    manifest = {
        "schema_version": 1,
        "run_id": checkpoint.run_id,
        "anchor_iteration": anchor_iteration,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "moved_paths": moved_paths,
    }
    (archive_root / "manifest.json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _collect_post_anchor_paths(
    run_root: Path,
    *,
    anchor_iteration: int,
) -> Iterator[_ArchiveItem]:
    for step_dir in run_root.glob("step_*"):
        step = _parse_step_dir(step_dir.name)
        if step is not None and step > anchor_iteration and step_dir.is_dir():
            yield ("step_dir", step, step_dir, Path(step_dir.name))

    for subdir in _WORKSPACE_SUBDIRS:
        workspace_root = run_root / subdir
        if not workspace_root.is_dir():
            continue
        for workspace in workspace_root.iterdir():
            step = _parse_workspace_step(workspace.name)
            if step is not None and step > anchor_iteration and workspace.is_dir():
                yield (subdir.removesuffix("s"), step, workspace, Path(subdir) / workspace.name)


def _next_resume_archive_root(run_root: Path, *, anchor_iteration: int) -> Path:
    archive_parent = run_root / RESUME_ARCHIVE_SUBDIR
    archive_parent.mkdir(parents=True, exist_ok=True)
    next_index = 1
    for path in archive_parent.iterdir():
        if not path.is_dir() or not path.name.startswith("resume_"):
            continue
        token = path.name.removeprefix("resume_").split("__", 1)[0]
        if not token.isdigit():
            continue
        next_index = max(next_index, int(token) + 1)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return archive_parent / f"resume_{next_index:04d}__anchor_{anchor_iteration}__{timestamp}"


def _parse_step_dir(name: str) -> int | None:
    if not name.startswith("step_"):
        return None
    try:
        return int(name.removeprefix("step_"))
    except ValueError:
        return None


def _parse_workspace_step(name: str) -> int | None:
    marker = "_step_"
    if marker not in name:
        return None
    token = name.split(marker, 1)[1].split("_", 1)[0]
    try:
        return int(token)
    except ValueError:
        return None
