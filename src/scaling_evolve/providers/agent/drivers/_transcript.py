"""Shared transcript archival helpers for session drivers."""

from __future__ import annotations

import shutil
from pathlib import Path


def archive_transcript(
    source: Path,
    dest_root: Path,
    session_id: str,
    timestamp_ns: int | None = None,
) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    if timestamp_ns is None:
        destination = dest_root / source.name
    else:
        suffix = source.suffix or ".jsonl"
        destination = dest_root / f"{session_id}-{timestamp_ns}{suffix}"
    shutil.copy2(source, destination)
    return destination


def archive_subagent_transcripts(
    source_dir: Path,
    dest_root: Path,
    session_id: str,
) -> Path | None:
    if not source_dir.exists():
        return None
    archive_root = dest_root / session_id
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / "subagents"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    return destination
