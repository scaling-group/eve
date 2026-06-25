"""Workflow-required prompt file loading for Eve."""

from __future__ import annotations

from pathlib import Path


def read_required_prompt_text(prompt_root: Path, relative_path: str) -> str:
    """Read a required workflow prompt file, failing loudly if it is missing."""
    path = prompt_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"workflow-required prompt file missing: {path}. "
            f"EvE requires `{relative_path}` under optimizer.workers.items[].prompt; "
            "deleting or renaming "
            "this file prevents the workflow from running. Restore the file or point "
            "the worker prompt field at a directory that contains it."
        )
    return path.read_text(encoding="utf-8").strip()
