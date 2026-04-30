"""Shared workspace tree helpers for session drivers."""

from __future__ import annotations

from collections.abc import Collection
from difflib import unified_diff
from pathlib import Path

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".claude-driver-config",
    ".claude-driver-transcripts",
    ".codex-driver-home",
    ".codex-driver-transcripts",
}
DEFAULT_EXCLUDE_FILES = {
    ".claude-stop-hook.log",
    ".claude-task-stopped",
}


def read_workspace_tree(
    root: Path,
    exclude_dirs: Collection[str] | None = None,
    exclude_files: Collection[str] | None = None,
) -> dict[str, str]:
    ignored_roots = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
    ignored_files = set(exclude_files or DEFAULT_EXCLUDE_FILES)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in ignored_roots for part in path.relative_to(root).parts):
            continue
        if path.name in ignored_files:
            continue
        try:
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            continue
    return files


def changed_paths_from_tree(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    return sorted(changed)


def diff_patch_from_tree(before: dict[str, str], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for path in changed_paths_from_tree(before, after):
        before_lines = before.get(path, "").splitlines(keepends=True)
        after_lines = after.get(path, "").splitlines(keepends=True)
        chunks.extend(
            unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(chunks)
