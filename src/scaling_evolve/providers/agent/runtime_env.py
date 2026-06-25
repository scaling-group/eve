"""Agent runtime environment helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

RUNTIME_DIRNAME = ".agent-runtime"
REPO_ROOT = Path(__file__).resolve().parents[4]


def repo_runtime_bin() -> Path:
    """Return this checkout's repo-local agent runtime bin directory."""

    return REPO_ROOT / RUNTIME_DIRNAME / "bin"


def workspace_runtime_bin(workspace_root: Path | str) -> Path:
    """Return a workspace-local agent runtime bin directory."""

    return Path(workspace_root).expanduser().resolve(strict=False) / RUNTIME_DIRNAME / "bin"


def prepend_runtime_bins(
    env: Mapping[str, str],
    bins: Iterable[Path],
) -> dict[str, str]:
    """Return `env` with existing runtime bin directories prepended to PATH."""

    resolved_env = dict(env)
    path_parts: list[str] = []
    seen: set[str] = set()
    for bin_dir in bins:
        if not bin_dir.is_dir():
            continue
        bin_dir_str = str(bin_dir)
        if bin_dir_str in seen:
            continue
        seen.add(bin_dir_str)
        path_parts.append(bin_dir_str)

    current_path = resolved_env.get("PATH", os.environ.get("PATH", ""))
    for entry in current_path.split(os.pathsep):
        if not entry or entry in seen:
            continue
        seen.add(entry)
        path_parts.append(entry)

    if path_parts:
        resolved_env["PATH"] = os.pathsep.join(path_parts)
    return resolved_env


def prepend_agent_runtime_bins(
    env: Mapping[str, str],
    *,
    workspace_root: Path | str | None = None,
) -> dict[str, str]:
    """Return `env` with workspace and repo agent runtime bins on PATH."""

    bins: list[Path] = []
    if workspace_root is not None:
        bins.append(workspace_runtime_bin(workspace_root))
    bins.append(repo_runtime_bin())
    return prepend_runtime_bins(env, bins)
