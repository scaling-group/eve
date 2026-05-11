"""Shared Codex hook config and trust helpers."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

CODEX_HOOK_TRUST_EVENTS = (
    "pre_tool_use",
    "post_tool_use",
    "session_start",
    "user_prompt_submit",
)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up to the nearest pyproject.toml."""

    candidate = (start or Path(__file__)).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repository root from {candidate}")


def repo_codex_hooks_path(repo_root: Path | None = None) -> Path:
    """Return the stable repo-level Codex hooks.json path."""

    root = find_repo_root(repo_root)
    return root / ".codex" / "hooks.json"


def write_repo_codex_hooks(repo_root: Path | None = None) -> Path:
    """Write the stable repo-level Codex hooks.json file."""

    root = find_repo_root(repo_root)
    hooks_path = repo_codex_hooks_path(root)
    write_codex_hooks_file(hooks_path, repo_root=root)
    return hooks_path


def write_codex_hooks_file(hooks_path: Path, *, repo_root: Path | None = None) -> None:
    """Write a Codex hooks.json file at the provided path."""

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(codex_hooks_payload(repo_root=repo_root), indent=2) + "\n",
        encoding="utf-8",
    )


def codex_hooks_payload(*, repo_root: Path | None = None) -> dict[str, object]:
    """Return the Codex hooks.json payload used by Eve workspaces."""

    hook_command = workspace_hook_command(repo_root=repo_root)
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Read|Edit|Write|MultiEdit|Glob|Grep",
                    "hooks": [{"type": "command", "command": hook_command}],
                }
            ],
            "SessionStart": [{"hooks": [{"type": "command", "command": hook_command}]}],
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": hook_command}]}],
            "PostToolUse": [{"hooks": [{"type": "command", "command": hook_command}]}],
        }
    }


def workspace_hook_command(*, repo_root: Path | None = None) -> str:
    """Return the command invoked by Codex hooks inside workspaces."""

    repo_src_root = find_repo_root(repo_root) / "src"
    return (
        f"env PYTHONPATH={repo_src_root} {sys.executable} "
        "-m scaling_evolve.providers.agent.hooks.workspace_guard"
    )


def real_codex_config_path() -> Path:
    """Return the real user-level Codex config path."""

    return Path.home() / ".codex" / "config.toml"


def read_trusted_hook_state(
    hooks_json_path: str | Path,
    *,
    config_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read trusted hook state entries matching one hooks.json path."""

    resolved_hooks_path = str(Path(hooks_json_path).expanduser().resolve())
    prefix = f"{resolved_hooks_path}:"
    source_config = config_path or real_codex_config_path()
    if not source_config.exists():
        return {}
    payload = tomllib.loads(source_config.read_text(encoding="utf-8"))
    hooks_section = payload.get("hooks")
    if not isinstance(hooks_section, dict):
        return {}
    state_section = hooks_section.get("state")
    if not isinstance(state_section, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for key, value in state_section.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if isinstance(value, dict):
            entries[key] = dict(value)
    return entries


def missing_hook_trust_events(
    hooks_json_path: str | Path,
    *,
    config_path: Path | None = None,
) -> list[str]:
    """Return required Codex hook events missing trust state."""

    resolved_hooks_path = str(Path(hooks_json_path).expanduser().resolve())
    entries = read_trusted_hook_state(resolved_hooks_path, config_path=config_path)
    missing = []
    for event in CODEX_HOOK_TRUST_EVENTS:
        key = f"{resolved_hooks_path}:{event}:0:0"
        if key not in entries:
            missing.append(event)
    return missing


def is_repo_project_trusted(
    repo_root: Path,
    *,
    config_path: Path | None = None,
) -> bool:
    """Check if the repo root is trusted as a Codex project."""

    source_config = config_path or real_codex_config_path()
    if not source_config.exists():
        return False
    payload = tomllib.loads(source_config.read_text(encoding="utf-8"))
    projects = payload.get("projects")
    if not isinstance(projects, dict):
        return False
    resolved = str(repo_root.expanduser().resolve())
    entry = projects.get(resolved)
    if not isinstance(entry, dict):
        return False
    return entry.get("trust_level") == "trusted"


def ensure_codex_hooks_trusted(
    repo_root: Path | None = None,
    *,
    config_path: Path | None = None,
) -> None:
    """Exit early with instructions when repo-level Codex hooks are not trusted."""

    root = find_repo_root(repo_root)
    hooks_json_path = repo_codex_hooks_path(root)
    hooks_missing = missing_hook_trust_events(hooks_json_path, config_path=config_path)
    project_trusted = is_repo_project_trusted(root, config_path=config_path)
    if not hooks_missing and project_trusted:
        return

    print(_trust_error_message(root), file=sys.stderr)
    raise SystemExit(1)


def render_trusted_hook_state_toml(
    hooks_json_path: str | Path | None,
    *,
    config_path: Path | None = None,
) -> str:
    """Render matching real-home hook trust entries for an isolated Codex config."""

    if hooks_json_path is None:
        return ""
    entries = read_trusted_hook_state(hooks_json_path, config_path=config_path)
    if not entries:
        return ""
    lines: list[str] = []
    for key in _ordered_trust_keys(str(Path(hooks_json_path).expanduser().resolve()), entries):
        values = entries[key]
        lines.append(f"[hooks.state.{json.dumps(key)}]")
        for field_name, field_value in values.items():
            if not isinstance(field_name, str):
                continue
            rendered_value = _toml_scalar(field_value)
            if rendered_value is not None:
                lines.append(f"{field_name} = {rendered_value}")
        lines.append("")
    return "\n".join(lines)


def _ordered_trust_keys(hooks_json_path: str, entries: dict[str, dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for event in CODEX_HOOK_TRUST_EVENTS:
        key = f"{hooks_json_path}:{event}:0:0"
        if key in entries:
            ordered.append(key)
    ordered.extend(sorted(key for key in entries if key not in set(ordered)))
    return ordered


def _trust_error_message(repo_root: Path) -> str:
    return "\n".join(
        [
            "Codex hooks are not trusted for this repository.",
            "",
            "Eve requires workspace hooks for sandbox protection and budget prompts.",
            "Run this once to trust hooks:",
            "",
            f"    codex -C {repo_root}",
            "    (type /hooks -> press t to trust all -> Esc -> Ctrl-C)",
            "",
            "Then re-run Eve.",
        ]
    )


def _toml_scalar(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    return None


if __name__ == "__main__":
    root = find_repo_root()
    hooks_path = write_repo_codex_hooks(root)
    print(f"Created {hooks_path}")
    print(f"\nNow trust hooks:\n    codex -C {root}\n    (type /hooks -> press t -> Esc -> Ctrl-C)")
