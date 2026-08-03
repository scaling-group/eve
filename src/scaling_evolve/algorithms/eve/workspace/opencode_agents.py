"""Render Eve's local helper agents for OpenCode."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

_AGENT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def project_opencode_agents(
    workspace: Path,
    *,
    config_root: Path,
    steps: int,
) -> None:
    """Project helper agents while letting them inherit OpenCode's selected model."""

    source_root = workspace / ".codex" / "agents"
    target_root = config_root / "agents"
    if not source_root.is_dir():
        if target_root.is_dir():
            for stale_path in target_root.glob("*.md"):
                stale_path.unlink()
        return

    definitions: dict[str, tuple[str, str]] = {}
    for source in sorted(source_root.rglob("*.toml")):
        try:
            payload = tomllib.loads(source.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"Invalid helper agent definition: {source}") from error
        name = _required_string(payload, "name", source)
        if not _AGENT_NAME.fullmatch(name):
            raise ValueError(f"Invalid helper agent name `{name}` in {source}.")
        if name in definitions:
            raise ValueError(f"Duplicate helper agent name `{name}` in {source_root}.")
        definitions[name] = (
            _required_string(payload, "description", source),
            _required_string(payload, "developer_instructions", source),
        )

    if target_root.is_dir():
        for stale_path in target_root.glob("*.md"):
            stale_path.unlink()
    if not definitions:
        return
    target_root.mkdir(parents=True, exist_ok=True)
    for name, (description, prompt) in definitions.items():
        rendered = "\n".join(
            [
                "---",
                f"description: {json.dumps(description, ensure_ascii=False)}",
                "mode: subagent",
                f"steps: {steps}",
                "permission:",
                "  task: deny",
                "---",
                "",
                prompt.strip(),
                "",
            ]
        )
        (target_root / f"{name}.md").write_text(rendered, encoding="utf-8")


def _required_string(payload: dict[str, object], key: str, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Helper agent definition {source} requires non-empty `{key}`.")
    return value.strip()
