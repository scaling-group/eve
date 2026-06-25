"""File-tree helpers: materialize and read {relative_path: content} dicts."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path

_BINARY_PREFIX = "__scaling_evolve_binary_v1__:"


def write_file_tree(root: Path, files: dict[str, str]) -> None:
    """Materialize a {relative_path: content} dict into a directory tree.

    Text files are written as UTF-8. Values encoded with the portable binary
    marker are decoded back into their original bytes before writing.
    """
    for rel_path, content in files.items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = decode_portable_file_content(content)
        if payload is None:
            dest.write_text(content, encoding="utf-8")
            continue
        dest.write_bytes(payload)


def read_file_tree(root: Path) -> dict[str, str]:
    """Recursively read a directory into a portable {relative_path: content} dict.

    Text files are returned unchanged. Binary files are represented with a
    reversible ASCII envelope so the tree can be serialized and later
    materialized without losing bytes.
    """
    result: dict[str, str] = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        payload = path.read_bytes()
        if _is_probably_binary(payload):
            result[rel] = encode_portable_binary_content(payload)
            continue
        result[rel] = payload.decode("utf-8")
    return result


def encode_portable_binary_content(payload: bytes) -> str:
    """Encode raw bytes into a JSON-backed ASCII envelope."""

    envelope = {
        "encoding": "base64",
        "content": base64.b64encode(payload).decode("ascii"),
    }
    return _BINARY_PREFIX + json.dumps(envelope, separators=(",", ":"), sort_keys=True)


def decode_portable_file_content(content: str) -> bytes | None:
    """Return decoded bytes for portable-binary content, else ``None``."""

    if not content.startswith(_BINARY_PREFIX):
        return None
    payload = json.loads(content.removeprefix(_BINARY_PREFIX))
    if not isinstance(payload, dict):
        raise ValueError("Portable binary envelope must be a JSON object.")
    if payload.get("encoding") != "base64":
        raise ValueError("Portable binary envelope uses an unsupported encoding.")
    encoded = payload.get("content")
    if not isinstance(encoded, str):
        raise ValueError("Portable binary envelope is missing string content.")
    return base64.b64decode(encoded)


def _is_probably_binary(payload: bytes) -> bool:
    if b"\x00" in payload:
        return True
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError:
        return True
    control_count = sum(1 for ch in decoded if ord(ch) < 32 and ch not in {"\n", "\r", "\t", "\f"})
    return control_count > 0


def expose_guidance_skills(workspace: Path) -> None:
    """Expose guidance/skills through workspace-root agent conventions."""
    skills_dir = workspace / "guidance" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    _ensure_symlink(
        workspace / ".claude" / "skills",
        Path("..") / "guidance" / "skills",
        target_is_directory=True,
    )
    _ensure_symlink(
        workspace / ".codex" / "skills",
        Path("..") / "guidance" / "skills",
        target_is_directory=True,
    )


def expose_guidance_agents(workspace: Path) -> None:
    """Expose optimizer-provided agent definitions through local tool conventions.

    Guidance agent provider directories are symlinked so edits and newly created
    guidance agents remain visible during the same workspace. Immutable files may
    later be written through these symlinks as overlay files.
    """
    agent_specs = (
        (workspace / "guidance" / "agents" / "codex", workspace / ".codex" / "agents"),
        (workspace / "guidance" / "agents" / "claude", workspace / ".claude" / "agents"),
    )
    for source_dir, destination_dir in agent_specs:
        source_dir.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(
            destination_dir,
            Path("..") / "guidance" / "agents" / source_dir.name,
            target_is_directory=True,
        )


def write_claude_stop_hook_settings(
    workspace: Path,
    *,
    signal_filename: str = ".claude-task-stopped",
    log_filename: str = ".claude-stop-hook.log",
) -> None:
    """Write workspace-local Claude Stop hook settings for tmux completion detection."""

    signal_path = workspace / signal_filename
    log_path = workspace / log_filename
    settings_path = workspace / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hooks": {
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                f"touch {shlex.quote(str(signal_path))} && "
                                f"date -u +%Y-%m-%dT%H:%M:%SZ >> {shlex.quote(str(log_path))} && "
                                "echo '{\"continue\": true}'"
                            ),
                            "timeout": 5,
                        }
                    ],
                }
            ]
        }
    }
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_symlink(path: Path, target: Path, *, target_is_directory: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target, target_is_directory=target_is_directory)
