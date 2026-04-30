"""Thin Codex trust-shim helpers for tmux-driven sessions."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CodexLaunchConfig:
    """Minimal launch configuration required for Codex trust bootstrap."""

    worktree_root: Path


@dataclass(frozen=True)
class IsolatedCodexHome:
    """Resolved per-workspace Codex HOME tree."""

    root: Path
    codex_dir: Path
    auth_path: Path
    config_path: Path

    def env(self) -> dict[str, str]:
        """Return the environment variables needed for Codex launch."""

        return {"HOME": str(self.root)}


def create_isolated_codex_home(
    *,
    home_root: Path,
    source_auth_path: Path | None,
    launch: CodexLaunchConfig,
) -> IsolatedCodexHome:
    """Create a thin HOME tree that only bootstraps project trust and auth."""

    resolved_home = home_root.expanduser().resolve()
    codex_dir = resolved_home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    auth_path = codex_dir / "auth.json"
    if source_auth_path is not None and source_auth_path.exists():
        if auth_path.exists() or auth_path.is_symlink():
            auth_path.unlink()
        auth_path.symlink_to(source_auth_path.expanduser().resolve())

    config_path = codex_dir / "config.toml"
    config_path.write_text(_render_config_toml(launch), encoding="utf-8")
    return IsolatedCodexHome(
        root=resolved_home,
        codex_dir=codex_dir,
        auth_path=auth_path,
        config_path=config_path,
    )


def reset_isolated_codex_home(
    *,
    home: IsolatedCodexHome,
    source_auth_path: Path | None,
    launch: CodexLaunchConfig,
) -> IsolatedCodexHome:
    """Rewrite the trust shim and prune provider-managed runtime residue."""

    refreshed = create_isolated_codex_home(
        home_root=home.root,
        source_auth_path=source_auth_path,
        launch=launch,
    )
    _prune_home(refreshed)
    refreshed.config_path.write_text(_render_config_toml(launch), encoding="utf-8")
    return refreshed


def latest_rollout_path(
    codex_home: Path,
    *,
    after_mtime_ns: int | None = None,
) -> Path | None:
    """Return the newest rollout JSONL path under a Codex home."""

    sessions_root = codex_home / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    candidates = sorted(sessions_root.rglob("rollout-*.jsonl"))
    if after_mtime_ns is not None:
        candidates = [path for path in candidates if path.stat().st_mtime_ns >= after_mtime_ns]
    return candidates[-1] if candidates else None


def rollout_path_for_session(codex_home: Path, *, session_id: str) -> Path | None:
    """Return the rollout JSONL path that belongs to one provider session id."""

    sessions_root = codex_home / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    for path in sorted(sessions_root.rglob("rollout-*.jsonl")):
        if extract_session_id(path) == session_id:
            return path
    return None


def extract_last_assistant_message(session_jsonl: Path) -> str | None:
    """Extract the last assistant-facing message text from one rollout JSONL file."""

    last_message: str | None = None
    for line in session_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = _load_json_line(line)
        if payload is None:
            continue
        if payload.get("type") == "event_msg":
            inner = _mapping(payload.get("payload"))
            if inner.get("type") == "agent_message":
                message = inner.get("message")
                if isinstance(message, str) and message.strip():
                    last_message = message.strip()
        if payload.get("type") == "response_item":
            item = _mapping(payload.get("payload"))
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_mapping = _mapping(block)
                text = block_mapping.get("text")
                if isinstance(text, str) and text.strip():
                    last_message = text.strip()
    return last_message


def extract_session_id(session_jsonl: Path) -> str | None:
    """Extract the provider session id from one rollout JSONL file."""

    for line in session_jsonl.read_text(encoding="utf-8").splitlines():
        payload = _load_json_line(line)
        if payload is None or payload.get("type") != "session_meta":
            continue
        meta = _mapping(payload.get("payload"))
        session_id = meta.get("id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
    return None


def extract_usage(session_jsonl: Path, *, from_line: int = 0) -> dict[str, int]:
    """Extract token usage fields from the latest Codex token-count event."""

    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "agent_turns": 0,
    }
    for line in session_jsonl.read_text(encoding="utf-8").splitlines()[from_line:]:
        payload = _load_json_line(line)
        if payload is None or payload.get("type") != "event_msg":
            continue
        outer = _mapping(payload.get("payload"))
        if outer.get("type") == "agent_message":
            usage["agent_turns"] += 1
            continue
        if outer.get("type") != "token_count":
            continue
        info = _mapping(outer.get("info"))
        total = _mapping(info.get("total_token_usage"))
        if not total:
            continue
        usage = {
            "input_tokens": _int_value(total.get("input_tokens")),
            "cached_input_tokens": _int_value(total.get("cached_input_tokens")),
            "output_tokens": _int_value(total.get("output_tokens")),
            "reasoning_output_tokens": _int_value(total.get("reasoning_output_tokens")),
            "agent_turns": usage["agent_turns"],
        }
    return usage


def _render_config_toml(launch: CodexLaunchConfig) -> str:
    worktree = str(launch.worktree_root.resolve())
    return (
        "[features]\n"
        "codex_hooks = true\n\n"
        f"[projects.{json.dumps(worktree)}]\n"
        'trust_level = "trusted"\n'
    )


def _prune_home(home: IsolatedCodexHome) -> None:
    keep = {"auth.json", "config.toml", "sessions"}
    for child in home.codex_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
            continue
        try:
            child.unlink()
        except FileNotFoundError:
            continue


def _load_json_line(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0
