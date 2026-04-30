"""Shared transcript helpers for rollout turn counting and tool-batch inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TranscriptTurnState:
    """Observed turn state derived from one transcript file."""

    format_name: Literal["claude", "codex_exec", "codex_tmux", "unknown"]
    turn_count: int
    latest_batch_tool_ids: tuple[str, ...]


def inspect_transcript_turn_state(transcript_path: Path) -> TranscriptTurnState:
    """Return the current turn count and latest tool batch ids for one transcript."""

    if not transcript_path.exists():
        return TranscriptTurnState("unknown", 0, ())
    payloads = _load_payloads(transcript_path)
    format_name = _detect_format(payloads)
    if format_name == "claude":
        return _inspect_claude_payloads(payloads)
    if format_name == "codex_exec":
        return _inspect_codex_exec_payloads(payloads)
    if format_name == "codex_tmux":
        return _inspect_codex_tmux_payloads(payloads)
    return TranscriptTurnState("unknown", 0, ())


def _inspect_claude_payloads(payloads: list[dict[str, Any]]) -> TranscriptTurnState:
    assistant_ids: set[str] = set()
    latest_batch: tuple[str, ...] = ()
    for index, payload in enumerate(payloads):
        if _string(payload.get("type")) != "assistant":
            continue
        message = _mapping(payload.get("message"))
        batch_ids = tuple(
            tool_id
            for tool_id in (
                _string(block.get("id"))
                for block in _content_blocks(message.get("content"))
                if _string(block.get("type")) == "tool_use"
            )
            if tool_id is not None
        )
        if not batch_ids:
            continue
        assistant_id = (
            _string(message.get("id"))
            or _string(payload.get("requestId"))
            or f"assistant-line:{index}"
        )
        assistant_ids.add(assistant_id)
        latest_batch = batch_ids
    return TranscriptTurnState("claude", len(assistant_ids), latest_batch)


def _inspect_codex_exec_payloads(payloads: list[dict[str, Any]]) -> TranscriptTurnState:
    turn_count = 0
    latest_batch: tuple[str, ...] = ()
    current_batch: list[str] = []
    for payload in payloads:
        payload_type = _string(payload.get("type"))
        if payload_type == "item.completed":
            item = _mapping(payload.get("item"))
            item_type = _string(item.get("type"))
            if item_type == "agent_message":
                turn_count += 1
            elif item_type == "function_call":
                call_id = _string(item.get("call_id"))
                if call_id is not None:
                    current_batch.append(call_id)
                continue
        if current_batch:
            latest_batch = tuple(current_batch)
            current_batch = []
    if current_batch:
        latest_batch = tuple(current_batch)
    return TranscriptTurnState("codex_exec", turn_count, latest_batch)


def _inspect_codex_tmux_payloads(payloads: list[dict[str, Any]]) -> TranscriptTurnState:
    turn_count = 0
    latest_batch: tuple[str, ...] = ()
    current_batch: list[str] = []
    for payload in payloads:
        payload_type = _string(payload.get("type"))
        if payload_type == "event_msg":
            event = _mapping(payload.get("payload"))
            if _string(event.get("type")) == "agent_message":
                turn_count += 1
            if current_batch:
                latest_batch = tuple(current_batch)
                current_batch = []
            continue
        if payload_type == "response_item":
            item = _mapping(payload.get("payload"))
            if _string(item.get("type")) == "function_call":
                call_id = _string(item.get("call_id"))
                if call_id is not None:
                    current_batch.append(call_id)
                continue
        if current_batch:
            latest_batch = tuple(current_batch)
            current_batch = []
    if current_batch:
        latest_batch = tuple(current_batch)
    return TranscriptTurnState("codex_tmux", turn_count, latest_batch)


def _detect_format(
    payloads: list[dict[str, Any]],
) -> Literal["claude", "codex_exec", "codex_tmux", "unknown"]:
    payload_types = {_string(payload.get("type")) for payload in payloads}
    if "assistant" in payload_types:
        return "claude"
    if "item.completed" in payload_types or "thread.started" in payload_types:
        return "codex_exec"
    if "event_msg" in payload_types or "response_item" in payload_types:
        return "codex_tmux"
    return "unknown"


def _load_payloads(transcript_path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        payload = _load_json_line(line)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _content_blocks(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [block for block in value if isinstance(block, dict)]


def _load_json_line(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
