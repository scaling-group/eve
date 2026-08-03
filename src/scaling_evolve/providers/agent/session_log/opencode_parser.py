"""OpenCode JSONL transcript parser for unified session logs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scaling_evolve.providers.agent.session_log.schema import (
    ParsedSession,
    ToolEvent,
    TraceTurn,
    infer_workspace_context,
    result_bytes,
    stringify_payload,
)


def parse_opencode_session(
    transcript_path: Path,
    *,
    session_id: str | None,
) -> ParsedSession | None:
    """Parse one cumulative OpenCode driver transcript."""

    if not transcript_path.exists():
        return None

    parsed = ParsedSession(provider="opencode", session_id=session_id)
    current_turn: TraceTurn | None = None
    last_agent_text: str | None = None
    observed_session_ids: set[str] = set()

    for line_number, line in enumerate(
        transcript_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = _load_json_line(line, line_number=line_number)
        event_session_id = _string(payload.get("sessionID"))
        if event_session_id is None:
            raise ValueError(f"line {line_number}: OpenCode event is missing sessionID")
        observed_session_ids.add(event_session_id)
        if session_id is not None and event_session_id != session_id:
            raise ValueError(
                f"line {line_number}: expected session `{session_id}`, got `{event_session_id}`"
            )
        parsed.session_id = parsed.session_id or event_session_id

        timestamp = _timestamp(payload.get("timestamp"))
        parsed.started_at = parsed.started_at or timestamp
        parsed.ended_at = timestamp or parsed.ended_at
        event_type = _string(payload.get("type"))
        if event_type == "eve.rollout":
            instruction = _string(payload.get("instruction"))
            if instruction is not None:
                parsed.instructions.append(instruction)
            parsed.model = _string(payload.get("model")) or parsed.model
            parsed.effort = _string(payload.get("variant")) or parsed.effort
            parsed.role = _string(payload.get("role")) or parsed.role
            parsed.cwd = _string(payload.get("cwd")) or parsed.cwd
            continue

        part = _mapping(payload.get("part"))
        if event_type == "step_start":
            current_turn = _flush_turn(parsed, current_turn)
            current_turn = TraceTurn()
        elif event_type == "reasoning":
            text = _string(part.get("text"))
            if text is not None:
                current_turn = current_turn or TraceTurn()
                current_turn.thinking.append(text)
        elif event_type == "text":
            text = _string(part.get("text"))
            if text is not None:
                current_turn = current_turn or TraceTurn()
                current_turn.agent.append(text)
                last_agent_text = text
        elif event_type == "tool_use":
            current_turn = current_turn or TraceTurn()
            current_turn.tools.append(_tool_event(part))
        elif event_type == "step_finish":
            current_turn = _flush_turn(parsed, current_turn)
        elif event_type == "error":
            raise ValueError(f"line {line_number}: OpenCode session error event in transcript")

    if len(observed_session_ids) > 1:
        raise ValueError(f"OpenCode transcript contains multiple sessions: {observed_session_ids}")
    _flush_turn(parsed, current_turn)
    parsed.final_response = last_agent_text
    inferred_role, inferred_iteration = infer_workspace_context(parsed.cwd)
    parsed.role = parsed.role or inferred_role
    parsed.iteration = inferred_iteration
    return parsed


def _tool_event(part: dict[str, Any]) -> ToolEvent:
    state = _mapping(part.get("state"))
    status = _string(state.get("status"))
    output: object | None = None
    success: bool | None = None
    if status == "completed":
        output = state.get("output")
        success = True
    elif status == "error":
        output = state.get("error")
        success = False
    return ToolEvent(
        name=_string(part.get("tool")) or "tool",
        args=stringify_payload(state.get("input")),
        tool_id=_string(part.get("callID")),
        result_bytes=result_bytes(output) if output is not None else None,
        result_success=success,
    )


def _flush_turn(parsed: ParsedSession, current_turn: TraceTurn | None) -> TraceTurn | None:
    if current_turn is not None and not current_turn.empty():
        parsed.turns.append(current_turn)
    return None


def _load_json_line(line: str, *, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line_number}: invalid OpenCode JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: OpenCode event must be an object")
    return payload


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _timestamp(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return (
            datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")
        )
    return _string(value)
