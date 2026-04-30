"""Codex transcript parser for markdown session logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scaling_evolve.providers.agent.session_log.schema import (
    ParsedSession,
    ToolEvent,
    TraceTurn,
    infer_workspace_context,
    result_bytes,
)


def parse_codex_session(
    transcript_path: Path,
    *,
    session_id: str | None,
) -> ParsedSession | None:
    """Parse one Codex JSONL transcript into structured session data."""

    if not transcript_path.exists():
        return None

    parsed = ParsedSession(provider="codex_tmux", session_id=session_id)
    root_cwd: str | None = None
    current_turn: TraceTurn | None = None
    last_agent_text: str | None = None

    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        payload = _load_json_line(line)
        if payload is None:
            continue
        timestamp = _string(payload.get("timestamp"))
        parsed.started_at = parsed.started_at or timestamp
        parsed.ended_at = timestamp or parsed.ended_at
        record_type = _string(payload.get("type"))
        if record_type == "session_meta":
            meta = _mapping(payload.get("payload"))
            meta_session_id = _string(meta.get("id"))
            if parsed.session_id is None and meta_session_id:
                parsed.session_id = meta_session_id
            if meta_session_id == parsed.session_id or (
                root_cwd is None and meta_session_id is not None
            ):
                root_cwd = _string(meta.get("cwd")) or root_cwd
                parsed.cwd = root_cwd or parsed.cwd
        elif record_type == "turn_context":
            context = _mapping(payload.get("payload"))
            context_cwd = _string(context.get("cwd"))
            if root_cwd is not None and context_cwd not in {None, root_cwd}:
                continue
            parsed.model = parsed.model or _string(context.get("model"))
            parsed.effort = parsed.effort or _string(context.get("effort"))
            parsed.cwd = parsed.cwd or context_cwd
        elif record_type == "event_msg":
            event = _mapping(payload.get("payload"))
            event_type = _string(event.get("type"))
            if event_type == "user_message":
                message = _string(event.get("message"))
                if message and message.strip():
                    parsed.instructions.append(message.strip())
            elif event_type == "agent_message":
                current_turn = _flush_turn(parsed, current_turn)
                message = _string(event.get("message"))
                if message and message.strip():
                    last_agent_text = message.strip()
                    current_turn = TraceTurn(agent=[last_agent_text])
        elif record_type == "response_item":
            item = _mapping(payload.get("payload"))
            item_type = _string(item.get("type"))
            if item_type == "function_call":
                current_turn = current_turn or TraceTurn()
                current_turn.tools.append(
                    ToolEvent(
                        name=_string(item.get("name")) or "tool",
                        args=_string(item.get("arguments")) or "",
                        tool_id=_string(item.get("call_id")),
                    )
                )
            elif item_type == "function_call_output":
                current_turn = current_turn or TraceTurn()
                output = item.get("output")
                tool = _last_pending_tool(current_turn)
                if tool is None:
                    tool = ToolEvent(name="tool_result", args="")
                    current_turn.tools.append(tool)
                tool.result_bytes = result_bytes(output)
                tool.result_success = _infer_codex_tool_success(output)
            elif item_type == "message" and _string(item.get("role")) == "assistant":
                text = _assistant_message_text(item)
                if text and text != last_agent_text:
                    current_turn = _flush_turn(parsed, current_turn)
                    last_agent_text = text
                    current_turn = TraceTurn(agent=[text])

    current_turn = _flush_turn(parsed, current_turn)
    parsed.final_response = last_agent_text
    parsed.role, parsed.iteration = infer_workspace_context(parsed.cwd)
    return parsed


def _flush_turn(parsed: ParsedSession, current_turn: TraceTurn | None) -> TraceTurn | None:
    if current_turn is not None and not current_turn.empty():
        parsed.turns.append(current_turn)
    return None


def _last_pending_tool(turn: TraceTurn) -> ToolEvent | None:
    for tool in reversed(turn.tools):
        if tool.result_bytes is None:
            return tool
    return None


def _assistant_message_text(item: dict[str, Any]) -> str | None:
    content = item.get("content")
    if not isinstance(content, list):
        return None
    blocks = []
    for block in content:
        mapping = _mapping(block)
        if _string(mapping.get("type")) != "output_text":
            continue
        text = _string(mapping.get("text"))
        if text and text.strip():
            blocks.append(text.strip())
    if not blocks:
        return None
    return "\n\n".join(blocks)


def _infer_codex_tool_success(output: object) -> bool | None:
    if not isinstance(output, str):
        return None
    if "Process exited with code 0" in output:
        return True
    if "Process exited with code " in output:
        return False
    return None


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
