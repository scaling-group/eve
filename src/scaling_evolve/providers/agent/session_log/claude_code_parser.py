"""Claude Code transcript parser for markdown session logs."""

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
    stringify_payload,
)


def parse_claude_code_session(
    transcript_path: Path,
    *,
    session_id: str | None,
    effort: str | None = None,
) -> ParsedSession | None:
    """Parse one Claude Code JSONL transcript into structured session data."""

    if not transcript_path.exists():
        return None

    parsed = ParsedSession(provider="claude_code", session_id=session_id, effort=effort)
    current_turn: TraceTurn | None = None
    current_turn_message_id: str | None = None
    last_agent_text: str | None = None

    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        payload = _load_json_line(line)
        if payload is None:
            continue
        record_session_id = _string(payload.get("sessionId")) or _string(payload.get("session_id"))
        if parsed.session_id is not None and record_session_id not in {None, parsed.session_id}:
            continue
        parsed.session_id = parsed.session_id or record_session_id
        timestamp = _string(payload.get("timestamp"))
        parsed.started_at = parsed.started_at or timestamp
        parsed.ended_at = timestamp or parsed.ended_at
        record_type = _string(payload.get("type"))
        if record_type == "system":
            if _string(payload.get("subtype")) == "init":
                parsed.cwd = parsed.cwd or _string(payload.get("cwd"))
                parsed.model = parsed.model or _string(payload.get("model"))
            continue
        if record_type == "assistant":
            message = _mapping(payload.get("message"))
            message_id = _string(message.get("id"))
            if current_turn is None:
                current_turn = TraceTurn()
                current_turn_message_id = message_id
            elif message_id is None:
                current_turn = _flush_turn(parsed, current_turn)
                current_turn = TraceTurn()
                current_turn_message_id = None
            elif message_id != current_turn_message_id:
                current_turn = _flush_turn(parsed, current_turn)
                current_turn = TraceTurn()
                current_turn_message_id = message_id
            parsed.model = parsed.model or _string(message.get("model"))
            parsed.cwd = parsed.cwd or _string(payload.get("cwd"))
            for block in _content_blocks(message.get("content")):
                block_type = _string(block.get("type"))
                if block_type == "thinking":
                    thinking = _string(block.get("thinking")) or _string(block.get("text"))
                    if thinking and thinking.strip():
                        current_turn.thinking.append(thinking.strip())
                elif block_type == "text":
                    text = _string(block.get("text"))
                    if text and text.strip():
                        cleaned = text.strip()
                        current_turn.agent.append(cleaned)
                        last_agent_text = cleaned
                elif block_type == "tool_use":
                    current_turn.tools.append(
                        ToolEvent(
                            name=_string(block.get("name")) or "tool",
                            args=stringify_payload(block.get("input")),
                            tool_id=_string(block.get("id")),
                        )
                    )
        elif record_type == "user":
            parsed.cwd = parsed.cwd or _string(payload.get("cwd"))
            top_level_tool_result = _mapping(payload.get("tool_use_result"))
            if top_level_tool_result:
                current_turn = current_turn or TraceTurn()
                tool = _pending_tool(current_turn)
                if tool is None:
                    tool = ToolEvent(name="tool_result", args="")
                    current_turn.tools.append(tool)
                tool.result_bytes = result_bytes(top_level_tool_result)
                tool.result_success = not bool(payload.get("is_error"))
            message = _mapping(payload.get("message"))
            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    parsed.instructions.append(text)
                continue
            for block in _content_blocks(content):
                block_type = _string(block.get("type"))
                if block_type == "tool_result":
                    current_turn = current_turn or TraceTurn()
                    tool = _tool_by_id_across_turns(
                        parsed,
                        current_turn,
                        _string(block.get("tool_use_id")),
                    )
                    if tool is None:
                        tool = _pending_tool(current_turn)
                    if tool is None:
                        tool = ToolEvent(name="tool_result", args="")
                        current_turn.tools.append(tool)
                    tool.result_bytes = result_bytes(block.get("content"))
                    tool.result_success = not bool(block.get("is_error"))
                elif block_type == "text":
                    text = _string(block.get("text"))
                    if text and text.strip():
                        parsed.instructions.append(text.strip())
        elif record_type == "result":
            result_text = _string(payload.get("result"))
            if result_text and result_text.strip():
                last_agent_text = result_text.strip()

    current_turn = _flush_turn(parsed, current_turn)
    parsed.final_response = last_agent_text
    parsed.role, parsed.iteration = infer_workspace_context(parsed.cwd)
    return parsed


def _flush_turn(parsed: ParsedSession, current_turn: TraceTurn | None) -> TraceTurn | None:
    if current_turn is not None and not current_turn.empty():
        parsed.turns.append(current_turn)
    return None


def _tool_by_id_across_turns(
    parsed: ParsedSession,
    current_turn: TraceTurn | None,
    tool_id: str | None,
) -> ToolEvent | None:
    if tool_id is None:
        return None
    turns: list[TraceTurn] = []
    if current_turn is not None:
        turns.append(current_turn)
    turns.extend(reversed(parsed.turns))
    for turn in turns:
        for tool in reversed(turn.tools):
            if tool.tool_id == tool_id:
                return tool
    return None


def _pending_tool(turn: TraceTurn) -> ToolEvent | None:
    for tool in reversed(turn.tools):
        if tool.result_bytes is None:
            return tool
    return None


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
