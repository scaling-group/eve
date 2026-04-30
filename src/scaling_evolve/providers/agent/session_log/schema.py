"""Shared markdown schema helpers for session logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

MAX_TOOL_ARGS_CHARS = 500


@dataclass
class ToolEvent:
    """One tool call plus its summarized result."""

    name: str
    args: str
    tool_id: str | None = None
    result_bytes: int | None = None
    result_success: bool | None = None


@dataclass
class TraceTurn:
    """One assistant turn in the rendered trace."""

    thinking: list[str] = field(default_factory=list)
    agent: list[str] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.thinking and not self.agent and not self.tools


@dataclass
class ParsedSession:
    """Structured session data before markdown rendering."""

    session_id: str | None = None
    provider: str = "unknown"
    model: str | None = None
    effort: str | None = None
    role: str | None = None
    iteration: int | None = None
    cwd: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    instructions: list[str] = field(default_factory=list)
    turns: list[TraceTurn] = field(default_factory=list)
    final_response: str | None = None


def infer_workspace_context(cwd: str | None) -> tuple[str | None, int | None]:
    """Infer role and iteration from an Eve workspace path."""

    if not cwd:
        return None, None
    role: str | None = None
    if "/solver_workspaces/" in cwd:
        role = "solver"
    elif "/optimizer_workspaces/" in cwd:
        role = "optimizer"
    match = re.search(r"(?:^|_)step_(\d+)(?:_|$)", cwd)
    iteration = int(match.group(1)) if match else None
    return role, iteration


def render_session_markdown(
    parsed: ParsedSession,
    *,
    rollout_count: int,
    usage: dict[str, int | float],
) -> str:
    """Render one parsed session into the shared markdown schema."""

    role = parsed.role or "agent"
    title = f"# Session Log - {role}"
    if parsed.iteration is not None:
        title += f" (iter {parsed.iteration})"
    lines = [
        title,
        "",
        f"- **Session ID**: {parsed.session_id or 'unknown'}",
        f"- **Provider**: {parsed.provider}",
        f"- **Model**: {_model_line(parsed.model, parsed.effort)}",
        f"- **Role**: {role}",
        f"- **Started / Ended**: {_pair(parsed.started_at, parsed.ended_at)}",
        f"- **Duration**: {_duration_text(parsed.started_at, parsed.ended_at)}",
        f"- **Cwd**: {parsed.cwd or 'unknown'}",
        f"- **Turns**: {len(parsed.turns)}",
        (
            "- **Tokens**: "
            f"in={int(usage.get('input_tokens', 0) or 0)}, "
            f"out={int(usage.get('output_tokens', 0) or 0)}, "
            f"cache_read={int(usage.get('cache_read_tokens', 0) or 0)}"
        ),
        f"- **Spawn/resume count**: {rollout_count}",
        "",
        "## Instruction",
        "",
    ]
    if parsed.instructions:
        for index, message in enumerate(parsed.instructions, start=1):
            if len(parsed.instructions) > 1:
                lines.extend([f"### User Message {index}", ""])
            lines.extend([message.rstrip(), ""])
    else:
        lines.extend(["(none)", ""])

    lines.extend(["## Trace", ""])
    if parsed.turns:
        for index, turn in enumerate(parsed.turns, start=1):
            lines.extend([f"### Turn {index}", ""])
            for thinking in turn.thinking:
                lines.extend(["**agent** (thinking):", "", thinking.rstrip(), ""])
            for agent in turn.agent:
                lines.extend(["**agent**:", "", agent.rstrip(), ""])
            for tool in turn.tools:
                lines.extend(_render_tool(tool))
            if lines[-1] != "":
                lines.append("")
    else:
        lines.extend(["(no trace events captured)", ""])

    lines.extend(["## Final Response", ""])
    if parsed.final_response:
        lines.extend([parsed.final_response.rstrip(), ""])
    else:
        lines.extend(["(none)", ""])
    return "\n".join(lines).rstrip() + "\n"


def stringify_payload(payload: object) -> str:
    """Convert a tool-argument payload into readable text."""

    if isinstance(payload, str):
        return payload.strip()
    if payload is None:
        return ""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def truncate_tool_args(args: str) -> str:
    """Cap tool arguments to a bounded size with a total-length marker."""

    text = args.strip()
    if len(text) <= MAX_TOOL_ARGS_CHARS:
        return text
    clipped = text[:MAX_TOOL_ARGS_CHARS].rstrip()
    return f"{clipped}\n... [truncated, total {len(text)} chars]"


def result_bytes(payload: object) -> int:
    """Return the serialized byte length of a tool result payload."""

    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def success_text(success: bool | None) -> str:
    """Render a compact success marker."""

    if success is True:
        return "ok"
    if success is False:
        return "error"
    return "unknown"


def _render_tool(tool: ToolEvent) -> list[str]:
    args = truncate_tool_args(tool.args)
    result_line = (
        f"result: {success_text(tool.result_success)}, {tool.result_bytes} bytes"
        if tool.result_bytes is not None
        else "result: unavailable"
    )
    lines = [f"**tool**: `{tool.name}`", ""]
    if args:
        lines.extend(["args:", "", "```text", args, "```", ""])
    else:
        lines.extend(["args: (empty)", ""])
    lines.extend([result_line, ""])
    return lines


def _model_line(model: str | None, effort: str | None) -> str:
    if model and effort:
        return f"{model} (effort={effort})"
    return model or "unknown"


def _pair(left: str | None, right: str | None) -> str:
    return f"{left or 'unknown'} / {right or 'unknown'}"


def _duration_text(started_at: str | None, ended_at: str | None) -> str:
    if started_at is None or ended_at is None:
        return "unknown"
    started = _parse_timestamp(started_at)
    ended = _parse_timestamp(ended_at)
    if started is None or ended is None:
        return "unknown"
    seconds = max((ended - started).total_seconds(), 0.0)
    return f"{seconds:.1f}s"


def _parse_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
