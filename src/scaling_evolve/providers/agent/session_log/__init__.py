"""Driver-agnostic session-log builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scaling_evolve.providers.agent.drivers.base import SessionRollout
from scaling_evolve.providers.agent.session_log.claude_code_parser import (
    parse_claude_code_session,
)
from scaling_evolve.providers.agent.session_log.codex_parser import parse_codex_session
from scaling_evolve.providers.agent.session_log.schema import render_session_markdown


def build_session_log_markdown(rollouts: list[SessionRollout]) -> str | None:
    """Build a markdown session log from the latest rollout transcript."""

    if not rollouts:
        return None
    rollout = rollouts[-1]
    metadata = getattr(rollout.state, "metadata", {}) or {}
    transcript_path = metadata.get("provider_transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return None
    parsed = _dispatch_parser(
        Path(transcript_path).expanduser(),
        rollout=rollout,
        driver_name=_driver_name(metadata),
    )
    if parsed is None:
        return None
    return render_session_markdown(
        parsed,
        rollout_count=len(rollouts),
        usage=_summarize_usage(rollouts),
    )


def _dispatch_parser(
    transcript_path: Path,
    *,
    rollout: SessionRollout,
    driver_name: str | None,
):
    if driver_name in {"codex_tmux", "codex_exec"}:
        parsed = parse_codex_session(transcript_path, session_id=rollout.state.session_id)
        if parsed is not None and driver_name is not None:
            parsed.provider = driver_name
        return parsed
    if driver_name in {"claude_code", "claude_code_tmux"}:
        return parse_claude_code_session(
            transcript_path,
            session_id=rollout.state.session_id,
            effort=_driver_effort(getattr(rollout.state, "metadata", {}) or {}),
        )
    return None


def _driver_name(metadata: dict[str, Any]) -> str | None:
    driver = metadata.get("driver")
    if isinstance(driver, str) and driver:
        return driver
    execution = metadata.get("driver_execution")
    if isinstance(execution, dict):
        driver = execution.get("driver")
        if isinstance(driver, str) and driver:
            return driver
    return None


def _driver_effort(metadata: dict[str, Any]) -> str | None:
    execution = metadata.get("driver_execution")
    if not isinstance(execution, dict):
        return None
    effort = execution.get("effort_level")
    return effort if isinstance(effort, str) and effort else None


def _summarize_usage(rollouts: list[SessionRollout]) -> dict[str, int | float]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
    }
    for rollout in rollouts:
        usage = getattr(rollout, "usage", None)
        if usage is None:
            continue
        totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
        totals["cache_read_tokens"] += int(getattr(usage, "cache_read_tokens", 0) or 0)
    return totals
