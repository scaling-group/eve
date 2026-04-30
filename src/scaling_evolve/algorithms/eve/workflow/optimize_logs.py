"""Helpers for persisting optimize-session logs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from scaling_evolve.algorithms.eve.workspace.file_tree import read_file_tree
from scaling_evolve.providers.agent.drivers.base import SessionRollout
from scaling_evolve.providers.agent.session_log import build_session_log_markdown


def build_optimize_log_tree(
    workspace: Path,
    rollouts: list[SessionRollout],
) -> dict[str, str]:
    """Read agent-written optimize logs and append token usage metadata."""

    logs = read_file_tree(workspace / "logs" / "optimize")
    final_response = _final_response(rollouts)
    if final_response is not None:
        logs["final_response.txt"] = final_response.rstrip() + "\n"
    logs["token_usage.json"] = (
        json.dumps(build_usage_report(rollouts), indent=2, sort_keys=True) + "\n"
    )
    session_markdown = build_session_log_markdown(rollouts)
    if session_markdown is not None:
        logs["session.md"] = session_markdown
    return logs


def build_usage_report(rollouts: list[SessionRollout]) -> dict[str, object]:
    """Return a usage report with total plus per-attempt entries."""

    return {
        "total": summarize_rollout_usage(rollouts),
        "attempts": [_usage_payload(rollout) for rollout in rollouts],
    }


def summarize_rollout_usage(rollouts: list[SessionRollout]) -> dict[str, int | float]:
    """Return total token/cost usage across one optimize run."""

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "agent_turns": 0,
        "model_cost_usd": 0.0,
        "wallclock_seconds": 0.0,
    }
    for rollout in rollouts:
        usage = getattr(rollout, "usage", None)
        if usage is None:
            continue
        totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
        totals["cache_read_tokens"] += int(getattr(usage, "cache_read_tokens", 0) or 0)
        totals["cache_creation_tokens"] += int(getattr(usage, "cache_creation_tokens", 0) or 0)
        totals["agent_turns"] += int(getattr(usage, "agent_turns", 0) or 0)
        totals["model_cost_usd"] += float(getattr(usage, "model_cost_usd", 0.0) or 0.0)
        totals["wallclock_seconds"] += float(getattr(usage, "wallclock_seconds", 0.0) or 0.0)
    return totals


def _usage_payload(rollout: SessionRollout) -> dict[str, object]:
    usage = getattr(rollout, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        payload = usage.model_dump(mode="json")
    elif isinstance(usage, Mapping):
        payload = dict(usage)
    else:
        payload = dict(vars(usage))
    return {key: value for key, value in payload.items() if value is not None}


def _final_response(rollouts: list[SessionRollout]) -> str | None:
    for rollout in reversed(rollouts):
        final_response = getattr(rollout, "summary", None)
        if isinstance(final_response, str) and final_response.strip():
            return final_response
    return None
