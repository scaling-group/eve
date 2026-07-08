"""Shared transcript helpers for rollout turn counting and tool-batch inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TranscriptTurnState:
    """Observed turn state derived from one transcript file."""

    format_name: Literal["codex_exec", "codex_tmux", "unknown"]
    turn_count: int
    latest_batch_tool_ids: tuple[str, ...]


def inspect_transcript_turn_state(transcript_path: Path) -> TranscriptTurnState:
    """Return the current turn count and latest tool batch ids for one transcript."""

    if not transcript_path.exists():
        return TranscriptTurnState("unknown", 0, ())

    exec_turn_count = 0
    exec_latest_batch: tuple[str, ...] = ()
    exec_current_batch: list[str] = []

    tmux_turn_count = 0
    tmux_latest_batch: tuple[str, ...] = ()
    tmux_current_batch: list[str] = []

    has_exec_indicator = False
    has_tmux_indicator = False

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue

            p_type = payload.get("type")

            if p_type == "item.completed":
                has_exec_indicator = True
                item = payload.get("item")
                if isinstance(item, dict):
                    itype = item.get("type")
                    if itype == "agent_message":
                        exec_turn_count += 1
                    elif itype == "function_call":
                        cid = item.get("call_id")
                        if isinstance(cid, str):
                            exec_current_batch.append(cid)
                        if tmux_current_batch:
                            tmux_latest_batch = tuple(tmux_current_batch)
                            tmux_current_batch = []
                        continue
            elif p_type == "response_item":
                has_tmux_indicator = True
                item = payload.get("payload")
                if isinstance(item, dict):
                    itype = item.get("type")
                    if itype == "function_call":
                        cid = item.get("call_id")
                        if isinstance(cid, str):
                            tmux_current_batch.append(cid)
                        if exec_current_batch:
                            exec_latest_batch = tuple(exec_current_batch)
                            exec_current_batch = []
                        continue
            elif p_type == "event_msg":
                has_tmux_indicator = True
                ev = payload.get("payload")
                if isinstance(ev, dict) and ev.get("type") == "agent_message":
                    tmux_turn_count += 1
            elif p_type == "thread.started":
                has_exec_indicator = True

            # Common reset for all other cases and fall-throughs
            if exec_current_batch:
                exec_latest_batch = tuple(exec_current_batch)
                exec_current_batch = []
            if tmux_current_batch:
                tmux_latest_batch = tuple(tmux_current_batch)
                tmux_current_batch = []

    # Final batch finalization
    if exec_current_batch:
        exec_latest_batch = tuple(exec_current_batch)
    if tmux_current_batch:
        tmux_latest_batch = tuple(tmux_current_batch)

    if has_exec_indicator:
        return TranscriptTurnState("codex_exec", exec_turn_count, exec_latest_batch)
    if has_tmux_indicator:
        return TranscriptTurnState("codex_tmux", tmux_turn_count, tmux_latest_batch)
    return TranscriptTurnState("unknown", 0, ())
