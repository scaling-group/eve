"""Helpers for transcript-based compaction detection."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scaling_evolve.core.common import DomainModel, JSONValue, _parse_timestamp


class CompactEvent(DomainModel):
    """One detected transcript compaction boundary."""

    timestamp: datetime
    line_number: int
    trigger: str | None = None
    pre_tokens: int | None = None


def detect_compact_events(transcript_path: Path) -> list[CompactEvent]:
    """Parse one session JSONL transcript and return detected compaction events."""

    events: list[CompactEvent] = []
    with transcript_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "system" or payload.get("subtype") != "compact_boundary":
                continue
            timestamp = _parse_timestamp(payload.get("timestamp"))
            if timestamp is None:
                continue
            compact_metadata = payload.get("compactMetadata")
            if not isinstance(compact_metadata, dict):
                compact_metadata = {}
            pre_tokens = compact_metadata.get("preTokens")
            events.append(
                CompactEvent(
                    timestamp=timestamp,
                    line_number=line_number,
                    trigger=(
                        str(compact_metadata["trigger"])
                        if isinstance(compact_metadata.get("trigger"), str)
                        else None
                    ),
                    pre_tokens=pre_tokens if isinstance(pre_tokens, int) else None,
                )
            )
    return events


def compact_metadata_from_transcript(transcript_path: Path) -> dict[str, JSONValue]:
    """Build compact metadata payload persisted onto runtime or edge records."""

    events = detect_compact_events(transcript_path)
    payload: dict[str, JSONValue] = {"compact_event_count": len(events)}
    if events:
        last_event = events[-1]
        payload["compact_boundary_timestamp"] = last_event.timestamp.isoformat()
        payload["last_compact_line_number"] = last_event.line_number
    return payload
