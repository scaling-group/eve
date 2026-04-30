"""Tests for compaction event detection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scaling_evolve.providers.agent.compaction import detect_compact_events


def test_detect_compact_events_reads_compact_boundaries(tmp_path: Path) -> None:
    transcript_path = tmp_path / "session.jsonl"
    rows = [
        {"type": "system", "subtype": "init", "timestamp": "2026-03-27T12:00:00.000Z"},
        {
            "type": "system",
            "subtype": "compact_boundary",
            "timestamp": "2026-03-27T12:30:00.000Z",
            "compactMetadata": {"trigger": "auto", "preTokens": 6200},
        },
        {"type": "assistant", "timestamp": "2026-03-27T12:31:00.000Z"},
        {
            "type": "system",
            "subtype": "compact_boundary",
            "timestamp": "2026-03-27T13:45:12.250Z",
            "compactMetadata": {"trigger": "manual", "preTokens": 7100},
        },
    ]
    transcript_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    events = detect_compact_events(transcript_path)

    assert len(events) == 2
    assert events[0].line_number == 2
    assert events[0].timestamp == datetime(2026, 3, 27, 12, 30, tzinfo=UTC)
    assert events[0].trigger == "auto"
    assert events[0].pre_tokens == 6200
    assert events[1].line_number == 4
    assert events[1].timestamp == datetime(2026, 3, 27, 13, 45, 12, 250000, tzinfo=UTC)
    assert events[1].trigger == "manual"
    assert events[1].pre_tokens == 7100


def test_detect_compact_events_handles_empty_and_malformed_jsonl(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")

    malformed_path = tmp_path / "mixed.jsonl"
    malformed_path.write_text(
        "\n".join(
            [
                '{"type":"system","subtype":"compact_boundary","timestamp":"2026-03-27T12:30:00Z"}',
                "{not-json}",
                '{"type":"assistant","timestamp":"2026-03-27T12:31:00Z"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert detect_compact_events(empty_path) == []
    assert len(detect_compact_events(malformed_path)) == 1
