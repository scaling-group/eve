from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scaling_evolve.providers.agent.session_log import build_session_log_markdown


def _line(event_type: str, *, timestamp: int, part: dict[str, object]) -> str:
    return json.dumps(
        {
            "type": event_type,
            "timestamp": timestamp,
            "sessionID": "ses_log",
            "part": part,
        }
    )


def test_build_session_log_markdown_for_cumulative_opencode_transcript(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "opencode.jsonl"
    cwd = "/private/solver_workspaces/run_step_4_worker"
    records = [
        json.dumps(
            {
                "type": "eve.rollout",
                "timestamp": 1_700_000_000_000,
                "sessionID": "ses_log",
                "instruction": "Patch candidate.py.",
                "model": "deepseek/deepseek-chat",
                "variant": "high",
                "role": "solver",
                "cwd": cwd,
            }
        ),
        _line(
            "step_start",
            timestamp=1_700_000_000_100,
            part={"type": "step-start"},
        ),
        _line(
            "reasoning",
            timestamp=1_700_000_000_200,
            part={"type": "reasoning", "text": "Inspecting the candidate."},
        ),
        _line(
            "tool_use",
            timestamp=1_700_000_000_300,
            part={
                "type": "tool",
                "tool": "edit",
                "callID": "call_1",
                "state": {
                    "status": "completed",
                    "input": {"filePath": "candidate.py"},
                    "output": "PRIVATE TOOL OUTPUT",
                },
            },
        ),
        _line(
            "text",
            timestamp=1_700_000_000_400,
            part={"type": "text", "text": "Initial patch complete."},
        ),
        _line(
            "step_finish",
            timestamp=1_700_000_000_500,
            part={"type": "step-finish"},
        ),
        json.dumps(
            {
                "type": "eve.rollout",
                "timestamp": 1_700_000_001_000,
                "sessionID": "ses_log",
                "instruction": "Repair the boundary issue.",
                "model": "deepseek/deepseek-chat",
                "variant": "high",
                "role": "solver",
                "cwd": cwd,
            }
        ),
        _line(
            "step_start",
            timestamp=1_700_000_001_100,
            part={"type": "step-start"},
        ),
        _line(
            "tool_use",
            timestamp=1_700_000_001_200,
            part={
                "type": "tool",
                "tool": "bash",
                "callID": "call_2",
                "state": {
                    "status": "error",
                    "input": {"command": "false"},
                    "error": "exit 1",
                },
            },
        ),
        _line(
            "text",
            timestamp=1_700_000_001_300,
            part={"type": "text", "text": "Boundary repaired."},
        ),
        _line(
            "step_finish",
            timestamp=1_700_000_001_400,
            part={"type": "step-finish"},
        ),
    ]
    transcript_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="ses_log",
            metadata={
                "driver": "opencode",
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=12, output_tokens=7, cache_read_tokens=3),
        summary="Boundary repaired.",
    )

    markdown = build_session_log_markdown([rollout, rollout])

    assert markdown is not None
    assert "# Session Log - solver (iter 4)" in markdown
    assert "- **Provider**: opencode" in markdown
    assert "- **Model**: deepseek/deepseek-chat (effort=high)" in markdown
    assert "- **Spawn/resume count**: 2" in markdown
    assert "Patch candidate.py." in markdown
    assert "Repair the boundary issue." in markdown
    assert "Inspecting the candidate." in markdown
    assert "Initial patch complete." in markdown
    assert "Boundary repaired." in markdown
    assert "result: ok," in markdown
    assert "result: error," in markdown
    assert "PRIVATE TOOL OUTPUT" not in markdown
