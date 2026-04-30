from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scaling_evolve.providers.agent.session_log import build_session_log_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM_JSON_PROBE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stream_json_probes"


def test_build_session_log_markdown_for_claude_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "claude.jsonl"
    long_command = "python3 - <<'PY'\n" + ("print('x')\n" * 120) + "PY"
    records = [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:00Z",
            "content": "Read README first.",
        },
        {
            "type": "user",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:01Z",
            "cwd": "/tmp/solver_workspaces/20260405T100000_step_4_abcd",
            "message": {"role": "user", "content": "Read README first."},
        },
        {
            "type": "assistant",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:02Z",
            "cwd": "/tmp/solver_workspaces/20260405T100000_step_4_abcd",
            "message": {
                "role": "assistant",
                "model": "openai/gpt-5.4-mini-20260317",
                "content": [
                    {"type": "thinking", "thinking": "First, inspect the workspace carefully."},
                    {"type": "text", "text": "Checking files and preparing a patch."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Bash",
                        "input": {"command": long_command, "description": "Run the check."},
                    },
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:03Z",
            "cwd": "/tmp/solver_workspaces/20260405T100000_step_4_abcd",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "SECRET TOOL RESULT BODY",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:04Z",
            "cwd": "/tmp/solver_workspaces/20260405T100000_step_4_abcd",
            "message": {
                "role": "assistant",
                "model": "openai/gpt-5.4-mini-20260317",
                "content": [{"type": "text", "text": "Finished the workspace update."}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "other-session",
            "timestamp": "2026-04-05T10:00:05Z",
            "cwd": "/tmp/solver_workspaces/20260405T100000_step_4_abcd",
            "message": {
                "role": "assistant",
                "model": "openai/gpt-5.4-mini-20260317",
                "content": [{"type": "text", "text": "ignore me"}],
            },
        },
    ]
    transcript_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="session-root",
            metadata={
                "driver_execution": {
                    "driver": "claude_code",
                    "effort_level": "low",
                },
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=20, output_tokens=8, cache_read_tokens=5),
        summary="Finished the workspace update.",
    )

    markdown = build_session_log_markdown([rollout])

    assert markdown is not None
    assert "# Session Log - solver (iter 4)" in markdown
    assert "- **Provider**: claude_code" in markdown
    assert "- **Model**: openai/gpt-5.4-mini-20260317 (effort=low)" in markdown
    assert "Read README first." in markdown
    assert "First, inspect the workspace carefully." in markdown
    assert "Checking files and preparing a patch." in markdown
    assert "Finished the workspace update." in markdown
    assert "result: ok," in markdown
    assert "... [truncated, total" in markdown
    assert "SECRET TOOL RESULT BODY" not in markdown
    assert "ignore me" not in markdown


def test_build_session_log_markdown_for_claude_tmux_transcript_uses_claude_parser(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "claude-tmux.jsonl"
    transcript_path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "session-root",
                "timestamp": "2026-04-09T10:00:00Z",
                "cwd": "/tmp/solver_workspaces/20260409T100000_step_4_abcd",
                "message": {
                    "role": "assistant",
                    "model": "claude-haiku-4-5-20251001",
                    "content": [{"type": "text", "text": "tmux transcript parsed"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="session-root",
            metadata={
                "driver_execution": {
                    "driver": "claude_code_tmux",
                    "effort_level": "medium",
                },
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=3, output_tokens=2, cache_read_tokens=1),
        summary="tmux transcript parsed",
    )

    markdown = build_session_log_markdown([rollout])

    assert markdown is not None
    assert "- **Provider**: claude_code" in markdown
    assert "tmux transcript parsed" in markdown


def test_claude_tool_results_match_tool_use_across_turns(tmp_path: Path) -> None:
    transcript_path = tmp_path / "claude-delayed.jsonl"
    records = [
        {
            "type": "user",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:00Z",
            "cwd": "/tmp/optimizer_workspaces/20260405T100000_step_1_abcd",
            "message": {"role": "user", "content": "Review the optimizer."},
        },
        {
            "type": "assistant",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:01Z",
            "cwd": "/tmp/optimizer_workspaces/20260405T100000_step_1_abcd",
            "message": {
                "role": "assistant",
                "model": "openai/gpt-5.4-mini-20260317",
                "content": [
                    {"type": "text", "text": "First pass."},
                    {
                        "type": "tool_use",
                        "id": "call_a",
                        "name": "Bash",
                        "input": {"command": "git status --short"},
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:02Z",
            "cwd": "/tmp/optimizer_workspaces/20260405T100000_step_1_abcd",
            "message": {
                "role": "assistant",
                "model": "openai/gpt-5.4-mini-20260317",
                "content": [
                    {"type": "text", "text": "Second pass."},
                    {
                        "type": "tool_use",
                        "id": "call_b",
                        "name": "Bash",
                        "input": {"command": "git diff --stat"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "session-root",
            "timestamp": "2026-04-05T10:00:03Z",
            "cwd": "/tmp/optimizer_workspaces/20260405T100000_step_1_abcd",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_a",
                        "content": "status output",
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_b",
                        "content": "diff output",
                        "is_error": False,
                    },
                ],
            },
        },
    ]
    transcript_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="session-root",
            metadata={
                "driver_execution": {"driver": "claude_code"},
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=10, output_tokens=4, cache_read_tokens=1),
        summary="Second pass.",
    )

    markdown = build_session_log_markdown([rollout])

    assert markdown is not None
    turn_1 = markdown.split("### Turn 1", 1)[1].split("### Turn 2", 1)[0]
    turn_2 = markdown.split("### Turn 2", 1)[1].split("## Final Response", 1)[0]
    assert "git status --short" in turn_1
    assert "result: ok," in turn_1
    assert "unavailable" not in turn_1
    assert "git diff --stat" in turn_2
    assert "result: ok," in turn_2
    assert "unavailable" not in turn_2
    assert "**tool**: `tool_result`" not in markdown


@pytest.mark.skipif(
    not STREAM_JSON_PROBE_ROOT.exists(),
    reason="stream-json probe fixtures not present (gitignored)",
)
def test_build_session_log_markdown_for_noninteractive_stream_json_fixture() -> None:
    transcript_path = STREAM_JSON_PROBE_ROOT / "p1_basic_agent_loop.jsonl"

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="b1fe867f-7aa9-4b4a-bf47-665af1f708fe",
            metadata={
                "driver_execution": {
                    "driver": "claude_code",
                    "effort_level": "medium",
                },
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=34, output_tokens=1148, cache_read_tokens=104291),
        summary="Fixed!",
    )

    markdown = build_session_log_markdown([rollout])

    assert markdown is not None
    assert "- **Session ID**: b1fe867f-7aa9-4b4a-bf47-665af1f708fe" in markdown
    assert "- **Provider**: claude_code" in markdown
    assert "**tool**: `Read`" in markdown
    assert "**tool**: `Edit`" in markdown
    assert "**tool**: `Bash`" in markdown
    assert "Fixed! Changed line 5 from `radius ** 3` to `radius ** 2`." in markdown


@pytest.mark.skipif(
    not STREAM_JSON_PROBE_ROOT.exists(),
    reason="stream-json probe fixtures not present (gitignored)",
)
def test_build_session_log_markdown_for_resume_stream_json_fixture() -> None:
    transcript_path = STREAM_JSON_PROBE_ROOT / "p3_resume_followup.jsonl"

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="bb8e7e11-6b5a-470a-88b5-cafcda373115",
            metadata={
                "driver_execution": {
                    "driver": "claude_code",
                    "effort_level": "medium",
                },
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=23, output_tokens=400, cache_read_tokens=90871),
        summary="Done!",
    )

    markdown = build_session_log_markdown([rollout])

    assert markdown is not None
    assert "- **Session ID**: bb8e7e11-6b5a-470a-88b5-cafcda373115" in markdown
    assert "ask_name()" in markdown
