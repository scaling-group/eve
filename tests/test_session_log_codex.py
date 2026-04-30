from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scaling_evolve.providers.agent.session_log import build_session_log_markdown


def test_build_session_log_markdown_for_codex_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "codex.jsonl"
    long_args = '{"cmd":"' + ("x" * 700) + '"}'
    transcript_path.write_text(
        "\n".join(
            [
                (
                    '{"timestamp":"2026-04-05T10:00:00Z","type":"session_meta","payload":'
                    '{"id":"session-root","cwd":"'
                    "/tmp/solver_workspaces/20260405T100000_step_3_abcd"
                    '"}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:00Z","type":"turn_context","payload":'
                    '{"cwd":"'
                    "/tmp/solver_workspaces/20260405T100000_step_3_abcd"
                    '","model":"gpt-5.4-mini","effort":"medium"}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:01Z","type":"event_msg","payload":'
                    '{"type":"user_message","message":"Read README first."}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:02Z","type":"event_msg","payload":'
                    '{"type":"agent_message","message":"Inspecting the workspace."}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:03Z","type":"response_item","payload":'
                    '{"type":"function_call","name":"exec_command","arguments":'
                    + __import__("json").dumps(long_args)
                    + ',"call_id":"call_1"}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:04Z","type":"response_item","payload":'
                    '{"type":"function_call_output","call_id":"call_1","output":'
                    '"Command...\\nProcess exited with code 0\\nSENSITIVE TOOL OUTPUT"}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:05Z","type":"event_msg","payload":'
                    '{"type":"user_message","message":"Repair the boundary issue."}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:06Z","type":"event_msg","payload":'
                    '{"type":"agent_message","message":"Boundary fixed and checks passed."}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="session-root",
            metadata={
                "driver": "codex_tmux",
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=12, output_tokens=6, cache_read_tokens=3),
        summary="Boundary fixed and checks passed.",
    )

    markdown = build_session_log_markdown([rollout, rollout])

    assert markdown is not None
    assert "# Session Log - solver (iter 3)" in markdown
    assert "- **Provider**: codex_tmux" in markdown
    assert "- **Model**: gpt-5.4-mini (effort=medium)" in markdown
    assert "- **Spawn/resume count**: 2" in markdown
    assert "Read README first." in markdown
    assert "Repair the boundary issue." in markdown
    assert "Inspecting the workspace." in markdown
    assert "Boundary fixed and checks passed." in markdown
    assert "result: ok," in markdown
    assert "... [truncated, total" in markdown
    assert "SENSITIVE TOOL OUTPUT" not in markdown


def test_build_session_log_markdown_for_codex_exec_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "codex-exec.jsonl"
    tool_args = '{"cmd":"sed -n 1,40p candidate.py"}'
    transcript_path.write_text(
        "\n".join(
            [
                (
                    '{"timestamp":"2026-04-09T08:23:40Z","type":"session_meta","payload":'
                    '{"id":"session-exec","cwd":"'
                    "/tmp/solver_workspaces/20260409T082340_step_2_abcd"
                    '"}}'
                ),
                (
                    '{"timestamp":"2026-04-09T08:23:40Z","type":"turn_context","payload":'
                    '{"cwd":"'
                    "/tmp/solver_workspaces/20260409T082340_step_2_abcd"
                    '","model":"gpt-5.4-mini","effort":"low"}}'
                ),
                (
                    '{"timestamp":"2026-04-09T08:23:41Z","type":"event_msg","payload":'
                    '{"type":"user_message","message":"Patch candidate.py."}}'
                ),
                (
                    '{"timestamp":"2026-04-09T08:23:42Z","type":"event_msg","payload":'
                    '{"type":"agent_message","message":"Applying the requested patch."}}'
                ),
                (
                    '{"timestamp":"2026-04-09T08:23:43Z","type":"response_item","payload":'
                    '{"type":"function_call","name":"exec_command","arguments":'
                    + __import__("json").dumps(tool_args)
                    + ',"call_id":"call_1"}}'
                ),
                (
                    '{"timestamp":"2026-04-09T08:23:44Z","type":"response_item","payload":'
                    '{"type":"function_call_output","call_id":"call_1","output":'
                    '"Command...\\nProcess exited with code 0\\nVALUE = 1"}}'
                ),
                (
                    '{"timestamp":"2026-04-09T08:23:45Z","type":"event_msg","payload":'
                    '{"type":"agent_message","message":"Patch applied."}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rollout = SimpleNamespace(
        state=SimpleNamespace(
            session_id="session-exec",
            metadata={
                "driver": "codex_exec",
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=8, output_tokens=4, cache_read_tokens=2),
        summary="Patch applied.",
    )

    markdown = build_session_log_markdown([rollout])

    assert markdown is not None
    assert "- **Provider**: codex_exec" in markdown
    assert "# Session Log - solver (iter 2)" in markdown
