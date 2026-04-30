from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scaling_evolve.providers.agent.config import AgentProviderConfig
from scaling_evolve.providers.agent.drivers._metadata import TokenPricing
from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease
from scaling_evolve.providers.agent.drivers.codex_exec import (
    CodexExecCommandResult,
    CodexExecSessionDriver,
    CodexExecStreamSummary,
)

_LIVE_CODEX_EXEC = os.environ.get("SCALING_EVOLVE_RUN_LIVE_CODEX_EXEC_TESTS") == "1"
_PRICING_TABLE = {
    "gpt-5.4-mini": TokenPricing(
        input_per_million=0.75,
        output_per_million=4.5,
        cache_read_per_million=0.075,
    ),
}


def _init_git_repo(worktree: Path, *, initial_contents: str = "VALUE = 1\n") -> None:
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text(initial_contents, encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)


def _workspace_lease(worktree: Path, *, workspace_id: str = "attempt-1") -> SessionWorkspaceLease:
    return SessionWorkspaceLease(
        workspace_id=workspace_id,
        target_repo_root=str(worktree),
        workspace_root=str(worktree),
        session_cwd=str(worktree),
    )


def test_codex_exec_parse_stdout_extracts_session_id_usage_and_summary() -> None:
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"session-123"}',
            '{"type":"turn.started"}',
            (
                '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
                '"text":"Inspecting the file."}}'
            ),
            (
                '{"type":"item.completed","item":{"id":"item_1","type":"agent_message",'
                '"text":"Mutation finished."}}'
            ),
            (
                '{"type":"turn.completed","usage":{"input_tokens":10,'
                '"cached_input_tokens":2,"output_tokens":3}}'
            ),
        ]
    )

    parsed = CodexExecSessionDriver._parse_stdout_jsonl(stdout)

    assert parsed == CodexExecStreamSummary(
        session_id="session-123",
        summary="Mutation finished.",
        usage={
            "input_tokens": 10,
            "cached_input_tokens": 2,
            "output_tokens": 3,
            "agent_turns": 2,
        },
    )


def test_codex_exec_spawn_collects_diff_and_transcript(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree, initial_contents="x = 1\n")
    candidate = worktree / "candidate.py"
    commands: list[list[str]] = []

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        commands.append(command)
        assert cwd == worktree
        assert command[:3] == ["codex", "exec", "Do the task"]
        assert command[3:7] == [
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            "-C",
            str(worktree),
        ]
        assert "-m" in command
        assert "gpt-5.4-mini" in command
        assert "-c" in command
        assert 'model_reasoning_effort="low"' in command
        assert 'web_search="disabled"' in command

        stdout = "\n".join(
            [
                '{"type":"thread.started","thread_id":"session-abc"}',
                (
                    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
                    '"text":"mutation finished"}}'
                ),
                (
                    '{"type":"turn.completed","usage":{"input_tokens":10,'
                    '"cached_input_tokens":2,"output_tokens":3}}'
                ),
            ]
        )
        stdout_live_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_live_path.write_text(stdout + "\n", encoding="utf-8")

        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "09" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-abc"}}',
                    (
                        '{"type":"event_msg","payload":{"type":"agent_message",'
                        '"message":"mutation finished"}}'
                    ),
                    (
                        '{"type":"event_msg","payload":{"type":"token_count","info":'
                        '{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,'
                        '"output_tokens":3,"reasoning_output_tokens":1}}}}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        candidate.write_text("x = 2\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=stdout + "\n",
            stderr="warning",
        )

    monkeypatch.setattr(CodexExecSessionDriver, "_run_command", fake_run)

    driver = CodexExecSessionDriver(
        run_root=tmp_path / "run-root",
        model="gpt-5.4-mini",
        reasoning_effort="low",
        pricing_table=_PRICING_TABLE,
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=_workspace_lease(worktree),
        )
    )

    assert commands
    assert rollout.summary == "mutation finished"
    assert rollout.changed_paths == ["candidate.py"]
    assert rollout.usage is not None
    assert rollout.usage.input_tokens == 10
    assert rollout.usage.cache_read_tokens == 2
    assert rollout.usage.output_tokens == 3
    assert rollout.usage.agent_turns == 1
    assert rollout.usage.model_cost_usd == pytest.approx(0.00002115)
    assert rollout.state.session_id == "session-abc"
    assert rollout.state.metadata["driver"] == "codex_exec"
    assert Path(rollout.state.metadata["provider_transcript_path"]).exists()
    assert Path(rollout.state.metadata["provider_transcript_live_path"]).exists()
    assert Path(rollout.state.metadata["driver_stdout_live_path"]).exists()
    assert Path(rollout.state.metadata["codex_driver_home"]).exists()
    completion_payload = Path(rollout.state.metadata["completion_path"]).read_text(encoding="utf-8")
    assert '"changed_files": [' in completion_payload
    assert '"candidate.py"' in completion_payload


def test_codex_exec_resume_reuses_session_and_driver_home(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree, initial_contents="x = 1\n")
    candidate = worktree / "candidate.py"
    launches: list[list[str]] = []

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        launches.append(command)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "09" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        existing = session_path.read_text(encoding="utf-8") if session_path.exists() else ""
        summary_text = "spawn finished" if len(launches) == 1 else "repair finished"
        input_tokens = 10 if len(launches) == 1 else 7
        stdout = (
            "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "session-resume"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {
                                "id": "item_0",
                                "type": "agent_message",
                                "text": summary_text,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {
                                "input_tokens": input_tokens,
                                "cached_input_tokens": 1,
                                "output_tokens": 2,
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        stdout_live_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_live_path.write_text(stdout, encoding="utf-8")
        session_path.write_text(
            existing
            + "\n".join(
                (
                    [json.dumps({"type": "session_meta", "payload": {"id": "session-resume"}})]
                    if not existing
                    else []
                )
                + [
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "agent_message",
                                "message": summary_text,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": {
                                        "input_tokens": 7,
                                        "cached_input_tokens": 1,
                                        "output_tokens": 2,
                                        "reasoning_output_tokens": 0,
                                    }
                                },
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        candidate.write_text(f"x = {len(launches) + 1}\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(CodexExecSessionDriver, "_run_command", fake_run)

    driver = CodexExecSessionDriver(
        run_root=tmp_path / "run-root",
        model="gpt-5.4-mini",
        reasoning_effort="low",
        pricing_table=_PRICING_TABLE,
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=_workspace_lease(worktree),
        )
    )
    resumed = driver.resume(rollout.state, instruction="Repair the boundary issue")

    assert launches[1][:5] == [
        "codex",
        "exec",
        "resume",
        "session-resume",
        "Repair the boundary issue",
    ]
    assert "-C" not in launches[1]
    assert rollout.state.session_id == "session-resume"
    assert resumed.state.session_id == rollout.state.session_id
    assert resumed.summary == "repair finished"
    assert resumed.usage is not None
    assert resumed.usage.agent_turns == 1
    assert resumed.usage.model_cost_usd == pytest.approx(0.000014325)
    assert (
        resumed.state.metadata["codex_driver_home"] == rollout.state.metadata["codex_driver_home"]
    )
    assert Path(resumed.state.metadata["provider_transcript_path"]).exists()


def test_codex_exec_driver_config_accepts_driver_literal() -> None:
    provider = AgentProviderConfig.model_validate(
        {
            "kind": "agent_fork",
            "driver": "codex_exec",
            "model": "gpt-5.4-mini",
        }
    )

    assert provider.driver == "codex_exec"


def test_codex_exec_run_command_enforces_rollout_max_turns(tmp_path: Path) -> None:
    driver = CodexExecSessionDriver(
        run_root=tmp_path / "run-root",
        rollout_max_turns=1,
        timeout_seconds=5,
    )
    live_log = tmp_path / "live.jsonl"
    script = "\n".join(
        [
            "import json, sys, time",
            'print(json.dumps({"type": "thread.started", "thread_id": "session-1"}), flush=True)',
            (
                'print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", '
                '"text": "Inspecting"}}), flush=True)'
            ),
            "time.sleep(30)",
        ]
    )

    result = driver._run_command(  # noqa: SLF001
        command=[sys.executable, "-u", "-c", script],
        cwd=tmp_path,
        env={},
        stdout_live_path=live_log,
    )

    assert isinstance(result, CodexExecCommandResult)
    assert result.rollout_max_turns_reached is True
    assert result.observed_turns == 1
    assert '"agent_message"' in live_log.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not _LIVE_CODEX_EXEC,
    reason="Set SCALING_EVOLVE_RUN_LIVE_CODEX_EXEC_TESTS=1 to run live Codex exec tests.",
)
def test_codex_exec_live_spawn_edits_file(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    driver = CodexExecSessionDriver(
        run_root=tmp_path / "run-root",
        model="gpt-5.4-mini",
        reasoning_effort="low",
        timeout_seconds=240,
    )

    rollout = driver.spawn(
        SessionSeed(
            instruction=(
                "Open candidate.py, replace `VALUE = 1` with `VALUE = 2`, save the file, and stop."
            ),
            workspace=_workspace_lease(worktree),
        )
    )

    assert (worktree / "candidate.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert rollout.state.session_id is not None
    assert "candidate.py" in rollout.changed_paths
    assert Path(rollout.state.metadata["provider_transcript_path"]).exists()


@pytest.mark.skipif(
    not _LIVE_CODEX_EXEC,
    reason="Set SCALING_EVOLVE_RUN_LIVE_CODEX_EXEC_TESTS=1 to run live Codex exec tests.",
)
def test_codex_exec_live_resume_preserves_session_context(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    driver = CodexExecSessionDriver(
        run_root=tmp_path / "run-root",
        model="gpt-5.4-mini",
        reasoning_effort="low",
        timeout_seconds=240,
    )

    initial = driver.spawn(
        SessionSeed(
            instruction=(
                "Open candidate.py, replace `VALUE = 1` with `VALUE = 2`, save the file, and stop."
            ),
            workspace=_workspace_lease(worktree),
        )
    )
    resumed = driver.resume(
        initial.state,
        instruction=(
            "Continue in the same workspace. Replace `VALUE = 2` with `VALUE = 3`, "
            "save the file, and stop."
        ),
    )

    assert initial.state.session_id == resumed.state.session_id
    assert (worktree / "candidate.py").read_text(encoding="utf-8") == "VALUE = 3\n"


@pytest.mark.skipif(
    not _LIVE_CODEX_EXEC,
    reason="Set SCALING_EVOLVE_RUN_LIVE_CODEX_EXEC_TESTS=1 to run live Codex exec tests.",
)
def test_codex_exec_live_concurrent_spawn_x4(tmp_path: Path) -> None:
    def run_one(index: int) -> tuple[str | None, str]:
        worktree = tmp_path / f"repo-{index}"
        _init_git_repo(worktree, initial_contents=f"VALUE = {index}\n")
        driver = CodexExecSessionDriver(
            run_root=tmp_path / f"run-root-{index}",
            model="gpt-5.4-mini",
            reasoning_effort="low",
            timeout_seconds=240,
        )
        rollout = driver.spawn(
            SessionSeed(
                instruction=(
                    f"Open candidate.py, replace `VALUE = {index}` with `VALUE = {index + 10}`, "
                    "save the file, and stop."
                ),
                workspace=_workspace_lease(worktree, workspace_id=f"attempt-{index}"),
            )
        )
        return rollout.state.session_id, (worktree / "candidate.py").read_text(encoding="utf-8")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_one, range(1, 5)))

    session_ids = [session_id for session_id, _ in results]
    assert len(set(session_ids)) == 4
    assert [contents for _, contents in results] == [
        "VALUE = 11\n",
        "VALUE = 12\n",
        "VALUE = 13\n",
        "VALUE = 14\n",
    ]
