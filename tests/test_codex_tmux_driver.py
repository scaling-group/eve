from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scaling_evolve.providers.agent.drivers._metadata import TokenPricing
from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease
from scaling_evolve.providers.agent.drivers.codex_tmux import (
    CodexCompletionState,
    CodexTmuxSessionDriver,
)

_PRICING_TABLE = {
    "gpt-5.4-mini": TokenPricing(
        input_per_million=0.75,
        output_per_million=4.5,
        cache_read_per_million=0.075,
    ),
}


def test_codex_tmux_spawn_collects_done_file_diff_and_session_log(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    candidate = worktree / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        assert pane_id == "%9"
        assert argv[0] == "codex"
        assert argv[1:5] == ["--no-alt-screen", "-a", "never", "-s"]
        assert argv[5] == "workspace-write"
        assert "-m" in argv
        assert "gpt-5.4-mini" in argv
        assert 'model_reasoning_effort="low"' in argv
        assert pane_title == "Gen ? | TASK | Slot ?"
        assert banner_lines[0] == "Gen ? | TASK | Slot ?"
        candidate.write_text("x = 2\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "probe"], check=True)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
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
                        '{"type":"event_msg","payload":{"type":"task_complete",'
                        '"turn_id":"turn-1","last_agent_message":"mutation finished"}}'
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

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: cleaned.append(pane_id),
    )
    reset_waits: list[str] = []
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: reset_waits.append(pane_id),
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        model="gpt-5.4-mini",
        reasoning_effort="low",
        pricing_table=_PRICING_TABLE,
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    assert rollout.summary == "mutation finished"
    assert rollout.changed_paths == ["candidate.py"]
    assert rollout.usage is not None
    assert rollout.usage.input_tokens == 10
    assert rollout.usage.cache_read_tokens == 2
    assert rollout.usage.output_tokens == 3
    assert rollout.usage.agent_turns == 1
    assert rollout.usage.model_cost_usd == pytest.approx(0.00002115)
    assert rollout.usage.wallclock_seconds is not None
    assert rollout.usage.wallclock_seconds < 10
    assert rollout.state.session_id == "session-abc"
    assert rollout.state.metadata["driver"] == "codex_tmux"
    diff_path = Path(rollout.state.metadata["diff_path"])
    assert diff_path.exists()
    assert "candidate.py" in diff_path.read_text(encoding="utf-8")
    assert Path(rollout.state.metadata["provider_transcript_path"]).exists()
    assert Path(rollout.state.metadata["codex_driver_home"]).exists()
    assert Path(rollout.state.metadata["attempt_root"]).name == ".codex-driver-transcripts"
    completion_payload = json.loads(Path(rollout.state.metadata["completion_path"]).read_text())
    assert completion_payload["changed_files"] == ["candidate.py"]
    assert cleaned == ["%9"]
    assert reset_waits == ["%9"]
    assert "instruction_path" not in rollout.state.metadata


def test_codex_tmux_spawn_uses_explicit_display_context_for_pane_title(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    titles: list[str] = []

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        titles.append(pane_title)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-xyz"}}',
                    '{"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1","last_agent_message":"done"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        role="solver",
    )
    driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
            display_context={"iteration": 2, "worker_index": 1},
        )
    )

    assert titles == ["[solver] Gen 2 | TASK | Slot 1"]


def test_codex_tmux_spawn_can_use_existing_prompt_file_without_rewriting_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    readme = worktree / "README.md"
    readme.write_text("# Existing prompt\n", encoding="utf-8")
    (worktree / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "README.md", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    launch_args: list[list[str]] = []

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        launch_args.append(argv)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-readme"}}',
                    '{"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1","last_agent_message":"done"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(pane_id="%9", run_root=tmp_path / "run-root", role="solver")
    rollout = driver.spawn(
        SessionSeed(
            instruction="unused teacher prompt",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
            prompt_file="README.md",
            write_prompt_file=False,
        )
    )

    assert launch_args[0][-1] == "unused teacher prompt"
    assert readme.read_text(encoding="utf-8") == "# Existing prompt\n"
    assert Path(rollout.state.metadata["instruction_path"]).name == "README.md"
    assert "runtime_contract_path" not in rollout.state.metadata


def test_codex_tmux_spawn_synthesizes_completion_when_agent_exits_without_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    candidate = worktree / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        candidate.write_text("x = 2\n", encoding="utf-8")
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-no-completion"}}',
                    (
                        '{"type":"event_msg","payload":{"type":"agent_message",'
                        '"message":"finished without completion file"}}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    pane_dead_state = {"calls": 0}

    def fake_pane_dead(pane_id):  # noqa: ANN001
        pane_dead_state["calls"] += 1
        return pane_dead_state["calls"] > 1

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        fake_pane_dead,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(pane_id="%9", run_root=tmp_path / "run-root")
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    archived_completion = Path(rollout.state.metadata["completion_path"])
    payload = json.loads(archived_completion.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["summary"] == "finished without completion file"
    assert payload["changed_files"] == ["candidate.py"]


def test_codex_tmux_spawn_skips_session_archive_when_jsonl_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    candidate = worktree / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    missing_session_jsonl = tmp_path / "missing-rollout.jsonl"

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        lambda **kwargs: candidate.write_text("x = 2\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    def fake_wait(self, **kwargs):  # noqa: ANN001, ARG001
        return CodexCompletionState(
            session_jsonl=missing_session_jsonl,
            task_complete_payload={"last_agent_message": "done without transcript"},
            session_jsonl_line_count=0,
        )

    monkeypatch.setattr(CodexTmuxSessionDriver, "_wait_for_completion", fake_wait)

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    assert rollout.summary == "done without transcript"
    assert rollout.changed_paths == ["candidate.py"]
    assert rollout.state.metadata.get("provider_transcript_path") is None
    assert rollout.state.metadata.get("provider_transcript_live_path") is None
    assert rollout.state.session_id


def test_codex_tmux_wait_for_completion_enforces_rollout_max_turns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    live_jsonl = tmp_path / "rollout.jsonl"
    live_jsonl.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"session-abc"}}',
                (
                    '{"type":"event_msg","payload":{"type":"agent_message",'
                    '"message":"Inspecting the workspace."}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.latest_rollout_path",
        lambda codex_home, after_mtime_ns=None: live_jsonl,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.time.sleep",
        lambda seconds: None,
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        rollout_max_turns=1,
    )
    completion = driver._wait_for_completion(
        isolated_home=type("Home", (), {"root": tmp_path})(),
        existing_session_jsonl=None,
        line_count_before_launch=0,
        launch_started_ns=0,
        pane_id="%9",
    )

    assert completion.rollout_max_turns_reached is True
    assert completion.observed_turns == 1


def test_codex_tmux_spawn_enables_search_only_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    commands: list[list[str]] = []

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        commands.append(argv)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-search"}}',
                    '{"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1","last_agent_message":"done"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        web_search="live",
    )
    driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    assert "--search" in commands[0]
    assert "-c" in commands[0]
    assert 'web_search="live"' in commands[0]


def test_codex_tmux_spawn_explicitly_disables_search_features_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    commands: list[list[str]] = []

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        commands.append(argv)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-search-disabled"}}',
                    '{"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1","last_agent_message":"done"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        web_search="disabled",
    )
    driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    assert "--search" not in commands[0]
    assert "-c" in commands[0]
    assert 'web_search="disabled"' in commands[0]


def test_codex_tmux_spawn_cleans_pane_on_timeout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        lambda **kwargs: None,
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: cleaned.append(pane_id),
    )
    reset_waits: list[str] = []
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: reset_waits.append(pane_id),
    )

    def fake_wait(self, **kwargs):  # noqa: ANN001, ARG001
        raise TimeoutError("boom")

    monkeypatch.setattr(CodexTmuxSessionDriver, "_wait_for_completion", fake_wait)
    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
    )

    with pytest.raises(TimeoutError, match="boom"):
        driver.spawn(
            SessionSeed(
                instruction="Do the task",
                workspace=SessionWorkspaceLease(
                    workspace_id="attempt-1",
                    target_repo_root=str(worktree),
                    workspace_root=str(worktree),
                    session_cwd=str(worktree),
                ),
            )
        )

    assert cleaned == ["%9"]
    assert reset_waits == ["%9"]


def test_codex_tmux_resume_uses_same_session_and_workspace_driver_home(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    candidate = worktree / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    launches: list[list[str]] = []
    session_path = tmp_path / "live-rollout.jsonl"

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        launches.append(argv)
        assert pane_id == "%9"
        assert cwd == worktree
        if len(launches) == 1:
            assert "Do the task" in argv[-1]
        else:
            assert "Repair the boundary issue" in argv[-1]
        candidate.write_text("x = 2\n", encoding="utf-8")
        session_path.parent.mkdir(parents=True, exist_ok=True)
        existing = session_path.read_text(encoding="utf-8") if session_path.exists() else ""
        session_path.write_text(
            existing
            + "\n".join(
                (
                    ['{"type":"session_meta","payload":{"id":"session-resume"}}']
                    if not existing
                    else []
                )
                + [
                    (
                        '{"type":"event_msg","payload":{"type":"agent_message",'
                        '"message":"repair finished"}}'
                    ),
                    (
                        '{"type":"event_msg","payload":{"type":"task_complete",'
                        f'"turn_id":"turn-{len(launches)}","last_agent_message":"repair finished"}}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    wait_calls: list[tuple[Path | None, int, str | None]] = []

    def fake_wait(
        self,
        *,
        isolated_home,
        existing_session_jsonl,
        line_count_before_launch,
        launch_started_ns,
        pane_id,
    ):  # noqa: ANN001
        _ = (self, isolated_home, launch_started_ns, pane_id)
        wait_calls.append((existing_session_jsonl, line_count_before_launch, launches[-1][-1]))
        return CodexCompletionState(
            session_jsonl=session_path,
            task_complete_payload={"last_agent_message": "repair finished"},
            session_jsonl_line_count=len(session_path.read_text(encoding="utf-8").splitlines()),
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        CodexTmuxSessionDriver,
        "_resolve_live_session_path",
        lambda self, *, isolated_home, session_id, metadata: session_path,  # noqa: ARG005
    )
    monkeypatch.setattr(CodexTmuxSessionDriver, "_wait_for_completion", fake_wait)
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        model="gpt-5.4-mini",
        reasoning_effort="low",
        role="solver",
        pricing_table=_PRICING_TABLE,
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    resumed = driver.resume(rollout.state, instruction="Repair the boundary issue")

    assert launches[1][:8] == [
        "codex",
        "--no-alt-screen",
        "-a",
        "never",
        "-s",
        "workspace-write",
        "-C",
        str(worktree),
    ]
    assert "resume" in launches[1]
    assert rollout.state.session_id in launches[1]
    assert rollout.state.session_id == "session-resume"
    assert resumed.state.session_id == rollout.state.session_id
    assert wait_calls[0][0] is None
    assert wait_calls[0][1] == 0
    assert wait_calls[1][0] == session_path
    assert wait_calls[1][1] > 0
    assert (
        resumed.state.metadata["codex_driver_home"] == rollout.state.metadata["codex_driver_home"]
    )
    assert resumed.state.metadata["pane_id"] == "%9"
    assert resumed.summary == "repair finished"
    assert resumed.usage is not None
    assert resumed.usage.agent_turns == 1


def test_codex_tmux_waits_for_root_turn_task_complete_not_subagent_completion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    candidate = worktree / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        candidate.write_text("x = 2\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "probe"], check=True)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "05" / "rollout-test.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-root"}}',
                    '{"type":"event_msg","payload":{"type":"task_started","turn_id":"root-turn"}}',
                    '{"type":"event_msg","payload":{"type":"task_started","turn_id":"subagent-turn"}}',
                    (
                        '{"type":"event_msg","payload":{"type":"task_complete",'
                        '"turn_id":"subagent-turn","last_agent_message":"subagent pass"}}'
                    ),
                    (
                        '{"type":"event_msg","payload":{"type":"agent_message",'
                        '"message":"root finished"}}'
                    ),
                    (
                        '{"type":"event_msg","payload":{"type":"task_complete",'
                        '"turn_id":"root-turn","last_agent_message":"root finished"}}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = CodexTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-1",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    assert rollout.summary == "root finished"
