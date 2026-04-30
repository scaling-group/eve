from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scaling_evolve.core.engine import RuntimeStateRef
from scaling_evolve.providers.agent.drivers._metadata import TokenPricing
from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease
from scaling_evolve.providers.agent.drivers.claude_code_tmux import (
    ClaudeCodeCompletionState,
    ClaudeCodeTmuxSessionDriver,
    _project_bucket_dir,
)

_PRICING_TABLE = {
    "claude-sonnet-4-6": TokenPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.3,
    ),
}


def _init_git_repo(worktree: Path) -> None:
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


def test_claude_code_tmux_spawn_archives_transcript_and_cleans_live_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)

    live_root = tmp_path / "live-projects" / "bucket"
    live_jsonl = live_root / "session-abc.jsonl"
    live_subagents_dir = live_root / "session-abc" / "subagents"

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        assert pane_id == "%9"
        assert cwd == worktree
        assert argv[0] == "claude"
        assert "--setting-sources" in argv
        assert "--disallowedTools" in argv
        assert "WebSearch,WebFetch" in argv
        assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "auto"
        candidate = worktree / "candidate.py"
        candidate.write_text("x = 2\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "probe"], check=True)

        live_subagents_dir.mkdir(parents=True, exist_ok=True)
        (live_subagents_dir / "agent-1.jsonl").write_text("{}", encoding="utf-8")
        live_jsonl.parent.mkdir(parents=True, exist_ok=True)
        live_jsonl.write_text(
            "\n".join(
                [
                    '{"type":"permission-mode","sessionId":"session-abc"}',
                    (
                        '{"type":"assistant","requestId":"req-1","message":{"content":'
                        '[{"type":"text","text":"mutation finished"}],"stop_reason":"end_turn",'
                        '"usage":{"input_tokens":10,"cache_read_input_tokens":2,"output_tokens":3}}}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        ClaudeCodeTmuxSessionDriver,
        "_wait_for_completion",
        lambda self, **kwargs: ClaudeCodeCompletionState(  # noqa: ARG005
            session_jsonl=live_jsonl,
            session_id="session-abc",
            session_jsonl_line_count=2,
        ),
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        model="claude-sonnet-4-6",
        effort_level="auto",
        disallowed_tools=("WebSearch", "WebFetch"),
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
    assert rollout.usage.model_cost_usd == pytest.approx(0.0000756)
    assert rollout.state.session_id == "session-abc"
    archive_path = Path(rollout.state.metadata["provider_transcript_path"])
    assert archive_path.exists()
    assert not live_jsonl.exists()
    assert not live_subagents_dir.exists()
    assert Path(rollout.state.metadata["provider_transcript_subagents_path"]).exists()
    assert rollout.state.metadata["provider_transcript_live_path"] == str(live_jsonl)


def test_claude_code_tmux_resume_restores_archived_transcript_and_uses_new_lines_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)

    transcript_root = worktree / ".claude-driver-transcripts" / "session-resume"
    transcript_root.mkdir(parents=True, exist_ok=True)
    archive_jsonl = transcript_root / "session-resume.jsonl"
    archive_jsonl.write_text(
        "\n".join(
            [
                '{"type":"permission-mode","sessionId":"session-resume"}',
                (
                    '{"type":"assistant","requestId":"req-old","message":{"content":'
                    '[{"type":"text","text":"old summary"}],"stop_reason":"end_turn",'
                    '"usage":{"input_tokens":100,"cache_read_input_tokens":20,"output_tokens":30}}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    archive_subagents = transcript_root / "subagents"
    archive_subagents.mkdir()
    (archive_subagents / "agent-old.jsonl").write_text("{}", encoding="utf-8")

    live_jsonl = tmp_path / "live-projects" / "bucket" / "session-resume.jsonl"
    launches: list[list[str]] = []

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        launches.append(argv)
        assert "--resume" in argv
        assert "session-resume" in argv
        assert live_jsonl.exists()
        candidate = worktree / "candidate.py"
        candidate.write_text("x = 3\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "repair"], check=True)
        live_jsonl.write_text(
            archive_jsonl.read_text(encoding="utf-8")
            + "\n".join(
                [
                    (
                        '{"type":"assistant","requestId":"req-new","message":{"content":'
                        '[{"type":"text","text":"repair finished"}],"stop_reason":"end_turn",'
                        '"usage":{"input_tokens":7,"cache_read_input_tokens":1,"output_tokens":2}}}'
                    )
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        ClaudeCodeTmuxSessionDriver,
        "_wait_for_completion",
        lambda self, **kwargs: ClaudeCodeCompletionState(  # noqa: ARG005
            session_jsonl=live_jsonl,
            session_id="session-resume",
            session_jsonl_line_count=3,
        ),
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )

    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        model="claude-sonnet-4-6",
        pricing_table=_PRICING_TABLE,
    )
    resumed = driver.resume(
        RuntimeStateRef(
            state_id="runtime:test",
            provider_kind="claude_code_tmux",
            session_id="session-resume",
            workspace_id="attempt-1",
            target_repo_root=str(worktree),
            workspace_root=str(worktree),
            session_cwd=str(worktree),
            metadata={
                "driver": "claude_code_tmux",
                "provider_transcript_live_path": str(live_jsonl),
                "provider_transcript_path": str(archive_jsonl),
                "provider_transcript_subagents_path": str(archive_subagents),
            },
        ),
        instruction="Repair the boundary issue",
    )

    assert launches
    assert resumed.summary == "repair finished"
    assert resumed.usage is not None
    assert resumed.usage.input_tokens == 7
    assert resumed.usage.cache_read_tokens == 1
    assert resumed.usage.output_tokens == 2
    assert resumed.usage.agent_turns == 1
    assert resumed.usage.model_cost_usd == pytest.approx(0.0000513)
    assert resumed.state.session_id == "session-resume"
    assert Path(resumed.state.metadata["provider_transcript_path"]).exists()
    assert not live_jsonl.exists()


def test_claude_code_tmux_wait_for_completion_requires_no_active_task_after_signal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    live_jsonl = worktree / "session-abc.jsonl"
    live_jsonl.write_text(
        "\n".join(
            [
                '{"type":"permission-mode","sessionId":"session-abc"}',
                (
                    '{"type":"assistant","requestId":"req-1","message":{"content":'
                    '[{"type":"text","text":"done"}],"stop_reason":"end_turn",'
                    '"usage":{"input_tokens":1,"cache_read_input_tokens":0,"output_tokens":1}}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    signal_file = worktree / ".claude-task-stopped"
    signal_file.touch()
    captures = iter(
        (
            "❯ continue\n1 local agent\n",
            "❯ done\n",
        )
    )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux._latest_session_jsonl",
        lambda worktree_root, after_mtime_ns=None: live_jsonl,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.capture_pane_tail",
        lambda pane_id, tail_lines=80: next(captures),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.time.sleep",
        lambda seconds: None,
    )

    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
    )
    completion = driver._wait_for_completion(
        worktree_root=worktree,
        existing_session_jsonl=None,
        line_count_before_launch=0,
        launch_started_ns=0,
        pane_id="%9",
    )

    assert completion.session_jsonl == live_jsonl
    assert completion.session_id == "session-abc"
    assert completion.session_jsonl_line_count == 2


def test_claude_code_tmux_wait_for_completion_enforces_rollout_max_turns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    live_jsonl = worktree / "session-abc.jsonl"
    live_jsonl.write_text(
        "\n".join(
            [
                '{"type":"permission-mode","sessionId":"session-abc"}',
                (
                    '{"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":'
                    '[{"type":"tool_use","id":"call-1","name":"Read","input":{"file_path":"candidate.py"}}]}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux._latest_session_jsonl",
        lambda worktree_root, after_mtime_ns=None: live_jsonl,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.pane_dead",
        lambda pane_id: False,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.time.sleep",
        lambda seconds: None,
    )

    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        rollout_max_turns=1,
    )
    completion = driver._wait_for_completion(
        worktree_root=worktree,
        existing_session_jsonl=None,
        line_count_before_launch=0,
        launch_started_ns=0,
        pane_id="%9",
    )

    assert completion.rollout_max_turns_reached is True
    assert completion.observed_turns == 1


def test_claude_code_tmux_spawn_recovers_transcript_when_wait_returns_without_jsonl(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)

    live_root = _project_bucket_dir(worktree)
    live_jsonl = live_root / "session-late.jsonl"

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        _ = (pane_id, env, argv, pane_title, banner_lines)
        assert cwd == worktree
        candidate = worktree / "candidate.py"
        candidate.write_text("x = 4\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "late-transcript"], check=True)
        live_jsonl.parent.mkdir(parents=True, exist_ok=True)
        live_jsonl.write_text(
            "\n".join(
                [
                    '{"type":"permission-mode","sessionId":"session-late"}',
                    (
                        '{"type":"assistant","requestId":"req-1","message":{"content":'
                        '[{"type":"text","text":"late transcript recovered"}],'
                        '"stop_reason":"end_turn",'
                        '"usage":{"input_tokens":11,"cache_read_input_tokens":4,"output_tokens":5}}}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    lookup_count = {"value": 0}

    def fake_latest_session_jsonl(worktree_root: Path, after_mtime_ns=None):  # noqa: ANN001
        _ = (worktree_root, after_mtime_ns)
        lookup_count["value"] += 1
        if lookup_count["value"] < 3:
            return None
        return live_jsonl

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.launch_in_pane",
        fake_launch,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux._latest_session_jsonl",
        fake_latest_session_jsonl,
    )
    monkeypatch.setattr(
        ClaudeCodeTmuxSessionDriver,
        "_wait_for_completion",
        lambda self, **kwargs: ClaudeCodeCompletionState(  # noqa: ARG005
            session_jsonl=None,
            session_id=None,
            session_jsonl_line_count=0,
        ),
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.kill_pane",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.wait_for_pane_reset",
        lambda pane_id: None,
    )
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.claude_code_tmux.time.sleep",
        lambda seconds: None,
    )

    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=SessionWorkspaceLease(
                workspace_id="attempt-late",
                target_repo_root=str(worktree),
                workspace_root=str(worktree),
                session_cwd=str(worktree),
            ),
        )
    )

    assert rollout.summary == "late transcript recovered"
    assert rollout.usage is not None
    assert rollout.usage.input_tokens == 11
    assert rollout.usage.cache_read_tokens == 4
    assert rollout.usage.output_tokens == 5
    assert Path(rollout.state.metadata["provider_transcript_path"]).exists()
    assert not live_jsonl.exists()
    assert lookup_count["value"] >= 3


def test_claude_code_tmux_argv_emits_disallowed_tools_flag() -> None:
    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root="/tmp/run-root",
        model="opus",
        disallowed_tools=("WebSearch", "WebFetch"),
    )

    argv = driver._argv(  # noqa: SLF001
        session_id=None,
        instruction="Do work.",
        prompt_file=None,
    )

    assert "--disallowedTools" in argv
    assert "WebSearch,WebFetch" in argv


def test_project_bucket_dir_slugifies_full_workspace_path(tmp_path: Path) -> None:
    worktree = (
        tmp_path
        / ".runs"
        / "eve"
        / "circle-packing"
        / "run-20260409_025002_291960-07d5e0036e8f"
        / "solver_workspaces"
        / "20260408T185004_step_1_a42028d44652"
    )
    worktree.mkdir(parents=True)

    bucket = _project_bucket_dir(worktree)

    resolved = str(worktree.resolve())
    expected_slug = re.sub(r"[^A-Za-z0-9]", "-", resolved)
    assert bucket.name == expected_slug
