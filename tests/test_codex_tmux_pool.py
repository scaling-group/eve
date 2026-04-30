from __future__ import annotations

import subprocess
from pathlib import Path

from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease
from scaling_evolve.providers.agent.drivers.codex_tmux import (
    CodexCompletionState,
    CodexTmuxPanePool,
    CodexTmuxSessionDriver,
)
from scaling_evolve.providers.agent.tmux_runtime import TmuxPanePoolSession


def test_codex_tmux_pool_reuses_one_pane_across_solver_eval_optimizer(
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

    banner_titles: list[str] = []
    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.show_banner_in_pane",
        lambda *, pane_id, cwd, title, lines=(): banner_titles.append(title),  # noqa: ARG005
    )
    pool = CodexTmuxPanePool.from_session(
        session=TmuxPanePoolSession(session_name="pool-test", pane_ids=("%7",)),
        cwd=tmp_path,
    )

    launches: list[tuple[str, str]] = []
    residue_checks: list[tuple[bool, bool, bool]] = []
    session_paths: dict[str, Path] = {}

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        launches.append((pane_id, pane_title))
        residue_checks.append(
            (
                (cwd / ".evolve-done.json").exists(),
                (cwd / ".codex-runtime.md").exists(),
                (cwd / ".evolve-instruction.md").exists(),
            )
        )
        assert residue_checks[-1] == (False, False, False)
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "05" / f"{pane_title}.jsonl"
        )
        session_paths[pane_title] = session_path
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"session-pool"}}',
                    '{"type":"event_msg","payload":{"type":"agent_message","message":"done"}}',
                    '{"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1","last_agent_message":"done"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def fake_wait(
        self,
        *,
        isolated_home,
        existing_session_jsonl,
        line_count_before_launch,
        launch_started_ns,
        pane_id,
    ):  # noqa: ANN001, ARG001
        _ = (
            self,
            isolated_home,
            existing_session_jsonl,
            line_count_before_launch,
            launch_started_ns,
        )
        pane_title = launches[-1][1]
        return CodexCompletionState(
            session_jsonl=session_paths[pane_title],
            task_complete_payload={"last_agent_message": "done"},
            session_jsonl_line_count=3,
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.drivers.codex_tmux.launch_in_pane",
        fake_launch,
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

    drivers = [
        CodexTmuxSessionDriver(
            pane_pool=pool,
            run_root=tmp_path / "run-root",
            role=role,
            model="gpt-5.4-mini",
        )
        for role in ("solver", "eval", "optimizer")
    ]
    for index, driver in enumerate(drivers, start=1):
        rollout = driver.spawn(
            SessionSeed(
                instruction=f"Role {driver.role}",
                workspace=SessionWorkspaceLease(
                    workspace_id=f"attempt-{index}",
                    target_repo_root=str(worktree),
                    workspace_root=str(worktree),
                    session_cwd=str(worktree),
                ),
            )
        )
        assert rollout.state.metadata["pane_id"] == "%7"

    assert launches == [
        ("%7", "[solver] Gen ? | TASK | Slot ?"),
        ("%7", "[eval] Gen ? | TASK | Slot ?"),
        ("%7", "[optimizer] Gen ? | TASK | Slot ?"),
    ]
    assert residue_checks == [(False, False, False)] * 3
    assert banner_titles[0] == "Codex Pool | idle"
    assert banner_titles[-1] == "Codex Pool | idle"
