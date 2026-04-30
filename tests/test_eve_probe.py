from __future__ import annotations

from pathlib import Path

from scaling_evolve.algorithms.eve import probe
from scaling_evolve.providers.agent.drivers.codex_tmux import CodexTmuxSessionDriver


def test_solver_probe_uses_workspace_seed_with_real_codex_tmux_driver(
    monkeypatch,
    tmp_path: Path,
) -> None:
    launches: list[dict[str, object]] = []

    def fake_launch(*, pane_id, cwd, env, argv, pane_title, banner_lines):  # noqa: ANN001
        launches.append(
            {
                "pane_id": pane_id,
                "cwd": cwd,
                "argv": argv,
                "pane_title": pane_title,
                "banner_lines": banner_lines,
            }
        )
        (cwd / "output" / "candidate.py").write_text(
            "VALUE = 2\n\n\ndef score() -> int:\n    return VALUE\n",
            encoding="utf-8",
        )
        session_path = (
            Path(env["HOME"]) / ".codex" / "sessions" / "2026" / "04" / "05" / "rollout-probe.jsonl"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"probe-session"}}',
                    (
                        '{"type":"event_msg","payload":{"type":"agent_message",'
                        '"message":"solver probe completed"}}'
                    ),
                    (
                        '{"type":"event_msg","payload":{"type":"task_complete",'
                        '"turn_id":"turn-1","last_agent_message":"solver probe completed"}}'
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
        role="solver",
        model="gpt-5.4-mini",
    )

    result = probe._probe_solver(tmp_path / "solver-probe", driver)  # noqa: SLF001

    assert result["agent"] == "solver"
    assert result["summary"] == "solver probe completed"
    assert result["changed_paths"] == ["output/candidate.py"]
    assert "return VALUE" in str(result["candidate"])
    assert launches[0]["pane_id"] == "%9"
    assert launches[0]["cwd"] == (tmp_path / "solver-probe").resolve()
