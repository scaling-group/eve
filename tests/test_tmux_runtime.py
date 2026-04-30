from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scaling_evolve.providers.agent import tmux_runtime


def test_create_grid_and_launch_in_pane(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    outputs = iter(["%1\n", "%2\n", "%3\n", "%4\n", "", "", "", "", "0\n"])

    def fake_run(command, check, capture_output, text):  # noqa: ANN001
        calls.append(command)
        return SimpleNamespace(stdout=next(outputs), returncode=0)

    monkeypatch.setattr(tmux_runtime.subprocess, "run", fake_run)

    grid = tmux_runtime.create_grid(session_name="evolve-run-test", cwd=tmp_path)
    tmux_runtime.launch_in_pane(
        pane_id="%1",
        cwd=tmp_path,
        env={"HOME": "/tmp/demo-home"},
        argv=["codex", "--flag", "value"],
    )
    tmux_runtime.kill_pane("%1")
    assert tmux_runtime.pane_dead("%1") is False

    assert grid.task_panes == ("%1", "%2")
    assert grid.eval_panes == ("%3", "%4")
    assert calls[0][:3] == ["tmux", "new-session", "-d"]
    assert calls[1][:3] == ["tmux", "split-window", "-d"]
    assert calls[4][:3] == ["tmux", "set-option", "-t"]
    assert calls[5][:3] == ["tmux", "set-option", "-t"]
    assert calls[6][1] == "respawn-pane"
    assert "HOME=/tmp/demo-home" in calls[6][-1]
    assert "codex --flag value" in calls[6][-1]
    assert calls[7][:3] == ["tmux", "respawn-pane", "-k"]
    assert calls[7][-1].startswith("/bin/sh -lc ")
    assert calls[8][:3] == ["tmux", "display-message", "-p"]


def test_create_single_pane_session(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    outputs = iter(["%9\n", "", ""])

    def fake_run(command, check, capture_output, text):  # noqa: ANN001
        calls.append(command)
        return SimpleNamespace(stdout=next(outputs), returncode=0)

    monkeypatch.setattr(tmux_runtime.subprocess, "run", fake_run)

    session = tmux_runtime.create_single_pane_session(
        session_name="evolve-probe-test",
        cwd=tmp_path,
    )

    assert session.session_name == "evolve-probe-test"
    assert session.pane_id == "%9"
    assert calls[0][:3] == ["tmux", "new-session", "-d"]
    assert calls[1][:3] == ["tmux", "set-option", "-t"]
    assert calls[2][:3] == ["tmux", "set-option", "-t"]


def test_create_pane_pool_session(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    outputs = iter(["%9\n", "%10\n", "%11\n", "", ""])

    def fake_run(command, check, capture_output, text):  # noqa: ANN001
        calls.append(command)
        return SimpleNamespace(stdout=next(outputs), returncode=0)

    monkeypatch.setattr(tmux_runtime.subprocess, "run", fake_run)

    session = tmux_runtime.create_pane_pool_session(
        session_name="evolve-pool-test",
        cwd=tmp_path,
        pane_count=3,
    )

    assert session.session_name == "evolve-pool-test"
    assert session.pane_ids == ("%9", "%11", "%10")
    assert calls[0][:3] == ["tmux", "new-session", "-d"]
    assert calls[1][1] == "split-window"
    assert calls[2][1] == "split-window"
    assert calls[3][:3] == ["tmux", "set-option", "-t"]
    assert calls[4][:3] == ["tmux", "set-option", "-t"]


def test_open_iterm2_window_for_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(tmux_runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(tmux_runtime.shutil, "which", lambda name: "/usr/bin/osascript")

    def fake_run(command, input, capture_output, text, check):  # noqa: ANN001
        captured["command"] = command
        captured["input"] = input
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tmux_runtime.subprocess, "run", fake_run)

    assert tmux_runtime.open_iterm2_window_for_session("evolve-run-test") is True
    assert captured["command"] == ["osascript"]
    assert "tmux attach -t evolve-run-test" in str(captured["input"])


def test_wait_for_pane_reset_polls_until_shell(monkeypatch) -> None:
    commands = iter(["codex", "codex", "zsh"])
    monkeypatch.setattr(
        tmux_runtime,
        "pane_current_command",
        lambda pane_id: next(commands),  # noqa: ARG005
    )
    monkeypatch.setattr(tmux_runtime.time, "sleep", lambda seconds: None)  # noqa: ARG005

    tmux_runtime.wait_for_pane_reset("%1", timeout_seconds=1.0, poll_interval_seconds=0.0)


def test_kill_session_restores_mouse_on(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_tmux(args, check=True):  # noqa: ANN001
        calls.append([*args, str(check)])
        return ""

    monkeypatch.setattr(tmux_runtime, "_run_tmux", fake_run_tmux)

    tmux_runtime.kill_session("evolve-run-test")

    assert calls == [
        ["kill-session", "-t", "evolve-run-test", "False"],
        ["set-option", "-g", "mouse", "on", "False"],
    ]


def test_show_banner_in_pane_sets_title_and_respawns_idle_shell(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, list[str], bool]] = []

    def fake_run_tmux(args, check=True):  # noqa: ANN001
        calls.append(("tmux", list(args), check))
        return ""

    monkeypatch.setattr(tmux_runtime, "_run_tmux", fake_run_tmux)

    tmux_runtime.show_banner_in_pane(
        pane_id="%1",
        cwd=tmp_path,
        title="Gen 2 | EVAL | waiting",
        lines=("waiting for task phase to finish",),
    )

    assert calls[0] == (
        "tmux",
        ["set-option", "-p", "-t", "%1", "@phase_label", "Gen 2 | EVAL | waiting"],
        False,
    )
    assert calls[1] == ("tmux", ["select-pane", "-t", "%1", "-T", "Gen 2 | EVAL | waiting"], False)
    assert calls[2][1][:3] == ["respawn-pane", "-k", "-t"]
    assert "Gen 2 | EVAL | waiting" in calls[2][1][-1]
    assert "waiting for task phase to finish" in calls[2][1][-1]
