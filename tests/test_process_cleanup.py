from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scaling_evolve.algorithms.eve.workflow.evaluation import _run_shell_step
from scaling_evolve.providers.agent.drivers._subprocess import (
    _LIVE_PROCESSES,
    _LIVE_TEMPORARY_DIRECTORIES,
    kill_all_live_process_trees,
    run_with_live_log,
    tracked_temporary_directory,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture(autouse=True)
def _balanced_process_registry() -> Iterator[None]:
    processes_before = set(_LIVE_PROCESSES)
    directories_before = set(_LIVE_TEMPORARY_DIRECTORIES)
    yield
    assert processes_before == set(_LIVE_PROCESSES)
    assert directories_before == set(_LIVE_TEMPORARY_DIRECTORIES)


def test_live_log_timeout_kills_descendants(tmp_path: Path) -> None:
    grandchild_pid_file = tmp_path / "grandchild.pid"
    grandchild = "; ".join(
        [
            "import os, time",
            f"open({str(grandchild_pid_file)!r}, 'w').write(str(os.getpid()))",
            "time.sleep(30)",
        ]
    )
    script = "\n".join(
        [
            "import pathlib, subprocess, sys, time",
            f"pid_file = pathlib.Path({str(grandchild_pid_file)!r})",
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}], start_new_session=True)",
            "while not pid_file.exists(): time.sleep(0.01)",
            "print('started', flush=True)",
            "time.sleep(30)",
        ]
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_live_log(
            command=[sys.executable, "-u", "-c", script],
            cwd=tmp_path,
            env=None,
            live_log_path=tmp_path / "live.log",
            timeout=1.0,
        )

    grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8"))
    assert _wait_until(lambda: not _process_alive(grandchild_pid))


def test_abort_cleanup_kills_registered_live_log_process(tmp_path: Path) -> None:
    processes_before = set(_LIVE_PROCESSES)

    def rollout() -> subprocess.CompletedProcess[str]:
        return run_with_live_log(
            command=[sys.executable, "-u", "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=None,
            live_log_path=tmp_path / "live.log",
            timeout=60,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rollout)
        assert _wait_until(lambda: bool(set(_LIVE_PROCESSES) - processes_before) or future.done())
        assert set(_LIVE_PROCESSES) - processes_before
        kill_all_live_process_trees()
        completed = future.result(timeout=5)

    assert completed.returncode == -signal.SIGKILL


def test_abort_cleanup_removes_registered_temporary_directory(tmp_path: Path) -> None:
    with tracked_temporary_directory(
        prefix="scaling-evolve-test-",
        parent=tmp_path,
    ) as runtime_root:
        assert runtime_root.parent == tmp_path
        (runtime_root / "future-version-cache").write_text("cache\n", encoding="utf-8")

        kill_all_live_process_trees()

        assert not runtime_root.exists()


def test_temporary_directory_stays_registered_until_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_rmtree = shutil.rmtree
    removal_calls = 0

    def abort_during_first_removal(path: str | Path) -> None:
        nonlocal removal_calls
        removal_calls += 1
        if removal_calls == 1:
            assert Path(path) in _LIVE_TEMPORARY_DIRECTORIES
            kill_all_live_process_trees()
            return
        real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", abort_during_first_removal)
    with tracked_temporary_directory(
        prefix="scaling-evolve-test-",
        parent=tmp_path,
    ) as runtime_root:
        (runtime_root / "future-version-cache").write_text("cache\n", encoding="utf-8")

    assert removal_calls == 2
    assert not runtime_root.exists()


def test_codex_turn_limit_terminates_descendants(tmp_path: Path) -> None:
    grandchild_pid_file = tmp_path / "grandchild.pid"
    grandchild = "; ".join(
        [
            "import os, time",
            f"open({str(grandchild_pid_file)!r}, 'w').write(str(os.getpid()))",
            "time.sleep(30)",
        ]
    )
    script = "\n".join(
        [
            "import json, pathlib, subprocess, sys, time",
            f"pid_file = pathlib.Path({str(grandchild_pid_file)!r})",
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}], start_new_session=True)",
            "while not pid_file.exists(): time.sleep(0.01)",
            (
                'print(json.dumps({"type": "item.completed", "item": '
                '{"type": "agent_message", "text": "Inspecting"}}), flush=True)'
            ),
            "time.sleep(30)",
        ]
    )
    driver = CodexExecSessionDriver(
        run_root=tmp_path / "run-root",
        rollout_max_turns=1,
        timeout_seconds=5,
    )

    result = driver._run_command(  # noqa: SLF001
        command=[sys.executable, "-u", "-c", script],
        cwd=tmp_path,
        env={},
        stdout_live_path=tmp_path / "live.jsonl",
    )

    assert result.rollout_max_turns_reached is True
    grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8"))
    assert _wait_until(lambda: not _process_alive(grandchild_pid))


def test_shell_evaluation_is_registered_for_abort_cleanup(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    grandchild_pid_file = tmp_path / "grandchild.pid"
    step = tmp_path / "evaluation.sh"
    step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"{shlex.quote(sys.executable)} - <<'PY' &",
                "import os, time",
                "os.setsid()",
                f"open({str(grandchild_pid_file)!r}, 'w').write(str(os.getpid()))",
                "time.sleep(30)",
                "PY",
                "wait",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    processes_before = set(_LIVE_PROCESSES)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_shell_step,
            step=step,
            step_index=1,
            workspace_root=workspace_root,
        )
        assert _wait_until(
            lambda: (
                (bool(set(_LIVE_PROCESSES) - processes_before) and grandchild_pid_file.exists())
                or future.done()
            )
        )
        assert set(_LIVE_PROCESSES) - processes_before
        assert grandchild_pid_file.exists()
        kill_all_live_process_trees()
        with pytest.raises(RuntimeError, match="Evaluation shell step failed"):
            future.result(timeout=5)

    grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8"))
    assert _wait_until(lambda: not _process_alive(grandchild_pid))
    assert (
        workspace_root / "logs" / "evaluate" / "steps" / "step_01_evaluation" / "status.txt"
    ).read_text(encoding="utf-8") == "failed\n"
