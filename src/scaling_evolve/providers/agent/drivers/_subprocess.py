"""Shared subprocess helpers for non-interactive session drivers."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

_LIVE_PROCESSES: dict[int, subprocess.Popen[str]] = {}
_LIVE_TEMPORARY_DIRECTORIES: set[Path] = set()


def _process_tree(root_pid: int) -> list[int]:
    """Return a root process and its descendants from one process-table snapshot."""
    completed = subprocess.run(
        ["ps", "-Ao", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=False,
    )
    children: dict[int, list[int]] = {}
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            pid, parent_pid = (int(field) for field in fields)
            children.setdefault(parent_pid, []).append(pid)

    process_ids: list[int] = []
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        process_ids.append(pid)
        pending.extend(children.get(pid, ()))
    return process_ids


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a subprocess and every current descendant."""
    for pid in reversed(_process_tree(process.pid)):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)


def kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Freeze and kill a subprocess tree, including descendants that called setsid."""
    process_ids = [process.pid]
    known_process_ids = {process.pid}
    with suppress(ProcessLookupError, PermissionError):
        os.kill(process.pid, signal.SIGSTOP)

    # Each pass freezes newly discovered children before taking another snapshot,
    # so a stopped parent cannot spawn a descendant that escapes the next pass.
    while True:
        new_process_ids = [
            pid for pid in _process_tree(process.pid) if pid not in known_process_ids
        ]
        if not new_process_ids:
            break
        for pid in new_process_ids:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGSTOP)
        process_ids.extend(new_process_ids)
        known_process_ids.update(new_process_ids)

    for pid in reversed(process_ids):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


@contextmanager
def tracked_process_tree(process: subprocess.Popen[str]) -> Iterator[None]:
    """Track a subprocess tree for runner abort cleanup."""
    _LIVE_PROCESSES[process.pid] = process
    try:
        yield
    finally:
        if process.poll() is None:
            kill_process_tree(process)
        _LIVE_PROCESSES.pop(process.pid, None)


@contextmanager
def tracked_temporary_directory(*, prefix: str, parent: Path | None = None) -> Iterator[Path]:
    """Create a per-rollout directory that runner abort cleanup can remove."""
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    _LIVE_TEMPORARY_DIRECTORIES.add(path)
    try:
        yield path
    finally:
        try:
            with suppress(FileNotFoundError):
                shutil.rmtree(path)
        finally:
            _LIVE_TEMPORARY_DIRECTORIES.discard(path)


def kill_all_live_process_trees() -> None:
    """Clean up every live subprocess tree and per-rollout runtime directory."""
    try:
        for process in list(_LIVE_PROCESSES.values()):
            kill_process_tree(process)
    finally:
        for path in list(_LIVE_TEMPORARY_DIRECTORIES):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            finally:
                _LIVE_TEMPORARY_DIRECTORIES.discard(path)


def run_with_live_log(
    command: Sequence[str],
    cwd: str | Path,
    env: Mapping[str, str] | None,
    live_log_path: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    live_log_path = live_log_path.expanduser()
    live_log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    process_env.update(dict(env or {}))
    stderr_text = ""
    with live_log_path.open("w", encoding="utf-8") as stdout_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=process_env,
            start_new_session=True,
        )
        with tracked_process_tree(process):
            try:
                _, stderr_text = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                kill_process_tree(process)
                _, stderr_text = process.communicate()
                stdout_handle.flush()
                raise subprocess.TimeoutExpired(
                    error.cmd,
                    error.timeout,
                    output=_read_text_if_exists(live_log_path),
                    stderr=stderr_text,
                ) from error
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode,
        stdout=_read_text_if_exists(live_log_path),
        stderr=stderr_text,
    )


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
