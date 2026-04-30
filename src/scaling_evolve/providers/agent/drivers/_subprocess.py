"""Shared subprocess helpers for non-interactive session drivers."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


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
        )
        try:
            _, stderr_text = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
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
