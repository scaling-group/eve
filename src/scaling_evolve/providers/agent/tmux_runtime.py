"""Minimal tmux helpers for the demo Codex runtime."""

from __future__ import annotations

import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TmuxGrid:
    """Resolved 2x2 tmux layout for one evolve run."""

    session_name: str
    task_panes: tuple[str, str]
    eval_panes: tuple[str, str]

    @property
    def all_panes(self) -> tuple[str, str, str, str]:
        return (*self.task_panes, *self.eval_panes)


@dataclass(frozen=True)
class TmuxPaneSession:
    """Resolved single-pane tmux session for probe-style runs."""

    session_name: str
    pane_id: str


@dataclass(frozen=True)
class TmuxPanePoolSession:
    """Resolved single-window multi-pane session for pool-backed runs."""

    session_name: str
    pane_ids: tuple[str, ...]


_PROMPT_RE = re.compile(r"^\s*[›>❯]\s*", re.UNICODE)
_BOOTSTRAP_PATTERNS = (
    re.compile(r"\b(loading|initializing|starting up)\b", re.IGNORECASE),
    re.compile(r"\bmodel:\s*loading\b", re.IGNORECASE),
    re.compile(r"\bconnecting\s+to\b", re.IGNORECASE),
)
_SPINNER_RE = re.compile(
    r"^[·✻]\s+[A-Za-z][A-Za-z0-9''-]*(?:\s+[A-Za-z][A-Za-z0-9''-]*){0,3}(?:…|\.{3})$",
    re.UNICODE,
)
_SPECIAL_ACTIVE_RE = re.compile(r"^•\s.+\(.+•\s*esc to interrupt\)$", re.IGNORECASE)
_ACTIVE_TASK_PATTERNS = (
    re.compile(r"\b\d+\s+background terminal running\b", re.IGNORECASE),
    re.compile(r"\bbackground terminal running\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+background tasks?\b", re.IGNORECASE),
    re.compile(r"\bbackground tasks?\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+local agents?\b", re.IGNORECASE),
    re.compile(r"\blocal agents?\b", re.IGNORECASE),
    re.compile(r"esc to interrupt", re.IGNORECASE),
    re.compile(r"ctrl\+x ctrl\+k to stop agents", re.IGNORECASE),
)


def create_grid(*, session_name: str, cwd: Path) -> TmuxGrid:
    """Create a detached 2x2 session and return pane ids."""

    top_left = _run_tmux(
        [
            "new-session",
            "-d",
            "-s",
            session_name,
            "-c",
            str(cwd),
            "-P",
            "-F",
            "#{pane_id}",
        ]
    ).strip()
    top_right = _run_tmux(
        [
            "split-window",
            "-d",
            "-h",
            "-t",
            top_left,
            "-c",
            str(cwd),
            "-P",
            "-F",
            "#{pane_id}",
        ]
    ).strip()
    bottom_left = _run_tmux(
        [
            "split-window",
            "-d",
            "-v",
            "-t",
            top_left,
            "-c",
            str(cwd),
            "-P",
            "-F",
            "#{pane_id}",
        ]
    ).strip()
    bottom_right = _run_tmux(
        [
            "split-window",
            "-d",
            "-v",
            "-t",
            top_right,
            "-c",
            str(cwd),
            "-P",
            "-F",
            "#{pane_id}",
        ]
    ).strip()
    _configure_session(session_name)
    return TmuxGrid(
        session_name=session_name,
        task_panes=(top_left, top_right),
        eval_panes=(bottom_left, bottom_right),
    )


def create_single_pane_session(*, session_name: str, cwd: Path) -> TmuxPaneSession:
    """Create a detached single-pane session and return its pane id."""

    pane_id = _run_tmux(
        [
            "new-session",
            "-d",
            "-s",
            session_name,
            "-c",
            str(cwd),
            "-P",
            "-F",
            "#{pane_id}",
        ]
    ).strip()
    _configure_session(session_name)
    return TmuxPaneSession(session_name=session_name, pane_id=pane_id)


def _compute_grid(n: int) -> tuple[int, int]:
    """Return (rows, cols) for the squarest landscape grid holding *n* panes."""
    if n <= 1:
        return 1, 1
    k = math.ceil(math.log2(n))
    cols = 1 << math.ceil(k / 2)
    rows = math.ceil(n / cols)
    return rows, cols


def _split_equal(pane_id: str, count: int, *, horizontal: bool, cwd: Path) -> list[str]:
    """Split *pane_id* into *count* equal parts using exact percentages."""
    if count <= 1:
        return [pane_id]
    flag = "-h" if horizontal else "-v"
    panes: list[str] = []
    remainder = pane_id
    for i in range(1, count):
        pct = round((count - i) / (count - i + 1) * 100)
        new_pane = _run_tmux(
            [
                "split-window",
                flag,
                "-d",
                "-t",
                remainder,
                "-l",
                f"{pct}%",
                "-c",
                str(cwd),
                "-P",
                "-F",
                "#{pane_id}",
            ]
        ).strip()
        panes.append(remainder)
        remainder = new_pane
    panes.append(remainder)
    return panes


def create_pane_pool_session(
    *,
    session_name: str,
    cwd: Path,
    pane_count: int,
) -> TmuxPanePoolSession:
    """Create one detached tmux window with ``pane_count`` fungible panes."""

    _TMUX_MAX_PANES_PER_WINDOW = 50

    if pane_count <= 0:
        raise ValueError("pane_count must be positive")
    if pane_count > _TMUX_MAX_PANES_PER_WINDOW:
        raise ValueError(
            f"pane_count={pane_count} exceeds tmux hard limit of "
            f"{_TMUX_MAX_PANES_PER_WINDOW} panes per window"
        )

    rows, cols = _compute_grid(pane_count)
    # Each pane needs ~2 rows (1 content + 1 border).
    session_height = pane_count * 2 + 1

    first_pane = _run_tmux(
        [
            "new-session",
            "-d",
            "-s",
            session_name,
            "-x",
            "200",
            "-y",
            str(session_height),
            "-c",
            str(cwd),
            "-P",
            "-F",
            "#{pane_id}",
        ]
    ).strip()

    # Split into rows, then split each row into columns.
    row_panes = _split_equal(first_pane, rows, horizontal=False, cwd=cwd)
    pane_ids: list[str] = []
    remaining = pane_count
    for row_pane in row_panes:
        n_cols = min(cols, remaining)
        col_panes = _split_equal(row_pane, n_cols, horizontal=True, cwd=cwd)
        pane_ids.extend(col_panes)
        remaining -= n_cols

    _configure_session(session_name)
    return TmuxPanePoolSession(session_name=session_name, pane_ids=tuple(pane_ids))


def launch_in_pane(
    *,
    pane_id: str,
    cwd: Path,
    env: dict[str, str],
    argv: list[str],
    pane_title: str | None = None,
    banner_lines: tuple[str, ...] = (),
) -> None:
    """Launch one interactive Codex command in a pane by replacing the pane process."""

    if pane_title:
        set_pane_phase_label(pane_id, pane_title)
        set_pane_title(pane_id, pane_title)
    env_prefix = _isolated_env_prefix(env)
    command = " ".join(shlex.quote(part) for part in argv)
    banner_prefix = _banner_prefix(banner_lines)
    script = f"cd {shlex.quote(str(cwd))} && {banner_prefix}{env_prefix} {command}".strip()
    shell_command = _shell_wrapper(script)
    _run_tmux(["respawn-pane", "-k", "-t", pane_id, "-c", str(cwd), shell_command])


def pane_dead(pane_id: str) -> bool:
    """Return whether tmux considers the pane dead."""

    try:
        result = _run_tmux(["display-message", "-p", "-t", pane_id, "#{pane_dead}"]).strip()
    except subprocess.CalledProcessError:
        return True
    if result == "":
        return True
    return result == "1"


def capture_pane(pane_id: str) -> str:
    """Capture visible pane contents for debugging."""

    return _run_tmux(["capture-pane", "-p", "-t", pane_id])


def capture_pane_tail(pane_id: str, *, tail_lines: int = 80) -> str:
    """Capture the last ``tail_lines`` of one pane for readiness checks."""

    return _run_tmux(["capture-pane", "-p", "-t", pane_id, "-S", f"-{tail_lines}"])


def pane_looks_ready(captured: str) -> bool:
    """Return whether a Claude/Codex prompt is visible in the captured tail."""

    content = captured.rstrip()
    if not content:
        return False
    lines = _normalize_pane_lines(content)
    if not lines:
        return False
    if any(pattern.search(line) for pattern in _BOOTSTRAP_PATTERNS for line in lines):
        return False
    if _PROMPT_RE.search(lines[-1]):
        return True
    return any(re.match(r"^\s*[›❯]\s*", line, re.UNICODE) for line in lines)


def pane_has_active_task(captured: str) -> bool:
    """Return whether the pane tail still shows task activity."""

    tail = [line.strip() for line in _normalize_pane_lines(captured)][-40:]
    if any(_SPECIAL_ACTIVE_RE.match(line) for line in tail):
        return True
    if any(_SPINNER_RE.match(line) for line in tail):
        return True
    return any(pattern.search(line) for pattern in _ACTIVE_TASK_PATTERNS for line in tail)


def _wait_for_pane_idle(
    pane_id: str,
    *,
    timeout_seconds: float,
    poll_interval: float = 0.5,
    idle_confirmation_seconds: float = 3.0,
) -> bool:
    """Wait until a pane shows a stable ready prompt with no active task markers."""

    deadline = time.monotonic() + timeout_seconds
    first_idle_at: float | None = None
    while time.monotonic() < deadline:
        captured = capture_pane_tail(pane_id, tail_lines=80)
        idle = pane_looks_ready(captured) and not pane_has_active_task(captured)
        if idle:
            if first_idle_at is None:
                first_idle_at = time.monotonic()
            if time.monotonic() - first_idle_at >= idle_confirmation_seconds:
                return True
        else:
            first_idle_at = None
        if pane_dead(pane_id):
            return False
        time.sleep(poll_interval)
    return False


def pane_current_command(pane_id: str) -> str:
    """Return the current foreground command name for a pane."""

    return _run_tmux(["display-message", "-p", "-t", pane_id, "#{pane_current_command}"]).strip()


def set_pane_title(pane_id: str, title: str) -> None:
    """Set one pane title so tmux border labels reflect the active phase."""

    _run_tmux(["select-pane", "-t", pane_id, "-T", title], check=False)


def set_pane_phase_label(pane_id: str, label: str) -> None:
    """Persist a stable tmux-owned phase label that apps cannot overwrite."""

    _run_tmux(["set-option", "-p", "-t", pane_id, "@phase_label", label], check=False)


def rename_window(session_name: str, name: str) -> None:
    """Rename the visible window for clearer phase/generation navigation."""

    _run_tmux(["rename-window", "-t", session_name, name], check=False)


def show_banner_in_pane(
    *,
    pane_id: str,
    cwd: Path,
    title: str,
    lines: tuple[str, ...] = (),
) -> None:
    """Reset a pane to an idle shell after showing a phase banner."""

    set_pane_phase_label(pane_id, title)
    set_pane_title(pane_id, title)
    shell_path = os.environ.get("SHELL", "/bin/zsh")
    script = (
        f"cd {shlex.quote(str(cwd))} && "
        f"{_banner_prefix((title, *lines))}"
        f"exec {shlex.quote(shell_path)} -i"
    )
    _run_tmux(
        ["respawn-pane", "-k", "-t", pane_id, "-c", str(cwd), _shell_wrapper(script)],
        check=False,
    )


def kill_pane(pane_id: str) -> None:
    """Best-effort pane cleanup that preserves the pane slot for reuse."""

    shell_path = os.environ.get("SHELL", "/bin/zsh")
    idle_command = _shell_wrapper(f"exec {shlex.quote(shell_path)} -i")
    _run_tmux(["respawn-pane", "-k", "-t", pane_id, idle_command], check=False)


def kill_session(session_name: str) -> None:
    """Best-effort session cleanup."""

    _run_tmux(["kill-session", "-t", session_name], check=False)
    _run_tmux(["set-option", "-g", "mouse", "on"], check=False)


def open_iterm2_window_for_session(session_name: str) -> bool:
    """Open a dedicated iTerm2 window and attach it to the run session when possible."""

    if platform.system() != "Darwin":
        return False
    if shutil.which("osascript") is None:
        return False
    script = "\n".join(
        [
            'tell application id "com.googlecode.iterm2"',
            "  activate",
            "  set newWindow to (create window with default profile)",
            "  tell current session of newWindow",
            f'    write text "tmux attach -t {session_name}"',
            "  end tell",
            "end tell",
        ]
    )
    completed = subprocess.run(
        ["osascript"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def wait_for_pane_reset(
    pane_id: str,
    *,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
) -> None:
    """Wait until a pane is no longer running the interactive Codex TUI."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            current_command = pane_current_command(pane_id)
        except subprocess.CalledProcessError:
            return
        if current_command == "":
            return
        if current_command and current_command != "codex":
            return
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for pane {pane_id} to reset to an idle shell.")


def wait_for_panes_reset(
    pane_ids: tuple[str, ...],
    *,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
) -> None:
    """Wait until all listed panes have reset back to an idle shell."""

    for pane_id in pane_ids:
        wait_for_pane_reset(
            pane_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )


def _run_tmux(args: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["tmux", *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _normalize_pane_lines(captured: str) -> list[str]:
    return [line.replace("\r", "").rstrip() for line in str(captured).split("\n") if line.strip()]


def _banner_prefix(lines: tuple[str, ...]) -> str:
    clear_command = "command -v clear >/dev/null 2>&1 && clear || printf '\\033[2J\\033[H'; "
    if not lines:
        return clear_command
    banner = (
        "========================================",
        *lines,
        "========================================",
        "",
    )
    quoted_lines = " ".join(shlex.quote(line) for line in banner)
    return clear_command + f"printf '%s\\n' {quoted_lines}; "


def _isolated_env_prefix(env: dict[str, str]) -> str:
    retained_keys = (
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "USER",
        "LOGNAME",
        "COLORTERM",
    )
    retained = {key: value for key in retained_keys if (value := os.environ.get(key))}
    term_value = retained.get("TERM")
    if not term_value or term_value == "dumb":
        retained["TERM"] = "screen-256color"
    retained.update(env)
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(retained.items()))
    return f"env -i {exports}".strip()


def _shell_wrapper(script: str) -> str:
    return f"/bin/sh -lc {shlex.quote(script)}"


def _configure_session(session_name: str) -> None:
    _run_tmux(["set-option", "-t", session_name, "pane-border-status", "top"], check=False)
    _run_tmux(
        [
            "set-option",
            "-t",
            session_name,
            "pane-border-format",
            "#{?@phase_label,#{@phase_label},#{pane_title}}",
        ],
        check=False,
    )
