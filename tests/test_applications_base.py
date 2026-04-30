"""Tests for shared application-layer helpers."""

from __future__ import annotations

from scaling_evolve.applications.base import terminate_process_with_fallback


class _FakeProcess:
    def __init__(self, *, alive_after_terminate: bool) -> None:
        self.alive_after_terminate = alive_after_terminate
        self.calls: list[object] = []
        self.killed = False

    def terminate(self) -> None:
        self.calls.append("terminate")

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))

    def kill(self) -> None:
        self.calls.append("kill")
        self.killed = True

    def is_alive(self) -> bool:
        self.calls.append("is_alive")
        if self.killed:
            return False
        return self.alive_after_terminate


def test_terminate_process_with_fallback_kills_when_terminate_is_not_enough() -> None:
    process = _FakeProcess(alive_after_terminate=True)

    terminate_process_with_fallback(process, timeout_seconds=0.25)

    assert process.calls == [
        "terminate",
        ("join", 0.25),
        "is_alive",
        "kill",
        ("join", 0.25),
    ]


def test_terminate_process_with_fallback_skips_kill_when_process_exits() -> None:
    process = _FakeProcess(alive_after_terminate=False)

    terminate_process_with_fallback(process, timeout_seconds=0.25)

    assert process.calls == [
        "terminate",
        ("join", 0.25),
        "is_alive",
    ]
