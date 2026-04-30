"""Session-store helpers for stateful execution providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from scaling_evolve.core.common import DomainModel, JSONValue


class SessionInfo(DomainModel):
    """Provider-managed session metadata keyed by node id."""

    session_id: str | None = None
    workspace_id: str | None = None
    lease_id: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


@runtime_checkable
class SessionStore(Protocol):
    """Minimal session-store protocol."""

    def get(self, node_id: str) -> SessionInfo | None:
        """Return session info for `node_id`."""

        ...

    def put(self, node_id: str, info: SessionInfo) -> None:
        """Persist session info for `node_id`."""

        ...

    def has_session(self, node_id: str) -> bool:
        """Return whether `node_id` owns a tracked session."""

        ...


class InMemorySessionStore(SessionStore):
    """Small default session-store implementation used during migration."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionInfo] = {}

    def get(self, node_id: str) -> SessionInfo | None:
        return self._sessions.get(node_id)

    def put(self, node_id: str, info: SessionInfo) -> None:
        self._sessions[node_id] = info

    def has_session(self, node_id: str) -> bool:
        return node_id in self._sessions
