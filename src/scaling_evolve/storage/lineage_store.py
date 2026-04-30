"""Lineage storage protocol surface."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import Field

from scaling_evolve.core.common import DomainModel, JSONValue, RunId, utc_now
from scaling_evolve.core.node import (
    EdgeRecordLike,
    ExecutionSegmentRecord,
    NodeRecordLike,
    SessionInstanceRecord,
)
from scaling_evolve.core.storage.models import ArtifactRef


class RunRecord(DomainModel):
    """Minimal persisted run record."""

    run_id: RunId
    run_name: str | None = None
    config_ref: ArtifactRef | None = None
    status: str = "pending"
    app_kind: str = "unknown"
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    seed: int | None = None
    notes: dict[str, JSONValue] = Field(default_factory=dict)


@runtime_checkable
class LineageStore(Protocol):
    """Protocol for storing run, node, and edge lineage."""

    def create_run(self, run: RunRecord) -> None:
        """Persist a new run record."""

        ...

    def save_node(self, node: NodeRecordLike) -> None:
        """Persist a node record."""

        ...

    def save_edge(self, edge: EdgeRecordLike) -> None:
        """Persist an edge record."""

        ...

    def list_nodes(self, run_id: RunId) -> Sequence[NodeRecordLike]:
        """List node records for a run."""

        ...

    def save_edge_execution(self, **kwargs: object) -> None:
        """Persist canonical per-edge execution facts."""

        ...

    def list_edge_executions(self, run_id: RunId) -> Sequence[dict[str, JSONValue]]:
        """List canonical per-edge execution facts for a run."""

        ...

    def save_session_instance(self, record: SessionInstanceRecord) -> None:
        """Persist a session-instance record."""

        ...

    def list_session_instances(self, run_id: RunId) -> Sequence[SessionInstanceRecord]:
        """List session-instance records for a run."""

        ...

    def save_execution_segment(self, record: ExecutionSegmentRecord) -> None:
        """Persist an execution-segment record."""

        ...

    def list_execution_segments(self, run_id: RunId) -> Sequence[ExecutionSegmentRecord]:
        """List execution-segment records for a run."""

        ...
