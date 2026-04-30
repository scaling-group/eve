"""Workspace management protocols."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from scaling_evolve.core.common import DomainModel, NodeId
from scaling_evolve.core.engine import PortableStateRef, WorkspaceLease

WorkspacePurpose = Literal["mutation", "evaluation", "review", "debug"]
WorkspaceStrategy = Literal[
    "artifact_only",
    "full_workspace",
]


class WorkspacePlan(DomainModel):
    """Resolved workspace strategy for a concrete runtime purpose."""

    strategy: WorkspaceStrategy
    purpose: WorkspacePurpose
    support_roots: list[str] = Field(default_factory=list)
    write_roots: list[str] = Field(default_factory=list)
    needs_git_semantics: bool = False


class WorkspaceLeaseRequest(DomainModel):
    """Request describing a workspace lease need."""

    run_id: str
    node_id: str | None = None
    purpose: WorkspacePurpose
    owner_node_id: str | None = None
    target_repo_root: str | None = None
    base_snapshot_id: str | None = None
    plan: WorkspacePlan | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    retain_on_failure: bool = False


class WorkspacePressure(DomainModel):
    """Aggregate pressure hints for a workspace manager."""

    used_bytes: int = 0
    soft_limit_bytes: int | None = None
    hard_limit_bytes: int | None = None
    active_full_workspaces: int = 0


@runtime_checkable
class WorkspaceManager(Protocol):
    """Protocol for leasing and releasing workspaces."""

    def acquire(self, owner_node_id: NodeId | None = None) -> WorkspaceLease:
        """Acquire a workspace lease."""

        ...

    def release(self, lease: WorkspaceLease) -> None:
        """Release a workspace lease."""

        ...


@runtime_checkable
class WorkspaceMaterializer(Protocol):
    """Protocol for capturing portable state from a workspace."""

    def snapshot(self, lease: WorkspaceLease) -> PortableStateRef:
        """Create a portable snapshot from a workspace lease."""

        ...


@runtime_checkable
class WorkspaceLeaseManager(Protocol):
    """Strategy-aware workspace lease protocol."""

    def acquire(self, request: WorkspaceLeaseRequest) -> WorkspaceLease:
        """Acquire a workspace lease for a specific runtime purpose."""

        ...

    def release(self, lease: WorkspaceLease, *, snapshot: bool = False) -> None:
        """Release a workspace lease, optionally retaining snapshot metadata."""

        ...

    def pressure(self) -> WorkspacePressure:
        """Return current workspace pressure indicators."""

        ...
