"""Session driver protocol surface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from scaling_evolve.core.common import DomainModel
from scaling_evolve.core.engine import ArtifactRef, PortableStateRef, RuntimeStateRef
from scaling_evolve.core.mutation import ProviderUsage


def _empty_attachments() -> list[ArtifactRef]:
    return []


def _empty_changed_paths() -> list[str]:
    return []


def _empty_display_context() -> dict[str, str | int]:
    return {}


class SessionDriverCapabilities(DomainModel):
    """Capability surface exposed by a session driver."""

    supports_native_fork: bool = False
    supports_cross_workspace_fork: bool = False


class SessionWorkspaceLease(DomainModel):
    """Resolved workspace lease passed to a session driver."""

    workspace_id: str
    target_repo_root: str
    workspace_root: str
    session_cwd: str
    strategy: str | None = None


class SessionSeed(DomainModel):
    """Input required to spawn a new session."""

    instruction: str
    workspace: SessionWorkspaceLease | None = None
    working_directory: str | None = None
    attachments: list[ArtifactRef] = Field(default_factory=_empty_attachments)
    display_context: dict[str, str | int] = Field(default_factory=_empty_display_context)
    prompt_file: str | None = None
    write_prompt_file: bool = True

    @model_validator(mode="after")
    def _sync_workspace_alias(self) -> SessionSeed:
        if self.workspace is None and self.working_directory is not None:
            self.workspace = SessionWorkspaceLease(
                workspace_id="workspace:legacy",
                target_repo_root=self.working_directory,
                workspace_root=self.working_directory,
                session_cwd=self.working_directory,
            )
        if self.working_directory is None and self.workspace is not None:
            self.working_directory = self.workspace.session_cwd
        return self


class SessionRollout(DomainModel):
    """Result of a single session mutation or resume step."""

    state: RuntimeStateRef
    transcript: ArtifactRef | None = None
    changed_files_manifest: ArtifactRef | None = None
    primary_path: str | None = None
    changed_paths: list[str] = Field(default_factory=_empty_changed_paths)
    summary: str | None = None
    fallback_reason: str | None = None
    usage: ProviderUsage | None = None

    @model_validator(mode="after")
    def _sync_changed_path_defaults(self) -> SessionRollout:
        if self.primary_path is None and self.changed_paths:
            self.primary_path = self.changed_paths[0]
        return self


class SessionSnapshot(DomainModel):
    """Portable snapshot captured from a session."""

    state: RuntimeStateRef
    portable_state: PortableStateRef | None = None
    transcript_digest: ArtifactRef | None = None
    summary: str | None = None


@runtime_checkable
class SessionDriver(Protocol):
    """Protocol for persistent session management."""

    def capabilities(self) -> SessionDriverCapabilities:
        """Return the driver's feature capabilities."""

        ...

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        """Create a new session from a seed and return the completed rollout."""

        ...

    def fork_session(self, parent: RuntimeStateRef) -> str:
        """Create a child session id from an existing runtime state."""

        ...

    def migrate_session(
        self,
        *,
        parent_cwd: str,
        child_cwd: str,
        session_id: str,
    ) -> str:
        """Move a forked provider session into the child cwd bucket."""

        ...

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        """Fork a session from an existing runtime state and return the completed rollout."""

        ...

    def resume(
        self,
        state: RuntimeStateRef,
        instruction: str | None = None,
    ) -> SessionRollout:
        """Resume a session and return the rollout summary."""

        ...

    def snapshot(self, state: RuntimeStateRef) -> SessionSnapshot:
        """Capture a portable snapshot from a session."""

        ...
