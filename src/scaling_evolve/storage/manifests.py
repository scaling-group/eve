"""Storage-side manifest models used by concrete persistence implementations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from scaling_evolve.core.common import DomainModel, JSONValue, utc_now
from scaling_evolve.core.storage.models import ArtifactRef


def _empty_manifest_files() -> list[ManifestFile]:
    return []


def _empty_delta_entries() -> list[PortableDeltaEntry]:
    return []


def _empty_paths() -> list[str]:
    return []


class ManifestFile(DomainModel):
    """Single file entry captured in a portable or workspace manifest."""

    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    artifact_ref: ArtifactRef | None = None


class PortableDeltaEntry(DomainModel):
    """Changed file metadata carried across portable snapshots."""

    path: str
    change_type: str
    artifact_ref: ArtifactRef | None = None


class PortableManifest(DomainModel):
    """Portable snapshot manifest that can be replayed later."""

    files: list[ManifestFile] = Field(default_factory=_empty_manifest_files)
    deltas: list[PortableDeltaEntry] = Field(default_factory=_empty_delta_entries)


class ChangedFilesManifest(DomainModel):
    """Changed files produced by a session lane rollout."""

    files: list[str] = Field(default_factory=_empty_paths)
    primary_path: str | None = None
    changed_paths: list[str] = Field(default_factory=_empty_paths)
    workspace_strategy: str | None = None
    manifest_source: Literal["driver", "projection_fallback"] = "driver"

    @model_validator(mode="after")
    def _sync_changed_path_aliases(self) -> ChangedFilesManifest:
        if not self.changed_paths:
            self.changed_paths = list(self.files)
        if not self.files:
            self.files = list(self.changed_paths)
        if self.primary_path is None and self.changed_paths:
            self.primary_path = self.changed_paths[0]
        return self


class WorkspaceSnapshotManifest(DomainModel):
    """Workspace materialization metadata captured for snapshot round-trips."""

    strategy: str
    target_repo_root: str | None = None
    workspace_root: str | None = None
    support_roots: list[str] = Field(default_factory=_empty_paths)
    write_roots: list[str] = Field(default_factory=_empty_paths)
    files: list[ManifestFile] = Field(default_factory=_empty_manifest_files)


class ArtifactRecord(DomainModel):
    """Indexed artifact metadata stored in SQLite."""

    ref: ArtifactRef
    run_id: str
    node_id: str | None = None
    edge_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EventRecord(DomainModel):
    """Persisted event for retries, failures, and stop conditions."""

    event_id: str
    run_id: str
    node_id: str | None = None
    edge_id: str | None = None
    event_type: str
    payload: Mapping[str, JSONValue]
    created_at: datetime = Field(default_factory=utc_now)


@runtime_checkable
class ArtifactIndexer(Protocol):
    """Protocol implemented by lineage stores that index artifact metadata."""

    def register_artifact(
        self,
        ref: ArtifactRef,
        *,
        run_id: str,
        node_id: str | None = None,
        edge_id: str | None = None,
    ) -> None:
        """Persist artifact metadata for later lookup."""

        ...
