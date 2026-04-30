"""Storage-side artifact, manifest, and event models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, cast, runtime_checkable
from uuid import uuid4

from pydantic import Field, model_validator

from scaling_evolve.core.common import DomainModel, JSONValue, NodeId, utc_now


class ArtifactKind(StrEnum):
    """Artifact categories produced by the framework."""

    SOURCE = "source"
    SUMMARY = "summary"
    METRICS = "metrics"
    TRANSCRIPT = "transcript"
    SESSION_ARCHIVE_JSONL = "session.archive.jsonl"
    CHANGED_FILES_MANIFEST_JSON = "changed_files.manifest.json"
    TRANSCRIPT_DIGEST_JSON = "transcript.digest.json"
    SNAPSHOT = "snapshot"
    CONFIG_JSON = "config.json"
    INSTRUCTION_JSON = "instruction.json"
    INSTRUCTION_TEXT = "instruction.text"
    MUTATION_RESULT_JSON = "mutation.result.json"
    PROMPT_REQUEST_JSON = "prompt.request.json"
    MODEL_RESPONSE_RAW_JSON = "model.response.raw.json"
    MODEL_RESPONSE_PARSED_JSON = "model.response.parsed.json"
    CANDIDATE_SOURCE_PY = "candidate.source.py"
    EVALUATION_SUMMARY_JSON = "evaluation.summary.json"
    EVALUATION_TRACEBACK_TXT = "evaluation.traceback.txt"
    LINEAGE_SUMMARY_JSON = "lineage.summary.json"
    INHERITED_CONTEXT_JSON = "inherited_context.json"
    PROJECTED_STATE_JSON = "projected_state.json"
    PORTABLE_STATE_JSON = "portable_state.json"
    SCORE_JSON = "score.json"
    FAILURE_SUMMARY_TXT = "failure.summary.txt"
    APPROACH_CARD_YAML = "approach_card.yaml"
    MUTATION_NOTE_YAML = "mutation_note.yaml"


class MaterializationKind(StrEnum):
    """Kinds of materialized candidate state."""

    SINGLE_FILE = "single_file"
    WORKSPACE_SNAPSHOT = "workspace_snapshot"
    WORKSPACE = "workspace"
    SNAPSHOT = "snapshot"
    ARTIFACT = "artifact"


def _empty_changed_artifacts() -> list[ArtifactRef]:
    return []


class ArtifactRef(DomainModel):
    """Reference to a persisted artifact or projected candidate artifact."""

    artifact_id: str | None = None
    kind: ArtifactKind | str | None = None
    location: str | None = None
    relpath: str = ""
    uri: str | None = None
    sha256: str = ""
    size_bytes: int | None = None
    primary_artifact: Any | None = None
    changed_artifacts: list[Any] = Field(default_factory=list)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_aliases(self) -> ArtifactRef:
        if not self.relpath and isinstance(self.location, str):
            self.relpath = self.location
        if not self.relpath and self.uri is not None:
            self.relpath = self.uri
        if self.uri is None and self.relpath:
            self.uri = self.relpath
        if self.location is None and self.relpath:
            self.location = self.relpath
        if self.location is not None:
            self.metadata.setdefault("location", self.location)
        return self


class MaterializationRef(DomainModel):
    """Reference to a candidate's materialized state."""

    materialization_id: str = Field(default_factory=lambda: f"mat-{uuid4().hex}")
    kind: MaterializationKind = MaterializationKind.SINGLE_FILE
    snapshot_artifact: ArtifactRef | None = None
    primary_artifact: ArtifactRef | None = None
    manifest_artifact: ArtifactRef | None = None
    changed_artifacts: list[ArtifactRef] = Field(default_factory=_empty_changed_artifacts)
    node_id: NodeId | None = None
    location: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_payload(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        typed_value = cast(Mapping[str, JSONValue], value)
        payload: dict[str, JSONValue] = {
            str(raw_key): raw_item for raw_key, raw_item in typed_value.items()
        }
        location = payload.pop("location", None)
        if location is None:
            if "snapshot_artifact" not in payload and "primary_artifact" in payload:
                payload["snapshot_artifact"] = payload["primary_artifact"]
            return payload
        location_str = str(location)
        materialization_id = str(payload.get("materialization_id", f"legacy-{uuid4().hex}"))
        placeholder = {
            "artifact_id": f"{materialization_id}:primary",
            "kind": ArtifactKind.SOURCE,
            "location": location_str,
            "relpath": location_str,
            "uri": location_str,
            "metadata": {"legacy_location": True},
        }
        payload.setdefault("primary_artifact", placeholder)
        payload.setdefault("snapshot_artifact", placeholder)
        payload["location"] = location_str
        metadata_value = payload.get("metadata")
        metadata: dict[str, JSONValue] = {}
        if isinstance(metadata_value, dict):
            typed_metadata = cast(dict[str, JSONValue], metadata_value)
            for raw_key, raw_item in typed_metadata.items():
                metadata[str(raw_key)] = raw_item
        payload["metadata"] = metadata
        metadata.setdefault("location", location_str)
        return payload

    @model_validator(mode="after")
    def _sync_location_alias(self) -> MaterializationRef:
        if self.snapshot_artifact is None and self.primary_artifact is not None:
            self.snapshot_artifact = self.primary_artifact
        if self.location is None and self.primary_artifact is not None:
            self.location = self.primary_artifact.location or self.primary_artifact.relpath
        if self.location is None and self.changed_artifacts:
            self.location = self.changed_artifacts[0].location or self.changed_artifacts[0].relpath
        if self.location is None:
            metadata_location = self.metadata.get("location")
            if isinstance(metadata_location, str):
                self.location = metadata_location
        if self.location is not None:
            self.metadata.setdefault("location", self.location)
        return self


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


class SessionInstanceRecord(DomainModel):
    """Persisted session-instance metadata for a mutation execution."""

    session_instance_id: str
    run_id: str
    individual_id: str
    base_checkpoint_id: str | None = None
    driver_name: str | None = None
    provider_session_id: str | None = None
    workspace_id: str | None = None
    status: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class ExecutionSegmentRecord(DomainModel):
    """Persisted execution-segment metadata for a session instance."""

    segment_id: str
    run_id: str
    session_instance_id: str
    reason: str | None = None
    native_ref: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    transcript_ref: ArtifactRef | None = None
    cost: dict[str, JSONValue] = Field(default_factory=dict)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


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
