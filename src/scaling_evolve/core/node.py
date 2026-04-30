"""Node models and node-adjacent compatibility types."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Literal, cast

from pydantic import Field, model_validator

from scaling_evolve.core.common import (
    DomainModel,
    EdgeId,
    JSONValue,
    NodeId,
    RunId,
    normalize_provider_kind,
    utc_now,
    warn_legacy_once,
)
from scaling_evolve.core.engine import PortableState, RuntimeState
from scaling_evolve.core.evaluation import ScoreCard
from scaling_evolve.core.storage.models import (
    ArtifactRef,
    MaterializationRef,
)
from scaling_evolve.core.storage.models import (
    ExecutionSegmentRecord as _ExecutionSegmentRecord,
)
from scaling_evolve.core.storage.models import (
    SessionInstanceRecord as _SessionInstanceRecord,
)

NodeLifecycleStatus = Literal["seed", "evaluated", "invalid", "failed"]
EdgeLifecycleStatus = Literal["started", "mutated", "evaluated", "failed"]
InheritanceModeValue = Literal["summary_only", "native"]
ProviderKindValue = str
EdgeKindValue = Literal["fork", "continuation", "inspiration"]
_LOGGER = logging.getLogger(__name__)


def _normalize_inheritance_mode(value: object) -> object:
    if value == "rehydrate":
        warn_legacy_once(
            "Legacy inheritance mode 'rehydrate' mapped to 'summary_only'.",
            logger=_LOGGER,
        )
        return "summary_only"
    if value == "fresh":
        return "summary_only"
    return value


class NodeStatus(StrEnum):
    """Lifecycle states for candidate nodes."""

    PENDING = "pending"
    SEED = "seed"
    MATERIALIZED = "materialized"
    EVALUATED = "evaluated"
    INVALID = "invalid"
    FAILED = "failed"


class EdgeStatus(StrEnum):
    """Lifecycle states for lineage edges."""

    PENDING = "pending"
    STARTED = "started"
    APPLIED = "applied"
    MUTATED = "mutated"
    EVALUATED = "evaluated"
    FAILED = "failed"


class Node(DomainModel):
    """Minimal persisted version record."""

    id: str
    artifact: ArtifactRef
    score: ScoreCard | None = None
    parent_id: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return self.id


class NodeRecordLike(DomainModel):
    """Minimal node record used across protocol contracts."""

    node_id: NodeId
    run_id: RunId
    status: NodeStatus = NodeStatus.PENDING
    score: float | None = None
    parent_id: NodeId | None = None


class EdgeRecordLike(DomainModel):
    """Minimal edge record used across protocol contracts."""

    edge_id: EdgeId
    run_id: RunId
    parent_id: NodeId
    child_id: NodeId | None = None
    status: EdgeStatus = EdgeStatus.PENDING


class NodeRecord(Node):
    """Compatibility node wrapper that still accepts legacy payloads."""

    score: ScoreCard | None = None
    approach_summary: str | None = None
    approach_card_ref: ArtifactRef | None = None

    @property
    def primary_score(self) -> float:
        if self.score is None:
            return float("-inf")
        return self.score.primary_score

    @property
    def run_id(self) -> RunId:
        value = self.metadata.get("run_id")
        return str(value) if value is not None else ""

    @property
    def individual_id(self) -> str | None:
        value = self.metadata.get("individual_id")
        if isinstance(value, str):
            return value
        return self.id

    @individual_id.setter
    def individual_id(self, value: str | None) -> None:
        self.metadata["individual_id"] = self.id if value is None else value

    @property
    def parent_node_id(self) -> NodeId | None:
        return self.parent_id

    @property
    def generation(self) -> int:
        value = self.metadata.get("generation")
        return int(value) if isinstance(value, int | float) else 0

    @property
    def workspace_path(self) -> str | None:
        value = self.metadata.get("workspace_path")
        if isinstance(value, str) and value:
            return value
        materialization = self.metadata.get("materialization")
        if isinstance(materialization, MaterializationRef):
            workspace_root = materialization.metadata.get("workspace_root")
            if isinstance(workspace_root, str) and workspace_root:
                return workspace_root
        if isinstance(materialization, Mapping):
            materialization_root = cast(Mapping[object, object], materialization).get("metadata")
            if isinstance(materialization_root, Mapping):
                workspace_root = cast(Mapping[object, object], materialization_root).get(
                    "workspace_root"
                )
                if isinstance(workspace_root, str) and workspace_root:
                    return workspace_root
        return None

    @workspace_path.setter
    def workspace_path(self, value: str | None) -> None:
        self.metadata["workspace_path"] = value

    @property
    def materialization(self) -> MaterializationRef:
        value = self.metadata.get("materialization")
        if isinstance(value, MaterializationRef):
            return value
        if isinstance(value, Mapping):
            return MaterializationRef.model_validate(value)
        return MaterializationRef(
            materialization_id=self.artifact.artifact_id or f"mat:{self.id}",
            primary_artifact=self.artifact,
            snapshot_artifact=self.artifact,
            changed_artifacts=[],
            location=self.artifact.location,
            metadata=dict(self.artifact.metadata),
        )

    @property
    def portable_state(self) -> PortableState:
        value = self.metadata.get("portable_state")
        if isinstance(value, PortableState):
            return value
        if isinstance(value, Mapping):
            return PortableState.model_validate(value)
        return PortableState()

    @property
    def runtime_state(self) -> RuntimeState | None:
        value = self.metadata.get("runtime_state")
        if isinstance(value, RuntimeState):
            return value
        if isinstance(value, Mapping):
            return RuntimeState.model_validate(value)
        return None

    @property
    def budget(self):
        from scaling_evolve.core.engine import BudgetLedger

        value = self.metadata.get("budget")
        if isinstance(value, BudgetLedger):
            return value
        if isinstance(value, Mapping):
            return BudgetLedger.model_validate(value)
        return BudgetLedger()

    @property
    def status(self) -> NodeLifecycleStatus:
        value = self.metadata.get("status")
        if isinstance(value, str) and value in {"seed", "evaluated", "invalid", "failed"}:
            return cast(NodeLifecycleStatus, value)
        return "evaluated"

    @property
    def created_at(self) -> datetime:
        value = self.metadata.get("created_at")
        if isinstance(value, datetime):
            return value
        return utc_now()

    @property
    def tags(self) -> dict[str, str]:
        value = self.metadata.get("tags")
        if not isinstance(value, dict):
            return {}
        return {
            str(key): str(tag_value)
            for key, tag_value in cast(dict[object, object], value).items()
            if isinstance(tag_value, str | int | float | bool)
        }

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_payload(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        typed_value = cast(Mapping[object, object], value)
        payload = {str(key): item for key, item in typed_value.items()}
        metadata_value = payload.pop("metadata", {})
        metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
        materialization = payload.pop("materialization", metadata.get("materialization"))
        if materialization is not None and not isinstance(materialization, MaterializationRef):
            materialization = MaterializationRef.model_validate(materialization)
        workspace_path = payload.pop("workspace_path", metadata.get("workspace_path"))
        if "node_id" in payload and "id" not in payload:
            payload["id"] = payload.pop("node_id")
        if "parent_node_id" in payload and "parent_id" not in payload:
            payload["parent_id"] = payload.pop("parent_node_id")
        metadata.setdefault("run_id", payload.pop("run_id", metadata.get("run_id")))
        metadata.setdefault(
            "individual_id",
            payload.pop("individual_id", metadata.get("individual_id")),
        )
        metadata["generation"] = payload.pop("generation", metadata.get("generation", 0))
        metadata["materialization"] = materialization
        metadata["portable_state"] = payload.pop(
            "portable_state",
            metadata.get("portable_state", PortableState()),
        )
        metadata["runtime_state"] = payload.pop("runtime_state", metadata.get("runtime_state"))
        metadata["budget"] = payload.pop("budget", metadata.get("budget"))
        metadata["status"] = payload.pop("status", metadata.get("status", "evaluated"))
        metadata["created_at"] = payload.pop("created_at", metadata.get("created_at", utc_now()))
        metadata["tags"] = payload.pop("tags", metadata.get("tags", {}))
        metadata["workspace_path"] = workspace_path
        payload["approach_summary"] = payload.pop(
            "approach_summary",
            metadata.get("approach_summary"),
        )
        payload["approach_card_ref"] = payload.pop(
            "approach_card_ref",
            metadata.get("approach_card_ref"),
        )
        artifact = payload.pop("artifact", None)
        if artifact is None:
            artifact = ArtifactRef(
                artifact_id=materialization.materialization_id if materialization else None,
                kind=(str(materialization.kind) if materialization is not None else "source"),
                location=(materialization.location if materialization is not None else None),
                metadata=(dict(materialization.metadata) if materialization is not None else {}),
            )
        payload["artifact"] = artifact
        payload["metadata"] = metadata
        return payload

    @model_validator(mode="after")
    def _default_individual_id(self) -> NodeRecord:
        if self.individual_id is None:
            self.individual_id = self.node_id
        return self

    def model_copy(
        self,
        *,
        update: dict[str, object] | None = None,
        deep: bool = False,
    ) -> NodeRecord:
        if not update:
            return cast(NodeRecord, super().model_copy(update=update, deep=deep))
        payload = self.model_dump(mode="python")
        payload.update(update)
        return type(self).model_validate(payload)


class EdgeRecord(DomainModel):
    """Persisted mutation edge record."""

    edge_id: EdgeId
    run_id: RunId
    iteration: int
    parent_node_id: NodeId
    child_node_id: NodeId | None = None
    edge_kind: EdgeKindValue = "fork"
    provider_kind: ProviderKindValue
    inheritance_mode: InheritanceModeValue
    instruction_ref: ArtifactRef | None = None
    projected_state_ref: ArtifactRef | None = None
    result_ref: ArtifactRef | None = None
    delta_summary: str | None = None
    mutation_note_ref: ArtifactRef | None = None
    status: EdgeLifecycleStatus = "started"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = {str(key): item for key, item in cast(Mapping[object, object], value).items()}
        if "backend_kind" in payload and "provider_kind" not in payload:
            payload["provider_kind"] = payload.pop("backend_kind")
        if "provider_kind" in payload:
            payload["provider_kind"] = normalize_provider_kind(
                payload["provider_kind"],
                logger=_LOGGER,
            )
        if "inheritance_mode" in payload:
            payload["inheritance_mode"] = _normalize_inheritance_mode(payload["inheritance_mode"])
        return payload

    @property
    def parent_id(self) -> NodeId:
        return self.parent_node_id

    @property
    def child_id(self) -> NodeId | None:
        return self.child_node_id


def fingerprint_materialization(materialization: MaterializationRef) -> str | None:
    """Build a stable fingerprint from persisted artifact hashes."""

    for artifact in (
        materialization.primary_artifact,
        materialization.snapshot_artifact,
        *materialization.changed_artifacts,
    ):
        if artifact is not None and artifact.sha256:
            return artifact.sha256
    return None


def fingerprint_node(node: NodeRecord) -> str | None:
    """Read or derive a stable fingerprint for a persisted node."""

    candidate_fingerprint = node.tags.get("candidate_fingerprint")
    if candidate_fingerprint:
        return candidate_fingerprint
    return fingerprint_materialization(node.materialization)


ExecutionSegmentRecord = _ExecutionSegmentRecord
SessionInstanceRecord = _SessionInstanceRecord
