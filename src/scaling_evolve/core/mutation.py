"""Mutation-provider models and protocols."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import Field, field_validator, model_validator

from scaling_evolve.core.common import DomainModel, JSONValue, normalize_provider_kind
from scaling_evolve.core.engine import BudgetLedger, PortableStateRef, RuntimeStateRef
from scaling_evolve.core.storage.models import ArtifactRef, MaterializationRef

_LOGGER = logging.getLogger(__name__)


class InheritanceMode(StrEnum):
    """Supported state inheritance modes."""

    SUMMARY_ONLY = "summary_only"
    NATIVE = "native"


class CapabilitySet(DomainModel):
    """Capabilities surfaced by an execution provider."""

    supports_tools: bool = False
    supports_runtime_state: bool = False
    supports_native_fork: bool = False
    supports_summary_only: bool = True


class ProviderSpec(DomainModel):
    """Stable identity for selecting a concrete mutation provider."""

    kind: str
    provider: str | None = None
    model: str | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: object) -> object:
        return normalize_provider_kind(value, logger=_LOGGER)


class MutationSurface(DomainModel):
    """Filesystem and shell boundaries exposed to a mutation provider."""

    read_roots: list[str] = Field(default_factory=list)
    write_roots: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    shell_allowlist: list[str] = Field(default_factory=list)


class MutationInstructionLike(DomainModel):
    """Minimal mutation instruction placeholder."""

    content: str
    strategy: str | None = None
    system_message: str | None = None
    task_instruction: str | None = None
    output_format: str | None = None
    diversity_guidance: str | None = None
    search_replace_example: str | None = None
    workspace_note: str | None = None
    fork_warning: str | None = None
    context_queries: str | None = None


class MutationInstruction(MutationInstructionLike):
    """Structured mutation instruction shared across policies and providers."""

    objective: str
    constraints: list[str] = Field(default_factory=list)
    focus: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    expected_output: str
    candidate_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _empty_projected_artifacts() -> list[ProjectedArtifact]:
    return []


def _empty_projected_programs() -> list[ProjectedProgram]:
    return []


def _empty_artifact_feedback() -> list[ProjectedArtifactFeedback]:
    return []


def _empty_projected_feedback() -> list[ProjectedFeedback]:
    return []


def _empty_inspiration_summaries() -> list[str]:
    return []


def _empty_inspiration_programs() -> list[ProjectedProgram]:
    return []


class ProjectedArtifact(DomainModel):
    """Artifact included in a projected mutation context."""

    ref: ArtifactRef | None = None
    path: str | None = None
    summary: str | None = None


class ProjectedProgram(DomainModel):
    """Prompt-facing program context from the active search population."""

    node_id: str
    score: float | None = None
    summary: str | None = None
    source: str | None = None
    path: str | None = None
    code: str | None = None
    metrics: dict[str, JSONValue] = Field(default_factory=dict)


class ProjectedArtifactFeedback(DomainModel):
    """Prompt-facing evaluator artifact payload."""

    title: str
    content: str
    artifact_kind: str | None = None
    source_node_id: str | None = None


class ProjectedFeedback(DomainModel):
    """Feedback carried into a projected state."""

    source: str
    content: str


class ProjectedState(DomainModel):
    """State projected into a mutation provider."""

    parent_node_id: str | None = None
    portable_state: PortableStateRef | None = None
    runtime_state: RuntimeStateRef | None = None
    artifacts: list[ProjectedArtifact] = Field(default_factory=_empty_projected_artifacts)
    previous_attempts: list[ProjectedProgram] = Field(default_factory=_empty_projected_programs)
    top_programs: list[ProjectedProgram] = Field(default_factory=_empty_projected_programs)
    diverse_programs: list[ProjectedProgram] = Field(default_factory=_empty_projected_programs)
    artifact_feedback: list[ProjectedArtifactFeedback] = Field(
        default_factory=_empty_artifact_feedback
    )
    feedback: list[ProjectedFeedback] = Field(default_factory=_empty_projected_feedback)
    summary: str | None = None
    lineage_summary: str | None = None
    latest_score_summary: str | None = None
    latest_failure_summary: str | None = None
    best_ancestor_summary: str | None = None
    inspiration_summaries: list[str] = Field(default_factory=_empty_inspiration_summaries)
    inspiration_programs: list[ProjectedProgram] = Field(
        default_factory=_empty_inspiration_programs
    )
    inherited_context: dict[str, JSONValue] = Field(default_factory=dict)
    mutation_surface: dict[str, JSONValue] = Field(default_factory=dict)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


def _empty_tool_grants() -> list[ToolGrant]:
    return []


class ToolGrant(DomainModel):
    """Permission to use a named tool surface."""

    name: str
    scope: str | None = None
    allow_shell: bool = False


class MutationPlan(DomainModel):
    """Planner-facing mutation metadata."""

    summary: str
    expected_outputs: list[str] = Field(default_factory=list)
    tool_grants: list[ToolGrant] = Field(default_factory=_empty_tool_grants)


class MutationRequest(DomainModel):
    """Normalized request passed to a mutation provider."""

    request_id: str
    provider: ProviderSpec
    instruction: MutationInstructionLike
    projected_state: ProjectedState = Field(default_factory=ProjectedState)
    plan: MutationPlan | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider_key(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = {str(key): item for key, item in cast(Mapping[object, object], value).items()}
        if "backend" in payload and "provider" not in payload:
            payload["provider"] = payload.pop("backend")
        projected_state = payload.get("projected_state")
        if projected_state is not None and not isinstance(projected_state, Mapping):
            model_dump = getattr(projected_state, "model_dump", None)
            if callable(model_dump):
                payload["projected_state"] = model_dump(mode="python")
        return payload


def _spec_keys(spec: ProviderSpec) -> tuple[str, ...]:
    keys: list[str] = []
    if spec.provider is not None and spec.model is not None:
        keys.append(f"{spec.kind}:{spec.provider}:{spec.model}")
    if spec.provider is not None:
        keys.append(f"{spec.kind}:{spec.provider}")
    keys.append(spec.kind)
    return tuple(keys)


@runtime_checkable
class ProviderRegistry(Protocol):
    """Resolve a stable provider spec to a concrete provider instance."""

    def resolve(self, spec: ProviderSpec) -> object:
        """Return the provider that should execute the requested spec."""

        ...


class StaticProviderRegistry(ProviderRegistry):
    """Static mapping-based registry with exact-to-broad fallback."""

    def __init__(self, providers: Mapping[str, object]) -> None:
        self._providers = dict(providers)

    def resolve(self, spec: ProviderSpec) -> object:
        for key in _spec_keys(spec):
            provider = self._providers.get(key)
            if provider is not None:
                return provider
        available = ", ".join(sorted(self._providers))
        raise LookupError(
            f"No provider registered for {spec.model_dump(mode='json')}; "
            f"tried keys={list(_spec_keys(spec))}; available=[{available}]"
        )

    @property
    def providers(self) -> Mapping[str, object]:
        return dict(self._providers)


class ProviderUsage(DomainModel):
    """Usage accounting reported by a mutation provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model_cost_usd: float = 0.0
    wallclock_seconds: float | None = None
    agent_turns: int = 0

    def as_budget(self) -> BudgetLedger:
        return BudgetLedger(
            model_input_tokens=self.input_tokens,
            model_output_tokens=self.output_tokens,
            model_cache_read_tokens=self.cache_read_tokens,
            model_cache_creation_tokens=self.cache_creation_tokens,
            model_cost_usd=self.model_cost_usd,
            wallclock_seconds=self.wallclock_seconds or 0.0,
            agent_turns=self.agent_turns,
        )


class ProviderFailure(DomainModel):
    """Structured failure surfaced by a mutation provider."""

    kind: str
    message: str
    retryable: bool = False


def _empty_result_artifacts() -> list[ProjectedArtifact]:
    return []


def _empty_artifact_refs() -> list[ArtifactRef]:
    return []


class MutationResult(DomainModel):
    """Outcome of a mutation provider execution."""

    request_id: str
    provider_kind: str
    status: Literal["ok", "invalid_output", "backend_error", "timeout"] = "ok"
    output_text: str | None = None
    materialization: MaterializationRef | None = None
    portable_state: PortableStateRef | None = None
    runtime_state: RuntimeStateRef | None = None
    child_materialization: MaterializationRef | None = None
    child_portable_state: PortableStateRef | None = None
    child_runtime_state: RuntimeStateRef | None = None
    artifacts: list[ProjectedArtifact] = Field(default_factory=_empty_result_artifacts)
    artifact_refs: list[ArtifactRef] = Field(default_factory=_empty_artifact_refs)
    usage: ProviderUsage | None = None
    budget: BudgetLedger = Field(default_factory=BudgetLedger)
    failure: ProviderFailure | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider_kind_key(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = {str(key): item for key, item in cast(Mapping[object, object], value).items()}
        if "backend_kind" in payload and "provider_kind" not in payload:
            payload["provider_kind"] = payload.pop("backend_kind")
        raw_artifacts = payload.get("artifacts")
        if isinstance(raw_artifacts, list):
            normalized_artifacts: list[object] = []
            for artifact in raw_artifacts:
                if isinstance(artifact, Mapping):
                    normalized_artifacts.append(artifact)
                    continue
                model_dump = getattr(artifact, "model_dump", None)
                normalized_artifacts.append(
                    model_dump(mode="python") if callable(model_dump) else artifact
                )
            payload["artifacts"] = normalized_artifacts
        return payload

    @model_validator(mode="after")
    def _sync_compatibility_fields(self) -> MutationResult:
        if self.child_materialization is None:
            self.child_materialization = self.materialization
        if self.materialization is None:
            self.materialization = self.child_materialization
        if self.child_portable_state is None:
            self.child_portable_state = self.portable_state
        if self.portable_state is None:
            self.portable_state = self.child_portable_state
        if self.child_runtime_state is None:
            self.child_runtime_state = self.runtime_state
        if self.runtime_state is None:
            self.runtime_state = self.child_runtime_state
        if self.usage is not None and self.budget == BudgetLedger():
            self.budget = self.usage.as_budget()
        if self.failure is not None and self.status == "ok":
            self.status = "backend_error"
        return self
