"""Engine-level protocols and portable runtime models."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from scaling_evolve.core.common import (
    DomainModel,
    JSONValue,
    normalize_provider_kind,
    primary_score,
)
from scaling_evolve.core.storage.models import ArtifactRef, MaterializationRef


class ExecutionLifecycle(StrEnum):
    """Lifecycle states for provider-owned runtime execution."""

    SPAWNED = "spawned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    DETACHED = "detached"


class ComputeUsage(DomainModel):
    """Accrued compute usage for a node or mutation result."""

    model_input_tokens: int = 0
    model_output_tokens: int = 0
    model_cache_read_tokens: int = 0
    model_cache_creation_tokens: int = 0
    model_cost_usd: float = 0.0
    evaluator_calls: int = 0
    evaluator_cpu_seconds: float = 0.0
    wallclock_seconds: float = 0.0
    agent_turns: int = 0
    tool_calls: int = 0
    human_attention_budget: int | None = None

    def merged(self, *others: ComputeUsage) -> ComputeUsage:
        """Return a new usage object with numeric fields added together."""

        payload = self.model_dump()
        human_attention_total = self.human_attention_budget
        for other in others:
            payload["model_input_tokens"] += other.model_input_tokens
            payload["model_output_tokens"] += other.model_output_tokens
            payload["model_cache_read_tokens"] += other.model_cache_read_tokens
            payload["model_cache_creation_tokens"] += other.model_cache_creation_tokens
            payload["model_cost_usd"] += other.model_cost_usd
            payload["evaluator_calls"] += other.evaluator_calls
            payload["evaluator_cpu_seconds"] += other.evaluator_cpu_seconds
            payload["wallclock_seconds"] += other.wallclock_seconds
            payload["agent_turns"] += other.agent_turns
            payload["tool_calls"] += other.tool_calls
            if other.human_attention_budget is not None:
                human_attention_total = (human_attention_total or 0) + other.human_attention_budget
        payload["human_attention_budget"] = human_attention_total
        return ComputeUsage.model_validate(payload)

    def compute_units(
        self,
        *,
        cache_credit: float = 0.1,
        output_weight: float = 1.5,
    ) -> float:
        """Return normalized compute units from cache-aware model usage."""

        uncached = self.model_input_tokens + self.model_cache_creation_tokens
        return (
            uncached
            + cache_credit * self.model_cache_read_tokens
            + output_weight * self.model_output_tokens
        )


BudgetLedger = ComputeUsage

# `types.refs` historically surfaced these names alongside runtime refs.
ArtifactRef = ArtifactRef
MaterializationRef = MaterializationRef


class SearchStateLike(DomainModel):
    """Minimal search state placeholder."""

    generation: int = 0
    active_provider: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_active_provider(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "active_backend" in payload and "active_provider" not in payload:
            payload["active_provider"] = payload.pop("active_backend")
        return payload


class PortableStateRef(DomainModel):
    """Reference to provider-portable state."""

    state_id: str
    summary: str | None = None
    artifact: ArtifactRef | None = None
    lineage_summary_ref: ArtifactRef | None = None
    inherited_context_ref: ArtifactRef | None = None
    transcript_digest_ref: ArtifactRef | None = None


class RuntimeStateRef(DomainModel):
    """Reference to provider-owned runtime state."""

    state_id: str
    provider_kind: str = "llm"
    lifecycle: ExecutionLifecycle = ExecutionLifecycle.SPAWNED
    session_id: str | None = None
    workspace_id: str | None = None
    target_repo_root: str | None = None
    workspace_root: str | None = None
    session_cwd: str | None = None
    lease_owner_run_id: str | None = None
    lease_owner_node_id: str | None = None
    lease_purpose: str | None = None
    lease_created_at: str | None = None
    lease_detached_at: str | None = None
    lease_released_at: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @classmethod
    def _normalize_kind(cls, value: object) -> object:
        return normalize_provider_kind(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "backend_kind" in payload and "provider_kind" not in payload:
            payload["provider_kind"] = payload.pop("backend_kind")
        if "provider_kind" in payload:
            payload["provider_kind"] = cls._normalize_kind(payload["provider_kind"])
        return payload

    @field_validator("provider_kind", mode="before")
    @classmethod
    def _normalize_provider_kind_value(cls, value: object) -> object:
        return cls._normalize_kind(value)


class RuntimeState(RuntimeStateRef):
    """Backend-owned runtime state retained on a node."""

    native_ref: str | None = None
    transcript_ref: ArtifactRef | None = None
    changed_files_manifest_ref: ArtifactRef | None = None


class WorkspaceLease(DomainModel):
    """Handle for a leased workspace."""

    workspace_id: str
    root: str
    owner_node_id: str | None = None
    strategy: str | None = None
    purpose: str | None = None
    target_repo_root: str | None = None
    workspace_root: str | None = None
    session_cwd: str | None = None
    created_at: datetime | None = None
    base_snapshot_id: str | None = None
    used_bytes: int | None = None
    used_git_worktree: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.workspace_root is None:
            self.workspace_root = self.root
        if self.session_cwd is None:
            self.session_cwd = self.workspace_root
        if not self.root:
            self.root = self.workspace_root or self.session_cwd or ""


def _empty_trace_refs() -> list[ArtifactRef]:
    return []


class PortableState(DomainModel):
    """Portable context state reconstructed by a mutation context builder."""

    state_id: str | None = None
    workspace_snapshot_ref: ArtifactRef | None = None
    lineage_summary_ref: ArtifactRef | None = None
    inherited_context_ref: ArtifactRef | None = None
    transcript_digest_ref: ArtifactRef | None = None
    artifact_index_ref: ArtifactRef | None = None
    selected_trace_refs: list[ArtifactRef] = Field(default_factory=_empty_trace_refs)
    evaluator_feedback_ref: ArtifactRef | None = None
    inherited_context: dict[str, JSONValue] = Field(default_factory=dict)
    mutation_surface: dict[str, JSONValue] = Field(default_factory=dict)
    inheritance_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    summary: str | None = None


class StopDecision(DomainModel):
    """Decision produced by the stop-condition checker."""

    reached: bool = False
    reason: str | None = None
    detail: str | None = None


class StopConditions:
    """Evaluate run-level iteration and budget limits."""

    def __init__(
        self,
        *,
        max_iterations: int,
        max_model_input_tokens: int | None = None,
        max_model_output_tokens: int | None = None,
        max_compute_units: float | None = None,
        max_evaluator_calls: int | None = None,
        max_wallclock_seconds: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_model_input_tokens = max_model_input_tokens
        self.max_model_output_tokens = max_model_output_tokens
        self.max_compute_units = max_compute_units
        self.max_evaluator_calls = max_evaluator_calls
        self.max_wallclock_seconds = max_wallclock_seconds

    def max_iterations_reached(self, *, completed_iterations: int) -> StopDecision:
        """Return a stop decision when the iteration budget is exhausted."""

        if completed_iterations < self.max_iterations:
            return StopDecision()
        return StopDecision(
            reached=True,
            reason="max_iterations",
            detail=(
                f"Reached max_iterations={self.max_iterations} after "
                f"{completed_iterations} completed iterations."
            ),
        )

    def reached(
        self,
        *,
        elapsed_seconds: float,
        run_budget: BudgetLedger,
    ) -> StopDecision:
        """Evaluate non-iteration stop conditions."""

        if self.max_wallclock_seconds is not None and elapsed_seconds >= float(
            self.max_wallclock_seconds
        ):
            return StopDecision(
                reached=True,
                reason="max_wallclock_seconds",
                detail=(
                    f"Elapsed wallclock {elapsed_seconds:.2f}s exceeded "
                    f"{self.max_wallclock_seconds}s."
                ),
            )
        if (
            self.max_model_input_tokens is not None
            and run_budget.model_input_tokens >= self.max_model_input_tokens
        ):
            return StopDecision(
                reached=True,
                reason="max_model_input_tokens",
                detail=(
                    f"Model input tokens {run_budget.model_input_tokens} reached "
                    f"{self.max_model_input_tokens}."
                ),
            )
        if (
            self.max_model_output_tokens is not None
            and run_budget.model_output_tokens >= self.max_model_output_tokens
        ):
            return StopDecision(
                reached=True,
                reason="max_model_output_tokens",
                detail=(
                    f"Model output tokens {run_budget.model_output_tokens} reached "
                    f"{self.max_model_output_tokens}."
                ),
            )
        if self.max_compute_units is not None:
            compute_units = run_budget.compute_units()
            if compute_units >= self.max_compute_units:
                return StopDecision(
                    reached=True,
                    reason="max_compute_units",
                    detail=(
                        f"Compute units {compute_units:.2f} reached {self.max_compute_units:.2f}."
                    ),
                )
        if (
            self.max_evaluator_calls is not None
            and run_budget.evaluator_calls >= self.max_evaluator_calls
        ):
            return StopDecision(
                reached=True,
                reason="max_evaluator_calls",
                detail=(
                    f"Evaluator calls {run_budget.evaluator_calls} reached "
                    f"{self.max_evaluator_calls}."
                ),
            )
        return StopDecision()


EngineStopConditions = StopConditions


def _empty_node_ids() -> list[str]:
    return []


def _empty_archive_slots() -> dict[str, str]:
    return {}


class RunCheckpointState(DomainModel):
    """Minimal engine state needed for edge-committed resume."""

    schema_version: int = 1
    run_id: str
    checkpoint_seq: int
    resume_count: int = 0
    next_iteration: int = 0
    next_birth_index: int = 0
    next_node_counter: int = 0
    next_edge_counter: int = 0
    active_pool_node_ids: list[str] = Field(default_factory=_empty_node_ids)
    archive_slots: dict[str, str] = Field(default_factory=_empty_archive_slots)
    run_budget: BudgetLedger = Field(default_factory=BudgetLedger)
    stop_logged: bool = False
    last_completed_edge_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def write_run_checkpoint(path: str | Path, checkpoint: RunCheckpointState) -> None:
    """Write one checkpoint JSON atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(f"{target.suffix}.tmp")
    temp_path.write_text(
        f"{json.dumps(checkpoint.model_dump(mode='json'), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temp_path.replace(target)


def load_run_checkpoint(path: str | Path) -> RunCheckpointState:
    """Load one checkpoint JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunCheckpointState.model_validate(payload)


class RunSummary(DomainModel):
    """Compact summary emitted after an engine run completes."""

    run_id: str
    status: str
    node_count: int
    edge_count: int
    best_node_id: str | None = None
    best_score: float | None = None
    best_status: str | None = None


def _best_node(nodes: Sequence[object]) -> object | None:
    if not nodes:
        return None
    return max(
        nodes,
        key=lambda node: (
            primary_score(node),
            -int(getattr(node, "generation", 0) or 0),
            getattr(node, "node_id", ""),
        ),
    )


def build_run_summary(
    lineage_store: object,
    *,
    run_id: str,
    status: str,
) -> RunSummary:
    """Build a compact summary from persisted run lineage."""

    nodes = tuple(lineage_store.list_nodes(run_id))
    edges = tuple(lineage_store.list_edges(run_id))
    best = _best_node(nodes)
    return RunSummary(
        run_id=run_id,
        status=status,
        node_count=len(nodes),
        edge_count=len(edges),
        best_node_id=(getattr(best, "node_id", None) if best is not None else None),
        best_score=(getattr(best, "primary_score", None) if best is not None else None),
        best_status=(getattr(best, "status", None) if best is not None else None),
    )


@runtime_checkable
class SurvivorPolicy(Protocol):
    """Protocol for survivor selection."""

    def select(self, population: Sequence[object]) -> Sequence[object]:
        """Choose which nodes survive into the next population."""

        ...


class TopScoreSurvivorPolicy:
    """Preserve the current top-score-first survivor ordering."""

    def select(self, population: Sequence[object]) -> Sequence[object]:
        return tuple(
            sorted(
                population,
                key=lambda candidate: (
                    -primary_score(candidate),
                    getattr(candidate, "node_id", ""),
                ),
            )
        )


@runtime_checkable
class IterationReporter(Protocol):
    """Optional runtime hook for iteration-level reporting."""

    def on_seed(self, seed: object, *, run_budget: BudgetLedger) -> None:
        """Record seed state before the first mutation iteration."""

    def on_iteration_complete(self, *, iteration: int, run_budget: BudgetLedger) -> None:
        """Record a completed search iteration."""

    def on_child_complete(
        self,
        *,
        prepared: object,
        executed: object,
        child_node: object,
        archive_decision: object,
        run_budget: BudgetLedger,
        novelty_payload: dict[str, object] | None = None,
    ) -> None:
        """Record one completed child mutation."""


@runtime_checkable
class EnginePersister(Protocol):
    """Persist engine-side mutations and bookkeeping."""

    def persist(self, *args: object, **kwargs: object) -> object:
        """Persist one engine-side transition."""

        ...
