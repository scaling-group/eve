"""Application protocol surface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from scaling_evolve.core.bindings import AssessmentContract, TargetBinding
from scaling_evolve.core.common import DomainModel, JSONValue
from scaling_evolve.core.engine import MaterializationRef
from scaling_evolve.core.enums import ArtifactKind
from scaling_evolve.core.evaluation import EvaluationResult
from scaling_evolve.core.mutation import MutationSurface


def _empty_primary_artifacts() -> list[PrimaryArtifact]:
    return []


class PrimaryArtifact(DomainModel):
    """Primary artifact surfaced by an application."""

    path: str
    kind: ArtifactKind = ArtifactKind.SOURCE
    description: str | None = None


class SeedBundle(DomainModel):
    """Seed material used to initialize a run."""

    seed_id: str
    summary: str | None = None
    primary_artifacts: list[PrimaryArtifact] = Field(default_factory=_empty_primary_artifacts)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ApplicationContext(DomainModel):
    """Minimal application execution context."""

    app_kind: str
    repo_root: str | None = None


@runtime_checkable
class Application(Protocol):
    """Protocol for application-specific materialization and evaluation."""

    kind: str

    def seed(self) -> SeedBundle:
        """Return the seed material for a new run."""

        ...

    def materialize(self, seed: SeedBundle) -> MaterializationRef:
        """Materialize a candidate from the seed bundle."""

        ...

    def evaluate(self, candidate: MaterializationRef) -> EvaluationResult:
        """Evaluate a materialized candidate."""

        ...

    def mutation_surface(self) -> MutationSurface:
        """Describe the allowed mutation surface for this application."""

        ...

    def extract_primary_artifact(self, candidate: MaterializationRef) -> str | None:
        """Return the primary artifact path for a materialized candidate."""

        ...


@runtime_checkable
class TargetBindingProvider(Protocol):
    """Optional application extension for surfacing target binding metadata."""

    def target_binding(self) -> TargetBinding:
        """Return the application target binding."""

        ...


@runtime_checkable
class AssessmentContractProvider(Protocol):
    """Optional application extension for surfacing assessment metadata."""

    def assessment_contract(self) -> AssessmentContract:
        """Return the application assessment contract."""

        ...


@runtime_checkable
class KillableProcess(Protocol):
    """Minimal process interface for evaluator timeout cleanup."""

    def terminate(self) -> object:
        """Request graceful termination."""

        ...

    def join(self, timeout: float | None = None) -> object:
        """Wait for process exit."""

        ...

    def kill(self) -> object:
        """Force termination when graceful shutdown fails."""

        ...

    def is_alive(self) -> bool:
        """Return whether the child process still exists."""

        ...


def terminate_process_with_fallback(
    process: KillableProcess,
    *,
    timeout_seconds: float = 0.1,
) -> None:
    """Terminate a child process, escalating to kill when needed."""

    process.terminate()
    process.join(timeout=timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=timeout_seconds)


def resolve_target_binding(
    app: Application,
    *,
    repo_root: str | None = None,
) -> TargetBinding:
    """Resolve target binding metadata without widening the base protocol."""

    if isinstance(app, TargetBindingProvider):
        return app.target_binding()
    return TargetBinding(kind="managed", repo_root=repo_root)


def resolve_assessment_contract(app: Application) -> AssessmentContract:
    """Resolve assessment metadata without widening the base protocol."""

    if isinstance(app, AssessmentContractProvider):
        return app.assessment_contract()
    return AssessmentContract(kind="objective")
