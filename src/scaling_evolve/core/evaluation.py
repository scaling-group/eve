"""Evaluation protocols and data models."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from scaling_evolve.core.common import DomainModel, JSONValue
from scaling_evolve.core.engine import BudgetLedger
from scaling_evolve.core.storage.models import ArtifactKind, ArtifactRef

FeatureValue = str | int | float | bool
ScoreStatus = str
RecommendedAction = str


class EvalStatus(StrEnum):
    """Evaluation outcomes."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


def _empty_artifact_refs() -> list[ArtifactRef]:
    return []


class ScoreCard(DomainModel):
    """Evaluation result with strict separation between numeric and boolean outputs."""

    primary_score: float
    metrics: dict[str, float] = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)
    features: dict[str, FeatureValue] = Field(default_factory=dict)
    status: ScoreStatus = "ok"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    recommended_action: RecommendedAction | None = None
    summary: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=_empty_artifact_refs)

    def is_better_than(self, other: ScoreCard | None) -> bool:
        """Compare candidates using the primary numeric objective only."""

        if other is None:
            return True
        return self.primary_score > other.primary_score


def _empty_stages() -> list[EvaluationStageResult]:
    return []


def _empty_artifacts() -> list[EvaluationArtifact]:
    return []


class MetricBundle(DomainModel):
    """Metrics and checks emitted by evaluation."""

    metrics: dict[str, float] = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)


class EvaluationArtifact(DomainModel):
    """Artifact payload emitted by an application evaluator."""

    kind: ArtifactKind | str | None = None
    filename: str | None = None
    payload_text: str | None = None
    payload_json: dict[str, JSONValue] | None = None


class EvaluationStageResult(DomainModel):
    """One stage of application evaluation."""

    stage: str
    status: EvalStatus
    detail: str | None = None
    score: float | None = None
    bundle: MetricBundle = Field(default_factory=MetricBundle)
    score_card: ScoreCard | None = None
    budget: BudgetLedger = Field(default_factory=BudgetLedger)
    artifacts: list[EvaluationArtifact] = Field(default_factory=_empty_artifacts)

    @model_validator(mode="after")
    def _sync_stage_scorecard_fields(self) -> EvaluationStageResult:
        if self.score_card is None:
            return self
        if self.score is None:
            self.score = self.score_card.primary_score
        if not self.bundle.metrics and not self.bundle.checks:
            self.bundle = MetricBundle(
                metrics=dict(self.score_card.metrics),
                checks=dict(self.score_card.checks),
            )
        return self


class EvaluationResult(DomainModel):
    """Aggregate evaluation outcome for a candidate."""

    status: EvalStatus
    score: float | None = None
    summary: str | None = None
    metrics: MetricBundle = Field(default_factory=MetricBundle)
    stages: list[EvaluationStageResult] = Field(default_factory=_empty_stages)
    score_card: ScoreCard | None = None
    budget: BudgetLedger = Field(default_factory=BudgetLedger)
    artifacts: list[EvaluationArtifact] = Field(default_factory=_empty_artifacts)

    @model_validator(mode="after")
    def _sync_scorecard_fields(self) -> EvaluationResult:
        if self.score_card is None:
            return self
        if self.score is None:
            self.score = self.score_card.primary_score
        if self.summary is None:
            self.summary = self.score_card.summary
        if not self.metrics.metrics and not self.metrics.checks:
            self.metrics = MetricBundle(
                metrics=dict(self.score_card.metrics),
                checks=dict(self.score_card.checks),
            )
        return self


def terminal_stage_result(evaluation: EvaluationResult) -> EvaluationStageResult | None:
    """Return the last meaningful stage result for a completed evaluation."""

    if not evaluation.stages:
        return None
    if evaluation.status == EvalStatus.FAILED:
        for stage in reversed(evaluation.stages):
            if stage.status == EvalStatus.FAILED:
                return stage
    return evaluation.stages[-1]


def stage_feedback(stage: EvaluationStageResult | None) -> str | None:
    """Extract the most useful prompt-facing feedback for a stage."""

    if stage is None:
        return None
    if stage.detail:
        return stage.detail
    if stage.score_card is not None:
        return stage.score_card.summary
    return None


@runtime_checkable
class RelativeEvaluation(Protocol):
    """Placeholder seam for relative evaluations such as ELO matches."""

    def compare(
        self,
        challenger: object,
        incumbent: object,
        **kwargs: object,
    ) -> EvaluationResult:
        """Compare two candidates and return a relative evaluation result."""

        ...
