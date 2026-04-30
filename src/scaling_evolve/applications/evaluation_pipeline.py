"""Reusable cascade evaluation pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from scaling_evolve.core.engine import MaterializationRef
from scaling_evolve.core.enums import EvalStatus
from scaling_evolve.core.evaluation import (
    BudgetLedger,
    EvaluationArtifact,
    EvaluationResult,
    EvaluationStageResult,
    MetricBundle,
    ScoreCard,
    stage_feedback,
)


def _merge_bundle(left: MetricBundle, right: MetricBundle) -> MetricBundle:
    return MetricBundle(
        metrics={**left.metrics, **right.metrics},
        checks={**left.checks, **right.checks},
    )


@runtime_checkable
class EvaluationStage(Protocol):
    """One named stage in a cascade evaluation pipeline."""

    name: str

    def run(self, candidate: MaterializationRef) -> EvaluationStageResult:
        """Run the stage against a materialized candidate."""

        ...


class CallableEvaluationStage:
    """Simple adapter for stage functions."""

    def __init__(
        self,
        name: str,
        runner: Callable[[MaterializationRef], EvaluationStageResult],
    ) -> None:
        self.name = name
        self._runner = runner

    def run(self, candidate: MaterializationRef) -> EvaluationStageResult:
        result = self._runner(candidate)
        if result.stage == self.name:
            return result
        return result.model_copy(update={"stage": self.name})


class CascadeEvaluationPipeline:
    """Run staged evaluation with fail-fast and threshold short-circuiting."""

    def __init__(
        self,
        stages: Sequence[EvaluationStage],
        *,
        thresholds: Sequence[float] = (0.5, 0.75, 0.9),
    ) -> None:
        self.stages = tuple(stages)
        self.thresholds = tuple(thresholds)

    def evaluate(self, candidate: MaterializationRef) -> EvaluationResult:
        """Run stages in order until failure, threshold miss, or completion."""

        stage_results: list[EvaluationStageResult] = []
        merged_bundle = MetricBundle()
        merged_budget = BudgetLedger()
        merged_artifacts: list[EvaluationArtifact] = []
        final_score_card: ScoreCard | None = None

        for index, stage in enumerate(self.stages):
            result = stage.run(candidate)
            if result.stage != stage.name:
                result = result.model_copy(update={"stage": stage.name})
            stage_results.append(result)
            merged_bundle = _merge_bundle(merged_bundle, result.bundle)
            merged_budget = merged_budget.merged(result.budget)
            merged_artifacts.extend(result.artifacts)
            if result.score_card is not None:
                final_score_card = result.score_card

            if result.status != EvalStatus.PASSED:
                return self._result_from_failure(
                    result,
                    stage_results=stage_results,
                    bundle=merged_bundle,
                    budget=merged_budget,
                    artifacts=merged_artifacts,
                )

            threshold = self.thresholds[index] if index < len(self.thresholds) else None
            threshold_value = self._threshold_value(result, merged_bundle)
            if (
                threshold is not None
                and threshold_value is not None
                and threshold_value < threshold
            ):
                return self._result_from_threshold_miss(
                    result,
                    threshold=threshold,
                    threshold_value=threshold_value,
                    stage_results=stage_results,
                    bundle=merged_bundle,
                    budget=merged_budget,
                    artifacts=merged_artifacts,
                )

        if final_score_card is not None:
            return EvaluationResult(
                status=EvalStatus.PASSED,
                score_card=final_score_card,
                stages=stage_results,
                budget=merged_budget,
                artifacts=merged_artifacts,
            )

        last_stage = stage_results[-1] if stage_results else None
        return EvaluationResult(
            status=EvalStatus.PASSED,
            score=last_stage.score if last_stage is not None else None,
            summary=stage_feedback(last_stage) or "All evaluation stages passed.",
            metrics=merged_bundle,
            stages=stage_results,
            budget=merged_budget,
            artifacts=merged_artifacts,
        )

    def _result_from_failure(
        self,
        stage: EvaluationStageResult,
        *,
        stage_results: Sequence[EvaluationStageResult],
        bundle: MetricBundle,
        budget: BudgetLedger,
        artifacts: Sequence[EvaluationArtifact],
    ) -> EvaluationResult:
        if stage.score_card is not None:
            return EvaluationResult(
                status=EvalStatus.FAILED,
                score_card=stage.score_card,
                stages=list(stage_results),
                budget=budget,
                artifacts=list(artifacts),
            )
        return EvaluationResult(
            status=EvalStatus.FAILED,
            score=stage.score,
            summary=stage_feedback(stage) or f"Stage `{stage.stage}` failed.",
            metrics=bundle,
            stages=list(stage_results),
            budget=budget,
            artifacts=list(artifacts),
        )

    def _result_from_threshold_miss(
        self,
        stage: EvaluationStageResult,
        *,
        threshold: float,
        threshold_value: float,
        stage_results: Sequence[EvaluationStageResult],
        bundle: MetricBundle,
        budget: BudgetLedger,
        artifacts: Sequence[EvaluationArtifact],
    ) -> EvaluationResult:
        cascade_note = (
            f"Short-circuited after `{stage.stage}`: score {threshold_value:.4f} "
            f"missed threshold {threshold:.4f}."
        )
        score_card = stage.score_card
        if score_card is None:
            score_card = ScoreCard(
                primary_score=threshold_value,
                metrics=dict(bundle.metrics),
                checks=dict(bundle.checks),
                status="ok",
                summary=cascade_note,
            )
        else:
            score_card = score_card.model_copy(
                update={
                    "primary_score": threshold_value,
                    "metrics": dict(bundle.metrics),
                    "checks": dict(bundle.checks),
                }
            )
        return EvaluationResult(
            status=EvalStatus.PASSED,
            score_card=score_card,
            summary=cascade_note,
            stages=list(stage_results),
            budget=budget,
            artifacts=list(artifacts),
        )

    def _threshold_value(
        self,
        stage: EvaluationStageResult,
        bundle: MetricBundle,
    ) -> float | None:
        for metric_name in ("primary_score", "score", "combined_score", "target_ratio"):
            metric_value = bundle.metrics.get(metric_name)
            if metric_value is not None:
                return metric_value
        if stage.score_card is not None:
            return stage.score_card.primary_score
        return stage.score
