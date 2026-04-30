"""Reusable helpers for agent-driven evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from scaling_evolve.core.engine import BudgetLedger
from scaling_evolve.core.enums import EvalStatus
from scaling_evolve.core.evaluation import EvaluationResult, MetricBundle, ScoreCard
from scaling_evolve.providers.agent.drivers.base import SessionRollout, SessionSeed


@runtime_checkable
class SpawnOnlyDriver(Protocol):
    """Minimal driver surface required for one-shot evaluation sessions."""

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        """Start an agent session and return its rollout."""

        ...


@dataclass(frozen=True)
class AgentEvaluationRun:
    """Result of executing an agent-backed evaluator."""

    rollout: SessionRollout | None
    result_text: str | None = None
    error: str | None = None


def run_agent_for_result(
    *,
    driver: SpawnOnlyDriver,
    seed: SessionSeed,
    load_result_text,
) -> AgentEvaluationRun:
    """Spawn a one-shot agent session and load its persisted result text."""

    rollout: SessionRollout | None = None
    try:
        rollout = driver.spawn(seed)
        result_text = str(load_result_text(rollout))
    except Exception as error:  # noqa: BLE001
        return AgentEvaluationRun(rollout=rollout, error=str(error))
    return AgentEvaluationRun(rollout=rollout, result_text=result_text)


def load_json_completion_text(rollout: SessionRollout) -> str:
    """Load a JSON completion file path recorded in rollout metadata."""

    completion_path = Path(rollout.state.metadata["completion_path"])  # type: ignore[index]
    return completion_path.read_text(encoding="utf-8")


def evaluation_from_agent_payload(payload: dict[str, object]) -> EvaluationResult:
    """Normalize a JSON agent completion payload into an EvaluationResult."""

    score = float(payload.get("score") or 0.0)
    summary = str(payload.get("summary") or f"score={score:.6f}")
    raw_status = str(payload.get("status") or "ok")
    status = EvalStatus.PASSED if raw_status in {"ok", "passed"} else EvalStatus.FAILED
    raw_metrics = payload.get("metrics")
    metrics = (
        {
            str(key): float(value)
            for key, value in raw_metrics.items()
            if isinstance(key, str) and isinstance(value, int | float)
        }
        if isinstance(raw_metrics, dict)
        else {"score": score}
    )
    raw_checks = payload.get("checks")
    checks = (
        {
            str(key): bool(value)
            for key, value in raw_checks.items()
            if isinstance(key, str) and isinstance(value, bool)
        }
        if isinstance(raw_checks, dict)
        else {}
    )
    wallclock_seconds = (
        float(payload["wallclock_seconds"])
        if isinstance(payload.get("wallclock_seconds"), int | float)
        else 0.0
    )
    return EvaluationResult(
        status=status,
        score=score,
        summary=summary,
        metrics=MetricBundle(metrics=metrics, checks=checks),
        budget=BudgetLedger(
            evaluator_calls=1,
            wallclock_seconds=wallclock_seconds,
        ),
        score_card=ScoreCard(
            primary_score=score,
            metrics=metrics,
            checks=checks,
            summary=summary,
            status="ok" if status == EvalStatus.PASSED else "failed",
        ),
    )


def evaluation_from_agent_completion_text(result_text: str) -> EvaluationResult:
    """Parse a JSON completion file emitted by an evaluation agent."""

    payload = json.loads(result_text)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation completion payload must be a JSON object")
    return evaluation_from_agent_payload(payload)
