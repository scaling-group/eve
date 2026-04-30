"""Phase 3 optimizer scoring helpers."""

from __future__ import annotations

import logging
from copy import deepcopy

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result

_LOGGER = logging.getLogger(__name__)


def score_optimizers(
    *,
    optimizers: list[PopulationEntry],
    phase2_results: list[Phase2Result],
    optimizer_pop,
    optimizer_evaluator: object,
) -> None:
    if not phase2_results:
        return
    current_elos = {optimizer.id: optimizer.score for optimizer in optimizers}
    task_scores = {
        result.optimizer.id: result.produced_solver.score
        for result in phase2_results
        if result.produced_solver is not None
    }
    participating_elos = {
        optimizer_id: elo
        for optimizer_id, elo in current_elos.items()
        if optimizer_id in task_scores
    }
    updated_scores = optimizer_evaluator.update(
        participating_elos,
        task_scores,
    )
    optimizer_pop.update_scores(updated_scores)
    produced_optimizer_scores: dict[str, object] = {}
    for result in phase2_results:
        if result.produced_optimizer is None:
            continue
        updated_score = updated_scores.get(result.optimizer.id)
        if updated_score is None:
            continue
        synced_score = deepcopy(updated_score)
        result.produced_optimizer.score = synced_score
        produced_optimizer_scores[result.produced_optimizer.id] = synced_score
    if produced_optimizer_scores:
        optimizer_pop.update_scores(produced_optimizer_scores)
    _LOGGER.info(
        "Phase 3: updated Elo for %d optimizers and synced %d Phase 2 optimizers",
        len(updated_scores),
        len(produced_optimizer_scores),
    )
