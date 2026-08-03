"""Phase 3 optimizer scoring helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy

import optree

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
    """Score Phase 2 optimizer samples without collapsing duplicate parents.

    Each successful Phase 2 result gets a unique temporary result id and starts
    from its parent optimizer's current score. If that result produced a new
    optimizer, the new optimizer receives that result id's updated score.

    The parent optimizer receives the same Elo movement as the temporary result
    id. When the same parent optimizer appears multiple times in one Phase 2
    batch, those per-result Elo deltas are summed back onto the parent.
    """
    if not phase2_results:
        return
    # Treat each successful Phase 2 result as its own temporary result id. A
    # parent optimizer may be sampled multiple times in one batch, and those
    # samples should be scored independently before their deltas are folded back.
    scored_results = [
        (f"phase2_result_{index}", result)
        for index, result in enumerate(phase2_results)
        if result.produced_solver is not None
    ]
    if len(scored_results) < 2:
        return

    current_scores = {result_id: result.optimizer.score for result_id, result in scored_results}
    task_scores = {
        result_id: result.produced_solver.score
        for result_id, result in scored_results
        if result.produced_solver is not None
    }
    updated_result_scores = optimizer_evaluator.update(
        current_scores,
        task_scores,
    )
    parent_base_scores: dict[str, object] = {}
    parent_result_scores: dict[str, list[object]] = {}
    produced_optimizer_scores: dict[str, object] = {}
    for result_id, result in scored_results:
        updated_score = updated_result_scores.get(result_id)
        if updated_score is None:
            continue
        parent_base_scores.setdefault(result.optimizer.id, current_scores[result_id])
        parent_result_scores.setdefault(result.optimizer.id, []).append(updated_score)
        if result.produced_optimizer is not None:
            synced_score = deepcopy(updated_score)
            result.produced_optimizer.score = deepcopy(synced_score)
            produced_optimizer_scores[result.produced_optimizer.id] = synced_score

    updated_scores = {
        optimizer_id: _aggregate_parent_score(
            parent_base_scores[optimizer_id],
            result_scores,
        )
        for optimizer_id, result_scores in parent_result_scores.items()
    }
    optimizer_pop.update_scores(updated_scores)
    if produced_optimizer_scores:
        optimizer_pop.update_scores(produced_optimizer_scores)
    _LOGGER.info(
        "Phase 3: updated Elo for %d optimizers from %d Phase 2 results and synced %d "
        "Phase 2 optimizers",
        len(updated_scores),
        len(scored_results),
        len(produced_optimizer_scores),
    )


def _aggregate_parent_score(
    current_score: object,
    updated_result_scores: Sequence[object],
) -> object:
    """Fold one parent's temporary result scores back into its own score.

    Every temporary result score was computed from the same parent base score.
    For each numeric leaf, the parent moves by the sum of each result's
    ``updated - base`` delta. Non-numeric leaves, including strings, keep the
    parent value. Produced optimizers keep their individual result-level scores;
    only the original parent receives this summed score.
    """
    return optree.tree_map(
        _aggregate_score_leaf,
        current_score,
        *updated_result_scores,
    )


def _aggregate_score_leaf(current_leaf: object, *updated_leaves: object) -> object:
    if not isinstance(current_leaf, (int, float)) or isinstance(current_leaf, bool):
        return current_leaf

    value = current_leaf
    for updated_leaf in updated_leaves:
        if isinstance(updated_leaf, (int, float)) and not isinstance(updated_leaf, bool):
            value += updated_leaf - current_leaf
    return value
