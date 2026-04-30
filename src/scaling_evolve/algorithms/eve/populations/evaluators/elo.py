"""Optimizer evaluation helpers for Eve.

Canonical config lives in ``optimizer.evaluation``.

Supported evaluators:

1. ``ScalarEloEvaluator``

   Use a single scalar task score per optimizer. The optimizer's own persisted
   score is expected to be a mapping with numeric ``elo``.

2. ``VectorEloEvaluator``

   Use multiple solver-score features, each with its own ``k_factor``. The
   ``k_factors`` field is itself a PyTree whose structure matches the nested
   solver score. Every numeric leaf defines one feature path plus the Elo
   strength to apply at that path.
"""

from __future__ import annotations

import math

import optree
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.score import scalar


class VectorEloEvaluator:
    def __init__(self, k_factors: object, initial_score: object | None = None) -> None:
        self.k_factors = k_factors
        self.initial_score = initial_score

    def update(
        self,
        current_elos: dict[str, PyTree],
        task_scores: dict[str, PyTree],
    ) -> dict[str, PyTree]:
        optimizer_ids = [
            optimizer_id for optimizer_id in current_elos if optimizer_id in task_scores
        ]
        if len(optimizer_ids) < 2:
            return dict(current_elos)

        feature_specs = self._flatten_numeric_tree(
            self.k_factors,
            tree_name="vector_elo k_factors",
        )
        ratings = {
            optimizer_id: scalar(current_elos[optimizer_id], preferred_key="elo")
            for optimizer_id in optimizer_ids
        }
        deltas = {optimizer_id: 0.0 for optimizer_id in optimizer_ids}
        for path, k_factor in feature_specs:
            outcomes = {
                optimizer_id: scalar(self._get_pytree_path(task_scores[optimizer_id], path))
                for optimizer_id in optimizer_ids
            }
            feature_deltas = self._elo_deltas(ratings, outcomes, k_factor=k_factor)
            for optimizer_id, delta in feature_deltas.items():
                deltas[optimizer_id] += delta
        updated = dict(current_elos)
        for optimizer_id, rating in ratings.items():
            updated[optimizer_id] = {"elo": rating + deltas[optimizer_id]}
        return updated

    def _elo_deltas(
        self,
        ratings: dict[str, float],
        outcomes: dict[str, float],
        *,
        k_factor: float,
    ) -> dict[str, float]:
        optimizer_ids = [optimizer_id for optimizer_id in ratings if optimizer_id in outcomes]
        deltas = {optimizer_id: 0.0 for optimizer_id in optimizer_ids}
        for index, left_id in enumerate(optimizer_ids):
            left_rating = ratings[left_id]
            left_outcome = outcomes[left_id]
            for right_id in optimizer_ids[index + 1 :]:
                right_rating = ratings[right_id]
                right_outcome = outcomes[right_id]
                expected_left = 1.0 / (1.0 + math.pow(10.0, (right_rating - left_rating) / 400.0))
                actual_left = self._elo_outcome(left_outcome, right_outcome)
                deltas[left_id] += k_factor * (actual_left - expected_left)
                deltas[right_id] += k_factor * ((1.0 - actual_left) - (1.0 - expected_left))
        return deltas

    def _elo_outcome(self, left: float, right: float) -> float:
        if left > right:
            return 1.0
        if left < right:
            return 0.0
        return 0.5

    def _flatten_numeric_tree(
        self,
        tree: object,
        *,
        tree_name: str,
    ) -> list[tuple[tuple[str, ...], float]]:
        paths, leaves, _treespec = optree.tree_flatten_with_path(tree)
        if not leaves:
            raise TypeError(f"{tree_name} must contain at least one numeric leaf")

        flattened: list[tuple[tuple[str, ...], float]] = []
        for raw_path, leaf in zip(paths, leaves, strict=True):
            if not isinstance(leaf, (int, float)) or isinstance(leaf, bool):
                raise TypeError(f"{tree_name} must be a PyTree of numeric leaves")
            flattened.append((tuple(raw_path), float(leaf)))
        return flattened

    def _get_pytree_path(self, tree: PyTree, path: tuple[str, ...]) -> PyTree:
        current = tree
        for key in path:
            if not isinstance(current, dict):
                joined = ".".join(path)
                raise TypeError(f"score path {joined!r} does not resolve to a nested mapping")
            if key not in current:
                joined = ".".join(path)
                raise KeyError(f"score path {joined!r} is missing key {key!r}")
            current = current[key]
        return current


class ScalarEloEvaluator(VectorEloEvaluator):
    def __init__(self, k_factor: float = 32.0, initial_score: object | None = None) -> None:
        self.k_factor = k_factor
        super().__init__(k_factors={"score": k_factor}, initial_score=initial_score)

    def update(
        self,
        current_elos: dict[str, PyTree],
        task_scores: dict[str, PyTree],
    ) -> dict[str, PyTree]:
        scalar_task_scores = {
            optimizer_id: {"score": scalar(score)} for optimizer_id, score in task_scores.items()
        }
        return super().update(current_elos, scalar_task_scores)
