"""Rank-based softmax samplers for Eve populations.

Canonical config lives in ``loop.sampling``. Each sampling site has its own
config mapping, for example:

```yaml
sampling:
  phase1_solver_population:
    _target_:
      scaling_evolve.algorithms.eve.populations.samplers.rank_softmax.RankExponentialSumSampler
    features:
      score:
        weight: 1.0
        temperature: 1.0
  phase4_lead_optimizer:
    _target_:
      scaling_evolve.algorithms.eve.populations.samplers.rank_softmax.RankSoftmaxSampler
    temperature: 1.0
```

Supported samplers in this file:

1. ``RankSoftmaxSampler``

   Sort items by a scalar view of their score, assign weight
   ``exp(-rank / temperature)``, then sample without replacement.

2. ``RankExponentialSumSampler``

   Configure a feature PyTree whose structure matches the score PyTree. Each
   leaf is a mapping with numeric ``weight`` and ``temperature``.

   For each configured feature path, items are ranked by that feature value,
   best-first. The final sampling weight is:

   ``sum_i weight_i * exp(-rank_i / temperature_i)``

   These summed weights are used directly for weighted sampling without
   replacement.

With a single configured feature, ``RankExponentialSumSampler`` is equivalent
to ``RankSoftmaxSampler`` when the same temperature is used.
"""

from __future__ import annotations

import math
import random
from typing import TypeVar

import optree
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.samplers.base import (
    ReplacementMode,
    WeightedSamplerBase,
)
from scaling_evolve.algorithms.eve.populations.score import scalar

_T = TypeVar("_T")


class RankExponentialSumSampler(WeightedSamplerBase[_T]):
    def __init__(
        self,
        features: PyTree,
        *,
        replacement_mode: ReplacementMode,
    ) -> None:
        super().__init__(replacement_mode=replacement_mode)
        self.features = features

    def sample(
        self,
        items: list[_T],
        scores: list[PyTree],
        num: int,
        rng: random.Random | None = None,
    ) -> list[_T]:
        rng = rng or random.Random()
        if num <= 0 or not items:
            return []

        feature_specs = self._feature_specs(self.features)
        total_weights = [0.0 for _ in items]
        for path, feature_weight, temperature in feature_specs:
            path_values = [scalar(self._read_pytree_path(score, path)) for score in scores]
            ranked_indices = sorted(
                range(len(items)),
                key=lambda index: path_values[index],
                reverse=True,
            )
            for rank, item_index in enumerate(ranked_indices):
                total_weights[item_index] += feature_weight * math.exp(-(rank / temperature))

        if not any(weight > 0 for weight in total_weights):
            raise ValueError("rank_exponential_sum produced no positive sampling weights")
        return self._sample_by_mode(items, total_weights, num=num, rng=rng)

    def _feature_specs(self, tree: PyTree) -> list[tuple[tuple[str, ...], float, float]]:
        paths, leaves, _treespec = optree.tree_flatten_with_path(
            tree,
            is_leaf=lambda value: (
                isinstance(value, dict) and set(value) == {"weight", "temperature"}
            ),
        )
        specs: list[tuple[tuple[str, ...], float, float]] = []
        for raw_path, leaf in zip(paths, leaves, strict=True):
            if not isinstance(leaf, dict):
                continue
            weight = leaf.get("weight")
            temperature = leaf.get("temperature")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise TypeError("rank_exponential_sum feature weight must be numeric")
            if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
                raise TypeError("rank_exponential_sum feature temperature must be numeric")
            if temperature <= 0:
                raise ValueError("rank_exponential_sum feature temperature must be > 0")
            specs.append((tuple(raw_path), float(weight), float(temperature)))
        if not specs:
            raise TypeError(
                "rank_exponential_sum features must provide leaves with weight and temperature"
            )
        return specs

    def _read_pytree_path(self, tree: PyTree, path: tuple[str, ...]) -> PyTree:
        current = tree
        for key in path:
            if not isinstance(current, dict):
                joined = ".".join(path)
                raise TypeError(f"score path {joined!r} does not resolve to a nested mapping")
            current = current[key]
        return current


class RankSoftmaxSampler(RankExponentialSumSampler):
    def __init__(
        self,
        temperature: float = 1.0,
        *,
        replacement_mode: ReplacementMode,
    ) -> None:
        if temperature <= 0:
            raise ValueError("rank_softmax temperature must be > 0")
        self.temperature = temperature
        super().__init__(
            features=None,
            replacement_mode=replacement_mode,
        )

    def sample(
        self,
        items: list[_T],
        scores: list[PyTree],
        num: int,
        rng: random.Random | None = None,
    ) -> list[_T]:
        rng = rng or random.Random()
        if num <= 0 or not items:
            return []

        scalar_scores = [scalar(score) for score in scores]
        ranked_indices = sorted(
            range(len(items)),
            key=lambda index: scalar_scores[index],
            reverse=True,
        )
        total_weights = [0.0 for _ in items]
        for rank, item_index in enumerate(ranked_indices):
            total_weights[item_index] = math.exp(-(rank / self.temperature))

        return self._sample_by_mode(items, total_weights, num=num, rng=rng)
