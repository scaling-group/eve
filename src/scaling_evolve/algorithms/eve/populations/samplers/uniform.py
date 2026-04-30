"""Uniform samplers for Eve populations."""

from __future__ import annotations

import random
from typing import TypeVar

from optree import PyTree

from scaling_evolve.algorithms.eve.populations.samplers.base import (
    ReplacementMode,
    WeightedSamplerBase,
)

_T = TypeVar("_T")


class UniformSampler(WeightedSamplerBase[_T]):
    def __init__(
        self,
        *,
        replacement_mode: ReplacementMode,
    ) -> None:
        super().__init__(replacement_mode=replacement_mode)

    def sample(
        self,
        items: list[_T],
        scores: list[PyTree],
        num: int,
        rng: random.Random | None = None,
    ) -> list[_T]:
        _ = scores
        rng = rng or random.Random()
        return self._sample_by_mode(items, [1.0] * len(items), num=num, rng=rng)
