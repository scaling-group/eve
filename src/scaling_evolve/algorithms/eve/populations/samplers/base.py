"""Shared replacement-mode sampling helpers for Eve samplers."""

from __future__ import annotations

import random
from typing import Generic, TypeVar

_T = TypeVar("_T")
ReplacementMode = str


class WeightedSamplerBase(Generic[_T]):
    def __init__(self, *, replacement_mode: ReplacementMode) -> None:
        if replacement_mode not in {"no_replacement", "replacement", "auto"}:
            raise ValueError(f"Unsupported replacement_mode: {replacement_mode}")
        self.replacement_mode = replacement_mode

    def _sample_by_mode(
        self,
        items: list[_T],
        weights: list[float],
        *,
        num: int,
        rng: random.Random,
    ) -> list[_T]:
        if num <= 0 or not items:
            return []
        if self.replacement_mode == "replacement":
            return self._weighted_sample_with_replacement(items, weights, num=num, rng=rng)
        if self.replacement_mode == "no_replacement":
            return self._weighted_sample_without_replacement(
                items,
                weights,
                num=min(num, len(items)),
                rng=rng,
            )
        if self.replacement_mode == "auto":
            return self._weighted_sample_auto(items, weights, num=num, rng=rng)
        raise ValueError(f"Unsupported replacement_mode: {self.replacement_mode}")

    def _weighted_sample_without_replacement(
        self,
        items: list[_T],
        weights: list[float],
        *,
        num: int,
        rng: random.Random,
    ) -> list[_T]:
        remaining = sorted(
            list(zip(items, weights, strict=True)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        result: list[_T] = []
        for _ in range(min(num, len(remaining))):
            chosen_index = rng.choices(
                range(len(remaining)),
                weights=[weight for _item, weight in remaining],
                k=1,
            )[0]
            item, _weight = remaining.pop(chosen_index)
            result.append(item)
            remaining.sort(key=lambda pair: pair[1], reverse=True)
        return result

    def _weighted_sample_with_replacement(
        self,
        items: list[_T],
        weights: list[float],
        *,
        num: int,
        rng: random.Random,
    ) -> list[_T]:
        if num <= 0:
            return []
        return rng.choices(items, weights=weights, k=num)

    def _weighted_sample_auto(
        self,
        items: list[_T],
        weights: list[float],
        *,
        num: int,
        rng: random.Random,
    ) -> list[_T]:
        selected = self._weighted_sample_without_replacement(
            items,
            weights,
            num=min(num, len(items)),
            rng=rng,
        )
        if len(selected) >= num:
            return selected
        selected.extend(
            self._weighted_sample_with_replacement(
                items,
                weights,
                num=num - len(selected),
                rng=rng,
            )
        )
        return selected
