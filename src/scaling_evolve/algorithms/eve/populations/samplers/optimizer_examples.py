"""Joint optimizer-reference sampling policies for Phase 2 worker batches.

The concrete policies cover two orthogonal choices: whether workers share one
draw or receive independent draws, and whether eligibility excludes all working
optimizers or only the current worker. Returned lists are ordered to match the
``working_optimizers`` input, including when a working optimizer is repeated.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Protocol

from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry


class _PopulationEntrySampler(Protocol):
    def sample(
        self,
        items: list[PopulationEntry],
        scores: list[PyTree],
        num: int,
        rng: random.Random | None = None,
    ) -> list[PopulationEntry]: ...


class JointOptimizerExampleSampler(ABC):
    """Allocate reference optimizers jointly across a Phase 2 worker batch."""

    def __init__(self, base_sampler: _PopulationEntrySampler) -> None:
        self.base_sampler = base_sampler

    @abstractmethod
    def sample(
        self,
        optimizer_entries: list[PopulationEntry],
        working_optimizers: list[PopulationEntry],
        num: int,
        rng: random.Random,
    ) -> list[list[PopulationEntry]]:
        """Return one ordered reference list per working optimizer."""

    def _sample_entries(
        self,
        entries: list[PopulationEntry],
        num: int,
        rng: random.Random,
    ) -> list[PopulationEntry]:
        if num <= 0 or not entries:
            return []
        return list(
            self.base_sampler.sample(
                entries,
                [entry.score for entry in entries],
                num,
                rng=rng,
            )
        )


class SharedExcludeAllWorkingSampler(JointOptimizerExampleSampler):
    """Share one sample drawn after excluding every working optimizer."""

    def sample(
        self,
        optimizer_entries: list[PopulationEntry],
        working_optimizers: list[PopulationEntry],
        num: int,
        rng: random.Random,
    ) -> list[list[PopulationEntry]]:
        working_ids = {entry.id for entry in working_optimizers}
        eligible_entries = [entry for entry in optimizer_entries if entry.id not in working_ids]
        shared_references = self._sample_entries(eligible_entries, num, rng)
        return [list(shared_references) for _ in working_optimizers]


class PerWorkerExcludeAllWorkingSampler(JointOptimizerExampleSampler):
    """Draw independently per worker after excluding every working optimizer."""

    def sample(
        self,
        optimizer_entries: list[PopulationEntry],
        working_optimizers: list[PopulationEntry],
        num: int,
        rng: random.Random,
    ) -> list[list[PopulationEntry]]:
        working_ids = {entry.id for entry in working_optimizers}
        eligible_entries = [entry for entry in optimizer_entries if entry.id not in working_ids]
        return [self._sample_entries(eligible_entries, num, rng) for _ in working_optimizers]


class PerWorkerExcludeSelfSampler(JointOptimizerExampleSampler):
    """Draw independently per worker after excluding only that worker itself."""

    def sample(
        self,
        optimizer_entries: list[PopulationEntry],
        working_optimizers: list[PopulationEntry],
        num: int,
        rng: random.Random,
    ) -> list[list[PopulationEntry]]:
        return [
            self._sample_entries(
                [entry for entry in optimizer_entries if entry.id != optimizer.id],
                num,
                rng,
            )
            for optimizer in working_optimizers
        ]


class SharedCandidatesExcludeSelfSampler(JointOptimizerExampleSampler):
    """Share an ordered candidate draw, then remove each worker itself."""

    def sample(
        self,
        optimizer_entries: list[PopulationEntry],
        working_optimizers: list[PopulationEntry],
        num: int,
        rng: random.Random,
    ) -> list[list[PopulationEntry]]:
        if num <= 0:
            return [[] for _ in working_optimizers]
        shared_candidates = self._sample_entries(optimizer_entries, num + 1, rng)
        return [
            [entry for entry in shared_candidates if entry.id != optimizer.id][:num]
            for optimizer in working_optimizers
        ]
