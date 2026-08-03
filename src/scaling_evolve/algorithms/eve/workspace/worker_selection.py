"""Worker-selection strategies for Phase 2."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scaling_evolve.algorithms.eve.workspace.solver_workspace import SolverWorkerConfig


class RandomWorkerSelector:
    """Select workers randomly according to their configured weights."""

    def select(
        self,
        workers: Sequence[SolverWorkerConfig],
        *,
        worker_index: int,
        rng: random.Random,
    ) -> SolverWorkerConfig:
        _ = worker_index
        return rng.choices(
            workers,
            weights=[worker.weight for worker in workers],
            k=1,
        )[0]


class RoundRobinWorkerSelector:
    """Select workers in configured order using the Phase 2 worker index."""

    def select(
        self,
        workers: Sequence[SolverWorkerConfig],
        *,
        worker_index: int,
        rng: random.Random,
    ) -> SolverWorkerConfig:
        _ = rng
        return workers[(worker_index - 1) % len(workers)]
