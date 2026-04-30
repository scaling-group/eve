"""Solver population: thin alias over the shared Population base.

The solver population is the Phase 2 candidate pool; keep its wording aligned
with the solver workspace README and Phase 2 logs.
"""

from __future__ import annotations

from scaling_evolve.algorithms.eve.populations.base import Population
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore

SolverEntry = PopulationEntry


class SolverPopulation(Population):
    """Solver candidate population (T)."""

    def __init__(
        self,
        lineage_store: SQLiteLineageStore,
        artifact_store: FSArtifactStore,
        run_id: str,
        **kwargs: object,
    ) -> None:
        super().__init__(lineage_store, artifact_store, run_id, app_kind="eve.solver", **kwargs)
