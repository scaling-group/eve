"""Optimizer population: thin wrapper over the shared Population base."""

from __future__ import annotations

from scaling_evolve.algorithms.eve.populations.base import Population
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore

# Re-export for backward compatibility.
OptimizerEntry = PopulationEntry


class OptimizerPopulation(Population):
    """Optimizer population (O)."""

    def __init__(
        self,
        lineage_store: SQLiteLineageStore,
        artifact_store: FSArtifactStore,
        run_id: str,
        **kwargs: object,
    ) -> None:
        super().__init__(lineage_store, artifact_store, run_id, app_kind="eve.optimizer", **kwargs)
