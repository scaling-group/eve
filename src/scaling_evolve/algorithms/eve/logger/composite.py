"""Composite logger for Eve runs."""

from __future__ import annotations

from scaling_evolve.algorithms.eve.logger.base import EveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result


class CompositeEveLogger(EveLogger):
    """Fan-out logger that forwards events to multiple Eve loggers."""

    def __init__(self, loggers: list[EveLogger]) -> None:
        self._loggers = list(loggers)

    def on_iteration(
        self,
        *,
        iteration: int,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        phase2_results: list[Phase2Result],
    ) -> None:
        for logger in self._loggers:
            logger.on_iteration(
                iteration=iteration,
                solver_entries=solver_entries,
                optimizer_entries=optimizer_entries,
                phase2_results=phase2_results,
            )

    def finish(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> None:
        for logger in self._loggers:
            logger.finish(
                solver_entries=solver_entries,
                optimizer_entries=optimizer_entries,
                iterations_completed=iterations_completed,
            )

    def write_resume_anchor_summary(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> None:
        for logger in self._loggers:
            logger.write_resume_anchor_summary(
                solver_entries=solver_entries,
                optimizer_entries=optimizer_entries,
                iterations_completed=iterations_completed,
            )
