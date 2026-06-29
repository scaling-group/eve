"""Top-level Eve workflow orchestration."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.logger import EveLogger
from scaling_evolve.algorithms.eve.populations.base import Population
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.runtime.resume import EveCheckpoint, write_checkpoint
from scaling_evolve.algorithms.eve.runtime.snapshots import write_lineage_snapshots
from scaling_evolve.algorithms.eve.workflow.evaluation import SolverEvaluator
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2BatchRunner
from scaling_evolve.algorithms.eve.workflow.phase3 import score_optimizers
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkspaceBuilder,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver

_LOGGER = logging.getLogger(__name__)
_SEED_WORKSPACE_ID = "seed"


class Eve:
    """Explicit co-evolution loop."""

    def __init__(
        self,
        run_id: str,
        solver_pop: Population,
        optimizer_pop: Population,
        solver_workspace_builder: SolverWorkspaceBuilder,
        solver_driver: SessionDriver,
        solver_evaluator: SolverEvaluator,
        config: DictConfig,
        optimizer_evaluator: object,
        phase2_optimizer_sampler: object,
        phase2_solver_sampler: object,
        phase2_prefill_sampler: object,
        phase2_optimizer_examples_sampler: object,
        phase2_produced_optimizer_sampler: object,
        logger: EveLogger | None = None,
    ) -> None:
        self.run_id = run_id
        self.solver_pop = solver_pop
        self.optimizer_pop = optimizer_pop
        self.solver_workspace_builder = solver_workspace_builder
        self.solver_driver = solver_driver
        self.solver_evaluator = solver_evaluator
        self.config = config
        self.optimizer_evaluator = optimizer_evaluator
        self.phase2_optimizer_sampler = phase2_optimizer_sampler
        self.phase2_solver_sampler = phase2_solver_sampler
        self.phase2_prefill_sampler = phase2_prefill_sampler
        self.phase2_optimizer_examples_sampler = phase2_optimizer_examples_sampler
        self.phase2_produced_optimizer_sampler = phase2_produced_optimizer_sampler
        self.logger = logger

    def run(self, *, start_iteration: int = 0) -> None:
        total_solver_evals = self._ensure_seed_solver()
        self._iterations_completed = start_iteration

        for iteration in range(start_iteration, self.config.max_iterations):
            step = iteration + 1
            self._persist_iteration_boundary(step - 1)
            _LOGGER.info("Iteration %d / %d", step, self.config.max_iterations)

            phase2_batch_runner = Phase2BatchRunner(
                solver_workspace_builder=self.solver_workspace_builder,
                driver=self.solver_driver,
                solver_evaluator=self.solver_evaluator,
                step_label=self._step_log_dir(step),
                iteration=step,
                total_iterations=self.config.max_iterations,
                solver_pop=self.solver_pop,
                optimizer_pop=self.optimizer_pop,
                n_workers_phase2=self.config.n_workers_phase2,
                n_solver_examples_phase2=self.config.n_solver_examples_phase2,
                n_optimizer_examples_phase2=self.config.n_optimizer_examples_phase2,
                exclude_all_working_optimizers_from_examples=(
                    self.config.exclude_all_working_optimizers_from_examples
                ),
                n_produced_optimizers_phase2=self.config.produce_optimizer_in_phase2,
                optimizer_sampler=self.phase2_optimizer_sampler,
                solver_sampler=self.phase2_solver_sampler,
                prefill_sampler=self.phase2_prefill_sampler,
                optimizer_examples_sampler=self.phase2_optimizer_examples_sampler,
                produced_optimizer_sampler=self.phase2_produced_optimizer_sampler,
            )
            phase2_results = phase2_batch_runner.run()
            if not phase2_results:
                continue
            optimizers = [result.optimizer for result in phase2_results]
            total_solver_evals += len(phase2_results)

            score_optimizers(
                optimizers=optimizers,
                phase2_results=phase2_results,
                optimizer_pop=self.optimizer_pop,
                optimizer_evaluator=self.optimizer_evaluator,
            )
            try:
                if self.logger is not None:
                    self.logger.on_iteration(
                        iteration=step,
                        solver_entries=self.solver_pop.entries(),
                        optimizer_entries=self.optimizer_pop.entries(),
                        phase2_results=phase2_results,
                    )
            except Exception:
                _LOGGER.exception("logger.on_iteration failed; continuing.")
            self._iterations_completed = step
            self._persist_iteration_boundary(step)

        _LOGGER.info("Eve finished. Total solver evaluations: %d", total_solver_evals)

    @property
    def iterations_completed(self) -> int:
        return getattr(self, "_iterations_completed", 0)

    def _ensure_seed_solver(self) -> int:
        if self.solver_pop.size() > 0:
            return 0

        seed_solver_id = PopulationEntry.make_id("solver")
        workspace, _ = self.solver_workspace_builder.build({}, [], workspace_id=_SEED_WORKSPACE_ID)
        seed_entry, evaluated = self.solver_evaluator.build_seed_entry(
            workspace=workspace,
            seed_solver_id=seed_solver_id,
            solver_workspace_builder=self.solver_workspace_builder,
        )
        self.solver_pop.add(seed_entry)
        _LOGGER.info("Seeded solver population with %s", seed_solver_id)
        return 1 if evaluated else 0

    @staticmethod
    def _step_log_dir(iteration: int) -> str:
        return f"step_{iteration}"

    def _persist_iteration_boundary(self, anchor_iteration: int) -> None:
        # Without a lineage snapshot there is no anchor to roll back to, so the
        # checkpoint would be unusable on resume. A snapshot is required because
        # each iteration mutates existing rows (Phase 3 re-scores optimizer elo
        # in place), so resume cannot be done by simply dropping new rows.
        if not bool(self.config.get("enable_resume", True)):
            return
        write_lineage_snapshots(
            run_root=self.config.workspace_root,
            solver_db_path=self.config.solver_db_path,
            optimizer_db_path=self.config.optimizer_db_path,
            anchor_iteration=anchor_iteration,
        )
        write_checkpoint(
            self.config.workspace_root,
            EveCheckpoint(
                run_id=self.run_id,
                last_completed_iteration=anchor_iteration,
                max_iterations=int(self.config.max_iterations),
            ),
        )
