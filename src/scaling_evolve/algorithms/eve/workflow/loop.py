"""Top-level Eve workflow orchestration."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.logger import EveLogger
from scaling_evolve.algorithms.eve.populations.base import Population
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.evaluation import SolverEvaluator
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2BatchRunner
from scaling_evolve.algorithms.eve.workflow.phase3 import score_optimizers
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4BatchRunner
from scaling_evolve.algorithms.eve.workspace.optimizer_workspace import (
    OptimizerWorkspaceBuilder,
)
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkspaceBuilder,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver

_LOGGER = logging.getLogger(__name__)
_SEED_WORKSPACE_ID = "seed"


class Eve:
    """Explicit 4-phase co-evolution loop."""

    def __init__(
        self,
        solver_pop: Population,
        optimizer_pop: Population,
        solver_workspace_builder: SolverWorkspaceBuilder,
        optimizer_workspace_builder: OptimizerWorkspaceBuilder,
        solver_driver: SessionDriver,
        optimizer_driver: SessionDriver,
        solver_evaluator: SolverEvaluator,
        config: DictConfig,
        instructions: dict[str, object],
        optimizer_evaluator: object,
        phase2_optimizer_sampler: object,
        phase2_solver_sampler: object,
        phase2_prefill_sampler: object,
        phase2_optimizer_examples_sampler: object,
        phase2_produced_optimizer_sampler: object,
        phase4_lead_sampler: object,
        phase4_optimizer_examples_sampler: object,
        phase4_prefill_sampler: object,
        phase4_history_log_sampler: object,
        logger: EveLogger | None = None,
    ) -> None:
        self.solver_pop = solver_pop
        self.optimizer_pop = optimizer_pop
        self.solver_workspace_builder = solver_workspace_builder
        self.optimizer_workspace_builder = optimizer_workspace_builder
        self.solver_driver = solver_driver
        self.optimizer_driver = optimizer_driver
        self.solver_evaluator = solver_evaluator
        self.config = config
        self.instructions = instructions
        self.optimizer_evaluator = optimizer_evaluator
        self.phase2_optimizer_sampler = phase2_optimizer_sampler
        self.phase2_solver_sampler = phase2_solver_sampler
        self.phase2_prefill_sampler = phase2_prefill_sampler
        self.phase2_optimizer_examples_sampler = phase2_optimizer_examples_sampler
        self.phase2_produced_optimizer_sampler = phase2_produced_optimizer_sampler
        self.phase4_lead_sampler = phase4_lead_sampler
        self.phase4_optimizer_examples_sampler = phase4_optimizer_examples_sampler
        self.phase4_prefill_sampler = phase4_prefill_sampler
        self.phase4_history_log_sampler = phase4_history_log_sampler
        self.logger = logger

    def run(self) -> None:
        total_solver_evals = self._ensure_seed_solver()
        self._iterations_completed = 0

        for iteration in range(self.config.max_iterations):
            step = iteration + 1
            self._snapshot_lineage_state(step - 1)
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
            # Keep Phase 2 logging and Phase 4 scoring in the same iteration so
            # workspace artifacts remain easy to inspect after a run.
            total_solver_evals += len(phase2_results)

            score_optimizers(
                optimizers=optimizers,
                phase2_results=phase2_results,
                optimizer_pop=self.optimizer_pop,
                optimizer_evaluator=self.optimizer_evaluator,
            )
            if int(self.config.n_workers_phase4) <= 0:
                _LOGGER.info(
                    "Phase 4 skipped because n_workers_phase4=%d.",
                    int(self.config.n_workers_phase4),
                )
                phase4_results = []
            else:
                phase4_batch_runner = Phase4BatchRunner(
                    optimizer_workspace_builder=self.optimizer_workspace_builder,
                    driver=self.optimizer_driver,
                    step_label=self._step_log_dir(step),
                    iteration=step,
                    optimizer_pop=self.optimizer_pop,
                    lead_sampler=self.phase4_lead_sampler,
                    optimizer_examples_sampler=self.phase4_optimizer_examples_sampler,
                    prefill_sampler=self.phase4_prefill_sampler,
                    history_log_sampler=self.phase4_history_log_sampler,
                    n_workers_phase4=self.config.n_workers_phase4,
                    n_optimizer_examples_phase4=self.config.n_optimizer_examples_phase4,
                    n_latest_solver_logs=self.config.n_logs_per_example_phase4,
                )
                phase4_results = phase4_batch_runner.run()
            try:
                if self.logger is not None:
                    self.logger.on_iteration(
                        iteration=step,
                        solver_entries=self.solver_pop.entries(),
                        optimizer_entries=self.optimizer_pop.entries(),
                        phase2_results=phase2_results,
                        phase4_results=phase4_results,
                    )
            except Exception:
                _LOGGER.exception("logger.on_iteration failed; continuing.")
            self._iterations_completed = step

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

    def _snapshot_lineage_state(self, anchor_iteration: int) -> None:
        if not bool(self.config.get("enable_iter_snapshots", True)):
            return

        snapshot_root = Path(str(self.config.workspace_root)) / ".snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        solver_snapshot = snapshot_root / f"solver_lineage_iter_{anchor_iteration}.db"
        optimizer_snapshot = snapshot_root / f"optimizer_lineage_iter_{anchor_iteration}.db"

        self._backup_sqlite_db(Path(str(self.config.solver_db_path)), solver_snapshot)
        self._backup_sqlite_db(Path(str(self.config.optimizer_db_path)), optimizer_snapshot)
        self._prune_old_snapshots(snapshot_root)

    @staticmethod
    def _backup_sqlite_db(source_path: Path, dest_path: Path) -> None:
        if not source_path.exists():
            return
        temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp")
        source_conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        dest_conn = sqlite3.connect(temp_path)
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
            source_conn.close()
        temp_path.replace(dest_path)

    def _prune_old_snapshots(self, snapshot_root: Path) -> None:
        retain = int(self.config.get("iter_snapshot_retain", 3))
        if retain <= 0:
            return
        for prefix in ("solver_lineage_iter_", "optimizer_lineage_iter_"):
            snapshots = sorted(snapshot_root.glob(f"{prefix}*.db"))
            if len(snapshots) <= retain:
                continue
            for snapshot_path in snapshots[:-retain]:
                snapshot_path.unlink(missing_ok=True)
