"""EveFactory: assembles all components and wires them together.

The factory is the single place that should know how solver and optimizer
workspaces, storage, and the loop runtime fit together.

Usage:

    config = OmegaConf.create({
        "run_id": "my-run-id",
        "run_root": "runs/my_run",
        "loop": {"max_iterations": 50},
    })
    factory = EveFactory.from_config(config)

    # Seed the populations before running
    factory.seed_solver_population([PopulationEntry(...)])
    factory.seed_optimizer_population([PopulationEntry(...)])

    factory.run()

The factory constructs:
  - role-specific session drivers (spawn() only — no AgentProvider, no session store)
  - Two population instances (SolverPopulation + OptimizerPopulation)
    each backed by their own SQLiteLineageStore + FSArtifactStore
  - SolverWorkspaceBuilder + OptimizerWorkspaceBuilder
  - Eve

LocalWorkspaceManager is not used. Workspace directories are created directly
by the workspace builders using loop.workspace_root from config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from scaling_evolve.algorithms.eve.logger import EveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.optimizer_population import (
    OptimizerPopulation,
)
from scaling_evolve.algorithms.eve.populations.solver_population import (
    SolverPopulation,
)
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.runtime.restore import (
    RestoreResult,
    restore_populations_from_run,
)
from scaling_evolve.algorithms.eve.workflow.evaluation import SolverEvaluator
from scaling_evolve.algorithms.eve.workflow.loop import Eve
from scaling_evolve.algorithms.eve.workspace.file_tree import read_file_tree
from scaling_evolve.algorithms.eve.workspace.optimizer_workspace import (
    OptimizerWorkspaceBuilder,
)
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkspaceBuilder,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore

_LOGGER = logging.getLogger(__name__)


@dataclass
class EveFactory:
    """Assembled Eve runtime ready to run."""

    config: DictConfig
    loop: Eve
    solver_lineage_store: SQLiteLineageStore
    optimizer_lineage_store: SQLiteLineageStore

    def seed_solver_population(self, seeds: list[PopulationEntry]) -> None:
        """Add initial solver candidates before the first iteration."""
        for entry in seeds:
            self.loop.solver_pop.add(entry)
        _LOGGER.info("Seeded solver population with %d candidates.", len(seeds))

    def seed_optimizer_population(self, seeds: list[PopulationEntry]) -> None:
        """Add initial optimizers before the first iteration."""
        for entry in seeds:
            self.loop.optimizer_pop.add(entry)
        _LOGGER.info("Seeded optimizer population with %d optimizers.", len(seeds))

    def seed_initial_optimizer(self, *, search_root: Path) -> None:
        """Seed the initial optimizer from config-managed initial optimizer files."""
        initial_optimizer_value = OmegaConf.select(self.config, "optimizer.initial_optimizer")
        initial_optimizer_root = (search_root / str(initial_optimizer_value)).resolve()
        if not initial_optimizer_root.exists():
            raise SystemExit(f"initial_optimizer directory not found: {initial_optimizer_root}")
        if not initial_optimizer_root.is_dir():
            raise SystemExit(f"initial_optimizer must be a directory: {initial_optimizer_root}")
        initial_optimizer_files = read_file_tree(initial_optimizer_root)

        self.seed_optimizer_population(
            [
                PopulationEntry(
                    id=PopulationEntry.make_id("optimizer"),
                    files=initial_optimizer_files,
                    score=self.loop.optimizer_evaluator.initial_score,
                    logs={},
                )
            ]
        )

    def run(self) -> None:
        """Execute the Eve."""
        self.loop.run()

    def restore_from_path(self, source_path: str | Path) -> RestoreResult:
        """Restore previous solver and optimizer populations into this run."""

        return restore_populations_from_run(
            source_path,
            solver_population=self.loop.solver_pop,
            optimizer_population=self.loop.optimizer_pop,
        )

    def restore_from_spec(self, spec) -> RestoreResult:
        """Restore previous populations using an explicit restore spec."""

        return restore_populations_from_run(
            spec.path,
            solver_population=self.loop.solver_pop,
            optimizer_population=self.loop.optimizer_pop,
            solver_ids=spec.solver_ids,
            optimizer_ids=spec.optimizer_ids,
        )

    def close(self) -> None:
        """Release database connections."""
        self.solver_lineage_store.close()
        self.optimizer_lineage_store.close()

    def __enter__(self) -> EveFactory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        solver_evaluator: SolverEvaluator,
        *,
        solver_driver: SessionDriver,
        optimizer_driver: SessionDriver,
        logger: EveLogger | None = None,
        task_problem: RepoTaskProblem | None = None,
    ) -> EveFactory:
        """Assemble an EveFactory from an DictConfig.

        Args:
            config: fully populated DictConfig.
            solver_evaluator: assembled solver evaluator.
            solver_driver: pre-built solver session driver.
            optimizer_driver: pre-built optimizer session driver.

        Returns:
            Assembled EveFactory.
        """
        loop_cfg = config.loop
        workspace_root = Path(loop_cfg.workspace_root)
        workspace_root.mkdir(parents=True, exist_ok=True)

        # --- Storage ---
        solver_run_id = f"{config.run_id}_solver"
        optimizer_run_id = f"{config.run_id}_optimizer"

        solver_lineage_store = SQLiteLineageStore(Path(loop_cfg.solver_db_path))
        optimizer_lineage_store = SQLiteLineageStore(Path(loop_cfg.optimizer_db_path))

        artifact_root = Path(loop_cfg.artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)

        solver_artifact_store = FSArtifactStore(artifact_root, run_id=solver_run_id)
        optimizer_artifact_store = FSArtifactStore(artifact_root, run_id=optimizer_run_id)

        sampling_keys = (
            "phase1_optimizer_population",
            "phase1_solver_population",
            "solver_workspace_prefill",
            "phase2_optimizer_examples",
            "phase2_produced_optimizers",
            "phase4_lead_optimizer",
            "phase4_optimizer_examples",
            "optimizer_workspace_prefill",
            "optimizer_history_logs",
        )
        samplers = {
            key: instantiate(dict(loop_cfg.sampling[key]), _convert_="all") for key in sampling_keys
        }
        optimizer_evaluator = instantiate(config.optimizer.evaluation, _convert_="all")
        instruction_fields = (
            "phase2_readme",
            "phase2_entrypoint",
            "phase2_agent",
            "phase4_readme",
            "phase4_entrypoint",
            "phase4_agent",
        )
        instruction_cfg = {key: dict(config.prompt[key]) for key in instruction_fields}
        instructions = {
            field_name: instantiate(instruction_cfg[field_name], _convert_="all")
            for field_name in instruction_fields
        }
        rollout_prompt_fields = ("budget",)
        rollout_prompt_cfg = {
            key: dict(config.prompt.rollout[key]) for key in rollout_prompt_fields
        }
        rollout_prompts = {
            field_name: instantiate(rollout_prompt_cfg[field_name], _convert_="all")
            for field_name in rollout_prompt_fields
        }

        # --- Populations ---
        solver_pop = SolverPopulation(
            solver_lineage_store,
            solver_artifact_store,
            run_id=solver_run_id,
            config=loop_cfg,
        )
        optimizer_pop = OptimizerPopulation(
            optimizer_lineage_store,
            optimizer_artifact_store,
            run_id=optimizer_run_id,
            config=loop_cfg,
        )

        # --- Workspace builders ---
        solver_ws_root = workspace_root / "solver_workspaces"
        optimizer_ws_root = workspace_root / "optimizer_workspaces"
        if task_problem is None:
            raise ValueError("task_problem is required for Eve solver workspaces")
        solver_workspace_builder = SolverWorkspaceBuilder(
            solver_ws_root,
            problem=task_problem,
            config=loop_cfg,
            instructions=instructions,
            rollout_prompts=rollout_prompts,
        )
        optimizer_workspace_builder = OptimizerWorkspaceBuilder(
            optimizer_ws_root,
            problem=task_problem,
            config=loop_cfg,
            instructions=instructions,
            rollout_prompts=rollout_prompts,
        )

        # --- Loop ---
        loop = Eve(
            solver_pop=solver_pop,
            optimizer_pop=optimizer_pop,
            solver_workspace_builder=solver_workspace_builder,
            optimizer_workspace_builder=optimizer_workspace_builder,
            solver_driver=solver_driver,
            optimizer_driver=optimizer_driver,
            solver_evaluator=solver_evaluator,
            config=loop_cfg,
            instructions=instructions,
            optimizer_evaluator=optimizer_evaluator,
            phase2_optimizer_sampler=samplers["phase1_optimizer_population"],
            phase2_solver_sampler=samplers["phase1_solver_population"],
            phase2_prefill_sampler=samplers["solver_workspace_prefill"],
            phase2_optimizer_examples_sampler=samplers["phase2_optimizer_examples"],
            phase2_produced_optimizer_sampler=samplers["phase2_produced_optimizers"],
            phase4_lead_sampler=samplers["phase4_lead_optimizer"],
            phase4_optimizer_examples_sampler=samplers["phase4_optimizer_examples"],
            phase4_prefill_sampler=samplers["optimizer_workspace_prefill"],
            phase4_history_log_sampler=samplers["optimizer_history_logs"],
            logger=logger,
        )

        return cls(
            config=config,
            loop=loop,
            solver_lineage_store=solver_lineage_store,
            optimizer_lineage_store=optimizer_lineage_store,
        )
