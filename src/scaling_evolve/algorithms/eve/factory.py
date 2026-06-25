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
  - SolverWorkspaceBuilder
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
from scaling_evolve.algorithms.eve.prompt_assets import read_required_prompt_text
from scaling_evolve.algorithms.eve.rollout_prompts.default import BudgetPrompt
from scaling_evolve.algorithms.eve.runtime.imports import (
    ImportResult,
    import_populations_from_run,
)
from scaling_evolve.algorithms.eve.workflow.evaluation import SolverEvaluator
from scaling_evolve.algorithms.eve.workflow.loop import Eve
from scaling_evolve.algorithms.eve.workspace.file_tree import read_file_tree
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkerConfig,
    SolverWorkspaceBuilder,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore

_LOGGER = logging.getLogger(__name__)


def _require_directory(path: Path, *, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    if not path.is_dir():
        raise SystemExit(f"{label} must be a directory: {path}")


def _load_solver_worker_config(
    *,
    config: DictConfig,
    search_root: Path,
    name: str,
    weight: float,
    immutable_value: object,
    prompt_value: object,
    immutable_renderer_config: object | None,
    label: str,
) -> SolverWorkerConfig:
    immutable_root = (search_root / str(immutable_value)).resolve()
    _require_directory(immutable_root, label=f"{label} immutable asset directory")
    prompt_root = (search_root / str(prompt_value)).resolve()
    _require_directory(prompt_root, label=f"{label} workflow prompt directory")
    renderer_config = (
        immutable_renderer_config
        if immutable_renderer_config is not None
        else config.optimizer.immutable_renderer
    )
    entrypoint_prompt = read_required_prompt_text(prompt_root, "ENTRYPOINT.md")
    boundary_repair_prompt = read_required_prompt_text(prompt_root, "BOUNDARY_REPAIR.md")
    return SolverWorkerConfig(
        name=name,
        weight=weight,
        immutable_files=read_file_tree(immutable_root),
        immutable_renderer=instantiate(
            renderer_config,
            entrypoint=entrypoint_prompt,
            _convert_="all",
        ),
        boundary_repair_prompt=boundary_repair_prompt,
        rollout_prompts={"budget": BudgetPrompt(prompt_root=prompt_root)},
    )


def _load_solver_worker_configs(
    config: DictConfig,
    *,
    search_root: Path,
) -> list[SolverWorkerConfig]:
    workers_cfg = OmegaConf.select(config, "optimizer.workers")
    if workers_cfg is None:
        raise SystemExit(
            "optimizer.workers is required. Configure optimizer.workers.items with at least "
            "one solver worker."
        )
    selection = OmegaConf.select(workers_cfg, "selection", default="random")
    if selection != "random":
        raise SystemExit("optimizer.workers.selection only supports `random`.")
    raw_items = OmegaConf.select(workers_cfg, "items")
    if raw_items is None or len(raw_items) == 0:
        raise SystemExit("optimizer.workers.items must contain at least one worker.")

    worker_configs: list[SolverWorkerConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_items):
        name = OmegaConf.select(item, "name")
        if not name:
            raise SystemExit(f"optimizer.workers.items[{index}].name is required.")
        name_value = str(name)
        if name_value in seen_names:
            raise SystemExit(f"optimizer.workers.items[{index}].name duplicates `{name_value}`.")
        seen_names.add(name_value)

        weight = OmegaConf.select(item, "weight")
        if weight is None:
            raise SystemExit(f"optimizer.workers.items[{index}].weight is required.")
        try:
            weight_value = float(weight)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"optimizer.workers.items[{index}].weight must be a number.") from exc
        if weight_value <= 0:
            raise SystemExit(f"optimizer.workers.items[{index}].weight must be positive.")

        immutable_value = OmegaConf.select(item, "immutable")
        if immutable_value is None:
            raise SystemExit(f"optimizer.workers.items[{index}].immutable is required.")
        prompt_value = OmegaConf.select(item, "prompt")
        if prompt_value is None:
            raise SystemExit(f"optimizer.workers.items[{index}].prompt is required.")

        worker_configs.append(
            _load_solver_worker_config(
                config=config,
                search_root=search_root,
                name=name_value,
                weight=weight_value,
                immutable_value=immutable_value,
                prompt_value=prompt_value,
                immutable_renderer_config=OmegaConf.select(item, "immutable_renderer"),
                label=f"optimizer worker `{name_value}`",
            )
        )
    return worker_configs


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

    def seed_initial_guidance(self, *, search_root: Path) -> None:
        """Seed the initial optimizer population from config-managed guidance files."""
        initial_guidance_value = OmegaConf.select(self.config, "optimizer.initial_guidance")
        initial_guidance_root = (search_root / str(initial_guidance_value)).resolve()
        if not initial_guidance_root.exists():
            raise SystemExit(f"initial_guidance directory not found: {initial_guidance_root}")
        if not initial_guidance_root.is_dir():
            raise SystemExit(f"initial_guidance must be a directory: {initial_guidance_root}")
        initial_guidance_files = read_file_tree(initial_guidance_root)

        self.seed_optimizer_population(
            [
                PopulationEntry(
                    id=PopulationEntry.make_id("optimizer"),
                    files=initial_guidance_files,
                    score=self.loop.optimizer_evaluator.initial_score,
                    logs={},
                )
            ]
        )

    def run(self, *, start_iteration: int = 0) -> None:
        """Execute the Eve."""
        self.loop.run(start_iteration=start_iteration)

    def import_from_path(self, source_path: str | Path) -> ImportResult:
        """Import previous solver and optimizer populations into this run."""

        return import_populations_from_run(
            source_path,
            solver_population=self.loop.solver_pop,
            optimizer_population=self.loop.optimizer_pop,
        )

    def import_from_spec(self, spec) -> ImportResult:
        """Import previous populations using an explicit import spec."""

        return import_populations_from_run(
            spec.path,
            solver_population=self.loop.solver_pop,
            optimizer_population=self.loop.optimizer_pop,
            solver_ids=spec.solver_ids,
            optimizer_ids=spec.optimizer_ids,
            import_solvers=spec.import_solvers,
            import_optimizers=spec.import_optimizers,
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
        logger: EveLogger | None = None,
        task_problem: RepoTaskProblem | None = None,
        search_root: Path | None = None,
    ) -> EveFactory:
        """Assemble an EveFactory from an DictConfig.

        Args:
            config: fully populated DictConfig.
            solver_evaluator: assembled solver evaluator.
            solver_driver: pre-built solver session driver.

        Returns:
            Assembled EveFactory.
        """
        loop_cfg = config.loop
        search_root = Path(__file__).resolve().parents[4] if search_root is None else search_root
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
        )
        samplers = {
            key: instantiate(dict(loop_cfg.sampling[key]), _convert_="all") for key in sampling_keys
        }
        optimizer_evaluator = instantiate(config.optimizer.evaluation, _convert_="all")
        solver_worker_configs = _load_solver_worker_configs(config, search_root=search_root)

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
        if task_problem is None:
            raise ValueError("task_problem is required for Eve solver workspaces")
        solver_workspace_builder = SolverWorkspaceBuilder(
            solver_ws_root,
            problem=task_problem,
            config=loop_cfg,
            immutable_files={},
            worker_configs=solver_worker_configs,
        )

        # --- Loop ---
        loop = Eve(
            run_id=str(config.run_id),
            solver_pop=solver_pop,
            optimizer_pop=optimizer_pop,
            solver_workspace_builder=solver_workspace_builder,
            solver_driver=solver_driver,
            solver_evaluator=solver_evaluator,
            config=loop_cfg,
            optimizer_evaluator=optimizer_evaluator,
            phase2_optimizer_sampler=samplers["phase1_optimizer_population"],
            phase2_solver_sampler=samplers["phase1_solver_population"],
            phase2_prefill_sampler=samplers["solver_workspace_prefill"],
            phase2_optimizer_examples_sampler=samplers["phase2_optimizer_examples"],
            phase2_produced_optimizer_sampler=samplers["phase2_produced_optimizers"],
            logger=logger,
        )

        return cls(
            config=config,
            loop=loop,
            solver_lineage_store=solver_lineage_store,
            optimizer_lineage_store=optimizer_lineage_store,
        )
