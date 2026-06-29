"""Solver workspace builder (Phase 2)."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.workflow.boundary import (
    BoundaryCheckResult,
    check_workspace_boundary,
)
from scaling_evolve.algorithms.eve.workspace.file_tree import (
    expose_guidance_agents,
    expose_guidance_skills,
    read_file_tree,
    write_claude_stop_hook_settings,
    write_file_tree,
)
from scaling_evolve.algorithms.eve.workspace.immutable_renderers.base import (
    render_immutable_files,
)
from scaling_evolve.algorithms.eve.workspace.immutable_renderers.default import (
    DefaultRenderer,
)


@dataclass(frozen=True)
class SolverWorkerConfig:
    """Loaded prompt/immutable assets for one optimizer-side solver worker type."""

    name: str
    weight: float
    immutable_files: dict[str, str]
    immutable_renderer: DefaultRenderer
    boundary_repair_prompt: str
    rollout_prompts: dict[str, object]


class SolverWorkspaceBuilder:
    """Build and extract solver workspaces for Phase 2."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        immutable_files: dict[str, str],
        immutable_renderer: DefaultRenderer | None = None,
        boundary_repair_prompt: str | None = None,
        rollout_prompts: dict[str, object] | None = None,
        worker_configs: list[SolverWorkerConfig] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.config = config
        explicit_worker_configs = worker_configs is not None
        self.worker_configs = list(
            worker_configs
            if worker_configs is not None
            else [
                SolverWorkerConfig(
                    name="default",
                    weight=1.0,
                    immutable_files=dict(immutable_files),
                    immutable_renderer=immutable_renderer or DefaultRenderer(),
                    boundary_repair_prompt=boundary_repair_prompt or "",
                    rollout_prompts=rollout_prompts or {},
                )
            ]
        )
        if not self.worker_configs:
            raise ValueError("worker_configs must contain at least one worker.")
        for worker_config in self.worker_configs:
            self._validate_worker_config(
                worker_config,
                require_boundary_repair=explicit_worker_configs,
            )
        self.rollout_prompts = self.worker_configs[0].rollout_prompts
        self._rng = rng or random.Random()

    def _validate_worker_config(
        self,
        worker_config: SolverWorkerConfig,
        *,
        require_boundary_repair: bool = False,
    ) -> None:
        if worker_config.weight <= 0:
            raise ValueError(f"worker `{worker_config.name}` weight must be positive.")
        if require_boundary_repair and not worker_config.boundary_repair_prompt:
            raise ValueError(f"worker `{worker_config.name}` requires prompt/BOUNDARY_REPAIR.md.")

    def select_worker_config(self, *, worker_index: int | None = None) -> SolverWorkerConfig:
        """Return a worker config using weighted random selection."""
        _ = worker_index
        return self._rng.choices(
            self.worker_configs,
            weights=[worker.weight for worker in self.worker_configs],
            k=1,
        )[0]

    def build(
        self,
        optimizer_files: dict[str, str],
        solvers: list[PopulationEntry],
        workspace_id: str,
        *,
        optimizer: PopulationEntry | None = None,
        worker_index: int | None = None,
        prefill_solver: PopulationEntry | None = None,
        optimizer_examples: list[PopulationEntry] | None = None,
        worker_config: SolverWorkerConfig | None = None,
    ) -> tuple[Path, PopulationEntry | None]:
        """Create a fresh solver workspace directory and populate it.

        Args:
            optimizer_files: extra optimizer files to write at workspace root (e.g.
                optimizer approach, solver description, skills).
            solvers: sampled solver entries to include as reference material.
            workspace_id: unique ID for this workspace.

        Returns:
            (workspace_path, prefill_solver).
        """
        _ = worker_index
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        ws = self.workspace_root / f"{ts}_{workspace_id}"
        ws.mkdir(parents=True, exist_ok=True)
        optimizer_examples = optimizer_examples or []
        use_optimizer_examples = int(self.config.n_optimizer_examples_phase2) > 0

        # Optimizer files
        write_file_tree(ws / "guidance", optimizer_files)
        expose_guidance_skills(ws)
        expose_guidance_agents(ws)

        # Reference examples
        examples_dir = ws / "solver_examples"
        examples_dir.mkdir(exist_ok=True)
        for entry in solvers:
            example_dir = examples_dir / entry.id
            example_dir.mkdir(exist_ok=True)
            write_file_tree(example_dir / "solver", entry.files)
            write_file_tree(
                example_dir / "logs",
                {
                    path: content
                    for path, content in entry.logs.items()
                    if path.startswith("evaluate/")
                },
            )
            self._write_yaml(
                example_dir / "score.yaml",
                {
                    "example_id": entry.id,
                    "score": entry.score,
                },
            )

        if use_optimizer_examples:
            optimizer_examples_dir = ws / "guidance_examples"
            optimizer_examples_dir.mkdir(exist_ok=True)
            for entry in optimizer_examples:
                example_dir = optimizer_examples_dir / entry.id
                example_dir.mkdir(exist_ok=True)
                write_file_tree(example_dir / "guidance", entry.files)
                write_file_tree(example_dir / "logs", entry.logs)
                self._write_yaml(
                    example_dir / "score.yaml",
                    {
                        "example_id": entry.id,
                        "score": entry.score,
                    },
                )

        # Prefill solver/ with a base repo checkout overlaid by a randomly
        # chosen reference example.
        solver_dir = ws / "solver"
        shutil.copytree(self.problem.snapshot_root, solver_dir)
        if prefill_solver is not None:
            write_file_tree(solver_dir, prefill_solver.files)
        write_claude_stop_hook_settings(ws)

        return ws, prefill_solver

    def write_immutable_assets(
        self,
        workspace: Path,
        *,
        optimizer: PopulationEntry | None = None,
        solvers: list[PopulationEntry],
        prefill_solver: PopulationEntry,
        optimizer_examples: list[PopulationEntry] | None = None,
        worker_config: SolverWorkerConfig | None = None,
    ) -> None:
        """Copy immutable assets into a Phase 2 workspace and render README."""
        worker_config = worker_config or self.worker_configs[0]
        immutable_files = self._render_runtime_immutable_files(worker_config)
        write_file_tree(
            workspace,
            render_immutable_files(
                renderer=worker_config.immutable_renderer,
                immutable_files=immutable_files,
                problem=self.problem,
                config=self.config,
                optimizer=optimizer,
                solvers=solvers,
                prefill_solver=prefill_solver,
                optimizer_examples=optimizer_examples,
            ),
        )
        self._write_worker_metadata(workspace, worker_config)

    def _render_runtime_immutable_files(
        self,
        worker_config: SolverWorkerConfig,
    ) -> dict[str, str]:
        return {
            path: self.problem.render_runtime_template(content)
            for path, content in worker_config.immutable_files.items()
        }

    def _write_worker_metadata(self, workspace: Path, worker_config: SolverWorkerConfig) -> None:
        metadata_dir = workspace / ".scaling_evolve"
        metadata_dir.mkdir(exist_ok=True)
        self._write_yaml(
            metadata_dir / "worker.yaml",
            {
                "name": worker_config.name,
                "weight": worker_config.weight,
            },
        )

    def entrypoint_instruction(
        self,
        *,
        optimizer: PopulationEntry | None = None,
        solvers: list[PopulationEntry],
        prefill_solver: PopulationEntry,
        optimizer_examples: list[PopulationEntry] | None = None,
        worker_config: SolverWorkerConfig | None = None,
    ) -> str:
        """Return the Phase 2 entrypoint instruction from the immutable renderer."""
        worker_config = worker_config or self.worker_configs[0]
        return worker_config.immutable_renderer.entrypoint(
            problem=self.problem,
            config=self.config,
            optimizer=optimizer,
            solvers=solvers,
            prefill_solver=prefill_solver,
            optimizer_examples=optimizer_examples,
        )

    def boundary_repair_instruction(
        self,
        boundary_result: object,
        *,
        worker_config: SolverWorkerConfig | None = None,
    ) -> str:
        """Return the configured boundary-repair instruction plus runtime summary."""
        worker_config = worker_config or self.worker_configs[0]
        if not worker_config.boundary_repair_prompt:
            raise ValueError(
                "prompt/BOUNDARY_REPAIR.md is required before EvE can repair boundary "
                "violations. Load it from optimizer.workers.items[].prompt and pass it "
                "to the workspace builder."
            )
        return "\n".join(
            [worker_config.boundary_repair_prompt.strip(), "", boundary_result.summary()]
        )

    def extract(self, workspace: Path) -> dict[str, str]:
        """Read the editable candidate files from workspace solver/.

        Raises:
            FileNotFoundError: if solver/ directory is missing.
            ValueError: if any editable file is missing.
        """
        solver_dir = workspace / "solver"
        if not solver_dir.exists():
            raise FileNotFoundError(f"solver/ directory not found in {workspace}")
        files: dict[str, str] = {}
        for rel_path in self.problem.editable_files:
            path = solver_dir / rel_path
            if not path.exists():
                raise ValueError(f"editable file {rel_path} is missing in {solver_dir}")
            files[rel_path] = path.read_text(encoding="utf-8")
        for folder in self.problem.editable_folders:
            folder_root = solver_dir / folder
            if not folder_root.exists():
                continue
            for rel_path, content in read_file_tree(folder_root).items():
                files[str(Path(folder) / rel_path)] = content
        return files

    def extract_optimizer(
        self,
        workspace: Path,
        *,
        worker_config: SolverWorkerConfig | None = None,
    ) -> dict[str, str]:
        """Read the optional Phase 2 optimizer artifact from guidance/."""
        optimizer_dir = workspace / "guidance"
        if not optimizer_dir.exists():
            return {}
        files = read_file_tree(optimizer_dir)
        overlay_paths = self._immutable_guidance_overlay_paths(
            worker_config or self.worker_configs[0]
        )
        return {path: content for path, content in files.items() if path not in overlay_paths}

    @staticmethod
    def _immutable_guidance_overlay_paths(worker_config: SolverWorkerConfig) -> set[str]:
        """Return guidance paths overwritten by immutable files through symlinked surfaces."""
        overlay_paths: set[str] = set()
        prefixes = (
            (".codex/skills/", "skills/"),
            (".claude/skills/", "skills/"),
            (".codex/agents/", "agents/codex/"),
            (".claude/agents/", "agents/claude/"),
        )
        for path in worker_config.immutable_files:
            for source_prefix, guidance_prefix in prefixes:
                if path.startswith(source_prefix):
                    overlay_paths.add(guidance_prefix + path.removeprefix(source_prefix))
                    break
        return overlay_paths

    def boundary_check_result(self, workspace: Path) -> BoundaryCheckResult:
        """Return the boundary-check result for workspace solver/."""
        solver_dir = workspace / "solver"
        return check_workspace_boundary(
            baseline_root=self.problem.snapshot_root,
            candidate_root=solver_dir,
            editable={
                "files": self.problem.editable_files,
                "folders": self.problem.editable_folders,
            },
        )

    def write_run_logs(
        self,
        workspace: Path,
        *,
        optimize_logs: dict[str, str],
        evaluate_logs: dict[str, str],
    ) -> dict[str, str]:
        """Persist current-run optimize logs and return the full entry log tree.

        Evaluation runs in a separate workspace. Its logs are included in the
        returned population-entry log tree, but are not copied back into this
        Phase 2 solver workspace.
        """
        logs = {
            **{f"optimize/{path}": content for path, content in optimize_logs.items()},
            **{f"evaluate/{path}": content for path, content in evaluate_logs.items()},
        }
        write_file_tree(
            workspace / "logs",
            {f"optimize/{path}": content for path, content in optimize_logs.items()},
        )
        return logs

    def build_phase2_optimizer_log_tree(
        self,
        *,
        run_id: str,
        produced_solver: PopulationEntry,
        optimize_logs: dict[str, str],
        evaluate_logs: dict[str, str],
    ) -> dict[str, str]:
        """Build the optimizer log subtree for one Phase 2 solver-optimization run."""
        step_root = f"{run_id}_{produced_solver.id}"
        logs = {
            **{
                f"{step_root}/solver/{path}": content
                for path, content in produced_solver.files.items()
            },
            **{
                f"{step_root}/logs/optimize/{path}": content
                for path, content in optimize_logs.items()
            },
            **{
                f"{step_root}/logs/evaluate/{path}": content
                for path, content in evaluate_logs.items()
            },
        }
        logs[f"{step_root}/score.yaml"] = yaml.safe_dump(
            {
                "solver_id": produced_solver.id,
                "score": produced_solver.score,
            },
            sort_keys=False,
        )
        return logs

    def _write_yaml(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _write_workspace_doc_copy(self, workspace: Path, *, filename: str, content: str) -> None:
        (workspace / filename).write_text(content, encoding="utf-8")
