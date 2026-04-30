"""Solver workspace builder (Phase 2)."""

from __future__ import annotations

import random
import shutil
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
    expose_guidance_skills,
    read_file_tree,
    write_claude_stop_hook_settings,
    write_file_tree,
    write_project_agent_definitions,
)


class SolverWorkspaceBuilder:
    """Build and extract solver workspaces for Phase 2."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        instructions: dict[str, object],
        rollout_prompts: dict[str, object] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.config = config
        self.instructions = instructions
        self.rollout_prompts = rollout_prompts or {}
        self._rng = rng or random.Random()
        self.worker_index: int | None = None

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
        self.worker_index = worker_index
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        ws = self.workspace_root / f"{ts}_{workspace_id}"
        ws.mkdir(parents=True, exist_ok=True)
        optimizer_examples = optimizer_examples or []
        use_optimizer_examples = int(self.config.n_optimizer_examples_phase2) > 0
        solver_examples_dirname = "solver_examples" if use_optimizer_examples else "examples"

        # Optimizer files
        write_file_tree(ws / "guidance", optimizer_files)
        expose_guidance_skills(ws)

        # Reference examples
        examples_dir = ws / solver_examples_dirname
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
                write_file_tree(example_dir / "optimizer", entry.files)
                write_file_tree(example_dir / "logs", entry.logs)
                self._write_yaml(
                    example_dir / "score.yaml",
                    {
                        "example_id": entry.id,
                        "score": entry.score,
                    },
                )

        # Prefill output/ with a base repo checkout overlaid by a randomly
        # chosen reference example.
        output_dir = ws / "output"
        shutil.copytree(self.problem.snapshot_root, output_dir)
        self._write_yaml(
            ws / "score.yaml",
            {
                "optimizer_id": optimizer.id if optimizer is not None else None,
                "sampled_optimizer_example_ids": [entry.id for entry in optimizer_examples],
                "prefill_solver_id": (prefill_solver.id if prefill_solver is not None else None),
                "prefill_solver_score": (
                    prefill_solver.score if prefill_solver is not None else None
                ),
            },
        )
        if prefill_solver is not None:
            write_file_tree(output_dir, prefill_solver.files)
        write_project_agent_definitions(
            ws,
            name="check-runner",
            claude_content=self.problem.render_check_agent_definition(
                self.problem.check_agent_paths["claude"]
            ),
            codex_content=self.problem.render_check_agent_definition(
                self.problem.check_agent_paths["codex"]
            ),
        )
        write_claude_stop_hook_settings(ws)

        return ws, prefill_solver

    def write_readme(self, workspace: Path, content: str) -> None:
        """Persist the full Phase 2 README."""
        (workspace / "README.md").write_text(content.strip() + "\n", encoding="utf-8")

    def write_workspace_agent_instructions(self, workspace: Path, content: str) -> None:
        """Persist workspace-root agent instruction files."""
        payload = content.strip() + "\n"
        for filename in ("AGENTS.md", "CLAUDE.md"):
            (workspace / filename).write_text(payload, encoding="utf-8")

    def extract(self, workspace: Path) -> dict[str, str]:
        """Read the editable candidate files from workspace output/.

        Raises:
            FileNotFoundError: if output/ directory is missing.
            ValueError: if any editable file is missing.
        """
        output_dir = workspace / "output"
        if not output_dir.exists():
            raise FileNotFoundError(f"output/ directory not found in {workspace}")
        files: dict[str, str] = {}
        for rel_path in self.problem.editable_files:
            path = output_dir / rel_path
            if not path.exists():
                raise ValueError(f"editable file {rel_path} is missing in {output_dir}")
            files[rel_path] = path.read_text(encoding="utf-8")
        for folder in self.problem.editable_folders:
            folder_root = output_dir / folder
            if not folder_root.exists():
                continue
            for rel_path, content in read_file_tree(folder_root).items():
                files[str(Path(folder) / rel_path)] = content
        return files

    def extract_optimizer(self, workspace: Path) -> dict[str, str]:
        """Read the optional Phase 2 optimizer artifact from guidance/."""
        optimizer_dir = workspace / "guidance"
        if not optimizer_dir.exists():
            return {}
        return read_file_tree(optimizer_dir)

    def boundary_check_result(self, workspace: Path) -> BoundaryCheckResult:
        """Return the boundary-check result for workspace output/."""
        output_dir = workspace / "output"
        return check_workspace_boundary(
            baseline_root=self.problem.snapshot_root,
            candidate_root=output_dir,
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
        """Persist current-run logs into the workspace and return the same log tree.

        The returned mapping matches the persisted solver entry log shape.
        """
        logs = {
            **{f"optimize/{path}": content for path, content in optimize_logs.items()},
            **{f"evaluate/{path}": content for path, content in evaluate_logs.items()},
        }
        write_file_tree(workspace / "logs", logs)
        return logs

    def write_score_manifest(
        self,
        workspace: Path,
        *,
        sampled_solvers: list[PopulationEntry],
        optimizer: PopulationEntry | None = None,
        sampled_optimizers: list[PopulationEntry] | None = None,
        prefill_solver: PopulationEntry | None,
        produced_solver: PopulationEntry,
    ) -> None:
        """Persist workspace-level score metadata."""
        self._write_yaml(
            workspace / "score.yaml",
            {
                "sampled_solver_examples": [
                    {
                        "example_id": entry.id,
                        "score": entry.score,
                    }
                    for entry in sampled_solvers
                ],
                "optimizer_id": optimizer.id if optimizer is not None else None,
                "sampled_optimizer_example_ids": [entry.id for entry in (sampled_optimizers or [])],
                "prefill_solver_id": prefill_solver.id if prefill_solver is not None else None,
                "produced_solver": {
                    "solver_id": produced_solver.id,
                    "score": produced_solver.score,
                },
            },
        )

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
