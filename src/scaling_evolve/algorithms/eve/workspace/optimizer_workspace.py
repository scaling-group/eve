"""Optimizer workspace builder (Phase 4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.workspace.file_tree import (
    expose_guidance_skills,
    read_file_tree,
    write_claude_stop_hook_settings,
    write_file_tree,
)


class OptimizerWorkspaceBuilder:
    """Build and extract optimizer workspaces for Phase 4."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        instructions: dict[str, object],
        rollout_prompts: dict[str, object] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.config = config
        self.instructions = instructions
        self.rollout_prompts = rollout_prompts or {}
        self.worker_index: int | None = None

    def build(
        self,
        lead: PopulationEntry,
        optimizers: list[PopulationEntry],
        workspace_id: str,
        *,
        worker_index: int | None = None,
        prefill_optimizer: PopulationEntry | None = None,
        example_logs_by_optimizer: dict[str, dict[str, str]] | None = None,
    ) -> tuple[Path, PopulationEntry | None]:
        """Create a fresh optimizer workspace directory and populate it.

        Args:
            lead: the lead optimizer (o*).
            optimizers: sampled optimizer entries used as reference material.
            workspace_id: unique ID for this workspace.

        Returns:
            (workspace_path, prefill_optimizer_entry).
        """
        self.worker_index = worker_index
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        ws = self.workspace_root / f"{ts}_{workspace_id}"
        ws.mkdir(parents=True, exist_ok=True)

        # Guidance files
        write_file_tree(ws / "guidance", lead.files)
        expose_guidance_skills(ws)
        write_claude_stop_hook_settings(ws)
        self.problem.copy_base_repo(ws / "task_base")

        # Reference optimizer examples
        examples_dir = ws / "examples"
        examples_dir.mkdir(exist_ok=True)
        for entry in optimizers:
            example_dir = examples_dir / entry.id
            example_dir.mkdir(exist_ok=True)
            write_file_tree(example_dir / "optimizer", entry.files)
            write_file_tree(
                example_dir / "logs",
                (
                    {}
                    if example_logs_by_optimizer is None
                    else example_logs_by_optimizer.get(entry.id, {})
                ),
            )
            self._write_yaml(
                example_dir / "score.yaml",
                {
                    "optimizer_id": entry.id,
                    "score": entry.score,
                },
            )

        # Prefill output/ with a randomly chosen optimizer example.
        output_dir = ws / "output"
        output_dir.mkdir(exist_ok=True)
        if prefill_optimizer is not None:
            write_file_tree(output_dir, prefill_optimizer.files)
        self._write_yaml(
            ws / "score.yaml",
            {
                "prefill_optimizer_id": (
                    prefill_optimizer.id if prefill_optimizer is not None else None
                ),
                "prefill_example_score": (
                    prefill_optimizer.score if prefill_optimizer is not None else None
                ),
                "sampled_solver_history_ids": self.sampled_solver_history_ids(
                    optimizers,
                    example_logs_by_optimizer,
                ),
            },
        )

        return ws, prefill_optimizer

    def write_readme(self, workspace: Path, content: str) -> None:
        """Persist the full Phase 4 README."""
        (workspace / "README.md").write_text(content.strip() + "\n", encoding="utf-8")

    def write_workspace_agent_instructions(self, workspace: Path, content: str) -> None:
        """Persist workspace-root agent instruction files."""
        payload = content.strip() + "\n"
        for filename in ("AGENTS.md", "CLAUDE.md"):
            (workspace / filename).write_text(payload, encoding="utf-8")

    def extract(self, workspace: Path) -> dict[str, str]:
        """Read the agent-produced optimizer artifact from output/.

        Raises:
            FileNotFoundError: if output/ directory is missing.
            ValueError: if output/ directory is empty.
        """
        output_dir = workspace / "output"
        if not output_dir.exists():
            raise FileNotFoundError(f"output/ directory not found in {workspace}")
        files = read_file_tree(output_dir)
        if not files:
            raise ValueError(f"output/ directory is empty in {workspace}")
        return files

    def write_optimize_log(
        self,
        workspace: Path,
        *,
        optimize_logs: dict[str, str],
    ) -> dict[str, str]:
        """Persist the current optimizer-improvement log into the workspace."""
        logs = {f"optimize/{path}": content for path, content in optimize_logs.items()}
        write_file_tree(workspace / "logs", logs)
        return logs

    def write_score_manifest(
        self,
        workspace: Path,
        *,
        sampled_optimizers: list[PopulationEntry],
        sampled_solver_history_ids: list[str] | None = None,
        lead_optimizer: PopulationEntry,
        prefill_optimizer: PopulationEntry | None,
        produced_optimizer: PopulationEntry,
    ) -> None:
        """Persist workspace-level optimizer score metadata."""
        self._write_yaml(
            workspace / "score.yaml",
            {
                "sampled_examples": [
                    {
                        "optimizer_id": entry.id,
                        "score": entry.score,
                    }
                    for entry in sampled_optimizers
                ],
                "lead_optimizer": {
                    "optimizer_id": lead_optimizer.id,
                    "score": lead_optimizer.score,
                },
                "prefill_optimizer_id": (
                    prefill_optimizer.id if prefill_optimizer is not None else None
                ),
                "sampled_solver_history_ids": list(sampled_solver_history_ids or []),
                "produced_optimizer": {
                    "optimizer_id": produced_optimizer.id,
                    "initial_score": produced_optimizer.score,
                },
            },
        )

    def sampled_solver_history_ids(
        self,
        optimizers: list[PopulationEntry],
        example_logs_by_optimizer: dict[str, dict[str, str]] | None,
    ) -> list[str]:
        """Return the sampled solver-history ids exposed in this workspace."""
        if not example_logs_by_optimizer:
            return []
        sampled_ids: list[str] = []
        seen: set[str] = set()
        for entry in optimizers:
            for path in example_logs_by_optimizer.get(entry.id, {}):
                root, separator, _ = path.partition("/")
                if not separator or root in seen:
                    continue
                sampled_ids.append(root)
                seen.add(root)
        return sampled_ids

    def _write_yaml(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
