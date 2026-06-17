"""Build standalone workspaces for evaluating solver candidates."""

from __future__ import annotations

import shutil
from pathlib import Path

from scaling_evolve.algorithms.eve.workspace.file_tree import write_file_tree

EVALUATION_WORKSPACES_DIRNAME = "evaluation_workspaces"


class EvaluationWorkspaceBuilder:
    """Build evaluation workspaces for single solver candidates."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        immutable_files: dict[str, str] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.immutable_files = dict(immutable_files or {})

    def build(
        self,
        solver_workspace: Path,
        *,
        optimize_logs: dict[str, str],
    ) -> Path:
        """Create a fresh evaluation workspace from the current solver workspace."""
        eval_workspace = self.workspace_root / solver_workspace.name
        if eval_workspace.exists():
            shutil.rmtree(eval_workspace)
        eval_workspace.mkdir(parents=True)

        shutil.copytree(solver_workspace / "output", eval_workspace / "output")
        logs_dir = solver_workspace / "logs"
        if logs_dir.exists():
            shutil.copytree(logs_dir, eval_workspace / "logs")
        else:
            (eval_workspace / "logs").mkdir()
        write_file_tree(eval_workspace / "logs" / "optimize", optimize_logs)
        self._write_agent_assets(eval_workspace)
        return eval_workspace

    def _write_agent_assets(self, eval_workspace: Path) -> None:
        write_file_tree(eval_workspace, self.immutable_files)
