"""Worker-indexed instruction variants for Eve."""

from __future__ import annotations

from scaling_evolve.algorithms.eve.instructions.default import (
    Phase2ReadmeInstruction,
    _read_instruction_text,
)
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry


class IndexedPhase2ReadmeInstruction(Phase2ReadmeInstruction):
    """Select one Phase 2 README template by taking worker_index modulo file_lists."""

    def __init__(self, *, file_lists: list[list[str]]) -> None:
        if not file_lists:
            raise ValueError("IndexedPhase2ReadmeInstruction requires at least one file list.")
        self._contents = [
            "\n\n".join(
                part for part in (_read_instruction_text(path) for path in paths) if part
            ).strip()
            for paths in file_lists
        ]
        self.content = self._contents[0]

    def render(
        self,
        *,
        workspace_builder: object,
        optimizer: PopulationEntry | None = None,
        solvers: list[PopulationEntry],
        prefill_solver: PopulationEntry,
        optimizer_examples: list[PopulationEntry] | None = None,
        **_: object,
    ) -> str:
        worker_index = getattr(workspace_builder, "worker_index", None)
        template = (
            self._contents[0]
            if worker_index is None
            else self._contents[int(worker_index) % len(self._contents)]
        )

        problem = workspace_builder.problem
        optimizer_examples = optimizer_examples or []
        use_optimizer_examples = int(workspace_builder.config.n_optimizer_examples_phase2) > 0
        solver_examples_dir = "solver_examples" if use_optimizer_examples else "examples"
        return template.format(
            editable_files_block=self.render_path_list(
                heading="Editable files:",
                entries=tuple(problem.editable_files),
            ),
            editable_folders_block=self.render_path_list(
                heading="Editable folders:",
                entries=tuple(problem.editable_folders),
                suffix="/",
            ),
            solver_examples_block=self.render_solver_examples_block(
                solvers=solvers,
                prefill_solver=prefill_solver,
                solver_examples_dir=solver_examples_dir,
            ),
            solver_examples_dir=solver_examples_dir,
            optimizer_examples_block=(
                self.render_optimizer_examples_block(
                    optimizer_examples=optimizer_examples,
                    prefill_optimizer=optimizer,
                )
                if use_optimizer_examples
                else ""
            ),
        ).strip()
