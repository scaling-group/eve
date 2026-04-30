"""Hydra-instantiable default instruction objects for Eve."""

from __future__ import annotations

from pathlib import Path

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines


def _read_instruction_text(path: str) -> str:
    return Path(path).resolve().read_text(encoding="utf-8").strip()


class _BaseInstruction:
    def __init__(self, *, file_list: list[str]) -> None:
        parts = [_read_instruction_text(path) for path in file_list]
        self.content = "\n\n".join(part for part in parts if part).strip()

    def render(self, **_: object) -> str:
        return self.content.strip()

    @staticmethod
    def render_path_list(*, heading: str, entries: tuple[str, ...], suffix: str = "") -> str:
        if not entries:
            return ""
        rendered = "\n".join(f"- `{path}{suffix}`" for path in entries)
        return f"{heading}\n{rendered}\n\n"


class Phase2EntrypointInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


class Phase4EntrypointInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


class Phase2AgentInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


class Phase4AgentInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


class Phase2ReadmeInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)

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
        problem = workspace_builder.problem
        optimizer_examples = optimizer_examples or []
        use_optimizer_examples = int(workspace_builder.config.n_optimizer_examples_phase2) > 0
        solver_examples_dir = "solver_examples" if use_optimizer_examples else "examples"
        return self.content.format(
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

    def render_solver_examples_block(
        self,
        *,
        solvers: list[PopulationEntry],
        prefill_solver: PopulationEntry,
        solver_examples_dir: str,
    ) -> str:
        blocks = []
        for entry in solvers:
            marker = " <- prefill (copied to output/)" if entry.id == prefill_solver.id else ""
            blocks.append(f"- `{solver_examples_dir}/{entry.id}/`{marker}")
            blocks.extend(score_block_lines(entry.score, indent=2))
        blocks.append("")
        return "\n".join(blocks) + "\n"

    def render_optimizer_examples_block(
        self,
        *,
        optimizer_examples: list[PopulationEntry],
        prefill_optimizer: PopulationEntry | None,
    ) -> str:
        blocks = []
        for entry in optimizer_examples:
            marker = (
                " <- prefill (copied to guidance/)"
                if prefill_optimizer is not None and entry.id == prefill_optimizer.id
                else ""
            )
            blocks.append(f"- `guidance_examples/{entry.id}/`{marker}")
            blocks.extend(score_block_lines(entry.score, indent=2))
        blocks.append("")
        return "\n".join(blocks)


class Phase4ReadmeInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)

    def render(
        self,
        *,
        optimizers: list[PopulationEntry],
        prefill_optimizer: PopulationEntry,
        **_: object,
    ) -> str:
        reference_optimizers_block = (
            self.render_reference_optimizers_block(
                optimizers=optimizers,
                prefill_optimizer=prefill_optimizer,
            ),
        )
        return self.content.format(reference_optimizers_block=reference_optimizers_block).strip()

    def render_reference_optimizers_block(
        self,
        *,
        optimizers: list[PopulationEntry],
        prefill_optimizer: PopulationEntry,
    ) -> str:
        blocks = []
        for entry in optimizers:
            markers: list[str] = []
            if entry.id == prefill_optimizer.id:
                markers.append("prefill (copied to output/)")
            marker_text = f" <- {', '.join(markers)}" if markers else ""
            blocks.append(f"- `examples/{entry.id}/`{marker_text}")
            blocks.extend(score_block_lines(entry.score, indent=2))
        blocks.append("")
        return "\n".join(blocks) + "\n"
