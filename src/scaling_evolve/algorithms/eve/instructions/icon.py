"""Icon-specific instruction variants for Eve."""

from __future__ import annotations

from scaling_evolve.algorithms.eve.instructions import default
from scaling_evolve.algorithms.eve.instructions.default import _BaseInstruction
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines


class Phase2EntrypointInstruction(default.Phase2EntrypointInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


class Phase2ReadmeInstruction(default.Phase2ReadmeInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


class Phase4EntrypointInstruction(_BaseInstruction):
    def __init__(self, *, file_list: list[str]) -> None:
        super().__init__(file_list=file_list)


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
        reference_optimizers_block = self.render_reference_optimizers_block(
            optimizers=optimizers,
            prefill_optimizer=prefill_optimizer,
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
