"""Default immutable README renderer for Eve.

Canonical config lives under ``optimizer.immutable_renderer``. A single renderer
handles every immutable asset's runtime markers; the file is selected by the
workspace builder, not by the renderer. Example:

```yaml
optimizer:
  immutable: configs/eve/optimizer/circle_packing/immutable
  immutable_renderer:
    _target_: scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer
```

``DefaultRenderer`` fills the README template by literal marker replacement
(``str.replace``, not ``str.format``), so static prose may contain bare ``{`` /
``}`` (JSON, LaTeX, code) without breaking. The required markers are validated
up front; removing any of them is a hard error.
"""

from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem


class DefaultRenderer:
    """Render immutable README templates via literal marker replacement."""

    README_MARKERS: dict[str, str] = {
        "{editable_files_block}": "the editable file list for the current application",
        "{editable_folders_block}": "the editable folder list for the current application",
        "{solver_examples_block}": "runtime solver examples and their score cards",
        "{optimizer_examples_block}": "runtime optimizer examples and their score cards",
    }

    def render(
        self,
        template: str,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        optimizer: PopulationEntry | None = None,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry,
        optimizer_examples: Sequence[PopulationEntry] | None = None,
    ) -> str:
        """Render a copied immutable README using literal marker replacement."""

        self.validate(template)
        optimizer_examples = tuple(optimizer_examples or ())
        use_optimizer_examples = int(config.n_optimizer_examples_phase2) > 0
        solver_examples_dir = "solver_examples" if use_optimizer_examples else "examples"
        replacements = {
            "{editable_files_block}": self._render_path_list(
                heading="Editable files:",
                entries=tuple(problem.editable_files),
            ),
            "{editable_folders_block}": self._render_path_list(
                heading="Editable folders:",
                entries=tuple(problem.editable_folders),
                suffix="/",
            ),
            "{solver_examples_block}": self._render_solver_examples_block(
                solvers=solvers,
                prefill_solver=prefill_solver,
                solver_examples_dir=solver_examples_dir,
            ),
            "{optimizer_examples_block}": (
                self._render_optimizer_examples_block(
                    optimizer_examples=optimizer_examples,
                    prefill_optimizer=optimizer,
                )
                if use_optimizer_examples
                else ""
            ),
        }
        rendered = template
        for marker, value in replacements.items():
            rendered = rendered.replace(marker, value)
        return rendered.strip()

    def validate(self, template: str) -> None:
        """Raise if the README template is missing a required marker."""
        missing = [marker for marker in self.README_MARKERS if marker not in template]
        if not missing:
            return
        details = "\n".join(f"- {marker}: {self.README_MARKERS[marker]}" for marker in missing)
        raise ValueError(
            "immutable/README.md is missing required placeholder(s):\n"
            f"{details}\n"
            "EvE injects runtime data through these exact markers. Removing them prevents "
            "the Phase 2 workspace instructions from receiving the data the algorithm "
            "needs. Please keep the required placeholder strings in immutable/README.md."
        )

    @staticmethod
    def _render_path_list(*, heading: str, entries: tuple[str, ...], suffix: str = "") -> str:
        if not entries:
            return ""
        rendered = "\n".join(f"- `{path}{suffix}`" for path in entries)
        return f"{heading}\n{rendered}\n\n"

    @staticmethod
    def _render_solver_examples_block(
        *,
        solvers: Sequence[PopulationEntry],
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

    @staticmethod
    def _render_optimizer_examples_block(
        *,
        optimizer_examples: Sequence[PopulationEntry],
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
