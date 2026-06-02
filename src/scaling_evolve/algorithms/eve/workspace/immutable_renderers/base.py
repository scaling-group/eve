"""Base immutable-asset renderer for Eve.

A renderer owns the *dynamic* side of a Phase 2 workspace: it renders the
immutable README template and supplies the prompt-folder entrypoint instruction.
Both go through the same literal marker replacement (``str.replace``, not
``str.format``), so static prose may contain bare ``{`` / ``}`` (JSON, LaTeX,
code) without breaking. README rendering and entrypoint delivery stay separate,
while their source text is visible under ``optimizer.prompt``:

```yaml
optimizer:
  immutable: configs/eve/optimizer/circle_packing/immutable
  prompt: configs/eve/optimizer/circle_packing/prompt
  immutable_renderer:
    _target_: scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer
```

The required README markers are validated up front; removing any of them is a
hard error.
"""

from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem


class ImmutableRenderer:
    """Render immutable README templates and the agent entrypoint instruction."""

    README_MARKERS: dict[str, str] = {
        "{editable_files_block}": "the editable file list for the current application",
        "{editable_folders_block}": "the editable folder list for the current application",
        "{solver_examples_block}": "runtime solver examples and their score cards",
        "{optimizer_examples_block}": "runtime optimizer examples and their score cards",
    }

    def __init__(self, *, entrypoint: str | None = None) -> None:
        self._entrypoint_template = entrypoint

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
        return self._replace_markers(
            template,
            problem=problem,
            config=config,
            optimizer=optimizer,
            solvers=solvers,
            prefill_solver=prefill_solver,
            optimizer_examples=optimizer_examples,
        ).strip()

    def entrypoint(
        self,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        optimizer: PopulationEntry | None = None,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry,
        optimizer_examples: Sequence[PopulationEntry] | None = None,
    ) -> str:
        """Return the agent entrypoint instruction, with markers replaced."""

        if self._entrypoint_template is None:
            raise ValueError(
                "prompt/ENTRYPOINT.md is required before EvE can spawn a Phase 2 agent. "
                "Load it from optimizer.prompt and pass it to the immutable renderer."
            )
        return self._replace_markers(
            self._entrypoint_template,
            problem=problem,
            config=config,
            optimizer=optimizer,
            solvers=solvers,
            prefill_solver=prefill_solver,
            optimizer_examples=optimizer_examples,
        ).strip()

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

    def _replace_markers(
        self,
        text: str,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        optimizer: PopulationEntry | None,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry,
        optimizer_examples: Sequence[PopulationEntry] | None,
    ) -> str:
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
        rendered = text
        for marker, value in replacements.items():
            rendered = rendered.replace(marker, value)
        return rendered

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
