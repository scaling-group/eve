"""Base immutable-asset renderer for Eve.

A renderer owns the *dynamic* side of a Phase 2 workspace: it renders the
immutable README template and supplies the prompt-folder entrypoint instruction.
Both go through the same literal marker replacement (``str.replace``, not
``str.format``), so static prose may contain bare ``{`` / ``}`` (JSON, LaTeX,
code) without breaking. README rendering and entrypoint delivery stay separate,
while their source text is visible under ``optimizer.workers.items[].prompt``:

```yaml
optimizer:
  workers:
    items:
      - name: normal
        weight: 1.0
        immutable: configs/eve/optimizer/circle_packing/immutable
        prompt: configs/eve/optimizer/circle_packing/prompt
```

The required README markers are validated up front; removing any of them is a
hard error.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem


class ImmutableRenderer:
    """Render immutable templates and the agent entrypoint instruction."""

    def __init__(
        self,
        *,
        entrypoint: str | None = None,
    ) -> None:
        self._entrypoint_template = entrypoint

    def render(
        self,
        template: str,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        immutable_files: dict[str, str] | None = None,
        optimizer: PopulationEntry | None = None,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry | None,
        optimizer_examples: Sequence[PopulationEntry] | None = None,
    ) -> str:
        """Render copied immutable text using literal marker replacement."""

        return self._replace_markers(
            template,
            problem=problem,
            config=config,
            immutable_files=immutable_files,
            optimizer=optimizer,
            solvers=solvers,
            prefill_solver=prefill_solver,
            optimizer_examples=optimizer_examples,
        )

    def entrypoint(
        self,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        optimizer: PopulationEntry | None = None,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry | None,
        optimizer_examples: Sequence[PopulationEntry] | None = None,
    ) -> str:
        """Return the agent entrypoint instruction, with markers replaced."""

        if self._entrypoint_template is None:
            raise ValueError(
                "prompt/ENTRYPOINT.md is required before EvE can spawn a Phase 2 agent. "
                "Load it from optimizer.workers.items[].prompt and pass it to the immutable "
                "renderer."
            )
        return self._replace_markers(
            self._entrypoint_template,
            problem=problem,
            config=config,
            immutable_files=None,
            optimizer=optimizer,
            solvers=solvers,
            prefill_solver=prefill_solver,
            optimizer_examples=optimizer_examples,
        ).strip()

    def _replace_markers(
        self,
        text: str,
        *,
        problem: RepoTaskProblem,
        config: DictConfig,
        immutable_files: dict[str, str] | None,
        optimizer: PopulationEntry | None,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry | None,
        optimizer_examples: Sequence[PopulationEntry] | None,
    ) -> str:
        optimizer_examples = tuple(optimizer_examples or ())
        use_optimizer_examples = int(config.n_optimizer_examples_phase2) > 0
        solver_examples_dir = "solver_examples"
        replacements = {
            "{editable_files_block}": self._render_path_list(
                heading="Editable files:",
                entries=self._solver_workspace_paths(tuple(problem.editable_files)),
            ),
            "{editable_folders_block}": self._render_path_list(
                heading="Editable folders:",
                entries=self._solver_workspace_paths(tuple(problem.editable_folders)),
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
            "{immutable_overlay_block}": self._render_immutable_overlay_block(
                immutable_files or {}
            ),
        }
        rendered = text
        for marker, value in replacements.items():
            rendered = rendered.replace(marker, value)
        return rendered

    @staticmethod
    def _render_immutable_overlay_block(immutable_files: dict[str, str]) -> str:
        overlays: list[tuple[str, str]] = []
        prefixes = (
            (".codex/skills/", "guidance/skills/"),
            (".claude/skills/", "guidance/skills/"),
            (".codex/agents/", "guidance/agents/codex/"),
            (".claude/agents/", "guidance/agents/claude/"),
        )
        for path in sorted(immutable_files):
            for source_prefix, guidance_prefix in prefixes:
                if path.startswith(source_prefix):
                    overlays.append((path, guidance_prefix + path.removeprefix(source_prefix)))
                    break
        if not overlays:
            return "No immutable overlay files target guidance-exposed directories.\n"
        lines = []
        lines.extend(f"- `{source}` overlays `{target}`" for source, target in overlays)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_path_list(*, heading: str, entries: tuple[str, ...], suffix: str = "") -> str:
        if not entries:
            return ""
        rendered = "\n".join(f"- `{path}{suffix}`" for path in entries)
        return f"{heading}\n{rendered}\n\n"

    @staticmethod
    def _solver_workspace_paths(entries: tuple[str, ...]) -> tuple[str, ...]:
        """Render editable paths relative to the phase workspace root."""
        return tuple(str(Path("solver") / entry) for entry in entries)

    @staticmethod
    def _render_solver_examples_block(
        *,
        solvers: Sequence[PopulationEntry],
        prefill_solver: PopulationEntry | None,
        solver_examples_dir: str,
    ) -> str:
        blocks = []
        for entry in solvers:
            marker = (
                " <- prefill"
                if prefill_solver is not None and entry.id == prefill_solver.id
                else ""
            )
            blocks.append(f"- `{solver_examples_dir}/{entry.id}/`{marker}")
            blocks.append("  prior score:")
            blocks.extend(score_block_lines(entry.score, indent=4))
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


def render_immutable_files(
    *,
    renderer: ImmutableRenderer,
    immutable_files: dict[str, str],
    problem: RepoTaskProblem,
    config: DictConfig,
    optimizer: PopulationEntry | None = None,
    solvers: Sequence[PopulationEntry],
    prefill_solver: PopulationEntry | None,
    optimizer_examples: Sequence[PopulationEntry] | None = None,
) -> dict[str, str]:
    """Render every immutable file with the same runtime marker context."""
    return {
        rel_path: renderer.render(
            content,
            problem=problem,
            config=config,
            immutable_files=immutable_files,
            optimizer=optimizer,
            solvers=solvers,
            prefill_solver=prefill_solver,
            optimizer_examples=optimizer_examples,
        )
        for rel_path, content in immutable_files.items()
    }
