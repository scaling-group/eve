"""Immutable workspace asset rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem

README_MARKERS: dict[str, str] = {
    "{editable_files_block}": "the editable file list for the current application",
    "{editable_folders_block}": "the editable folder list for the current application",
    "{solver_examples_block}": "runtime solver examples and their score cards",
    "{optimizer_examples_block}": "runtime optimizer examples and their score cards",
}


def render_readme_template(
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

    validate_readme_template(template)
    optimizer_examples = tuple(optimizer_examples or ())
    use_optimizer_examples = int(config.n_optimizer_examples_phase2) > 0
    solver_examples_dir = "solver_examples" if use_optimizer_examples else "examples"
    replacements = {
        "{editable_files_block}": render_path_list(
            heading="Editable files:",
            entries=tuple(problem.editable_files),
        ),
        "{editable_folders_block}": render_path_list(
            heading="Editable folders:",
            entries=tuple(problem.editable_folders),
            suffix="/",
        ),
        "{solver_examples_block}": render_solver_examples_block(
            solvers=solvers,
            prefill_solver=prefill_solver,
            solver_examples_dir=solver_examples_dir,
        ),
        "{optimizer_examples_block}": (
            render_optimizer_examples_block(
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


def render_path_list(*, heading: str, entries: tuple[str, ...], suffix: str = "") -> str:
    if not entries:
        return ""
    rendered = "\n".join(f"- `{path}{suffix}`" for path in entries)
    return f"{heading}\n{rendered}\n\n"


def render_solver_examples_block(
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


def render_optimizer_examples_block(
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


def validate_readme_template(template: str) -> None:
    missing = [marker for marker in README_MARKERS if marker not in template]
    if not missing:
        return
    details = "\n".join(f"- {marker}: {README_MARKERS[marker]}" for marker in missing)
    raise ValueError(
        "immutable/README.md is missing required placeholder(s):\n"
        f"{details}\n"
        "EvE injects runtime data through these exact markers. Removing them prevents "
        "the Phase 2 workspace instructions from receiving the data the algorithm "
        "needs. Please keep the required placeholder strings in immutable/README.md."
    )
