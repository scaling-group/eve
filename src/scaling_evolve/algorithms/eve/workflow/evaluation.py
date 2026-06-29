"""Agent-backed evaluation helpers for Eve solver candidates."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.prompt_assets import read_required_prompt_text
from scaling_evolve.algorithms.eve.rollout_prompts.default import BudgetPrompt, PromptContext
from scaling_evolve.algorithms.eve.workflow.optimize_logs import build_usage_report
from scaling_evolve.algorithms.eve.workspace.file_tree import (
    read_file_tree,
    write_file_tree,
)
from scaling_evolve.algorithms.eve.workspace.immutable_renderers.base import (
    ImmutableRenderer,
    render_immutable_files,
)
from scaling_evolve.algorithms.eve.workspace.immutable_renderers.static import (
    StaticRenderer,
)
from scaling_evolve.algorithms.eve.workspace.runtime_hooks import install_workspace_runtime_hooks
from scaling_evolve.providers.agent.drivers.base import SessionSeed
from scaling_evolve.providers.agent.evaluation import (
    load_completion_summary_verdict,
    run_agent_for_result,
)
from scaling_evolve.providers.agent.runtime_env import prepend_agent_runtime_bins
from scaling_evolve.providers.agent.session_log import build_session_log_markdown

_LOGGER = logging.getLogger(__name__)
_EVAL_LOG_ROOT = Path("logs") / "evaluate"
_EVAL_SCORE_YAML_PATH = _EVAL_LOG_ROOT / "score.yaml"
_EVAL_STEP_LOG_ROOT = _EVAL_LOG_ROOT / "steps"


@dataclass(frozen=True)
class EvaluationStep:
    """One ordered evaluation step.

    A step is a *judge* step iff it carries an `entrypoint` (a judge always has a
    `prompt`); otherwise it is a programmatic *shell* step (`path` points at a `.sh`
    file). A judge's `immutable` is OPTIONAL: it may be null/empty, meaning no scaffold
    is landed and the judge runs from its ENTRYPOINT alone. There is no `kind` field —
    dispatch is by form.
    """

    name: str
    path: Path | None = None
    immutable_files: dict[str, str] | None = None
    immutable_renderer: ImmutableRenderer | None = None
    entrypoint: str | None = None
    rollout_prompts: dict[str, object] | None = None

    @property
    def is_judge(self) -> bool:
        return self.entrypoint is not None

    @property
    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class EvaluationPlan:
    """Loaded evaluation config for a solver candidate."""

    steps: tuple[EvaluationStep, ...]


@dataclass(frozen=True)
class SolverEvaluator:
    """Owns solver-evaluation policy and runtime execution."""

    problem: RepoTaskProblem
    evaluation_plan: EvaluationPlan
    evaluation_driver_factory: Callable[[], object]
    evaluation_failure_score: PyTree
    boundary_failure_score: PyTree
    seed_solver_score: PyTree | None
    seed_solver_skip_evaluation: bool
    include_solver_examples: bool = False

    def __call__(
        self,
        workspace_root: Path,
        *,
        candidate_files: dict[str, str] | None = None,
        optimize_logs: dict[str, str] | None = None,
        solver_examples: list[PopulationEntry] | None = None,
        prefill_solver: PopulationEntry | None = None,
        display_context: dict[str, str | int] | None = None,
    ) -> tuple[PyTree, dict[str, str]]:
        return self.evaluate(
            workspace_root,
            candidate_files=candidate_files,
            optimize_logs=optimize_logs,
            solver_examples=solver_examples,
            prefill_solver=prefill_solver,
            display_context=display_context,
        )

    def evaluate(
        self,
        workspace_root: Path,
        *,
        candidate_files: dict[str, str] | None = None,
        optimize_logs: dict[str, str] | None = None,
        solver_examples: list[PopulationEntry] | None = None,
        prefill_solver: PopulationEntry | None = None,
        display_context: dict[str, str | int] | None = None,
    ) -> tuple[PyTree, dict[str, str]]:
        eval_ws = _judge_workspace_for_solver_workspace(workspace_root)
        log_root = eval_ws / _EVAL_LOG_ROOT
        _LOGGER.info("Evaluation: starting workspace `%s`", workspace_root.name)
        try:
            if candidate_files is None:
                candidate_files = read_file_tree(workspace_root / "solver")
            if optimize_logs is None:
                optimize_logs = read_file_tree(workspace_root / "logs" / "optimize")
            solver_examples = list(solver_examples or [])
            score = _run_evaluation_steps(
                plan=self.evaluation_plan,
                workspace_root=workspace_root,
                candidate_files=candidate_files,
                optimize_logs=optimize_logs,
                snapshot_root=self.problem.snapshot_root,
                problem=self.problem,
                solver_examples=solver_examples,
                prefill_solver=prefill_solver,
                include_solver_examples=self.include_solver_examples,
                driver_factory=self.evaluation_driver_factory,
                display_context=dict(display_context or {}),
            )
            _validate_score_shape(score, self.evaluation_failure_score)

            logs = read_file_tree(log_root)
            rendered_score = "\n".join(score_block_lines(score, indent=2))
            _LOGGER.info(
                "Evaluation: parsed score for workspace `%s`\n%s",
                workspace_root.name,
                rendered_score,
            )
            return score, logs
        except Exception:
            return (
                self.evaluation_failure_score,
                (read_file_tree(log_root) if log_root.exists() else {}),
            )

    def evaluate_candidate(
        self,
        workspace_root: Path,
        *,
        boundary_result: object,
        candidate_files: dict[str, str] | None = None,
        optimize_logs: dict[str, str] | None = None,
        solver_examples: list[PopulationEntry] | None = None,
        prefill_solver: PopulationEntry | None = None,
        display_context: dict[str, str | int] | None = None,
    ) -> tuple[PyTree, dict[str, str]]:
        if not boundary_result.ok:
            return (
                self.boundary_failure_score,
                {"boundary/forbidden_changes.txt": boundary_result.summary() + "\n"},
            )
        return self.evaluate(
            workspace_root,
            candidate_files=candidate_files,
            optimize_logs=optimize_logs,
            solver_examples=solver_examples,
            prefill_solver=prefill_solver,
            display_context=display_context,
        )

    @staticmethod
    def check_boundary(
        workspace_root: Path,
        *,
        solver_workspace_builder: object,
    ) -> object:
        return solver_workspace_builder.boundary_check_result(workspace_root)

    def build_seed_entry(
        self,
        *,
        workspace: Path,
        seed_solver_id: str,
        solver_workspace_builder: object,
    ) -> tuple[PopulationEntry, bool]:
        if self.seed_solver_skip_evaluation:
            if self.seed_solver_score is None:
                raise ValueError(
                    "evaluation.seed_solver_score is required when "
                    "evaluation.seed_solver_skip_evaluation "
                    "is true."
                )
            return (
                self._seed_entry_from_fixed_score(
                    workspace=workspace,
                    seed_solver_id=seed_solver_id,
                    score=self.seed_solver_score,
                    summary="Seed evaluation skipped; used evaluation.seed_solver_score.",
                    solver_workspace_builder=solver_workspace_builder,
                ),
                False,
            )

        candidate_files = solver_workspace_builder.extract(workspace)
        score, evaluate_logs = self.evaluate(
            workspace,
            candidate_files=candidate_files,
            optimize_logs={},
            display_context={"iteration": 0, "worker_index": 0},
        )
        solver_logs = solver_workspace_builder.write_run_logs(
            workspace,
            optimize_logs={},
            evaluate_logs=evaluate_logs,
        )
        seed_entry = PopulationEntry(
            id=seed_solver_id,
            files=candidate_files,
            score=score,
            logs=solver_logs,
        )
        return seed_entry, True

    @staticmethod
    def _seed_entry_from_fixed_score(
        *,
        workspace: Path,
        seed_solver_id: str,
        score: PyTree,
        summary: str,
        solver_workspace_builder: object,
    ) -> PopulationEntry:
        candidate_files = solver_workspace_builder.extract(workspace)
        evaluate_logs = {
            "score.yaml": yaml.safe_dump(score, sort_keys=False),
            "summary.txt": f"{summary}\n",
        }
        solver_logs = solver_workspace_builder.write_run_logs(
            workspace,
            optimize_logs={},
            evaluate_logs=evaluate_logs,
        )
        seed_entry = PopulationEntry(
            id=seed_solver_id,
            files=candidate_files,
            score=score,
            logs=solver_logs,
        )
        return seed_entry


def _step_log_dir(*, step_name: str, step_index: int, workspace_root: Path) -> Path:
    return workspace_root / _EVAL_STEP_LOG_ROOT / f"step_{step_index:02d}_{step_name}"


def _run_shell_step(*, step: Path, step_index: int, workspace_root: Path) -> None:
    step_log_dir = _step_log_dir(
        step_name=step.stem,
        step_index=step_index,
        workspace_root=workspace_root,
    )
    step_log_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(
        "Evaluation: starting shell step %02d `%s` for workspace `%s`; "
        "waiting for completion, logs under %s",
        step_index,
        step.name,
        workspace_root.name,
        step_log_dir,
    )
    env = prepend_agent_runtime_bins(
        {
            **os.environ,
            "EVE_WORKSPACE_ROOT": str(workspace_root),
            "EVE_SOLVER_ROOT": str(workspace_root / "solver"),
            "EVE_EVAL_LOG_ROOT": str(workspace_root / _EVAL_LOG_ROOT),
            # solver/ is frozen read-only (it is the candidate being scored; the .sh measures it,
            # never modifies it). Stop Python from trying to write __pycache__ there when the
            # task's evaluator imports the candidate.
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        workspace_root=workspace_root,
    )
    started_at = time.monotonic()
    completed = subprocess.run(
        ["bash", str(step)],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.monotonic() - started_at
    if completed.stdout:
        (step_log_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (step_log_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode == 0:
        (step_log_dir / "status.txt").write_text("ok\n", encoding="utf-8")
        _LOGGER.info(
            "Evaluation: shell step %02d `%s` finished successfully in %.1fs",
            step_index,
            step.name,
            elapsed,
        )
        return
    (step_log_dir / "status.txt").write_text("failed\n", encoding="utf-8")
    _LOGGER.warning(
        "Evaluation: shell step %02d `%s` failed with return code %d after %.1fs",
        step_index,
        step.name,
        completed.returncode,
        elapsed,
    )
    raise RuntimeError(
        f"Evaluation shell step failed: {step.name}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _judge_rollout_logs(rollout) -> dict[str, str]:
    """Render a judge agent's rollout into the same artifacts the solver gets in `logs/optimize/`
    (final response, token usage, full session transcript), so a judge's reasoning is auditable
    from the persisted entry — not only from the transient driver transcripts. Mirrors
    `build_optimize_log_tree` for a single judge run."""
    logs: dict[str, str] = {}
    if rollout is None:
        return logs
    final_response = getattr(rollout, "summary", None)
    if isinstance(final_response, str) and final_response.strip():
        logs["final_response.txt"] = final_response.rstrip() + "\n"
    logs["token_usage.json"] = (
        json.dumps(build_usage_report([rollout]), indent=2, sort_keys=True) + "\n"
    )
    try:
        session_markdown = build_session_log_markdown([rollout])
    except Exception:
        session_markdown = None  # a missing/unparseable transcript must not fail evaluation
    if session_markdown is not None:
        logs["session.md"] = session_markdown
    return logs


def _run_judge_step(
    *,
    step: EvaluationStep,
    step_index: int,
    eval_ws: Path,
    previously_landed: set[str],
    problem: RepoTaskProblem,
    solver_examples: list[PopulationEntry],
    prefill_solver: PopulationEntry | None,
    driver_factory,
    display_context: dict[str, str | int],
) -> set[str]:
    step_log_dir = _step_log_dir(
        step_name=step.display_name,
        step_index=step_index,
        workspace_root=eval_ws,
    )
    step_log_dir.mkdir(parents=True, exist_ok=True)
    landed = _land_judge_assets(
        step=step,
        eval_ws=eval_ws,
        previously_landed=previously_landed,
        problem=problem,
        solver_examples=solver_examples,
        prefill_solver=prefill_solver,
    )
    score_path = eval_ws / _EVAL_SCORE_YAML_PATH
    score_before_text = score_path.read_text(encoding="utf-8") if score_path.exists() else None
    _LOGGER.info(
        "Evaluation: starting judge step %02d `%s` in `%s`",
        step_index,
        step.display_name,
        eval_ws,
    )
    driver = driver_factory()
    prompt_specs = _build_judge_prompt_specs(step=step, driver=driver, eval_ws=eval_ws)
    install_workspace_runtime_hooks(eval_ws, driver=driver, prompt_specs=prompt_specs)
    run = run_agent_for_result(
        driver=driver,
        seed=SessionSeed(
            instruction=_require_judge_entrypoint(step),
            working_directory=str(eval_ws),
            prompt_file="README.md" if (eval_ws / "README.md").is_file() else None,
            write_prompt_file=False,
            display_context=display_context,
        ),
        load_result_text=lambda _rollout: "",
    )
    if run.error is not None:
        (step_log_dir / "status.txt").write_text("failed\n", encoding="utf-8")
        (step_log_dir / "error.txt").write_text(f"{run.error}\n", encoding="utf-8")
        _LOGGER.warning("Evaluation: judge step %02d failed: %s", step_index, run.error)
        raise RuntimeError(f"Evaluation judge step failed: {run.error}")
    try:
        score = _capture_judge_score(
            judge_workspace=eval_ws,
            rollout=run.rollout,
            score_before_text=score_before_text,
        )
    except Exception:
        (step_log_dir / "status.txt").write_text("failed\n", encoding="utf-8")
        raise
    _write_score_yaml(step_log_dir / "score.yaml", score)
    write_file_tree(step_log_dir, _judge_rollout_logs(run.rollout))
    (step_log_dir / "status.txt").write_text("ok\n", encoding="utf-8")
    _LOGGER.info("Evaluation: judge step %02d finished successfully", step_index)
    return landed


def _land_judge_assets(
    *,
    step: EvaluationStep,
    eval_ws: Path,
    previously_landed: set[str],
    problem: RepoTaskProblem,
    solver_examples: list[PopulationEntry],
    prefill_solver: PopulationEntry | None,
) -> set[str]:
    """Clean-swap THIS judge's immutable assets into the (built) eval workspace.

    Each judge step REPLACES the prior judge's scaffold: the previously-landed files are
    removed first, then this step's immutable is written. A clean swap (not an overlay) means
    one judge does NOT inherit stale files from an earlier judge. The episode (`solver/`,
    `logs/optimize/`) and accumulated `logs/evaluate/` are never touched, so the accumulated
    score persists across judge steps.

    Returns the set of relative paths this step landed (removed before the next judge step).
    """
    for rel_path in previously_landed:
        target = eval_ws / rel_path
        if target.is_file() or target.is_symlink():
            target.unlink()

    renderer = step.immutable_renderer or StaticRenderer()
    immutable_files = dict(step.immutable_files or {})
    eval_renderer_config = OmegaConf.create({"n_optimizer_examples_phase2": 0})
    rendered_files = render_immutable_files(
        renderer=renderer,
        immutable_files=immutable_files,
        problem=problem,
        config=eval_renderer_config,
        optimizer=None,
        solvers=solver_examples,
        prefill_solver=prefill_solver,
        optimizer_examples=(),
    )
    write_file_tree(eval_ws, rendered_files)
    return set(rendered_files)


def _build_judge_prompt_specs(
    *,
    step: EvaluationStep,
    driver: object,
    eval_ws: Path,
) -> list[dict[str, object]]:
    """Build the budget `prompt_specs` for a judge step, mirroring phase2's solver path."""
    rollout_prompts = step.rollout_prompts or {}
    if not rollout_prompts or not _driver_budget_prompt_enabled(driver):
        return []
    ctx = PromptContext(
        workspace=eval_ws,
        rollout_max_turns=_driver_rollout_max_turns(driver),
    )
    prompt_specs: list[dict[str, object]] = []
    for name, prompt in rollout_prompts.items():
        prompt_specs.append(
            {
                "name": name,
                "system_text": prompt.system(ctx),
                "user_text": prompt.user(ctx),
                "turn_template": _turn_template_source(prompt),
                "turn_format_kwargs": _turn_format_kwargs(prompt, ctx),
            }
        )
    return prompt_specs


def _driver_rollout_max_turns(driver: object) -> int | None:
    value = getattr(driver, "rollout_max_turns", None)
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _driver_budget_prompt_enabled(driver: object) -> bool:
    value = getattr(driver, "budget_prompt", True)
    return value if isinstance(value, bool) else True


def _turn_template_source(prompt: object) -> str | None:
    turn_template_source = getattr(prompt, "turn_template_source", None)
    if callable(turn_template_source):
        value = turn_template_source()
        return value if isinstance(value, str) and value else None
    value = getattr(prompt, "_turn_template", None)
    return value if isinstance(value, str) and value else None


def _turn_format_kwargs(prompt: object, ctx: PromptContext) -> dict[str, object]:
    turn_format_kwargs = getattr(prompt, "turn_format_kwargs", None)
    if not callable(turn_format_kwargs):
        return {}
    value = turn_format_kwargs(ctx)
    return value if isinstance(value, dict) else {}


def _capture_judge_score(
    *, judge_workspace: Path, rollout, score_before_text: str | None
) -> PyTree:
    """Capture a judge's contribution into the accumulating score.yaml.

    Prefer the file the judge wrote IN PLACE (its full updated score). If the judge did NOT
    change score.yaml (e.g. it only emitted a fenced verdict in its final message — common with
    tight budgets), recover that verdict and merge it onto the prior accumulated score, so the
    judge's dimensions are not silently dropped and earlier dimensions are kept.
    """
    score_path = judge_workspace / _EVAL_SCORE_YAML_PATH
    current_text = score_path.read_text(encoding="utf-8") if score_path.exists() else None

    # The judge wrote (or created) score.yaml in place -> that is its full updated score.
    if current_text is not None and current_text != score_before_text:
        return _load_evaluation_score(judge_workspace)

    # The judge did not update the file. Recover its verdict from the completion, if any.
    if rollout is not None:
        verdict = None
        try:
            verdict = load_completion_summary_verdict(rollout)
        except Exception:
            verdict = None
        if isinstance(verdict, dict):
            prior = yaml.safe_load(score_before_text) if score_before_text else {}
            if not isinstance(prior, dict):
                prior = {}
            merged = {**prior, **verdict}
            _write_score_yaml(score_path, merged)
            return merged

    # No file change and no usable verdict: keep whatever is already on disk, else error.
    if current_text is not None:
        return _load_evaluation_score(judge_workspace)
    raise ValueError(
        "Evaluation judge did not write logs/evaluate/score.yaml and returned no parseable "
        "completion verdict."
    )


def _run_evaluation_steps(
    *,
    plan: EvaluationPlan,
    workspace_root: Path,
    candidate_files: dict[str, str] | None,
    optimize_logs: dict[str, str] | None,
    snapshot_root: Path,
    problem: RepoTaskProblem,
    solver_examples: list[PopulationEntry],
    prefill_solver: PopulationEntry | None,
    include_solver_examples: bool,
    driver_factory,
    display_context: dict[str, str | int],
) -> PyTree:
    # Build ONE eval workspace for every evaluation plan, including sh-only plans.
    # The eval workspace is the canonical scoring environment; the solver
    # workspace remains the Phase 2 agent episode and is not mutated by eval.
    eval_ws = _build_eval_workspace(
        workspace_root,
        candidate_files=candidate_files or {},
        optimize_logs=optimize_logs or {},
        snapshot_root=snapshot_root,
        solver_examples=solver_examples,
        include_solver_examples=include_solver_examples,
    )
    landed_assets: set[str] = set()
    for step_index, step in enumerate(plan.steps, start=1):
        if step.is_judge:
            landed_assets = _run_judge_step(
                step=step,
                step_index=step_index,
                eval_ws=eval_ws,
                previously_landed=landed_assets,
                problem=problem,
                solver_examples=solver_examples,
                prefill_solver=prefill_solver,
                driver_factory=driver_factory,
                display_context=display_context,
            )
            continue
        if step.path is None:
            raise ValueError(f"Evaluation shell step requires a path: {step}")
        if step.path.suffix != ".sh":
            raise ValueError(
                f"Unsupported evaluation step type for {step.path}. "
                "Shell evaluation steps must be `.sh` files."
            )
        _run_shell_step(step=step.path, step_index=step_index, workspace_root=eval_ws)
    return _load_evaluation_score(eval_ws)


def _build_eval_workspace(
    workspace_root: Path,
    *,
    candidate_files: dict[str, str],
    optimize_logs: dict[str, str],
    snapshot_root: Path,
    solver_examples: list[PopulationEntry] | None = None,
    include_solver_examples: bool = False,
) -> Path:
    """Build the eval workspace ONCE from the canonical episode representation.

    `solver/` is reconstructed from `snapshot_root` (full harness, e.g. evaluate.py)
    overlaid with the candidate edits — NOT copied from the solver workspace (decoupling,
    design §5). There is intentionally no `guidance/` (design §4).
    """
    eval_ws = _judge_workspace_for_solver_workspace(workspace_root)
    _remove_tree(eval_ws)
    eval_ws.mkdir(parents=True, exist_ok=True)

    solver_root = eval_ws / "solver"
    shutil.copytree(snapshot_root, solver_root)
    write_file_tree(solver_root, candidate_files)

    optimize_root = eval_ws / "logs" / "optimize"
    write_file_tree(optimize_root, optimize_logs)

    if include_solver_examples:
        examples_root = eval_ws / "solver_examples"
        _write_solver_examples(examples_root, solver_examples or [])

    _make_tree_read_only(solver_root)
    _make_tree_read_only(optimize_root)
    if include_solver_examples:
        _make_tree_read_only(eval_ws / "solver_examples")

    (eval_ws / _EVAL_LOG_ROOT).mkdir(parents=True, exist_ok=True)
    return eval_ws


def _write_solver_examples(examples_root: Path, solver_examples: list[PopulationEntry]) -> None:
    examples_root.mkdir(exist_ok=True)
    for entry in solver_examples:
        example_dir = examples_root / entry.id
        example_dir.mkdir(exist_ok=True)
        write_file_tree(example_dir / "solver", entry.files)
        write_file_tree(
            example_dir / "logs",
            {path: content for path, content in entry.logs.items() if path.startswith("evaluate/")},
        )
        _write_score_yaml(
            example_dir / "score.yaml",
            {
                "example_id": entry.id,
                "score": entry.score,
            },
        )


def _judge_workspace_for_solver_workspace(workspace_root: Path) -> Path:
    if workspace_root.parent.name == "solver_workspaces":
        return workspace_root.parent.parent / "evaluation_workspaces" / workspace_root.name
    return workspace_root.parent / "evaluation_workspaces" / workspace_root.name


def _require_judge_entrypoint(step: EvaluationStep) -> str:
    if step.entrypoint is None:
        raise ValueError("evaluation judge step prompt/ENTRYPOINT.md is required.")
    return step.entrypoint


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    _make_tree_writable(path)
    shutil.rmtree(path)


def _make_tree_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | 0o700)
        except FileNotFoundError:
            continue
    try:
        root.chmod(root.stat().st_mode | 0o700)
    except FileNotFoundError:
        return


def _make_tree_read_only(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode & ~0o222)
    if not root.is_symlink():
        root.chmod(root.stat().st_mode & ~0o222)


def build_solver_evaluator(
    problem: RepoTaskProblem,
    *,
    evaluation_plan: EvaluationPlan,
    evaluation_failure_score: PyTree,
    boundary_failure_score: PyTree,
    seed_solver_score: PyTree | None,
    seed_solver_skip_evaluation: bool,
    include_solver_examples: bool = False,
    evaluation_driver_factory,
) -> SolverEvaluator:
    return SolverEvaluator(
        problem=problem,
        evaluation_plan=evaluation_plan,
        evaluation_driver_factory=evaluation_driver_factory,
        evaluation_failure_score=evaluation_failure_score,
        boundary_failure_score=boundary_failure_score,
        seed_solver_score=seed_solver_score,
        seed_solver_skip_evaluation=seed_solver_skip_evaluation,
        include_solver_examples=include_solver_examples,
    )


def build_evaluation_plan(
    problem: RepoTaskProblem,
    *,
    evaluation_config: DictConfig | None,
    search_root: Path,
) -> EvaluationPlan:
    """Build ordered evaluation steps from `config.evaluation`, when present.

    Dispatch is by form (no `kind`): a bare `.sh` path (or a `{path: ...}` mapping) is a
    shell step; a mapping carrying `immutable` AND `prompt` is a judge step.
    """

    if evaluation_config is None:
        raise ValueError(
            "No `evaluation:` config group is composed for this run. Add an `evaluation:` group "
            "to the run config's `defaults:` list (e.g. `- evaluation: <app>` for a "
            "single-step programmatic evaluator, or `- evaluation: <app>.judge` for a judge "
            "pipeline). See configs/eve/circle_packing.yaml for the default form and "
            "configs/eve/evaluation/circle_packing.judge.yaml for the judge form."
        )

    raw_steps = OmegaConf.select(evaluation_config, "steps")
    if raw_steps is None or len(raw_steps) == 0:
        raise SystemExit("evaluation.steps must contain at least one step.")

    shared_renderer = _evaluation_immutable_renderer(evaluation_config)

    steps: list[EvaluationStep] = []
    seen_judge_names: set[str] = set()
    for index, raw_step in enumerate(raw_steps):
        step = _parse_evaluation_step(
            index=index,
            raw_step=raw_step,
            search_root=search_root,
            shared_renderer=shared_renderer,
            seen_judge_names=seen_judge_names,
        )
        steps.append(step)

    return EvaluationPlan(steps=tuple(steps))


def _evaluation_immutable_renderer(evaluation_config: DictConfig) -> ImmutableRenderer:
    renderer_config = OmegaConf.select(evaluation_config, "immutable_renderer")
    if renderer_config is None:
        return StaticRenderer()
    return instantiate(renderer_config, _convert_="all")


def _parse_evaluation_step(
    *,
    index: int,
    raw_step: object,
    search_root: Path,
    shared_renderer: ImmutableRenderer,
    seen_judge_names: set[str],
) -> EvaluationStep:
    # Form 1: bare string path (canonical shell form).
    if isinstance(raw_step, str):
        return _shell_step_from_path(search_root, raw_step)

    # Mappings: judge (immutable+prompt) or shell ({path: ...}).
    immutable_value = OmegaConf.select(raw_step, "immutable")
    prompt_value = OmegaConf.select(raw_step, "prompt")
    path_value = OmegaConf.select(raw_step, "path")

    if prompt_value is not None:
        return _judge_step(
            index=index,
            raw_step=raw_step,
            immutable_value=immutable_value,
            prompt_value=prompt_value,
            search_root=search_root,
            shared_renderer=shared_renderer,
            seen_judge_names=seen_judge_names,
        )

    if path_value is not None and immutable_value is None:
        return _shell_step_from_path(search_root, path_value)

    raise SystemExit(
        f"evaluation.steps[{index}] is not a valid step: expected a bare `.sh` path string, "
        "a `{path: ...}` shell mapping, or a `{name, prompt[, immutable]}` judge mapping "
        "(a judge's `immutable` is optional; `prompt` is required)."
    )


def _shell_step_from_path(search_root: Path, raw_path: object) -> EvaluationStep:
    path = _resolve_config_path(search_root, raw_path)
    if path.suffix != ".sh":
        raise SystemExit(f"Evaluation shell step must be a `.sh` file: {path}")
    return EvaluationStep(name=path.stem, path=path)


def _judge_step(
    *,
    index: int,
    raw_step: object,
    immutable_value: object,
    prompt_value: object,
    search_root: Path,
    shared_renderer: ImmutableRenderer,
    seen_judge_names: set[str],
) -> EvaluationStep:
    prompt_root = _resolve_required_dir(
        search_root, prompt_value, label=f"evaluation.steps[{index}] judge prompt directory"
    )
    # The judge name is DERIVED from its prompt dir (e.g. `prompt_assess` -> "assess"), so the
    # config needs no redundant `name:` field; an explicit `name` is still honored if given.
    configured_name = OmegaConf.select(raw_step, "name")
    name_value = (
        str(configured_name)
        if configured_name
        else (prompt_root.name.removeprefix("prompt_") or prompt_root.name)
    )
    if name_value in seen_judge_names:
        raise SystemExit(f"evaluation.steps[{index}] duplicates judge name `{name_value}`.")
    seen_judge_names.add(name_value)
    entrypoint_path = prompt_root / "ENTRYPOINT.md"
    if not entrypoint_path.is_file():
        raise SystemExit(f"evaluation judge `{name_value}` prompt ENTRYPOINT.md not found.")

    immutable_files: dict[str, str] = {}
    if immutable_value is not None:
        immutable_root = _resolve_required_dir(
            search_root,
            immutable_value,
            label=f"evaluation judge `{name_value}` immutable directory",
        )
        immutable_files = read_file_tree(immutable_root)

    return EvaluationStep(
        name=name_value,
        immutable_files=immutable_files,
        immutable_renderer=shared_renderer,
        entrypoint=read_required_prompt_text(prompt_root, "ENTRYPOINT.md"),
        rollout_prompts={"budget": BudgetPrompt(prompt_root=prompt_root)},
    )


def _resolve_config_path(search_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = search_root / path
    return path.resolve()


def _resolve_required_dir(search_root: Path, raw_path: object, *, label: str) -> Path:
    if raw_path is None:
        raise SystemExit(f"{label} is required.")
    path = _resolve_config_path(search_root, raw_path)
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    if not path.is_dir():
        raise SystemExit(f"{label} must be a directory: {path}")
    return path


def _load_evaluation_score(workspace_root: Path) -> PyTree:
    score_yaml_path = workspace_root / _EVAL_SCORE_YAML_PATH
    if not score_yaml_path.exists():
        raise ValueError("Evaluation did not write logs/evaluate/score.yaml")
    return yaml.safe_load(score_yaml_path.read_text(encoding="utf-8"))


def _validate_score_shape(score: PyTree, failure_score: PyTree) -> None:
    """Validate the final evaluation score before it can enter a population.

    `evaluation.failure_score` is the route's fallback score. Its numeric leaves define
    the minimum score shape that a successful `logs/evaluate/score.yaml` must provide.
    Non-numeric metadata in the failure score, such as `summary`, is intentionally ignored.
    """
    for path in _numeric_leaf_paths(failure_score):
        value = _read_score_path(score, path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            joined = ".".join(path) if path else "<score>"
            raise TypeError(f"evaluation score path `{joined}` must be numeric")


def _numeric_leaf_paths(score: PyTree) -> tuple[tuple[str, ...], ...]:
    """Return paths to numeric score leaves in a score PyTree.

    These paths are used as the evaluation output contract. For example,
    `{score: 0.0, summary: ...}` yields `("score",)`, while a dimensions score yields
    paths such as `("dimensions", "coverage")`.
    """
    paths: list[tuple[str, ...]] = []

    def _walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            paths.append(path)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(child, (*path, str(key)))

    _walk(score, ())
    return tuple(paths)


def _read_score_path(score: PyTree, path: tuple[str, ...]) -> object:
    """Read a nested score path, raising a clear error when the final score is incomplete."""
    current = score
    for key in path:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path) if path else "<score>"
            raise ValueError(f"evaluation score is missing required path `{joined}`")
        current = current[key]
    return current


def _write_score_yaml(path: Path, score: PyTree) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(score, sort_keys=False), encoding="utf-8")
