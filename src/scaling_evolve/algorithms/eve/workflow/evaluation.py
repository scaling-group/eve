"""Agent-backed evaluation helpers for Eve solver candidates."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import score_block_lines
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.workspace.file_tree import read_file_tree
from scaling_evolve.providers.agent.drivers.base import SessionSeed
from scaling_evolve.providers.agent.evaluation import run_agent_for_result

_LOGGER = logging.getLogger(__name__)
_EVAL_LOG_ROOT = Path("logs") / "evaluate"
_EVAL_SCORE_YAML_PATH = _EVAL_LOG_ROOT / "score.yaml"
_EVAL_STEP_LOG_ROOT = _EVAL_LOG_ROOT / "steps"


class RemoteTransportHaltError(RuntimeError):
    """Wrapper-level transport halt that must stop the runner."""

    def __init__(
        self,
        *,
        step: Path,
        workspace_root: Path,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(
            "wrapper transport halt: "
            f"workspace={workspace_root.name} step={step.name} returncode={returncode}"
        )
        self.step = step
        self.workspace_root = workspace_root
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class SolverEvaluator:
    """Owns solver-evaluation policy and runtime execution."""

    problem: RepoTaskProblem
    evaluation_driver_factory: Callable[[], object]
    evaluation_failure_score: PyTree
    boundary_failure_score: PyTree
    seed_solver_score: PyTree | None
    seed_solver_skip_evaluation: bool

    def __call__(
        self,
        workspace_root: Path,
        *,
        display_context: dict[str, str | int] | None = None,
    ) -> tuple[PyTree, dict[str, str]]:
        return self.evaluate(workspace_root, display_context=display_context)

    def evaluate(
        self,
        workspace_root: Path,
        *,
        display_context: dict[str, str | int] | None = None,
    ) -> tuple[PyTree, dict[str, str]]:
        log_root = workspace_root / _EVAL_LOG_ROOT
        _LOGGER.info("Evaluation: starting workspace `%s`", workspace_root.name)
        try:
            log_root.mkdir(parents=True, exist_ok=True)
            _run_evaluation_steps(
                problem=self.problem,
                workspace_root=workspace_root,
                driver_factory=self.evaluation_driver_factory,
                display_context=dict(display_context or {}),
            )
            score = _load_evaluation_score(workspace_root)

            logs = read_file_tree(log_root)
            rendered_score = "\n".join(score_block_lines(score, indent=2))
            _LOGGER.info(
                "Evaluation: parsed score for workspace `%s`\n%s",
                workspace_root.name,
                rendered_score,
            )
            return score, logs
        except RemoteTransportHaltError:
            raise
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
        display_context: dict[str, str | int] | None = None,
    ) -> tuple[PyTree, dict[str, str]]:
        if not boundary_result.ok:
            return (
                self.boundary_failure_score,
                {"boundary/forbidden_changes.txt": boundary_result.summary() + "\n"},
            )
        return self.evaluate(workspace_root, display_context=display_context)

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
                    "application.seed_solver_score is required when "
                    "application.seed_solver_skip_evaluation "
                    "is true."
                )
            return (
                self._seed_entry_from_fixed_score(
                    workspace=workspace,
                    seed_solver_id=seed_solver_id,
                    score=self.seed_solver_score,
                    summary="Seed evaluation skipped; used application.seed_solver_score.",
                    solver_workspace_builder=solver_workspace_builder,
                ),
                False,
            )

        score, evaluate_logs = self.evaluate(
            workspace,
            display_context={"iteration": 0, "worker_index": 0},
        )
        candidate_files = solver_workspace_builder.extract(workspace)
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
        solver_workspace_builder.write_score_manifest(
            workspace,
            sampled_solvers=[],
            prefill_solver=None,
            produced_solver=seed_entry,
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
        normalized_score = (
            OmegaConf.to_container(score, resolve=True) if OmegaConf.is_config(score) else score
        )
        candidate_files = solver_workspace_builder.extract(workspace)
        evaluate_logs = {
            "score.yaml": yaml.safe_dump(normalized_score, sort_keys=False),
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
            score=normalized_score,
            logs=solver_logs,
        )
        solver_workspace_builder.write_score_manifest(
            workspace,
            sampled_solvers=[],
            prefill_solver=None,
            produced_solver=seed_entry,
        )
        return seed_entry


def _step_log_dir(*, step: Path, step_index: int, workspace_root: Path) -> Path:
    return workspace_root / _EVAL_STEP_LOG_ROOT / f"step_{step_index:02d}_{step.stem}"


def _run_shell_step(*, step: Path, step_index: int, workspace_root: Path) -> None:
    step_log_dir = _step_log_dir(step=step, step_index=step_index, workspace_root=workspace_root)
    step_log_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(
        "Evaluation: starting shell step %02d `%s` for workspace `%s`; "
        "waiting for completion, logs under %s",
        step_index,
        step.name,
        workspace_root.name,
        step_log_dir,
    )
    env = {
        **os.environ,
        "EVE_WORKSPACE_ROOT": str(workspace_root),
        "EVE_OUTPUT_ROOT": str(workspace_root / "output"),
        "EVE_EVAL_LOG_ROOT": str(workspace_root / _EVAL_LOG_ROOT),
    }
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
    raise RemoteTransportHaltError(
        step=step,
        workspace_root=workspace_root,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _run_markdown_step(
    *,
    step: Path,
    step_index: int,
    workspace_root: Path,
    driver_factory,
    display_context: dict[str, str | int],
) -> None:
    step_log_dir = _step_log_dir(step=step, step_index=step_index, workspace_root=workspace_root)
    step_log_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(
        "Evaluation: starting markdown step %02d `%s` for workspace `%s`",
        step_index,
        step.name,
        workspace_root.name,
    )
    driver = driver_factory()
    run = run_agent_for_result(
        driver=driver,
        seed=SessionSeed(
            instruction=step.read_text(encoding="utf-8"),
            working_directory=str(workspace_root),
            write_prompt_file=False,
            display_context=display_context,
        ),
        load_result_text=lambda _rollout: "",
    )
    if run.error is not None:
        (step_log_dir / "status.txt").write_text("failed\n", encoding="utf-8")
        (step_log_dir / "error.txt").write_text(f"{run.error}\n", encoding="utf-8")
        _LOGGER.warning(
            "Evaluation: markdown step %02d `%s` failed: %s",
            step_index,
            step.name,
            run.error,
        )
        raise RuntimeError(f"Evaluation markdown step failed: {step.name}\n{run.error}")
    (step_log_dir / "status.txt").write_text("ok\n", encoding="utf-8")
    if run.rollout is not None and run.rollout.summary:
        (step_log_dir / "summary.txt").write_text(f"{run.rollout.summary}\n", encoding="utf-8")
    _LOGGER.info(
        "Evaluation: markdown step %02d `%s` finished successfully",
        step_index,
        step.name,
    )


def _run_evaluation_steps(
    *,
    problem: RepoTaskProblem,
    workspace_root: Path,
    driver_factory,
    display_context: dict[str, str | int],
) -> None:
    for step_index, step in enumerate(problem.evaluation_steps, start=1):
        if step.suffix == ".sh":
            _run_shell_step(step=step, step_index=step_index, workspace_root=workspace_root)
            continue
        if step.suffix == ".md":
            _run_markdown_step(
                step=step,
                step_index=step_index,
                workspace_root=workspace_root,
                driver_factory=driver_factory,
                display_context=display_context,
            )
            continue
        raise ValueError(f"Unsupported evaluation step type: {step}")


def build_solver_evaluator(
    problem: RepoTaskProblem,
    *,
    evaluation_failure_score: PyTree,
    boundary_failure_score: PyTree,
    seed_solver_score: PyTree | None,
    seed_solver_skip_evaluation: bool,
    evaluation_driver_factory,
) -> SolverEvaluator:
    return SolverEvaluator(
        problem=problem,
        evaluation_driver_factory=evaluation_driver_factory,
        evaluation_failure_score=evaluation_failure_score,
        boundary_failure_score=boundary_failure_score,
        seed_solver_score=seed_solver_score,
        seed_solver_skip_evaluation=seed_solver_skip_evaluation,
    )


def _load_evaluation_score(workspace_root: Path) -> PyTree:
    score_yaml_path = workspace_root / _EVAL_SCORE_YAML_PATH
    if not score_yaml_path.exists():
        raise ValueError("Evaluation did not write logs/evaluate/score.yaml")
    return yaml.safe_load(score_yaml_path.read_text(encoding="utf-8"))
