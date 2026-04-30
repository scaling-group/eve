"""Phase 2 solver-generation workflow helpers."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.rollout_prompts.default import PromptContext
from scaling_evolve.algorithms.eve.workflow.evaluation import (
    RemoteTransportHaltError,
    SolverEvaluator,
)
from scaling_evolve.algorithms.eve.workflow.optimize_logs import (
    build_optimize_log_tree,
)
from scaling_evolve.algorithms.eve.workspace.runtime_hooks import (
    install_workspace_runtime_hooks,
)
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkspaceBuilder,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver, SessionRollout, SessionSeed

_LOGGER = logging.getLogger(__name__)
_WORKSPACE_ID_HEX_LENGTH = 12


@dataclass
class Phase2Result:
    """Committed result produced by one Phase 2 worker."""

    optimizer: PopulationEntry
    sampled_solvers: list[PopulationEntry] = field(default_factory=list)
    sampled_optimizers: list[PopulationEntry] = field(default_factory=list)
    prefill_solver: PopulationEntry | None = None
    produced_solver: PopulationEntry | None = None
    produced_optimizer: PopulationEntry | None = None
    rollouts: list[SessionRollout] = field(default_factory=list)
    optimizer_log_tree: dict[str, str] = field(default_factory=dict)
    workspace_id: str = ""


@dataclass(frozen=True)
class SolverWorkspaceEvaluation:
    """Evaluated solver workspace ready for population commit."""

    entry: PopulationEntry
    score: PyTree
    evaluate_log_tree: dict[str, str]


def phase2_boundary_repair_instruction(boundary_result: object) -> str:
    summary = boundary_result.summary()
    return "\n".join(
        [
            "# Boundary Check Failed",
            "",
            "Your previous edit changed files outside the allowed editable surface.",
            "Restore every forbidden change, then rerun the predefined `check-runner`",
            "after each repair pass until it succeeds.",
            "",
            summary,
        ]
    )


class Phase2Runner:
    def __init__(
        self,
        *,
        solver_workspace_builder: SolverWorkspaceBuilder,
        driver: SessionDriver,
        solver_evaluator: SolverEvaluator,
        step_label: str,
        iteration: int,
        total_iterations: int | None = None,
        optimizer_pop: object | None = None,
    ) -> None:
        self.solver_workspace_builder = solver_workspace_builder
        self.driver = driver
        self.solver_evaluator = solver_evaluator
        self.step_label = step_label
        self.iteration = iteration
        self.total_iterations = total_iterations
        self.optimizer_pop = optimizer_pop
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.optimizer: PopulationEntry | None = None
        self.solvers: list[PopulationEntry] = []
        self.optimizer_examples: list[PopulationEntry] = []
        self.prefill_solver: PopulationEntry | None = None
        self.workspace: Path | None = None
        self.optimize_rollouts: list[SessionRollout] = []
        self.optimize_log_tree: dict[str, str] = {}
        self.boundary_result: object | None = None
        self.worker_index: int = 0

    def run_single(
        self,
        *,
        optimizer: PopulationEntry,
        solvers: list[PopulationEntry],
        optimizer_examples: list[PopulationEntry] | None = None,
        prefill_solver: PopulationEntry | None,
        worker_index: int,
    ) -> Phase2Result:
        self._reset_run_state()
        self.optimizer = optimizer
        self.solvers = list(solvers)
        self.optimizer_examples = list(optimizer_examples or [])
        self.prefill_solver = prefill_solver
        self.worker_index = worker_index
        self._build_workspace()
        self._run_agent()
        if self.workspace is None or self.boundary_result is None:
            raise ValueError("Phase2Runner.run_single requires completed workspace state.")
        evaluation = self.evaluate_workspace(
            workspace_root=self.workspace,
            solver_id=PopulationEntry.make_id("solver"),
            sampled_solvers=self.solvers,
            prefill_solver=self.prefill_solver,
            optimize_logs=self.optimize_log_tree,
            worker_index=self.worker_index,
            boundary_result=self.boundary_result,
        )
        result = self._finalize_result(evaluation)
        self._update_optimizer_logs(result)
        return result

    def _build_workspace(self) -> None:
        if self.optimizer is None:
            raise ValueError("Phase2Runner._build_workspace requires optimizer.")
        workspace_id = f"{self.step_label}_{uuid.uuid4().hex[:_WORKSPACE_ID_HEX_LENGTH]}"
        workspace, _ = self.solver_workspace_builder.build(
            self.optimizer.files,
            self.solvers,
            workspace_id=workspace_id,
            optimizer=self.optimizer,
            worker_index=self.worker_index,
            prefill_solver=self.prefill_solver,
            optimizer_examples=self.optimizer_examples,
        )
        readme = self.solver_workspace_builder.instructions["phase2_readme"].render(
            workspace_builder=self.solver_workspace_builder,
            optimizer=self.optimizer,
            solvers=self.solvers,
            prefill_solver=self.prefill_solver,
            optimizer_examples=self.optimizer_examples,
        )
        phase2_agent = self.solver_workspace_builder.instructions["phase2_agent"].render(
            workspace_builder=self.solver_workspace_builder,
            optimizer=self.optimizer,
            solvers=self.solvers,
            prefill_solver=self.prefill_solver,
            optimizer_examples=self.optimizer_examples,
        )
        self.solver_workspace_builder.write_readme(workspace, readme)
        self.solver_workspace_builder.write_workspace_agent_instructions(
            workspace,
            phase2_agent,
        )
        self.workspace = workspace

    def _run_agent(self) -> None:
        if self.workspace is None or self.optimizer is None:
            raise ValueError("Phase2Runner._run_agent requires workspace and optimizer.")
        prompt_specs = self._build_prompt_specs()
        install_workspace_runtime_hooks(
            self.workspace, driver=self.driver, prompt_specs=prompt_specs
        )
        rollout = self.driver.spawn(
            SessionSeed(
                instruction=self.solver_workspace_builder.instructions["phase2_entrypoint"].render(
                    workspace_builder=self.solver_workspace_builder,
                    optimizer=self.optimizer,
                    solvers=self.solvers,
                    prefill_solver=self.prefill_solver,
                ),
                working_directory=str(self.workspace),
                prompt_file="README.md",
                write_prompt_file=False,
                display_context={"iteration": self.iteration, "worker_index": self.worker_index},
            )
        )
        self.optimize_rollouts = [rollout]
        boundary_result = self.solver_evaluator.check_boundary(
            self.workspace,
            solver_workspace_builder=self.solver_workspace_builder,
        )
        repair_attempts = 0
        while (
            not boundary_result.ok
            and repair_attempts < self.solver_workspace_builder.config.boundary_repair_attempts
        ):
            repair_attempts += 1
            rollout = self.driver.resume(
                rollout.state,
                instruction=phase2_boundary_repair_instruction(boundary_result),
            )
            self.optimize_rollouts.append(rollout)
            boundary_result = self.solver_evaluator.check_boundary(
                self.workspace,
                solver_workspace_builder=self.solver_workspace_builder,
            )
        self.optimize_log_tree = build_optimize_log_tree(self.workspace, self.optimize_rollouts)
        if not boundary_result.ok:
            self.optimize_log_tree.setdefault(
                "boundary/forbidden_changes.txt", boundary_result.summary() + "\n"
            )
        self.boundary_result = boundary_result

    def _build_prompt_specs(self) -> list[dict[str, object]]:
        if self.workspace is None:
            raise ValueError("Phase2Runner._build_prompt_specs requires workspace.")
        if not _driver_budget_prompt_enabled(self.driver):
            return []
        ctx = PromptContext(
            workspace=self.workspace,
            rollout_max_turns=_driver_rollout_max_turns(self.driver),
        )
        prompt_specs: list[dict[str, object]] = []
        for name, prompt in self.solver_workspace_builder.rollout_prompts.items():
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

    def evaluate_workspace(
        self,
        *,
        workspace_root: Path,
        solver_id: str,
        sampled_solvers: list[PopulationEntry],
        prefill_solver: PopulationEntry | None,
        optimize_logs: dict[str, str],
        worker_index: int,
        boundary_result: object | None = None,
    ) -> SolverWorkspaceEvaluation:
        if boundary_result is None:
            boundary_result = self.solver_evaluator.check_boundary(
                workspace_root,
                solver_workspace_builder=self.solver_workspace_builder,
            )
        _LOGGER.info(
            "%s: evaluating workspace `%s` as solver candidate `%s`",
            self._phase_log_prefix(),
            workspace_root.name,
            solver_id,
        )
        candidate_files = self.solver_workspace_builder.extract(workspace_root)
        score, evaluate_log_tree = self.solver_evaluator.evaluate_candidate(
            workspace_root=workspace_root,
            boundary_result=boundary_result,
            display_context={"iteration": self.iteration, "worker_index": worker_index},
        )
        solver_logs = self.solver_workspace_builder.write_run_logs(
            workspace_root,
            optimize_logs=optimize_logs,
            evaluate_logs=evaluate_log_tree,
        )
        produced_solver = PopulationEntry(
            id=solver_id,
            files=candidate_files,
            score=score,
            logs=solver_logs,
        )
        self.solver_workspace_builder.write_score_manifest(
            workspace_root,
            sampled_solvers=sampled_solvers,
            optimizer=self.optimizer,
            sampled_optimizers=self.optimizer_examples,
            prefill_solver=prefill_solver,
            produced_solver=produced_solver,
        )
        _LOGGER.info(
            "%s: evaluation finished for solver candidate `%s`",
            self._phase_log_prefix(),
            solver_id,
        )
        return SolverWorkspaceEvaluation(
            entry=produced_solver,
            score=score,
            evaluate_log_tree=evaluate_log_tree,
        )

    def _finalize_result(
        self,
        evaluation: SolverWorkspaceEvaluation,
    ) -> Phase2Result:
        if self.workspace is None or self.optimizer is None:
            raise ValueError("Phase2Runner._finalize_result requires completed run state.")
        optimizer_log_tree = self.solver_workspace_builder.build_phase2_optimizer_log_tree(
            run_id=self.step_label,
            produced_solver=evaluation.entry,
            optimize_logs=self.optimize_log_tree,
            evaluate_logs=evaluation.evaluate_log_tree,
        )
        produced_optimizer = self._build_phase2_optimizer(
            produced_solver=evaluation.entry,
            optimizer_log_tree=optimizer_log_tree,
        )
        return Phase2Result(
            optimizer=self.optimizer,
            sampled_solvers=list(self.solvers),
            sampled_optimizers=list(self.optimizer_examples),
            prefill_solver=self.prefill_solver,
            produced_solver=evaluation.entry,
            produced_optimizer=produced_optimizer,
            rollouts=list(self.optimize_rollouts),
            optimizer_log_tree=optimizer_log_tree,
            workspace_id=self.workspace.name,
        )

    def _build_phase2_optimizer(
        self,
        *,
        produced_solver: PopulationEntry,
        optimizer_log_tree: dict[str, str],
    ) -> PopulationEntry | None:
        if (
            self.workspace is None
            or int(self.solver_workspace_builder.config.produce_optimizer_in_phase2) <= 0
        ):
            return None
        optimizer_files = self.solver_workspace_builder.extract_optimizer(self.workspace)
        if not optimizer_files:
            _LOGGER.warning(
                "%s: produced no optimizer files in %s.",
                self._phase_log_prefix(),
                self.workspace,
            )
            return None
        if self.optimizer is not None and optimizer_files == self.optimizer.files:
            _LOGGER.warning(
                "%s: produce_optimizer_in_phase2 is enabled but the guidance tree was not "
                "modified; no optimizer candidate will be produced for solver %s.",
                self._phase_log_prefix(),
                produced_solver.id,
            )
            return None
        return PopulationEntry(
            id=PopulationEntry.make_id("optimizer"),
            files=optimizer_files,
            score=deepcopy(self.optimizer.score if self.optimizer is not None else {}),
            logs=optimizer_log_tree,
        )

    def _update_optimizer_logs(self, result: Phase2Result) -> None:
        if self.optimizer_pop is None or result.produced_solver is None:
            return
        self.optimizer_pop.update_logs({result.optimizer.id: result.optimizer_log_tree})

    def _phase_log_prefix(self) -> str:
        if self.total_iterations is None:
            return f"Iteration {self.iteration} | Phase 2"
        return f"Iteration {self.iteration}/{self.total_iterations} | Phase 2"


def _driver_rollout_max_turns(driver: SessionDriver) -> int | None:
    value = getattr(driver, "rollout_max_turns", None)
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _driver_budget_prompt_enabled(driver: SessionDriver) -> bool:
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


class Phase2BatchRunner:
    def __init__(
        self,
        *,
        solver_workspace_builder: SolverWorkspaceBuilder,
        driver: SessionDriver,
        solver_evaluator: SolverEvaluator,
        step_label: str,
        iteration: int,
        total_iterations: int | None = None,
        solver_pop: object,
        optimizer_pop: object,
        n_workers_phase2: int,
        n_solver_examples_phase2: int,
        n_optimizer_examples_phase2: int,
        n_produced_optimizers_phase2: int,
        optimizer_sampler: object,
        solver_sampler: object,
        prefill_sampler: object,
        optimizer_examples_sampler: object,
        produced_optimizer_sampler: object,
    ) -> None:
        self.solver_workspace_builder = solver_workspace_builder
        self.driver = driver
        self.solver_evaluator = solver_evaluator
        self.step_label = step_label
        self.iteration = iteration
        self.total_iterations = total_iterations
        self.solver_pop = solver_pop
        self.optimizer_pop = optimizer_pop
        self.n_workers_phase2 = n_workers_phase2
        self.n_solver_examples_phase2 = n_solver_examples_phase2
        self.n_optimizer_examples_phase2 = n_optimizer_examples_phase2
        self.n_produced_optimizers_phase2 = n_produced_optimizers_phase2
        self.optimizer_sampler = optimizer_sampler
        self.solver_sampler = solver_sampler
        self.prefill_sampler = prefill_sampler
        self.optimizer_examples_sampler = optimizer_examples_sampler
        self.produced_optimizer_sampler = produced_optimizer_sampler

    def run(self) -> list[Phase2Result]:
        optimizers = self._sample_optimizers()
        solvers = self._sample_solvers()
        if not optimizers:
            _LOGGER.warning("Optimizer population is empty; skipping iteration.")
            return []
        shared_optimizer_examples = self._sample_shared_optimizer_examples(optimizers)
        results: list[Phase2Result] = []
        with ThreadPoolExecutor(max_workers=len(optimizers)) as pool:
            future_to_optimizer = {
                pool.submit(
                    Phase2Runner(
                        solver_workspace_builder=self.solver_workspace_builder,
                        driver=self.driver,
                        solver_evaluator=self.solver_evaluator,
                        step_label=self.step_label,
                        iteration=self.iteration,
                        total_iterations=self.total_iterations,
                        optimizer_pop=self.optimizer_pop,
                    ).run_single,
                    optimizer=optimizer,
                    solvers=solvers,
                    optimizer_examples=self._build_worker_optimizer_examples(
                        optimizer,
                        shared_optimizer_examples,
                    ),
                    prefill_solver=self._sample_prefill_solver(solvers),
                    worker_index=worker_index,
                ): optimizer
                for worker_index, optimizer in enumerate(optimizers, start=1)
            }
            for future in as_completed(future_to_optimizer):
                optimizer = future_to_optimizer[future]
                try:
                    results.append(future.result())
                except RemoteTransportHaltError:
                    _LOGGER.error(
                        "%s: transport halt while evaluating optimizer %s; aborting iteration.",
                        self._phase_log_prefix(),
                        optimizer.id,
                    )
                    raise
                except Exception:
                    _LOGGER.exception(
                        "%s: failed for optimizer %s; skipping.",
                        self._phase_log_prefix(),
                        optimizer.id,
                    )
        for result in results:
            if result.produced_solver is None:
                continue
            self.solver_pop.add(result.produced_solver)
            _LOGGER.info(
                "%s: added solver candidate %s",
                self._phase_log_prefix(),
                result.produced_solver.id,
            )
        self._add_selected_phase2_optimizers(results)
        return results

    def _add_selected_phase2_optimizers(self, results: list[Phase2Result]) -> None:
        produced_results = [
            result
            for result in results
            if result.produced_optimizer is not None and result.produced_solver is not None
        ]
        if not produced_results:
            return
        if self.n_produced_optimizers_phase2 <= 0:
            for result in produced_results:
                result.produced_optimizer = None
            return
        selected_results = list(
            self.produced_optimizer_sampler.sample(
                produced_results,
                [result.produced_solver.score for result in produced_results],
                self.n_produced_optimizers_phase2,
                rng=self.optimizer_pop._rng,
            )
        )
        selected_result_ids = {id(result) for result in selected_results}
        for result in produced_results:
            if id(result) not in selected_result_ids:
                result.produced_optimizer = None
                continue
            assert result.produced_optimizer is not None
            self.optimizer_pop.add(result.produced_optimizer)
            _LOGGER.info(
                "%s: added optimizer candidate %s",
                self._phase_log_prefix(),
                result.produced_optimizer.id,
            )

    def _sample_optimizers(self) -> list[PopulationEntry]:
        optimizer_entries = self.optimizer_pop.entries()
        return list(
            self.optimizer_sampler.sample(
                optimizer_entries,
                [entry.score for entry in optimizer_entries],
                self.n_workers_phase2,
                rng=self.optimizer_pop._rng,
            )
        )

    def _sample_solvers(self) -> list[PopulationEntry]:
        solver_entries = self.solver_pop.entries()
        return list(
            self.solver_sampler.sample(
                solver_entries,
                [entry.score for entry in solver_entries],
                self.n_solver_examples_phase2,
                rng=self.solver_pop._rng,
            )
        )

    def _sample_prefill_solver(
        self,
        solvers: list[PopulationEntry],
    ) -> PopulationEntry | None:
        prefill_candidates = self.prefill_sampler.sample(
            solvers,
            [entry.score for entry in solvers],
            1,
            rng=self.solver_workspace_builder._rng,
        )
        return prefill_candidates[0] if prefill_candidates else None

    def _sample_shared_optimizer_examples(
        self,
        selected_optimizers: list[PopulationEntry],
    ) -> list[PopulationEntry]:
        if self.n_optimizer_examples_phase2 <= 1:
            return []
        selected_ids = {entry.id for entry in selected_optimizers}
        optimizer_entries = [
            entry for entry in self.optimizer_pop.entries() if entry.id not in selected_ids
        ]
        return list(
            self.optimizer_examples_sampler.sample(
                optimizer_entries,
                [entry.score for entry in optimizer_entries],
                self.n_optimizer_examples_phase2 - 1,
                rng=self.optimizer_pop._rng,
            )
        )

    def _build_worker_optimizer_examples(
        self,
        optimizer: PopulationEntry,
        shared_optimizer_examples: list[PopulationEntry],
    ) -> list[PopulationEntry]:
        """Build the per-worker optimizer examples for `guidance_examples/`.

        Semantics are determined by `n_optimizer_examples_phase2`:

        - `n_optimizer_examples_phase2 <= 0`: do not create optimizer examples.
        - `n_optimizer_examples_phase2 == 1`: include only the current worker's
          optimizer, which is also copied into `guidance/`.
        - `n_optimizer_examples_phase2 > 1`: include the current worker's
          optimizer plus the shared `n-1` reference optimizers sampled for this
          iteration.
        """
        if self.n_optimizer_examples_phase2 <= 0:
            return []
        return [optimizer, *shared_optimizer_examples[: self.n_optimizer_examples_phase2 - 1]]

    def _phase_log_prefix(self) -> str:
        if self.total_iterations is None:
            return f"Iteration {self.iteration} | Phase 2"
        return f"Iteration {self.iteration}/{self.total_iterations} | Phase 2"
