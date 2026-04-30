"""Phase 4 optimizer-improvement helpers."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.rollout_prompts.default import PromptContext
from scaling_evolve.algorithms.eve.workflow.optimize_logs import (
    build_optimize_log_tree,
)
from scaling_evolve.algorithms.eve.workspace.optimizer_workspace import (
    OptimizerWorkspaceBuilder,
)
from scaling_evolve.algorithms.eve.workspace.runtime_hooks import (
    install_workspace_runtime_hooks,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver, SessionRollout, SessionSeed

_LOGGER = logging.getLogger(__name__)
_WORKSPACE_ID_HEX_LENGTH = 12


@dataclass(frozen=True)
class Phase4Result:
    """Structured outcome of one Phase 4 optimizer-improvement run."""

    lead_optimizer: PopulationEntry
    sampled_optimizers: list[PopulationEntry] = field(default_factory=list)
    prefill_optimizer: PopulationEntry | None = None
    produced_optimizer: PopulationEntry | None = None
    rollouts: list[SessionRollout] = field(default_factory=list)
    workspace_id: str = ""


@dataclass(frozen=True)
class OptimizerWorkspaceResult:
    """Finalized optimizer workspace ready for population commit."""

    entry: PopulationEntry


@dataclass(frozen=True)
class _TaskLogCandidate:
    root: str
    solver_id: str
    score: PyTree
    files: dict[str, str]
    display_name: str


class Phase4Runner:
    def __init__(
        self,
        *,
        optimizer_workspace_builder: OptimizerWorkspaceBuilder,
        driver: SessionDriver,
        step_label: str,
        iteration: int,
        optimizer_pop: object | None = None,
    ) -> None:
        self.optimizer_workspace_builder = optimizer_workspace_builder
        self.driver = driver
        self.step_label = step_label
        self.iteration = iteration
        self.optimizer_pop = optimizer_pop
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.lead_optimizer: PopulationEntry | None = None
        self.sampled_optimizers: list[PopulationEntry] = []
        self.prefill_optimizer: PopulationEntry | None = None
        self.workspace: Path | None = None
        self.rollout: SessionRollout | None = None
        self.optimize_log_tree: dict[str, str] = {}
        self.example_logs_by_optimizer: dict[str, dict[str, str]] = {}
        self.worker_index: int = 0

    def run_single(
        self,
        *,
        lead: PopulationEntry,
        sampled_optimizers: list[PopulationEntry],
        prefill_optimizer: PopulationEntry | None,
        example_logs_by_optimizer: dict[str, dict[str, str]],
        worker_index: int,
    ) -> Phase4Result:
        self._reset_run_state()
        self.lead_optimizer = lead
        self.sampled_optimizers = list(sampled_optimizers)
        self.prefill_optimizer = prefill_optimizer
        self.example_logs_by_optimizer = dict(example_logs_by_optimizer)
        self.worker_index = worker_index
        self._build_workspace()
        self._run_agent()
        workspace_result = self._finalize_result()
        return Phase4Result(
            lead_optimizer=self.lead_optimizer,
            sampled_optimizers=list(self.sampled_optimizers),
            prefill_optimizer=self.prefill_optimizer,
            produced_optimizer=workspace_result.entry,
            rollouts=[self.rollout] if self.rollout is not None else [],
            workspace_id=self.workspace.name if self.workspace is not None else "",
        )

    def _build_workspace(self) -> None:
        if self.lead_optimizer is None:
            raise ValueError("Phase4Runner._build_workspace requires lead optimizer.")
        workspace_id = f"{self.step_label}_{uuid.uuid4().hex[:_WORKSPACE_ID_HEX_LENGTH]}"
        workspace, _ = self.optimizer_workspace_builder.build(
            self.lead_optimizer,
            self.sampled_optimizers,
            workspace_id=workspace_id,
            worker_index=self.worker_index,
            prefill_optimizer=self.prefill_optimizer,
            example_logs_by_optimizer=self.example_logs_by_optimizer,
        )
        readme = self.optimizer_workspace_builder.instructions["phase4_readme"].render(
            workspace_builder=self.optimizer_workspace_builder,
            lead_optimizer=self.lead_optimizer,
            optimizers=self.sampled_optimizers,
            prefill_optimizer=self.prefill_optimizer,
        )
        phase4_agent = self.optimizer_workspace_builder.instructions["phase4_agent"].render(
            workspace_builder=self.optimizer_workspace_builder,
            lead_optimizer=self.lead_optimizer,
            optimizers=self.sampled_optimizers,
            prefill_optimizer=self.prefill_optimizer,
        )
        self.optimizer_workspace_builder.write_readme(workspace, readme)
        self.optimizer_workspace_builder.write_workspace_agent_instructions(
            workspace,
            phase4_agent,
        )
        self.workspace = workspace

    def _run_agent(self) -> None:
        if self.workspace is None or self.lead_optimizer is None:
            raise ValueError("Phase4Runner._run_agent requires workspace and lead optimizer.")
        prompt_specs = self._build_prompt_specs()
        install_workspace_runtime_hooks(
            self.workspace, driver=self.driver, prompt_specs=prompt_specs
        )
        rollout = self.driver.spawn(
            SessionSeed(
                instruction=self.optimizer_workspace_builder.instructions[
                    "phase4_entrypoint"
                ].render(
                    workspace_builder=self.optimizer_workspace_builder,
                    lead_optimizer=self.lead_optimizer,
                    optimizers=self.sampled_optimizers,
                    prefill_optimizer=self.prefill_optimizer,
                ),
                working_directory=str(self.workspace),
                prompt_file="README.md",
                write_prompt_file=False,
                display_context={"iteration": self.iteration, "worker_index": self.worker_index},
            )
        )
        optimize_log_tree = build_optimize_log_tree(self.workspace, [rollout])
        self.optimizer_workspace_builder.write_optimize_log(
            self.workspace,
            optimize_logs=optimize_log_tree,
        )
        self.rollout = rollout
        self.optimize_log_tree = optimize_log_tree

    def _build_prompt_specs(self) -> list[dict[str, object]]:
        if self.workspace is None:
            raise ValueError("Phase4Runner._build_prompt_specs requires workspace.")
        if not _driver_budget_prompt_enabled(self.driver):
            return []
        ctx = PromptContext(
            workspace=self.workspace,
            rollout_max_turns=_driver_rollout_max_turns(self.driver),
        )
        prompt_specs: list[dict[str, object]] = []
        for name, prompt in self.optimizer_workspace_builder.rollout_prompts.items():
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

    def _inherit_score(self, prefill_optimizer: PopulationEntry | None) -> PyTree:
        if prefill_optimizer is None:
            raise ValueError("Phase 4 requires a prefill optimizer to inherit score.")
        return deepcopy(prefill_optimizer.score)

    def _finalize_result(self) -> OptimizerWorkspaceResult:
        if self.workspace is None or self.lead_optimizer is None:
            raise ValueError("Phase4Runner._finalize_result requires completed run state.")
        new_files = self.optimizer_workspace_builder.extract(self.workspace)
        initial_score = self._inherit_score(self.prefill_optimizer)
        if not new_files:
            _LOGGER.warning(
                "Phase 4 produced no optimizer files for lead %s.",
                self.lead_optimizer.id,
            )
        produced_optimizer = PopulationEntry(
            id=PopulationEntry.make_id("optimizer"),
            files=new_files,
            score=initial_score,
            logs={},
        )
        self.optimizer_workspace_builder.write_score_manifest(
            self.workspace,
            sampled_optimizers=self.sampled_optimizers,
            sampled_solver_history_ids=self.optimizer_workspace_builder.sampled_solver_history_ids(
                self.sampled_optimizers,
                self.example_logs_by_optimizer,
            ),
            lead_optimizer=self.lead_optimizer,
            prefill_optimizer=self.prefill_optimizer,
            produced_optimizer=produced_optimizer,
        )
        return OptimizerWorkspaceResult(entry=produced_optimizer)


class Phase4BatchRunner:
    def __init__(
        self,
        *,
        optimizer_workspace_builder: OptimizerWorkspaceBuilder,
        driver: SessionDriver,
        step_label: str,
        iteration: int,
        optimizer_pop: object,
        lead_sampler: object,
        optimizer_examples_sampler: object,
        prefill_sampler: object,
        history_log_sampler: object,
        n_workers_phase4: int,
        n_optimizer_examples_phase4: int,
        n_latest_solver_logs: int,
    ) -> None:
        self.optimizer_workspace_builder = optimizer_workspace_builder
        self.driver = driver
        self.step_label = step_label
        self.iteration = iteration
        self.optimizer_pop = optimizer_pop
        self.lead_sampler = lead_sampler
        self.optimizer_examples_sampler = optimizer_examples_sampler
        self.prefill_sampler = prefill_sampler
        self.history_log_sampler = history_log_sampler
        self.n_workers_phase4 = n_workers_phase4
        self.n_optimizer_examples_phase4 = n_optimizer_examples_phase4
        self.n_latest_solver_logs = n_latest_solver_logs

    def run(self) -> list[Phase4Result]:
        leads = self._sample_leads(n=self.n_workers_phase4)
        if not leads:
            return []

        worker_inputs = [
            (
                worker_index,
                lead,
                self._sample_optimizer_examples(),
            )
            for worker_index, lead in enumerate(leads, start=1)
        ]
        indexed_results: list[tuple[int, Phase4Result]] = []
        with ThreadPoolExecutor(max_workers=self.n_workers_phase4) as pool:
            future_to_input = {
                pool.submit(
                    Phase4Runner(
                        optimizer_workspace_builder=self.optimizer_workspace_builder,
                        driver=self.driver,
                        step_label=self.step_label,
                        iteration=self.iteration,
                    ).run_single,
                    lead=lead,
                    sampled_optimizers=sampled_optimizers,
                    prefill_optimizer=self._sample_prefill_optimizer(sampled_optimizers),
                    example_logs_by_optimizer=self._sample_example_logs(sampled_optimizers),
                    worker_index=worker_index,
                ): (worker_index, lead)
                for worker_index, lead, sampled_optimizers in worker_inputs
            }
            for future in as_completed(future_to_input):
                worker_index, lead = future_to_input[future]
                try:
                    indexed_results.append((worker_index, future.result()))
                except Exception:
                    _LOGGER.exception("Phase 4 failed for lead optimizer %s.", lead.id)
                    raise
        indexed_results.sort(key=lambda item: item[0])
        results = [result for _, result in indexed_results]
        for result in results:
            if result.produced_optimizer is None:
                continue
            self.optimizer_pop.add(result.produced_optimizer)
            _LOGGER.info("Phase 4: added new optimizer %s", result.produced_optimizer.id)
        return results

    def _sample_leads(self, *, n: int) -> list[PopulationEntry]:
        optimizer_entries = self.optimizer_pop.entries()
        leads = self.lead_sampler.sample(
            optimizer_entries,
            [entry.score for entry in optimizer_entries],
            n,
            rng=self.optimizer_pop._rng,
        )
        if not leads:
            _LOGGER.warning("Optimizer population empty in Phase 4; skipping.")
            return []
        return list(leads)

    def _sample_optimizer_examples(self) -> list[PopulationEntry]:
        optimizer_entries = self.optimizer_pop.entries()
        return list(
            self.optimizer_examples_sampler.sample(
                optimizer_entries,
                [entry.score for entry in optimizer_entries],
                self.n_optimizer_examples_phase4,
                rng=self.optimizer_pop._rng,
            )
        )

    def _sample_prefill_optimizer(
        self,
        optimizers: list[PopulationEntry],
    ) -> PopulationEntry | None:
        prefill_candidates = self.prefill_sampler.sample(
            optimizers,
            [entry.score for entry in optimizers],
            1,
            rng=self.optimizer_pop._rng,
        )
        return prefill_candidates[0] if prefill_candidates else None

    def _sample_example_logs(
        self,
        optimizers: list[PopulationEntry],
    ) -> dict[str, dict[str, str]]:
        return {
            entry.id: self._sample_solver_logs(
                entry.logs,
                sample_size=self.n_latest_solver_logs,
                sampler=self.history_log_sampler,
                rng=self.optimizer_pop._rng,
            )
            for entry in optimizers
        }

    def _sample_solver_logs(
        self,
        logs: dict[str, str],
        *,
        sample_size: int,
        sampler: object,
        rng: object,
    ) -> dict[str, str]:
        if sample_size <= 0:
            return {}

        passthrough: dict[str, str] = {}
        grouped: dict[str, dict[str, str]] = {}
        for path, content in sorted(logs.items()):
            if "/" not in path:
                passthrough[path] = content
                continue
            root, _, remainder = path.partition("/")
            grouped.setdefault(root, {})[remainder] = content

        solver_logs: list[_TaskLogCandidate] = []
        for root, files in grouped.items():
            score_text = files.get("score.yaml")
            if score_text is None:
                passthrough.update({f"{root}/{path}": content for path, content in files.items()})
                continue
            solver_id, score = self._solver_log_metadata(root, score_text)
            solver_logs.append(
                _TaskLogCandidate(
                    root=root,
                    solver_id=solver_id,
                    score=score,
                    files=files,
                    display_name=solver_id,
                )
            )

        sampled = sampler.sample(
            solver_logs,
            [solver_log.score for solver_log in solver_logs],
            sample_size,
            rng=rng,
        )
        filtered = dict(passthrough)
        for candidate in sampled:
            for relative_path, content in candidate.files.items():
                filtered[f"{candidate.display_name}/{relative_path}"] = content
        return filtered

    def _solver_log_metadata(self, root: str, score_text: str) -> tuple[str, PyTree]:
        payload = yaml.safe_load(score_text) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping in optimizer task score.yaml for {root}")
        solver_id = payload.get("solver_id", root)
        if not isinstance(solver_id, str) or not solver_id:
            raise ValueError(f"Missing solver_id in optimizer task score.yaml for {root}")
        if "score" not in payload:
            raise ValueError(f"Missing score in optimizer task score.yaml for {root}")
        return solver_id, payload["score"]


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
