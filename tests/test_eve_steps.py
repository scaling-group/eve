from __future__ import annotations

import json
import logging
import random
import sqlite3
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.evaluators.elo import (
    ScalarEloEvaluator,
    VectorEloEvaluator,
)
from scaling_evolve.algorithms.eve.populations.samplers.rank_softmax import (
    RankExponentialSumSampler,
    RankSoftmaxSampler,
)
from scaling_evolve.algorithms.eve.populations.samplers.uniform import UniformSampler
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.rollout_prompts.default import BudgetPrompt
from scaling_evolve.algorithms.eve.workflow.boundary import (
    BoundaryCheckResult,
    check_workspace_boundary,
)
from scaling_evolve.algorithms.eve.workflow.evaluation import (
    RemoteTransportHaltError,
    build_solver_evaluator,
)
from scaling_evolve.algorithms.eve.workflow.loop import Eve
from scaling_evolve.algorithms.eve.workflow.phase2 import (
    Phase2BatchRunner,
    Phase2Result,
    Phase2Runner,
    phase2_boundary_repair_instruction,
)
from scaling_evolve.algorithms.eve.workflow.phase3 import score_optimizers
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4BatchRunner, Phase4Runner
from scaling_evolve.algorithms.eve.workspace.optimizer_workspace import (
    OptimizerWorkspaceBuilder,
)
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkspaceBuilder,
)


def _make_test_config(workspace_root: Path | str = "run", **overrides) -> DictConfig:
    """Build a test DictConfig with defaults matching loop/default.yaml."""
    _S = "scaling_evolve.algorithms.eve"
    _RS = f"{_S}.populations.samplers.rank_softmax"
    _US = f"{_S}.populations.samplers.uniform"
    base = "configs/eve/prompt/templates/built_in"
    cfg = {
        "max_iterations": 2,
        "n_workers_phase2": 2,
        "n_workers_phase4": 1,
        "n_solver_examples_phase2": 4,
        "n_optimizer_examples_phase2": 4,
        "n_optimizer_examples_phase4": 2,
        "n_logs_per_example_phase4": 8,
        "boundary_repair_attempts": 3,
        "enable_iter_snapshots": True,
        "iter_snapshot_retain": 3,
        "retain_workspaces": True,
        "produce_optimizer_in_phase2": 0,
        "sampling": {
            "phase1_optimizer_population": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "auto",
            },
            "phase1_solver_population": {
                "_target_": f"{_RS}.RankExponentialSumSampler",
                "features": {
                    "score": {"weight": 1.0, "temperature": 1.0},
                },
                "replacement_mode": "no_replacement",
            },
            "solver_workspace_prefill": {
                "_target_": f"{_US}.UniformSampler",
                "replacement_mode": "no_replacement",
            },
            "phase2_optimizer_examples": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "no_replacement",
            },
            "phase2_produced_optimizers": {
                "_target_": f"{_RS}.RankExponentialSumSampler",
                "features": {
                    "score": {"weight": 1.0, "temperature": 1.0},
                },
                "replacement_mode": "no_replacement",
            },
            "phase4_lead_optimizer": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "no_replacement",
            },
            "phase4_optimizer_examples": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "no_replacement",
            },
            "optimizer_workspace_prefill": {
                "_target_": f"{_US}.UniformSampler",
                "replacement_mode": "no_replacement",
            },
            "optimizer_history_logs": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "no_replacement",
            },
        },
        "instructions": {
            "phase2_readme": {
                "_target_": f"{_S}.instructions.default.Phase2ReadmeInstruction",
                "file_list": [
                    f"{base}/design_doc.md",
                    f"{base}/phase2_readme.md",
                    f"{base}/phase2_score_semantics.md",
                ],
            },
            "phase2_entrypoint": {
                "_target_": f"{_S}.instructions.default.Phase2EntrypointInstruction",
                "file_list": [f"{base}/phase2_entrypoint.md"],
            },
            "phase2_agent": {
                "_target_": f"{_S}.instructions.default.Phase2AgentInstruction",
                "file_list": [
                    f"{base}/workspace_agent.md",
                ],
            },
            "phase4_readme": {
                "_target_": f"{_S}.instructions.default.Phase4ReadmeInstruction",
                "file_list": [
                    f"{base}/design_doc.md",
                    f"{base}/phase4_readme.md",
                    f"{base}/phase4_score_semantics.md",
                    f"{base}/skills_doc.md",
                ],
            },
            "phase4_entrypoint": {
                "_target_": f"{_S}.instructions.default.Phase4EntrypointInstruction",
                "file_list": [f"{base}/phase4_entrypoint.md"],
            },
            "phase4_agent": {
                "_target_": f"{_S}.instructions.default.Phase4AgentInstruction",
                "file_list": [
                    f"{base}/workspace_agent.md",
                ],
            },
        },
        "rollout": {
            "budget": {
                "_target_": f"{_S}.rollout_prompts.default.BudgetPrompt",
            },
        },
    }
    ws = Path(workspace_root) if isinstance(workspace_root, str) else workspace_root
    cfg["workspace_root"] = str(ws)
    cfg["run_id"] = "test-run"
    cfg["label"] = ""
    cfg["artifact_root"] = str(ws / "artifacts")
    cfg["solver_db_path"] = str(ws / "solver_lineage.db")
    cfg["optimizer_db_path"] = str(ws / "optimizer_lineage.db")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged = {**cfg[key]}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict) and isinstance(merged.get(nested_key), dict):
                    merged[nested_key] = {**merged[nested_key], **nested_value}
                else:
                    merged[nested_key] = nested_value
            cfg[key] = merged
        else:
            cfg[key] = value
    return OmegaConf.create(cfg)


class _FakeDriver:
    def __init__(self) -> None:
        self.resume_calls = 0
        self.spawn_instructions: list[str] = []
        self.optimizer_guidance_update: dict[str, str] = {}

    def spawn(self, seed: object) -> object:
        self.spawn_instructions.append(seed.instruction)
        workspace = Path(seed.working_directory)
        for rel_path, content in self.optimizer_guidance_update.items():
            path = workspace / "guidance" / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        optimize_dir = workspace / "logs" / "optimize"
        optimize_dir.mkdir(parents=True, exist_ok=True)
        (optimize_dir / "agent-note.txt").write_text("agent log\n", encoding="utf-8")
        return SimpleNamespace(
            summary="phase2 transcript",
            state=SimpleNamespace(),
            usage=SimpleNamespace(model_cost_usd=0.5),
        )

    def resume(self, state: object, instruction: str | None = None) -> object:
        _ = (state, instruction)
        self.resume_calls += 1
        return SimpleNamespace(
            summary=f"repair transcript {self.resume_calls}",
            state=SimpleNamespace(),
            usage=SimpleNamespace(model_cost_usd=0.25),
        )


class _FakePopulation:
    def __init__(self) -> None:
        self.entries: list[PopulationEntry] = []

    def add(self, entry: PopulationEntry) -> None:
        self.entries.append(entry)

    def sample(self, n: int) -> list[PopulationEntry]:
        return self.entries[:n]

    def size(self) -> int:
        return len(self.entries)


def _instantiate_test_instructions(config: DictConfig) -> DictConfig:
    """Instantiate instruction objects in-place on a DictConfig."""
    config.instructions._set_flag("allow_objects", True)
    for field_name in (
        "phase2_readme",
        "phase2_entrypoint",
        "phase2_agent",
        "phase4_readme",
        "phase4_entrypoint",
        "phase4_agent",
    ):
        value = config.instructions[field_name]
        if isinstance(value, (dict, DictConfig)):
            config.instructions[field_name] = instantiate(
                OmegaConf.to_container(value, resolve=True)
                if isinstance(value, DictConfig)
                else dict(value),
                _convert_="all",
            )
    return config


def _instruction_objects(config: DictConfig) -> dict[str, object]:
    return {
        field_name: config.instructions[field_name]
        for field_name in (
            "phase2_readme",
            "phase2_entrypoint",
            "phase2_agent",
            "phase4_readme",
            "phase4_entrypoint",
            "phase4_agent",
        )
    }


def _render_phase2_readme_for_test(
    config: DictConfig,
    *,
    problem: RepoTaskProblem,
    solvers: list[PopulationEntry],
    prefill_solver_id: str,
    optimizer_examples: list[PopulationEntry] | None = None,
) -> str:
    prefill_solver = next(entry for entry in solvers if entry.id == prefill_solver_id)
    workspace_builder = SimpleNamespace(problem=problem, config=config)
    return config.instructions["phase2_readme"].render(
        workspace_builder=workspace_builder,
        solvers=solvers,
        prefill_solver=prefill_solver,
        optimizer_examples=optimizer_examples or [],
    )


def _render_phase4_readme_for_test(
    config: DictConfig,
    *,
    optimizers: list[PopulationEntry],
    prefill_optimizer_id: str,
    lead_optimizer_id: str,
) -> str:
    lead_optimizer = next(entry for entry in optimizers if entry.id == lead_optimizer_id)
    prefill_optimizer = next(entry for entry in optimizers if entry.id == prefill_optimizer_id)
    return config.instructions["phase4_readme"].render(
        lead_optimizer=lead_optimizer,
        optimizers=optimizers,
        prefill_optimizer=prefill_optimizer,
    )


def _score(value: float, *, summary: str | None = None) -> object:
    return {"score": value, "summary": summary or f"score={value}"}


def _optimizer_score(value: float) -> object:
    return {"elo": value}


def _make_problem(tmp_path: Path) -> RepoTaskProblem:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    check_claude = tmp_path / "check_claude.md"
    check_claude.write_text(
        "\n".join(
            [
                "---",
                "name: check-runner",
                'description: "Run the check workflow and report PASS or FAIL."',
                "tools: Bash, Read",
                "---",
                "",
                "Run `python3 -m py_compile candidate.py`.",
                "",
                "```bash",
                "{{BOUNDARY_CHECK_COMMAND}}",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    check_codex = tmp_path / "check_codex.toml"
    check_codex.write_text(
        "\n".join(
            [
                'name = "check-runner"',
                'description = "Run the check workflow and report PASS or FAIL."',
                'developer_instructions = """Run `python3 -m py_compile candidate.py`.',
                "",
                "```bash",
                "{{BOUNDARY_CHECK_COMMAND}}",
                "```",
                '"""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.md"
    evaluation.write_text("# Formal evaluation\n", encoding="utf-8")
    (snapshot / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (snapshot / "README.md").write_text("# repo\n", encoding="utf-8")
    return RepoTaskProblem(
        name="circle-packing",
        github_url="https://github.com/scaling-group/eve",
        commit="abc123def456",
        editable_files=("candidate.py",),
        editable_folders=(),
        check_agent_paths={"claude": check_claude, "codex": check_codex},
        evaluation_steps=(evaluation,),
        local_checkout=snapshot,
        snapshot_root=snapshot,
        boundary_checker_path=tmp_path / "boundary.py",
    )


def _make_loop(
    tmp_path: Path,
    *,
    eval_fn,
    **config_overrides: object,
) -> tuple[Eve, _FakePopulation]:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            max_iterations=1,
            **config_overrides,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    optimizer_workspace_builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    solver_evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=config_overrides.get("seed_solver_score"),
        seed_solver_skip_evaluation=bool(
            config_overrides.get("seed_solver_skip_evaluation", False)
        ),
        evaluation_driver_factory=lambda: object(),
    )
    object.__setattr__(solver_evaluator, "evaluate", eval_fn)
    solver_pop = _FakePopulation()
    optimizer_pop = _FakePopulation()
    loop = Eve(
        solver_pop=solver_pop,  # type: ignore[arg-type]
        optimizer_pop=optimizer_pop,  # type: ignore[arg-type]
        solver_workspace_builder=solver_workspace_builder,
        optimizer_workspace_builder=optimizer_workspace_builder,
        solver_driver=object(),  # type: ignore[arg-type]
        optimizer_driver=object(),  # type: ignore[arg-type]
        solver_evaluator=solver_evaluator,
        config=config,
        instructions=_instruction_objects(config),
        optimizer_evaluator=ScalarEloEvaluator(),
        phase2_optimizer_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_solver_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_optimizer_examples_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_produced_optimizer_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase4_lead_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase4_optimizer_examples_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase4_prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase4_history_log_sampler=UniformSampler(replacement_mode="no_replacement"),
    )
    return loop, solver_pop


def _minimal_instruction_overrides() -> dict[str, dict[str, object]]:
    base = "configs/eve/prompt/templates/built_in"
    _S = "scaling_evolve.algorithms.eve"
    return {
        "phase2_readme": {
            "_target_": f"{_S}.instructions.default.Phase2ReadmeInstruction",
            "file_list": [f"{base}/phase2_readme.md"],
        },
        "phase2_entrypoint": {
            "_target_": f"{_S}.instructions.default.Phase2EntrypointInstruction",
            "file_list": [f"{base}/phase2_entrypoint.md"],
        },
        "phase2_agent": {
            "_target_": f"{_S}.instructions.default.Phase2AgentInstruction",
            "file_list": [f"{base}/workspace_agent.md"],
        },
        "phase4_readme": {
            "_target_": f"{_S}.instructions.default.Phase4ReadmeInstruction",
            "file_list": [f"{base}/phase4_readme.md"],
        },
        "phase4_entrypoint": {
            "_target_": f"{_S}.instructions.default.Phase4EntrypointInstruction",
            "file_list": [f"{base}/phase4_entrypoint.md"],
        },
        "phase4_agent": {
            "_target_": f"{_S}.instructions.default.Phase4AgentInstruction",
            "file_list": [f"{base}/workspace_agent.md"],
        },
    }


def _make_solver_evaluator(problem: RepoTaskProblem, *, eval_fn) -> object:
    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: object(),
    )
    object.__setattr__(evaluator, "evaluate", eval_fn)
    return evaluator


def test_seed_solver_evaluates_by_default(tmp_path: Path) -> None:
    calls: list[Path] = []

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        calls.append(workspace_root)
        return _score(1.25), {"summary.txt": "seed evaluation\n"}

    loop, solver_pop = _make_loop(tmp_path, eval_fn=eval_fn)

    created = loop._ensure_seed_solver()  # noqa: SLF001

    assert created == 1
    assert len(calls) == 1
    assert len(solver_pop.entries) == 1
    assert solver_pop.entries[0].score["score"] == 1.25
    assert "evaluate/summary.txt" in solver_pop.entries[0].logs


def test_seed_solver_score_does_not_bypass_seed_evaluation_by_itself(tmp_path: Path) -> None:
    calls: list[Path] = []

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        calls.append(workspace_root)
        return _score(9.0), {"summary.txt": "eval ran\n"}

    loop, solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        seed_solver_score=3.5,
    )

    created = loop._ensure_seed_solver()  # noqa: SLF001

    assert created == 1
    assert len(calls) == 1
    assert len(solver_pop.entries) == 1
    assert solver_pop.entries[0].score["score"] == 9.0
    assert solver_pop.entries[0].logs["evaluate/summary.txt"] == "eval ran\n"
    assert next((tmp_path / "solver_workspaces").glob("*")).exists()


def test_seed_solver_skip_evaluation_requires_seed_solver_score(tmp_path: Path) -> None:
    calls: list[Path] = []

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        calls.append(workspace_root)
        return _score(9.0), {"summary.txt": "should not run\n"}

    loop, _solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        seed_solver_skip_evaluation=True,
    )

    try:
        loop._ensure_seed_solver()  # noqa: SLF001
    except ValueError as exc:
        assert "application.seed_solver_score is required" in str(exc)
    else:
        raise AssertionError(
            "Expected seed_solver_skip_evaluation without seed_solver_score to fail"
        )


def test_seed_solver_skip_evaluation_uses_seed_solver_score_when_present(tmp_path: Path) -> None:
    calls: list[Path] = []

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        calls.append(workspace_root)
        return _score(9.0), {"summary.txt": "should not run\n"}

    loop, solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        seed_solver_score=3.5,
        seed_solver_skip_evaluation=True,
    )

    created = loop._ensure_seed_solver()  # noqa: SLF001

    assert created == 0
    assert calls == []
    assert len(solver_pop.entries) == 1
    assert solver_pop.entries[0].score == 3.5
    assert solver_pop.entries[0].logs["evaluate/score.yaml"] == "3.5\n...\n"
    assert "application.seed_solver_score" in solver_pop.entries[0].logs["evaluate/summary.txt"]


def test_seed_solver_skip_evaluation_normalizes_mapping_score_configs(tmp_path: Path) -> None:
    calls: list[Path] = []

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        calls.append(workspace_root)
        return _score(9.0), {"summary.txt": "should not run\n"}

    loop, solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        seed_solver_score=OmegaConf.create(
            {
                "score": -0.0887646175810432,
                "summary": "AVP + HiEx 4/12 seed, beta_v2, smoke (MAX_STEPS=50)",
            }
        ),
        seed_solver_skip_evaluation=True,
    )

    created = loop._ensure_seed_solver()  # noqa: SLF001

    assert created == 0
    assert calls == []
    assert len(solver_pop.entries) == 1
    assert solver_pop.entries[0].score == {
        "score": -0.0887646175810432,
        "summary": "AVP + HiEx 4/12 seed, beta_v2, smoke (MAX_STEPS=50)",
    }
    assert "score: -0.0887646175810432" in solver_pop.entries[0].logs["evaluate/score.yaml"]
    assert (
        "summary: AVP + HiEx 4/12 seed, beta_v2, smoke (MAX_STEPS=50)"
        in solver_pop.entries[0].logs["evaluate/score.yaml"]
    )


def test_loop_skips_phase4_when_n_workers_phase4_is_non_positive(
    tmp_path: Path, monkeypatch
) -> None:
    phase4_called = False

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        _ = workspace_root, display_context
        return _score(1.0), {"summary.txt": "ok\n"}

    loop, _solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        instructions=_minimal_instruction_overrides(),
        n_workers_phase4=0,
    )

    class _FakePhase2Result:
        def __init__(self) -> None:
            self.optimizer = PopulationEntry(
                id="optimizer_seed",
                files={"APPROACH.md": "seed\n"},
                score=_optimizer_score(1500.0),
                logs={},
            )

    def _fake_phase2_run(self):  # noqa: ANN001
        _ = self
        return [_FakePhase2Result()]

    def _fake_phase4_run(self):  # noqa: ANN001
        _ = self
        nonlocal phase4_called
        phase4_called = True
        return []

    monkeypatch.setattr(Phase2BatchRunner, "run", _fake_phase2_run)
    monkeypatch.setattr(Phase4BatchRunner, "run", _fake_phase4_run)
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.workflow.loop.score_optimizers",
        lambda **kwargs: None,
    )

    loop.run()

    assert phase4_called is False


def test_loop_runs_phase4_when_n_workers_phase4_is_positive(tmp_path: Path, monkeypatch) -> None:
    phase4_called = False

    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        _ = workspace_root, display_context
        return _score(1.0), {"summary.txt": "ok\n"}

    loop, _solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        instructions=_minimal_instruction_overrides(),
        n_workers_phase4=1,
    )

    class _FakePhase2Result:
        def __init__(self) -> None:
            self.optimizer = PopulationEntry(
                id="optimizer_seed",
                files={"APPROACH.md": "seed\n"},
                score=_optimizer_score(1500.0),
                logs={},
            )

    def _fake_phase2_run(self):  # noqa: ANN001
        _ = self
        return [_FakePhase2Result()]

    def _fake_phase4_run(self):  # noqa: ANN001
        _ = self
        nonlocal phase4_called
        phase4_called = True
        return []

    monkeypatch.setattr(Phase2BatchRunner, "run", _fake_phase2_run)
    monkeypatch.setattr(Phase4BatchRunner, "run", _fake_phase4_run)
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.workflow.loop.score_optimizers",
        lambda **kwargs: None,
    )

    loop.run()

    assert phase4_called is True


def test_loop_writes_wal_safe_iter_snapshots(tmp_path: Path, monkeypatch) -> None:
    def eval_fn(workspace_root: Path, display_context=None):  # noqa: ARG001
        _ = workspace_root, display_context
        return _score(1.0), {"summary.txt": "ok\n"}

    loop, _solver_pop = _make_loop(
        tmp_path,
        eval_fn=eval_fn,
        instructions=_minimal_instruction_overrides(),
        n_workers_phase4=0,
        enable_iter_snapshots=True,
        iter_snapshot_retain=2,
    )
    loop.config.max_iterations = 3

    solver_db_path = Path(str(loop.config.solver_db_path))
    optimizer_db_path = Path(str(loop.config.optimizer_db_path))

    def _init_marker_db(path: Path, *, value: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
            conn.execute("INSERT INTO marker(value) VALUES (?)", (value,))
            conn.commit()
        finally:
            conn.close()

    def _write_marker(path: Path, *, value: int) -> None:
        conn = sqlite3.connect(path)
        try:
            conn.execute("DELETE FROM marker")
            conn.execute("INSERT INTO marker(value) VALUES (?)", (value,))
            conn.commit()
        finally:
            conn.close()

    def _read_marker(path: Path) -> int:
        conn = sqlite3.connect(path)
        try:
            row = conn.execute("SELECT value FROM marker").fetchone()
        finally:
            conn.close()
        assert row is not None
        return int(row[0])

    _init_marker_db(solver_db_path, value=0)
    _init_marker_db(optimizer_db_path, value=100)

    def _fake_phase2_run(self):  # noqa: ANN001
        _write_marker(solver_db_path, value=self.iteration)
        _write_marker(optimizer_db_path, value=100 + self.iteration)
        return []

    monkeypatch.setattr(Phase2BatchRunner, "run", _fake_phase2_run)

    loop.run()

    snapshot_root = Path(str(loop.config.workspace_root)) / ".snapshots"
    solver_snapshots = sorted(snapshot_root.glob("solver_lineage_iter_*.db"))
    optimizer_snapshots = sorted(snapshot_root.glob("optimizer_lineage_iter_*.db"))

    assert [path.name for path in solver_snapshots] == [
        "solver_lineage_iter_1.db",
        "solver_lineage_iter_2.db",
    ]
    assert [path.name for path in optimizer_snapshots] == [
        "optimizer_lineage_iter_1.db",
        "optimizer_lineage_iter_2.db",
    ]
    assert _read_marker(snapshot_root / "solver_lineage_iter_1.db") == 1
    assert _read_marker(snapshot_root / "solver_lineage_iter_2.db") == 2
    assert _read_marker(snapshot_root / "optimizer_lineage_iter_1.db") == 101
    assert _read_marker(snapshot_root / "optimizer_lineage_iter_2.db") == 102


def test_optimize_logs_preserve_agent_files_and_append_token_usage(tmp_path: Path) -> None:
    rollout = SimpleNamespace(
        summary="final summary",
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "model_cost_usd": 0.5,
            "agent_turns": 3,
            "wallclock_seconds": None,
        },
        state=SimpleNamespace(metadata={}),
    )

    from scaling_evolve.algorithms.eve.workflow.optimize_logs import build_optimize_log_tree

    workspace = tmp_path / "workspace"
    (workspace / "logs" / "optimize").mkdir(parents=True)
    (workspace / "logs" / "optimize" / "agent-note.txt").write_text(
        "important note\n", encoding="utf-8"
    )

    logs = build_optimize_log_tree(workspace, [rollout])

    assert logs["agent-note.txt"] == "important note\n"
    assert logs["final_response.txt"] == "final summary\n"
    assert '"agent_turns": 3' in logs["token_usage.json"]
    assert '"input_tokens": 10' in logs["token_usage.json"]
    assert '"attempts"' in logs["token_usage.json"]
    assert '"total"' in logs["token_usage.json"]
    assert '"cache_read_tokens": 0' in logs["token_usage.json"]
    assert "session.md" not in logs


def test_optimize_logs_append_session_markdown_when_transcript_exists(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "\n".join(
            [
                (
                    '{"timestamp":"2026-04-05T10:00:00Z","type":"session_meta","payload":'
                    '{"id":"session-root","cwd":"'
                    "/tmp/optimizer_workspaces/20260405T100000_step_2_abcd"
                    '"}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:00Z","type":"turn_context","payload":'
                    '{"cwd":"'
                    "/tmp/optimizer_workspaces/20260405T100000_step_2_abcd"
                    '","model":"gpt-5.4-mini","effort":"medium"}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:01Z","type":"event_msg","payload":'
                    '{"type":"user_message","message":"Improve the optimizer."}}'
                ),
                (
                    '{"timestamp":"2026-04-05T10:00:02Z","type":"event_msg","payload":'
                    '{"type":"agent_message","message":"Optimizer updated."}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rollout = SimpleNamespace(
        summary="Optimizer updated.",
        usage=SimpleNamespace(input_tokens=7, output_tokens=3, cache_read_tokens=1),
        state=SimpleNamespace(
            session_id="session-root",
            metadata={
                "driver": "codex_tmux",
                "provider_transcript_path": str(transcript),
            },
        ),
    )

    from scaling_evolve.algorithms.eve.workflow.optimize_logs import build_optimize_log_tree

    workspace = tmp_path / "workspace"
    (workspace / "logs" / "optimize").mkdir(parents=True)

    logs = build_optimize_log_tree(workspace, [rollout])

    assert "session.md" in logs
    assert "# Session Log - optimizer (iter 2)" in logs["session.md"]
    assert "Improve the optimizer." in logs["session.md"]
    assert "Optimizer updated." in logs["session.md"]


def test_phase2_workspace_logs_are_direct_and_optimizer_history_uses_iteration_step(
    tmp_path: Path,
) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach", "TASK.md": "task"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={
            "optimize/transcript.txt": "old optimize transcript",
            "evaluate/summary.txt": "seed summary",
        },
    )

    result = Phase2Runner(
        solver_workspace_builder=solver_workspace_builder,
        driver=driver,
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
    ).run_single(
        optimizer=optimizer,
        solvers=[candidate],
        prefill_solver=candidate,
        worker_index=1,
    )

    assert result.produced_solver is not None
    assert "optimize/agent-note.txt" in result.produced_solver.logs
    assert "optimize/final_response.txt" in result.produced_solver.logs
    assert "optimize/token_usage.json" in result.produced_solver.logs
    assert '"attempts"' in result.produced_solver.logs["optimize/token_usage.json"]
    assert "evaluate/summary.txt" in result.produced_solver.logs
    step_root = f"step_3_{result.produced_solver.id}"
    assert f"{step_root}/solver/candidate.py" in result.optimizer_log_tree
    assert f"{step_root}/logs/optimize/agent-note.txt" in result.optimizer_log_tree
    assert f"{step_root}/logs/optimize/token_usage.json" in result.optimizer_log_tree
    assert '"attempts"' in result.optimizer_log_tree[f"{step_root}/logs/optimize/token_usage.json"]
    assert f"{step_root}/logs/evaluate/summary.txt" in result.optimizer_log_tree
    assert f"{step_root}/score.yaml" in result.optimizer_log_tree
    assert "solver_id:" in result.optimizer_log_tree[f"{step_root}/score.yaml"]
    assert driver.spawn_instructions == [
        solver_workspace_builder.instructions["phase2_entrypoint"].render(
            workspace_builder=solver_workspace_builder,
            optimizer=optimizer,
            solvers=[candidate],
            prefill_solver=candidate,
        )
    ]
    workspace = next((tmp_path / "solver_workspaces").glob("*"))
    assert not (workspace / "eve.md").exists()
    assert (workspace / "output" / "candidate.py").exists()
    assert (workspace / "output" / "README.md").exists()
    assert not (workspace / "check.md").exists()
    assert (
        workspace / "solver_examples" / "solver_1" / "logs" / "evaluate" / "summary.txt"
    ).exists()
    assert not (
        workspace / "examples" / "solver_1" / "logs" / "optimize" / "transcript.txt"
    ).exists()
    assert not (workspace / "guidance" / "PROBLEM.md").exists()
    assert not (workspace / "guidance" / "FORMAL_EVALUATION.md").exists()


def test_phase2_can_produce_optimizer_when_configured(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    driver.optimizer_guidance_update = {"APPROACH.md": "updated approach"}
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            produce_optimizer_in_phase2=1,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach", "TASK.md": "task"},
        score=_optimizer_score(1500.0),
        logs={},
    )
    optimizer_example = PopulationEntry(
        id="opt_ref_1",
        files={"APPROACH.md": "reference guidance"},
        score=_optimizer_score(1495.0),
        logs={"optimize/summary.txt": "reference log"},
    )
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )

    result = Phase2Runner(
        solver_workspace_builder=solver_workspace_builder,
        driver=driver,
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
    ).run_single(
        optimizer=optimizer,
        solvers=[candidate],
        optimizer_examples=[optimizer_example],
        prefill_solver=candidate,
        worker_index=1,
    )

    assert result.produced_solver is not None
    assert result.produced_optimizer is not None
    assert result.produced_optimizer.files == {
        "APPROACH.md": "updated approach",
        "TASK.md": "task",
    }
    assert result.produced_optimizer.score == optimizer.score
    assert result.produced_optimizer.logs == result.optimizer_log_tree
    workspace = next((tmp_path / "solver_workspaces").glob("*"))
    assert (workspace / "guidance" / "APPROACH.md").exists()
    manifest = yaml.safe_load((workspace / "score.yaml").read_text(encoding="utf-8"))
    assert manifest["optimizer_id"] == "opt-1"
    assert manifest["sampled_optimizer_example_ids"] == ["opt_ref_1"]
    assert manifest["produced_solver"]["solver_id"] == result.produced_solver.id


def test_phase2_skips_optimizer_candidate_when_guidance_is_unchanged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            produce_optimizer_in_phase2=1,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach", "TASK.md": "task"},
        score=_optimizer_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )

    result = Phase2Runner(
        solver_workspace_builder=solver_workspace_builder,
        driver=_FakeDriver(),
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
    ).run_single(
        optimizer=optimizer,
        solvers=[candidate],
        prefill_solver=candidate,
        worker_index=1,
    )

    assert result.produced_solver is not None
    assert result.produced_optimizer is None
    assert (
        "produce_optimizer_in_phase2 is enabled but the guidance tree was not modified"
        in caplog.text
    )
    assert "no optimizer candidate will be produced for solver solver_" in caplog.text


def test_phase2_batch_adds_configured_optimizer_candidate(tmp_path: Path) -> None:
    class _Population:
        def __init__(self, entries: list[PopulationEntry]) -> None:
            self._entries = list(entries)
            self._rng = random.Random(1)

        def entries(self) -> list[PopulationEntry]:
            return list(self._entries)

        def add(self, entry: PopulationEntry) -> None:
            self._entries.append(entry)

        def update_logs(self, logs_by_id: dict[str, dict[str, str]]) -> None:
            _ = logs_by_id

    class _HeadSampler:
        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = scores
            _ = rng
            return list(entries[:n])

    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    driver.optimizer_guidance_update = {"APPROACH.md": "updated approach"}
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            produce_optimizer_in_phase2=1,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    solver_pop = _Population(
        [
            PopulationEntry(
                id="solver_1",
                files={"candidate.py": "print('seed')\n"},
                score=_score(0.4),
                logs={"evaluate/summary.txt": "seed summary"},
            )
        ]
    )
    optimizer_pop = _Population(
        [
            PopulationEntry(
                id="opt-1",
                files={"APPROACH.md": "approach"},
                score=_optimizer_score(1500.0),
                logs={},
            )
        ]
    )

    results = Phase2BatchRunner(
        solver_workspace_builder=solver_workspace_builder,
        driver=driver,
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
        solver_pop=solver_pop,
        optimizer_pop=optimizer_pop,
        n_workers_phase2=1,
        n_solver_examples_phase2=1,
        n_optimizer_examples_phase2=0,
        n_produced_optimizers_phase2=1,
        optimizer_sampler=_HeadSampler(),
        solver_sampler=_HeadSampler(),
        prefill_sampler=_HeadSampler(),
        optimizer_examples_sampler=_HeadSampler(),
        produced_optimizer_sampler=_HeadSampler(),
    ).run()

    assert len(results) == 1
    assert results[0].produced_optimizer is not None
    assert len(solver_pop.entries()) == 2
    assert len(optimizer_pop.entries()) == 2
    assert optimizer_pop.entries()[1].score == optimizer_pop.entries()[0].score


def test_phase2_workspace_uses_optimizer_examples_when_enabled(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            n_optimizer_examples_phase2=2,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    optimizer = PopulationEntry(
        id="opt_current",
        files={"APPROACH.md": "current guidance"},
        score=_optimizer_score(1500.0),
        logs={},
    )
    solver = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )
    optimizer_example = PopulationEntry(
        id="opt_ref_1",
        files={"APPROACH.md": "reference guidance"},
        score=_optimizer_score(1495.0),
        logs={"optimize/summary.txt": "reference log"},
    )

    workspace, _ = solver_workspace_builder.build(
        optimizer.files,
        [solver],
        workspace_id="workspace_1",
        optimizer=optimizer,
        prefill_solver=solver,
        optimizer_examples=[optimizer_example],
    )

    assert (workspace / "solver_examples" / "solver_1" / "solver" / "candidate.py").exists()
    assert (workspace / "guidance_examples" / "opt_ref_1" / "optimizer" / "APPROACH.md").exists()
    assert (
        workspace / "guidance_examples" / "opt_ref_1" / "logs" / "optimize/summary.txt"
    ).exists()
    assert not (workspace / "examples").exists()
    manifest = yaml.safe_load((workspace / "score.yaml").read_text(encoding="utf-8"))
    assert manifest["optimizer_id"] == "opt_current"
    assert manifest["sampled_optimizer_example_ids"] == ["opt_ref_1"]


def test_phase2_batch_reuses_same_optimizer_examples_for_all_workers(
    tmp_path: Path, monkeypatch
) -> None:
    class _Population:
        def __init__(self, entries: list[PopulationEntry]) -> None:
            self._entries = list(entries)
            self._rng = random.Random(0)

        def entries(self) -> list[PopulationEntry]:
            return list(self._entries)

        def add(self, entry: PopulationEntry) -> None:
            self._entries.append(entry)

        def update_logs(self, logs_by_id: dict[str, dict[str, str]]) -> None:
            _ = logs_by_id

    class _HeadSampler:
        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = scores
            _ = rng
            return list(entries[:n])

    class _RecordingOptimizerExampleSampler:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = scores
            _ = rng
            entry_ids = [entry.id for entry in entries]
            self.calls.append(entry_ids)
            return list(entries[:n])

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            n_workers_phase2=2,
            n_solver_examples_phase2=1,
            n_optimizer_examples_phase2=2,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    solver_pop = _Population(
        [
            PopulationEntry(
                id="solver_1",
                files={"candidate.py": "print('seed')\n"},
                score=_score(0.4),
                logs={"evaluate/summary.txt": "seed summary"},
            )
        ]
    )
    optimizer_pop = _Population(
        [
            PopulationEntry(
                id="opt_1",
                files={"APPROACH.md": "opt1"},
                score=_optimizer_score(1500.0),
                logs={},
            ),
            PopulationEntry(
                id="opt_2",
                files={"APPROACH.md": "opt2"},
                score=_optimizer_score(1490.0),
                logs={},
            ),
            PopulationEntry(
                id="opt_3",
                files={"APPROACH.md": "opt3"},
                score=_optimizer_score(1480.0),
                logs={},
            ),
            PopulationEntry(
                id="opt_4",
                files={"APPROACH.md": "opt4"},
                score=_optimizer_score(1470.0),
                logs={},
            ),
        ]
    )
    optimizer_examples_seen: list[list[str]] = []

    def _fake_run_single(
        self, *, optimizer, solvers, optimizer_examples, prefill_solver, worker_index
    ):
        _ = (self, solvers, prefill_solver, worker_index)
        optimizer_examples_seen.append([entry.id for entry in optimizer_examples])
        return Phase2Result(optimizer=optimizer)

    optimizer_examples_sampler = _RecordingOptimizerExampleSampler()
    monkeypatch.setattr(Phase2Runner, "run_single", _fake_run_single)

    Phase2BatchRunner(
        solver_workspace_builder=solver_workspace_builder,
        driver=_FakeDriver(),
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
        solver_pop=solver_pop,
        optimizer_pop=optimizer_pop,
        n_workers_phase2=2,
        n_solver_examples_phase2=1,
        n_optimizer_examples_phase2=2,
        n_produced_optimizers_phase2=0,
        optimizer_sampler=_HeadSampler(),
        solver_sampler=_HeadSampler(),
        prefill_sampler=_HeadSampler(),
        optimizer_examples_sampler=optimizer_examples_sampler,
        produced_optimizer_sampler=_HeadSampler(),
    ).run()

    assert optimizer_examples_sampler.calls == [["opt_3", "opt_4"]]
    assert optimizer_examples_seen == [["opt_1", "opt_3"], ["opt_2", "opt_3"]]


def test_phase2_batch_samples_produced_optimizers_when_configured(
    tmp_path: Path, monkeypatch
) -> None:
    class _Population:
        def __init__(self, entries: list[PopulationEntry]) -> None:
            self._entries = list(entries)
            self._rng = random.Random(0)

        def entries(self) -> list[PopulationEntry]:
            return list(self._entries)

        def add(self, entry: PopulationEntry) -> None:
            self._entries.append(entry)

        def update_logs(self, logs_by_id: dict[str, dict[str, str]]) -> None:
            _ = logs_by_id

    class _HeadSampler:
        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = scores
            _ = rng
            return list(entries[:n])

    class _ProducedOptimizerSampler:
        def __init__(self) -> None:
            self.calls: list[list[float]] = []

        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = entries
            _ = rng
            self.calls.append([float(score["score"]) for score in scores])
            ranked = sorted(
                zip(entries, scores, strict=True),
                key=lambda item: float(item[1]["score"]),
                reverse=True,
            )
            return [entry for entry, _score in ranked[:n]]

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            n_workers_phase2=2,
            n_solver_examples_phase2=1,
            n_optimizer_examples_phase2=0,
            produce_optimizer_in_phase2=1,
        )
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    solver_pop = _Population(
        [
            PopulationEntry(
                id="solver_1",
                files={"candidate.py": "print('seed')\n"},
                score=_score(0.4),
                logs={"evaluate/summary.txt": "seed summary"},
            )
        ]
    )
    optimizer_pop = _Population(
        [
            PopulationEntry(
                id="opt_1",
                files={"APPROACH.md": "opt1"},
                score=_optimizer_score(1500.0),
                logs={},
            ),
            PopulationEntry(
                id="opt_2",
                files={"APPROACH.md": "opt2"},
                score=_optimizer_score(1490.0),
                logs={},
            ),
        ]
    )

    def _fake_run_single(
        self, *, optimizer, solvers, optimizer_examples, prefill_solver, worker_index
    ):
        _ = (self, solvers, optimizer_examples, prefill_solver)
        return Phase2Result(
            optimizer=optimizer,
            produced_solver=PopulationEntry(
                id=f"solver_new_{worker_index}",
                files={"candidate.py": f"print({worker_index})\n"},
                score=_score(float(worker_index)),
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id=f"optimizer_new_{worker_index}",
                files={"APPROACH.md": f"opt{worker_index}"},
                score=_optimizer_score(1500.0),
                logs={},
            ),
        )

    produced_optimizer_sampler = _ProducedOptimizerSampler()
    monkeypatch.setattr(Phase2Runner, "run_single", _fake_run_single)

    results = Phase2BatchRunner(
        solver_workspace_builder=solver_workspace_builder,
        driver=_FakeDriver(),
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
        solver_pop=solver_pop,
        optimizer_pop=optimizer_pop,
        n_workers_phase2=2,
        n_solver_examples_phase2=1,
        n_optimizer_examples_phase2=0,
        n_produced_optimizers_phase2=1,
        optimizer_sampler=_HeadSampler(),
        solver_sampler=_HeadSampler(),
        prefill_sampler=_HeadSampler(),
        optimizer_examples_sampler=_HeadSampler(),
        produced_optimizer_sampler=produced_optimizer_sampler,
    ).run()

    assert [sorted(call) for call in produced_optimizer_sampler.calls] == [[1.0, 2.0]]
    kept_optimizer_ids = sorted(
        result.produced_optimizer.id for result in results if result.produced_optimizer is not None
    )
    assert kept_optimizer_ids == ["optimizer_new_2"]
    assert [entry.id for entry in optimizer_pop.entries()] == ["opt_1", "opt_2", "optimizer_new_2"]


def test_phase3_syncs_phase2_optimizer_score_to_updated_parent_score() -> None:
    class _OptimizerPopulation:
        def __init__(self, entries: list[PopulationEntry]) -> None:
            self._entries = {entry.id: entry for entry in entries}

        def update_scores(self, updated_scores: dict[str, object]) -> None:
            for entry_id, score in updated_scores.items():
                entry = self._entries[entry_id]
                self._entries[entry_id] = PopulationEntry(
                    id=entry.id,
                    files=entry.files,
                    score=score,
                    logs=entry.logs,
                )

        def get(self, entry_id: str) -> PopulationEntry:
            return self._entries[entry_id]

    parent_a = PopulationEntry(
        id="optimizer_a",
        files={"APPROACH.md": "a"},
        score={"elo": 1500.0},
        logs={},
    )
    parent_b = PopulationEntry(
        id="optimizer_b",
        files={"APPROACH.md": "b"},
        score={"elo": 1500.0},
        logs={},
    )
    produced_a = PopulationEntry(
        id="optimizer_new_a",
        files={"APPROACH.md": "a2"},
        score={"elo": 1500.0},
        logs={},
    )
    produced_b = PopulationEntry(
        id="optimizer_new_b",
        files={"APPROACH.md": "b2"},
        score={"elo": 1500.0},
        logs={},
    )
    phase2_results = [
        Phase2Result(
            optimizer=parent_a,
            produced_solver=PopulationEntry(
                id="solver_a",
                files={"candidate.py": "a"},
                score=_score(1.0),
                logs={},
            ),
            produced_optimizer=produced_a,
        ),
        Phase2Result(
            optimizer=parent_b,
            produced_solver=PopulationEntry(
                id="solver_b",
                files={"candidate.py": "b"},
                score=_score(0.0),
                logs={},
            ),
            produced_optimizer=produced_b,
        ),
    ]
    optimizer_pop = _OptimizerPopulation([parent_a, parent_b, produced_a, produced_b])

    score_optimizers(
        optimizers=[parent_a, parent_b],
        phase2_results=phase2_results,
        optimizer_pop=optimizer_pop,
        optimizer_evaluator=ScalarEloEvaluator(k_factor=32.0),
    )

    assert optimizer_pop.get("optimizer_a").score == {"elo": 1516.0}
    assert optimizer_pop.get("optimizer_new_a").score == {"elo": 1516.0}
    assert phase2_results[0].produced_optimizer is not None
    assert phase2_results[0].produced_optimizer.score == {"elo": 1516.0}
    assert optimizer_pop.get("optimizer_b").score == {"elo": 1484.0}
    assert optimizer_pop.get("optimizer_new_b").score == {"elo": 1484.0}


def test_phase2_system_prompt_omits_removed_important_message(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )

    Phase2Runner(
        solver_workspace_builder=solver_workspace_builder,
        driver=driver,
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
    ).run_single(
        optimizer=optimizer,
        solvers=[candidate],
        prefill_solver=candidate,
        worker_index=1,
    )

    assert len(driver.spawn_instructions) == 1
    assert "Read `README.md` first and follow it." in driver.spawn_instructions[0]
    assert "VERY IMPORTANT RULE" not in driver.spawn_instructions[0]


def test_phase2_runner_writes_rollout_prompt_specs_into_workspace(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    driver.rollout_max_turns = 12
    driver.budget_prompt = True
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
        rollout_prompts={"budget": BudgetPrompt()},
    )
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )

    Phase2Runner(
        solver_workspace_builder=solver_workspace_builder,
        driver=driver,
        solver_evaluator=_make_solver_evaluator(
            problem,
            eval_fn=lambda files, display_context=None: (
                _score(1.0),
                {"summary.txt": "evaluation summary"},
            ),
        ),
        step_label="step_3",
        iteration=3,
    ).run_single(
        optimizer=optimizer,
        solvers=[candidate],
        prefill_solver=candidate,
        worker_index=1,
    )

    workspace = next((tmp_path / "solver_workspaces").glob("*"))
    rollout_payload = json.loads(
        (workspace / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert rollout_payload["version"] == 2
    assert rollout_payload["prompts"] == [
        {
            "name": "budget",
            "system_text": None,
            "user_text": (
                "Turn budget enabled: this session has 12 turns per rollout. "
                "After each turn you will see `[Budget] N/12 turns remaining`. "
                "Use that signal to pace your work - the current rollout will be terminated "
                "when the budget runs out."
            ),
            "turn_template": "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining",
            "turn_format_kwargs": {"rollout_max_turns": 12},
        }
    ]


def test_solver_workspace_exposes_context_skills_via_root_links(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={},
    )

    workspace, _ = builder.build(
        {"skills/read-eval/SKILL.md": "skill body\n"},
        [candidate],
        "workspace",
    )

    assert (workspace / "guidance" / "skills" / "read-eval" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "skill body\n"
    assert not (workspace / "skills").exists()
    assert (workspace / ".claude" / "skills").is_symlink()
    assert (workspace / ".codex" / "skills").is_symlink()
    assert (workspace / ".claude" / "skills" / "read-eval" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "skill body\n"
    assert (workspace / ".codex" / "skills" / "read-eval" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "skill body\n"
    assert (workspace / ".claude" / "agents" / "check-runner.md").exists()
    assert (workspace / ".codex" / "agents" / "check-runner.toml").exists()
    assert (workspace / ".claude" / "settings.local.json").exists()
    assert "python3 -m py_compile candidate.py" in (
        workspace / ".claude" / "agents" / "check-runner.md"
    ).read_text(encoding="utf-8")
    assert ".claude-task-stopped" in (workspace / ".claude" / "settings.local.json").read_text(
        encoding="utf-8"
    )
    assert (
        "python3 -m py_compile candidate.py"
        in (
            tomllib.loads(
                (workspace / ".codex" / "agents" / "check-runner.toml").read_text(encoding="utf-8")
            )["developer_instructions"]
        )
    )


def test_optimizer_workspace_exposes_context_skills_via_root_links(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    lead = PopulationEntry(
        id="optimizer_0",
        files={"skills/read-eval/SKILL.md": "skill body\n"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="optimizer_1",
        files={"read-eval.md": "candidate\n"},
        score=_score(1490.0),
        logs={},
    )

    workspace, _ = builder.build(lead, [candidate], "workspace")

    assert (workspace / "guidance" / "skills" / "read-eval" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "skill body\n"
    assert not (workspace / "skills").exists()
    assert (workspace / ".claude" / "skills").is_symlink()
    assert (workspace / ".codex" / "skills").is_symlink()
    assert (workspace / ".claude" / "settings.local.json").exists()
    assert ".claude-task-stopped" in (workspace / ".claude" / "settings.local.json").read_text(
        encoding="utf-8"
    )
    assert (workspace / "task_base" / "candidate.py").exists()


def test_optimizer_workspace_samples_solver_history_by_score_and_hides_step_ids(
    tmp_path: Path,
) -> None:
    problem = _make_problem(tmp_path)

    class _TopRankOnlyRandom:
        def choice(self, seq):  # noqa: ANN001
            return seq[0]

        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            _ = weights
            return [population[0]]

    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    lead = PopulationEntry(
        id="optimizer_0",
        files={"read-eval.md": "base\n"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="optimizer_1",
        files={"read-eval.md": "candidate\n"},
        score=_score(1490.0),
        logs={
            "step_1_solver_a/solver/candidate.py": "v1\n",
            "step_1_solver_a/score.yaml": (
                "solver_id: solver_a\nscore:\n  score: 9.0\n  summary: score=9.0\n"
            ),
            "step_2_solver_b/score.yaml": (
                "solver_id: solver_b\nscore:\n  score: 1.0\n  summary: score=1.0\n"
            ),
            "step_2_solver_b/solver/candidate.py": "v2\n",
            "step_3_solver_c/score.yaml": (
                "solver_id: solver_c\nscore:\n  score: 8.0\n  summary: score=8.0\n"
            ),
            "step_3_solver_c/solver/candidate.py": "v3\n",
            "notes.txt": "keep\n",
        },
    )

    batch_runner = Phase4BatchRunner(
        optimizer_workspace_builder=builder,
        driver=object(),  # type: ignore[arg-type]
        step_label="step_1",
        iteration=1,
        optimizer_pop=SimpleNamespace(_rng=_TopRankOnlyRandom()),  # type: ignore[arg-type]
        lead_sampler=UniformSampler(replacement_mode="no_replacement"),
        optimizer_examples_sampler=UniformSampler(replacement_mode="no_replacement"),
        prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        history_log_sampler=RankSoftmaxSampler(
            temperature=1.0,
            replacement_mode="no_replacement",
        ),
        n_workers_phase4=1,
        n_optimizer_examples_phase4=1,
        n_latest_solver_logs=2,
    )
    workspace, _ = builder.build(
        lead,
        [candidate],
        "workspace",
        example_logs_by_optimizer=batch_runner._sample_example_logs([candidate]),  # noqa: SLF001
    )

    logs_root = workspace / "examples" / "optimizer_1" / "logs"
    assert (logs_root / "notes.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (logs_root / "step_1_solver_a").exists()
    assert not (logs_root / "step_2_solver_b").exists()
    assert not (logs_root / "step_3_solver_c").exists()
    selected_dirs = sorted(path.name for path in logs_root.iterdir() if path.is_dir())
    assert len(selected_dirs) == 2
    assert all(name.startswith("solver_") for name in selected_dirs)
    selected_solver_ids = {
        yaml.safe_load((logs_root / name / "score.yaml").read_text(encoding="utf-8"))["solver_id"]
        for name in selected_dirs
    }
    assert selected_solver_ids == {"solver_a", "solver_c"}
    manifest = yaml.safe_load((workspace / "score.yaml").read_text(encoding="utf-8"))
    assert manifest["sampled_solver_history_ids"] == ["solver_a", "solver_c"]


def test_optimizer_workspace_score_manifest_records_sampled_solver_history_ids(
    tmp_path: Path,
) -> None:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    workspace = tmp_path / "optimizer_workspaces" / "workspace"
    lead = PopulationEntry(
        id="opt-1",
        files={"read-eval.md": "guidance"},
        score=_score(1500.0),
        logs={},
    )
    sampled = PopulationEntry(
        id="opt-2",
        files={"read-eval.md": "sampled"},
        score=_score(1490.0),
        logs={},
    )
    produced = PopulationEntry(
        id="opt-3",
        files={"read-eval.md": "produced"},
        score=_score(1490.0),
        logs={},
    )

    builder.write_score_manifest(
        workspace,
        sampled_optimizers=[sampled],
        sampled_solver_history_ids=["solver_a", "solver_c"],
        lead_optimizer=lead,
        prefill_optimizer=sampled,
        produced_optimizer=produced,
    )

    manifest = yaml.safe_load((workspace / "score.yaml").read_text(encoding="utf-8"))
    assert manifest["sampled_solver_history_ids"] == ["solver_a", "solver_c"]


def test_optimizer_workspace_feature_mode_sampling_keeps_opaque_score_labels(
    tmp_path: Path,
) -> None:
    problem = _make_problem(tmp_path)

    class _TopRankOnlyRandom:
        def choice(self, seq):  # noqa: ANN001
            return seq[0]

        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            _ = weights
            return [population[0]]

    config = _make_test_config(
        workspace_root=tmp_path / "run",
    )
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    lead = PopulationEntry(
        id="optimizer_0",
        files={"read-eval.md": "base\n"},
        score={"score": 1500.0, "quality": 1500.0, "speed": 1500.0},
        logs={},
    )
    candidate = PopulationEntry(
        id="optimizer_1",
        files={"read-eval.md": "candidate\n"},
        score={"score": 1490.0, "quality": 1490.0, "speed": 1490.0},
        logs={
            "step_1_solver_a/score.yaml": (
                "solver_id: solver_a\n"
                "score:\n"
                "  score: 99.0\n"
                "  quality: 1.0\n"
                "  speed: 2.0\n"
                "  summary: a\n"
            ),
            "step_1_solver_a/logs/evaluate/score.yaml": (
                "score: 99.0\nquality: 1.0\nspeed: 2.0\nsummary: a\n"
            ),
            "step_1_solver_a/solver/candidate.py": "a\n",
            "step_2_solver_b/score.yaml": (
                "solver_id: solver_b\n"
                "score:\n"
                "  score: 0.0\n"
                "  quality: 8.0\n"
                "  speed: 10.0\n"
                "  summary: b\n"
            ),
            "step_2_solver_b/logs/evaluate/score.yaml": (
                "score: 0.0\nquality: 8.0\nspeed: 10.0\nsummary: b\n"
            ),
            "step_2_solver_b/solver/candidate.py": "b\n",
        },
    )

    batch_runner = Phase4BatchRunner(
        optimizer_workspace_builder=builder,
        driver=object(),  # type: ignore[arg-type]
        step_label="step_1",
        iteration=1,
        optimizer_pop=SimpleNamespace(_rng=_TopRankOnlyRandom()),  # type: ignore[arg-type]
        lead_sampler=UniformSampler(replacement_mode="no_replacement"),
        optimizer_examples_sampler=UniformSampler(replacement_mode="no_replacement"),
        prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        history_log_sampler=RankSoftmaxSampler(
            temperature=1.0,
            replacement_mode="no_replacement",
        ),
        n_workers_phase4=1,
        n_optimizer_examples_phase4=1,
        n_latest_solver_logs=1,
    )
    workspace, _ = builder.build(
        lead,
        [candidate],
        "workspace",
        example_logs_by_optimizer=batch_runner._sample_example_logs([candidate]),  # noqa: SLF001
    )
    logs_root = workspace / "examples" / "optimizer_1" / "logs"
    selected_dirs = [path.name for path in logs_root.iterdir() if path.is_dir()]
    assert len(selected_dirs) == 1
    assert selected_dirs[0].startswith("solver_")


def test_phase4_parallel_runs_multiple_optimizer_updates(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            n_workers_phase4=2,
            n_optimizer_examples_phase4=1,
        )
    )
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )

    class _Phase4Driver:
        def spawn(self, seed: object) -> object:
            workspace = Path(seed.working_directory)
            optimize_dir = workspace / "logs" / "optimize"
            optimize_dir.mkdir(parents=True, exist_ok=True)
            (optimize_dir / "agent-note.txt").write_text("optimizer log\n", encoding="utf-8")
            return SimpleNamespace(
                summary="phase4 transcript",
                state=SimpleNamespace(),
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=4,
                    cache_read_tokens=2,
                    cache_creation_tokens=0,
                    agent_turns=1,
                    model_cost_usd=0.3,
                    wallclock_seconds=3.0,
                ),
            )

    class _OptimizerPopulation:
        def __init__(self, entries: list[PopulationEntry]) -> None:
            self._entries = list(entries)
            self._rng = random.Random(0)

        def entries(self) -> list[PopulationEntry]:
            return list(self._entries)

        def add(self, entry: PopulationEntry) -> None:
            self._entries.append(entry)

    class _HeadSampler:
        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = scores
            _ = rng
            return list(entries[:n])

    optimizer_entries = [
        PopulationEntry(
            id="optimizer_1",
            files={"APPROACH.md": "one\n"},
            score={"elo": 1500.0},
            logs={},
        ),
        PopulationEntry(
            id="optimizer_2",
            files={"APPROACH.md": "two\n"},
            score={"elo": 1490.0},
            logs={},
        ),
        PopulationEntry(
            id="optimizer_3",
            files={"APPROACH.md": "three\n"},
            score={"elo": 1480.0},
            logs={},
        ),
    ]
    optimizer_pop = _OptimizerPopulation(optimizer_entries)
    phase4_results = Phase4BatchRunner(
        optimizer_workspace_builder=builder,
        driver=_Phase4Driver(),  # type: ignore[arg-type]
        step_label="step_1",
        iteration=1,
        optimizer_pop=optimizer_pop,
        lead_sampler=_HeadSampler(),
        optimizer_examples_sampler=_HeadSampler(),
        prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        history_log_sampler=RankSoftmaxSampler(
            temperature=1.0,
            replacement_mode="no_replacement",
        ),
        n_workers_phase4=2,
        n_optimizer_examples_phase4=1,
        n_latest_solver_logs=8,
    ).run()

    assert len(phase4_results) == 2
    assert [result.lead_optimizer.id for result in phase4_results] == ["optimizer_1", "optimizer_2"]
    assert all(result.produced_optimizer is not None for result in phase4_results)
    assert all(result.rollouts for result in phase4_results)

    assert len(optimizer_pop.entries()) == 5
    produced_ids = [
        result.produced_optimizer.id
        for result in phase4_results
        if result.produced_optimizer is not None
    ]
    assert set(produced_ids).issubset({entry.id for entry in optimizer_pop.entries()})


def test_phase4_system_prompt_omits_removed_important_message(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    lead = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "lead"},
        score=_score(1500.0),
        logs={},
    )
    sampled = PopulationEntry(
        id="opt-2",
        files={"APPROACH.md": "sample"},
        score=_score(1490.0),
        logs={},
    )

    Phase4Runner(
        optimizer_workspace_builder=builder,
        driver=driver,
        step_label="step_4",
        iteration=1,
    ).run_single(
        lead=lead,
        sampled_optimizers=[sampled],
        prefill_optimizer=sampled,
        example_logs_by_optimizer={},
        worker_index=1,
    )

    assert len(driver.spawn_instructions) == 1
    assert "Read `README.md` first and follow it." in driver.spawn_instructions[0]
    assert "VERY IMPORTANT RULE" not in driver.spawn_instructions[0]


def test_readme_renders_current_score_shape(tmp_path: Path) -> None:
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score={"score": 0.5, "quality": 8.0, "speed": 10.0},
        logs={},
    )

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
        )
    )
    _ = optimizer
    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "score:" in instruction
    assert "Their score cards are shown below:" in instruction
    assert "quality: 8.0" in instruction
    assert "speed: 10.0" in instruction


def test_solver_workspace_builder_writes_workspace_agent_instruction_files(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            instructions=_minimal_instruction_overrides(),
        )
    )
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    builder.write_workspace_agent_instructions(workspace, "# Workspace Agent Instructions\n")

    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == (
        "# Workspace Agent Instructions\n"
    )
    assert (workspace / "CLAUDE.md").read_text(encoding="utf-8") == (
        "# Workspace Agent Instructions\n"
    )


def test_scalar_elo_uses_numeric_score_field() -> None:
    current = {
        "optimizer_a": {"score": 1500.0, "summary": "a"},
        "optimizer_b": {"score": 1500.0, "summary": "b"},
    }
    task_scores = {
        "optimizer_a": {"score": 1.0, "summary": "a task"},
        "optimizer_b": {"score": 0.0, "summary": "b task"},
    }

    updated = ScalarEloEvaluator(k_factor=32.0).update(
        current,
        task_scores,
    )

    assert updated["optimizer_a"] == {"elo": 1516.0}
    assert updated["optimizer_b"] == {"elo": 1484.0}


def test_vector_elo_uses_nested_k_factor_tree() -> None:
    current = {
        "optimizer_a": {"elo": 1500.0},
        "optimizer_b": {"elo": 1500.0},
    }
    task_scores = {
        "optimizer_a": {
            "server_score": {
                "quality": 1.0,
                "speed": 1.0,
            }
        },
        "optimizer_b": {
            "server_score": {
                "quality": 0.0,
                "speed": 0.0,
            }
        },
    }

    updated = VectorEloEvaluator(
        k_factors={
            "server_score": {
                "quality": 32.0,
                "speed": 16.0,
            }
        }
    ).update(
        current,
        task_scores,
    )

    assert updated["optimizer_a"] == {"elo": 1524.0}
    assert updated["optimizer_b"] == {"elo": 1476.0}


def test_rank_exponential_sum_uses_nested_weight_and_temperature_trees() -> None:
    items = ["a", "b", "c"]
    scores = [
        {"solver_score": {"quality": 9.0, "speed": 1.0}},
        {"solver_score": {"quality": 3.0, "speed": 8.0}},
        {"solver_score": {"quality": 1.0, "speed": 0.0}},
    ]

    class _TopWeightRandom:
        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            best_index = max(range(len(weights)), key=weights.__getitem__)
            return [population[best_index]]

    selected = RankExponentialSumSampler(
        features={
            "solver_score": {
                "quality": {"weight": 2.0, "temperature": 1.0},
                "speed": {"weight": 1.0, "temperature": 1.0},
            }
        },
        replacement_mode="no_replacement",
    ).sample(
        items,
        scores,
        1,
        rng=_TopWeightRandom(),  # type: ignore[arg-type]
    )

    assert selected == ["a"]


def test_rank_softmax_accepts_optimizer_elo_scores() -> None:
    items = ["a", "b"]
    scores = [{"elo": 1500.0}, {"elo": 1520.0}]

    class _TopWeightRandom:
        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            best_index = max(range(len(weights)), key=weights.__getitem__)
            return [population[best_index]]

    selected = RankSoftmaxSampler(temperature=1.0, replacement_mode="no_replacement").sample(
        items,
        scores,
        1,
        rng=_TopWeightRandom(),  # type: ignore[arg-type]
    )

    assert selected == ["b"]


def test_uniform_sampler_auto_fills_after_unique_pass() -> None:
    class _DeterministicRandom:
        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            return [population[0]] * k

    selected = UniformSampler(replacement_mode="auto").sample(
        ["a", "b"],
        [],
        4,
        rng=_DeterministicRandom(),  # type: ignore[arg-type]
    )

    assert selected == ["a", "b", "a", "a"]


def test_rank_softmax_auto_fills_after_unique_pass() -> None:
    class _TopWeightRandom:
        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            best_index = max(range(len(weights)), key=weights.__getitem__)
            return [population[best_index]] * k

    selected = RankSoftmaxSampler(
        temperature=1.0,
        replacement_mode="auto",
    ).sample(
        ["a", "b"],
        [{"elo": 1510.0}, {"elo": 1500.0}],
        4,
        rng=_TopWeightRandom(),  # type: ignore[arg-type]
    )

    assert selected == ["a", "b", "a", "a"]


def test_boundary_check_detects_forbidden_non_editable_changes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (baseline / "README.md").write_text("# repo\n", encoding="utf-8")
    (candidate / "candidate.py").write_text("print('child')\n", encoding="utf-8")
    (candidate / "README.md").write_text("# changed\n", encoding="utf-8")

    result = check_workspace_boundary(
        baseline_root=baseline,
        candidate_root=candidate,
        editable={"files": ("candidate.py",), "folders": ()},
    )

    assert not result.ok
    assert result.forbidden_modified == ("README.md",)


def test_boundary_check_ignores_gitignored_generated_files(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (candidate / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (baseline / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (candidate / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (candidate / "__pycache__").mkdir()
    (candidate / "__pycache__" / "candidate.cpython-313.pyc").write_bytes(b"compiled")

    result = check_workspace_boundary(
        baseline_root=baseline,
        candidate_root=candidate,
        editable={"files": ("candidate.py",), "folders": ()},
    )

    assert result.ok


def test_boundary_check_ignores_auto_generated_workspace_skills(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (candidate / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (candidate / ".agents" / "skills" / "workflow").mkdir(parents=True)
    (candidate / ".claude" / "skills" / "workflow").mkdir(parents=True)
    (candidate / ".agents" / "skills" / "workflow" / "SKILL.md").write_text(
        "generated\n", encoding="utf-8"
    )
    (candidate / ".claude" / "skills" / "workflow" / "SKILL.md").write_text(
        "generated\n", encoding="utf-8"
    )

    result = check_workspace_boundary(
        baseline_root=baseline,
        candidate_root=candidate,
        editable={"files": ("candidate.py",), "folders": ()},
    )

    assert result.ok


def test_boundary_check_still_protects_gitignore_itself(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (candidate / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (baseline / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (candidate / "candidate.py").write_text("print('seed')\n", encoding="utf-8")

    result = check_workspace_boundary(
        baseline_root=baseline,
        candidate_root=candidate,
        editable={"files": ("candidate.py",), "folders": ()},
    )

    assert not result.ok
    assert result.forbidden_modified == (".gitignore",)


def test_task_context_tells_agent_to_use_boundary_check_during_editing(tmp_path: Path) -> None:
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    _ = optimizer
    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "Current phase: Phase 2 solver optimization." in instruction
    assert "improved solver candidate" in instruction
    assert "Editable files:" in instruction
    assert "invoke the predefined `check-runner` sub-agent" in instruction
    assert "repository root in `output/`" in instruction


def test_solver_readme_includes_configured_score_explanation(tmp_path: Path) -> None:
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
        )
    )
    _ = optimizer
    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "## Score Semantics" in instruction
    assert "Higher solver score is better." in instruction


def test_solver_readme_omits_removed_phase2_important_message(tmp_path: Path) -> None:
    optimizer = PopulationEntry(
        id="opt-1",
        files={"APPROACH.md": "approach"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    _ = optimizer
    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "VERY IMPORTANT RULE" not in instruction
    assert "Don't keep improving!" not in instruction
    assert "Do not ask the human for clarification, approval, or feedback" in instruction


def test_phase2_self_optimize_readme_allows_output_and_guidance_edits(
    tmp_path: Path,
) -> None:
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            instructions={
                "phase2_readme": {
                    "_target_": (
                        "scaling_evolve.algorithms.eve.instructions.icon.Phase2ReadmeInstruction"
                    ),
                    "file_list": [
                        "configs/eve/prompt/templates/built_in/design_doc.md",
                        "configs/eve/prompt/templates/icon/phase2_self_optimize_readme.md",
                        "configs/eve/prompt/templates/built_in/phase2_score_semantics.md",
                    ],
                },
            },
        )
    )
    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "Current phase: Phase 2 solver and optimizer optimization." in instruction
    assert "improved solver candidate in `output/`" in instruction
    assert "improve the files in `guidance/`" in instruction
    assert "modify optimizer files inside `guidance/`" in instruction
    assert "The optimizer's guidance lives in `guidance/docs/`." in instruction
    assert "Do not edit files outside `output/`, `guidance/`, and `logs/optimize/`." in instruction
    assert (
        "Reference solver examples are in `solver_examples/` when optimizer examples are enabled, "
        "otherwise in `examples/`." in instruction
    )


def test_indexed_phase2_readme_uses_worker_index_modulo(tmp_path: Path) -> None:
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )
    base = tmp_path / "instructions"
    base.mkdir()
    (base / "variant_a.md").write_text("Variant A\n\n{solver_examples_block}\n", encoding="utf-8")
    (base / "variant_b.md").write_text("Variant B\n\n{solver_examples_block}\n", encoding="utf-8")

    problem = _make_problem(tmp_path)
    instruction = instantiate(
        {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.indexed.IndexedPhase2ReadmeInstruction"
            ),
            "file_lists": [
                [str(base / "variant_a.md")],
                [str(base / "variant_b.md")],
            ],
        }
    )
    workspace_builder = SimpleNamespace(
        problem=problem,
        config=SimpleNamespace(n_optimizer_examples_phase2=0),
        worker_index=0,
    )

    first_instruction = instruction.render(
        workspace_builder=workspace_builder,
        solvers=[candidate],
        prefill_solver=candidate,
    )
    workspace_builder.worker_index = 1
    second_instruction = instruction.render(
        workspace_builder=workspace_builder,
        solvers=[candidate],
        prefill_solver=candidate,
    )
    workspace_builder.worker_index = 3
    wrapped_instruction = instruction.render(
        workspace_builder=workspace_builder,
        solvers=[candidate],
        prefill_solver=candidate,
    )

    assert "Variant A" in first_instruction
    assert "Variant B" in second_instruction
    assert "Variant B" in wrapped_instruction


def test_optimizer_readme_says_output_guides_phase2_not_phase4() -> None:
    lead = PopulationEntry(
        id="opt-1",
        files={"read-eval.md": "guidance"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="opt-2",
        files={"read-eval.md": "guidance"},
        score=_score(1490.0),
        logs={},
    )

    config = _instantiate_test_instructions(_make_test_config(workspace_root=Path("run")))
    _ = (lead, "empty")
    instruction = _render_phase4_readme_for_test(
        config,
        optimizers=[lead, candidate],
        lead_optimizer_id="opt-1",
        prefill_optimizer_id="opt-2",
    )

    assert "they will later be copied into Phase 2 solver workspaces" in instruction
    assert "solver-optimization agent" in instruction
    assert "copied into the Phase 2 `guidance/` folder" in instruction
    assert "Use the reference optimizers and the current `output/` files together" in instruction
    assert "Write any important optimization notes" in instruction
    assert "save your final response there" in instruction
    assert "## Skills Convention" in instruction
    assert "Recommended `SKILL.md` shape:" in instruction
    assert 'description: "<one-line summary>"' in instruction
    assert "output/\n└── skills/" in instruction
    assert "Treat `guidance/` as supporting guidance, if it exists" in instruction


def test_optimizer_readme_includes_configured_score_explanation() -> None:
    lead = PopulationEntry(
        id="opt-1",
        files={"read-eval.md": "guidance"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="opt-2",
        files={"read-eval.md": "guidance"},
        score=_score(1490.0),
        logs={},
    )

    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=Path("run"),
        )
    )
    _ = (lead, "empty")
    instruction = _render_phase4_readme_for_test(
        config,
        optimizers=[lead, candidate],
        lead_optimizer_id="opt-1",
        prefill_optimizer_id="opt-2",
    )

    assert "## Score Semantics" in instruction
    assert "Higher optimizer Elo is better." in instruction


def test_optimizer_readme_omits_removed_phase4_important_message() -> None:
    lead = PopulationEntry(
        id="opt-1",
        files={"read-eval.md": "guidance"},
        score=_score(1500.0),
        logs={},
    )
    candidate = PopulationEntry(
        id="opt-2",
        files={"read-eval.md": "guidance"},
        score=_score(1490.0),
        logs={},
    )

    config = _instantiate_test_instructions(_make_test_config(workspace_root=Path("run")))
    _ = (lead, "empty")
    instruction = _render_phase4_readme_for_test(
        config,
        optimizers=[lead, candidate],
        lead_optimizer_id="opt-1",
        prefill_optimizer_id="opt-2",
    )

    assert "VERY IMPORTANT RULE" not in instruction
    assert "Don't keep improving!" not in instruction
    assert "Do not ask the human for clarification, approval, or feedback" in instruction


def test_boundary_check_allows_editable_folder_changes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    (baseline / "prompts").mkdir(parents=True)
    (candidate / "prompts").mkdir(parents=True)
    (baseline / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (candidate / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (baseline / "prompts" / "system.md").write_text("v1\n", encoding="utf-8")
    (candidate / "prompts" / "system.md").write_text("v2\n", encoding="utf-8")
    (candidate / "prompts" / "extra.md").write_text("new\n", encoding="utf-8")

    result = check_workspace_boundary(
        baseline_root=baseline,
        candidate_root=candidate,
        editable={"files": ("candidate.py",), "folders": ("prompts",)},
    )

    assert result.ok


def test_extract_includes_editable_folder_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "prompts").mkdir(parents=True)
    (snapshot / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (snapshot / "prompts" / "system.md").write_text("v1\n", encoding="utf-8")
    problem = RepoTaskProblem(
        name="circle-packing",
        github_url="https://github.com/scaling-group/eve",
        commit="abc123def456",
        editable_files=("candidate.py",),
        editable_folders=("prompts",),
        check_agent_paths={
            "claude": tmp_path / "check_claude.md",
            "codex": tmp_path / "check_codex.toml",
        },
        evaluation_steps=(tmp_path / "evaluation.md",),
        local_checkout=snapshot,
        snapshot_root=snapshot,
        boundary_checker_path=tmp_path / "boundary.py",
    )
    problem.check_agent_paths["claude"].write_text(
        "\n".join(
            [
                "---",
                "name: check-runner",
                'description: "Run the check workflow and report PASS or FAIL."',
                "tools: Bash, Read",
                "---",
                "",
                "Run `python3 -m py_compile candidate.py`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    problem.check_agent_paths["codex"].write_text(
        "\n".join(
            [
                'name = "check-runner"',
                'description = "Run the check workflow and report PASS or FAIL."',
                'developer_instructions = """Run `python3 -m py_compile candidate.py`.',
                '"""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    problem.evaluation_steps[0].write_text("# Formal evaluation\n", encoding="utf-8")
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    workspace, _ = builder.build({}, [], workspace_id="workspace-1")
    output_dir = workspace / "output"
    (output_dir / "prompts" / "extra.md").write_text("new\n", encoding="utf-8")

    files = builder.extract(workspace)

    assert files["candidate.py"] == "print('seed')\n"
    assert files["prompts/system.md"] == "v1\n"
    assert files["prompts/extra.md"] == "new\n"


def test_boundary_repair_instruction_requires_rechecking_each_repair_pass() -> None:
    result = BoundaryCheckResult(forbidden_modified=("README.md",))

    instruction = phase2_boundary_repair_instruction(result)

    assert "after each repair pass" in instruction
    assert "`check-runner`" in instruction


def test_solver_workspace_embeds_check_agent_from_config(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )
    workspace, _ = builder.build({}, [], workspace_id="workspace-1")
    builder.write_readme(workspace, "# README\n")
    claude_check = (workspace / ".claude" / "agents" / "check-runner.md").read_text(
        encoding="utf-8"
    )
    codex_check = tomllib.loads(
        (workspace / ".codex" / "agents" / "check-runner.toml").read_text(encoding="utf-8")
    )["developer_instructions"]

    assert "python3 -m py_compile candidate.py" in claude_check
    assert "--baseline-root" in claude_check
    assert "python3 -m py_compile candidate.py" in codex_check
    assert "--baseline-root" in codex_check


def test_solver_workspace_name_uses_underscore_timestamp_format(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )

    workspace, _ = builder.build({}, [], workspace_id="workspace-1")

    timestamp_prefix = workspace.name.split("_workspace-1", maxsplit=1)[0]
    assert "T" not in timestamp_prefix
    assert len(timestamp_prefix) == len("20260411_175422")


def test_optimizer_workspace_name_uses_underscore_timestamp_format(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = OptimizerWorkspaceBuilder(
        tmp_path / "optimizer_workspaces",
        problem=problem,
        config=config,
        instructions=_instruction_objects(config),
    )

    workspace, _ = builder.build(
        PopulationEntry(id="lead", files={"optimizer.py": "pass\n"}, score={"elo": 1.0}, logs={}),
        [],
        workspace_id="workspace-1",
    )

    timestamp_prefix = workspace.name.split("_workspace-1", maxsplit=1)[0]
    assert "T" not in timestamp_prefix
    assert len(timestamp_prefix) == len("20260411_175422")


def test_eval_reads_score_yaml_and_preserves_log_tree(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)

    class _EvalDriver:
        def spawn(self, seed: object) -> object:
            _ = seed
            (repo_root / "logs" / "evaluate" / "score.yaml").write_text(
                "score: 1.25\nsummary: good run\n",
                encoding="utf-8",
            )
            (repo_root / "logs" / "evaluate" / "summary.txt").write_text(
                "good run\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="eval transcript")

    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: _EvalDriver(),
    )

    score, logs = evaluator(repo_root)

    assert score["score"] == 1.25
    assert "score: 1.25" in logs["score.yaml"]
    assert logs["summary.txt"] == "good run\n"


def test_eval_emits_progress_logs_for_shell_steps(
    tmp_path: Path,
    caplog,
) -> None:
    problem = _make_problem(tmp_path)
    shell_step = tmp_path / "evaluation_progress.sh"
    shell_step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "mkdir -p logs/evaluate",
                "cat <<'EOF' > logs/evaluate/score.yaml",
                "score: 0.5",
                "summary: progress",
                "EOF",
                "printf 'progress\\n' > logs/evaluate/summary.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    problem = RepoTaskProblem(
        name=problem.name,
        github_url=problem.github_url,
        commit=problem.commit,
        editable_files=problem.editable_files,
        editable_folders=problem.editable_folders,
        check_agent_paths=problem.check_agent_paths,
        evaluation_steps=(shell_step,),
        local_checkout=problem.local_checkout,
        snapshot_root=problem.snapshot_root,
        boundary_checker_path=problem.boundary_checker_path,
    )
    workspace = tmp_path / "workspace"
    (workspace / "output").mkdir(parents=True)
    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: SimpleNamespace(),
    )

    with caplog.at_level(logging.INFO):
        score, logs = evaluator(workspace)

    assert score["score"] == 0.5
    assert "score: 0.5" in logs["score.yaml"]
    assert "Evaluation: starting workspace `workspace`" in caplog.text
    assert "Evaluation: starting shell step 01 `evaluation_progress.sh`" in caplog.text
    assert "waiting for completion, logs under" in caplog.text
    assert "Evaluation: shell step 01 `evaluation_progress.sh` finished successfully" in caplog.text
    assert "Evaluation: parsed score for workspace `workspace`" in caplog.text


def test_eval_runs_shell_and_markdown_steps_in_workspace_root(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = tmp_path / "evaluation.sh"
    shell_step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "test -f README.md",
                "test ! -f candidate.py",
                "mkdir -p logs/evaluate",
                "printf 'from-shell\\n' > logs/evaluate/from_shell.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_step = tmp_path / "evaluation_agent.md"
    markdown_step.write_text("# Agent step\n", encoding="utf-8")
    problem = RepoTaskProblem(
        name=problem.name,
        github_url=problem.github_url,
        commit=problem.commit,
        editable_files=problem.editable_files,
        editable_folders=problem.editable_folders,
        check_agent_paths=problem.check_agent_paths,
        evaluation_steps=(shell_step, markdown_step),
        local_checkout=problem.local_checkout,
        snapshot_root=problem.snapshot_root,
        boundary_checker_path=problem.boundary_checker_path,
    )
    workspace = tmp_path / "workspace"
    output_dir = workspace / "output"
    output_dir.mkdir(parents=True)
    (workspace / "README.md").write_text("# workspace\n", encoding="utf-8")
    (output_dir / "candidate.py").write_text("print('candidate')\n", encoding="utf-8")

    class _EvalDriver:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def spawn(self, seed: object) -> object:
            self.calls.append(seed)
            current = Path(seed.working_directory)
            assert current == workspace
            assert (current / "README.md").exists()
            assert not (current / "candidate.py").exists()
            (current / "logs" / "evaluate").mkdir(parents=True, exist_ok=True)
            (current / "logs" / "evaluate" / "score.yaml").write_text(
                "score: 2.5\nsummary: eval transcript\n",
                encoding="utf-8",
            )
            (current / "logs" / "evaluate" / "from_agent.txt").write_text(
                "from-agent\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="eval transcript")

    driver = _EvalDriver()
    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: driver,
    )

    score, logs = evaluator(workspace)

    assert score["score"] == 2.5
    assert "score: 2.5" in logs["score.yaml"]
    assert logs["from_shell.txt"] == "from-shell\n"
    assert logs["from_agent.txt"] == "from-agent\n"
    assert logs["steps/step_01_evaluation/status.txt"] == "ok\n"
    assert "steps/step_01_evaluation/stdout.txt" not in logs
    assert "steps/step_01_evaluation/stderr.txt" not in logs
    assert logs["steps/step_02_evaluation_agent/status.txt"] == "ok\n"
    assert logs["steps/step_02_evaluation_agent/summary.txt"] == "eval transcript\n"
    assert len(driver.calls) == 1
    assert driver.calls[0].instruction == "# Agent step\n"
    assert driver.calls[0].write_prompt_file is False


def test_eval_preserves_shell_step_logs_on_failure(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = tmp_path / "evaluation_fail.sh"
    shell_step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf 'hello\\n'",
                "printf 'boom\\n' >&2",
                "exit 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    problem = RepoTaskProblem(
        name=problem.name,
        github_url=problem.github_url,
        commit=problem.commit,
        editable_files=problem.editable_files,
        editable_folders=problem.editable_folders,
        check_agent_paths=problem.check_agent_paths,
        evaluation_steps=(shell_step,),
        local_checkout=problem.local_checkout,
        snapshot_root=problem.snapshot_root,
        boundary_checker_path=problem.boundary_checker_path,
    )
    workspace = tmp_path / "workspace"
    (workspace / "output").mkdir(parents=True)

    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: SimpleNamespace(),
    )

    with pytest.raises(RemoteTransportHaltError):
        evaluator(workspace)

    logs = {
        str(path.relative_to(workspace / "logs" / "evaluate")): path.read_text(encoding="utf-8")
        for path in (workspace / "logs" / "evaluate").rglob("*")
        if path.is_file()
    }

    assert logs["steps/step_01_evaluation_fail/status.txt"] == "failed\n"
    assert logs["steps/step_01_evaluation_fail/stdout.txt"] == "hello\n"
    assert logs["steps/step_01_evaluation_fail/stderr.txt"] == "boom\n"


def test_phase2_batch_reraises_transport_halt(tmp_path: Path, monkeypatch) -> None:
    class _Population:
        def __init__(self, entries: list[PopulationEntry]) -> None:
            self._entries = list(entries)
            self._rng = random.Random(1)

        def entries(self) -> list[PopulationEntry]:
            return list(self._entries)

        def add(self, entry: PopulationEntry) -> None:
            self._entries.append(entry)

        def update_logs(self, logs_by_id: dict[str, dict[str, str]]) -> None:
            _ = logs_by_id

    class _HeadSampler:
        def sample(self, entries, scores, n, rng):  # noqa: ANN001, ARG002
            _ = scores
            _ = rng
            return list(entries[:n])

    def _raise_transport_halt(self, **kwargs):  # noqa: ANN001
        _ = (self, kwargs)
        raise RemoteTransportHaltError(
            step=Path("evaluate.sh"),
            workspace_root=tmp_path / "workspace",
            returncode=1,
            stdout="",
            stderr="[REMOTE-L4] breaker open window exceeded",
        )

    monkeypatch.setattr(Phase2Runner, "run_single", _raise_transport_halt)

    solver_pop = _Population(
        [
            PopulationEntry(
                id="solver_1", files={"candidate.py": "seed\n"}, score=_score(0.4), logs={}
            )
        ]
    )
    optimizer_pop = _Population(
        [
            PopulationEntry(
                id="opt-1",
                files={"APPROACH.md": "approach"},
                score=_optimizer_score(1500.0),
                logs={},
            )
        ]
    )

    batch_runner = Phase2BatchRunner(
        solver_workspace_builder=SimpleNamespace(_rng=random.Random(1)),
        driver=object(),
        solver_evaluator=object(),
        step_label="step_1",
        iteration=1,
        solver_pop=solver_pop,
        optimizer_pop=optimizer_pop,
        n_workers_phase2=1,
        n_solver_examples_phase2=1,
        n_optimizer_examples_phase2=0,
        n_produced_optimizers_phase2=0,
        optimizer_sampler=_HeadSampler(),
        solver_sampler=_HeadSampler(),
        prefill_sampler=_HeadSampler(),
        optimizer_examples_sampler=_HeadSampler(),
        produced_optimizer_sampler=_HeadSampler(),
    )

    with pytest.raises(RemoteTransportHaltError):
        batch_runner.run()


def test_eval_rejects_invalid_score_yaml(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)

    class _EvalDriver:
        def spawn(self, seed: object) -> object:
            _ = seed
            (repo_root / "logs" / "evaluate" / "score.yaml").write_text(
                "- not-a-score-card\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="eval transcript")

    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: _EvalDriver(),
    )

    score, logs = evaluator(repo_root)

    assert score == ["not-a-score-card"]
    assert logs["score.yaml"] == "- not-a-score-card\n"


def test_eval_fails_when_score_file_is_missing(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)

    class _EvalDriver:
        def spawn(self, seed: object) -> object:
            _ = seed
            (repo_root / "logs" / "evaluate" / "summary.txt").write_text(
                "missing score\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="eval transcript")

    evaluator = build_solver_evaluator(
        problem,
        evaluation_failure_score={"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: _EvalDriver(),
    )

    score, logs = evaluator(repo_root)

    assert score == {"score": 0.0, "summary": "evaluation failed"}
    assert logs["summary.txt"] == "missing score\n"
