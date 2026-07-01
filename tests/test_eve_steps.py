from __future__ import annotations

import json
import logging
import random
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from scaling_evolve.algorithms.eve.factory import _load_solver_worker_configs
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.evaluators.elo import (
    EvalScalarEloEvaluator,
    ScalarEloEvaluator,
    VectorEloEvaluator,
)
from scaling_evolve.algorithms.eve.populations.samplers.rank_softmax import (
    EvalRankSoftmaxSampler,
    RankExponentialSumSampler,
    RankSoftmaxSampler,
)
from scaling_evolve.algorithms.eve.populations.samplers.uniform import UniformSampler
from scaling_evolve.algorithms.eve.populations.score import scalar
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.prompt_assets import read_required_prompt_text
from scaling_evolve.algorithms.eve.rollout_prompts.default import BudgetPrompt
from scaling_evolve.algorithms.eve.workflow.boundary import (
    BoundaryCheckResult,
    check_workspace_boundary,
)
from scaling_evolve.algorithms.eve.workflow.evaluation import (
    EvaluationPlan,
    EvaluationStep,
    _build_eval_workspace,
    build_evaluation_plan,
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
from scaling_evolve.algorithms.eve.workspace.file_tree import read_file_tree
from scaling_evolve.algorithms.eve.workspace.immutable_renderers.default import (
    DefaultRenderer,
)
from scaling_evolve.algorithms.eve.workspace.immutable_renderers.static import (
    StaticRenderer,
)
from scaling_evolve.algorithms.eve.workspace.solver_workspace import (
    SolverWorkerConfig,
    SolverWorkspaceBuilder,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_immutable_files() -> dict[str, str]:
    return read_file_tree(_REPO_ROOT / "configs/eve/optimizer/circle_packing/immutable")


def _default_prompt_root() -> Path:
    return _REPO_ROOT / "configs/eve/optimizer/circle_packing/prompt"


def _default_renderer() -> DefaultRenderer:
    return DefaultRenderer(
        entrypoint=read_required_prompt_text(_default_prompt_root(), "ENTRYPOINT.md")
    )


def _default_boundary_repair_prompt() -> str:
    return read_required_prompt_text(_default_prompt_root(), "BOUNDARY_REPAIR.md")


def _default_budget_prompt() -> BudgetPrompt:
    return BudgetPrompt(prompt_root=_default_prompt_root())


def _immutable_files(config: DictConfig) -> dict[str, str]:
    return dict(OmegaConf.to_container(config.immutable_files, resolve=True))


def _solver_workspace_builder_kwargs(config: DictConfig) -> dict[str, object]:
    return {
        "immutable_files": _immutable_files(config),
        "immutable_renderer": _default_renderer(),
        "boundary_repair_prompt": _default_boundary_repair_prompt(),
    }


def _make_test_config(workspace_root: Path | str = "run", **overrides) -> DictConfig:
    """Build a test DictConfig with defaults matching loop/default.yaml."""
    _S = "scaling_evolve.algorithms.eve"
    _RS = f"{_S}.populations.samplers.rank_softmax"
    _US = f"{_S}.populations.samplers.uniform"
    cfg = {
        "max_iterations": 2,
        "n_workers_phase2": 2,
        "n_solver_examples_phase2": 4,
        "n_optimizer_examples_phase2": 4,
        "exclude_all_working_optimizers_from_examples": False,
        "boundary_repair_attempts": 3,
        "enable_resume": True,
        "retain_workspaces": True,
        "produce_optimizer_in_phase2": 0,
        "sampling": {
            "working_optimizer": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "auto",
            },
            "solver_examples": {
                "_target_": f"{_RS}.RankExponentialSumSampler",
                "features": {
                    "score": {"weight": 1.0, "temperature": 1.0},
                },
                "replacement_mode": "no_replacement",
            },
            "solver_prefill": {
                "_target_": f"{_US}.UniformSampler",
                "replacement_mode": "no_replacement",
            },
            "optimizer_examples": {
                "_target_": f"{_RS}.RankSoftmaxSampler",
                "temperature": 1.0,
                "replacement_mode": "no_replacement",
            },
            "produced_optimizers": {
                "_target_": f"{_RS}.RankExponentialSumSampler",
                "features": {
                    "score": {"weight": 1.0, "temperature": 1.0},
                },
                "replacement_mode": "no_replacement",
            },
        },
        "immutable_files": _default_immutable_files(),
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
        self.spawn_prompt_files: list[str | None] = []
        self.optimizer_guidance_update: dict[str, str] = {}

    def spawn(self, seed: object) -> object:
        self.spawn_instructions.append(seed.instruction)
        self.spawn_prompt_files.append(seed.prompt_file)
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
    """Compatibility shim for tests that predate immutable workspace assets."""
    return config


def _render_phase2_readme_for_test(
    config: DictConfig,
    *,
    problem: RepoTaskProblem,
    solvers: list[PopulationEntry],
    prefill_solver_id: str,
    optimizer_examples: list[PopulationEntry] | None = None,
) -> str:
    prefill_solver = next(entry for entry in solvers if entry.id == prefill_solver_id)
    return _default_renderer().render(
        _immutable_files(config)["README.md"],
        problem=problem,
        config=config,
        solvers=solvers,
        prefill_solver=prefill_solver,
        optimizer_examples=optimizer_examples or [],
    )


def test_static_renderer_returns_template_unchanged() -> None:
    template = "  Static rubric with {solver_examples_block}\n\n"

    assert StaticRenderer().render(template, object()) == template


def _score(value: float, *, summary: str | None = None) -> object:
    return {"score": value, "summary": summary or f"score={value}"}


def _optimizer_score(value: float) -> object:
    return {"elo": value}


def _make_problem(tmp_path: Path) -> RepoTaskProblem:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    (snapshot / "README.md").write_text("# repo\n", encoding="utf-8")
    (snapshot / "evaluate.py").write_text(
        'print(\'{"score": 0.0, "summary": "harness"}\')\n', encoding="utf-8"
    )
    return RepoTaskProblem(
        name="evolve-testbed",
        path=None,
        github_url="https://github.com/scaling-group/evolve-testbed",
        commit="abc123def456",
        editable_files=("candidate.py",),
        editable_folders=(),
        local_checkout=snapshot,
        snapshot_root=snapshot,
        boundary_checker_path=tmp_path / "boundary.py",
    )


def test_repo_task_problem_path_snapshot_reads_local_task_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    task_root = repo_root / "examples" / "math_proof" / "local_task"
    proof_root = task_root / "proof"
    problem_root = task_root / "problem"
    proof_root.mkdir(parents=True)
    problem_root.mkdir(parents=True)
    (proof_root / ".gitkeep").write_text("", encoding="utf-8")
    (problem_root / "problem.md").write_text("Uncommitted problem text.\n", encoding="utf-8")

    problem = RepoTaskProblem.from_config(
        {
            "name": "math-proof-local-task",
            "path": "examples/math_proof/local_task",
            "editable": {"files": [], "folders": ["proof"]},
        },
        cache_root=tmp_path / "cache",
        search_root=repo_root,
    )

    assert problem.local_checkout == repo_root.resolve()
    assert problem.path == "examples/math_proof/local_task"
    assert problem.github_url is None
    assert problem.commit is None
    assert (problem.snapshot_root / "problem" / "problem.md").read_text(
        encoding="utf-8"
    ) == "Uncommitted problem text.\n"
    assert problem.seed_files() == {"proof/.gitkeep": ""}


def test_repo_task_problem_rejects_path_with_git_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="application.path cannot be combined"):
        RepoTaskProblem.from_config(
            {
                "name": "math-proof-local-task",
                "path": "examples/math_proof/local_task",
                "github_url": f"file://{tmp_path}",
                "commit": "HEAD",
                "editable": {"files": [], "folders": ["proof"]},
            },
            cache_root=tmp_path / "cache",
            search_root=tmp_path,
        )


def test_repo_task_problem_requires_complete_source_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either path or both github_url and commit"):
        RepoTaskProblem.from_config(
            {
                "name": "math-proof-local-task",
                "github_url": f"file://{tmp_path}",
                "editable": {"files": [], "folders": ["proof"]},
            },
            cache_root=tmp_path / "cache",
            search_root=tmp_path,
        )


def test_runtime_template_renderer_uses_solver_as_candidate_root(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)

    rendered = problem.render_runtime_template("{{BOUNDARY_CHECK_COMMAND}}\n")

    assert "{{BOUNDARY_CHECK_COMMAND}}" not in rendered
    assert f"--baseline-root {problem.snapshot_root}" in rendered
    assert f"--editable-file {problem.editable_files[0]}" in rendered
    assert "--candidate-root solver" in rendered


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
        **_solver_workspace_builder_kwargs(config),
    )
    solver_evaluator = build_solver_evaluator(
        problem,
        evaluation_plan=EvaluationPlan(steps=()),
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
        run_id=str(config.run_id),
        solver_pop=solver_pop,  # type: ignore[arg-type]
        optimizer_pop=optimizer_pop,  # type: ignore[arg-type]
        solver_workspace_builder=solver_workspace_builder,
        solver_driver=object(),  # type: ignore[arg-type]
        solver_evaluator=solver_evaluator,
        config=config,
        optimizer_evaluator=ScalarEloEvaluator(),
        phase2_optimizer_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_solver_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_optimizer_examples_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_produced_optimizer_sampler=UniformSampler(replacement_mode="no_replacement"),
    )
    return loop, solver_pop


def _minimal_immutable_files() -> dict[str, str]:
    return {
        "README.md": "\n".join(
            [
                "# Workspace Notes",
                "",
                "{editable_files_block}",
                "{editable_folders_block}",
                "{solver_examples_block}",
                "{optimizer_examples_block}",
            ]
        )
        + "\n",
        "AGENTS.md": "# Workspace Agent Instructions\n",
        "CLAUDE.md": "# Workspace Agent Instructions\n",
    }


def _minimal_immutable_files_with_marker(marker: str) -> dict[str, str]:
    files = _minimal_immutable_files()
    files["README.md"] = f"# {marker}\n\n{files['README.md']}"
    files["AGENTS.md"] = f"# {marker} Agent Instructions\n"
    files["CLAUDE.md"] = f"# {marker} Agent Instructions\n"
    return files


def _worker_config(
    *,
    name: str,
    weight: float,
    marker: str,
    entrypoint: str,
    boundary_repair_prompt: str = "Repair only allowed files.",
) -> SolverWorkerConfig:
    return SolverWorkerConfig(
        name=name,
        weight=weight,
        immutable_files=_minimal_immutable_files_with_marker(marker),
        immutable_renderer=DefaultRenderer(entrypoint=entrypoint),
        boundary_repair_prompt=boundary_repair_prompt,
        rollout_prompts={"budget": _default_budget_prompt()},
    )


def _write_file_tree(root: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _write_prompt_root(root: Path, *, entrypoint: str, boundary: str = "Repair boundary.") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ENTRYPOINT.md").write_text(entrypoint, encoding="utf-8")
    (root / "BOUNDARY_REPAIR.md").write_text(boundary, encoding="utf-8")
    budget_root = root / "budget"
    budget_root.mkdir()
    (budget_root / "USER.md").write_text("Budget user prompt.\n", encoding="utf-8")
    (budget_root / "TURN.md").write_text("Budget turn prompt.\n", encoding="utf-8")


def test_factory_loads_worker_items_and_rejects_missing_workers(tmp_path: Path) -> None:
    normal_immutable = tmp_path / "normal_immutable"
    exploratory_immutable = tmp_path / "exploratory_immutable"
    normal_prompt = tmp_path / "normal_prompt"
    exploratory_prompt = tmp_path / "exploratory_prompt"
    _write_file_tree(normal_immutable, _minimal_immutable_files_with_marker("Normal"))
    _write_file_tree(exploratory_immutable, _minimal_immutable_files_with_marker("Exploratory"))
    _write_prompt_root(normal_prompt, entrypoint="Normal entrypoint.")
    _write_prompt_root(exploratory_prompt, entrypoint="Exploratory entrypoint.")
    renderer_cfg = {
        "_target_": (
            "scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer"
        ),
    }
    cfg = OmegaConf.create(
        {
            "optimizer": {
                "workers": {
                    "selection": "random",
                    "items": [
                        {
                            "name": "normal",
                            "weight": 1.0,
                            "immutable": "normal_immutable",
                            "prompt": "normal_prompt",
                        },
                        {
                            "name": "exploratory",
                            "weight": 2.5,
                            "immutable": "exploratory_immutable",
                            "prompt": "exploratory_prompt",
                        },
                    ],
                },
                "immutable_renderer": renderer_cfg,
            }
        }
    )

    worker_configs = _load_solver_worker_configs(cfg, search_root=tmp_path)

    assert [worker.name for worker in worker_configs] == ["normal", "exploratory"]
    assert [worker.weight for worker in worker_configs] == [1.0, 2.5]
    assert worker_configs[0].immutable_files["AGENTS.md"] == "# Normal Agent Instructions\n"
    assert worker_configs[1].boundary_repair_prompt == "Repair boundary."

    old_shape_cfg = OmegaConf.create(
        {
            "optimizer": {
                "immutable": "normal_immutable",
                "prompt": "normal_prompt",
                "immutable_renderer": renderer_cfg,
            }
        }
    )

    with pytest.raises(SystemExit, match="optimizer\\.workers is required"):
        _load_solver_worker_configs(old_shape_cfg, search_root=tmp_path)


def test_solver_workspace_builder_weighted_selection_is_seeded(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    worker_configs = [
        _worker_config(name="normal", weight=1.0, marker="Normal", entrypoint="Normal."),
        _worker_config(
            name="exploratory",
            weight=3.0,
            marker="Exploratory",
            entrypoint="Exploratory.",
        ),
    ]

    def selection_sequence(seed: int) -> list[str]:
        builder = SolverWorkspaceBuilder(
            tmp_path / f"solver_workspaces_{seed}",
            problem=problem,
            config=config,
            immutable_files={},
            worker_configs=worker_configs,
            rng=random.Random(seed),
        )
        return [builder.select_worker_config(worker_index=index).name for index in range(200)]

    first = selection_sequence(7)
    second = selection_sequence(7)

    assert first == second
    assert first.count("exploratory") > first.count("normal") > 0

    single_worker_builder = SolverWorkspaceBuilder(
        tmp_path / "single_worker",
        problem=problem,
        config=config,
        immutable_files={},
        worker_configs=[worker_configs[0]],
        rng=random.Random(11),
    )

    assert {
        single_worker_builder.select_worker_config(worker_index=index).name for index in range(20)
    } == {"normal"}


def test_solver_workspace_materializes_selected_worker_assets(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    normal_worker = _worker_config(
        name="normal",
        weight=1.0,
        marker="Normal",
        entrypoint="Normal entrypoint.",
    )
    exploratory_worker = _worker_config(
        name="exploratory",
        weight=1.0,
        marker="Exploratory",
        entrypoint="Exploratory entrypoint.",
    )
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        immutable_files={},
        worker_configs=[normal_worker, exploratory_worker],
    )
    optimizer = PopulationEntry(id="opt", files={"APPROACH.md": "approach"}, score={}, logs={})
    solver = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )

    workspace, _ = builder.build(
        optimizer.files,
        [solver],
        workspace_id="workspace_1",
        optimizer=optimizer,
        prefill_solver=solver,
        worker_config=exploratory_worker,
    )
    builder.write_immutable_assets(
        workspace,
        optimizer=optimizer,
        solvers=[solver],
        prefill_solver=solver,
        worker_config=exploratory_worker,
    )

    assert "# Exploratory" in (workspace / "README.md").read_text(encoding="utf-8")
    assert "# Exploratory Agent Instructions" in (workspace / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    assert "Exploratory entrypoint." in builder.entrypoint_instruction(
        optimizer=optimizer,
        solvers=[solver],
        prefill_solver=solver,
        worker_config=exploratory_worker,
    )
    assert "Normal entrypoint." not in builder.entrypoint_instruction(
        optimizer=optimizer,
        solvers=[solver],
        prefill_solver=solver,
        worker_config=exploratory_worker,
    )
    assert yaml.safe_load((workspace / ".scaling_evolve" / "worker.yaml").read_text()) == {
        "name": "exploratory",
        "weight": 1.0,
    }


def test_phase2_worker_without_readme_uses_inline_instruction_only(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    worker = SolverWorkerConfig(
        name="no_readme",
        weight=1.0,
        immutable_files={"AGENTS.md": "agent instructions\n"},
        immutable_renderer=DefaultRenderer(entrypoint="Inline phase2 instruction."),
        boundary_repair_prompt="Repair only allowed files.",
        rollout_prompts={"budget": _default_budget_prompt()},
    )
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        immutable_files={},
        worker_configs=[worker],
    )
    optimizer = PopulationEntry(id="opt-1", files={"APPROACH.md": "approach"}, score={}, logs={})
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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

    assert driver.spawn_instructions == ["Inline phase2 instruction."]
    assert driver.spawn_prompt_files == [None]


def test_normal_worker_matches_default_materialized_assets(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    optimizer = PopulationEntry(id="opt", files={"APPROACH.md": "approach"}, score={}, logs={})
    solver = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={"evaluate/summary.txt": "seed summary"},
    )
    normal_worker = SolverWorkerConfig(
        name="normal",
        weight=1.0,
        immutable_files=_default_immutable_files(),
        immutable_renderer=_default_renderer(),
        boundary_repair_prompt=_default_boundary_repair_prompt(),
        rollout_prompts={"budget": _default_budget_prompt()},
    )
    default_builder = SolverWorkspaceBuilder(
        tmp_path / "default_solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )
    worker_builder = SolverWorkspaceBuilder(
        tmp_path / "worker_solver_workspaces",
        problem=problem,
        config=config,
        immutable_files={},
        worker_configs=[normal_worker],
    )

    def materialize(
        builder: SolverWorkspaceBuilder,
        *,
        workspace_id: str,
        worker_config: SolverWorkerConfig | None = None,
    ) -> tuple[Path, str]:
        workspace, _ = builder.build(
            optimizer.files,
            [solver],
            workspace_id=workspace_id,
            optimizer=optimizer,
            prefill_solver=solver,
            worker_config=worker_config,
        )
        builder.write_immutable_assets(
            workspace,
            optimizer=optimizer,
            solvers=[solver],
            prefill_solver=solver,
            worker_config=worker_config,
        )
        entrypoint = builder.entrypoint_instruction(
            optimizer=optimizer,
            solvers=[solver],
            prefill_solver=solver,
            worker_config=worker_config,
        )
        return workspace, entrypoint

    default_workspace, default_entrypoint = materialize(
        default_builder,
        workspace_id="default",
    )
    worker_workspace, worker_entrypoint = materialize(
        worker_builder,
        workspace_id="worker",
        worker_config=normal_worker,
    )

    for filename in ("README.md", "AGENTS.md", "CLAUDE.md"):
        assert (default_workspace / filename).read_bytes() == (
            worker_workspace / filename
        ).read_bytes()
    assert default_entrypoint == worker_entrypoint


def _make_solver_evaluator(problem: RepoTaskProblem, *, eval_fn) -> object:
    evaluator = build_solver_evaluator(
        problem,
        evaluation_plan=EvaluationPlan(steps=()),
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

    def eval_fn(workspace_root: Path, display_context=None, **kwargs):  # noqa: ARG001
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

    def eval_fn(workspace_root: Path, display_context=None, **kwargs):  # noqa: ARG001
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

    def eval_fn(workspace_root: Path, display_context=None, **kwargs):  # noqa: ARG001
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
        assert "evaluation.seed_solver_score is required" in str(exc)
    else:
        raise AssertionError(
            "Expected seed_solver_skip_evaluation without seed_solver_score to fail"
        )


def test_seed_solver_skip_evaluation_uses_seed_solver_score_when_present(tmp_path: Path) -> None:
    calls: list[Path] = []

    def eval_fn(workspace_root: Path, display_context=None, **kwargs):  # noqa: ARG001
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
    assert "evaluation.seed_solver_score" in solver_pop.entries[0].logs["evaluate/summary.txt"]


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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        solver_workspace_builder.entrypoint_instruction(
            optimizer=optimizer,
            solvers=[candidate],
            prefill_solver=candidate,
        )
    ]
    workspace = next((tmp_path / "solver_workspaces").glob("*"))
    assert not (workspace / "eve.md").exists()
    assert (workspace / "solver" / "candidate.py").exists()
    assert (workspace / "solver" / "README.md").exists()
    assert not (workspace / "check.md").exists()
    assert (
        workspace / "solver_examples" / "solver_1" / "logs" / "evaluate" / "summary.txt"
    ).exists()
    assert not (
        workspace / "examples" / "solver_1" / "logs" / "optimize" / "transcript.txt"
    ).exists()
    assert not (workspace / "logs" / "evaluate").exists()
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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
    assert not (workspace / "score.yaml").exists()


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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        exclude_all_working_optimizers_from_examples=False,
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
        **_solver_workspace_builder_kwargs(config),
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
    assert (workspace / "guidance_examples" / "opt_ref_1" / "guidance" / "APPROACH.md").exists()
    assert (
        workspace / "guidance_examples" / "opt_ref_1" / "logs" / "optimize/summary.txt"
    ).exists()
    assert not (workspace / "examples").exists()
    assert not (workspace / "score.yaml").exists()


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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        exclude_all_working_optimizers_from_examples=True,
        n_produced_optimizers_phase2=0,
        optimizer_sampler=_HeadSampler(),
        solver_sampler=_HeadSampler(),
        prefill_sampler=_HeadSampler(),
        optimizer_examples_sampler=optimizer_examples_sampler,
        produced_optimizer_sampler=_HeadSampler(),
    ).run()

    assert optimizer_examples_sampler.calls == [["opt_3", "opt_4"]]
    assert optimizer_examples_seen == [["opt_1", "opt_3"], ["opt_2", "opt_3"]]


def test_phase2_batch_can_sample_optimizer_examples_from_working_optimizers(
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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        exclude_all_working_optimizers_from_examples=False,
        n_produced_optimizers_phase2=0,
        optimizer_sampler=_HeadSampler(),
        solver_sampler=_HeadSampler(),
        prefill_sampler=_HeadSampler(),
        optimizer_examples_sampler=optimizer_examples_sampler,
        produced_optimizer_sampler=_HeadSampler(),
    ).run()

    assert optimizer_examples_sampler.calls == [["opt_1", "opt_2", "opt_3"]]
    assert optimizer_examples_seen == [["opt_1", "opt_1"], ["opt_2", "opt_1"]]


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
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        exclude_all_working_optimizers_from_examples=False,
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


def test_phase2_system_prompt_includes_important_message_when_enabled(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    driver = _FakeDriver()
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    solver_workspace_builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        **_solver_workspace_builder_kwargs(config),
        rollout_prompts={"budget": _default_budget_prompt()},
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
            eval_fn=lambda files, display_context=None, **kwargs: (
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
        **_solver_workspace_builder_kwargs(config),
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
    assert (workspace / ".claude" / "settings.local.json").exists()
    assert ".claude-task-stopped" in (workspace / ".claude" / "settings.local.json").read_text(
        encoding="utf-8"
    )


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
    assert "prior score:" in instruction
    assert "quality: 8.0" in instruction
    assert "speed: 10.0" in instruction


def test_solver_workspace_builder_copies_workspace_agent_instruction_files(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(
        _make_test_config(
            workspace_root=tmp_path / "run",
            immutable_files=_minimal_immutable_files(),
        )
    )
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = PopulationEntry(
        id="solver_1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.4),
        logs={},
    )

    builder.write_immutable_assets(
        workspace,
        solvers=[candidate],
        prefill_solver=candidate,
    )

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


def test_eval_scalar_elo_uses_score_expression() -> None:
    current = {
        "optimizer_a": {"elo": 1500.0},
        "optimizer_b": {"elo": 1500.0},
    }
    task_scores = {
        "optimizer_a": {
            "dimensions": {
                "coverage": 90.0,
                "correctness": 80.0,
                "dependency": 40.0,
            }
        },
        "optimizer_b": {
            "dimensions": {
                "coverage": 70.0,
                "correctness": 72.0,
                "dependency": 68.0,
            }
        },
    }

    updated = EvalScalarEloEvaluator(
        expression=(
            'min(score["dimensions"]["coverage"], '
            'score["dimensions"]["correctness"], '
            'score["dimensions"]["dependency"])'
        ),
        k_factor=32.0,
    ).update(
        current,
        task_scores,
    )

    assert updated["optimizer_a"] == {"elo": 1484.0}
    assert updated["optimizer_b"] == {"elo": 1516.0}


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


def test_eval_rank_softmax_ranks_by_expression() -> None:
    items = ["coverage_peak", "balanced", "weak"]
    scores = [
        {"dimensions": {"coverage": 90.0, "correctness": 80.0, "dependency": 40.0}},
        {"dimensions": {"coverage": 70.0, "correctness": 72.0, "dependency": 68.0}},
        {"dimensions": {"coverage": 30.0, "correctness": 90.0, "dependency": 90.0}},
    ]

    class _TopWeightRandom:
        def choices(self, population, weights, k):  # noqa: ANN001, ARG002
            best_index = max(range(len(weights)), key=weights.__getitem__)
            return [population[best_index]]

    selected = EvalRankSoftmaxSampler(
        expression=(
            'min(score["dimensions"]["coverage"], '
            'score["dimensions"]["correctness"], '
            'score["dimensions"]["dependency"])'
        ),
        temperature=1.0,
        replacement_mode="no_replacement",
    ).sample(
        items,
        scores,
        1,
        rng=_TopWeightRandom(),  # type: ignore[arg-type]
    )

    assert selected == ["balanced"]


def test_math_proof_score_expression_is_shared_by_solver_sampling_and_optimizer_elo() -> None:
    config = OmegaConf.merge(
        OmegaConf.load(_REPO_ROOT / "configs/eve/evaluation/math_proof.yaml"),
        OmegaConf.load(_REPO_ROOT / "configs/eve/loop/math_proof.yaml"),
        OmegaConf.load(_REPO_ROOT / "configs/eve/optimizer/math_proof.yaml"),
    )

    expression = config.evaluation.scalar_score_expression
    assert config.loop.sampling.solver_examples.expression == expression
    assert config.loop.sampling.solver_prefill.expression == expression
    assert config.optimizer.evaluation.expression == expression
    assert config.optimizer.evaluation._target_.endswith("EvalScalarEloEvaluator")


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

    assert "Current phase: Phase 2 solver and optimizer optimization." in instruction
    assert "improved Circle Packing solver candidate" in instruction
    assert "Editable files:" in instruction
    assert "- `solver/candidate.py`" in instruction
    assert "invoke the predefined `check-runner`" in instruction
    assert "sub-agent from `.claude/agents/check-runner.md`" in instruction
    assert "Other files inside\n`solver/` are read-only." in instruction


def test_editable_folders_render_relative_to_workspace_root(tmp_path: Path) -> None:
    candidate = PopulationEntry(
        id="task-1",
        files={"proof/main.md": "seed\n"},
        score=_score(0.5),
        logs={},
    )
    problem = replace(_make_problem(tmp_path), editable_files=(), editable_folders=("proof",))
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))

    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "Editable folders:" in instruction
    assert "- `solver/proof/`" in instruction
    assert "- `proof/`" not in instruction


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


def test_solver_readme_includes_phase2_important_message_when_enabled(tmp_path: Path) -> None:
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

    assert "Do not ask the human for clarification, approval, or feedback" in instruction


def test_phase2_self_optimize_readme_allows_solver_and_guidance_edits(
    tmp_path: Path,
) -> None:
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )

    problem = _make_problem(tmp_path)
    config = _instantiate_test_instructions(_make_test_config(workspace_root=tmp_path / "run"))
    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert "Current phase: Phase 2 solver and optimizer optimization." in instruction
    assert "You have two goals:" in instruction
    assert "improved Circle Packing solver candidate" in instruction
    assert "Improve files in `guidance/`" in instruction
    assert "The whole `guidance/` folder is editable" in instruction


def test_readme_literal_replace_allows_bare_braces(tmp_path: Path) -> None:
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )

    problem = _make_problem(tmp_path)
    immutable_files = _minimal_immutable_files()
    immutable_files["README.md"] = "\n".join(
        [
            "# Workspace Notes",
            "",
            'Static JSON example: {"radius": 1, "ok": true}',
            "",
            "{editable_files_block}",
            "{editable_folders_block}",
            "{solver_examples_block}",
            "{optimizer_examples_block}",
        ]
    )
    config = _instantiate_test_instructions(
        _make_test_config(workspace_root=tmp_path / "run", immutable_files=immutable_files)
    )

    instruction = _render_phase2_readme_for_test(
        config,
        problem=problem,
        solvers=[candidate],
        prefill_solver_id="task-1",
    )

    assert 'Static JSON example: {"radius": 1, "ok": true}' in instruction
    assert "`solver_examples/task-1/` <- prefill" in instruction


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
        name="evolve-testbed",
        path=None,
        github_url="https://github.com/scaling-group/evolve-testbed",
        commit="abc123def456",
        editable_files=("candidate.py",),
        editable_folders=("prompts",),
        local_checkout=snapshot,
        snapshot_root=snapshot,
        boundary_checker_path=tmp_path / "boundary.py",
    )
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )
    workspace, _ = builder.build({}, [], workspace_id="workspace-1")
    solver_dir = workspace / "solver"
    (solver_dir / "prompts" / "extra.md").write_text("new\n", encoding="utf-8")

    files = builder.extract(workspace)

    assert files["candidate.py"] == "print('seed')\n"
    assert files["prompts/system.md"] == "v1\n"
    assert files["prompts/extra.md"] == "new\n"


def test_boundary_repair_instruction_requires_rechecking_each_repair_pass() -> None:
    result = BoundaryCheckResult(forbidden_modified=("README.md",))

    instruction = phase2_boundary_repair_instruction(result, _default_boundary_repair_prompt())

    assert "after each repair pass" in instruction
    assert "`check-runner`" in instruction


def test_boundary_repair_instruction_uses_configured_prompt(tmp_path: Path) -> None:
    result = BoundaryCheckResult(forbidden_modified=("README.md",))
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )

    instruction = builder.boundary_repair_instruction(result)

    assert instruction == phase2_boundary_repair_instruction(
        result, _default_boundary_repair_prompt()
    )


def test_solver_workspace_embeds_check_agent_from_immutable(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )
    workspace, _ = builder.build({}, [candidate], workspace_id="workspace-1")
    builder.write_immutable_assets(
        workspace,
        solvers=[candidate],
        prefill_solver=candidate,
    )
    claude_check = (workspace / ".claude" / "agents" / "check-runner.md").read_text(
        encoding="utf-8"
    )
    codex_check = tomllib.loads(
        (workspace / ".codex" / "agents" / "check-runner.toml").read_text(encoding="utf-8")
    )["developer_instructions"]

    assert "python3 -m py_compile solver/candidate.py solver/evaluate.py" in claude_check
    assert "--baseline-root" in claude_check
    assert "python3 -m py_compile solver/candidate.py solver/evaluate.py" in codex_check
    assert "--baseline-root" in codex_check


def test_solver_workspace_can_use_immutable_check_agents(tmp_path: Path) -> None:
    immutable_files = _minimal_immutable_files()
    immutable_files["README.md"] += "\n{immutable_overlay_block}\n"
    immutable_files[".claude/agents/check-runner.md"] = "\n".join(
        [
            "---",
            "name: check-runner",
            "---",
            "",
            "Immutable Claude check.",
            "{{BOUNDARY_CHECK_COMMAND}}",
        ]
    )
    immutable_files[".codex/agents/check-runner.toml"] = "\n".join(
        [
            'name = "check-runner"',
            'description = "Immutable check"',
            'developer_instructions = """Immutable Codex check.',
            "{{BOUNDARY_CHECK_COMMAND}}",
            '"""',
        ]
    )
    candidate = PopulationEntry(
        id="task-1",
        files={"candidate.py": "print('seed')\n"},
        score=_score(0.5),
        logs={},
    )
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run", immutable_files=immutable_files)
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )

    workspace, _ = builder.build({}, [candidate], workspace_id="workspace-1")
    builder.write_immutable_assets(
        workspace,
        solvers=[candidate],
        prefill_solver=candidate,
    )

    claude_check = (workspace / ".claude" / "agents" / "check-runner.md").read_text(
        encoding="utf-8"
    )
    codex_check = tomllib.loads(
        (workspace / ".codex" / "agents" / "check-runner.toml").read_text(encoding="utf-8")
    )["developer_instructions"]

    assert "Immutable Claude check." in claude_check
    assert "Immutable Codex check." in codex_check
    assert "{{BOUNDARY_CHECK_COMMAND}}" not in claude_check
    assert "{{BOUNDARY_CHECK_COMMAND}}" not in codex_check
    assert "--baseline-root" in claude_check
    assert "--baseline-root" in codex_check
    assert (workspace / "guidance" / "agents" / "claude" / "check-runner.md").exists()
    assert (workspace / "guidance" / "agents" / "codex" / "check-runner.toml").exists()
    assert "agents/claude/check-runner.md" not in builder.extract_optimizer(
        workspace,
        worker_config=builder.worker_configs[0],
    )
    assert "agents/codex/check-runner.toml" not in builder.extract_optimizer(
        workspace,
        worker_config=builder.worker_configs[0],
    )

    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert "{immutable_overlay_block}" not in readme
    assert "`.claude/agents/check-runner.md` overlays" in readme


def test_solver_workspace_name_uses_underscore_timestamp_format(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    config = _make_test_config(workspace_root=tmp_path / "run")
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=config,
        **_solver_workspace_builder_kwargs(config),
    )

    workspace, _ = builder.build({}, [], workspace_id="workspace-1")

    timestamp_prefix = workspace.name.split("_workspace-1", maxsplit=1)[0]
    assert "T" not in timestamp_prefix
    assert len(timestamp_prefix) == len("20260411_175422")


def _write_score_shell_step(path: Path, score_yaml: str) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'mkdir -p "$EVE_EVAL_LOG_ROOT"',
                "cat <<'EOF' > \"$EVE_EVAL_LOG_ROOT/score.yaml\"",
                score_yaml.rstrip(),
                "EOF",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _make_evaluation_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    (workspace / "solver").mkdir(parents=True)
    (workspace / "logs" / "optimize").mkdir(parents=True)
    (workspace / "guidance").mkdir(parents=True)
    (workspace / "solver" / "candidate.py").write_text("print('candidate')\n", encoding="utf-8")
    (workspace / "logs" / "optimize" / "transcript.txt").write_text(
        "solver transcript\n", encoding="utf-8"
    )
    (workspace / "guidance" / "APPROACH.md").write_text("optimizer guidance\n", encoding="utf-8")
    return workspace


def _judge_step(
    name: str,
    *,
    immutable_files: dict[str, str],
    entrypoint: str = "Judge entrypoint",
    immutable_renderer=None,
    rollout_prompts: dict[str, object] | None = None,
) -> EvaluationStep:
    return EvaluationStep(
        name=name,
        immutable_files=immutable_files,
        immutable_renderer=immutable_renderer or StaticRenderer(),
        entrypoint=entrypoint,
        rollout_prompts=rollout_prompts,
    )


def _judge_evaluator(
    problem,
    plan,
    driver,
    *,
    evaluation_failure_score=None,
    include_solver_examples: bool = False,
):
    return build_solver_evaluator(
        problem,
        evaluation_plan=plan,
        evaluation_failure_score=evaluation_failure_score
        or {"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        include_solver_examples=include_solver_examples,
        evaluation_driver_factory=lambda: driver,
    )


def test_build_evaluation_plan_dispatches_by_form(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    sh_path = tmp_path / "performance.sh"
    sh_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    immutable_dir = tmp_path / "judge_immutable"
    prompt_dir = tmp_path / "judge_prompt"
    _write_file_tree(immutable_dir, {"README.md": "rubric\n"})
    _write_prompt_root(prompt_dir, entrypoint="Judge entrypoint.")

    cfg = OmegaConf.create(
        {
            "evaluation": {
                "steps": [
                    str(sh_path),
                    {"name": "judge_a", "immutable": str(immutable_dir), "prompt": str(prompt_dir)},
                ]
            }
        }
    )
    plan = build_evaluation_plan(problem, evaluation_config=cfg.evaluation, search_root=tmp_path)
    assert [step.name for step in plan.steps] == ["performance", "judge_a"]
    assert plan.steps[0].is_judge is False
    assert plan.steps[0].path == sh_path.resolve()
    assert plan.steps[1].is_judge is True
    assert plan.steps[1].entrypoint == "Judge entrypoint."
    assert "budget" in plan.steps[1].rollout_prompts

    bad_cfg = OmegaConf.create({"evaluation": {"steps": [{"foo": "bar"}]}})
    with pytest.raises(SystemExit, match="not a valid step"):
        build_evaluation_plan(problem, evaluation_config=bad_cfg.evaluation, search_root=tmp_path)

    dup_cfg = OmegaConf.create(
        {
            "evaluation": {
                "steps": [
                    {"name": "dup", "immutable": str(immutable_dir), "prompt": str(prompt_dir)},
                    {"name": "dup", "immutable": str(immutable_dir), "prompt": str(prompt_dir)},
                ]
            }
        }
    )
    with pytest.raises(SystemExit, match="duplicates judge name"):
        build_evaluation_plan(problem, evaluation_config=dup_cfg.evaluation, search_root=tmp_path)


def test_structured_eval_accumulates_across_judge_steps(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(
        tmp_path / "performance.sh",
        "performance: 0.75\nsummary: programmatic",
    )
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)
    build_count = {"n": 0}

    class _AccumulatingDriver:
        """One driver instance shared by both judge steps; records what each judge sees."""

        def __init__(self) -> None:
            self.seen_by_step: list[dict] = []

        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            existing = yaml.safe_load(
                (eval_ws / "logs" / "evaluate" / "score.yaml").read_text(encoding="utf-8")
            )
            self.seen_by_step.append(dict(existing))
            rubric = (eval_ws / "README.md").read_text(encoding="utf-8")
            if "judge1" in rubric:
                # judge1 keeps performance, adds scientific.
                updated = {**existing, "scientific": 0.9}
            else:
                # judge2 reads judge1's output, folds into a headline `score`.
                assert "scientific" in existing, "judge2 must see judge1's accumulated dim"
                updated = {**existing, "score": 0.5}
            (eval_ws / "logs" / "evaluate" / "score.yaml").write_text(
                yaml.safe_dump(updated, sort_keys=False), encoding="utf-8"
            )
            return SimpleNamespace(summary="judge transcript")

    driver = _AccumulatingDriver()
    real_build = _build_eval_workspace

    def _counting_build(*args, **kwargs):
        build_count["n"] += 1
        return real_build(*args, **kwargs)

    import scaling_evolve.algorithms.eve.workflow.evaluation as eval_mod

    original = eval_mod._build_eval_workspace
    eval_mod._build_eval_workspace = _counting_build
    try:
        plan = EvaluationPlan(
            steps=(
                EvaluationStep(name="performance", path=shell_step),
                _judge_step("judge1", immutable_files={"README.md": "judge1 rubric\n"}),
                _judge_step("judge2", immutable_files={"README.md": "judge2 rubric\n"}),
            ),
        )
        evaluator = _judge_evaluator(problem, plan, driver)
        score, _logs = evaluator(
            workspace,
            candidate_files={"candidate.py": "print('candidate')\n"},
            optimize_logs={"transcript.txt": "solver transcript\n"},
        )
    finally:
        eval_mod._build_eval_workspace = original

    # The eval workspace is built exactly ONCE for both judge steps.
    assert build_count["n"] == 1
    # judge1 saw the sh's performance; judge2 saw judge1's accumulated scientific.
    assert driver.seen_by_step[0] == {"performance": 0.75, "summary": "programmatic"}
    assert driver.seen_by_step[1] == {
        "performance": 0.75,
        "summary": "programmatic",
        "scientific": 0.9,
    }
    # Engine passes the final score through UNCHANGED (no merge / transform).
    assert score == {
        "performance": 0.75,
        "summary": "programmatic",
        "scientific": 0.9,
        "score": 0.5,
    }


def test_eval_workspace_is_reconstructed_from_snapshot_and_candidate(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)
    captured: dict[str, Path] = {}

    class _JudgeDriver:
        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            captured["eval_ws"] = eval_ws
            # solver/ has BOTH the candidate edit AND the snapshot harness (evaluate.py).
            assert (eval_ws / "solver" / "candidate.py").read_text(
                encoding="utf-8"
            ) == "print('edited candidate')\n"
            assert (eval_ws / "solver" / "evaluate.py").exists()
            # logs/optimize materialized from the canonical optimize_logs dict.
            assert (eval_ws / "logs" / "optimize" / "transcript.txt").read_text(
                encoding="utf-8"
            ) == "solver transcript\n"
            (eval_ws / "logs" / "evaluate" / "score.yaml").write_text(
                "score: 0.6\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="judge transcript")

    plan = EvaluationPlan(
        steps=(_judge_step("judge", immutable_files={"README.md": "rubric\n"}),),
    )
    evaluator = _judge_evaluator(problem, plan, _JudgeDriver())
    score, _logs = evaluator(
        workspace,
        candidate_files={"candidate.py": "print('edited candidate')\n"},
        optimize_logs={"transcript.txt": "solver transcript\n"},
    )

    eval_ws = captured["eval_ws"]
    assert score == {"score": 0.6}
    assert eval_ws.parent == tmp_path / "run" / "evaluation_workspaces"
    # solver/ + logs/optimize/ frozen read-only; logs/evaluate/ writable.
    assert (eval_ws / "solver").stat().st_mode & 0o222 == 0
    assert (eval_ws / "logs" / "optimize").stat().st_mode & 0o222 == 0
    assert (eval_ws / "logs" / "evaluate").stat().st_mode & 0o222 != 0
    # No guidance/ in the eval workspace (design §4).
    assert not (eval_ws / "guidance").exists()
    # Solver examples are not materialized unless evaluation.include_solver_examples is enabled.
    assert not (eval_ws / "solver_examples").exists()


def test_eval_workspace_includes_solver_examples_and_renders_readme_block(
    tmp_path: Path,
) -> None:
    problem = _make_problem(tmp_path)
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)
    example = PopulationEntry(
        id="solver_example_1",
        files={"proof/proof.md": "example proof\n"},
        score={"dimensions": {"coverage": 77.0}},
        logs={
            "evaluate/score.yaml": "dimensions:\n  coverage: 77.0\n",
            "optimize/transcript.txt": "not copied\n",
        },
    )
    captured: dict[str, Path] = {}

    class _JudgeDriver:
        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            assert seed.prompt_file == "README.md"
            captured["eval_ws"] = eval_ws
            readme = (eval_ws / "README.md").read_text(encoding="utf-8")
            assert "{solver_examples_block}" not in readme
            assert "- `solver_examples/solver_example_1/` <- prefill" in readme
            assert "  prior score:\n    dimensions:" in readme
            assert "coverage: 77.0" in readme
            assert (
                eval_ws / "solver_examples" / "solver_example_1" / "solver" / "proof" / "proof.md"
            ).read_text(encoding="utf-8") == "example proof\n"
            assert (
                eval_ws
                / "solver_examples"
                / "solver_example_1"
                / "logs"
                / "evaluate"
                / "score.yaml"
            ).exists()
            assert not (
                eval_ws
                / "solver_examples"
                / "solver_example_1"
                / "logs"
                / "optimize"
                / "transcript.txt"
            ).exists()
            (eval_ws / "logs" / "evaluate" / "score.yaml").write_text(
                "score: 0.6\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="judge transcript")

    plan = EvaluationPlan(
        steps=(
            _judge_step(
                "judge",
                immutable_files={
                    "README.md": "Reference examples:\n\n{solver_examples_block}\n",
                    "AGENTS.md": "Agent examples:\n\n{solver_examples_block}\n",
                    "CLAUDE.md": "Judge agent instructions without runtime markers.\n",
                },
                immutable_renderer=DefaultRenderer(),
            ),
        ),
    )
    evaluator = _judge_evaluator(
        problem,
        plan,
        _JudgeDriver(),
        include_solver_examples=True,
    )

    score, _logs = evaluator(
        workspace,
        candidate_files={"candidate.py": "print('candidate')\n"},
        optimize_logs={"transcript.txt": "solver transcript\n"},
        solver_examples=[example],
        prefill_solver=example,
    )

    assert score == {"score": 0.6}
    eval_ws = captured["eval_ws"]
    agents_text = (eval_ws / "AGENTS.md").read_text(encoding="utf-8")
    assert "{solver_examples_block}" not in agents_text
    assert "- `solver_examples/solver_example_1/` <- prefill" in agents_text
    assert "  prior score:\n    dimensions:" in agents_text
    assert (eval_ws / "CLAUDE.md").read_text(
        encoding="utf-8"
    ) == "Judge agent instructions without runtime markers.\n"
    assert (eval_ws / "solver_examples").stat().st_mode & 0o222 == 0


def test_null_immutable_judge_runs_clean_without_inheriting_prior_scaffold(
    tmp_path: Path,
) -> None:
    # A judge's `immutable` is optional (null = no scaffold landed). A null-immutable
    # judge running after a scaffolded judge must see a CLEAN workspace — the prior
    # judge's scaffold is swapped out, not inherited.
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(
        tmp_path / "performance.sh", "performance: 0.5\nsummary: prog"
    )
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)
    scaffold_seen: dict[str, dict[str, bool]] = {}

    class _ScaffoldProbeDriver:
        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            tag = "j1" if "judge1" in seed.instruction else "j2"
            if tag == "j1":
                assert seed.prompt_file == "README.md"
            else:
                assert seed.prompt_file is None
            scaffold_seen[tag] = {
                "README.md": (eval_ws / "README.md").exists(),
                "AGENTS.md": (eval_ws / "AGENTS.md").exists(),
                "ENTRYPOINT.md": (eval_ws / "ENTRYPOINT.md").exists(),
            }
            existing = yaml.safe_load(
                (eval_ws / "logs" / "evaluate" / "score.yaml").read_text(encoding="utf-8")
            )
            existing[tag] = 1.0
            (eval_ws / "logs" / "evaluate" / "score.yaml").write_text(
                yaml.safe_dump(existing, sort_keys=False), encoding="utf-8"
            )
            return SimpleNamespace(summary="judge transcript")

    plan = EvaluationPlan(
        steps=(
            EvaluationStep(name="performance", path=shell_step),
            _judge_step(
                "scaffolded",
                immutable_files={"README.md": "judge1 rubric\n", "AGENTS.md": "judge1 agents\n"},
                entrypoint="judge1 entrypoint",
            ),
            # null-immutable judge: no scaffold landed, runs from the inline instruction only.
            _judge_step("bare", immutable_files={}, entrypoint="judge2 entrypoint"),
        ),
    )
    evaluator = _judge_evaluator(
        problem,
        plan,
        _ScaffoldProbeDriver(),
        evaluation_failure_score={"performance": 0.0, "j1": 0.0, "j2": 0.0},
    )
    score, _logs = evaluator(
        workspace,
        candidate_files={"candidate.py": "print('candidate')\n"},
        optimize_logs={"transcript.txt": "solver transcript\n"},
    )

    # The scaffolded judge saw its own immutable landed.
    assert scaffold_seen["j1"] == {"README.md": True, "AGENTS.md": True, "ENTRYPOINT.md": False}
    # The null-immutable judge saw a CLEAN workspace — the prior scaffold was swapped out.
    assert scaffold_seen["j2"] == {"README.md": False, "AGENTS.md": False, "ENTRYPOINT.md": False}
    # The accumulated score survived the clean swap.
    assert score == {
        "performance": 0.5,
        "summary": "prog",
        "j1": 1.0,
        "j2": 1.0,
    }


def test_build_evaluation_plan_allows_judge_without_immutable(tmp_path: Path) -> None:
    # `immutable` is optional in config: a `{name, prompt}` judge mapping (no immutable)
    # parses as a judge with an empty scaffold, not as a shell step or an error.
    prompt_root = tmp_path / "bare_judge_prompt"
    _write_prompt_root(prompt_root, entrypoint="Bare judge entrypoint.")
    evaluation_config = OmegaConf.create({"steps": [{"name": "bare", "prompt": str(prompt_root)}]})
    plan = build_evaluation_plan(
        _make_problem(tmp_path), evaluation_config=evaluation_config, search_root=tmp_path
    )
    (step,) = plan.steps
    assert step.is_judge
    assert step.immutable_files == {}
    assert step.entrypoint == "Bare judge entrypoint."


def test_structured_eval_captures_judge_score_from_completion_verdict(tmp_path: Path) -> None:
    # Judge-only pipeline: there is no prior step, so no score.yaml exists when the judge runs.
    # The fake judge writes no file, so the engine must capture the fenced completion verdict.
    problem = _make_problem(tmp_path)
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)
    captured: dict[str, Path] = {}

    class _JudgeDriver:
        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            captured["eval_ws"] = eval_ws
            # Fake judge writes NO score.yaml; engine must capture the fenced verdict.
            completion_path = tmp_path / "judge-completion.json"
            completion_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "summary": "\n".join(
                            [
                                "Final verdict:",
                                "```yaml",
                                "scientific: 0.7",
                                "score: 0.6",
                                "```",
                            ]
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(
                summary="judge transcript",
                state=SimpleNamespace(metadata={"completion_path": str(completion_path)}),
            )

    plan = EvaluationPlan(
        steps=(_judge_step("judge", immutable_files={"README.md": "rubric\n"}),),
    )
    evaluator = _judge_evaluator(problem, plan, _JudgeDriver())
    score, logs = evaluator(
        workspace,
        candidate_files={"candidate.py": "print('candidate')\n"},
        optimize_logs={"transcript.txt": "solver transcript\n"},
    )

    assert score == {"scientific": 0.7, "score": 0.6}
    eval_ws = captured["eval_ws"]
    written_score = (eval_ws / "logs" / "evaluate" / "score.yaml").read_text(encoding="utf-8")
    assert "scientific: 0.7" in written_score
    assert "scientific: 0.7" in logs["steps/step_01_judge/score.yaml"]


def test_structured_eval_headline_score_is_scalar_consumable(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)

    class _JudgeDriver:
        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            (eval_ws / "logs" / "evaluate" / "score.yaml").write_text(
                "performance: 0.8\nscientific: 0.9\nscore: 0.84\n", encoding="utf-8"
            )
            return SimpleNamespace(summary="judge transcript")

    plan = EvaluationPlan(
        steps=(_judge_step("aggregate", immutable_files={"README.md": "rubric\n"}),),
    )
    evaluator = _judge_evaluator(problem, plan, _JudgeDriver())
    score, _logs = evaluator(
        workspace,
        candidate_files={"candidate.py": "print('candidate')\n"},
        optimize_logs={"transcript.txt": "solver transcript\n"},
    )
    assert scalar(score) == 0.84


def _shell_step_evaluator(
    problem: RepoTaskProblem,
    shell_step: Path,
    tmp_path: Path,
    *,
    evaluation_failure_score=None,
) -> object:
    """Build an evaluator whose plan is a single `.sh` step, routed through the
    first-class `build_evaluation_plan` mechanism. Exercises the same sh-only
    step executor as the live run path."""
    plan = build_evaluation_plan(
        problem,
        evaluation_config=OmegaConf.create({"steps": [str(shell_step)]}),
        search_root=tmp_path,
    )
    return build_solver_evaluator(
        problem,
        evaluation_plan=plan,
        evaluation_failure_score=evaluation_failure_score
        or {"score": 0.0, "summary": "evaluation failed"},
        boundary_failure_score={"score": 0.0, "summary": "boundary check failed"},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: SimpleNamespace(),
    )


def test_shell_step_eval_preserves_existing_score_yaml_behavior(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(
        tmp_path / "score_evaluation.sh",
        "score: 1.25\nsummary: shell",
    )
    workspace = _make_evaluation_workspace(tmp_path)
    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

    score, logs = evaluator(workspace)

    assert score == {"score": 1.25, "summary": "shell"}
    assert logs["score.yaml"] == "score: 1.25\nsummary: shell\n"
    assert logs["steps/step_01_score_evaluation/status.txt"] == "ok\n"
    eval_ws = tmp_path / "run" / "evaluation_workspaces" / "candidate-a"
    assert (eval_ws / "logs" / "evaluate" / "score.yaml").exists()
    assert not (workspace / "logs" / "evaluate").exists()


def test_eval_reads_score_yaml_and_preserves_log_tree(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = tmp_path / "evaluation.sh"
    shell_step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'mkdir -p "$EVE_EVAL_LOG_ROOT"',
                "cat <<'EOF' > \"$EVE_EVAL_LOG_ROOT/score.yaml\"",
                "score: 1.25",
                "summary: good run",
                "EOF",
                'printf "good run\\n" > "$EVE_EVAL_LOG_ROOT/summary.txt"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)

    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

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
    workspace = tmp_path / "workspace"
    (workspace / "solver").mkdir(parents=True)
    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

    with caplog.at_level(logging.INFO):
        score, logs = evaluator(workspace)

    assert score["score"] == 0.5
    assert "score: 0.5" in logs["score.yaml"]
    assert "Evaluation: starting workspace `workspace`" in caplog.text
    assert "Evaluation: starting shell step 01 `evaluation_progress.sh`" in caplog.text
    assert "waiting for completion, logs under" in caplog.text
    assert "Evaluation: shell step 01 `evaluation_progress.sh` finished successfully" in caplog.text
    assert "Evaluation: parsed score for workspace `workspace`" in caplog.text


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
    workspace = tmp_path / "workspace"
    (workspace / "solver").mkdir(parents=True)

    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

    score, logs = evaluator(workspace)

    assert score == {"score": 0.0, "summary": "evaluation failed"}
    assert logs["steps/step_01_evaluation_fail/status.txt"] == "failed\n"
    assert logs["steps/step_01_evaluation_fail/stdout.txt"] == "hello\n"
    assert logs["steps/step_01_evaluation_fail/stderr.txt"] == "boom\n"


def test_eval_rejects_invalid_score_yaml(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(tmp_path / "invalid_score.sh", "- not-a-score-card")
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)

    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

    score, logs = evaluator(repo_root)

    assert score == {"score": 0.0, "summary": "evaluation failed"}
    assert logs["score.yaml"] == "- not-a-score-card\n"


def test_eval_failure_score_numeric_shape_is_required(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(
        tmp_path / "missing_headline_score.sh",
        "performance: 0.0\nsummary: invalid solver",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)
    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

    score, logs = evaluator(repo_root)

    assert score == {"score": 0.0, "summary": "evaluation failed"}
    assert logs["score.yaml"] == "performance: 0.0\nsummary: invalid solver\n"


def test_eval_failure_score_numeric_shape_allows_dimension_scores(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(
        tmp_path / "dimension_score.sh",
        "\n".join(
            [
                "dimensions:",
                "  coverage: 0.2",
                "  correctness: 0.3",
                "  dependency: 0.4",
                "  clarity: 0.5",
                "  strategy: 0.6",
            ]
        ),
    )
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)
    failure_score = {
        "dimensions": {
            "coverage": 0.0,
            "correctness": 0.0,
            "dependency": 0.0,
            "clarity": 0.0,
            "strategy": 0.0,
        }
    }
    evaluator = _shell_step_evaluator(
        problem,
        shell_step,
        tmp_path,
        evaluation_failure_score=failure_score,
    )

    score, _logs = evaluator(repo_root)

    assert score == {
        "dimensions": {
            "coverage": 0.2,
            "correctness": 0.3,
            "dependency": 0.4,
            "clarity": 0.5,
            "strategy": 0.6,
        }
    }


def test_eval_fails_when_score_file_is_missing(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    shell_step = tmp_path / "missing_score.sh"
    shell_step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'mkdir -p "$EVE_EVAL_LOG_ROOT"',
                'printf "missing score\\n" > "$EVE_EVAL_LOG_ROOT/summary.txt"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "logs" / "evaluate").mkdir(parents=True)

    evaluator = _shell_step_evaluator(problem, shell_step, tmp_path)

    score, logs = evaluator(repo_root)

    assert score == {"score": 0.0, "summary": "evaluation failed"}
    assert logs["summary.txt"] == "missing score\n"


def test_judge_verdict_recovered_when_file_not_written(tmp_path: Path) -> None:
    # A programmatic sh step lands {performance: 0.5}; the following judge writes NOTHING to
    # score.yaml but its rollout completion carries a fenced verdict {scientific: 0.9}. The engine
    # (_capture_judge_score) must RECOVER that verdict and MERGE it onto the prior accumulated
    # score, so the judge dim is added and the prior dim is kept (not silently dropped).
    problem = _make_problem(tmp_path)
    shell_step = _write_score_shell_step(
        tmp_path / "performance.sh",
        "performance: 0.5",
    )
    workspace = tmp_path / "run" / "solver_workspaces" / "candidate-a"
    workspace.mkdir(parents=True)

    class _NoFileJudgeDriver:
        def spawn(self, seed: object) -> object:
            eval_ws = Path(seed.working_directory)
            # The sh step's prior dim must already be accumulated when the judge runs.
            prior = yaml.safe_load(
                (eval_ws / "logs" / "evaluate" / "score.yaml").read_text(encoding="utf-8")
            )
            assert prior == {"performance": 0.5}
            # Fake judge writes NO score.yaml; the engine must capture the fenced verdict instead.
            completion_path = tmp_path / "judge-completion.json"
            completion_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "summary": "\n".join(
                            [
                                "Final verdict:",
                                "```yaml",
                                "scientific: 0.9",
                                "```",
                            ]
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(
                summary="judge transcript",
                state=SimpleNamespace(metadata={"completion_path": str(completion_path)}),
            )

    plan = EvaluationPlan(
        steps=(
            EvaluationStep(name="performance", path=shell_step),
            _judge_step("assess", immutable_files={"README.md": "rubric\n"}),
        ),
    )
    evaluator = _judge_evaluator(
        problem,
        plan,
        _NoFileJudgeDriver(),
        evaluation_failure_score={"performance": 0.0, "scientific": 0.0},
    )
    score, _logs = evaluator(
        workspace,
        candidate_files={"candidate.py": "print('candidate')\n"},
        optimize_logs={"transcript.txt": "solver transcript\n"},
    )

    # Prior dim kept, judge dim added — recovered verdict MERGED onto the prior, not dropped.
    assert score == {"performance": 0.5, "scientific": 0.9}
