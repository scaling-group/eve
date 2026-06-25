from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve import runner as runner_module
from scaling_evolve.algorithms.eve.workflow.evaluation import (
    EvaluationPlan,
    build_solver_evaluator,
)

_LOOP_CFG = {
    "n_workers_phase2": 1,
}


def _make_cfg(run_root: Path, *, logger_cfg: dict | None = None) -> object:
    run_id = "run-20260411_000000_000000-e2e-short-abcdef123456"
    raw = {
        "application": {
            "name": "fake-problem",
            "github_url": "https://github.com/example/fake-problem",
            "boundary_failure_score": {"score": 0.0, "summary": "boundary check failed"},
        },
        "evaluation": {
            "steps": ["configs/eve/evaluation/circle_packing/evaluation.sh"],
            "failure_score": {"score": 0.0, "summary": "evaluation failed"},
            "seed_solver_score": None,
            "seed_solver_skip_evaluation": False,
        },
        "optimizer": {
            "initial_guidance": None,
            "immutable": "configs/eve/optimizer/circle_packing/immutable",
            "prompt": "configs/eve/optimizer/circle_packing/prompt",
            "immutable_renderer": {
                "_target_": (
                    "scaling_evolve.algorithms.eve.workspace.immutable_renderers.default."
                    "DefaultRenderer"
                ),
            },
            "evaluation": {
                "_target_": (
                    "scaling_evolve.algorithms.eve.populations.evaluators.elo.ScalarEloEvaluator"
                ),
                "k_factor": 32.0,
                "initial_score": {"elo": 1500.0},
            },
        },
        "driver": {"provider": "codex_tmux", "open_iterm2": False},
        "loop": {
            **dict(_LOOP_CFG),
            "workspace_root": "${run_root}",
            "artifact_root": "${loop.workspace_root}/artifacts",
            "solver_db_path": "${loop.workspace_root}/solver_lineage.db",
            "optimizer_db_path": "${loop.workspace_root}/optimizer_lineage.db",
        },
        "label": "e2e-short",
        "run_id": run_id,
        "run_root": str(run_root),
    }
    if logger_cfg is not None:
        raw["logger"] = logger_cfg
    return OmegaConf.create(raw)


def test_runner_closes_drivers_when_logger_instantiate_fails(monkeypatch, tmp_path: Path) -> None:
    close_called = False

    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()

        def close(self) -> None:
            nonlocal close_called
            close_called = True

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root, logger_cfg={"_target_": "fake.logger.Target"})

    monkeypatch.setattr(runner_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner_module, "write_repo_codex_hooks", lambda repo_root: None)
    monkeypatch.setattr(runner_module, "ensure_codex_hooks_trusted", lambda repo_root: None)
    monkeypatch.setattr(
        runner_module.RepoTaskProblem,
        "from_config",
        classmethod(
            lambda cls, application, cache_root, search_root: SimpleNamespace(slug="fake-problem")
        ),
    )
    monkeypatch.setattr(runner_module, "build_role_drivers", lambda *args, **kwargs: _FakeDrivers())
    monkeypatch.setattr(runner_module, "build_evaluation_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module,
        "instantiate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("logger boom")),
    )

    with pytest.raises(RuntimeError, match="logger boom"):
        runner_module.run(cfg)

    assert close_called is True
    assert run_root.exists()


def test_runner_instantiates_logger_non_recursively(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()

        def close(self) -> None:
            return None

    class _FakeFactory:
        def __init__(self) -> None:
            self.loop = SimpleNamespace(
                solver_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                optimizer_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                iterations_completed=0,
            )
            self.seeded = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_guidance(self, *, search_root: Path) -> None:
            _ = search_root
            self.seeded = True

        def run(self, *, start_iteration: int = 0) -> None:
            _ = start_iteration
            return None

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root, logger_cfg={"_target_": "fake.logger.Target", "enabled": False})

    monkeypatch.setattr(runner_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner_module, "write_repo_codex_hooks", lambda repo_root: None)
    monkeypatch.setattr(runner_module, "ensure_codex_hooks_trusted", lambda repo_root: None)
    monkeypatch.setattr(
        runner_module.RepoTaskProblem,
        "from_config",
        classmethod(
            lambda cls, application, cache_root, search_root: SimpleNamespace(slug="fake-problem")
        ),
    )
    monkeypatch.setattr(runner_module, "build_role_drivers", lambda *args, **kwargs: _FakeDrivers())
    monkeypatch.setattr(runner_module, "build_evaluation_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: _FakeFactory()),
    )

    class _FakeLogger:
        def finish(self, **kwargs: object) -> None:
            _ = kwargs

    def _fake_instantiate(*args: object, **kwargs: object) -> object:
        _ = args
        captured_kwargs.update(kwargs)
        return _FakeLogger()

    monkeypatch.setattr(runner_module, "instantiate", _fake_instantiate)

    runner_module.run(cfg)
    assert captured_kwargs["_recursive_"] is False


def test_runner_normalizes_structured_score_payloads(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()

        def close(self) -> None:
            return None

    class _FakeFactory:
        loop = SimpleNamespace(
            solver_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
            optimizer_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
            iterations_completed=0,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_guidance(self, *, search_root: Path) -> None:
            _ = search_root

        def run(self, *, start_iteration: int = 0) -> None:
            _ = start_iteration

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root)
    cfg.evaluation.failure_score = {
        "score": 0.0,
        "dimensions": {"coverage": 0.0},
    }
    cfg.application.boundary_failure_score = {
        "score": 0.0,
        "dimensions": {"coverage": 0.0},
    }
    cfg.evaluation.seed_solver_score = {
        "score": 0.0,
        "dimensions": {"coverage": 0.0},
    }
    cfg.evaluation.seed_solver_skip_evaluation = True

    monkeypatch.setattr(runner_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner_module, "write_repo_codex_hooks", lambda repo_root: None)
    monkeypatch.setattr(runner_module, "ensure_codex_hooks_trusted", lambda repo_root: None)
    monkeypatch.setattr(
        runner_module.RepoTaskProblem,
        "from_config",
        classmethod(
            lambda cls, application, cache_root, search_root: SimpleNamespace(slug="fake-problem")
        ),
    )
    monkeypatch.setattr(runner_module, "build_role_drivers", lambda *args, **kwargs: _FakeDrivers())
    monkeypatch.setattr(
        runner_module,
        "build_solver_evaluator",
        lambda *args, **kwargs: captured_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: _FakeFactory()),
    )

    runner_module.run(cfg)

    for key in (
        "evaluation_failure_score",
        "boundary_failure_score",
        "seed_solver_score",
    ):
        score = captured_kwargs[key]
        assert isinstance(score, dict)
        assert isinstance(score["dimensions"], dict)
        assert not OmegaConf.is_config(score)
        assert not OmegaConf.is_config(score["dimensions"])


def test_build_solver_evaluator_preserves_nullable_seed_score() -> None:
    evaluator = build_solver_evaluator(
        SimpleNamespace(),
        evaluation_plan=EvaluationPlan(steps=()),
        evaluation_failure_score={"score": 0.0},
        boundary_failure_score={"score": 0.0},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: None,
    )

    assert evaluator.seed_solver_score is None


def test_runner_skips_optimizer_seed_when_importing_from_prior_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()

        def close(self) -> None:
            return None

    class _FakeFactory:
        def __init__(self) -> None:
            self.loop = SimpleNamespace(
                solver_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                optimizer_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                iterations_completed=0,
            )
            self.seed_calls = 0
            self.import_specs: list[object] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_guidance(self, *, search_root: Path) -> None:
            _ = search_root
            self.seed_calls += 1

        def import_from_spec(self, import_spec: object) -> object:
            self.import_specs.append(import_spec)
            return SimpleNamespace(
                source_run_root=Path("/tmp/prior-run"),
                solver=SimpleNamespace(entries_imported=1),
                optimizer=SimpleNamespace(entries_imported=1),
            )

        def run(self, *, start_iteration: int = 0) -> None:
            _ = start_iteration
            return None

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root, logger_cfg={"enabled": False})
    cfg.import_from = str(tmp_path / "prior-run")
    factory = _FakeFactory()

    monkeypatch.setattr(runner_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner_module, "write_repo_codex_hooks", lambda repo_root: None)
    monkeypatch.setattr(runner_module, "ensure_codex_hooks_trusted", lambda repo_root: None)
    monkeypatch.setattr(
        runner_module.RepoTaskProblem,
        "from_config",
        classmethod(
            lambda cls, application, cache_root, search_root: SimpleNamespace(slug="fake-problem")
        ),
    )
    monkeypatch.setattr(runner_module, "build_role_drivers", lambda *args, **kwargs: _FakeDrivers())
    monkeypatch.setattr(runner_module, "build_evaluation_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: factory),
    )

    runner_module.run(cfg)

    assert factory.seed_calls == 0
    assert len(factory.import_specs) == 1


def test_runner_resume_from_pins_run_id_and_start_iteration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()

        def close(self) -> None:
            return None

    class _FakeFactory:
        def __init__(self) -> None:
            self.loop = SimpleNamespace(
                solver_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                optimizer_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                iterations_completed=7,
            )
            self.seed_calls = 0
            self.start_iteration: int | None = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_guidance(self, *, search_root: Path) -> None:
            _ = search_root
            self.seed_calls += 1

        def run(self, *, start_iteration: int = 0) -> None:
            self.start_iteration = start_iteration

    run_root = tmp_path / "resume-run"
    cfg = _make_cfg(tmp_path / "fresh-run", logger_cfg={"enabled": False})
    cfg.resume_from = str(run_root)
    cfg.resume_iteration = 7
    factory = _FakeFactory()
    captured_config: dict[str, object] = {}
    captured_resume: dict[str, object] = {}
    captured_logger: dict[str, object] = {}

    monkeypatch.setattr(runner_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner_module, "write_repo_codex_hooks", lambda repo_root: None)
    monkeypatch.setattr(runner_module, "ensure_codex_hooks_trusted", lambda repo_root: None)

    def _fake_prepare_resume(run_root_arg, resume_iteration=None):  # noqa: ANN001
        captured_resume.update(
            {
                "run_root": str(run_root_arg),
                "resume_iteration": resume_iteration,
            }
        )
        return SimpleNamespace(
            run_root=Path(run_root_arg),
            run_id="run-original",
            start_iteration=7,
            checkpoint=SimpleNamespace(max_iterations=9),
        )

    monkeypatch.setattr(
        runner_module,
        "prepare_resume",
        _fake_prepare_resume,
    )
    monkeypatch.setattr(
        runner_module.RepoTaskProblem,
        "from_config",
        classmethod(
            lambda cls, application, cache_root, search_root: SimpleNamespace(slug="fake-problem")
        ),
    )
    monkeypatch.setattr(runner_module, "build_role_drivers", lambda *args, **kwargs: _FakeDrivers())
    monkeypatch.setattr(runner_module, "build_evaluation_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module,
        "_instantiate_eve_logger_group",
        lambda logger_cfg, run_config, *, resume_anchor_iteration=None: (
            captured_logger.update({"resume_anchor_iteration": resume_anchor_iteration}) or None
        ),
    )

    def _fake_from_config(cls, run_config, *args, **kwargs):  # noqa: ANN001, ARG001
        captured_config["run_id"] = str(run_config.run_id)
        captured_config["run_root"] = str(run_config.run_root)
        captured_config["max_iterations"] = int(run_config.loop.max_iterations)
        captured_config["workspace_root"] = str(run_config.loop.workspace_root)
        return factory

    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(_fake_from_config),
    )

    runner_module.run(cfg)

    assert captured_config == {
        "run_id": "run-original",
        "run_root": str(run_root),
        "max_iterations": 9,
        "workspace_root": str(run_root),
    }
    assert captured_resume == {
        "run_root": str(run_root),
        "resume_iteration": 7,
    }
    assert captured_logger == {"resume_anchor_iteration": 7}
    assert factory.seed_calls == 0
    assert factory.start_iteration == 7


def test_runner_resume_writes_anchor_summary_before_next_iteration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    solver_entries = [object()]
    optimizer_entries = [object()]

    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()

        def close(self) -> None:
            return None

    class _FakeLogger:
        def __init__(self) -> None:
            self.anchor_summary: dict[str, object] | None = None

        def write_resume_anchor_summary(self, **kwargs: object) -> None:
            events.append("anchor_summary")
            self.anchor_summary = dict(kwargs)

        def finish(self, **kwargs: object) -> None:
            _ = kwargs
            events.append("finish")

    class _FakeFactory:
        def __init__(self) -> None:
            self.loop = SimpleNamespace(
                solver_pop=SimpleNamespace(entries=lambda: solver_entries, size=lambda: 1),
                optimizer_pop=SimpleNamespace(entries=lambda: optimizer_entries, size=lambda: 1),
                iterations_completed=3,
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_guidance(self, *, search_root: Path) -> None:
            _ = search_root

        def run(self, *, start_iteration: int = 0) -> None:
            assert start_iteration == 2
            events.append("run")

    run_root = tmp_path / "resume-run"
    cfg = _make_cfg(run_root, logger_cfg={"enabled": False})
    cfg.resume_from = str(run_root)
    fake_logger = _FakeLogger()

    monkeypatch.setattr(runner_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner_module, "write_repo_codex_hooks", lambda repo_root: None)
    monkeypatch.setattr(runner_module, "ensure_codex_hooks_trusted", lambda repo_root: None)
    monkeypatch.setattr(
        runner_module,
        "prepare_resume",
        lambda run_root_arg, resume_iteration=None: SimpleNamespace(
            run_root=Path(run_root_arg),
            run_id="run-original",
            start_iteration=2,
            checkpoint=SimpleNamespace(max_iterations=4),
        ),
    )
    monkeypatch.setattr(
        runner_module.RepoTaskProblem,
        "from_config",
        classmethod(
            lambda cls, application, cache_root, search_root: SimpleNamespace(slug="fake-problem")
        ),
    )
    monkeypatch.setattr(runner_module, "build_role_drivers", lambda *args, **kwargs: _FakeDrivers())
    monkeypatch.setattr(runner_module, "build_evaluation_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module,
        "_instantiate_eve_logger_group",
        lambda *args, **kwargs: fake_logger,
    )
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: _FakeFactory()),
    )

    runner_module.run(cfg)

    assert events == ["anchor_summary", "run", "finish"]
    assert fake_logger.anchor_summary == {
        "solver_entries": solver_entries,
        "optimizer_entries": optimizer_entries,
        "iterations_completed": 2,
    }


def test_runner_rejects_resume_iteration_without_resume_from(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "artifacts", logger_cfg={"enabled": False})
    cfg.resume_iteration = 2

    with pytest.raises(SystemExit, match="requires `resume_from`"):
        runner_module.run(cfg)


def test_runner_rejects_negative_resume_iteration(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "artifacts", logger_cfg={"enabled": False})
    cfg.resume_from = str(tmp_path / "resume-run")
    cfg.resume_iteration = -1

    with pytest.raises(SystemExit, match="non-negative integer"):
        runner_module.run(cfg)


def test_runner_rejects_resume_from_and_import_from_together(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "artifacts", logger_cfg={"enabled": False})
    cfg.resume_from = str(tmp_path / "resume-run")
    cfg.import_from = str(tmp_path / "prior-run")

    with pytest.raises(SystemExit, match="mutually exclusive"):
        runner_module.run(cfg)
