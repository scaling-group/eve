from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve import runner as runner_module
from scaling_evolve.algorithms.eve.workflow.evaluation import build_solver_evaluator

_LOOP_CFG = {
    "n_workers_phase2": 1,
    "instructions": {
        "phase2_readme": {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.default.Phase2ReadmeInstruction"
            ),
        },
        "phase2_entrypoint": {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.default.Phase2EntrypointInstruction"
            ),
        },
        "phase2_agent": {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.default.Phase2AgentInstruction"
            ),
        },
        "phase4_readme": {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.default.Phase4ReadmeInstruction"
            ),
        },
        "phase4_entrypoint": {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.default.Phase4EntrypointInstruction"
            ),
        },
        "phase4_agent": {
            "_target_": (
                "scaling_evolve.algorithms.eve.instructions.default.Phase4AgentInstruction"
            ),
        },
    },
}


def _make_cfg(run_root: Path, *, logger_cfg: dict | None = None) -> object:
    run_id = "run-20260411_000000_000000-e2e-short-abcdef123456"
    raw = {
        "application": {
            "name": "fake-problem",
            "github_url": "https://example.com/fake-problem",
            "evaluation_failure_score": {"score": 0.0, "summary": "evaluation failed"},
            "boundary_failure_score": {"score": 0.0, "summary": "boundary check failed"},
            "seed_solver_score": None,
            "seed_solver_skip_evaluation": False,
        },
        "optimizer": {
            "initial_optimizer": None,
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
            self.optimizer_driver = object()

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
            self.optimizer_driver = object()

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

        def seed_initial_optimizer(self, *, search_root: Path) -> None:
            _ = search_root
            self.seeded = True

        def run(self) -> None:
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


def test_runner_skips_finish_when_logger_group_returns_none(monkeypatch, tmp_path: Path) -> None:
    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()
            self.optimizer_driver = object()

        def close(self) -> None:
            return None

    class _FakeFactory:
        def __init__(self) -> None:
            self.loop = SimpleNamespace(
                solver_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                optimizer_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                iterations_completed=0,
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_optimizer(self, *, search_root: Path) -> None:
            _ = search_root

        def run(self) -> None:
            return None

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root, logger_cfg={"enabled": False})

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
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: _FakeFactory()),
    )

    runner_module.run(cfg)


def test_runner_logs_and_exits_on_remote_transport_halt(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()
            self.optimizer_driver = object()

        def close(self) -> None:
            return None

    class _FakeFactory:
        def __init__(self) -> None:
            self.loop = SimpleNamespace(
                solver_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                optimizer_pop=SimpleNamespace(entries=lambda: [], size=lambda: 0),
                iterations_completed=0,
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_optimizer(self, *, search_root: Path) -> None:
            _ = search_root

        def run(self) -> None:
            raise runner_module.RemoteTransportHaltError(
                step=Path("evaluate.sh"),
                workspace_root=Path("/tmp/workspace"),
                returncode=1,
                stdout="",
                stderr="[REMOTE-L4] breaker open window exceeded",
            )

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root, logger_cfg={"enabled": False})

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
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: _FakeFactory()),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner_module.run(cfg)

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "[REMOTE-HALT]" in captured.out


def test_runner_skips_optimizer_seed_when_restoring_from_prior_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeDrivers:
        def __init__(self) -> None:
            self.eval_driver_factory = lambda: None
            self.solver_driver = object()
            self.optimizer_driver = object()

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
            self.restore_specs: list[object] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def seed_initial_optimizer(self, *, search_root: Path) -> None:
            _ = search_root
            self.seed_calls += 1

        def restore_from_spec(self, restore_spec: object) -> object:
            self.restore_specs.append(restore_spec)
            return SimpleNamespace(
                source_run_root=Path("/tmp/prior-run"),
                solver=SimpleNamespace(entries_restored=1),
                optimizer=SimpleNamespace(entries_restored=1),
            )

        def run(self) -> None:
            return None

    run_root = tmp_path / "artifacts"
    cfg = _make_cfg(run_root, logger_cfg={"enabled": False})
    cfg.restore_from = str(tmp_path / "prior-run")
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
    monkeypatch.setattr(runner_module, "build_solver_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner_module.EveFactory,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: factory),
    )

    runner_module.run(cfg)

    assert factory.seed_calls == 0
    assert len(factory.restore_specs) == 1


def test_build_solver_evaluator_preserves_nullable_seed_score() -> None:
    evaluator = build_solver_evaluator(
        SimpleNamespace(evaluation_steps=()),
        evaluation_failure_score={"score": 0.0},
        boundary_failure_score={"score": 0.0},
        seed_solver_score=None,
        seed_solver_skip_evaluation=False,
        evaluation_driver_factory=lambda: None,
    )

    assert evaluator.seed_solver_score is None


def _compose_eve_config(config_name: str, overrides: list[str] | None = None):
    if not OmegaConf.has_resolver("eve_run_id"):
        OmegaConf.register_new_resolver(
            "eve_run_id",
            runner_module.build_run_id,
            use_cache=True,
        )

    config_dir = Path(__file__).resolve().parents[1] / "configs" / "eve"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_icon_config_defaults() -> None:
    cfg = _compose_eve_config("icon")

    assert cfg.prompt.phase2_readme._target_ == (
        "scaling_evolve.algorithms.eve.instructions.icon.Phase2ReadmeInstruction"
    )
    assert cfg.prompt.phase2_readme.file_list[1] == (
        "configs/eve/prompt/templates/icon/phase2_self_optimize_readme.md"
    )
    assert cfg.loop.produce_optimizer_in_phase2 == cfg.loop.n_workers_phase2
    assert cfg.loop.n_workers_phase4 == 0


def test_icon_fixed_optimizer_disables_optimizer_evolution() -> None:
    cfg = _compose_eve_config("icon", overrides=["loop=fixed_optimizer"])

    assert cfg.loop.produce_optimizer_in_phase2 == 0
    assert cfg.loop.n_optimizer_examples_phase2 == 0
    assert cfg.loop.n_workers_phase4 == 0


def test_icon_smoke_config() -> None:
    cfg = _compose_eve_config("icon.smoke")

    assert cfg.label == "icon-pe-demo-smoke"
    assert cfg.loop.max_iterations == 2
    assert cfg.logger.enabled is False
    assert cfg.application.evaluation_steps == ["configs/eve/application/icon/evaluate_short.sh"]


def test_circle_packing_config_defaults() -> None:
    cfg = _compose_eve_config("circle_packing")

    assert cfg.prompt.phase2_readme._target_ == (
        "scaling_evolve.algorithms.eve.instructions.default.Phase2ReadmeInstruction"
    )
    assert cfg.prompt.phase2_readme.file_list[1] == (
        "configs/eve/prompt/templates/built_in/phase2_self_optimize_readme.md"
    )
    assert cfg.loop.produce_optimizer_in_phase2 == cfg.loop.n_workers_phase2
    assert cfg.loop.n_workers_phase4 == 0


def test_circle_packing_smoke_config() -> None:
    cfg = _compose_eve_config("circle_packing.smoke")

    assert cfg.label == "circle-packing-smoke"
    assert cfg.loop.max_iterations == 3
    assert cfg.driver.rollout_max_turns == 3
