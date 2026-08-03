from __future__ import annotations

from pathlib import Path

import pytest

from scaling_evolve.algorithms.eve.runtime.driver import (
    build_driver,
    build_role_drivers,
    load_pricing_table,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver
from scaling_evolve.providers.agent.drivers.codex_tmux import CodexTmuxSessionDriver


def test_build_role_drivers_opens_iterm2_for_codex_pool(monkeypatch, tmp_path: Path) -> None:
    opened: list[str] = []
    pool_kwargs: dict[str, object] = {}

    class _FakePool:
        def __init__(self) -> None:
            self.session_name = "codex-pool-test"
            self.cwd = tmp_path

        def close(self) -> None:
            return None

    def _create_pool(**kwargs):  # noqa: ANN003
        pool_kwargs.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.CodexTmuxPanePool.create",
        _create_pool,
    )
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.open_iterm2_window_for_session",
        lambda session_name: opened.append(session_name),
    )

    drivers = build_role_drivers(
        {
            "provider": "codex_tmux",
            "model": "gpt-5.4-mini",
            "open_iterm2": True,
        },
        run_root=tmp_path / "run-root",
        worker_slots=2,
    )

    assert opened == ["codex-pool-test"]
    assert pool_kwargs["pane_count"] == 2
    drivers.close()


def test_driver_builders_reject_legacy_pool_size(tmp_path: Path) -> None:
    config = {
        "driver": "codex_tmux",
        "pool_size": 2,
        "open_iterm2": False,
    }
    message = r"driver\.pool_size.*loop\.n_parallel_phase2"

    with pytest.raises(SystemExit, match=message):
        build_role_drivers(config, run_root=tmp_path / "role-run", worker_slots=2)
    with pytest.raises(SystemExit, match=message):
        build_driver(config, run_root=tmp_path / "standalone-run")


def test_build_driver_uses_worker_slots_for_standalone_tmux_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pool_kwargs: dict[str, object] = {}

    class _FakePool:
        session_name = "standalone-pool-test"
        cwd = tmp_path

        def close(self) -> None:
            return None

    def _create_pool(**kwargs):  # noqa: ANN003
        pool_kwargs.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.CodexTmuxPanePool.create",
        _create_pool,
    )

    driver = build_driver(
        {"driver": "codex_tmux"},
        run_root=tmp_path / "standalone-run",
        worker_slots=3,
    )

    assert pool_kwargs["pane_count"] == 3
    driver.close()


def test_build_role_drivers_uses_workspace_write_when_web_search_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakePool:
        def __init__(self) -> None:
            self.session_name = "codex-pool-test"
            self.cwd = tmp_path

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.CodexTmuxPanePool.create",
        lambda **kwargs: _FakePool(),
    )
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.open_iterm2_window_for_session",
        lambda session_name: None,
    )

    drivers = build_role_drivers(
        {
            "provider": "codex_tmux",
            "model": "gpt-5.4-mini",
            "rollout_max_turns": 16,
            "budget_prompt": False,
            "web_search": "disabled",
        },
        run_root=tmp_path / "run-root",
        worker_slots=2,
    )

    solver_driver = drivers.solver_driver
    assert solver_driver.sandbox_mode == "workspace-write"
    assert solver_driver.rollout_max_turns == 16
    assert solver_driver.budget_prompt is False
    assert solver_driver.web_search == "disabled"
    drivers.close()


def test_build_role_drivers_injects_model_provider_env(monkeypatch, tmp_path: Path) -> None:
    class _FakePool:
        def __init__(self) -> None:
            self.session_name = "codex-pool-test"
            self.cwd = tmp_path

        def close(self) -> None:
            return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.CodexTmuxPanePool.create",
        lambda **kwargs: _FakePool(),
    )
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.open_iterm2_window_for_session",
        lambda session_name: None,
    )

    drivers = build_role_drivers(
        {
            "provider": "codex_tmux",
            "model": "openai/gpt-5.4-mini",
            "model_provider": "openrouter",
            "model_providers": {
                "openrouter": {
                    "name": "OpenRouter",
                    "base_url": "https://openrouter.ai/api/v1",
                    "env_key": "OPENROUTER_API_KEY",
                    "wire_api": "responses",
                }
            },
        },
        run_root=tmp_path / "run-root",
        worker_slots=2,
    )

    solver_driver = drivers.solver_driver
    assert solver_driver.model_provider == "openrouter"
    assert solver_driver.provider_env == {"OPENROUTER_API_KEY": "test-key"}
    drivers.close()


def test_build_role_drivers_creates_shared_pool_for_codex_tmux_eval_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []

    class _FakePool:
        def __init__(self) -> None:
            self.session_name = "mixed-pool-test"
            self.cwd = tmp_path

        def close(self) -> None:
            return None

        def acquire(self, *, preferred_pane_id=None):  # noqa: ANN001, ARG002
            return "%9"

        def release(self, pane_id):  # noqa: ANN001, ARG002
            return None

        def reset_idle_banner(self, pane_id):  # noqa: ANN001, ARG002
            return None

    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.CodexTmuxPanePool.create",
        lambda **kwargs: _FakePool(),
    )
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.runtime.driver.open_iterm2_window_for_session",
        lambda session_name: opened.append(session_name),
    )

    drivers = build_role_drivers(
        {
            "driver": "codex_exec",
            "model": "gpt-5.4-mini",
            "open_iterm2": True,
            "overrides": {
                "eval": {
                    "driver": "codex_tmux",
                    "model": "gpt-5.4-mini",
                }
            },
        },
        run_root=tmp_path / "run-root",
        worker_slots=2,
    )

    assert isinstance(drivers.solver_driver, CodexExecSessionDriver)
    eval_driver = drivers.eval_driver_factory()
    assert isinstance(eval_driver, CodexTmuxSessionDriver)
    assert drivers.pane_pool is not None
    assert opened == ["mixed-pool-test"]
    drivers.close()


@pytest.mark.parametrize("driver_name", ["legacy_exec", "legacy_tmux", "unsupported_cli"])
def test_build_role_drivers_rejects_unsupported_drivers(
    driver_name: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="Unsupported driver"):
        build_role_drivers(
            {
                "driver": driver_name,
                "model": "gpt-5.4-mini",
            },
            run_root=tmp_path / "run-root",
            worker_slots=1,
        )


def test_build_role_drivers_builds_codex_exec_driver(tmp_path: Path) -> None:
    drivers = build_role_drivers(
        {
            "driver": "codex_exec",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "medium",
            "rollout_max_turns": 12,
            "budget_prompt": False,
            "web_search": "disabled",
        },
        run_root=tmp_path / "run-root",
        worker_slots=1,
    )

    assert isinstance(drivers.solver_driver, CodexExecSessionDriver)
    assert drivers.solver_driver.reasoning_effort == "medium"
    assert drivers.solver_driver.rollout_max_turns == 12
    assert drivers.solver_driver.budget_prompt is False
    assert drivers.solver_driver.web_search == "disabled"


def test_build_role_drivers_resolves_system_prompt_per_role(
    tmp_path: Path,
) -> None:
    # Generic `system_prompt_file` config key set only on the solver role; presence
    # is the switch. The codex driver receives it as `codex_system_prompt_file`.
    rel = "configs/eve/optimizer/circle_packing/prompt/CODEX_SYSTEM_PROMPT.md"
    drivers = build_role_drivers(
        {
            "driver": "codex_exec",
            "model": "gpt-5.4-mini",
            "overrides": {
                "solver": {"system_prompt_file": rel},
            },
        },
        run_root=tmp_path / "run-root",
        worker_slots=1,
    )

    assert isinstance(drivers.solver_driver, CodexExecSessionDriver)
    solver_path = drivers.solver_driver.codex_system_prompt_file
    assert solver_path is not None
    # Resolved to an absolute path anchored at the repo root (the shipped file).
    assert Path(solver_path).is_absolute()
    assert Path(solver_path).is_file()
    assert solver_path.endswith(rel)
    # No override on eval -> Codex keeps its built-in system prompt.
    eval_driver = drivers.eval_driver_factory()
    assert isinstance(eval_driver, CodexExecSessionDriver)
    assert eval_driver.codex_system_prompt_file is None


def test_build_role_drivers_plumbs_codex_multi_agent_role_overrides(
    tmp_path: Path,
) -> None:
    drivers = build_role_drivers(
        {
            "driver": "codex_exec",
            "model": "gpt-5.4-mini",
            "enable_multi_agent": True,
            "overrides": {
                "eval": {
                    "enable_multi_agent": False,
                }
            },
        },
        run_root=tmp_path / "run-root",
        worker_slots=1,
    )

    assert isinstance(drivers.solver_driver, CodexExecSessionDriver)
    assert drivers.solver_driver.enable_multi_agent is True
    eval_driver = drivers.eval_driver_factory()
    assert isinstance(eval_driver, CodexExecSessionDriver)
    assert eval_driver.enable_multi_agent is False


def test_load_pricing_table_reads_yaml(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "\n".join(
            [
                "gpt-5.4-mini:",
                "  input_per_million: 0.75",
                "  output_per_million: 4.5",
                "  cache_read_per_million: 0.075",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    table = load_pricing_table(pricing_path)

    assert table["gpt-5.4-mini"].input_per_million == pytest.approx(0.75)
    assert table["gpt-5.4-mini"].output_per_million == pytest.approx(4.5)
    assert table["gpt-5.4-mini"].cache_read_per_million == pytest.approx(0.075)
