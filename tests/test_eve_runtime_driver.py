from __future__ import annotations

from pathlib import Path

import pytest

from scaling_evolve.algorithms.eve.runtime.driver import (
    build_role_drivers,
    load_pricing_table,
)
from scaling_evolve.providers.agent.drivers.claude_code import ClaudeCodeSessionDriver
from scaling_evolve.providers.agent.drivers.claude_code_tmux import ClaudeCodeTmuxSessionDriver
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver
from scaling_evolve.providers.agent.drivers.codex_tmux import CodexTmuxSessionDriver


def test_build_role_drivers_opens_iterm2_for_codex_pool(monkeypatch, tmp_path: Path) -> None:
    opened: list[str] = []

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
        lambda session_name: opened.append(session_name),
    )

    drivers = build_role_drivers(
        {
            "provider": "codex_tmux",
            "model": "gpt-5.4-mini",
            "open_iterm2": True,
        },
        run_root=tmp_path / "run-root",
        workers=2,
    )

    assert opened == ["codex-pool-test"]
    drivers.close()


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
        workers=2,
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
        workers=2,
    )

    solver_driver = drivers.solver_driver
    assert solver_driver.model_provider == "openrouter"
    assert solver_driver.provider_env == {"OPENROUTER_API_KEY": "test-key"}
    drivers.close()


def test_build_role_drivers_creates_shared_pool_for_claude_tmux_override(
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
            "provider": "codex_tmux",
            "model": "gpt-5.4-mini",
            "open_iterm2": True,
            "overrides": {
                "optimizer": {
                    "provider": "claude_code_tmux",
                    "model": "claude-sonnet-4-6",
                }
            },
        },
        run_root=tmp_path / "run-root",
        workers=2,
    )

    assert isinstance(drivers.solver_driver, CodexTmuxSessionDriver)
    assert isinstance(drivers.optimizer_driver, ClaudeCodeTmuxSessionDriver)
    assert drivers.pane_pool is not None
    assert opened == ["mixed-pool-test"]
    drivers.close()


def test_build_role_drivers_passes_effort_level_to_claude_tmux_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakePool:
        def __init__(self) -> None:
            self.session_name = "claude-pool-test"
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
        lambda session_name: None,
    )

    drivers = build_role_drivers(
        {
            "provider": "claude_code_tmux",
            "model": "opus",
            "effort_level": "auto",
            "rollout_max_turns": 18,
            "budget_prompt": False,
            "overrides": {
                "eval": {
                    "effort_level": "low",
                }
            },
        },
        run_root=tmp_path / "run-root",
        workers=2,
    )

    assert isinstance(drivers.solver_driver, ClaudeCodeTmuxSessionDriver)
    assert drivers.solver_driver.effort_level == "auto"
    assert drivers.solver_driver.rollout_max_turns == 18
    assert drivers.solver_driver.budget_prompt is False
    eval_driver = drivers.eval_driver_factory()
    assert isinstance(eval_driver, ClaudeCodeTmuxSessionDriver)
    assert eval_driver.effort_level == "low"
    assert eval_driver.rollout_max_turns == 18
    assert eval_driver.budget_prompt is False
    drivers.close()


def test_build_role_drivers_disables_web_tools_for_claude_tmux(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakePool:
        def __init__(self) -> None:
            self.session_name = "claude-pool-test"
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
        lambda session_name: None,
    )

    drivers = build_role_drivers(
        {
            "provider": "claude_code_tmux",
            "model": "opus",
            "web_search": "disabled",
        },
        run_root=tmp_path / "run-root",
        workers=2,
    )

    assert isinstance(drivers.solver_driver, ClaudeCodeTmuxSessionDriver)
    assert drivers.solver_driver.disallowed_tools == ("WebSearch", "WebFetch")
    drivers.close()


def test_build_role_drivers_disables_web_tools_for_claude_noninteractive(tmp_path: Path) -> None:
    drivers = build_role_drivers(
        {
            "driver": "claude_code",
            "model": "sonnet",
            "web_search": "disabled",
            "dangerously_skip_permissions": False,
        },
        run_root=tmp_path / "run-root",
        workers=1,
    )

    assert isinstance(drivers.solver_driver, ClaudeCodeSessionDriver)
    assert drivers.solver_driver.disallowed_tools == ("WebSearch", "WebFetch")
    assert drivers.solver_driver.dangerously_skip_permissions is True
    assert drivers.solver_driver.setting_sources == ("project", "local")


def test_build_role_drivers_passes_budget_prompt_to_claude_noninteractive(tmp_path: Path) -> None:
    drivers = build_role_drivers(
        {
            "driver": "claude_code",
            "model": "sonnet",
            "budget_prompt": False,
        },
        run_root=tmp_path / "run-root",
        workers=1,
    )

    assert isinstance(drivers.solver_driver, ClaudeCodeSessionDriver)
    assert drivers.solver_driver.budget_prompt is False
    eval_driver = drivers.eval_driver_factory()
    assert isinstance(eval_driver, ClaudeCodeSessionDriver)
    assert eval_driver.budget_prompt is False


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
        workers=1,
    )

    assert isinstance(drivers.solver_driver, CodexExecSessionDriver)
    assert drivers.solver_driver.reasoning_effort == "medium"
    assert drivers.solver_driver.rollout_max_turns == 12
    assert drivers.solver_driver.budget_prompt is False
    assert drivers.solver_driver.web_search == "disabled"


def test_load_pricing_table_reads_yaml(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "\n".join(
            [
                "haiku:",
                "  input_per_million: 1.0",
                "  output_per_million: 5.0",
                "  cache_read_per_million: 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    table = load_pricing_table(pricing_path)

    assert table["haiku"].input_per_million == pytest.approx(1.0)
    assert table["haiku"].output_per_million == pytest.approx(5.0)
    assert table["haiku"].cache_read_per_million == pytest.approx(0.1)
