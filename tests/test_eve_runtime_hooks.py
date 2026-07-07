from __future__ import annotations

import json
from pathlib import Path

from scaling_evolve.algorithms.eve.workspace.runtime_hooks import (
    install_workspace_runtime_hooks,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver


def test_install_workspace_runtime_hooks_writes_rollout_prompt_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    driver = CodexExecSessionDriver(run_root=tmp_path / "run-root", rollout_max_turns=20)
    prompt_specs = [
        {
            "name": "budget",
            "system_text": None,
            "user_text": "Turn budget enabled: this session has 20 turns per rollout.",
            "turn_template": "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining",
            "turn_format_kwargs": {"rollout_max_turns": 20},
        }
    ]

    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=prompt_specs)

    prompt_payload = json.loads(
        (workspace / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert not (workspace / ".sandbox_config.json").exists()
    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == prompt_specs


def test_install_workspace_runtime_hooks_leaves_codex_hooks_to_driver_launch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    driver = CodexExecSessionDriver(run_root=tmp_path / "run-root", rollout_max_turns=12)
    prompt_specs = [
        {
            "name": "budget",
            "system_text": None,
            "user_text": "Turn budget enabled: this session has 12 turns per rollout.",
            "turn_template": "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining",
            "turn_format_kwargs": {"rollout_max_turns": 12},
        }
    ]

    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=prompt_specs)

    prompt_payload = json.loads(
        (workspace / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert not (workspace / ".sandbox_config.json").exists()
    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == prompt_specs
    assert not (workspace / ".codex" / "hooks.json").exists()


def test_install_workspace_runtime_hooks_does_not_copy_repo_codex_hooks(
    tmp_path: Path,
) -> None:
    repo_hooks_path = tmp_path / "repo" / ".codex" / "hooks.json"
    repo_hooks_path.parent.mkdir(parents=True)
    repo_hooks_path.write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    driver = CodexExecSessionDriver(run_root=tmp_path / "run-root", rollout_max_turns=12)

    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=[])

    assert (workspace / ".hooks" / "rollout_prompts.json").exists()
    assert not (workspace / ".codex" / "hooks.json").exists()


def test_install_workspace_runtime_hooks_omits_budget_prompt_when_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    driver = CodexExecSessionDriver(run_root=tmp_path / "run-root", rollout_max_turns=12)

    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=[])

    prompt_payload = json.loads(
        (workspace / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == []
