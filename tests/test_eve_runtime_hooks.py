from __future__ import annotations

import json
from pathlib import Path

from scaling_evolve.algorithms.eve.workspace.runtime_hooks import (
    install_workspace_runtime_hooks,
)
from scaling_evolve.providers.agent.drivers.claude_code_tmux import (
    ClaudeCodeTmuxSessionDriver,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver


def test_install_workspace_runtime_hooks_writes_claude_settings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    driver = ClaudeCodeTmuxSessionDriver(
        pane_id="%9",
        run_root=tmp_path / "run-root",
        rollout_max_turns=20,
    )
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
    settings_payload = json.loads(
        (workspace / ".claude" / "settings.json").read_text(encoding="utf-8")
    )

    assert not (workspace / ".sandbox_config.json").exists()
    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == prompt_specs
    assert settings_payload["hooks"]["SessionStart"][0]["hooks"][0]["type"] == "command"
    assert settings_payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]["type"] == "command"
    assert settings_payload["hooks"]["PostToolUse"][0]["hooks"][0]["type"] == "command"


def test_install_workspace_runtime_hooks_writes_codex_hooks(tmp_path: Path) -> None:
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
    hooks_payload = json.loads((workspace / ".codex" / "hooks.json").read_text(encoding="utf-8"))

    assert not (workspace / ".sandbox_config.json").exists()
    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == prompt_specs
    assert (
        hooks_payload["hooks"]["PreToolUse"][0]["matcher"]
        == "Bash|Read|Edit|Write|MultiEdit|Glob|Grep"
    )
    assert hooks_payload["hooks"]["SessionStart"][0]["hooks"][0]["type"] == "command"
    assert hooks_payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]["type"] == "command"
    assert hooks_payload["hooks"]["PostToolUse"][0]["hooks"][0]["type"] == "command"


def test_install_workspace_runtime_hooks_omits_budget_prompt_when_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    driver = CodexExecSessionDriver(run_root=tmp_path / "run-root", rollout_max_turns=12)

    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=[])

    prompt_payload = json.loads(
        (workspace / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == []
