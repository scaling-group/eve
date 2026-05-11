"""Install runtime hook config into Eve workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from scaling_evolve.providers.agent.codex_hooks import (
    repo_codex_hooks_path,
    workspace_hook_command,
    write_codex_hooks_file,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver
from scaling_evolve.providers.agent.drivers.claude_code import ClaudeCodeSessionDriver
from scaling_evolve.providers.agent.drivers.claude_code_tmux import (
    ClaudeCodeTmuxSessionDriver,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver
from scaling_evolve.providers.agent.drivers.codex_tmux import CodexTmuxSessionDriver


def install_workspace_runtime_hooks(
    workspace: Path,
    *,
    driver: SessionDriver,
    prompt_specs: list[dict[str, object]],
) -> None:
    """Write prompt-injection and sandbox hook config for one Eve workspace."""

    _write_rollout_prompt_config(workspace, prompt_specs=prompt_specs)
    if _is_claude_driver(driver):
        _write_claude_settings(workspace)
    if _is_codex_driver(driver):
        _write_codex_hooks(workspace)


def _write_rollout_prompt_config(
    workspace: Path,
    *,
    prompt_specs: list[dict[str, object]],
) -> None:
    hooks_dir = workspace / ".hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "prompts": prompt_specs,
    }
    (hooks_dir / "rollout_prompts.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_claude_settings(workspace: Path) -> None:
    claude_dir = workspace / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    hook_command = workspace_hook_command()
    payload = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Read|Edit|Write|MultiEdit|Glob|Grep",
                    "hooks": [{"type": "command", "command": hook_command}],
                }
            ],
            "SessionStart": [{"hooks": [{"type": "command", "command": hook_command}]}],
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": hook_command}]}],
            "PostToolUse": [{"hooks": [{"type": "command", "command": hook_command}]}],
        }
    }
    (claude_dir / "settings.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_codex_hooks(workspace: Path) -> None:
    if repo_codex_hooks_path().exists():
        return
    write_codex_hooks_file(workspace / ".codex" / "hooks.json")


def _is_claude_driver(driver: SessionDriver) -> bool:
    return isinstance(driver, ClaudeCodeSessionDriver | ClaudeCodeTmuxSessionDriver)


def _is_codex_driver(driver: SessionDriver) -> bool:
    return isinstance(driver, CodexExecSessionDriver | CodexTmuxSessionDriver)
