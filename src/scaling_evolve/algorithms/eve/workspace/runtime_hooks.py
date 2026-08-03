"""Install runtime hook config into Eve workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from scaling_evolve.algorithms.eve.workspace.opencode_agents import (
    project_opencode_agents,
)
from scaling_evolve.providers.agent.drivers.base import SessionDriver
from scaling_evolve.providers.agent.drivers.opencode import OpenCodeSessionDriver


def install_workspace_runtime_hooks(
    workspace: Path,
    *,
    driver: SessionDriver,
    prompt_specs: list[dict[str, object]],
) -> None:
    """Install provider runtime configuration for one Eve workspace."""

    if isinstance(driver, OpenCodeSessionDriver):
        config_root = driver.workspace_config_dir(workspace)
        project_opencode_agents(
            workspace,
            config_root=config_root,
            steps=driver.rollout_max_turns,
        )
        _project_opencode_skills(workspace, config_root=config_root)
    _write_rollout_prompt_config(workspace, prompt_specs=prompt_specs)


def _project_opencode_skills(workspace: Path, *, config_root: Path) -> None:
    source_root = workspace / "guidance" / "skills"
    target_root = config_root / "skills"
    if target_root.is_symlink():
        target_root.unlink()
    elif target_root.exists():
        raise RuntimeError(f"OpenCode skills projection path is not a symlink: {target_root}")
    if not source_root.is_dir():
        return
    target_root.parent.mkdir(parents=True, exist_ok=True)
    target_root.symlink_to(source_root.resolve(), target_is_directory=True)


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
