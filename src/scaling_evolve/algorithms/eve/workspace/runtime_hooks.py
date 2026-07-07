"""Install runtime hook config into Eve workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from scaling_evolve.providers.agent.drivers.base import SessionDriver


def install_workspace_runtime_hooks(
    workspace: Path,
    *,
    driver: SessionDriver,
    prompt_specs: list[dict[str, object]],
) -> None:
    """Write prompt-injection and sandbox hook config for one Eve workspace."""

    _ = driver
    _write_rollout_prompt_config(workspace, prompt_specs=prompt_specs)


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
