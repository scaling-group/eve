from __future__ import annotations

import json
from pathlib import Path

from scaling_evolve.algorithms.eve.workspace.file_tree import expose_guidance_agents
from scaling_evolve.algorithms.eve.workspace.runtime_hooks import (
    install_workspace_runtime_hooks,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver
from scaling_evolve.providers.agent.drivers.opencode import OpenCodeSessionDriver


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


def test_install_workspace_runtime_hooks_projects_opencode_subagents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    skill = workspace / "guidance" / "skills" / "proof-check" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill_text = "---\nname: proof-check\ndescription: Check proofs.\n---\n\nCheck the proof.\n"
    skill.write_text(skill_text, encoding="utf-8")
    source = workspace / "guidance" / "agents" / "codex" / "proof-auditor.toml"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                'name = "proof-auditor"',
                'description = "Audit the proof."',
                'model = "gpt-5.4-mini"',
                'model_reasoning_effort = "high"',
                "developer_instructions = '''",
                "# Proof auditor",
                "",
                "Read the proof and write the report.",
                "'''",
                "",
            ]
        ),
        encoding="utf-8",
    )
    expose_guidance_agents(workspace)
    driver = OpenCodeSessionDriver(
        run_root=tmp_path / "run-root",
        rollout_max_turns=17,
    )

    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=[])

    rendered = (
        workspace / ".opencode-driver-transcripts" / "config" / "agents" / "proof-auditor.md"
    ).read_text(encoding="utf-8")
    assert rendered == "\n".join(
        [
            "---",
            'description: "Audit the proof."',
            "mode: subagent",
            "steps: 17",
            "permission:",
            "  task: deny",
            "---",
            "",
            "# Proof auditor",
            "",
            "Read the proof and write the report.",
            "",
        ]
    )
    assert "gpt-5.4-mini" not in rendered
    projected_skills = driver.workspace_config_dir(workspace) / "skills"
    assert projected_skills.is_symlink()
    assert (projected_skills / "proof-check" / "SKILL.md").read_text(encoding="utf-8") == skill_text

    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "Read the proof and write the report.",
            "Read the revised proof and write the report.",
        ),
        encoding="utf-8",
    )
    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=[])
    assert "Read the revised proof" in (
        driver.workspace_config_dir(workspace) / "agents" / "proof-auditor.md"
    ).read_text(encoding="utf-8")

    source.unlink()
    install_workspace_runtime_hooks(workspace, driver=driver, prompt_specs=[])
    assert not (driver.workspace_config_dir(workspace) / "agents" / "proof-auditor.md").exists()
