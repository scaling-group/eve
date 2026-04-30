from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_eve_initial_optimizer_skills_have_valid_yaml_frontmatter() -> None:
    skill_paths = tuple(
        sorted(Path("configs/eve/optimizer").glob("*/initial_optimizer/skills/**/SKILL.md"))
    )
    assert skill_paths

    for skill_path in skill_paths:
        content = skill_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"]
        assert payload["description"]


def test_eve_claude_check_agents_have_valid_yaml_frontmatter() -> None:
    agent_paths = tuple(sorted(Path("configs/eve/application").glob("*/check_claude.md")))
    assert agent_paths

    for agent_path in agent_paths:
        content = agent_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"] == "check-runner"
        assert payload["description"]


def test_eve_codex_check_agents_have_valid_toml() -> None:
    agent_paths = tuple(sorted(Path("configs/eve/application").glob("*/check_codex.toml")))
    assert agent_paths

    for agent_path in agent_paths:
        payload = tomllib.loads(agent_path.read_text(encoding="utf-8"))
        assert payload["name"] == "check-runner"
        assert payload["description"]
        assert payload["developer_instructions"]
