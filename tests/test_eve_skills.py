from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_eve_initial_guidance_skills_have_valid_yaml_frontmatter() -> None:
    skill_paths = (
        Path(
            "configs/eve/optimizer/circle_packing/initial_guidance/skills/"
            "optimize-from-examples/SKILL.md"
        ),
        Path("configs/eve/optimizer/circle_packing/initial_guidance/skills/read-eval/SKILL.md"),
        Path("configs/eve/optimizer/icon/initial_guidance/skills/read-eval/SKILL.md"),
    )

    for skill_path in skill_paths:
        content = skill_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"]
        assert payload["description"]


def test_repo_docs_skills_have_valid_yaml_frontmatter() -> None:
    skill_paths = sorted(Path("docs/skills").glob("*/SKILL.md"))

    for skill_path in skill_paths:
        content = skill_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"]
        assert payload["description"]

    assert skill_paths


def test_public_docs_do_not_include_core_only_workflows() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    using_eve = Path("docs/using-eve.md").read_text(encoding="utf-8")

    assert not Path("docs/skills/scaling-evolve-core-runbook").exists()
    assert not Path("docs/skills/open-pull-request").exists()
    assert "scaling-group/scaling-evolve-core" not in readme
    assert "docs/skills/scaling-evolve-core-runbook/" not in readme
    assert "docs/skills/open-pull-request/" not in readme
    assert "scaling-evolve-core-runbook" not in using_eve
    assert "open-pull-request" not in using_eve


def test_root_skill_symlinks_match_public_agent_surface() -> None:
    assert Path(".agents/skills").is_symlink()
    assert Path(".agents/skills").readlink() == Path("../docs/skills")
    assert Path(".claude/skills").is_symlink()
    assert Path(".claude/skills").readlink() == Path("../docs/skills")
    assert not Path(".agent").exists()
    assert not Path(".codex/skills").exists()


def test_public_wandb_configs_do_not_set_group_defaults() -> None:
    wandb_cfg = yaml.safe_load(Path("configs/eve/logger/wandb.yaml").read_text())
    circle_cfg = yaml.safe_load(Path("configs/eve/circle_packing.yaml").read_text())

    assert wandb_cfg["enabled"] is False
    assert wandb_cfg["project"] == "eve"
    assert wandb_cfg["entity"] is None
    assert "wandb" not in circle_cfg.get("logger", {})


def test_eve_claude_check_agents_have_valid_yaml_frontmatter() -> None:
    agent_paths = (
        (
            Path("configs/eve/optimizer/circle_packing/immutable/.claude/agents/check-runner.md"),
            "check-runner",
        ),
        (
            Path("configs/eve/optimizer/icon/immutable/.claude/agents/check-runner.md"),
            "check-runner",
        ),
        (
            Path(
                "configs/eve/evaluation/circle_packing/immutable_assess/"
                ".claude/agents/score-check.md"
            ),
            "score-check",
        ),
    )

    for agent_path, expected_name in agent_paths:
        content = agent_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"] == expected_name
        assert payload["description"]


def test_eve_codex_check_agents_have_valid_toml() -> None:
    agent_paths = (
        (
            Path("configs/eve/optimizer/circle_packing/immutable/.codex/agents/check-runner.toml"),
            "check-runner",
        ),
        (
            Path("configs/eve/optimizer/icon/immutable/.codex/agents/check-runner.toml"),
            "check-runner",
        ),
        (
            Path(
                "configs/eve/evaluation/circle_packing/immutable_assess/"
                ".codex/agents/score-check.toml"
            ),
            "score-check",
        ),
    )

    for agent_path, expected_name in agent_paths:
        payload = tomllib.loads(agent_path.read_text(encoding="utf-8"))
        assert payload["name"] == expected_name
        assert payload["description"]
        assert payload["developer_instructions"]


def test_circle_packing_smoke_uses_single_root_preset_with_override_recipes() -> None:
    smoke_path = Path("configs/eve/circle_packing.smoke.yaml")
    smoke_text = smoke_path.read_text(encoding="utf-8")

    assert smoke_path.is_file()
    assert not Path("configs/eve/circle_packing.codex_prompt.smoke.yaml").exists()
    assert not Path("configs/eve/circle_packing.judge.smoke.yaml").exists()
    assert yaml.safe_load(smoke_text)["loop"]["max_iterations"] == 2
    assert "evaluation=circle_packing.judge" in smoke_text
    assert "+driver.overrides.solver.system_prompt_file" in smoke_text
    assert "+driver.overrides.eval.system_prompt_file" in smoke_text


def test_icon_port_does_not_reintroduce_old_schema_names() -> None:
    roots = (
        Path("configs/eve/evaluation/icon"),
        Path("configs/eve/optimizer/icon"),
        Path("configs/eve/icon.yaml"),
        Path("configs/eve/application/icon.yaml"),
        Path("configs/eve/evaluation/icon.yaml"),
        Path("configs/eve/optimizer/icon.yaml"),
        Path("examples/icon"),
    )
    banned = (
        "EVE_OUTPUT_ROOT",
        "output/",
        "initial_optimizer",
        "restore_from",
        "codex_tmux_icon",
        "evaluation_steps",
        "evaluation_failure_score",
        "evaluate_short",
        "wandb_env.sh",
    )

    scanned: list[Path] = []
    for root in roots:
        paths = (
            [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        )
        for path in paths:
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            scanned.append(path)
            content = path.read_text(encoding="utf-8")
            for token in banned:
                assert token not in content, f"{token!r} found in {path}"

    assert scanned
