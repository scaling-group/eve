from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scaling_evolve.algorithms.eve.runner  # noqa: F401


def _compose_eve_config(name: str):
    config_dir = Path("configs/eve").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=name)


def test_icon_config_composes_to_current_core_schema() -> None:
    cfg = _compose_eve_config("icon")

    assert OmegaConf.select(cfg, "application.name") == "icon-pe"
    assert OmegaConf.select(cfg, "driver.driver") == "codex_exec"
    assert OmegaConf.select(cfg, "logger.wandb.enabled") is False
    assert (
        OmegaConf.select(cfg, "optimizer.initial_guidance")
        == "configs/eve/optimizer/icon/initial_guidance"
    )
    assert (
        OmegaConf.select(cfg, "optimizer.workers.items.0.immutable")
        == "configs/eve/optimizer/icon/immutable"
    )
    assert OmegaConf.select(cfg, "optimizer.workers.items.0.prompt") == (
        "configs/eve/optimizer/icon/prompt"
    )
    assert OmegaConf.select(cfg, "evaluation.steps") == [
        "configs/eve/evaluation/icon/evaluation.sh"
    ]


def test_icon_config_points_at_existing_ported_assets() -> None:
    app_cfg = yaml.safe_load(Path("configs/eve/application/icon.yaml").read_text())
    opt_cfg = yaml.safe_load(Path("configs/eve/optimizer/icon.yaml").read_text())
    eval_cfg = yaml.safe_load(Path("configs/eve/evaluation/icon.yaml").read_text())

    editable_files = app_cfg["application"]["editable"]["files"]
    assert editable_files == [
        "configs/experiment/evolve_base.yaml",
        "configs/model/icon_evolve.yaml",
        "src/models/icon/icon_evolve.py",
        "src/models/base/transformer_evolve.py",
        "src/models/icon/pe_evolve.py",
    ]
    for relative_path in editable_files:
        assert (Path("examples/icon") / relative_path).is_file()

    assert Path(eval_cfg["evaluation"]["steps"][0]).is_file()
    assert not Path("configs/eve/application/icon").exists()
    assert Path("configs/eve/evaluation/icon/evaluation.sh").is_file()
    assert Path("configs/eve/evaluation/icon/evaluation_orchestrator.sh").is_file()
    assert Path("configs/eve/evaluation/icon/support/helpers.sh").is_file()
    assert Path(opt_cfg["optimizer"]["initial_guidance"]).is_dir()
    worker = opt_cfg["optimizer"]["workers"]["items"][0]
    immutable_root = Path(worker["immutable"])
    assert immutable_root.is_dir()
    assert (immutable_root / ".claude/agents/check-runner.md").is_file()
    assert (immutable_root / ".codex/agents/check-runner.toml").is_file()
    assert (immutable_root / "check_runner/check.sh").is_file()
    assert Path(worker["prompt"]).is_dir()


def test_icon_port_uses_public_placeholders_and_icon_core_convention() -> None:
    app_cfg = yaml.safe_load(Path("configs/eve/application/icon.yaml").read_text())
    application = app_cfg["application"]
    readme = Path("examples/icon/README.md").read_text(encoding="utf-8")

    assert application["github_url"] == "https://github.com/YOUR_USER/your-icon-core-fork"
    assert application["commit"] == "YOUR_FORK_COMMIT_SHA"
    assert "https://github.com/scaling-group/icon-core" in readme
    assert "configs/eve/application/icon.yaml" in readme
    assert "configs/eve/evaluation/icon/evaluation.sh" in readme
    assert "configs/eve/optimizer/icon/immutable/check_runner/" in readme

    icon_roots = (
        Path("configs/eve/optimizer/icon/immutable/check_runner"),
        Path("configs/eve/evaluation/icon"),
        Path("configs/eve/application/icon.yaml"),
        Path("examples/icon"),
    )
    scanned = ""
    for root in icon_roots:
        paths = (
            [root]
            if root.is_file()
            else sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
        )
        paths = [path for path in paths if path.suffix not in {".pyc", ".pyo"}]
        scanned += "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "~/repos/icon-core" in scanned
    assert "/scratch/${USER}/envs/venvs/icon-core/bin/python" in scanned
    assert "scaling-icon-core" not in scanned


def test_icon_port_does_not_add_smoke_resume_or_tmux_presets() -> None:
    forbidden_paths = (
        Path("configs/eve/icon.smoke.yaml"),
        Path("configs/eve/icon.resume.yaml"),
        Path("configs/eve/driver/codex_tmux_icon.yaml"),
    )

    for path in forbidden_paths:
        assert not path.exists(), f"unexpected Icon preset: {path}"


def test_icon_port_contains_no_release_data_or_model_artifacts() -> None:
    roots = (
        Path("configs/eve/evaluation/icon"),
        Path("configs/eve/optimizer/icon"),
        Path("examples/icon"),
    )
    forbidden_dir_names = {
        "__pycache__",
        ".runs",
        "checkpoints",
        "data",
        "release",
        "runs",
        "wandb",
    }
    forbidden_suffixes = {
        ".ckpt",
        ".jsonl",
        ".log",
        ".npz",
        ".pkl",
        ".pt",
        ".pth",
        ".pyc",
        ".pyo",
    }

    for root in roots:
        for path in root.rglob("*"):
            assert not (path.is_dir() and path.name in forbidden_dir_names), (
                f"unexpected generated/artifact directory in Icon port: {path}"
            )
            assert not (path.is_file() and path.suffix in forbidden_suffixes), (
                f"unexpected generated/artifact file in Icon port: {path}"
            )


def test_public_eve_configs_disable_wandb_by_default() -> None:
    for config_name in (
        "circle_packing",
        "math_proof_quickstart",
        "math_proof_jensen_covering",
        "icon",
    ):
        cfg = _compose_eve_config(config_name)

        assert OmegaConf.select(cfg, "logger.wandb.enabled") is False
        assert OmegaConf.select(cfg, "logger.wandb.project") == "eve"
        assert OmegaConf.select(cfg, "logger.wandb.entity") is None
