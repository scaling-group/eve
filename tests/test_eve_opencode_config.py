from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scaling_evolve.algorithms.eve.runner  # noqa: F401
from scaling_evolve.algorithms.eve.runtime.driver import build_role_drivers
from scaling_evolve.providers.agent.drivers.opencode import OpenCodeSessionDriver


def _compose_with_driver(driver: str, *, config_name: str = "circle_packing.smoke"):
    config_dir = Path("configs/eve").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name=config_name,
            overrides=[f"driver={driver}"],
        )


def test_opencode_smoke_preset_composes() -> None:
    cfg = _compose_with_driver("opencode_smoke")

    assert OmegaConf.select(cfg, "driver.driver") == "opencode"
    assert OmegaConf.select(cfg, "driver.executable") == "opencode"
    assert OmegaConf.select(cfg, "driver.model") is None
    assert OmegaConf.select(cfg, "driver.variant") is None
    assert OmegaConf.select(cfg, "driver.rollout_max_turns") == 50
    assert OmegaConf.select(cfg, "driver.budget_prompt") is False
    assert OmegaConf.select(cfg, "driver.timeout_seconds") == 900
    assert OmegaConf.select(cfg, "loop.n_workers_phase2") == 2
    assert OmegaConf.select(cfg, "loop.max_iterations") == 2


def test_opencode_max_preset_composes() -> None:
    cfg = _compose_with_driver("opencode_max")

    assert OmegaConf.select(cfg, "driver.driver") == "opencode"
    assert OmegaConf.select(cfg, "driver.rollout_max_turns") == 10000
    assert OmegaConf.select(cfg, "driver.budget_prompt") is False
    assert OmegaConf.select(cfg, "driver.timeout_seconds") == 7200


def test_math_proof_preset_builds_opencode_role_drivers(tmp_path: Path) -> None:
    cfg = _compose_with_driver(
        "opencode_smoke",
        config_name="math_proof_quickstart",
    )
    driver_cfg = OmegaConf.to_container(cfg.driver, resolve=True)
    assert isinstance(driver_cfg, dict)

    drivers = build_role_drivers(
        driver_cfg,
        run_root=tmp_path,
        worker_slots=1,
    )

    assert isinstance(drivers.solver_driver, OpenCodeSessionDriver)
    assert isinstance(drivers.eval_driver_factory(), OpenCodeSessionDriver)
    assert drivers.solver_driver.system_prompt_file == (
        Path("configs/eve/optimizer/math_proof/prompt/SYSTEM_PROMPT.md").resolve()
    )
    assert drivers.eval_driver_factory().system_prompt_file == (
        Path("configs/eve/evaluation/math_proof/prompt/SYSTEM_PROMPT.md").resolve()
    )
