from __future__ import annotations

from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve.logger import (
    CompositeEveLogger,
)
from scaling_evolve.algorithms.eve.runner import _instantiate_eve_logger_group


def test_instantiate_eve_logger_group_returns_none_without_targets() -> None:
    logger = _instantiate_eve_logger_group(
        OmegaConf.create({"enabled": True}),
        OmegaConf.create({"run_id": "run-1", "run_root": "/tmp/run-1"}),
    )

    assert logger is None


def test_instantiate_eve_logger_group_supports_single_logger_config(tmp_path) -> None:
    logger = _instantiate_eve_logger_group(
        OmegaConf.create(
            {
                "_target_": "scaling_evolve.algorithms.eve.logger.CSVEveLogger",
                "enabled": False,
                "output_dir": str(tmp_path / "telemetry"),
                "excluded_score_fields": ["info"],
            }
        ),
        OmegaConf.create({"run_id": "run-1", "run_root": str(tmp_path)}),
    )

    assert logger is not None


def test_instantiate_eve_logger_group_supports_many_loggers(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "csv": {
                "_target_": "scaling_evolve.algorithms.eve.logger.CSVEveLogger",
                "enabled": True,
                "output_dir": str(tmp_path / "telemetry"),
                "excluded_score_fields": ["info"],
            },
            "wandb": {
                "_target_": "scaling_evolve.algorithms.eve.logger.WandbEveLogger",
                "enabled": False,
                "excluded_score_fields": ["info"],
            },
        }
    )

    logger = _instantiate_eve_logger_group(
        cfg,
        OmegaConf.create({"run_id": "run-1", "run_root": str(tmp_path)}),
    )

    assert isinstance(logger, CompositeEveLogger)
