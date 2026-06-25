from __future__ import annotations

import csv

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


def test_instantiate_eve_logger_group_passes_resume_anchor_to_csv_only(tmp_path) -> None:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    with (telemetry / "iteration_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "usage/cumulative/model_cost_usd"])
        writer.writeheader()
        writer.writerow({"iteration": "1", "usage/cumulative/model_cost_usd": "0.1"})
        writer.writerow({"iteration": "2", "usage/cumulative/model_cost_usd": "0.3"})
    cfg = OmegaConf.create(
        {
            "csv": {
                "_target_": "scaling_evolve.algorithms.eve.logger.CSVEveLogger",
                "enabled": True,
                "output_dir": str(telemetry),
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
        resume_anchor_iteration=1,
    )

    assert isinstance(logger, CompositeEveLogger)
    with (telemetry / "iteration_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in rows] == ["1"]
