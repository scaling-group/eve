"""Run an Eve experiment."""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from pathlib import Path
from uuid import uuid4

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig as OmegaDictConfig

from scaling_evolve.algorithms.eve.factory import EveFactory
from scaling_evolve.algorithms.eve.logger import (
    CompositeEveLogger,
)
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.runtime.driver import (
    build_role_drivers,
    load_pricing_table,
)
from scaling_evolve.algorithms.eve.runtime.restore import RestoreSpec, parse_restore_spec
from scaling_evolve.algorithms.eve.workflow.evaluation import (
    RemoteTransportHaltError,
    build_solver_evaluator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
_LOG = logging.getLogger(__name__)


def build_run_id(ts: str, label: object = "") -> str:
    return "-".join(
        [
            "run",
            ts,
            *([text] if (text := str(label).strip()) else []),
            uuid4().hex[:12],
        ]
    )


if not OmegaConf.has_resolver("eve_run_id"):
    OmegaConf.register_new_resolver("eve_run_id", build_run_id, use_cache=True)


def _normalize_restore_sources(raw_value: object) -> list[RestoreSpec]:
    if raw_value is None or raw_value == "":
        return []
    if OmegaConf.is_config(raw_value):
        raw_value = OmegaConf.to_container(raw_value, resolve=True)
    if isinstance(raw_value, str):
        return [RestoreSpec(path=Path(raw_value))]
    if isinstance(raw_value, dict):
        try:
            return [parse_restore_spec(raw_value)]
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if isinstance(raw_value, list):
        specs: list[RestoreSpec] = []
        for item in raw_value:
            try:
                specs.append(parse_restore_spec(item))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        return specs
    raise SystemExit(
        "`restore_from` must be a path string, a restore mapping, or a list of restore specs."
    )


def _instantiate_eve_logger_group(
    logger_cfg: DictConfig | OmegaDictConfig | None,
    run_config: DictConfig,
):
    if not isinstance(logger_cfg, DictConfig):
        return None
    full_config = OmegaConf.to_container(run_config, resolve=True)
    candidates: list[DictConfig] = []
    if "_target_" in logger_cfg:
        candidates.append(logger_cfg)
    else:
        candidates.extend(
            candidate_cfg
            for _, candidate_cfg in logger_cfg.items()
            if isinstance(candidate_cfg, DictConfig) and "_target_" in candidate_cfg
        )
    loggers = []
    for candidate_cfg in candidates:
        if isinstance(candidate_cfg, DictConfig) and "_target_" in candidate_cfg:
            loggers.append(
                instantiate(
                    candidate_cfg,
                    run_id=run_config.run_id,
                    full_config=full_config,
                    _convert_="all",
                    _recursive_=False,
                )
            )
    if not loggers:
        return None
    if len(loggers) == 1:
        return loggers[0]
    return CompositeEveLogger(loggers)


def run(cfg: DictConfig) -> None:
    load_dotenv()

    repo_root = Path(__file__).resolve().parents[4]
    application_cfg = OmegaConf.to_container(cfg.application, resolve=True)
    restore_sources = _normalize_restore_sources(
        OmegaConf.select(cfg, "restore_from", default=None)
    )
    run_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    application_run_cfg = run_config.application
    run_root = Path(run_config.run_root)
    cache_root = run_root / ".repo_cache"
    problem = RepoTaskProblem.from_config(
        application_cfg,
        cache_root=cache_root,
        search_root=repo_root,
    )

    driver_cfg = OmegaConf.to_container(cfg.driver, resolve=True) if "driver" in cfg else {}
    drivers = None
    run_root.mkdir(parents=True, exist_ok=True)
    pricing_table = load_pricing_table(repo_root / "configs" / "pricing.yaml")

    def _cleanup_drivers() -> None:
        if drivers is not None:
            drivers.close()

    signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(128 + signal.SIGTERM))
    atexit.register(_cleanup_drivers)

    try:
        drivers = build_role_drivers(
            driver_cfg,
            run_root=run_root,
            workers=run_config.loop.n_workers_phase2,
            pricing_table=pricing_table,
        )
        solver_evaluator = build_solver_evaluator(
            problem,
            evaluation_failure_score=application_run_cfg.evaluation_failure_score,
            boundary_failure_score=application_run_cfg.boundary_failure_score,
            seed_solver_score=application_run_cfg.seed_solver_score,
            seed_solver_skip_evaluation=application_run_cfg.seed_solver_skip_evaluation,
            evaluation_driver_factory=drivers.eval_driver_factory,
        )

        run_logger = (
            _instantiate_eve_logger_group(cfg.logger, run_config) if "logger" in cfg else None
        )

        with EveFactory.from_config(
            run_config,
            solver_evaluator,
            solver_driver=drivers.solver_driver,
            optimizer_driver=drivers.optimizer_driver,
            logger=run_logger,
            task_problem=problem,
        ) as factory:
            if not restore_sources:
                factory.seed_initial_optimizer(search_root=repo_root)
            for restore_source in restore_sources:
                restore_result = factory.restore_from_spec(restore_source)
                _LOG.info(
                    "Restored Eve populations from %s: solver=%d optimizer=%d",
                    restore_result.source_run_root,
                    restore_result.solver.entries_restored,
                    restore_result.optimizer.entries_restored,
                )

            try:
                factory.run()
            except RemoteTransportHaltError as exc:
                halt_message = f"[REMOTE-HALT] {exc}"
                print(halt_message, flush=True)
                _LOG.error(halt_message)
                raise SystemExit(1) from exc

            solver_pop_size = factory.loop.solver_pop.size()
            optimizer_pop_size = factory.loop.optimizer_pop.size()

            print(f"Run ID   : {run_config.run_id}")
            print(f"Solver pop : {solver_pop_size} candidates")
            print(f"Opt pop  : {optimizer_pop_size} optimizers")
            print(f"Artifacts: {run_root}")

            if run_logger is not None:
                run_logger.finish(
                    solver_entries=factory.loop.solver_pop.entries(),
                    optimizer_entries=factory.loop.optimizer_pop.entries(),
                    iterations_completed=factory.loop.iterations_completed,
                )
    finally:
        if drivers is not None:
            drivers.close()


@hydra.main(version_base="1.3", config_path="../../../../configs/eve")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
