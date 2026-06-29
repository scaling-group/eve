"""Run an Eve experiment."""

from __future__ import annotations

import atexit
import logging
import os
import signal
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
from scaling_evolve.algorithms.eve.runtime.imports import ImportSpec, parse_import_spec
from scaling_evolve.algorithms.eve.runtime.resume import ResumeError, ResumePlan, prepare_resume
from scaling_evolve.algorithms.eve.workflow.evaluation import (
    build_evaluation_plan,
    build_solver_evaluator,
)
from scaling_evolve.providers.agent.codex_hooks import (
    ensure_codex_hooks_trusted,
    write_repo_codex_hooks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
_LOG = logging.getLogger(__name__)
_CSV_LOGGER_TARGET = "scaling_evolve.algorithms.eve.logger.CSVEveLogger"


def build_run_id(ts: str, label: object = "") -> str:
    return "-".join(
        [
            "run",
            ts,
            *([text] if (text := str(label).strip()) else []),
            uuid4().hex[:12],
        ]
    )


def build_run_root(
    resume_from: object,
    cwd: object,
    application_name: object,
    run_id: object,
) -> str:
    """Resolve the Hydra run root for fresh runs or explicit same-run resume."""

    if isinstance(resume_from, str) and resume_from.strip():
        return str(Path(resume_from).expanduser())
    return str(Path(str(cwd)) / ".runs" / "eve" / str(application_name) / str(run_id))


if not OmegaConf.has_resolver("eve_run_id"):
    OmegaConf.register_new_resolver("eve_run_id", build_run_id, use_cache=True)
if not OmegaConf.has_resolver("eve_run_root"):
    OmegaConf.register_new_resolver("eve_run_root", build_run_root, use_cache=True)


def _normalize_import_sources(raw_value: object) -> list[ImportSpec]:
    if raw_value is None or raw_value == "":
        return []
    if OmegaConf.is_config(raw_value):
        raw_value = OmegaConf.to_container(raw_value, resolve=True)
    if isinstance(raw_value, str):
        if raw_value == "???":
            raise SystemExit("`import_from` must be set to a prior Eve run path.")
        return [ImportSpec(path=Path(raw_value))]
    if isinstance(raw_value, dict):
        try:
            return [parse_import_spec(raw_value)]
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if isinstance(raw_value, list):
        specs: list[ImportSpec] = []
        for item in raw_value:
            try:
                specs.append(parse_import_spec(item))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        return specs
    raise SystemExit(
        "`import_from` must be a path string, an import mapping, or a list of import specs."
    )


def _normalize_score_payload(raw_value: object) -> object:
    if OmegaConf.is_config(raw_value):
        return OmegaConf.to_container(raw_value, resolve=True)
    return raw_value


def _instantiate_eve_logger_group(
    logger_cfg: DictConfig | OmegaDictConfig | None,
    run_config: DictConfig,
    *,
    resume_anchor_iteration: int | None = None,
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
            instantiate_kwargs = {}
            if (
                resume_anchor_iteration is not None
                and str(candidate_cfg.get("_target_")) == _CSV_LOGGER_TARGET
            ):
                instantiate_kwargs["resume_anchor_iteration"] = resume_anchor_iteration
            loggers.append(
                instantiate(
                    candidate_cfg,
                    run_id=run_config.run_id,
                    full_config=full_config,
                    **instantiate_kwargs,
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
    resume_plan, import_sources = _resolve_lifecycle_inputs(cfg)
    run_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if resume_plan is not None:
        _apply_resume_plan(run_config, resume_plan)
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
    if _driver_config_uses_codex(driver_cfg):
        write_repo_codex_hooks(repo_root)
        ensure_codex_hooks_trusted(repo_root)
    run_root.mkdir(parents=True, exist_ok=True)
    pricing_table = load_pricing_table(repo_root / "configs" / "pricing.yaml")

    def _cleanup_drivers() -> None:
        if drivers is not None:
            drivers.close()

    def _abort_on_signal(sig: int, _frame: object) -> None:
        _cleanup_drivers()
        os._exit(128 + sig)

    signal.signal(signal.SIGTERM, _abort_on_signal)
    signal.signal(signal.SIGINT, _abort_on_signal)
    atexit.register(_cleanup_drivers)

    try:
        drivers = build_role_drivers(
            driver_cfg,
            run_root=run_root,
            workers=run_config.loop.n_workers_phase2,
            pricing_table=pricing_table,
        )
        evaluation_plan = build_evaluation_plan(
            problem,
            evaluation_config=OmegaConf.select(run_config, "evaluation", default=None),
            search_root=repo_root,
        )
        evaluation_failure_score = _normalize_score_payload(run_config.evaluation.failure_score)
        boundary_failure_score = _normalize_score_payload(
            application_run_cfg.boundary_failure_score
        )
        seed_solver_score = _normalize_score_payload(
            OmegaConf.select(
                run_config,
                "evaluation.seed_solver_score",
                default=None,
            )
        )
        seed_solver_skip_evaluation = bool(
            OmegaConf.select(
                run_config,
                "evaluation.seed_solver_skip_evaluation",
                default=False,
            )
        )
        include_solver_examples = bool(
            OmegaConf.select(
                run_config,
                "evaluation.include_solver_examples",
                default=False,
            )
        )
        solver_evaluator = build_solver_evaluator(
            problem,
            evaluation_plan=evaluation_plan,
            evaluation_failure_score=evaluation_failure_score,
            boundary_failure_score=boundary_failure_score,
            seed_solver_score=seed_solver_score,
            seed_solver_skip_evaluation=seed_solver_skip_evaluation,
            include_solver_examples=include_solver_examples,
            evaluation_driver_factory=drivers.eval_driver_factory,
        )

        run_logger = (
            _instantiate_eve_logger_group(
                cfg.logger,
                run_config,
                resume_anchor_iteration=(
                    resume_plan.start_iteration if resume_plan is not None else None
                ),
            )
            if "logger" in cfg
            else None
        )

        with EveFactory.from_config(
            run_config,
            solver_evaluator,
            solver_driver=drivers.solver_driver,
            logger=run_logger,
            task_problem=problem,
            search_root=repo_root,
        ) as factory:
            seed_initial_on_import = _seed_initial_on_import(cfg)
            if resume_plan is not None and seed_initial_on_import:
                raise SystemExit(
                    "`optimizer.seed_initial_on_import` cannot be used with `resume_from`."
                )
            if resume_plan is None and (not import_sources or seed_initial_on_import):
                factory.seed_initial_guidance(search_root=repo_root)
            for import_source in import_sources:
                import_result = factory.import_from_spec(import_source)
                _LOG.info(
                    "Imported Eve populations from %s: solver=%d optimizer=%d",
                    import_result.source_run_root,
                    import_result.solver.entries_imported,
                    import_result.optimizer.entries_imported,
                )

            if resume_plan is not None and run_logger is not None:
                run_logger.write_resume_anchor_summary(
                    solver_entries=factory.loop.solver_pop.entries(),
                    optimizer_entries=factory.loop.optimizer_pop.entries(),
                    iterations_completed=resume_plan.start_iteration,
                )

            factory.run(
                start_iteration=resume_plan.start_iteration if resume_plan is not None else 0
            )

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


def _resolve_lifecycle_inputs(cfg: DictConfig) -> tuple[ResumePlan | None, list[ImportSpec]]:
    import_from = OmegaConf.select(cfg, "import_from", default=None)
    resume_from = OmegaConf.select(cfg, "resume_from", default=None)
    resume_iteration = _normalize_resume_iteration(
        OmegaConf.select(cfg, "resume_iteration", default=None)
    )

    if resume_iteration is not None and not _is_lifecycle_value_set(resume_from):
        raise SystemExit("`resume_iteration` requires `resume_from`.")

    if _is_lifecycle_value_set(resume_from) and _is_lifecycle_value_set(import_from):
        raise SystemExit("`resume_from` and `import_from` are mutually exclusive.")

    if _is_lifecycle_value_set(resume_from):
        if not isinstance(resume_from, str):
            raise SystemExit("`resume_from` must be an absolute Eve run root path.")
        if not Path(resume_from).expanduser().is_absolute():
            raise SystemExit("`resume_from` must be an absolute Eve run root path.")
        try:
            return prepare_resume(resume_from, resume_iteration=resume_iteration), []
        except ResumeError as exc:
            raise SystemExit(str(exc)) from exc

    return None, _normalize_import_sources(import_from)


def _normalize_resume_iteration(raw_value: object) -> int | None:
    if not _is_lifecycle_value_set(raw_value):
        return None
    if isinstance(raw_value, bool):
        raise SystemExit("`resume_iteration` must be a non-negative integer.")
    if isinstance(raw_value, int):
        iteration = raw_value
    elif isinstance(raw_value, str):
        try:
            iteration = int(raw_value)
        except ValueError as exc:
            raise SystemExit("`resume_iteration` must be a non-negative integer.") from exc
    else:
        raise SystemExit("`resume_iteration` must be a non-negative integer.")
    if iteration < 0:
        raise SystemExit("`resume_iteration` must be a non-negative integer.")
    return iteration


def _apply_resume_plan(run_config: DictConfig, resume_plan: ResumePlan) -> None:
    run_config.run_id = resume_plan.run_id
    run_config.run_root = str(resume_plan.run_root)
    run_config.loop.max_iterations = resume_plan.checkpoint.max_iterations
    run_config.loop.workspace_root = str(resume_plan.run_root)
    run_config.loop.artifact_root = str(resume_plan.run_root / "artifacts")
    run_config.loop.solver_db_path = str(resume_plan.run_root / "solver_lineage.db")
    run_config.loop.optimizer_db_path = str(resume_plan.run_root / "optimizer_lineage.db")


def _seed_initial_on_import(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "optimizer.seed_initial_on_import", default=False))


def _is_lifecycle_value_set(raw_value: object) -> bool:
    if raw_value is None:
        return False
    if OmegaConf.is_config(raw_value):
        raw_value = OmegaConf.to_container(raw_value, resolve=False)
    if isinstance(raw_value, str):
        return bool(raw_value.strip())
    if isinstance(raw_value, dict | list | tuple):
        return len(raw_value) > 0
    return True


def _driver_config_uses_codex(driver_cfg: object) -> bool:
    if not isinstance(driver_cfg, dict):
        return False
    return any(
        _driver_name(_driver_cfg_for_role(driver_cfg, role)) in {"codex_exec", "codex_tmux"}
        for role in ("solver", "eval", "optimizer")
    )


def _driver_cfg_for_role(driver_cfg: dict[str, object], role: str) -> dict[str, object]:
    resolved = {key: value for key, value in driver_cfg.items() if key != "overrides"}
    overrides = driver_cfg.get("overrides")
    if isinstance(overrides, dict):
        role_override = overrides.get(role)
        if isinstance(role_override, dict):
            resolved.update(role_override)
    return resolved


def _driver_name(driver_cfg: dict[str, object]) -> str:
    raw_driver = driver_cfg.get("driver")
    if isinstance(raw_driver, str) and raw_driver:
        return raw_driver
    raw_provider = driver_cfg.get("provider")
    if isinstance(raw_provider, str) and raw_provider:
        return raw_provider
    return "claude_code"


if __name__ == "__main__":
    main()
