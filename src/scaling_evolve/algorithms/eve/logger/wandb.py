"""Weights & Biases logger for Eve runs."""

from __future__ import annotations

import logging
import os
from typing import Any

from scaling_evolve.algorithms.eve.logger.base import EveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4Result

_LOG = logging.getLogger(__name__)


def _load_wandb_sdk() -> Any | None:
    try:
        import wandb
    except ImportError:
        return None
    return wandb


class WandbEveLogger(EveLogger):
    """Encapsulates W&B setup, per-iteration logging, and final summaries."""

    def __init__(
        self,
        *,
        run_id: str,
        full_config: dict[str, object],
        enabled: bool = False,
        project: str = "scaling-evolve",
        entity: str | None = None,
        mode: str | None = None,
        name: str | None = None,
        excluded_score_fields: list[str] | tuple[str, ...],
    ) -> None:
        self._sdk: Any | None = None
        self._run: Any | None = None
        self._run_id = run_id
        super().__init__(excluded_score_fields=excluded_score_fields)
        if not enabled:
            return

        sdk = _load_wandb_sdk()
        if sdk is None:
            _LOG.warning("wandb not installed; skipping W&B logging.")
            return

        try:
            os.environ.setdefault("WANDB_SILENT", "true")
            run = sdk.init(
                project=project,
                entity=entity,
                mode=mode,
                name=name or run_id,
                config=full_config,
                reinit="finish_previous",
            )
        except Exception as error:  # noqa: BLE001
            _LOG.warning("W&B init failed for `%s`: %s", run_id, error)
            return
        if run is None:
            return

        self._sdk = sdk
        self._run = run
        self._configure_run()

    def on_iteration(
        self,
        *,
        iteration: int,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        phase2_results: list[Phase2Result],
        phase4_results: list[Phase4Result],
    ) -> None:
        if self._sdk is None or self._run is None:
            return
        payload = self._build_iteration_payload(
            iteration=iteration,
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            phase2_results=phase2_results,
            phase4_results=phase4_results,
        )
        if self.phase2_solver_rows:
            payload["tables/phase2_solvers"] = self._build_result_table(
                self.phase2_solver_rows,
                entry_kind="solver",
            )
        if self.phase2_optimizer_rows:
            payload["tables/phase2_optimizers"] = self._build_result_table(
                self.phase2_optimizer_rows,
                entry_kind="optimizer",
            )
        if self.phase4_optimizer_rows:
            payload["tables/phase4_optimizers"] = self._build_result_table(
                self.phase4_optimizer_rows,
                entry_kind="optimizer",
            )
        self._sdk.log(payload, step=iteration)

    def finish(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> None:
        if self._sdk is None or self._run is None:
            return
        summary = self._build_finish_summary(
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            iterations_completed=iterations_completed,
        )
        self._run.summary.update(summary)
        self._sdk.finish()

    def _configure_run(self) -> None:
        assert self._sdk is not None
        assert self._run is not None
        self._sdk.define_metric("iteration")
        self._sdk.define_metric("population/*", step_metric="iteration")
        for prefix in (
            "usage/phase2",
            "usage/phase4",
            "usage/iteration",
            "usage/cumulative/phase2",
            "usage/cumulative/phase4",
            "usage/cumulative",
        ):
            summary = "last" if prefix.startswith("usage/cumulative") else None
            for key in (
                "model_cost_usd",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_creation_tokens",
                "agent_turns",
                "wallclock_seconds",
            ):
                self._sdk.define_metric(
                    f"{prefix}/{key}",
                    step_metric="iteration",
                    summary=summary,
                )
        self._sdk.define_metric("phase2/*", step_metric="iteration")
        self._sdk.define_metric("phase2/iteration/*", step_metric="iteration")
        self._sdk.define_metric("phase2/cumulative/*", step_metric="iteration", summary="last")
        self._sdk.define_metric("phase4/*", step_metric="iteration")
        self._sdk.define_metric("phase4/iteration/*", step_metric="iteration")
        self._sdk.define_metric("phase4/cumulative/*", step_metric="iteration", summary="last")

        run_url = getattr(self._run, "url", None)
        if isinstance(run_url, str) and run_url:
            _LOG.info("W&B run url: %s", run_url)

    def _build_result_table(
        self,
        rows: list[dict[str, object]],
        *,
        entry_kind: str,
    ) -> object:
        assert self._sdk is not None
        columns = list(self.result_table_columns(entry_kind=entry_kind))
        table = self._sdk.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(column) for column in columns])
        return table
