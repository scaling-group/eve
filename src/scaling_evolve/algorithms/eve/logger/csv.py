"""CSV logger for Eve runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scaling_evolve.algorithms.eve.logger.base import EveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4Result


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


class CSVEveLogger(EveLogger):
    """Writes Eve telemetry to CSV files under the run directory."""

    def __init__(
        self,
        *,
        run_id: str,
        full_config: dict[str, object],
        enabled: bool = False,
        output_dir: str | None = None,
        excluded_score_fields: list[str] | tuple[str, ...],
    ) -> None:
        _ = run_id
        super().__init__(excluded_score_fields=excluded_score_fields)
        self._iteration_rows: list[dict[str, object]] = []
        if not enabled:
            self._output_dir = None
            return
        resolved_output_dir = output_dir or str(Path(str(full_config["run_root"])) / "telemetry")
        self._output_dir = Path(resolved_output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def on_iteration(
        self,
        *,
        iteration: int,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        phase2_results: list[Phase2Result],
        phase4_results: list[Phase4Result],
    ) -> None:
        if self._output_dir is None:
            return
        payload = self._build_iteration_payload(
            iteration=iteration,
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            phase2_results=phase2_results,
            phase4_results=phase4_results,
        )
        phase2_solver_rows = self.phase2_solver_rows
        phase2_optimizer_rows = self.phase2_optimizer_rows
        phase4_optimizer_rows = self.phase4_optimizer_rows
        self._iteration_rows.append(dict(payload))
        self._write_iteration_metrics_csv()
        self._rewrite_result_rows_csv("phase2_solvers.csv", phase2_solver_rows, entry_kind="solver")
        self._rewrite_result_rows_csv(
            "phase2_optimizers.csv", phase2_optimizer_rows, entry_kind="optimizer"
        )
        self._rewrite_result_rows_csv(
            "phase4_optimizers.csv", phase4_optimizer_rows, entry_kind="optimizer"
        )
        self._write_summary(
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            iterations_completed=iteration,
        )

    def finish(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> None:
        if self._output_dir is None:
            return
        self._write_summary(
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            iterations_completed=iterations_completed,
        )

    def _write_iteration_metrics_csv(self) -> None:
        assert self._output_dir is not None
        fieldnames = sorted({key for row in self._iteration_rows for key in row})
        path = self._output_dir / "iteration_metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._iteration_rows:
                writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})

    def _rewrite_result_rows_csv(
        self,
        filename: str,
        rows: list[dict[str, object]],
        *,
        entry_kind: str,
    ) -> None:
        assert self._output_dir is not None
        path = self._output_dir / filename
        fieldnames = list(self.result_table_columns(entry_kind=entry_kind))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})

    def _write_summary(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> None:
        assert self._output_dir is not None
        summary = self._build_finish_summary(
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            iterations_completed=iterations_completed,
        )
        summary_path = self._output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
