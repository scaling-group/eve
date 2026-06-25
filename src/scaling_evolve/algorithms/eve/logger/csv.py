"""CSV logger for Eve runs."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from scaling_evolve.algorithms.eve.logger.base import USAGE_KEYS, EveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import scalar
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result

_LOGGER = logging.getLogger(__name__)


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        resume_anchor_iteration: int | None = None,
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
        if resume_anchor_iteration is not None:
            self._restore_for_resume(resume_anchor_iteration)

    def on_iteration(
        self,
        *,
        iteration: int,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        phase2_results: list[Phase2Result],
    ) -> None:
        if self._output_dir is None:
            return
        payload = self._build_iteration_payload(
            iteration=iteration,
            solver_entries=solver_entries,
            optimizer_entries=optimizer_entries,
            phase2_results=phase2_results,
        )
        phase2_solver_rows = self.phase2_solver_rows
        phase2_optimizer_rows = self.phase2_optimizer_rows
        self._iteration_rows.append(dict(payload))
        self._write_iteration_metrics_csv()
        self._rewrite_result_rows_csv("phase2_solvers.csv", phase2_solver_rows, entry_kind="solver")
        self._rewrite_result_rows_csv(
            "phase2_optimizers.csv", phase2_optimizer_rows, entry_kind="optimizer"
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

    def write_resume_anchor_summary(
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

    def _restore_for_resume(self, anchor_iteration: int) -> None:
        assert self._output_dir is not None
        self._iteration_rows = self._read_rows_through_anchor(
            self._output_dir / "iteration_metrics.csv",
            anchor_iteration=anchor_iteration,
        )
        self._phase2_solver_rows = self._read_rows_through_anchor(
            self._output_dir / "phase2_solvers.csv",
            anchor_iteration=anchor_iteration,
        )
        self._phase2_optimizer_rows = self._read_rows_through_anchor(
            self._output_dir / "phase2_optimizers.csv",
            anchor_iteration=anchor_iteration,
        )
        self._hydrate_cumulative_state()
        self._hydrate_best_records(anchor_iteration)
        self._write_iteration_metrics_csv()
        self._rewrite_result_rows_csv(
            "phase2_solvers.csv",
            self.phase2_solver_rows,
            entry_kind="solver",
        )
        self._rewrite_result_rows_csv(
            "phase2_optimizers.csv",
            self.phase2_optimizer_rows,
            entry_kind="optimizer",
        )

    @staticmethod
    def _read_rows_through_anchor(path: Path, *, anchor_iteration: int) -> list[dict[str, object]]:
        if not path.is_file():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        kept: list[dict[str, object]] = []
        for row in rows:
            iteration = _int_or_none(row.get("iteration"))
            if iteration is None:
                _LOGGER.warning("Skipping telemetry row without integer iteration in %s.", path)
                continue
            if iteration <= anchor_iteration:
                kept.append(dict(row))
        return kept

    def _hydrate_cumulative_state(self) -> None:
        if not self._iteration_rows:
            return
        last_row = self._iteration_rows[-1]
        for key in USAGE_KEYS:
            phase2_value = _float_or_none(last_row.get(f"usage/cumulative/phase2/{key}"))
            if phase2_value is not None:
                self._cumulative_phase2_usage[key] = phase2_value
            cumulative_value = _float_or_none(last_row.get(f"usage/cumulative/{key}"))
            if cumulative_value is not None:
                self._cumulative_usage[key] = cumulative_value
        prefix = "phase2/cumulative/max/"
        self._cumulative_phase2_max_scores = {
            key.removeprefix(prefix): value
            for key, raw_value in last_row.items()
            if key.startswith(prefix) and (value := _float_or_none(raw_value)) is not None
        }

    def _hydrate_best_records(self, anchor_iteration: int) -> None:
        summary = self._read_summary()
        self._best_solver_record = self._record_from_summary(
            summary,
            kind="solver",
            preferred_key="score",
            anchor_iteration=anchor_iteration,
        ) or self._best_record_from_rows(
            self._phase2_solver_rows,
            id_key="solver_id",
            preferred_key="score",
        )
        self._best_optimizer_record = self._record_from_summary(
            summary,
            kind="optimizer",
            preferred_key="elo",
            anchor_iteration=anchor_iteration,
        ) or self._best_record_from_rows(
            self._phase2_optimizer_rows,
            id_key="optimizer_id",
            preferred_key="elo",
        )

    def _read_summary(self) -> dict[str, object]:
        assert self._output_dir is not None
        summary_path = self._output_dir / "summary.json"
        if not summary_path.is_file():
            return {}
        try:
            raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw_summary, dict):
            return {}
        return raw_summary

    @staticmethod
    def _record_from_summary(
        summary: dict[str, object],
        *,
        kind: str,
        preferred_key: str,
        anchor_iteration: int,
    ) -> dict[str, object] | None:
        entry_id = summary.get(f"best_{kind}_id")
        score = summary.get(f"best_{kind}_score")
        if not isinstance(entry_id, str) or not entry_id or score is None:
            return None
        first_seen_iteration = _int_or_none(summary.get(f"best_{kind}_first_seen_iteration"))
        if first_seen_iteration is not None:
            if first_seen_iteration > anchor_iteration:
                return None
        else:
            total_iterations = _int_or_none(summary.get("total_iterations"))
            if total_iterations is not None and total_iterations > anchor_iteration:
                return None
        try:
            scalar_score = scalar(score, preferred_key=preferred_key)
        except TypeError:
            return None
        return {
            "id": entry_id,
            "score": score,
            "scalar_score": scalar_score,
            "first_seen_iteration": first_seen_iteration,
        }

    @staticmethod
    def _best_record_from_rows(
        rows: list[dict[str, object]],
        *,
        id_key: str,
        preferred_key: str,
    ) -> dict[str, object] | None:
        best_record: dict[str, object] | None = None
        best_score = float("-inf")
        for row in rows:
            entry_id = row.get(id_key)
            if not isinstance(entry_id, str) or not entry_id:
                continue
            iteration = _int_or_none(row.get("iteration"))
            try:
                score = json.loads(str(row.get("score_json", "")))
                scalar_score = scalar(score, preferred_key=preferred_key)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if scalar_score <= best_score:
                continue
            best_score = scalar_score
            best_record = {
                "id": entry_id,
                "score": score,
                "scalar_score": scalar_score,
                "first_seen_iteration": iteration,
            }
        return best_record
