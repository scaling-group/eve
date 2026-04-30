"""Base logger types for Eve runs."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import optree
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.score import scalar
from scaling_evolve.algorithms.eve.workflow.optimize_logs import summarize_rollout_usage
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4Result

_USAGE_KEYS = (
    "model_cost_usd",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "agent_turns",
    "wallclock_seconds",
)
_RESULT_TABLE_SHARED_COLUMNS = (
    "iteration",
    "worker_index",
    "workspace_id",
    "score_json",
    "agent_turns",
    "cache_creation_tokens",
    "cache_read_tokens",
    "input_tokens",
    "model_cost_usd",
    "output_tokens",
    "wallclock_seconds",
)


class EveLogger:
    """Base class for Eve loggers with shared payload construction."""

    def __init__(self, *, excluded_score_fields: list[str] | tuple[str, ...]) -> None:
        self._excluded_score_fields = set(excluded_score_fields)
        self._cumulative_phase2_usage = {key: 0.0 for key in _USAGE_KEYS}
        self._cumulative_phase4_usage = {key: 0.0 for key in _USAGE_KEYS}
        self._cumulative_usage = {key: 0.0 for key in _USAGE_KEYS}
        self._cumulative_phase2_max_scores: dict[str, float] = {}
        self._cumulative_phase4_max_scores: dict[str, float] = {}
        self._phase2_solver_rows: list[dict[str, object]] = []
        self._phase2_optimizer_rows: list[dict[str, object]] = []
        self._phase4_optimizer_rows: list[dict[str, object]] = []
        self._best_solver_record: dict[str, Any] | None = None
        self._best_optimizer_record: dict[str, Any] | None = None

    @property
    def phase2_solver_rows(self) -> list[dict[str, object]]:
        return list(self._phase2_solver_rows)

    @property
    def phase2_optimizer_rows(self) -> list[dict[str, object]]:
        return list(self._phase2_optimizer_rows)

    @property
    def phase4_optimizer_rows(self) -> list[dict[str, object]]:
        return list(self._phase4_optimizer_rows)

    @classmethod
    def result_table_columns(cls, *, entry_kind: str) -> tuple[str, ...]:
        return (
            _RESULT_TABLE_SHARED_COLUMNS[0],
            _RESULT_TABLE_SHARED_COLUMNS[1],
            f"{entry_kind}_id",
            *_RESULT_TABLE_SHARED_COLUMNS[2:],
        )

    def on_iteration(
        self,
        *,
        iteration: int,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        phase2_results: list[Phase2Result],
        phase4_results: list[Phase4Result],
    ) -> None:
        raise NotImplementedError

    def finish(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> None:
        raise NotImplementedError

    def _build_iteration_payload(
        self,
        *,
        iteration: int,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        phase2_results: list[Phase2Result],
        phase4_results: list[Phase4Result],
    ) -> dict[str, object]:
        projected_phase2_scores = [
            self._flatten_numeric_score(
                result.produced_solver.score,
                excluded_fields=self._excluded_score_fields,
            )
            for result in phase2_results
            if result.produced_solver is not None
        ]
        projected_phase4_scores = [
            self._flatten_numeric_score(
                result.produced_optimizer.score,
                excluded_fields=self._excluded_score_fields,
            )
            for result in phase4_results
            if result.produced_optimizer is not None
        ]
        phase2_usage_totals = summarize_rollout_usage(
            [rollout for result in phase2_results for rollout in result.rollouts]
        )
        phase4_usage_totals = summarize_rollout_usage(
            [rollout for result in phase4_results for rollout in result.rollouts]
        )
        iteration_usage_totals = {
            key: float(phase2_usage_totals[key] + phase4_usage_totals[key]) for key in _USAGE_KEYS
        }
        for key in _USAGE_KEYS:
            self._cumulative_phase2_usage[key] += float(phase2_usage_totals[key])
            self._cumulative_phase4_usage[key] += float(phase4_usage_totals[key])
            self._cumulative_usage[key] += iteration_usage_totals[key]
        payload: dict[str, object] = {
            "iteration": iteration,
            "population/solver_size": len(solver_entries),
            "population/optimizer_size": len(optimizer_entries),
            **self._prefix_usage_metrics("usage/iteration", iteration_usage_totals),
            **self._prefix_usage_metrics("usage/cumulative/phase2", self._cumulative_phase2_usage),
            **self._prefix_usage_metrics("usage/cumulative/phase4", self._cumulative_phase4_usage),
            **self._prefix_usage_metrics("usage/cumulative", self._cumulative_usage),
        }
        if phase2_results:
            phase2_score_payload, phase2_iteration_max_scores, phase2_features = (
                self._score_feature_metrics("phase2", projected_phase2_scores)
            )
            for feature, value in phase2_iteration_max_scores.items():
                self._cumulative_phase2_max_scores[feature] = max(
                    self._cumulative_phase2_max_scores.get(feature, value),
                    value,
                )
            payload.update(phase2_score_payload)
            payload.update(
                {
                    f"phase2/cumulative/max/{feature}": self._cumulative_phase2_max_scores[feature]
                    for feature in phase2_features
                }
            )
            payload.update(self._prefix_usage_metrics("usage/phase2", phase2_usage_totals))
        if phase4_results:
            phase4_score_payload, phase4_iteration_max_scores, phase4_features = (
                self._score_feature_metrics("phase4", projected_phase4_scores)
            )
            for feature, value in phase4_iteration_max_scores.items():
                self._cumulative_phase4_max_scores[feature] = max(
                    self._cumulative_phase4_max_scores.get(feature, value),
                    value,
                )
            payload.update(phase4_score_payload)
            payload.update(
                {
                    f"phase4/cumulative/max/{feature}": self._cumulative_phase4_max_scores[feature]
                    for feature in phase4_features
                }
            )
            payload.update(self._prefix_usage_metrics("usage/phase4", phase4_usage_totals))

        phase2_solver_rows = self._build_result_rows(
            iteration=iteration,
            phase_results=phase2_results,
            result_attr="produced_solver",
            entry_kind="solver",
        )
        phase2_optimizer_rows = self._build_result_rows(
            iteration=iteration,
            phase_results=phase2_results,
            result_attr="produced_optimizer",
            entry_kind="optimizer",
        )
        phase4_optimizer_rows = self._build_result_rows(
            iteration=iteration,
            phase_results=phase4_results,
            result_attr="produced_optimizer",
            entry_kind="optimizer",
        )
        self._phase2_solver_rows.extend(phase2_solver_rows)
        self._phase2_optimizer_rows.extend(phase2_optimizer_rows)
        self._phase4_optimizer_rows.extend(phase4_optimizer_rows)
        self._update_best_entry_record(
            entries=solver_entries,
            iteration=iteration,
            preferred_key="score",
            current_attr="_best_solver_record",
        )
        self._update_best_entry_record(
            entries=optimizer_entries,
            iteration=iteration,
            preferred_key="elo",
            current_attr="_best_optimizer_record",
        )
        return payload

    def _build_finish_summary(
        self,
        *,
        solver_entries: list[PopulationEntry],
        optimizer_entries: list[PopulationEntry],
        iterations_completed: int,
    ) -> dict[str, Any]:
        best_solver_record = self._resolve_best_record(
            entries=solver_entries,
            preferred_key="score",
            stored_record=self._best_solver_record,
        )
        best_optimizer_record = self._resolve_best_record(
            entries=optimizer_entries,
            preferred_key="elo",
            stored_record=self._best_optimizer_record,
        )
        return {
            "total_iterations": iterations_completed,
            "solver_pop_size": len(solver_entries),
            "optimizer_pop_size": len(optimizer_entries),
            "best_solver_score": best_solver_record["score"] if best_solver_record else None,
            "best_solver_first_seen_iteration": (
                best_solver_record["first_seen_iteration"] if best_solver_record else None
            ),
            "best_solver_id": best_solver_record["id"] if best_solver_record else None,
            "best_optimizer_score": best_optimizer_record["score"]
            if best_optimizer_record
            else None,
            "best_optimizer_first_seen_iteration": (
                best_optimizer_record["first_seen_iteration"] if best_optimizer_record else None
            ),
            "best_optimizer_id": best_optimizer_record["id"] if best_optimizer_record else None,
        }

    @staticmethod
    def _prefix_usage_metrics(prefix: str, usage_totals: dict[str, float]) -> dict[str, float]:
        return {f"{prefix}/{key}": float(usage_totals[key]) for key in _USAGE_KEYS}

    @staticmethod
    def _flatten_numeric_score(
        score: PyTree,
        *,
        excluded_fields: set[str],
        prefix: str = "",
    ) -> dict[str, float]:
        paths, leaves, _treespec = optree.tree_flatten_with_path(score)
        flat: dict[str, float] = {}
        for raw_path, leaf in zip(paths, leaves, strict=True):
            if any(isinstance(part, str) and part in excluded_fields for part in raw_path):
                continue
            if not isinstance(leaf, (int, float)) or isinstance(leaf, bool):
                continue
            path = "/".join(str(part) for part in raw_path)
            if not path:
                continue
            feature = f"{prefix}/{path}" if prefix else path
            flat[feature] = float(leaf)
        return flat

    @staticmethod
    def _score_feature_metrics(
        prefix: str,
        flattened_scores: list[dict[str, float]],
    ) -> tuple[dict[str, float | list[float]], dict[str, float], set[str]]:
        payload: dict[str, float | list[float]] = {}
        cumulative_candidates: dict[str, float] = {}
        feature_values: dict[str, list[float]] = defaultdict(list)

        for flattened in flattened_scores:
            for feature, value in flattened.items():
                feature_values[feature].append(value)

        seen_features = set(feature_values)
        for feature, values in feature_values.items():
            payload[f"{prefix}/scores/{feature}"] = list(values)
            payload[f"{prefix}/iteration/mean/{feature}"] = sum(values) / len(values)
            payload[f"{prefix}/iteration/max/{feature}"] = max(values)
            cumulative_candidates[feature] = max(values)
        return payload, cumulative_candidates, seen_features

    @staticmethod
    def _result_usage_row(rollouts: list[object]) -> dict[str, float]:
        usage = summarize_rollout_usage(rollouts)
        return {key: float(usage[key]) for key in _USAGE_KEYS}

    @staticmethod
    def _entry_record(
        *,
        entry: PopulationEntry,
        iteration: int | None,
        preferred_key: str,
    ) -> dict[str, Any]:
        return {
            "id": entry.id,
            "score": entry.score,
            "scalar_score": scalar(entry.score, preferred_key=preferred_key),
            "first_seen_iteration": iteration,
        }

    def _update_best_entry_record(
        self,
        *,
        entries: list[PopulationEntry],
        iteration: int,
        preferred_key: str,
        current_attr: str,
    ) -> None:
        best_entry = (
            max(entries, key=lambda entry: scalar(entry.score, preferred_key=preferred_key))
            if entries
            else None
        )
        if best_entry is None:
            return
        current_record = getattr(self, current_attr)
        current_best_score = (
            float(current_record["scalar_score"]) if current_record is not None else float("-inf")
        )
        candidate_score = scalar(best_entry.score, preferred_key=preferred_key)
        if candidate_score > current_best_score:
            setattr(
                self,
                current_attr,
                self._entry_record(
                    entry=best_entry,
                    iteration=iteration,
                    preferred_key=preferred_key,
                ),
            )
            return
        if candidate_score < current_best_score:
            return
        if current_record is None:
            setattr(
                self,
                current_attr,
                self._entry_record(
                    entry=best_entry,
                    iteration=iteration,
                    preferred_key=preferred_key,
                ),
            )
            return

    def _resolve_best_record(
        self,
        *,
        entries: list[PopulationEntry],
        preferred_key: str,
        stored_record: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if stored_record is not None:
            return dict(stored_record)
        if not entries:
            return None
        best_entry = max(
            entries, key=lambda entry: scalar(entry.score, preferred_key=preferred_key)
        )
        return self._entry_record(entry=best_entry, iteration=None, preferred_key=preferred_key)

    def _build_result_rows(
        self,
        *,
        iteration: int,
        phase_results: list[Phase2Result] | list[Phase4Result],
        result_attr: str,
        entry_kind: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, result in enumerate(phase_results):
            entry = getattr(result, result_attr)
            if entry is None:
                continue
            row: dict[str, object] = {
                "iteration": iteration,
                "worker_index": index,
                f"{entry_kind}_id": entry.id,
                "workspace_id": result.workspace_id,
                "score_json": json.dumps(entry.score, sort_keys=True, default=str),
            }
            row.update(self._result_usage_row(result.rollouts))
            rows.append(row)
        return rows
