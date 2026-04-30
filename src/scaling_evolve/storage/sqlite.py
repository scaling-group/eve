"""SQLite-backed lineage persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scaling_evolve.core.common import JSONValue
from scaling_evolve.core.engine import PortableState
from scaling_evolve.core.evaluation import (
    BudgetLedger,
    EvaluationResult,
    ScoreCard,
    stage_feedback,
    terminal_stage_result,
)
from scaling_evolve.core.node import (
    EdgeKindValue,
    EdgeLifecycleStatus,
    EdgeRecord,
    EdgeRecordLike,
    ExecutionSegmentRecord,
    InheritanceModeValue,
    Node,
    NodeLifecycleStatus,
    NodeRecord,
    NodeRecordLike,
    ProviderKindValue,
    SessionInstanceRecord,
)
from scaling_evolve.core.storage.models import ArtifactRef, MaterializationRef
from scaling_evolve.providers.agent.state import RuntimeState
from scaling_evolve.storage.lineage_store import RunRecord

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
      run_id TEXT PRIMARY KEY,
      config_ref TEXT,
      status TEXT NOT NULL,
      app_kind TEXT NOT NULL,
      started_at TEXT NOT NULL,
      finished_at TEXT,
      seed INTEGER,
      notes_json TEXT NOT NULL,
      run_name TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS nodes (
      run_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      individual_id TEXT,
      parent_node_id TEXT,
      generation INTEGER NOT NULL,
      status TEXT NOT NULL,
      workspace_path TEXT,
      approach_summary TEXT,
      approach_card_ref TEXT,
      materialization_ref TEXT NOT NULL,
      portable_state_ref TEXT NOT NULL,
      runtime_state_ref TEXT,
      score_ref TEXT NOT NULL,
      primary_score REAL NOT NULL,
      score_status TEXT NOT NULL,
      budget_json TEXT NOT NULL,
      tags_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL,
      PRIMARY KEY(run_id, node_id),
      FOREIGN KEY(run_id) REFERENCES runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
      run_id TEXT NOT NULL,
      edge_id TEXT NOT NULL,
      iteration INTEGER NOT NULL,
      parent_node_id TEXT NOT NULL,
      child_node_id TEXT,
      edge_kind TEXT NOT NULL DEFAULT 'fork',
      provider_kind TEXT NOT NULL,
      inheritance_mode TEXT NOT NULL,
      instruction_ref TEXT,
      projected_state_ref TEXT,
      result_ref TEXT,
      delta_summary TEXT,
      mutation_note_ref TEXT,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY(run_id, edge_id),
      FOREIGN KEY(run_id) REFERENCES runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edge_executions (
      run_id TEXT NOT NULL,
      edge_id TEXT NOT NULL,
      result_subtype TEXT NOT NULL,
      exit_code INTEGER,
      duration_seconds REAL,
      input_tokens INTEGER NOT NULL DEFAULT 0,
      output_tokens INTEGER NOT NULL DEFAULT 0,
      cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
      cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
      cost_usd REAL,
      cost_source TEXT,
      cache_discount_usd REAL NOT NULL DEFAULT 0.0,
      session_id TEXT,
      failure_kind TEXT,
      failure_message TEXT,
      created_at TEXT NOT NULL,
      PRIMARY KEY(run_id, edge_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS features (
      run_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      dimension TEXT NOT NULL,
      value REAL NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY(run_id, node_id, dimension)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluations (
      run_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      status TEXT NOT NULL,
      score REAL,
      summary TEXT,
      metrics_json TEXT NOT NULL DEFAULT '{}',
      checks_json TEXT NOT NULL DEFAULT '{}',
      feedback_text TEXT,
      created_at TEXT NOT NULL,
      PRIMARY KEY(run_id, node_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
      artifact_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      node_id TEXT,
      edge_id TEXT,
      artifact_kind TEXT NOT NULL,
      rel_path TEXT NOT NULL,
      sha256 TEXT NOT NULL,
      size_bytes INTEGER,
      created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      node_id TEXT,
      edge_id TEXT,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attention_events (
      event_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      node_id TEXT,
      edge_id TEXT,
      review_task_id TEXT,
      event_type TEXT NOT NULL,
      created_at TEXT NOT NULL,
      record_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_tasks (
      task_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      edge_id TEXT,
      status TEXT NOT NULL,
      recommended_action TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      record_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS human_decisions (
      decision_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      review_task_id TEXT,
      node_id TEXT,
      edge_id TEXT,
      decision_type TEXT NOT NULL,
      created_at TEXT NOT NULL,
      record_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_instances (
      session_instance_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      individual_id TEXT NOT NULL,
      base_checkpoint_id TEXT,
      driver_name TEXT,
      provider_session_id TEXT,
      workspace_id TEXT,
      status TEXT,
      started_at TEXT,
      ended_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS execution_segments (
      segment_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      session_instance_id TEXT NOT NULL,
      reason TEXT,
      native_ref TEXT,
      started_at TEXT,
      finished_at TEXT,
      transcript_ref TEXT,
      cost_json TEXT NOT NULL DEFAULT '{}',
      metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_nodes_run_score ON nodes(run_id, primary_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_run_generation ON nodes(run_id, generation DESC)",
    "CREATE INDEX IF NOT EXISTS idx_edges_run_iteration ON edges(run_id, iteration DESC)",
    """
    CREATE INDEX IF NOT EXISTS idx_edge_executions_run_created
    ON edge_executions(run_id, created_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_features_run_node
    ON features(run_id, node_id, dimension)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_evaluations_run_created
    ON evaluations(run_id, created_at ASC)
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind ON artifacts(run_id, artifact_kind)",
    """
    CREATE INDEX IF NOT EXISTS idx_attention_events_run_created
    ON attention_events(run_id, created_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_review_tasks_run_status
    ON review_tasks(run_id, status, updated_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_human_decisions_run_created
    ON human_decisions(run_id, created_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_instances_run_created
    ON session_instances(run_id, started_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_execution_segments_run_created
    ON execution_segments(run_id, started_at ASC)
    """,
)
_NODES_COLUMNS = (
    "run_id",
    "node_id",
    "individual_id",
    "parent_node_id",
    "generation",
    "status",
    "workspace_path",
    "approach_summary",
    "approach_card_ref",
    "materialization_ref",
    "portable_state_ref",
    "runtime_state_ref",
    "score_ref",
    "primary_score",
    "score_status",
    "budget_json",
    "tags_json",
    "created_at",
)
_EDGES_COLUMNS = (
    "run_id",
    "edge_id",
    "iteration",
    "parent_node_id",
    "child_node_id",
    "edge_kind",
    "provider_kind",
    "inheritance_mode",
    "instruction_ref",
    "projected_state_ref",
    "result_ref",
    "delta_summary",
    "mutation_note_ref",
    "status",
    "created_at",
)
_LOGGER = logging.getLogger(__name__)
_SEEN_LEGACY_WARNINGS: set[str] = set()


def _warn_legacy_once(message: str) -> None:
    if message in _SEEN_LEGACY_WARNINGS:
        return
    _SEEN_LEGACY_WARNINGS.add(message)
    _LOGGER.warning(message)


def _json_dumps(value: Any) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return json.dumps(payload, sort_keys=True)


def _load_model(raw: str | None, model_cls: type[Any], default: Any) -> Any:
    if raw is None or raw == "" or raw == "null":
        return default
    return model_cls.model_validate_json(raw)


def _load_tags(raw: str | None) -> dict[str, str]:
    if raw is None or raw == "" or raw == "null":
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()  # type: ignore[union-attr]
        if isinstance(key, str) and isinstance(value, str)
    }


def _load_notes(raw: str | None) -> dict[str, JSONValue]:
    if raw is None or raw == "" or raw == "null":
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, JSONValue], payload)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_node_status(value: str) -> NodeLifecycleStatus:
    if value in {"seed", "evaluated", "invalid", "failed"}:
        return cast(NodeLifecycleStatus, value)
    return "evaluated"


def _normalize_edge_status(value: str) -> EdgeLifecycleStatus:
    if value in {"started", "mutated", "evaluated", "failed"}:
        return cast(EdgeLifecycleStatus, value)
    return "failed"


def _normalize_provider_kind(value: str) -> ProviderKindValue:
    if value == "session":
        return "agent_fork"
    if value == "agent_sdk":
        _warn_legacy_once("Legacy provider kind 'agent_sdk' mapped to 'agent_fork'.")
        return "agent_fork"
    return cast(ProviderKindValue, value)


def _normalize_edge_kind(value: str) -> EdgeKindValue:
    if value in {"fork", "continuation", "inspiration"}:
        return cast(EdgeKindValue, value)
    return "fork"


def _normalize_inheritance_mode(value: str) -> InheritanceModeValue:
    if value == "rehydrate":
        _warn_legacy_once("Legacy inheritance mode 'rehydrate' mapped to 'summary_only'.")
        return "summary_only"
    if value == "fresh":
        return "summary_only"
    if value in {"summary_only", "native"}:
        return cast(InheritanceModeValue, value)
    return "summary_only"


def _coerce_node(node: NodeRecordLike | NodeRecord) -> NodeRecord:
    if isinstance(node, NodeRecord):
        return node

    score_value = node.score if node.score is not None else float("-inf")
    placeholder_ref = ArtifactRef(
        artifact_id=f"{node.node_id}:placeholder",
        kind="source",
        relpath=f"{node.node_id}/placeholder.py",
    )
    return NodeRecord(
        node_id=node.node_id,
        run_id=node.run_id,
        individual_id=node.node_id,
        parent_node_id=node.parent_id,
        generation=0,
        materialization=MaterializationRef(
            materialization_id=f"materialization-{node.node_id}",
            primary_artifact=placeholder_ref,
            snapshot_artifact=placeholder_ref,
        ),
        portable_state=PortableState(summary="coerced from NodeRecordLike"),
        score=ScoreCard(primary_score=score_value, status="ok", summary="coerced score"),
        budget=BudgetLedger(),
        status=_normalize_node_status(str(node.status)),
    )


def _coerce_edge(edge: EdgeRecordLike | EdgeRecord) -> EdgeRecord:
    if isinstance(edge, EdgeRecord):
        return edge

    return EdgeRecord(
        edge_id=edge.edge_id,
        run_id=edge.run_id,
        iteration=0,
        parent_node_id=edge.parent_id,
        child_node_id=edge.child_id,
        edge_kind="fork",
        provider_kind="llm",
        inheritance_mode="summary_only",
        status=_normalize_edge_status(str(edge.status)),
    )


class SQLiteLineageStore:
    """Persist run lineage, archive state, and artifacts in SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def _initialize(self) -> None:
        with self.connection:
            for statement in _SCHEMA_STATEMENTS:
                self.connection.execute(statement)
            self._ensure_nodes_tags_column()
            self._ensure_nodes_individual_id_column()
            self._ensure_nodes_workspace_path_column()
            self._ensure_nodes_approach_summary_column()
            self._ensure_nodes_approach_card_ref_column()
            self._ensure_edges_edge_kind_column()
            self._ensure_edges_delta_summary_column()
            self._ensure_edges_mutation_note_ref_column()
            self._ensure_edge_executions_cost_columns()
            self._ensure_composite_primary_key(
                table_name="nodes",
                expected_pk=("run_id", "node_id"),
                create_sql=_SCHEMA_STATEMENTS[1],
                columns=_NODES_COLUMNS,
            )
            self._ensure_composite_primary_key(
                table_name="edges",
                expected_pk=("run_id", "edge_id"),
                create_sql=_SCHEMA_STATEMENTS[2],
                columns=_EDGES_COLUMNS,
            )

    def _ensure_nodes_tags_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "tags_json" in columns:
            return
        self.connection.execute("ALTER TABLE nodes ADD COLUMN tags_json TEXT NOT NULL DEFAULT '{}'")

    def _ensure_nodes_individual_id_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "individual_id" not in columns:
            self.connection.execute("ALTER TABLE nodes ADD COLUMN individual_id TEXT")
        self.connection.execute(
            "UPDATE nodes SET individual_id = node_id "
            "WHERE individual_id IS NULL OR individual_id = ''"
        )

    def _ensure_nodes_workspace_path_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "workspace_path" in columns:
            return
        self.connection.execute("ALTER TABLE nodes ADD COLUMN workspace_path TEXT")

    def _ensure_nodes_approach_summary_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "approach_summary" in columns:
            return
        self.connection.execute("ALTER TABLE nodes ADD COLUMN approach_summary TEXT")

    def _ensure_nodes_approach_card_ref_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "approach_card_ref" in columns:
            return
        self.connection.execute("ALTER TABLE nodes ADD COLUMN approach_card_ref TEXT")

    def _ensure_edges_edge_kind_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "edge_kind" in columns:
            return
        self.connection.execute(
            "ALTER TABLE edges ADD COLUMN edge_kind TEXT NOT NULL DEFAULT 'fork'"
        )

    def _ensure_edges_delta_summary_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "delta_summary" in columns:
            return
        self.connection.execute("ALTER TABLE edges ADD COLUMN delta_summary TEXT")

    def _ensure_edges_mutation_note_ref_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "mutation_note_ref" in columns:
            return
        self.connection.execute("ALTER TABLE edges ADD COLUMN mutation_note_ref TEXT")

    def _ensure_edge_executions_cost_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(edge_executions)").fetchall()
        }
        if "cost_usd" not in columns:
            self.connection.execute("ALTER TABLE edge_executions ADD COLUMN cost_usd REAL")
        if "cost_source" not in columns:
            self.connection.execute("ALTER TABLE edge_executions ADD COLUMN cost_source TEXT")
        if "cache_discount_usd" not in columns:
            self.connection.execute(
                "ALTER TABLE edge_executions "
                "ADD COLUMN cache_discount_usd REAL NOT NULL DEFAULT 0.0"
            )

    def _ensure_composite_primary_key(
        self,
        *,
        table_name: str,
        expected_pk: tuple[str, ...],
        create_sql: str,
        columns: tuple[str, ...],
    ) -> None:
        if self._primary_key_columns(table_name) == expected_pk:
            return

        legacy_table = f"{table_name}__legacy_pk"
        column_list = ", ".join(columns)
        self.connection.execute("PRAGMA foreign_keys = OFF")
        try:
            self.connection.execute(f"ALTER TABLE {table_name} RENAME TO {legacy_table}")
            self.connection.execute(create_sql)
            self.connection.execute(
                f"""
                INSERT INTO {table_name}({column_list})
                SELECT {column_list}
                FROM {legacy_table}
                """
            )
            self.connection.execute(f"DROP TABLE {legacy_table}")
            for statement in _SCHEMA_STATEMENTS:
                if "INDEX" in statement:
                    self.connection.execute(statement)
        finally:
            self.connection.execute("PRAGMA foreign_keys = ON")

    def _primary_key_columns(self, table_name: str) -> tuple[str, ...]:
        rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        pk_rows = sorted(
            (row for row in rows if int(row["pk"]) > 0),
            key=lambda row: int(row["pk"]),
        )
        return tuple(str(row["name"]) for row in pk_rows)

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self.connection.close()

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def create_run(self, run: RunRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO runs(
                  run_id, config_ref, status, app_kind, started_at, finished_at, seed, notes_json,
                  run_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  config_ref = excluded.config_ref,
                  status = excluded.status,
                  app_kind = excluded.app_kind,
                  started_at = excluded.started_at,
                  finished_at = excluded.finished_at,
                  seed = excluded.seed,
                  notes_json = excluded.notes_json,
                  run_name = excluded.run_name
                """,
                (
                    run.run_id,
                    _json_dumps(run.config_ref),
                    run.status,
                    run.app_kind,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at is not None else None,
                    run.seed,
                    _json_dumps(run.notes),
                    run.run_name,
                ),
            )

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: datetime | None = None,
    ) -> None:
        """Update a run status as the engine advances."""

        with self.connection:
            self.connection.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, finished_at.isoformat() if finished_at is not None else None, run_id),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a persisted run record by id."""

        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT
                       run_id,
                       run_name,
                       config_ref,
                       status,
                       app_kind,
                       started_at,
                       finished_at,
                       seed,
                       notes_json
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=str(row["run_id"]),
            run_name=str(row["run_name"]) if row["run_name"] is not None else None,
            config_ref=_load_model(row["config_ref"], ArtifactRef, None),
            status=str(row["status"]),
            app_kind=str(row["app_kind"]),
            started_at=datetime.fromisoformat(str(row["started_at"])),
            finished_at=(
                datetime.fromisoformat(str(row["finished_at"]))
                if row["finished_at"] is not None
                else None
            ),
            seed=int(row["seed"]) if row["seed"] is not None else None,
            notes=_load_notes(cast(str | None, row["notes_json"])),
        )

    def put(self, node: Node) -> None:
        """Persist a Spec 3 node through the legacy lineage schema."""

        if isinstance(node, NodeRecord):
            self.save_node(node)
            return
        self.save_node(NodeRecord.model_validate(node.model_dump(mode="python")))

    def get(self, node_id: str) -> NodeRecord | None:
        """Load the most recent node matching `node_id`."""

        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM nodes
                WHERE node_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        return self._node_from_row(row)

    def query(self, **filters: object) -> list[NodeRecord]:
        """Filter nodes by metadata-backed fields."""

        return [
            node
            for node in self.all()
            if all(
                self._node_matches_filter(node, key, expected) for key, expected in filters.items()
            )
        ]

    def ancestors(self, node_id: str) -> list[NodeRecord]:
        """Return ancestors from parent to root."""

        nodes_by_id = {node.node_id: node for node in self.all()}
        lineage: list[NodeRecord] = []
        current = nodes_by_id.get(node_id)
        while current is not None and current.parent_id is not None:
            parent = nodes_by_id.get(current.parent_id)
            if parent is None:
                break
            lineage.append(parent)
            current = parent
        return lineage

    def children(self, node_id: str) -> list[NodeRecord]:
        """Return direct children for `node_id`."""

        return [node for node in self.all() if node.parent_id == node_id]

    def all(self) -> list[NodeRecord]:
        """Return every persisted node across runs."""

        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM nodes ORDER BY run_id ASC, generation ASC, created_at ASC"
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def save_node(self, node: NodeRecordLike | NodeRecord) -> None:
        persisted = _coerce_node(node)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO nodes(
                  run_id, node_id, individual_id, parent_node_id, generation, status,
                  workspace_path, approach_summary, approach_card_ref,
                  materialization_ref, portable_state_ref, runtime_state_ref, score_ref,
                  primary_score, score_status, budget_json, tags_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                  individual_id = excluded.individual_id,
                  parent_node_id = excluded.parent_node_id,
                  generation = excluded.generation,
                  status = excluded.status,
                  workspace_path = excluded.workspace_path,
                  approach_summary = excluded.approach_summary,
                  approach_card_ref = excluded.approach_card_ref,
                  materialization_ref = excluded.materialization_ref,
                  portable_state_ref = excluded.portable_state_ref,
                  runtime_state_ref = excluded.runtime_state_ref,
                  score_ref = excluded.score_ref,
                  primary_score = excluded.primary_score,
                  score_status = excluded.score_status,
                  budget_json = excluded.budget_json,
                  tags_json = excluded.tags_json,
                  created_at = excluded.created_at
                """,
                (
                    persisted.run_id,
                    persisted.node_id,
                    persisted.individual_id,
                    persisted.parent_node_id,
                    persisted.generation,
                    persisted.status,
                    persisted.workspace_path,
                    persisted.approach_summary,
                    _json_dumps(persisted.approach_card_ref),
                    _json_dumps(persisted.materialization),
                    _json_dumps(persisted.portable_state),
                    _json_dumps(persisted.runtime_state),
                    _json_dumps(persisted.score),
                    persisted.primary_score,
                    persisted.score.status,
                    _json_dumps(persisted.budget),
                    _json_dumps(persisted.tags),
                    persisted.created_at.isoformat(),
                ),
            )
            self.connection.execute(
                "DELETE FROM features WHERE run_id = ? AND node_id = ?",
                (persisted.run_id, persisted.node_id),
            )
            feature_rows = [
                (
                    persisted.run_id,
                    persisted.node_id,
                    dimension,
                    float(value),
                    _utc_now().isoformat(),
                )
                for dimension, value in persisted.score.features.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            if feature_rows:
                self.connection.executemany(
                    """
                    INSERT INTO features(run_id, node_id, dimension, value, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    feature_rows,
                )

    def save_edge(self, edge: EdgeRecordLike | EdgeRecord) -> None:
        persisted = _coerce_edge(edge)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO edges(
                  run_id, edge_id, iteration, parent_node_id, child_node_id, edge_kind,
                  provider_kind, inheritance_mode, instruction_ref, projected_state_ref,
                  result_ref, delta_summary, mutation_note_ref, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, edge_id) DO UPDATE SET
                  iteration = excluded.iteration,
                  parent_node_id = excluded.parent_node_id,
                  child_node_id = excluded.child_node_id,
                  edge_kind = excluded.edge_kind,
                  provider_kind = excluded.provider_kind,
                  inheritance_mode = excluded.inheritance_mode,
                  instruction_ref = excluded.instruction_ref,
                  projected_state_ref = excluded.projected_state_ref,
                  result_ref = excluded.result_ref,
                  delta_summary = excluded.delta_summary,
                  mutation_note_ref = excluded.mutation_note_ref,
                  status = excluded.status,
                  created_at = excluded.created_at
                """,
                (
                    persisted.run_id,
                    persisted.edge_id,
                    persisted.iteration,
                    persisted.parent_node_id,
                    persisted.child_node_id,
                    persisted.edge_kind,
                    persisted.provider_kind,
                    persisted.inheritance_mode,
                    _json_dumps(persisted.instruction_ref),
                    _json_dumps(persisted.projected_state_ref),
                    _json_dumps(persisted.result_ref),
                    persisted.delta_summary,
                    _json_dumps(persisted.mutation_note_ref),
                    persisted.status,
                    persisted.created_at.isoformat(),
                ),
            )

    def save_session_instance(self, record: SessionInstanceRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO session_instances(
                  session_instance_id, run_id, individual_id, base_checkpoint_id, driver_name,
                  provider_session_id, workspace_id, status, started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_instance_id) DO UPDATE SET
                  run_id = excluded.run_id,
                  individual_id = excluded.individual_id,
                  base_checkpoint_id = excluded.base_checkpoint_id,
                  driver_name = excluded.driver_name,
                  provider_session_id = excluded.provider_session_id,
                  workspace_id = excluded.workspace_id,
                  status = excluded.status,
                  started_at = excluded.started_at,
                  ended_at = excluded.ended_at
                """,
                (
                    record.session_instance_id,
                    record.run_id,
                    record.individual_id,
                    record.base_checkpoint_id,
                    record.driver_name,
                    record.provider_session_id,
                    record.workspace_id,
                    record.status,
                    record.started_at.isoformat() if record.started_at is not None else None,
                    record.ended_at.isoformat() if record.ended_at is not None else None,
                ),
            )

    def save_execution_segment(self, record: ExecutionSegmentRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO execution_segments(
                  segment_id, run_id, session_instance_id, reason, native_ref, started_at,
                  finished_at, transcript_ref, cost_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                  run_id = excluded.run_id,
                  session_instance_id = excluded.session_instance_id,
                  reason = excluded.reason,
                  native_ref = excluded.native_ref,
                  started_at = excluded.started_at,
                  finished_at = excluded.finished_at,
                  transcript_ref = excluded.transcript_ref,
                  cost_json = excluded.cost_json,
                  metadata_json = excluded.metadata_json
                """,
                (
                    record.segment_id,
                    record.run_id,
                    record.session_instance_id,
                    record.reason,
                    record.native_ref,
                    record.started_at.isoformat() if record.started_at is not None else None,
                    record.finished_at.isoformat() if record.finished_at is not None else None,
                    _json_dumps(record.transcript_ref),
                    _json_dumps(record.cost),
                    _json_dumps(record.metadata),
                ),
            )

    def save_edge_execution(
        self,
        *,
        run_id: str,
        edge_id: str,
        result_subtype: str,
        exit_code: int | None,
        duration_seconds: float | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cost_usd: float | None = None,
        cost_source: str | None = None,
        cache_discount_usd: float = 0.0,
        session_id: str | None = None,
        failure_kind: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        """Persist canonical per-edge execution facts."""

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO edge_executions(
                  run_id, edge_id, result_subtype, exit_code, duration_seconds, input_tokens,
                  output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                  cost_usd, cost_source, cache_discount_usd, session_id, failure_kind,
                  failure_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, edge_id) DO UPDATE SET
                  result_subtype = excluded.result_subtype,
                  exit_code = excluded.exit_code,
                  duration_seconds = excluded.duration_seconds,
                  input_tokens = excluded.input_tokens,
                  output_tokens = excluded.output_tokens,
                  cache_read_input_tokens = excluded.cache_read_input_tokens,
                  cache_creation_input_tokens = excluded.cache_creation_input_tokens,
                  cost_usd = excluded.cost_usd,
                  cost_source = excluded.cost_source,
                  cache_discount_usd = excluded.cache_discount_usd,
                  session_id = excluded.session_id,
                  failure_kind = excluded.failure_kind,
                  failure_message = excluded.failure_message,
                  created_at = excluded.created_at
                """,
                (
                    run_id,
                    edge_id,
                    result_subtype,
                    exit_code,
                    duration_seconds,
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    cost_usd,
                    cost_source,
                    cache_discount_usd,
                    session_id,
                    failure_kind,
                    failure_message,
                    _utc_now().isoformat(),
                ),
            )

    def save_evaluation(
        self,
        *,
        run_id: str,
        node_id: str,
        evaluation: EvaluationResult,
    ) -> None:
        """Persist evaluator-facing summary facts for one node."""

        feedback_text = stage_feedback(terminal_stage_result(evaluation)) or evaluation.summary
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO evaluations(
                  run_id, node_id, status, score, summary, metrics_json, checks_json,
                  feedback_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                  status = excluded.status,
                  score = excluded.score,
                  summary = excluded.summary,
                  metrics_json = excluded.metrics_json,
                  checks_json = excluded.checks_json,
                  feedback_text = excluded.feedback_text,
                  created_at = excluded.created_at
                """,
                (
                    run_id,
                    node_id,
                    evaluation.status.value,
                    evaluation.score,
                    evaluation.summary,
                    _json_dumps(evaluation.metrics.metrics),
                    _json_dumps(evaluation.metrics.checks),
                    feedback_text,
                    _utc_now().isoformat(),
                ),
            )

    def list_nodes(self, run_id: str) -> Sequence[NodeRecord]:
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM nodes WHERE run_id = ? ORDER BY generation ASC, created_at ASC",
                (run_id,),
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def list_edges(self, run_id: str) -> Sequence[EdgeRecord]:
        """List persisted mutation edges for a run."""

        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM edges WHERE run_id = ? ORDER BY iteration ASC, created_at ASC",
                (run_id,),
            ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def list_edge_executions(self, run_id: str) -> Sequence[dict[str, JSONValue]]:
        """List canonical per-edge execution facts for a run."""

        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT edge_id, result_subtype, exit_code, duration_seconds, input_tokens,
                       output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                       cost_usd, cost_source, cache_discount_usd, session_id, failure_kind,
                       failure_message
                FROM edge_executions
                WHERE run_id = ?
                ORDER BY created_at ASC, edge_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "edge_id": str(row["edge_id"]),
                "result_subtype": str(row["result_subtype"]),
                "exit_code": int(row["exit_code"]) if row["exit_code"] is not None else None,
                "duration_seconds": (
                    float(row["duration_seconds"]) if row["duration_seconds"] is not None else None
                ),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "cache_read_input_tokens": int(row["cache_read_input_tokens"]),
                "cache_creation_input_tokens": int(row["cache_creation_input_tokens"]),
                "cost_usd": float(row["cost_usd"]) if row["cost_usd"] is not None else None,
                "cost_source": str(row["cost_source"]) if row["cost_source"] is not None else None,
                "cache_discount_usd": float(row["cache_discount_usd"] or 0.0),
                "session_id": str(row["session_id"]) if row["session_id"] is not None else None,
                "failure_kind": (
                    str(row["failure_kind"]) if row["failure_kind"] is not None else None
                ),
                "failure_message": (
                    str(row["failure_message"]) if row["failure_message"] is not None else None
                ),
            }
            for row in rows
        ]

    def list_features(self, run_id: str) -> dict[str, dict[str, float]]:
        """Return numeric feature values grouped by node id."""

        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT node_id, dimension, value
                FROM features
                WHERE run_id = ?
                ORDER BY node_id ASC, dimension ASC
                """,
                (run_id,),
            ).fetchall()
        features: dict[str, dict[str, float]] = {}
        for row in rows:
            node_id = str(row["node_id"])
            features.setdefault(node_id, {})[str(row["dimension"])] = float(row["value"])
        return features

    def get_evaluation(self, run_id: str, node_id: str) -> dict[str, JSONValue] | None:
        """Load one persisted evaluation summary."""

        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT status, score, summary, metrics_json, checks_json, feedback_text
                FROM evaluations
                WHERE run_id = ? AND node_id = ?
                """,
                (run_id, node_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": str(row["status"]),
            "score": float(row["score"]) if row["score"] is not None else None,
            "summary": str(row["summary"]) if row["summary"] is not None else None,
            "metrics": _load_notes(cast(str | None, row["metrics_json"])),
            "checks": _load_notes(cast(str | None, row["checks_json"])),
            "feedback_text": (
                str(row["feedback_text"]) if row["feedback_text"] is not None else None
            ),
        }

    def list_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
    ) -> Sequence[dict[str, JSONValue]]:
        """List structured events for a run."""

        with self._read_connection() as connection:
            if event_type is None:
                rows = connection.execute(
                    """
                    SELECT event_id, node_id, edge_id, event_type, payload_json, created_at
                    FROM events
                    WHERE run_id = ?
                    ORDER BY created_at ASC, event_id ASC
                    """,
                    (run_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT event_id, node_id, edge_id, event_type, payload_json, created_at
                    FROM events
                    WHERE run_id = ? AND event_type = ?
                    ORDER BY created_at ASC, event_id ASC
                    """,
                    (run_id, event_type),
                ).fetchall()
        return [
            {
                "event_id": str(row["event_id"]),
                "node_id": str(row["node_id"]) if row["node_id"] is not None else None,
                "edge_id": str(row["edge_id"]) if row["edge_id"] is not None else None,
                "event_type": str(row["event_type"]),
                "payload": _load_notes(cast(str | None, row["payload_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def list_session_instances(self, run_id: str) -> Sequence[SessionInstanceRecord]:
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM session_instances
                WHERE run_id = ?
                ORDER BY started_at ASC, session_instance_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            SessionInstanceRecord(
                session_instance_id=str(row["session_instance_id"]),
                run_id=str(row["run_id"]),
                individual_id=str(row["individual_id"]),
                base_checkpoint_id=(
                    str(row["base_checkpoint_id"])
                    if row["base_checkpoint_id"] is not None
                    else None
                ),
                driver_name=str(row["driver_name"]) if row["driver_name"] is not None else None,
                provider_session_id=(
                    str(row["provider_session_id"])
                    if row["provider_session_id"] is not None
                    else None
                ),
                workspace_id=str(row["workspace_id"]) if row["workspace_id"] is not None else None,
                status=str(row["status"]) if row["status"] is not None else None,
                started_at=(
                    datetime.fromisoformat(str(row["started_at"]))
                    if row["started_at"] is not None
                    else None
                ),
                ended_at=(
                    datetime.fromisoformat(str(row["ended_at"]))
                    if row["ended_at"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def list_execution_segments(self, run_id: str) -> Sequence[ExecutionSegmentRecord]:
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_segments
                WHERE run_id = ?
                ORDER BY started_at ASC, segment_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            ExecutionSegmentRecord(
                segment_id=str(row["segment_id"]),
                run_id=str(row["run_id"]),
                session_instance_id=str(row["session_instance_id"]),
                reason=str(row["reason"]) if row["reason"] is not None else None,
                native_ref=str(row["native_ref"]) if row["native_ref"] is not None else None,
                started_at=(
                    datetime.fromisoformat(str(row["started_at"]))
                    if row["started_at"] is not None
                    else None
                ),
                finished_at=(
                    datetime.fromisoformat(str(row["finished_at"]))
                    if row["finished_at"] is not None
                    else None
                ),
                transcript_ref=_load_model(row["transcript_ref"], ArtifactRef, None),
                cost=_load_notes(cast(str | None, row["cost_json"])),
                metadata=_load_notes(cast(str | None, row["metadata_json"])),
            )
            for row in rows
        ]

    def save_artifact(
        self,
        ref: ArtifactRef,
        *,
        run_id: str,
        node_id: str | None = None,
        edge_id: str | None = None,
    ) -> None:
        """Persist a structured artifact row."""

        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO artifacts(
                  artifact_id, run_id, node_id, edge_id, artifact_kind, rel_path, sha256,
                  size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ref.artifact_id,
                    run_id,
                    node_id,
                    edge_id,
                    str(ref.kind),
                    ref.relpath,
                    ref.sha256,
                    ref.size_bytes,
                    _utc_now().isoformat(),
                ),
            )

    def list_artifacts(self, run_id: str) -> Sequence[ArtifactRef]:
        """List artifacts persisted for a run."""

        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT artifact_id, artifact_kind, rel_path, sha256, size_bytes
                FROM artifacts
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            ArtifactRef(
                artifact_id=str(row["artifact_id"]),
                kind=str(row["artifact_kind"]),
                relpath=str(row["rel_path"]),
                sha256=str(row["sha256"]),
                size_bytes=row["size_bytes"],
            )
            for row in rows
        ]

    def save_event(
        self,
        run_id: str,
        *,
        event_type: str,
        payload: Mapping[str, JSONValue],
        node_id: str | None = None,
        edge_id: str | None = None,
    ) -> str:
        """Persist a structured event emitted during the run."""

        event_id = uuid.uuid4().hex
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO events(
                  event_id, run_id, node_id, edge_id, event_type, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    run_id,
                    node_id,
                    edge_id,
                    event_type,
                    _json_dumps(payload),
                    _utc_now().isoformat(),
                ),
            )
        return event_id

    def _node_from_row(self, row: sqlite3.Row) -> NodeRecord:
        return NodeRecord(
            node_id=str(row["node_id"]),
            run_id=str(row["run_id"]),
            individual_id=(
                str(row["individual_id"])
                if row["individual_id"] is not None
                else str(row["node_id"])
            ),
            parent_node_id=(
                str(row["parent_node_id"]) if row["parent_node_id"] is not None else None
            ),
            generation=int(row["generation"]),
            workspace_path=(
                str(row["workspace_path"]) if row["workspace_path"] is not None else None
            ),
            approach_summary=(
                str(row["approach_summary"]) if row["approach_summary"] is not None else None
            ),
            approach_card_ref=_load_model(row["approach_card_ref"], ArtifactRef, None),
            materialization=_load_model(row["materialization_ref"], MaterializationRef, None),
            portable_state=_load_model(row["portable_state_ref"], PortableState, PortableState()),
            runtime_state=_load_model(row["runtime_state_ref"], RuntimeState, None),
            score=_load_model(
                row["score_ref"],
                ScoreCard,
                ScoreCard(primary_score=float(row["primary_score"]), summary="restored"),
            ),
            budget=_load_model(row["budget_json"], BudgetLedger, BudgetLedger()),
            status=_normalize_node_status(str(row["status"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            tags=_load_tags(cast(str | None, row["tags_json"])),
        )

    def _edge_from_row(self, row: sqlite3.Row) -> EdgeRecord:
        return EdgeRecord(
            edge_id=str(row["edge_id"]),
            run_id=str(row["run_id"]),
            iteration=int(row["iteration"]),
            parent_node_id=str(row["parent_node_id"]),
            child_node_id=str(row["child_node_id"]) if row["child_node_id"] is not None else None,
            edge_kind=_normalize_edge_kind(str(row["edge_kind"])),
            provider_kind=_normalize_provider_kind(str(row["provider_kind"])),
            inheritance_mode=_normalize_inheritance_mode(str(row["inheritance_mode"])),
            instruction_ref=_load_model(row["instruction_ref"], ArtifactRef, None),
            projected_state_ref=_load_model(row["projected_state_ref"], ArtifactRef, None),
            result_ref=_load_model(row["result_ref"], ArtifactRef, None),
            delta_summary=str(row["delta_summary"]) if row["delta_summary"] is not None else None,
            mutation_note_ref=_load_model(row["mutation_note_ref"], ArtifactRef, None),
            status=_normalize_edge_status(str(row["status"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    def _node_matches_filter(self, node: NodeRecord, key: str, expected: object) -> bool:
        if key == "id":
            return node.id == expected
        if key == "parent_id":
            return node.parent_id == expected
        if key == "run_id":
            return node.run_id == expected
        if key == "status":
            return node.status == expected
        if key == "generation":
            return node.generation == expected
        return node.metadata.get(key) == expected
