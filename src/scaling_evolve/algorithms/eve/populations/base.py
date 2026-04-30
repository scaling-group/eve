"""Eve population storage backed by artifact refs plus opaque score YAML refs."""

from __future__ import annotations

import json
import random
import sqlite3

import yaml
from omegaconf import DictConfig
from optree import PyTree

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.core.common import utc_now
from scaling_evolve.core.storage.models import ArtifactKind, ArtifactRef
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.lineage_store import RunRecord
from scaling_evolve.storage.sqlite import SQLiteLineageStore

_ENTRY_TABLE = "eve_population_entries"


class Population:
    """Generic Eve population with opaque score storage."""

    def __init__(
        self,
        lineage_store: SQLiteLineageStore,
        artifact_store: FSArtifactStore,
        run_id: str,
        app_kind: str,
        config: DictConfig,
        rng: random.Random | None = None,
    ) -> None:
        self._store = lineage_store
        self._artifacts = artifact_store.without_lineage_registration()
        self._run_id = run_id
        self._app_kind = app_kind
        self._config = config
        self._rng = rng or random.Random()

        self._store.create_run(RunRecord(run_id=run_id, app_kind=app_kind))
        self._initialize_entry_table()

    def add(self, entry: PopulationEntry) -> None:
        files_ref = self._artifacts.put_text(
            ArtifactKind.MUTATION_RESULT_JSON,
            json.dumps(entry.files, indent=2, sort_keys=True),
            filename=f"{entry.id}.json",
        )
        logs_ref = self._artifacts.put_text(
            ArtifactKind.TRANSCRIPT,
            json.dumps(entry.logs, indent=2, sort_keys=True),
            filename=f"{entry.id}_logs.json",
        )
        score_ref = self._artifacts.put_text(
            ArtifactKind.SCORE_JSON,
            yaml.safe_dump(entry.score, sort_keys=False),
            filename=f"{entry.id}_score.yaml",
        )
        with self._store.connection:
            self._store.connection.execute(
                f"""
                INSERT INTO {_ENTRY_TABLE}(
                  run_id, app_kind, entry_id, files_ref_json, logs_ref_json,
                  score_ref_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, entry_id) DO UPDATE SET
                  app_kind = excluded.app_kind,
                  files_ref_json = excluded.files_ref_json,
                  logs_ref_json = excluded.logs_ref_json,
                  score_ref_json = excluded.score_ref_json,
                  created_at = excluded.created_at
                """,
                (
                    self._run_id,
                    self._app_kind,
                    entry.id,
                    files_ref.model_dump_json(),
                    logs_ref.model_dump_json(),
                    score_ref.model_dump_json(),
                    utc_now().isoformat(),
                ),
            )

    def entries(self) -> list[PopulationEntry]:
        """Return all population entries in stable stored order."""
        return self._list_entries()

    def size(self) -> int:
        with self._store.connection:
            row = self._store.connection.execute(
                f"SELECT COUNT(*) AS count FROM {_ENTRY_TABLE} WHERE run_id = ?",
                (self._run_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def update_scores(self, updated_scores: dict[str, PyTree]) -> None:
        entries_by_id = {entry.id: entry for entry in self.entries()}
        for entry_id, score in updated_scores.items():
            entry = entries_by_id.get(entry_id)
            if entry is None:
                continue
            self.add(PopulationEntry(id=entry.id, files=entry.files, score=score, logs=entry.logs))

    def update_logs(self, new_logs: dict[str, dict[str, str]]) -> None:
        entries_by_id = {entry.id: entry for entry in self.entries()}
        for entry_id, extra_logs in new_logs.items():
            entry = entries_by_id.get(entry_id)
            if entry is None:
                continue
            merged_logs = dict(entry.logs)
            merged_logs.update(extra_logs)
            self.add(
                PopulationEntry(
                    id=entry.id,
                    files=entry.files,
                    score=entry.score,
                    logs=merged_logs,
                )
            )

    def _initialize_entry_table(self) -> None:
        with self._store.connection:
            self._store.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_ENTRY_TABLE} (
                  run_id TEXT NOT NULL,
                  app_kind TEXT NOT NULL,
                  entry_id TEXT NOT NULL,
                  files_ref_json TEXT NOT NULL,
                  logs_ref_json TEXT NOT NULL,
                  score_ref_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (run_id, entry_id)
                )
                """
            )

    def _list_entries(self) -> list[PopulationEntry]:
        with self._store.connection:
            rows = self._store.connection.execute(
                f"""
                SELECT entry_id, files_ref_json, logs_ref_json, score_ref_json
                FROM {_ENTRY_TABLE}
                WHERE run_id = ?
                ORDER BY created_at ASC, entry_id ASC
                """,
                (self._run_id,),
            ).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def _entry_from_row(self, row: sqlite3.Row) -> PopulationEntry:
        files_ref = ArtifactRef.model_validate_json(str(row["files_ref_json"]))
        logs_ref = ArtifactRef.model_validate_json(str(row["logs_ref_json"]))
        score_ref = ArtifactRef.model_validate_json(str(row["score_ref_json"]))
        files_payload = json.loads(self._artifacts.read_text(files_ref))
        logs_payload = json.loads(self._artifacts.read_text(logs_ref))
        score_payload = yaml.safe_load(self._artifacts.read_text(score_ref))
        if not isinstance(files_payload, dict):
            raise TypeError(f"Expected file tree object for {row['entry_id']}")
        if not isinstance(logs_payload, dict):
            raise TypeError(f"Expected log tree object for {row['entry_id']}")
        return PopulationEntry(
            id=str(row["entry_id"]),
            files={str(path): str(content) for path, content in files_payload.items()},
            score=score_payload,
            logs={str(path): str(content) for path, content in logs_payload.items()},
        )
