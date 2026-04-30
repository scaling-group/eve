"""Filesystem-backed artifact persistence."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from scaling_evolve.core.common import JSONValue
from scaling_evolve.core.enums import ArtifactKind
from scaling_evolve.core.storage.models import ArtifactRef
from scaling_evolve.storage.artifact_store import ArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore

_SNAPSHOT_KINDS = {ArtifactKind.SNAPSHOT}
_DIRECTORY_MAP: dict[ArtifactKind, str] = {
    ArtifactKind.TRANSCRIPT: "transcripts",
    ArtifactKind.SESSION_ARCHIVE_JSONL: "sessions",
    ArtifactKind.CHANGED_FILES_MANIFEST_JSON: "state",
    ArtifactKind.TRANSCRIPT_DIGEST_JSON: "state",
    ArtifactKind.PROMPT_REQUEST_JSON: "prompts",
    ArtifactKind.MODEL_RESPONSE_RAW_JSON: "responses",
    ArtifactKind.MODEL_RESPONSE_PARSED_JSON: "responses",
    ArtifactKind.EVALUATION_SUMMARY_JSON: "evaluations",
    ArtifactKind.EVALUATION_TRACEBACK_TXT: "evaluations",
    ArtifactKind.INSTRUCTION_JSON: "state",
    ArtifactKind.INSTRUCTION_TEXT: "state",
    ArtifactKind.MUTATION_RESULT_JSON: "state",
    ArtifactKind.PORTABLE_STATE_JSON: "state",
    ArtifactKind.INHERITED_CONTEXT_JSON: "state",
    ArtifactKind.PROJECTED_STATE_JSON: "state",
    ArtifactKind.SCORE_JSON: "state",
    ArtifactKind.LINEAGE_SUMMARY_JSON: "state",
    ArtifactKind.CONFIG_JSON: "state",
    ArtifactKind.APPROACH_CARD_YAML: "state",
    ArtifactKind.MUTATION_NOTE_YAML: "state",
}


class FSArtifactStore(ArtifactStore):
    """Persist artifacts under run-scoped filesystem directories."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        run_id: str,
        snapshot_root: str | Path | None = None,
        lineage_store: SQLiteLineageStore | None = None,
    ) -> None:
        self.run_id = run_id
        self.artifact_root = Path(artifact_root).resolve()
        self.snapshot_root = (
            Path(snapshot_root).resolve() if snapshot_root is not None else self.artifact_root
        )
        self.lineage_store = lineage_store
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def put_text(
        self,
        kind: ArtifactKind,
        text: str,
        *,
        filename: str | None = None,
        node_id: str | None = None,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Persist a UTF-8 text artifact."""

        return self.put_bytes(
            kind,
            text.encode("utf-8"),
            filename=filename,
            node_id=node_id,
            edge_id=edge_id,
            metadata=metadata,
        )

    def put_json(
        self,
        kind: ArtifactKind,
        payload: Mapping[str, JSONValue],
        *,
        filename: str | None = None,
        node_id: str | None = None,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Persist a formatted JSON artifact."""

        text = json.dumps(payload, indent=2, sort_keys=True)
        return self.put_text(
            kind,
            f"{text}\n",
            filename=filename,
            node_id=node_id,
            edge_id=edge_id,
            metadata=metadata,
        )

    def put_bytes(
        self,
        kind: ArtifactKind,
        payload: bytes,
        *,
        filename: str | None = None,
        node_id: str | None = None,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Persist raw bytes for binary artifacts such as snapshots."""

        artifact_id = uuid.uuid4().hex
        relpath = self._build_relative_path(kind, artifact_id, filename)
        path = self._root_for_kind(kind) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            relpath=relpath,
            uri=str(path.resolve()),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            metadata={"storage_root": self._storage_root_name(kind), **(metadata or {})},
        )
        if self.lineage_store is not None:
            self.lineage_store.save_artifact(
                ref,
                run_id=self.run_id,
                node_id=node_id,
                edge_id=edge_id,
            )
        return ref

    def read_text(self, ref: ArtifactRef) -> str:
        """Read a previously stored text artifact."""

        return self._resolve_ref_path(ref).read_text(encoding="utf-8")

    def read_json(self, ref: ArtifactRef) -> dict[str, JSONValue]:
        """Read a previously stored JSON artifact."""

        payload = json.loads(self.read_text(ref))
        if not isinstance(payload, dict):
            raise TypeError(f"Artifact {ref.artifact_id} does not contain a JSON object")
        return cast(dict[str, JSONValue], payload)

    def path_for(self, ref: ArtifactRef) -> Path:
        """Return the absolute filesystem path for a stored artifact reference."""

        return self._resolve_ref_path(ref)

    def _build_relative_path(
        self,
        kind: ArtifactKind,
        artifact_id: str,
        filename: str | None,
    ) -> str:
        directory = _DIRECTORY_MAP.get(kind, "artifacts")
        if filename is None:
            target_name = f"{artifact_id}__{str(kind)}"
        else:
            target_name = f"{artifact_id}__{Path(filename).name}"
        if self._root_is_run_scoped(self._root_for_kind(kind)):
            return str(Path(directory) / target_name)
        return str(Path(self.run_id) / directory / target_name)

    def _root_for_kind(self, kind: ArtifactKind) -> Path:
        return self.snapshot_root if kind in _SNAPSHOT_KINDS else self.artifact_root

    def _root_is_run_scoped(self, root: Path) -> bool:
        return root.name == self.run_id or root.parent.name == self.run_id

    def _resolve_ref_path(self, ref: ArtifactRef) -> Path:
        if ref.metadata.get("storage_root") == "snapshots":
            return self.snapshot_root / ref.relpath
        return self.artifact_root / ref.relpath

    def _storage_root_name(self, kind: ArtifactKind) -> str:
        return "snapshots" if kind in _SNAPSHOT_KINDS else "artifacts"

    def without_lineage_registration(self) -> FSArtifactStore:
        """Return a sibling store that writes files but skips artifact DB registration."""

        return FSArtifactStore(
            self.artifact_root,
            run_id=self.run_id,
            snapshot_root=self.snapshot_root,
            lineage_store=None,
        )
