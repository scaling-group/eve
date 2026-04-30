"""Filesystem-backed snapshot store."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from scaling_evolve.core.common import JSONValue
from scaling_evolve.core.engine import PortableState, PortableStateRef, RuntimeStateRef
from scaling_evolve.storage.snapshot_store import ReadableSnapshotStore


class FSSnapshotStore(ReadableSnapshotStore):
    """Persist portable/runtime snapshot metadata under a local filesystem root."""

    def __init__(self, root: str | Path, *, run_id: str | None = None) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self._scoped_root.mkdir(parents=True, exist_ok=True)

    def save_runtime_state(
        self,
        ref: RuntimeStateRef,
        payload: Mapping[str, JSONValue],
    ) -> None:
        self._write_state("runtime", ref.state_id, payload)

    def save_portable_state(
        self,
        ref: PortableStateRef,
        payload: Mapping[str, JSONValue],
    ) -> None:
        self._write_state("portable", ref.state_id, payload)

    def load_portable_state(self, ref: PortableStateRef) -> Mapping[str, JSONValue]:
        return self._load_state("portable", ref.state_id)

    def load_runtime_state(self, ref: RuntimeStateRef) -> Mapping[str, JSONValue]:
        return self._load_state("runtime", ref.state_id)

    def load_manifest(self, manifest_ref: str) -> Mapping[str, JSONValue] | list[JSONValue]:
        """Read a manifest payload referenced by stored metadata."""

        return self._load_json_path(manifest_ref)

    def load_delta(self, delta_ref: str) -> Mapping[str, JSONValue] | list[JSONValue]:
        """Read a delta payload referenced by stored metadata."""

        return self._load_json_path(delta_ref)

    def reconstruct_portable_state(self, ref: PortableStateRef) -> Mapping[str, JSONValue]:
        """Reinflate out-of-line portable manifests back into one payload."""

        payload = dict(self.load_portable_state(ref))
        manifest_ref = payload.get("manifest_ref")
        if isinstance(manifest_ref, str):
            payload["manifest"] = self.load_manifest(manifest_ref)
        delta_ref = payload.get("delta_ref")
        if isinstance(delta_ref, str):
            payload["delta"] = self.load_delta(delta_ref)
        changed_files_ref = payload.get("changed_files_manifest_ref")
        if isinstance(changed_files_ref, str):
            payload["changed_files_manifest"] = self.load_manifest(changed_files_ref)
        return payload

    def load_portable_state_model(self, ref: PortableStateRef) -> PortableState:
        """Rebuild a portable state model from stored snapshot metadata."""

        payload = dict(self.load_portable_state(ref))
        inheritance_metadata = payload.get("inheritance_metadata")
        if not isinstance(inheritance_metadata, dict):
            inheritance_metadata = {}
            payload["inheritance_metadata"] = inheritance_metadata
        manifest_ref = payload.get("manifest_ref")
        if isinstance(manifest_ref, str):
            cast(dict[str, JSONValue], inheritance_metadata)["manifest_ref"] = manifest_ref
        delta_ref = payload.get("delta_ref")
        if isinstance(delta_ref, str):
            cast(dict[str, JSONValue], inheritance_metadata)["delta_ref"] = delta_ref
        payload.setdefault("state_id", ref.state_id)
        return PortableState.model_validate(payload)

    def _write_state(
        self,
        namespace: str,
        state_id: str,
        payload: Mapping[str, JSONValue],
    ) -> None:
        metadata = dict(payload)
        manifest = metadata.pop("manifest", None)
        delta = metadata.pop("delta", None)
        deltas = metadata.pop("deltas", None)
        changed_files = metadata.pop("changed_files_manifest", None)
        if isinstance(manifest, (dict, list)):
            metadata["manifest_ref"] = self._write_blob(
                namespace,
                state_id,
                suffix="manifest",
                payload=cast(Mapping[str, JSONValue] | list[JSONValue], manifest),
            )
        if isinstance(delta, (dict, list)):
            metadata["delta_ref"] = self._write_blob(
                namespace,
                state_id,
                suffix="delta",
                payload=cast(Mapping[str, JSONValue] | list[JSONValue], delta),
            )
        if isinstance(deltas, (dict, list)):
            metadata["delta_ref"] = self._write_blob(
                namespace,
                state_id,
                suffix="delta",
                payload=cast(Mapping[str, JSONValue] | list[JSONValue], deltas),
            )
        if isinstance(changed_files, (dict, list)):
            metadata["changed_files_manifest_ref"] = self._write_blob(
                namespace,
                state_id,
                suffix="changed_files",
                payload=cast(Mapping[str, JSONValue] | list[JSONValue], changed_files),
            )
        path = self._state_path(namespace, state_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8")

    def _load_state(self, namespace: str, state_id: str) -> Mapping[str, JSONValue]:
        payload = json.loads(self._state_path(namespace, state_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Snapshot {namespace}/{state_id} did not decode to a JSON object")
        return cast(Mapping[str, JSONValue], payload)

    def _load_json_path(self, relpath: str) -> Mapping[str, JSONValue] | list[JSONValue]:
        payload = json.loads((self._scoped_root / relpath).read_text(encoding="utf-8"))
        if not isinstance(payload, (dict, list)):
            raise TypeError(f"Snapshot payload {relpath} did not decode to a JSON object or array")
        return cast(Mapping[str, JSONValue] | list[JSONValue], payload)

    def _write_blob(
        self,
        namespace: str,
        state_id: str,
        *,
        suffix: str,
        payload: Mapping[str, JSONValue] | list[JSONValue],
    ) -> str:
        relpath = f"manifests/{namespace}/{state_id}.{suffix}.json"
        path = self._scoped_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
        return relpath

    def _state_path(self, namespace: str, state_id: str) -> Path:
        return self._scoped_root / namespace / f"{state_id}.json"

    @property
    def _scoped_root(self) -> Path:
        if (
            self.run_id is None
            or self.root.name == self.run_id
            or self.root.parent.name == self.run_id
        ):
            return self.root
        return self.root / self.run_id
