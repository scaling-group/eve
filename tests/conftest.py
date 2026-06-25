"""Shared fake implementations for the EvE test suite."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from scaling_evolve.core.engine import (
    PortableStateRef,
    RuntimeStateRef,
)
from scaling_evolve.core.enums import ArtifactKind
from scaling_evolve.core.mutation import ProviderUsage
from scaling_evolve.providers.agent.local_workspace_manager import LocalWorkspaceManager
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore


class FakeSessionDriver:
    def __init__(self, artifact_store: FSArtifactStore, *, delay_seconds: float = 0.0) -> None:
        self.artifact_store = artifact_store
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []
        self.active_rollouts = 0
        self.max_concurrent_rollouts = 0
        self._fork_count = 0
        self._lock = threading.Lock()

    def capabilities(self) -> Any:
        from scaling_evolve.providers.agent.drivers.base import SessionDriverCapabilities

        return SessionDriverCapabilities(
            supports_native_fork=True,
            supports_cross_workspace_fork=True,
        )

    def _rollout(self, state: RuntimeStateRef, *, summary: str) -> Any:
        from scaling_evolve.providers.agent.drivers.base import SessionRollout

        with self._lock:
            self.active_rollouts += 1
            self.max_concurrent_rollouts = max(self.max_concurrent_rollouts, self.active_rollouts)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        workspace_root = Path(state.workspace_root or state.session_cwd or ".")
        workspace_root.mkdir(parents=True, exist_ok=True)
        candidate_path = workspace_root / "seed.py"
        candidate_path.write_text("print('patched')\n", encoding="utf-8")
        store = self.artifact_store.without_lineage_registration()
        try:
            transcript_ref = store.put_text(
                ArtifactKind.TRANSCRIPT,
                f"rollout:{state.state_id}\n",
                filename=f"{state.state_id}.transcript.txt",
                edge_id="edge-1",
            )
            changed_files_ref = store.put_json(
                ArtifactKind.CHANGED_FILES_MANIFEST_JSON,
                {
                    "files": [str(candidate_path)],
                    "primary_path": str(candidate_path),
                    "changed_paths": [str(candidate_path)],
                },
                filename=f"{state.state_id}.changed_files.json",
                edge_id="edge-1",
            )
            return SessionRollout(
                state=state,
                transcript=transcript_ref,
                changed_files_manifest=changed_files_ref,
                primary_path=str(candidate_path),
                changed_paths=[str(candidate_path)],
                summary=summary,
                usage=ProviderUsage(
                    input_tokens=21,
                    output_tokens=8,
                    cache_read_tokens=5,
                    cache_creation_tokens=1,
                    wallclock_seconds=4.5,
                ),
            )
        finally:
            with self._lock:
                self.active_rollouts -= 1

    def spawn(self, seed: Any) -> Any:
        self.calls.append("spawn")
        workspace = seed.workspace
        return self._rollout(
            RuntimeStateRef(
                state_id=f"runtime:{uuid4().hex}",
                provider_kind="agent_fork",
                session_id="session:spawn",
                workspace_id=workspace.workspace_id if workspace is not None else None,
                target_repo_root=workspace.target_repo_root if workspace is not None else None,
                workspace_root=workspace.workspace_root if workspace is not None else None,
                session_cwd=workspace.session_cwd if workspace is not None else None,
            ),
            summary=seed.instruction,
        )

    def fork_session(self, parent: RuntimeStateRef) -> str:
        self.calls.append("fork_session")
        self._fork_count += 1
        suffix = "" if self._fork_count == 1 else f"-{self._fork_count}"
        return f"{parent.session_id}:fork{suffix}"

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        self.calls.append("migrate_session")
        return f"{parent_cwd}->{child_cwd}:{session_id}"

    def fork(self, parent: RuntimeStateRef, instruction: str) -> Any:
        self.calls.append("resume")
        return self._rollout(
            parent.model_copy(
                update={
                    "state_id": "runtime:fork",
                    "session_id": "session:spawn:fork",
                }
            ),
            summary=instruction,
        )

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> Any:
        self.calls.append("resume")
        return self._rollout(state, summary=instruction or "session rollout")

    def snapshot(self, state: RuntimeStateRef) -> Any:
        from scaling_evolve.providers.agent.drivers.base import SessionSnapshot

        self.calls.append("snapshot")
        return SessionSnapshot(
            state=state,
            portable_state=PortableStateRef(
                state_id=f"portable:{state.state_id}",
                summary="snapshot",
            ),
        )


class StrictWorkspaceSessionDriver(FakeSessionDriver):
    def fork_session(self, parent: RuntimeStateRef) -> str:
        workspace_root = Path(parent.workspace_root or parent.session_cwd or ".")
        if not workspace_root.exists():
            raise AssertionError(f"expected retained workspace to exist: {workspace_root}")
        return super().fork_session(parent)


@pytest.fixture
def build_session_stack(tmp_path: Path):
    def _build(
        run_id: str = "run-1",
    ) -> tuple[SQLiteLineageStore, FSArtifactStore, LocalWorkspaceManager]:
        lineage_store = SQLiteLineageStore(tmp_path / f"{run_id}.sqlite3")
        artifact_store = FSArtifactStore(
            tmp_path / f"{run_id}-artifacts",
            snapshot_root=tmp_path / f"{run_id}-snapshots",
            run_id=run_id,
            lineage_store=lineage_store,
        )
        workspace_manager = LocalWorkspaceManager(tmp_path / f"{run_id}-workspaces")
        return lineage_store, artifact_store, workspace_manager

    return _build


@pytest.fixture
def fake_session_driver_factory():
    def _build(
        artifact_store: FSArtifactStore,
        *,
        delay_seconds: float = 0.0,
        strict_workspace: bool = False,
    ) -> FakeSessionDriver:
        driver_cls = StrictWorkspaceSessionDriver if strict_workspace else FakeSessionDriver
        return driver_cls(artifact_store, delay_seconds=delay_seconds)

    return _build
