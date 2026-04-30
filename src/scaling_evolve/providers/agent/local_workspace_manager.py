"""Local filesystem workspace lease manager."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from uuid import uuid4

from scaling_evolve.core.common import utc_now
from scaling_evolve.core.engine import WorkspaceLease
from scaling_evolve.providers.agent.workspaces import (
    WorkspaceLeaseManager,
    WorkspaceLeaseRequest,
    WorkspacePlan,
    WorkspacePressure,
)


class LocalWorkspaceManager(WorkspaceLeaseManager):
    """Strategy-aware local workspace manager used by offline tests and CLI wiring."""

    def __init__(
        self,
        root: str | Path,
        *,
        soft_limit_bytes: int | None = None,
        hard_limit_bytes: int | None = None,
        max_full_workspaces: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.soft_limit_bytes = soft_limit_bytes
        self.hard_limit_bytes = hard_limit_bytes
        self.max_full_workspaces = max_full_workspaces
        self._leases: dict[str, WorkspaceLease] = {}
        self._git_worktrees: dict[str, Path] = {}
        self._pending_full_workspaces = 0
        self._lock = threading.RLock()

    def acquire(
        self,
        request: WorkspaceLeaseRequest | str | None = None,
        owner_node_id: str | None = None,
    ) -> WorkspaceLease:
        if isinstance(request, WorkspaceLeaseRequest):
            owner = request.owner_node_id or request.node_id
            purpose = request.purpose
            target_repo_root = request.target_repo_root
            base_snapshot_id = request.base_snapshot_id
            plan = request.plan or WorkspacePlan(strategy="full_workspace", purpose=purpose)
            artifact_paths = list(request.artifact_paths)
        else:
            owner = request if isinstance(request, str) else owner_node_id
            purpose = "mutation"
            target_repo_root = None
            base_snapshot_id = None
            plan = WorkspacePlan(strategy="full_workspace", purpose=purpose)
            artifact_paths = []

        workspace_id = (
            request.node_id
            if isinstance(request, WorkspaceLeaseRequest) and request.node_id
            else (owner or f"workspace-{uuid4().hex[:12]}")
        )
        workspace_root = self.root / workspace_id
        reserved_full_workspace = False
        if plan.strategy == "full_workspace":
            with self._lock:
                active_full_workspaces = sum(
                    1 for lease in self._leases.values() if lease.strategy == "full_workspace"
                )
                if self.max_full_workspaces is not None and (
                    active_full_workspaces + self._pending_full_workspaces
                    >= self.max_full_workspaces
                ):
                    raise RuntimeError("max_full_workspaces limit reached")
                self._pending_full_workspaces += 1
                reserved_full_workspace = True
        try:
            self._materialize_workspace(
                workspace_root,
                target_repo_root=target_repo_root,
                plan=plan,
                artifact_paths=artifact_paths,
                workspace_id=workspace_id,
            )
        except Exception:
            if reserved_full_workspace:
                with self._lock:
                    self._pending_full_workspaces -= 1
            raise
        now = utc_now()
        with self._lock:
            used_git_worktree = workspace_id in self._git_worktrees
        lease = WorkspaceLease(
            workspace_id=workspace_id,
            root=str(workspace_root),
            owner_node_id=owner,
            strategy=plan.strategy,
            purpose=purpose,
            target_repo_root=target_repo_root,
            workspace_root=str(workspace_root),
            session_cwd=str(workspace_root),
            created_at=now,
            base_snapshot_id=base_snapshot_id,
            used_bytes=self._workspace_size_from_path(workspace_root),
            used_git_worktree=used_git_worktree,
        )
        with self._lock:
            if reserved_full_workspace:
                self._pending_full_workspaces -= 1
            self._leases[workspace_id] = lease
        return lease

    def get_lease(self, workspace_id: str) -> WorkspaceLease | None:
        """Return a currently tracked lease, if any."""

        with self._lock:
            return self._leases.get(workspace_id)

    def release(self, lease: WorkspaceLease, *, snapshot: bool = False) -> None:
        workspace_root = Path(lease.workspace_root or lease.root)
        with self._lock:
            self._leases.pop(lease.workspace_id, None)
            repo_root = self._git_worktrees.pop(lease.workspace_id, None)
        if snapshot:
            return
        if repo_root is not None and workspace_root.exists():
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(workspace_root)],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        if workspace_root.exists():
            shutil.rmtree(workspace_root, ignore_errors=True)

    def fork_workspace(self, source_root: str | Path, lease: WorkspaceLease) -> None:
        """Replace a leased workspace with a copy of an existing workspace tree."""

        with self._lock:
            source_path = Path(source_root).expanduser().resolve()
            workspace_root = Path(lease.workspace_root or lease.root).expanduser().resolve()
            if source_path == workspace_root:
                return
            if workspace_root.exists():
                shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.copytree(source_path, workspace_root)
            self._leases[lease.workspace_id] = lease.model_copy(
                update={"used_bytes": self._workspace_size_from_path(workspace_root)}
            )

    def pressure(self) -> WorkspacePressure:
        with self._lock:
            leases = tuple(self._leases.values())
        return WorkspacePressure(
            used_bytes=sum(self._workspace_size(lease) for lease in leases),
            soft_limit_bytes=self.soft_limit_bytes,
            hard_limit_bytes=self.hard_limit_bytes,
            active_full_workspaces=sum(1 for lease in leases if lease.strategy == "full_workspace"),
        )

    def _workspace_size(self, lease: WorkspaceLease) -> int:
        workspace_root = Path(lease.workspace_root or lease.root)
        return self._workspace_size_from_path(workspace_root)

    def _workspace_size_from_path(self, workspace_root: Path) -> int:
        if not workspace_root.exists():
            return 0
        size = 0
        for path in workspace_root.rglob("*"):
            if path.is_file():
                size += path.stat().st_size
        return size

    def _materialize_workspace(
        self,
        workspace_root: Path,
        *,
        target_repo_root: str | None,
        plan: WorkspacePlan,
        artifact_paths: list[str],
        workspace_id: str,
    ) -> None:
        if plan.strategy == "artifact_only" or target_repo_root is None:
            workspace_root.mkdir(parents=True, exist_ok=False)
            self._materialize_artifact_only(
                workspace_root,
                artifact_paths=artifact_paths,
                target_repo_root=target_repo_root,
            )
            return

        repo_root = Path(target_repo_root)
        if plan.strategy == "full_workspace":
            if plan.needs_git_semantics and self._is_git_repo(repo_root):
                self._create_git_worktree(repo_root, workspace_root)
                with self._lock:
                    self._git_worktrees[workspace_id] = repo_root
                return
            shutil.copytree(repo_root, workspace_root)
            return

        raise ValueError(f"Unsupported workspace strategy: {plan.strategy}")

    def _materialize_artifact_only(
        self,
        workspace_root: Path,
        *,
        artifact_paths: list[str],
        target_repo_root: str | None,
    ) -> None:
        repo_root = Path(target_repo_root) if target_repo_root is not None else None
        for item in artifact_paths:
            source = (
                self._resolve_source_path(repo_root, item) if repo_root is not None else Path(item)
            )
            if not source.exists():
                continue
            destination = workspace_root / self._relative_destination(source, repo_root)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    def _create_git_worktree(self, repo_root: Path, workspace_root: Path) -> None:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(workspace_root)],
            check=True,
            capture_output=True,
            text=True,
        )

    def _is_git_repo(self, repo_root: Path) -> bool:
        if (repo_root / ".git").exists():
            return True
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def _resolve_source_path(self, repo_root: Path, root: str) -> Path:
        candidate = Path(root)
        if candidate.is_absolute():
            return candidate
        return repo_root / candidate

    def _relative_destination(self, source: Path, repo_root: Path | None) -> Path:
        if repo_root is not None:
            try:
                return source.relative_to(repo_root)
            except ValueError:
                pass
        return Path(source.name)
