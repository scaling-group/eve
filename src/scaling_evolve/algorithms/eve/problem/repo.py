"""Repo-backed task problem helpers for Eve."""

from __future__ import annotations

import hashlib
import io
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from scaling_evolve.algorithms.eve.workspace.file_tree import read_file_tree


def _normalize_git_url(url: str) -> str:
    normalized = url.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.rstrip("/")


def _repo_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if name.endswith(".git"):
        name = name[:-4]
    return name or "repo"


def _repo_root_for_boundary_checker(path: Path) -> Path:
    for candidate in path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    if len(path.parents) > 5:
        return path.parents[5]
    return path.parent


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _extract_tar_bytes(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination.resolve())):
                raise ValueError(f"Refusing to extract unsafe archive path: {member.name}")
        archive.extractall(destination)


@dataclass(frozen=True)
class RepoTaskProblem:
    """Immutable repo-backed task problem definition."""

    name: str
    github_url: str | None
    commit: str | None
    editable_files: tuple[str, ...]
    editable_folders: tuple[str, ...]
    check_agent_paths: dict[str, Path]
    evaluation_steps: tuple[Path, ...]
    local_checkout: Path
    snapshot_root: Path
    boundary_checker_path: Path

    @property
    def repo_name(self) -> str:
        if self.github_url is None:
            return self.name
        return _repo_name_from_url(self.github_url)

    @property
    def slug(self) -> str:
        return self.name

    @classmethod
    def from_config(
        cls,
        raw: dict,
        *,
        cache_root: Path,
        search_root: Path,
    ) -> RepoTaskProblem:
        name = str(raw["name"])
        source_root_raw = raw.get("source_root")
        if source_root_raw is None:
            if "github_url" not in raw or "commit" not in raw:
                raise ValueError(
                    "application.github_url and application.commit are required "
                    "when application.source_root is not set"
                )
            github_url = str(raw["github_url"])
            commit = str(raw["commit"])
        else:
            github_url = str(raw["github_url"]) if raw.get("github_url") is not None else None
            commit = str(raw["commit"]) if raw.get("commit") is not None else None
        editable_raw = raw["editable"]
        editable_files = tuple(str(path) for path in editable_raw.get("files", []))
        editable_folders = tuple(str(path).rstrip("/") for path in editable_raw.get("folders", []))
        check_agent_raw = raw["check_agent"]
        check_agent_paths = {
            "claude": search_root / str(check_agent_raw["claude"]),
            "codex": search_root / str(check_agent_raw["codex"]),
        }
        raw_evaluation_steps = raw["evaluation_steps"]
        evaluation_steps = tuple(search_root / str(path) for path in raw_evaluation_steps)

        if not editable_files and not editable_folders:
            raise ValueError("application.editable.files/folders must not both be empty")

        cache_root.mkdir(parents=True, exist_ok=True)
        if source_root_raw is not None:
            source_root = search_root / str(source_root_raw)
            checkout = cls._resolve_local_source(source_root, cache_root=cache_root, name=name)
            snapshot_root = checkout
        else:
            assert github_url is not None
            assert commit is not None
            checkout = cls._resolve_checkout(
                github_url, cache_root=cache_root, search_root=search_root
            )
            snapshot_root = cls._export_snapshot(
                checkout=checkout,
                commit=commit,
                cache_root=cache_root,
            )
        boundary_checker_path = (
            search_root / "src/scaling_evolve/algorithms/eve/workflow/boundary.py"
        )
        return cls(
            name=name,
            github_url=github_url,
            commit=commit,
            editable_files=editable_files,
            editable_folders=editable_folders,
            check_agent_paths=check_agent_paths,
            evaluation_steps=evaluation_steps,
            local_checkout=checkout,
            snapshot_root=snapshot_root,
            boundary_checker_path=boundary_checker_path,
        )

    @staticmethod
    def _resolve_checkout(github_url: str, *, cache_root: Path, search_root: Path) -> Path:
        repo_name = _repo_name_from_url(github_url)
        normalized_target = _normalize_git_url(github_url)
        checkout_hash = hashlib.sha1(normalized_target.encode("utf-8")).hexdigest()[:12]
        checkout_name = f"{repo_name}-{checkout_hash}"
        cache_checkout = cache_root / "checkouts" / checkout_name
        if cache_checkout.exists():
            shutil.rmtree(cache_checkout)

        cache_checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", github_url, str(cache_checkout)],
            capture_output=True,
            text=True,
            check=True,
        )
        return cache_checkout.resolve()

    @staticmethod
    def _resolve_local_source(source_root: Path, *, cache_root: Path, name: str) -> Path:
        source_root = source_root.resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"application.source_root does not exist: {source_root}")
        source_hash = hashlib.sha1(str(source_root).encode("utf-8")).hexdigest()[:12]
        checkout_name = f"{name}-{source_hash}"
        cache_checkout = cache_root / "local_sources" / checkout_name
        if cache_checkout.exists():
            shutil.rmtree(cache_checkout)
        cache_checkout.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, cache_checkout)
        return cache_checkout.resolve()

    @staticmethod
    def _export_snapshot(*, checkout: Path, commit: str, cache_root: Path) -> Path:
        resolved_commit = _run_git(["rev-parse", commit], cwd=checkout)
        snapshot_root = cache_root / "snapshots" / resolved_commit
        if snapshot_root.exists():
            return snapshot_root.resolve()

        archive_bytes = subprocess.run(
            ["git", "-C", str(checkout), "archive", "--format=tar", resolved_commit],
            capture_output=True,
            check=True,
        ).stdout
        _extract_tar_bytes(archive_bytes, snapshot_root)
        return snapshot_root.resolve()

    def seed_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for rel_path in self.editable_files:
            path = self.snapshot_root / rel_path
            files[rel_path] = path.read_text(encoding="utf-8")
        for folder in self.editable_folders:
            folder_root = self.snapshot_root / folder
            if not folder_root.exists():
                continue
            for rel_path, content in read_file_tree(folder_root).items():
                files[str(Path(folder) / rel_path)] = content
        return files

    def copy_base_repo(self, destination: Path) -> None:
        shutil.copytree(self.snapshot_root, destination, dirs_exist_ok=True)

    def render_check_agent_definition(self, path: Path) -> str:
        return path.read_text(encoding="utf-8").replace(
            "{{BOUNDARY_CHECK_COMMAND}}",
            self.render_boundary_check_command(),
        )

    def render_boundary_check_command(self) -> str:
        repo_root = _repo_root_for_boundary_checker(self.boundary_checker_path)
        venv_bin = repo_root / ".venv" / "bin"
        command_parts = [
            f"PATH={shlex.quote(str(venv_bin))}:$PATH python3",
            shlex.quote(str(self.boundary_checker_path)),
            "--baseline-root",
            shlex.quote(str(self.snapshot_root)),
            "--candidate-root",
            ".",
        ]
        for rel_path in self.editable_files:
            command_parts.extend(["--editable-file", shlex.quote(rel_path)])
        for rel_path in self.editable_folders:
            command_parts.extend(["--editable-folder", shlex.quote(rel_path)])
        return " ".join(command_parts)
