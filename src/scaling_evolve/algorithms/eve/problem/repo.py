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


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _application_source(raw: dict) -> tuple[str | None, str | None, str | None]:
    path_value = raw.get("path")
    github_url_value = raw.get("github_url")
    commit_value = raw.get("commit")

    has_path = path_value is not None
    has_github_url = github_url_value is not None
    has_commit = commit_value is not None
    if has_path and (has_github_url or has_commit):
        raise ValueError("application.path cannot be combined with github_url or commit.")
    if has_path:
        return str(path_value), None, None
    if not has_github_url or not has_commit:
        raise ValueError("application must configure either path or both github_url and commit.")
    return None, str(github_url_value), str(commit_value)


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
    path: str | None
    github_url: str | None
    commit: str | None
    editable_files: tuple[str, ...]
    editable_folders: tuple[str, ...]
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
        path_value, github_url, commit = _application_source(raw)
        editable_raw = raw["editable"]
        editable_files = tuple(str(path) for path in editable_raw.get("files", []))
        editable_folders = tuple(str(path).rstrip("/") for path in editable_raw.get("folders", []))

        if not editable_files and not editable_folders:
            raise ValueError("application.editable.files/folders must not both be empty")

        cache_root.mkdir(parents=True, exist_ok=True)
        if path_value is not None:
            checkout = search_root.resolve()
            snapshot_root = cls._export_path_snapshot(
                search_root=search_root,
                path=path_value,
                cache_root=cache_root,
            )
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
            path=path_value,
            github_url=github_url,
            commit=commit,
            editable_files=editable_files,
            editable_folders=editable_folders,
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

    @staticmethod
    def _export_path_snapshot(*, search_root: Path, path: str, cache_root: Path) -> Path:
        raw_path = Path(path).expanduser()
        if not str(raw_path):
            raise ValueError("application.path must include a relative task path.")

        source_root = raw_path if raw_path.is_absolute() else search_root / raw_path
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"application.path not found: {source_root}")

        digest = hashlib.sha1()
        digest.update(str(source_root).encode("utf-8"))
        digest.update(b"\0")
        for path, content in read_file_tree(source_root).items():
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content.encode("utf-8"))
            digest.update(b"\0")

        snapshot_root = cache_root / "snapshots" / f"path-{digest.hexdigest()}"
        if snapshot_root.exists():
            return snapshot_root.resolve()

        shutil.copytree(source_root, snapshot_root)
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

    def render_runtime_template(self, text: str) -> str:
        return text.replace(
            "{{BOUNDARY_CHECK_COMMAND}}",
            self.render_boundary_check_command(),
        )

    def render_boundary_check_command(self) -> str:
        command_parts = [
            "python3",
            shlex.quote(str(self.boundary_checker_path)),
            "--baseline-root",
            shlex.quote(str(self.snapshot_root)),
            "--candidate-root",
            "solver",
        ]
        for rel_path in self.editable_files:
            command_parts.extend(["--editable-file", shlex.quote(rel_path)])
        for rel_path in self.editable_folders:
            command_parts.extend(["--editable-folder", shlex.quote(rel_path)])
        return " ".join(command_parts)
