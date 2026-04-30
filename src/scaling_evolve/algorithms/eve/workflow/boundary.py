"""Deterministic boundary checking for Eve solver workspaces."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

_AUTO_GENERATED_ALLOWED_PREFIXES = (
    ".agents/skills/",
    ".claude/skills/",
)


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _file_digest_map(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root))] = _sha256_bytes(path)
    return result


def _load_gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns: list[str] = []
    for raw_line in gitignore.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _pattern_matches(path: str, pattern: str) -> bool:
    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern[1:]

    if pattern.endswith("/"):
        directory = pattern.rstrip("/")
        if anchored or "/" in directory:
            return path == directory or path.startswith(f"{directory}/")
        return (
            any(part == directory for part in Path(path).parts[:-1])
            or path == directory
            or path.startswith(f"{directory}/")
        )

    if anchored or "/" in pattern:
        return fnmatch.fnmatch(path, pattern)

    parts = Path(path).parts
    return any(fnmatch.fnmatch(part, pattern) for part in parts)


def _ignored_path_set(root: Path, paths: set[str]) -> set[str]:
    patterns = _load_gitignore_patterns(root)
    ignored: set[str] = set()
    for path in sorted(paths):
        matched = False
        for pattern in patterns:
            negated = pattern.startswith("!")
            candidate_pattern = pattern[1:] if negated else pattern
            if _pattern_matches(path, candidate_pattern):
                matched = not negated
        if matched:
            ignored.add(path)
    return ignored


@dataclass(frozen=True)
class BoundaryCheckResult:
    forbidden_created: tuple[str, ...] = ()
    forbidden_deleted: tuple[str, ...] = ()
    forbidden_modified: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.forbidden_created or self.forbidden_deleted or self.forbidden_modified)

    def summary(self) -> str:
        if self.ok:
            return "Boundary check passed."
        lines = ["Boundary check failed."]
        if self.forbidden_created:
            lines.append("Created forbidden files:")
            lines.extend(f"- {path}" for path in self.forbidden_created)
        if self.forbidden_deleted:
            lines.append("Deleted forbidden files:")
            lines.extend(f"- {path}" for path in self.forbidden_deleted)
        if self.forbidden_modified:
            lines.append("Modified forbidden files:")
            lines.extend(f"- {path}" for path in self.forbidden_modified)
        return "\n".join(lines)


def _is_editable_path(
    path: str,
    *,
    editable: dict[str, tuple[str, ...] | set[str]],
) -> bool:
    editable_files = editable["files"]
    editable_folders = editable["folders"]
    if path in editable_files:
        return True
    return any(path == folder or path.startswith(f"{folder}/") for folder in editable_folders)


def _forbidden_paths(
    *,
    paths: set[str],
    editable: dict[str, tuple[str, ...] | set[str]],
    ignored_paths: set[str],
) -> set[str]:
    return {
        path
        for path in paths
        if path != ".gitignore"
        and not any(path.startswith(prefix) for prefix in _AUTO_GENERATED_ALLOWED_PREFIXES)
        and not _is_editable_path(
            path,
            editable=editable,
        )
        and path not in ignored_paths
    }


def _protected_path_changes(
    path: str,
    *,
    baseline: dict[str, str],
    candidate: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    created: set[str] = set()
    deleted: set[str] = set()
    modified: set[str] = set()
    if path in candidate and path not in baseline:
        created.add(path)
    elif path in baseline and path not in candidate:
        deleted.add(path)
    elif path in baseline and path in candidate and baseline[path] != candidate[path]:
        modified.add(path)
    return created, deleted, modified


def check_workspace_boundary(
    *,
    baseline_root: Path,
    candidate_root: Path,
    editable: dict[str, tuple[str, ...]],
) -> BoundaryCheckResult:
    """Compare a candidate repo against the pinned baseline outside editable paths."""
    editable_paths = {
        "files": set(editable.get("files", ())),
        "folders": editable.get("folders", ()),
    }
    baseline = _file_digest_map(baseline_root)
    candidate = _file_digest_map(candidate_root)
    all_paths = set(baseline) | set(candidate)
    baseline_ignored = _ignored_path_set(baseline_root, all_paths)
    candidate_ignored = _ignored_path_set(candidate_root, all_paths)

    created_paths = candidate.keys() - baseline.keys()
    deleted_paths = baseline.keys() - candidate.keys()
    modified_paths = {
        path for path in baseline.keys() & candidate.keys() if baseline[path] != candidate[path]
    }

    forbidden_created, forbidden_deleted, forbidden_modified = _protected_path_changes(
        ".gitignore",
        baseline=baseline,
        candidate=candidate,
    )
    forbidden_created |= _forbidden_paths(
        paths=created_paths,
        editable=editable_paths,
        ignored_paths=candidate_ignored,
    )
    forbidden_deleted |= _forbidden_paths(
        paths=deleted_paths,
        editable=editable_paths,
        ignored_paths=baseline_ignored,
    )
    forbidden_modified |= _forbidden_paths(
        paths=modified_paths,
        editable=editable_paths,
        ignored_paths=baseline_ignored | candidate_ignored,
    )

    return BoundaryCheckResult(
        forbidden_created=tuple(sorted(forbidden_created)),
        forbidden_deleted=tuple(sorted(forbidden_deleted)),
        forbidden_modified=tuple(sorted(forbidden_modified)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check that only editable files changed.")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--editable-file", action="append", dest="editable_files", default=[])
    parser.add_argument("--editable-folder", action="append", dest="editable_folders", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = check_workspace_boundary(
        baseline_root=Path(args.baseline_root).resolve(),
        candidate_root=Path(args.candidate_root).resolve(),
        editable={
            "files": tuple(args.editable_files),
            "folders": tuple(args.editable_folders),
        },
    )
    print(result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
