from __future__ import annotations

from pathlib import Path

import pytest

from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem


def _local_application_config(source_root: str) -> dict[str, object]:
    return {
        "name": "circle-packing",
        "source_root": source_root,
        "editable": {"files": ["candidate.py"], "folders": []},
        "check_agent": {
            "claude": "configs/eve/application/circle_packing/check_claude.md",
            "codex": "configs/eve/application/circle_packing/check_codex.toml",
        },
        "evaluation_steps": ["configs/eve/application/circle_packing/evaluation.sh"],
    }


def test_source_root_application_does_not_require_github_url_or_commit(
    tmp_path: Path,
) -> None:
    search_root = tmp_path / "repo"
    source_root = search_root / "examples" / "circle_packing" / "repo"
    source_root.mkdir(parents=True)
    (source_root / "candidate.py").write_text("print('seed')\n", encoding="utf-8")

    problem = RepoTaskProblem.from_config(
        _local_application_config("examples/circle_packing/repo"),
        cache_root=tmp_path / "cache",
        search_root=search_root,
    )

    assert problem.github_url is None
    assert problem.commit is None
    assert problem.repo_name == "circle-packing"
    assert problem.seed_files() == {"candidate.py": "print('seed')\n"}
    assert problem.local_checkout != source_root
    assert problem.snapshot_root == problem.local_checkout


def test_external_repo_application_requires_github_url_and_commit(tmp_path: Path) -> None:
    raw = _local_application_config("examples/circle_packing/repo")
    raw.pop("source_root")

    with pytest.raises(ValueError, match="github_url and application.commit"):
        RepoTaskProblem.from_config(
            raw,
            cache_root=tmp_path / "cache",
            search_root=tmp_path,
        )


def test_boundary_check_command_handles_short_tmp_paths(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    problem = RepoTaskProblem(
        name="local-app",
        github_url=None,
        commit=None,
        editable_files=("candidate.py",),
        editable_folders=(),
        check_agent_paths={},
        evaluation_steps=(),
        local_checkout=snapshot_root,
        snapshot_root=snapshot_root,
        boundary_checker_path=Path("/tmp/boundary.py"),
    )

    command = problem.render_boundary_check_command()

    assert "PATH=/tmp/.venv/bin:$PATH python3" in command
    assert "--baseline-root" in command
    assert "--editable-file candidate.py" in command


def test_boundary_check_command_prefers_project_root_marker(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    boundary_checker = (
        repo_root / "src" / "scaling_evolve" / "algorithms" / "eve" / "workflow" / "boundary.py"
    )
    boundary_checker.parent.mkdir(parents=True)
    boundary_checker.write_text("# boundary\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'eve'\n", encoding="utf-8")
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    problem = RepoTaskProblem(
        name="local-app",
        github_url=None,
        commit=None,
        editable_files=("candidate.py",),
        editable_folders=(),
        check_agent_paths={},
        evaluation_steps=(),
        local_checkout=snapshot_root,
        snapshot_root=snapshot_root,
        boundary_checker_path=boundary_checker,
    )

    command = problem.render_boundary_check_command()

    assert f"PATH={repo_root / '.venv' / 'bin'}:$PATH python3" in command
