from __future__ import annotations

import py_compile
from pathlib import Path

from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.workspace.solver_workspace import SolverWorkspaceBuilder


def _populate_editable_folder(root: Path) -> None:
    editable = root / "editable"
    editable.mkdir(parents=True)
    source = editable / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pyc_path = editable / "__pycache__" / "module.pyc"
    pyc_path.parent.mkdir(parents=True)
    py_compile.compile(str(source), cfile=str(pyc_path), doraise=True)


def test_extract_and_seed_files_preserve_binary_folder_entries(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    workspace = tmp_path / "workspace"
    _populate_editable_folder(snapshot_root)
    _populate_editable_folder(workspace / "output")
    problem = RepoTaskProblem(
        name="demo",
        github_url="https://example.com/demo",
        commit="HEAD",
        editable_files=(),
        editable_folders=("editable",),
        check_agent_paths={},
        evaluation_steps=(),
        local_checkout=tmp_path,
        snapshot_root=snapshot_root,
        boundary_checker_path=tmp_path / "boundary.py",
    )
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=OmegaConf.create({}),
        instructions={},
    )

    for files in (builder.extract(workspace), problem.seed_files()):
        assert files["editable/module.py"] == "VALUE = 1\n"
        pyc_path = next(path for path in files if path.endswith(".pyc"))
        assert files[pyc_path].startswith("__scaling_evolve_binary_v1__:")
