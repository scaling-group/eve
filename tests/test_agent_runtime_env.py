from __future__ import annotations

import os
from pathlib import Path

from scaling_evolve.providers.agent import runtime_env


def test_prepend_agent_runtime_bins_uses_workspace_then_repo_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_runtime_bin = repo_root / ".agent-runtime" / "bin"
    repo_runtime_bin.mkdir(parents=True)
    workspace_root = tmp_path / "workspace"
    workspace_runtime_bin = tmp_path / "workspace" / ".agent-runtime" / "bin"
    workspace_runtime_bin.mkdir(parents=True)
    monkeypatch.setattr(runtime_env, "REPO_ROOT", repo_root)

    env = runtime_env.prepend_agent_runtime_bins(
        {"PATH": f"{repo_runtime_bin}{os.pathsep}/usr/bin{os.pathsep}/usr/bin"},
        workspace_root=workspace_root,
    )

    assert env["PATH"].split(os.pathsep) == [
        str(workspace_runtime_bin),
        str(repo_runtime_bin),
        "/usr/bin",
    ]


def test_prepend_agent_runtime_bins_falls_back_to_repo_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_runtime_bin = repo_root / ".agent-runtime" / "bin"
    repo_runtime_bin.mkdir(parents=True)
    monkeypatch.setattr(runtime_env, "REPO_ROOT", repo_root)

    env = runtime_env.prepend_agent_runtime_bins(
        {"PATH": "/usr/bin"},
        workspace_root=tmp_path / "workspace",
    )

    assert env["PATH"].split(os.pathsep) == [str(repo_runtime_bin), "/usr/bin"]


def test_prepend_agent_runtime_bins_is_noop_when_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_env, "REPO_ROOT", tmp_path / "repo")

    env = runtime_env.prepend_agent_runtime_bins(
        {"PATH": "/usr/bin"},
        workspace_root=tmp_path / "workspace",
    )

    assert env == {"PATH": "/usr/bin"}
