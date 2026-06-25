from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.workspace.file_tree import expose_guidance_agents
from scaling_evolve.algorithms.eve.workspace.solver_workspace import SolverWorkspaceBuilder


def _make_problem(tmp_path: Path) -> RepoTaskProblem:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "candidate.py").write_text("print('seed')\n", encoding="utf-8")
    return RepoTaskProblem(
        name="test-problem",
        path=None,
        github_url="https://github.com/example/eve-problem",
        commit="abc123",
        editable_files=("candidate.py",),
        editable_folders=(),
        local_checkout=snapshot,
        snapshot_root=snapshot,
        boundary_checker_path=tmp_path / "boundary.py",
    )


def _make_config(tmp_path: Path):
    return OmegaConf.create(
        {
            "workspace_root": str(tmp_path / "run"),
            "n_optimizer_examples_phase2": 0,
        }
    )


def test_expose_guidance_agents_links_codex_and_claude_agents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    codex_source = workspace / "guidance" / "agents" / "codex"
    claude_source = workspace / "guidance" / "agents" / "claude"
    codex_source.mkdir(parents=True)
    claude_source.mkdir(parents=True)
    (codex_source / "planner.toml").write_text('name = "planner"\n', encoding="utf-8")
    (claude_source / "planner.md").write_text("# Planner\n", encoding="utf-8")

    expose_guidance_agents(workspace)

    assert (workspace / ".codex" / "agents").is_symlink()
    assert (workspace / ".claude" / "agents").is_symlink()
    codex_agent = workspace / ".codex" / "agents" / "planner.toml"
    claude_agent = workspace / ".claude" / "agents" / "planner.md"
    assert codex_agent.read_text(encoding="utf-8") == 'name = "planner"\n'
    assert claude_agent.read_text(encoding="utf-8") == "# Planner\n"

    (codex_source / "planner.toml").write_text('name = "planner-v2"\n', encoding="utf-8")
    (claude_source / "planner.md").write_text("# Planner v2\n", encoding="utf-8")

    assert codex_agent.read_text(encoding="utf-8") == 'name = "planner-v2"\n'
    assert claude_agent.read_text(encoding="utf-8") == "# Planner v2\n"

    (codex_source / "new-agent.toml").write_text('name = "new-agent"\n', encoding="utf-8")
    (claude_source / "new-agent.md").write_text("# New Agent\n", encoding="utf-8")

    assert (workspace / ".codex" / "agents" / "new-agent.toml").read_text(
        encoding="utf-8"
    ) == 'name = "new-agent"\n'
    assert (workspace / ".claude" / "agents" / "new-agent.md").read_text(
        encoding="utf-8"
    ) == "# New Agent\n"


def test_expose_guidance_agents_links_whole_provider_directories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    codex_source = workspace / "guidance" / "agents" / "codex"
    claude_source = workspace / "guidance" / "agents" / "claude"
    (codex_source / "nested").mkdir(parents=True)
    claude_source.mkdir(parents=True)
    (codex_source / "planner.txt").write_text("ignore\n", encoding="utf-8")
    (codex_source / "nested" / "nested.toml").write_text("ignore\n", encoding="utf-8")
    (claude_source / "planner.toml").write_text("ignore\n", encoding="utf-8")

    expose_guidance_agents(workspace)

    assert (workspace / ".codex" / "agents" / "planner.txt").read_text(
        encoding="utf-8"
    ) == "ignore\n"
    assert (workspace / ".codex" / "agents" / "nested" / "nested.toml").read_text(
        encoding="utf-8"
    ) == "ignore\n"
    assert (workspace / ".claude" / "agents" / "planner.toml").read_text(
        encoding="utf-8"
    ) == "ignore\n"


def test_solver_workspace_build_exposes_guidance_agents(tmp_path: Path) -> None:
    problem = _make_problem(tmp_path)
    builder = SolverWorkspaceBuilder(
        tmp_path / "solver_workspaces",
        problem=problem,
        config=_make_config(tmp_path),
        immutable_files={},
    )

    workspace, _ = builder.build(
        {
            "agents/codex/planner.toml": 'name = "planner"\n',
            "agents/claude/planner.md": "# Planner\n",
        },
        [],
        workspace_id="workspace",
    )

    assert (workspace / ".codex" / "agents" / "planner.toml").read_text(
        encoding="utf-8"
    ) == 'name = "planner"\n'
    assert (workspace / ".claude" / "agents" / "planner.md").read_text(
        encoding="utf-8"
    ) == "# Planner\n"
    assert (workspace / ".codex" / "agents").is_symlink()
    assert (workspace / ".claude" / "agents").is_symlink()
