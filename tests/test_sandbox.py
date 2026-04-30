"""Sandbox hook tests for workspace guard behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scaling_evolve.providers.agent.local_workspace_manager import LocalWorkspaceManager
from scaling_evolve.providers.agent.workspaces import WorkspaceLeaseRequest, WorkspacePlan

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_hook(
    *,
    cwd: Path,
    hook_event_name: str = "PreToolUse",
    tool_name: str,
    tool_input: dict[str, object],
    extra_payload: dict[str, object] | None = None,
) -> subprocess.CompletedProcess[str]:
    payload = {
        "cwd": str(cwd),
        "hook_event_name": hook_event_name,
        "tool_name": tool_name,
        "tool_input": tool_input,
    }
    if extra_payload:
        payload.update(extra_payload)
    return subprocess.run(
        [sys.executable, "-m", "scaling_evolve.providers.agent.hooks.workspace_guard"],
        cwd=cwd,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
    )


def _write_config(workspace: Path, *, evaluator_dir: Path) -> None:
    (workspace / ".sandbox_config.json").write_text(
        json.dumps(
            {
                "own_workspace": str(workspace),
                "evaluator_dirs": [str(evaluator_dir)],
            }
        ),
        encoding="utf-8",
    )


def _write_budget_status(workspace: Path, *, turns_used: int, rollout_max_turns: int) -> None:
    remaining = max(rollout_max_turns - turns_used, 0)
    (workspace / ".budget_status.json").write_text(
        json.dumps(
            {
                "turns_used": turns_used,
                "rollout_max_turns": rollout_max_turns,
                "turns_remaining": remaining,
                "estimated_cost_usd": 0.012,
                "estimated_cost_per_turn_usd": 0.001,
                "current_best_score": 1.25,
                "regime": "HIGH",
            }
        ),
        encoding="utf-8",
    )


def _write_agent_hooks_config(
    workspace: Path,
    *,
    rollout_max_turns: int,
    system_text: str | None = None,
) -> None:
    hooks_dir = workspace / ".hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    (hooks_dir / "rollout_prompts.json").write_text(
        json.dumps(
            {
                "version": 2,
                "prompts": [
                    {
                        "name": "budget",
                        "system_text": system_text,
                        "user_text": (
                            "Turn budget enabled: this session has "
                            f"{rollout_max_turns} turns per rollout."
                        ),
                        "turn_template": (
                            "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining"
                        ),
                        "turn_format_kwargs": {"rollout_max_turns": rollout_max_turns},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_workspace_guard_blocks_edit_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Edit",
        tool_input={"file_path": str(tmp_path / "outside.py")},
    )

    assert result.returncode == 2
    assert "outside your workspace" in result.stderr


def test_workspace_guard_allows_edit_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Edit",
        tool_input={"file_path": str(workspace / "candidate.py")},
    )

    assert result.returncode == 0


def test_workspace_guard_uses_config_workspace_when_cwd_is_nested(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    nested_cwd = workspace / "nested"
    nested_cwd.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    candidate = workspace / "candidate.py"
    candidate.write_text("print('ok')\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=nested_cwd,
        tool_name="Edit",
        tool_input={"file_path": str(candidate)},
    )

    assert result.returncode == 0


def test_workspace_guard_blocks_reads_of_evaluator_sources(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    evaluator_file = evaluator_dir / "secret.py"
    evaluator_file.write_text("SECRET = 1\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Read",
        tool_input={"file_path": str(evaluator_file)},
    )

    assert result.returncode == 2
    assert "off-limits" in result.stderr


def test_workspace_guard_allows_reads_of_other_node_workspaces(tmp_path: Path) -> None:
    root = tmp_path / "workspaces"
    workspace = root / "node-0001"
    other_workspace = root / "node-0002"
    workspace.mkdir(parents=True)
    other_workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    (other_workspace / "candidate.py").write_text("print('other')\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Read",
        tool_input={"file_path": str(other_workspace / "candidate.py")},
    )

    assert result.returncode == 0


def test_workspace_guard_blocks_bash_reads_outside_workspace_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Bash",
        tool_input={"command": "cat /etc/passwd"},
    )

    assert result.returncode == 2
    assert "outside your workspace" in result.stderr


def test_workspace_guard_allows_python_candidate_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    candidate = workspace / "candidate.py"
    candidate.write_text("print('ok')\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Bash",
        tool_input={"command": "python candidate.py"},
    )

    assert result.returncode == 0


def test_workspace_guard_blocks_python_target_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("print('x')\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Bash",
        tool_input={"command": f"python {outside}"},
    )

    assert result.returncode == 2
    assert "outside your workspace" in result.stderr


def test_workspace_guard_allows_claude_background_task_output_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    task_output = (
        tmp_path / "private" / "tmp" / "claude-501" / "session" / "tasks" / "task-1.output"
    )
    task_output.parent.mkdir(parents=True)
    task_output.write_text("done\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)

    result = _run_hook(
        cwd=workspace,
        tool_name="Bash",
        tool_input={"command": f"cat {task_output}"},
    )

    assert result.returncode == 0


def test_workspace_guard_post_tool_use_emits_budget_context_once_per_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    _write_config(workspace, evaluator_dir=evaluator_dir)
    _write_agent_hooks_config(workspace, rollout_max_turns=4)
    transcript_path = workspace / "rollout.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"session-1"}}',
                (
                    '{"type":"event_msg","payload":{"type":"agent_message",'
                    '"message":"Inspecting the workspace."}}'
                ),
                (
                    '{"type":"response_item","payload":{"type":"function_call",'
                    '"name":"exec_command","arguments":"{}","call_id":"call-1"}}'
                ),
                (
                    '{"type":"response_item","payload":{"type":"function_call",'
                    '"name":"exec_command","arguments":"{}","call_id":"call-2"}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first = _run_hook(
        cwd=workspace,
        hook_event_name="PostToolUse",
        tool_name="Read",
        tool_input={},
        extra_payload={"tool_use_id": "call-1", "transcript_path": str(transcript_path)},
    )
    second = _run_hook(
        cwd=workspace,
        hook_event_name="PostToolUse",
        tool_name="Read",
        tool_input={},
        extra_payload={"tool_use_id": "call-2", "transcript_path": str(transcript_path)},
    )

    first_payload = json.loads(first.stdout)

    assert first.returncode == 0
    assert second.returncode == 0
    assert (
        "[Budget] 3/4 turns remaining" in first_payload["hookSpecificOutput"]["additionalContext"]
    )
    assert second.stdout == ""


def test_workspace_guard_session_start_emits_system_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    _write_config(workspace, evaluator_dir=evaluator_dir)
    _write_agent_hooks_config(
        workspace,
        rollout_max_turns=4,
        system_text="system prompt from rollout config",
    )

    result = _run_hook(
        cwd=workspace,
        hook_event_name="SessionStart",
        tool_name="Read",
        tool_input={},
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["hookSpecificOutput"]["additionalContext"] == "system prompt from rollout config"


def test_workspace_guard_post_tool_use_emits_budget_context_once_per_turn_for_claude_transcript(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    _write_config(workspace, evaluator_dir=evaluator_dir)
    _write_agent_hooks_config(workspace, rollout_max_turns=4)
    transcript_path = workspace / "claude-rollout.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                (
                    '{"type":"assistant","message":{"id":"msg-1","content":'
                    '[{"type":"tool_use","id":"call-1","name":"Read","input":{}},'
                    '{"type":"tool_use","id":"call-2","name":"Bash","input":{}}]}}'
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first = _run_hook(
        cwd=workspace,
        hook_event_name="PostToolUse",
        tool_name="Read",
        tool_input={},
        extra_payload={"tool_use_id": "call-1", "transcript_path": str(transcript_path)},
    )
    second = _run_hook(
        cwd=workspace,
        hook_event_name="PostToolUse",
        tool_name="Read",
        tool_input={},
        extra_payload={"tool_use_id": "call-2", "transcript_path": str(transcript_path)},
    )

    first_payload = json.loads(first.stdout)

    assert first.returncode == 0
    assert second.returncode == 0
    assert (
        "[Budget] 3/4 turns remaining" in first_payload["hookSpecificOutput"]["additionalContext"]
    )
    assert second.stdout == ""


def test_workspace_guard_pre_tool_use_preserves_budget_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "node-0001"
    workspace.mkdir(parents=True)
    evaluator_dir = tmp_path / "src" / "scaling_evolve" / "applications"
    evaluator_dir.mkdir(parents=True)
    candidate = workspace / "candidate.py"
    candidate.write_text("print('ok')\n", encoding="utf-8")
    _write_config(workspace, evaluator_dir=evaluator_dir)
    _write_budget_status(workspace, turns_used=0, rollout_max_turns=4)

    result = _run_hook(
        cwd=workspace,
        tool_name="Read",
        tool_input={"file_path": str(candidate)},
    )
    budget_file = json.loads((workspace / ".budget_status.json").read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert result.stdout == ""
    assert budget_file["turns_used"] == 0
    assert budget_file["turns_remaining"] == 4


def test_local_workspace_manager_uses_requested_node_id_for_workspace_name(tmp_path: Path) -> None:
    manager = LocalWorkspaceManager(tmp_path / "workspaces")

    lease = manager.acquire(
        WorkspaceLeaseRequest(
            run_id="run-1",
            node_id="node-0004",
            purpose="mutation",
            plan=WorkspacePlan(strategy="artifact_only", purpose="mutation"),
        )
    )

    assert lease.workspace_id == "node-0004"
    assert Path(lease.workspace_root).name == "node-0004"


def test_local_workspace_manager_release_preserves_snapshotted_workspace(tmp_path: Path) -> None:
    manager = LocalWorkspaceManager(tmp_path / "workspaces")
    lease = manager.acquire(
        WorkspaceLeaseRequest(
            run_id="run-1",
            node_id="node-0002",
            purpose="mutation",
            plan=WorkspacePlan(strategy="artifact_only", purpose="mutation"),
        )
    )
    candidate = Path(lease.workspace_root) / "candidate.py"
    candidate.write_text("print('ok')\n", encoding="utf-8")

    manager.release(lease, snapshot=True)

    assert candidate.exists()


def test_local_workspace_manager_release_deletes_unsnapshotted_workspace(tmp_path: Path) -> None:
    manager = LocalWorkspaceManager(tmp_path / "workspaces")
    lease = manager.acquire(
        WorkspaceLeaseRequest(
            run_id="run-1",
            node_id="node-0003",
            purpose="mutation",
            plan=WorkspacePlan(strategy="artifact_only", purpose="mutation"),
        )
    )
    workspace_root = Path(lease.workspace_root)
    (workspace_root / "candidate.py").write_text("print('ok')\n", encoding="utf-8")

    manager.release(lease, snapshot=False)

    assert not workspace_root.exists()
