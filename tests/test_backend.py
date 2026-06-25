"""Slim contract tests for backend-facing protocol surfaces."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from scaling_evolve.core.engine import RuntimeStateRef
from scaling_evolve.core.enums import ArtifactKind
from scaling_evolve.core.mutation import (
    MutationInstructionLike,
    MutationRequest,
    MutationResult,
    ProjectedArtifact,
    ProjectedProgram,
    ProjectedState,
    ProviderSpec,
)
from scaling_evolve.providers.agent.backend import AgentProvider
from scaling_evolve.providers.agent.config import AgentProviderConfig
from scaling_evolve.providers.agent.drivers._metadata import TokenPricing, resolve_token_pricing
from scaling_evolve.providers.agent.drivers.base import (
    SessionDriverCapabilities,
    SessionRollout,
    SessionSeed,
)
from scaling_evolve.providers.agent.drivers.claude_code import ClaudeCodeSessionDriver
from scaling_evolve.providers.agent.drivers.factory import build_claude_code_session_driver

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM_JSON_PROBE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stream_json_probes"
_PRICING_TABLE = {
    "sonnet": TokenPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.3,
    ),
}


class _SessionSlotDriver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._fork_count = 0

    def capabilities(self) -> SessionDriverCapabilities:
        return SessionDriverCapabilities(
            supports_native_fork=True,
            supports_cross_workspace_fork=True,
        )

    def spawn(self, seed: object) -> SessionRollout:
        self.calls.append("spawn")
        workspace = seed.workspace  # type: ignore[attr-defined]
        return self._rollout(
            RuntimeStateRef(
                state_id="runtime:spawn",
                provider_kind="agent_fork",
                session_id="session:spawn",
                workspace_id=workspace.workspace_id,
                target_repo_root=workspace.target_repo_root,
                workspace_root=workspace.workspace_root,
                session_cwd=workspace.session_cwd,
            ),
            summary=str(seed.instruction),  # type: ignore[attr-defined]
        )

    def fork_session(self, parent: RuntimeStateRef) -> str:
        self.calls.append("fork_session")
        self._fork_count += 1
        return f"{parent.session_id}:fork-{self._fork_count}"

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        self.calls.append("migrate_session")
        return f"{parent_cwd}->{child_cwd}:{session_id}"

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        self.calls.append("resume")
        return self._rollout(parent, summary=instruction)

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        self.calls.append("resume")
        return self._rollout(state, summary=instruction or "resume")

    def snapshot(self, state: RuntimeStateRef) -> object:
        self.calls.append("snapshot")
        return None

    def _rollout(self, state: RuntimeStateRef, *, summary: str) -> SessionRollout:
        session_cwd = Path(state.session_cwd or state.workspace_root or ".")
        session_cwd.mkdir(parents=True, exist_ok=True)
        candidate_path = session_cwd / "candidate.py"
        candidate_path.write_text("print('slot-patched')\n", encoding="utf-8")
        return SessionRollout(
            state=state,
            primary_path=str(candidate_path),
            changed_paths=[str(candidate_path)],
            summary=summary,
        )


class _ConcurrentStableSlotDriver(_SessionSlotDriver):
    def __init__(self) -> None:
        super().__init__()
        self._active_by_cwd: dict[str, int] = {}
        self.max_concurrent_by_cwd: dict[str, int] = {}
        self.active_total = 0
        self.max_concurrent_total = 0
        self._resume_lock = threading.Lock()
        self.first_resume_started = threading.Event()
        self.release_resume = threading.Event()

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        cwd = str(state.session_cwd or state.workspace_root or ".")
        with self._resume_lock:
            current = self._active_by_cwd.get(cwd, 0) + 1
            self._active_by_cwd[cwd] = current
            self.max_concurrent_by_cwd[cwd] = max(
                self.max_concurrent_by_cwd.get(cwd, 0),
                current,
            )
            self.active_total += 1
            self.max_concurrent_total = max(self.max_concurrent_total, self.active_total)
            self.first_resume_started.set()
        try:
            self.release_resume.wait(timeout=5)
            return super().resume(state, instruction)
        finally:
            with self._resume_lock:
                remaining = self._active_by_cwd.get(cwd, 1) - 1
                if remaining <= 0:
                    self._active_by_cwd.pop(cwd, None)
                else:
                    self._active_by_cwd[cwd] = remaining
                self.active_total -= 1


def test_session_backend_uses_query_context_and_installs_sandbox(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
                "preferred_workspace_strategy": "artifact_only",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    parent_candidate = parent_workspace / "candidate.py"
    parent_candidate.write_text("print('parent')\n", encoding="utf-8")
    reference_workspace = tmp_path / "references" / "node-0007"
    reference_workspace.mkdir(parents=True)
    reference_candidate = reference_workspace / "candidate.py"
    reference_candidate.write_text("print('reference')\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-1",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
            task_instruction="Read ./candidate.py, modify it to improve the fitness score.",
            output_format="Edit the candidate directly in your workspace.",
            workspace_note=(
                "Your file: ./candidate.py\n"
                "Edit ONLY this file. Do NOT edit files in other workspaces."
            ),
            fork_warning=(
                "This file was copied from your parent ({parent_id}).\n"
                "IMPORTANT: Your workspace has CHANGED from the previous conversation.\n"
                "Ignore any file paths from the previous conversation.\n"
                "Use ONLY ./candidate.py in your new workspace."
            ),
            context_queries="Read `.context_snapshot.json` before editing.",
        ),
        projected_state=ProjectedState(
            parent_node_id="node-0000",
            artifacts=[ProjectedArtifact(path=str(parent_candidate), summary="candidate")],
            top_programs=[
                ProjectedProgram(
                    node_id="node-0007",
                    score=1.5,
                    summary="best so far",
                    path=str(reference_candidate),
                )
            ],
            inspiration_programs=[
                ProjectedProgram(
                    node_id="node-0008",
                    score=1.2,
                    summary="inspiration",
                    source="archive",
                    path=str(reference_candidate),
                )
            ],
            mutation_surface={
                "read_roots": ["candidate.py"],
                "write_roots": ["candidate.py"],
            },
            runtime_state=RuntimeStateRef(
                state_id="runtime:parent",
                provider_kind="agent_fork",
                workspace_id="node-0000",
                workspace_root=str(parent_workspace),
                session_cwd=str(parent_workspace),
            ),
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-1",
            "child_node_id": "node-0001",
            "sqlite_path": "relative/lineage.sqlite3",
            "config_path": "configs/eve/circle_packing.smoke.yaml",
            "budget_status": {
                "turns_used": 0,
                "rollout_max_turns": 4,
                "turns_remaining": 4,
                "estimated_cost_usd": 0.012,
                "estimated_cost_per_turn_usd": 0.001,
                "current_best_score": 1.25,
                "regime": "HIGH",
            },
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
            "requested_inheritance_mode": "summary_only",
        },
    )

    result = backend.execute(request)

    session_instruction_ref = next(
        ref for ref in result.artifact_refs if ref.kind == ArtifactKind.INSTRUCTION_TEXT
    )
    session_instruction = artifact_store.read_text(session_instruction_ref)
    workspace_root = Path(result.child_runtime_state.workspace_root or "")

    assert str(reference_candidate) not in session_instruction
    assert "# Workspace" not in session_instruction
    assert "Your file: ./candidate.py" in session_instruction
    assert "Edit ONLY this file. Do NOT edit files in other workspaces." in session_instruction
    assert "Read `.context_snapshot.json` before editing." not in session_instruction
    assert "# Program Evolution History" not in session_instruction
    assert "# Boundaries" not in session_instruction
    assert "Your workspace has CHANGED" not in session_instruction
    assert "./reference_programs/" not in session_instruction
    assert not (workspace_root / "reference_programs").exists()
    assert "./.budget_status.json" not in session_instruction
    assert "[Budget]" not in session_instruction
    assert (workspace_root / ".claude" / "settings.json").exists()
    assert (workspace_root / ".hooks" / "rollout_prompts.json").exists()
    assert (workspace_root / ".sandbox_config.json").exists()
    assert (workspace_root / ".budget_status.json").exists()
    assert (workspace_root / ".evolve" / "SUMMARY.yaml").exists()
    assert (workspace_root / ".evolve" / "DIFF.yaml").exists()
    sandbox_payload = json.loads(
        (workspace_root / ".sandbox_config.json").read_text(encoding="utf-8")
    )
    settings_payload = json.loads(
        (workspace_root / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert sandbox_payload["query_context"]["run_id"] == "run-1"
    assert sandbox_payload["query_context"]["parent_node_id"] == "node-0000"
    assert Path(sandbox_payload["query_context"]["sqlite_path"]).is_absolute()
    assert Path(sandbox_payload["query_context"]["config_path"]).is_absolute()
    prompt_payload = json.loads(
        (workspace_root / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )
    pre_tool_use = settings_payload["hooks"]["PreToolUse"]
    assert pre_tool_use[0]["matcher"] == "Bash|Read|Edit|Write|MultiEdit|Glob|Grep"
    assert pre_tool_use[0]["hooks"][0]["type"] == "command"
    assert (
        "scaling_evolve.providers.agent.hooks.workspace_guard"
        in pre_tool_use[0]["hooks"][0]["command"]
    )
    assert settings_payload["hooks"]["SessionStart"][0]["hooks"][0]["type"] == "command"
    assert settings_payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]["type"] == "command"
    assert settings_payload["hooks"]["PostToolUse"][0]["hooks"][0]["type"] == "command"
    assert prompt_payload["version"] == 2
    assert prompt_payload["prompts"] == [
        {
            "name": "budget",
            "system_text": None,
            "user_text": (
                "Turn budget enabled: this session has 200 turns per rollout. "
                "After each turn you will see `[Budget] N/200 turns remaining`. "
                "Use that signal to pace your work - the current rollout will be terminated "
                "when the budget runs out."
            ),
            "turn_template": "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining",
            "turn_format_kwargs": {"rollout_max_turns": 200},
        }
    ]
    assert workspace_root.name == "node-0001"

    backend.close()


def test_session_backend_native_fork_instruction_warns_about_workspace_change(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    parent_candidate = parent_workspace / "candidate.py"
    parent_candidate.write_text("print('parent')\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-2",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
            task_instruction="Read ./candidate.py, modify it to improve the fitness score.",
            output_format="Edit the candidate directly in your workspace.",
            workspace_note=(
                "Your file: ./candidate.py\n"
                "Edit ONLY this file. Do NOT edit files in other workspaces."
            ),
            fork_warning=(
                "<workspace_change>\n"
                "Your workspace has changed. Your candidate file is now at:\n"
                "{workspace_path}/candidate.py\n"
                "Read this file before making any edits.\n"
                "</workspace_change>"
            ),
        ),
        projected_state=ProjectedState(
            parent_node_id="node-0000",
            artifacts=[ProjectedArtifact(path=str(parent_candidate), summary="candidate")],
            mutation_surface={
                "read_roots": ["candidate.py"],
                "write_roots": ["candidate.py"],
            },
            runtime_state=RuntimeStateRef(
                state_id="runtime:parent",
                provider_kind="agent_fork",
                session_id="session:parent",
                workspace_id="node-0000",
                workspace_root=str(parent_workspace),
                session_cwd=str(parent_workspace),
            ),
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-2",
            "child_node_id": "node-0001",
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
            "requested_inheritance_mode": "native",
        },
    )

    result = backend.execute(request)

    session_instruction_ref = next(
        ref for ref in result.artifact_refs if ref.kind == ArtifactKind.INSTRUCTION_TEXT
    )
    session_instruction = artifact_store.read_text(session_instruction_ref)

    assert "Your workspace has changed. Your candidate file is now at:" in session_instruction
    assert str(result.child_runtime_state.session_cwd) in session_instruction
    assert f"{result.child_runtime_state.session_cwd}/candidate.py" in session_instruction
    assert result.child_runtime_state.metadata["actual_execution_mode"] == "fork"
    assert driver.calls[:3] == ["fork_session", "migrate_session", "resume"]

    backend.close()


def test_session_backend_leaves_codex_hooks_to_driver_launch(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "codex_exec",
                "model": "gpt-5.4-mini",
                "preferred_workspace_strategy": "artifact_only",
                "rollout_max_turns": 12,
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    parent_candidate = parent_workspace / "candidate.py"
    parent_candidate.write_text("print('parent')\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-codex-hooks",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(content="Improve candidate", strategy="agent"),
        projected_state=ProjectedState(
            parent_node_id="node-0000",
            artifacts=[ProjectedArtifact(path=str(parent_candidate), summary="candidate")],
            mutation_surface={"read_roots": ["candidate.py"], "write_roots": ["candidate.py"]},
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-codex-hooks",
            "child_node_id": "node-0001",
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
        },
    )

    result = backend.execute(request)

    workspace_root = Path(result.child_runtime_state.workspace_root or "")
    rollout_payload = json.loads(
        (workspace_root / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert not (workspace_root / ".codex" / "hooks.json").exists()
    assert rollout_payload["version"] == 2
    assert rollout_payload["prompts"][0]["turn_format_kwargs"] == {"rollout_max_turns": 12}

    backend.close()


def test_session_backend_skips_workspace_codex_hooks_when_repo_hooks_exist(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    repo_hooks_path = tmp_path / "repo" / ".codex" / "hooks.json"
    repo_hooks_path.parent.mkdir(parents=True)
    repo_hooks_path.write_text("{}\n", encoding="utf-8")
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "codex_exec",
                "model": "gpt-5.4-mini",
                "preferred_workspace_strategy": "artifact_only",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    parent_candidate = parent_workspace / "candidate.py"
    parent_candidate.write_text("print('parent')\n", encoding="utf-8")
    (tmp_path / "repo").mkdir(exist_ok=True)

    request = MutationRequest(
        request_id="request:edge-codex-repo-hooks",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(content="Improve candidate", strategy="agent"),
        projected_state=ProjectedState(
            parent_node_id="node-0000",
            artifacts=[ProjectedArtifact(path=str(parent_candidate), summary="candidate")],
            mutation_surface={"read_roots": ["candidate.py"], "write_roots": ["candidate.py"]},
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-codex-repo-hooks",
            "child_node_id": "node-0001",
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
        },
    )

    result = backend.execute(request)

    workspace_root = Path(result.child_runtime_state.workspace_root or "")
    assert not (workspace_root / ".codex" / "hooks.json").exists()
    assert (workspace_root / ".hooks" / "rollout_prompts.json").exists()

    backend.close()


def test_session_backend_writes_empty_rollout_prompts_when_budget_prompt_disabled(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
                "rollout_max_turns": 12,
                "budget_prompt": False,
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    parent_candidate = parent_workspace / "candidate.py"
    parent_candidate.write_text("print('parent')\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-budget-disabled",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
            task_instruction="Read ./candidate.py, modify it to improve the fitness score.",
            output_format="Edit the candidate directly in your workspace.",
            workspace_note=(
                "Your file: ./candidate.py\n"
                "Edit ONLY this file. Do NOT edit files in other workspaces."
            ),
        ),
        projected_state=ProjectedState(
            parent_node_id="node-0000",
            artifacts=[ProjectedArtifact(path=str(parent_candidate), summary="candidate")],
            mutation_surface={"read_roots": ["candidate.py"], "write_roots": ["candidate.py"]},
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-budget-disabled",
            "child_node_id": "node-0001",
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
        },
    )

    result = backend.execute(request)

    workspace_root = Path(result.child_runtime_state.workspace_root or "")
    rollout_payload = json.loads(
        (workspace_root / ".hooks" / "rollout_prompts.json").read_text(encoding="utf-8")
    )

    assert rollout_payload["version"] == 2
    assert rollout_payload["prompts"] == []

    backend.close()


def test_session_backend_native_fork_instruction_uses_score_feedback_for_scaling_evolve(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    parent_candidate = parent_workspace / "candidate.py"
    parent_candidate.write_text("print('parent')\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-3",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
            task_instruction="Read ./candidate.py, modify it to improve the fitness score.",
            output_format="Edit the candidate directly in your workspace.",
            workspace_note=(
                "Your file: ./candidate.py\n"
                "Edit ONLY this file. Do NOT edit files in other workspaces."
            ),
            context_queries="Read `.context_snapshot.json` before editing.",
        ),
        projected_state=ProjectedState(
            parent_node_id="node-0000",
            artifacts=[ProjectedArtifact(path=str(parent_candidate), summary="candidate")],
            mutation_surface={
                "read_roots": ["candidate.py"],
                "write_roots": ["candidate.py"],
            },
            runtime_state=RuntimeStateRef(
                state_id="runtime:parent",
                provider_kind="agent_fork",
                session_id="session:parent",
                workspace_id="node-0000",
                workspace_root=str(parent_workspace),
                session_cwd=str(parent_workspace),
            ),
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-3",
            "child_node_id": "node-0001",
            "parent_primary_score": 1.25,
            "algorithm": "scaling_evolve",
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
            "requested_inheritance_mode": "native",
        },
    )

    result = backend.execute(request)

    session_instruction_ref = next(
        ref for ref in result.artifact_refs if ref.kind == ArtifactKind.INSTRUCTION_TEXT
    )
    session_instruction = artifact_store.read_text(session_instruction_ref)

    assert "Your workspace has CHANGED" not in session_instruction
    assert "Your previous work scored 1.25." in session_instruction
    assert "Try a different approach to improve it." in session_instruction
    assert "Read `.context_snapshot.json` before editing." not in session_instruction
    assert "Your file: ./candidate.py" in session_instruction
    assert "Edit ONLY this file." in session_instruction
    assert result.child_runtime_state.metadata["actual_execution_mode"] == "fork"
    assert driver.calls[:3] == ["fork_session", "migrate_session", "resume"]

    backend.close()


def test_render_iteration_instruction_uses_spawn_specific_score_feedback() -> None:
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        _SessionSlotDriver(),
    )

    request = MutationRequest(
        request_id="request:edge-spawn-score",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
        ),
        projected_state=ProjectedState(),
        metadata={"parent_primary_score": 1.25},
    )

    session_instruction = backend._render_iteration_instruction(  # noqa: SLF001
        request,
        actual_execution_mode="spawn",
    )

    assert "The current program scores 1.25." in session_instruction
    assert "Try a different approach to improve it." in session_instruction
    assert "Your previous work scored" not in session_instruction

    backend.close()


def test_claude_code_driver_appends_system_prompt_to_all_commands() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        system_prompt_append="# Task\nAlways follow the shared system rules.",
    )

    spawn_command = driver._spawn_command(  # noqa: SLF001
        type(
            "Seed",
            (),
            {
                "instruction": "Continue.",
                "workspace": None,
                "working_directory": ".",
            },
        )()
    )
    resume_command = driver._resume_command(  # noqa: SLF001
        RuntimeStateRef(
            state_id="runtime:parent",
            provider_kind="agent_fork",
            session_id="session:parent",
        ),
        "Continue.",
    )

    assert "--append-system-prompt" in spawn_command
    assert "# Task\nAlways follow the shared system rules." in spawn_command
    assert "--append-system-prompt" in resume_command
    assert "# Task\nAlways follow the shared system rules." in resume_command


def test_claude_code_driver_sets_effort_level_env() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        model="sonnet",
        max_thinking_tokens=0,
        effort_level="low",
    )

    env = driver._provider_env("/tmp/workspace")  # noqa: SLF001

    assert env is not None
    assert env["ANTHROPIC_MODEL"] == "sonnet"
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "low"
    assert env["MAX_THINKING_TOKENS"] == "0"


def test_claude_code_driver_includes_hook_events_flag_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        ClaudeCodeSessionDriver,
        "_supports_include_hook_events",
        staticmethod(lambda executable: True),
    )

    driver = ClaudeCodeSessionDriver(executable="claude")

    assert "--include-hook-events" in driver._base_flags()  # noqa: SLF001


def test_claude_code_driver_omits_hook_events_flag_when_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        ClaudeCodeSessionDriver,
        "_supports_include_hook_events",
        staticmethod(lambda executable: False),
    )

    driver = ClaudeCodeSessionDriver(executable="claude")

    assert "--include-hook-events" not in driver._base_flags()  # noqa: SLF001


def test_claude_code_driver_injects_runtime_bin_and_adaptive_thinking_toggle(
    tmp_path: Path,
) -> None:
    runtime_bin = tmp_path / ".agent-runtime" / "bin"
    runtime_bin.mkdir(parents=True)
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        model="sonnet",
        disable_adaptive_thinking=True,
    )

    env = driver._provider_env(str(tmp_path))  # noqa: SLF001

    assert env is not None
    assert env["PATH"].startswith(f"{runtime_bin}:")
    assert env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] == "1"


def test_agent_provider_prefers_virtualenv_python_for_workspace_runtime(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    venv_bin = repo_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    expected_python = venv_bin / "python"
    expected_python.write_text("#!/bin/sh\n", encoding="utf-8")
    expected_python.chmod(0o755)
    monkeypatch.setenv("VIRTUAL_ENV", str(repo_root / ".venv"))
    monkeypatch.chdir(repo_root)

    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-venv")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "model": "haiku",
                "policy_profile": "benchmark_safe",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    runtime_python = backend._runtime_python_executable()  # noqa: SLF001

    assert runtime_python == expected_python.resolve(strict=False)


def test_claude_code_driver_extracts_openrouter_cache_fallback_fields() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        model="sonnet",
        pricing_table=_PRICING_TABLE,
    )

    usage = driver._extract_usage(  # noqa: SLF001
        {
            "duration_ms": 1234,
            "num_turns": 3,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
                "cached_tokens": 40,
                "cache_write_tokens": 12,
            },
        }
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 25
    assert usage.cache_read_tokens == 40
    assert usage.cache_creation_tokens == 12
    assert usage.agent_turns == 3
    assert usage.model_cost_usd == pytest.approx(0.000687)


def test_claude_code_driver_prefers_transcript_turn_count_when_cli_num_turns_overcounts(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "session.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "session_id": "session:child",
                        "message": {
                            "id": f"msg-{index}",
                            "model": "sonnet",
                            "content": [{"type": "text", "text": f"turn {index}"}],
                        },
                    }
                )
                for index in range(12)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    driver = ClaudeCodeSessionDriver(executable="claude", model="sonnet")
    result = {
        "type": "result",
        "subtype": "error_max_turns",
        "duration_ms": 1234,
        "num_turns": 13,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "cached_tokens": 40,
            "cache_write_tokens": 12,
        },
    }

    resolved_num_turns = driver._resolved_num_turns(  # noqa: SLF001
        result,
        transcript_path=str(transcript_path),
        session_id="session:child",
    )
    usage = driver._extract_usage(result, num_turns=resolved_num_turns)  # noqa: SLF001
    metadata = driver._execution_metadata(  # noqa: SLF001
        command=["claude", "-p", "Continue."],
        cwd=str(tmp_path),
        completed=subprocess.CompletedProcess(["claude"], 1, stdout="", stderr=""),
        result_line=result,
        num_turns=resolved_num_turns,
    )

    assert resolved_num_turns == 12
    assert usage.agent_turns == 12
    assert metadata["driver_execution"]["num_turns"] == 12
    assert (
        driver._summary_from_result(result, num_turns=resolved_num_turns)  # noqa: SLF001
        == "Claude Code reached rollout_max_turns after 12 turns."
    )


def test_resolve_token_pricing_matches_family_name_by_prefix() -> None:
    pricing = resolve_token_pricing(
        "claude-haiku-4-5-20251001",
        None,
        {
            "haiku": TokenPricing(
                input_per_million=1.0,
                output_per_million=5.0,
                cache_read_per_million=0.1,
            )
        },
    )

    assert pricing is not None
    assert pricing.input_per_million == 1.0


def test_claude_code_driver_factory_passes_timeout_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    driver = build_claude_code_session_driver(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
                "timeout_seconds": 123.0,
            }
        )
    )

    assert driver.timeout_seconds == 123.0


def test_claude_code_driver_default_runner_uses_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ClaudeCodeSessionDriver(executable="claude", timeout_seconds=123.0)
    observed: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    driver._default_runner(["claude", "--help"], "/tmp/workspace")  # noqa: SLF001

    assert observed["timeout"] == 123.0


def test_claude_code_driver_raises_runtime_error_on_timeout() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        runner=lambda command, cwd, env: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(command, timeout=600.0)
        ),
    )

    with pytest.raises(RuntimeError, match="Claude Code CLI timed out"):
        driver.spawn(
            type(
                "Seed",
                (),
                {
                    "instruction": "Continue.",
                    "workspace": None,
                    "working_directory": ".",
                },
            )()
        )


def test_claude_code_driver_treats_error_max_turns_as_usable_result() -> None:
    driver = ClaudeCodeSessionDriver(executable="claude")

    payload = {
        "type": "result",
        "subtype": "error_max_turns",
        "is_error": True,
        "session_id": "session:child",
        "num_turns": 12,
    }

    assert driver._result_is_error(payload) is False  # noqa: SLF001


def test_claude_code_driver_subscription_mode_skips_isolated_config_env() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        model="sonnet",
        isolate_config=False,
        effort_level="low",
    )

    env = driver._provider_env("/tmp/workspace")  # noqa: SLF001

    assert env is not None
    assert "CLAUDE_CONFIG_DIR" not in env
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "low"


def test_claude_code_driver_emits_setting_sources_flag() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        model="sonnet",
        isolate_config=False,
        setting_sources=("project", "local"),
    )

    command = driver._spawn_command(  # noqa: SLF001
        type(
            "Seed",
            (),
            {
                "instruction": "Continue.",
                "workspace": None,
                "working_directory": ".",
            },
        )()
    )

    assert "--setting-sources" in command
    assert "project,local" in command


def test_claude_code_driver_emits_alignment_flags() -> None:
    driver = ClaudeCodeSessionDriver(executable="claude", model="sonnet")

    command = driver._spawn_command(  # noqa: SLF001
        type(
            "Seed",
            (),
            {
                "instruction": "Continue.",
                "workspace": None,
                "working_directory": ".",
            },
        )()
    )

    assert "--dangerously-skip-permissions" in command
    assert "--setting-sources" in command
    assert "project,local" in command
    assert "--permission-mode" not in command
    assert "--bare" not in command
    assert "--allowedTools" not in command
    assert "--tools" not in command


def test_claude_code_driver_emits_disallowed_tools_flag() -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        disallowed_tools=("WebSearch", "WebFetch"),
    )

    command = driver._spawn_command(  # noqa: SLF001
        type(
            "Seed",
            (),
            {
                "instruction": "Continue.",
                "workspace": None,
                "working_directory": ".",
            },
        )()
    )

    assert "--disallowedTools" in command
    assert "WebSearch,WebFetch" in command


@pytest.mark.skipif(
    not STREAM_JSON_PROBE_ROOT.exists(),
    reason="stream-json probe fixtures not present (gitignored)",
)
def test_claude_code_driver_parses_stream_json_probe_fixture() -> None:
    driver = ClaudeCodeSessionDriver(executable="claude")
    stdout = (STREAM_JSON_PROBE_ROOT / "p1_basic_agent_loop.jsonl").read_text(encoding="utf-8")

    result = driver._maybe_extract_result_line(stdout)  # noqa: SLF001

    assert result is not None
    assert result["session_id"] == "b1fe867f-7aa9-4b4a-bf47-665af1f708fe"
    assert driver._result_is_error(result) is False  # noqa: SLF001
    assert driver._summary_from_result(result).startswith("Fixed!")  # noqa: SLF001
    usage = driver._extract_usage(result)  # noqa: SLF001
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.model_cost_usd > 0
    changed_paths = driver._extract_changed_paths_from_stream_json(stdout)  # noqa: SLF001
    assert any(path.endswith("/buggy.py") for path in changed_paths)


@pytest.mark.skipif(
    not STREAM_JSON_PROBE_ROOT.exists(),
    reason="stream-json probe fixtures not present (gitignored)",
)
def test_claude_code_driver_resume_probe_uses_result_session_id() -> None:
    driver = ClaudeCodeSessionDriver(executable="claude")
    spawn_stdout = (STREAM_JSON_PROBE_ROOT / "p3_resume_spawn.jsonl").read_text(encoding="utf-8")
    resume_stdout = (STREAM_JSON_PROBE_ROOT / "p3_resume_followup.jsonl").read_text(
        encoding="utf-8"
    )

    spawn_result = driver._maybe_extract_result_line(spawn_stdout)  # noqa: SLF001
    resume_result = driver._maybe_extract_result_line(resume_stdout)  # noqa: SLF001

    assert spawn_result is not None
    assert resume_result is not None
    assert spawn_result["session_id"] == "bb8e7e11-6b5a-470a-88b5-cafcda373115"
    assert resume_result["session_id"] == spawn_result["session_id"]


def test_claude_code_driver_snapshots_transcript_into_workspace(tmp_path: Path) -> None:
    driver = ClaudeCodeSessionDriver(
        executable="claude",
        model="sonnet",
        effort_level="low",
    )

    def fake_runner(command, cwd, env):  # noqa: ANN001
        _ = command
        assert env is not None
        live_path = driver._provider_session_path(cwd=cwd, session_id="session:child")  # noqa: SLF001
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(
            (
                '{"type":"user","sessionId":"session:child",'
                '"message":{"role":"user","content":"Do work."}}\n'
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "session_id": "session:child",
                    "summary": "done",
                    "num_turns": 1,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )
            + "\n",
            stderr="",
        )

    driver.runner = fake_runner

    rollout = driver.spawn(
        SessionSeed(
            instruction="Do work.",
            working_directory=str(tmp_path),
        )
    )

    metadata = rollout.state.metadata
    snapshot_path = Path(str(metadata["provider_transcript_path"]))
    live_path = Path(str(metadata["provider_transcript_live_path"]))
    stdout_live_path = Path(str(metadata["driver_stdout_live_path"]))
    assert live_path.exists()
    assert snapshot_path.exists()
    assert stdout_live_path.exists()
    assert snapshot_path != live_path
    assert snapshot_path.read_text(encoding="utf-8") == live_path.read_text(encoding="utf-8")
    assert stdout_live_path.read_text(encoding="utf-8") == str(metadata["driver_stdout"])
    assert Path(str(metadata["attempt_root"])) == tmp_path / ".claude-driver-transcripts"
    assert metadata["driver_execution"]["effort_level"] == "low"


def test_session_backend_prefers_rollout_summary_for_artifact_store_portable_state(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    request = MutationRequest(
        request_id="request:edge-semantic",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(content="Improve candidate"),
        projected_state=ProjectedState(summary="parent summary"),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-semantic",
        },
    )

    portable_state = backend._resolve_portable_state(  # noqa: SLF001
        request,
        None,
        None,
        rollout_summary="child summary",
    )

    assert portable_state is not None
    assert portable_state.summary == "child summary"

    backend.close()


def test_session_backend_native_fork_rewrites_query_context_after_workspace_copy(
    tmp_path: Path,
    build_session_stack,
    fake_session_driver_factory,
) -> None:
    _lineage_store, artifact_store, workspace_manager = build_session_stack("run-1")
    driver = fake_session_driver_factory(artifact_store)
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
        artifact_store=artifact_store,
        workspace_manager=workspace_manager,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    (parent_workspace / "candidate.py").write_text("print('parent')\n", encoding="utf-8")
    (parent_workspace / ".sandbox_config.json").write_text(
        json.dumps(
            {
                "query_context": {
                    "run_id": "run-stale",
                    "parent_node_id": "node-stale",
                    "child_node_id": "node-stale-child",
                    "sqlite_path": str(tmp_path / "stale.sqlite3"),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-4",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
            task_instruction="Read ./candidate.py, modify it to improve the fitness score.",
            output_format="Edit the candidate directly in your workspace.",
            workspace_note="Your file: ./candidate.py",
            context_queries="Read `.context_snapshot.json` before evaluating the parent.",
        ),
        projected_state=ProjectedState(
            parent_node_id="node-0007",
            artifacts=[
                ProjectedArtifact(
                    path=str(parent_workspace / "candidate.py"),
                    summary="candidate",
                )
            ],
            mutation_surface={
                "read_roots": ["candidate.py"],
                "write_roots": ["candidate.py"],
            },
            runtime_state=RuntimeStateRef(
                state_id="runtime:parent",
                provider_kind="agent_fork",
                session_id="session:parent",
                workspace_id="node-0007",
                workspace_root=str(parent_workspace),
                session_cwd=str(parent_workspace),
                metadata={
                    "compact_event_count": 2,
                    "last_compact_timestamp": "2026-03-27T15:30:00+00:00",
                    "last_compact_line_number": 48,
                },
            ),
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-4",
            "child_node_id": "node-0008",
            "algorithm": "scaling_evolve",
            "sqlite_path": str(tmp_path / "lineage.sqlite3"),
            "target_binding": {
                "kind": "managed",
                "repo_root": str(tmp_path / "repo"),
                "config_root": str(tmp_path / "repo"),
            },
            "assessment_contract": {"kind": "objective"},
            "requested_inheritance_mode": "native",
        },
    )

    result = backend.execute(request)

    workspace_root = Path(result.child_runtime_state.workspace_root or "")
    sandbox_payload = json.loads(
        (workspace_root / ".sandbox_config.json").read_text(encoding="utf-8")
    )

    assert sandbox_payload["query_context"]["run_id"] == "run-1"
    assert sandbox_payload["query_context"]["parent_node_id"] == "node-0007"
    assert sandbox_payload["query_context"]["child_node_id"] == "node-0008"
    assert sandbox_payload["query_context"]["compact_event_count"] == 2
    assert (
        sandbox_payload["query_context"]["compact_boundary_timestamp"]
        == "2026-03-27T15:30:00+00:00"
    )
    assert sandbox_payload["query_context"]["last_compact_line_number"] == 48

    backend.close()


def test_session_backend_native_fork_syncs_stable_session_slot_back_to_child_workspace(
    tmp_path: Path,
) -> None:
    driver = _SessionSlotDriver()
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    (parent_workspace / "candidate.py").write_text("print('parent')\n", encoding="utf-8")
    stable_slot = tmp_path / "session-slots" / "slot-0001"
    stable_slot.mkdir(parents=True)
    (stable_slot / "candidate.py").write_text("print('slot-parent')\n", encoding="utf-8")
    child_workspace = tmp_path / "child-node"
    (tmp_path / "repo").mkdir()

    request = MutationRequest(
        request_id="request:edge-slot",
        backend=ProviderSpec(kind="agent_fork"),
        instruction=MutationInstructionLike(
            content="Improve candidate",
            strategy="agent",
            task_instruction="Open ./candidate.py and improve it.",
            workspace_note="Your file: ./candidate.py",
        ),
        projected_state=ProjectedState(
            parent_node_id="node-0001",
            artifacts=[
                ProjectedArtifact(
                    path=str(parent_workspace / "candidate.py"),
                    summary="candidate",
                )
            ],
            mutation_surface={
                "read_roots": ["candidate.py"],
                "write_roots": ["candidate.py"],
            },
            runtime_state=RuntimeStateRef(
                state_id="runtime:parent",
                provider_kind="agent_fork",
                session_id="session:parent",
                workspace_id="node-0001",
                workspace_root=str(parent_workspace),
                session_cwd=str(stable_slot),
            ),
            metadata={
                "workspace": {
                    "workspace_id": "node-0002",
                    "target_repo_root": str(child_workspace),
                    "workspace_root": str(child_workspace),
                    "session_cwd": str(stable_slot),
                    "strategy": "artifact_only",
                }
            },
        ),
        metadata={
            "run_id": "run-1",
            "edge_id": "edge-slot",
            "child_node_id": "node-0002",
            "requested_inheritance_mode": "native",
        },
    )

    result = backend.execute(request)

    assert result.child_runtime_state is not None
    assert result.child_runtime_state.session_cwd == str(stable_slot)
    assert result.child_runtime_state.workspace_root == str(child_workspace)
    assert (child_workspace / "candidate.py").read_text(encoding="utf-8") == (
        "print('slot-patched')\n"
    )
    assert driver.calls[:3] == ["fork_session", "migrate_session", "resume"]

    backend.close()


def test_session_backend_serializes_concurrent_reuse_of_stable_session_slot(
    tmp_path: Path,
) -> None:
    driver = _ConcurrentStableSlotDriver()
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    (parent_workspace / "candidate.py").write_text("print('parent')\n", encoding="utf-8")
    stable_slot = tmp_path / "session-slots" / "slot-0001"
    stable_slot.mkdir(parents=True)
    (stable_slot / "candidate.py").write_text("print('slot-parent')\n", encoding="utf-8")

    def make_request(child_node_id: str) -> MutationRequest:
        child_workspace = tmp_path / child_node_id
        return MutationRequest(
            request_id=f"request:{child_node_id}",
            backend=ProviderSpec(kind="agent_fork"),
            instruction=MutationInstructionLike(
                content="Improve candidate",
                strategy="agent",
                task_instruction="Open ./candidate.py and improve it.",
                workspace_note="Your file: ./candidate.py",
            ),
            projected_state=ProjectedState(
                parent_node_id="node-0001",
                artifacts=[
                    ProjectedArtifact(
                        path=str(parent_workspace / "candidate.py"),
                        summary="candidate",
                    )
                ],
                mutation_surface={
                    "read_roots": ["candidate.py"],
                    "write_roots": ["candidate.py"],
                },
                runtime_state=RuntimeStateRef(
                    state_id="runtime:parent",
                    provider_kind="agent_fork",
                    session_id="session:parent",
                    workspace_id="node-0001",
                    workspace_root=str(parent_workspace),
                    session_cwd=str(stable_slot),
                ),
                metadata={
                    "workspace": {
                        "workspace_id": child_node_id,
                        "target_repo_root": str(child_workspace),
                        "workspace_root": str(child_workspace),
                        "session_cwd": str(stable_slot),
                        "strategy": "artifact_only",
                    }
                },
            ),
            metadata={
                "run_id": "run-1",
                "edge_id": f"edge:{child_node_id}",
                "child_node_id": child_node_id,
                "requested_inheritance_mode": "native",
            },
        )

    requests = [make_request("node-0002"), make_request("node-0003")]
    results: list[MutationResult | None] = [None, None]
    failures: list[BaseException] = []

    def run(index: int) -> None:
        try:
            results[index] = backend.execute(requests[index])
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            failures.append(exc)

    first = threading.Thread(target=run, args=(0,))
    second = threading.Thread(target=run, args=(1,))
    first.start()
    assert driver.first_resume_started.wait(timeout=5)
    second.start()
    time.sleep(0.2)
    driver.release_resume.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not failures
    assert all(result is not None for result in results)
    assert driver.max_concurrent_by_cwd.get(str(stable_slot), 0) == 1

    backend.close()


def test_session_backend_keeps_different_stable_session_slots_parallel(
    tmp_path: Path,
) -> None:
    driver = _ConcurrentStableSlotDriver()
    backend = AgentProvider(
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "claude_code",
                "provider": "openrouter",
            }
        ),
        driver,
    )

    parent_workspace = tmp_path / "parent-node"
    parent_workspace.mkdir()
    (parent_workspace / "candidate.py").write_text("print('parent')\n", encoding="utf-8")
    stable_slot_a = tmp_path / "session-slots" / "slot-0001"
    stable_slot_b = tmp_path / "session-slots" / "slot-0002"
    stable_slot_a.mkdir(parents=True)
    stable_slot_b.mkdir(parents=True)
    (stable_slot_a / "candidate.py").write_text("print('slot-a')\n", encoding="utf-8")
    (stable_slot_b / "candidate.py").write_text("print('slot-b')\n", encoding="utf-8")

    def make_request(child_node_id: str, slot: Path, parent_node_id: str) -> MutationRequest:
        child_workspace = tmp_path / child_node_id
        return MutationRequest(
            request_id=f"request:{child_node_id}",
            backend=ProviderSpec(kind="agent_fork"),
            instruction=MutationInstructionLike(
                content="Improve candidate",
                strategy="agent",
                task_instruction="Open ./candidate.py and improve it.",
                workspace_note="Your file: ./candidate.py",
            ),
            projected_state=ProjectedState(
                parent_node_id=parent_node_id,
                artifacts=[
                    ProjectedArtifact(
                        path=str(parent_workspace / "candidate.py"),
                        summary="candidate",
                    )
                ],
                mutation_surface={
                    "read_roots": ["candidate.py"],
                    "write_roots": ["candidate.py"],
                },
                runtime_state=RuntimeStateRef(
                    state_id=f"runtime:{parent_node_id}",
                    provider_kind="agent_fork",
                    session_id=f"session:{parent_node_id}",
                    workspace_id=parent_node_id,
                    workspace_root=str(parent_workspace),
                    session_cwd=str(slot),
                ),
                metadata={
                    "workspace": {
                        "workspace_id": child_node_id,
                        "target_repo_root": str(child_workspace),
                        "workspace_root": str(child_workspace),
                        "session_cwd": str(slot),
                        "strategy": "artifact_only",
                    }
                },
            ),
            metadata={
                "run_id": "run-1",
                "edge_id": f"edge:{child_node_id}",
                "child_node_id": child_node_id,
                "requested_inheritance_mode": "native",
            },
        )

    requests = [
        make_request("node-0002", stable_slot_a, "node-0001"),
        make_request("node-0003", stable_slot_b, "node-0009"),
    ]
    failures: list[BaseException] = []

    def run(index: int) -> None:
        try:
            backend.execute(requests[index])
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            failures.append(exc)

    first = threading.Thread(target=run, args=(0,))
    second = threading.Thread(target=run, args=(1,))
    first.start()
    assert driver.first_resume_started.wait(timeout=5)
    second.start()
    time.sleep(0.2)
    driver.release_resume.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not failures
    assert driver.max_concurrent_by_cwd.get(str(stable_slot_a), 0) == 1
    assert driver.max_concurrent_by_cwd.get(str(stable_slot_b), 0) == 1
    assert driver.max_concurrent_total >= 2

    backend.close()
