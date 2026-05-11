"""Persistent session provider implementation."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import cast

from scaling_evolve.core.bindings import AssessmentContract, TargetBinding
from scaling_evolve.core.common import JSONValue, utc_now
from scaling_evolve.core.engine import PortableStateRef, RuntimeStateRef, WorkspaceLease
from scaling_evolve.core.enums import ArtifactKind, ExecutionLifecycle, InheritanceMode
from scaling_evolve.core.mutation import (
    CapabilitySet,
    MutationRequest,
    MutationResult,
    ProjectedProgram,
    ProjectedState,
)
from scaling_evolve.core.storage.models import ArtifactRef, MaterializationRef
from scaling_evolve.providers.agent.codex_hooks import (
    repo_codex_hooks_path,
    workspace_hook_command,
    write_codex_hooks_file,
)
from scaling_evolve.providers.agent.compaction import compact_metadata_from_transcript
from scaling_evolve.providers.agent.config import AgentProviderConfig
from scaling_evolve.providers.agent.drivers.base import (
    SessionDriver,
    SessionRollout,
    SessionSeed,
    SessionSnapshot,
    SessionWorkspaceLease,
)
from scaling_evolve.providers.agent.python_runtime import (
    PythonExecutionPolicy,
    write_python_runtime,
)
from scaling_evolve.providers.agent.summarization import (
    build_lineage_summary,
    build_transcript_digest,
)
from scaling_evolve.providers.agent.workspace_resolver import resolve_workspace_plan
from scaling_evolve.providers.agent.workspaces import WorkspaceLeaseManager, WorkspaceLeaseRequest
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.manifests import ChangedFilesManifest

_LOGGER = logging.getLogger(__name__)


def _as_mapping(value: object) -> Mapping[str, JSONValue] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, JSONValue], value)


class AgentProvider:
    """Stateful mutation provider driven by a persistent session driver."""

    kind = "agent_fork"
    capabilities = CapabilitySet(
        supports_tools=True,
        supports_runtime_state=True,
        supports_native_fork=True,
        supports_summary_only=False,
    )

    def __init__(
        self,
        config: AgentProviderConfig,
        driver: SessionDriver,
        *,
        artifact_store: FSArtifactStore | None = None,
        workspace_manager: WorkspaceLeaseManager | None = None,
    ) -> None:
        self.config = config
        self.driver = driver
        self.artifact_store = artifact_store
        self.workspace_manager = workspace_manager
        driver_capabilities = driver.capabilities()
        self.capabilities = CapabilitySet(
            supports_tools=True,
            supports_runtime_state=True,
            supports_native_fork=driver_capabilities.supports_native_fork,
            supports_summary_only=False,
        )
        self._retained_leases: dict[str, WorkspaceLease] = {}
        self._session_cwd_locks: dict[str, threading.RLock] = {}
        self._session_cwd_locks_guard = threading.Lock()
        atexit.register(self.close)

    @contextmanager
    def _session_cwd_lock(self, session_cwd: str):
        # Scaling Evolve deliberately reuses a stable session cwd along a lineage so
        # forked children can keep continuation state and prompt-cache locality.
        # Once multiple children share that slot, the slot becomes a serialized
        # resource: concurrent prepare/edit/sync cycles would race on the same
        # candidate.py and Claude driver metadata.
        key = str(Path(session_cwd).expanduser().resolve(strict=False))
        with self._session_cwd_locks_guard:
            lock = self._session_cwd_locks.setdefault(key, threading.RLock())
        with lock:
            yield

    def execute(self, request: MutationRequest) -> MutationResult:
        requested_inheritance_mode = self._requested_inheritance_mode(request)
        workspace, acquired_lease, workspace_inheritance = self._resolve_workspace(request)
        workspace = self._resolve_session_workspace(request, workspace)
        parent_state = (
            request.projected_state.runtime_state
            if requested_inheritance_mode == InheritanceMode.NATIVE
            else None
        )
        fallback_reason: str | None = None
        active_state: RuntimeStateRef | None = None
        actual_execution_mode = "spawn"
        instruction_text_ref: ArtifactRef | None = None

        try:
            with self._session_cwd_lock(workspace.session_cwd):
                self._prepare_workspace_runtime(request, workspace)
                if self._can_native_fork(parent_state, workspace):
                    parent_runtime_state = cast(RuntimeStateRef, parent_state)
                    rollout, instruction_text_ref = self._execute_native_fork(
                        request,
                        parent_runtime_state,
                        workspace,
                        acquired_lease,
                    )
                    actual_execution_mode = "fork"
                    workspace_inheritance = "fork_copy"
                else:
                    session_instruction = self._render_session_instruction(
                        request,
                        workspace,
                        actual_execution_mode=actual_execution_mode,
                    )
                    instruction_text_ref = self._store_session_instruction_text(
                        request,
                        session_instruction,
                    )
                    if parent_state is not None and self.config.fork_mode == "native":
                        fallback_reason = "cross_workspace_fork_unsupported"
                    rollout = self.driver.spawn(
                        SessionSeed(
                            instruction=session_instruction,
                            workspace=workspace,
                            working_directory=workspace.session_cwd,
                            attachments=[
                                artifact.ref
                                for artifact in request.projected_state.artifacts
                                if artifact.ref is not None
                            ],
                        )
                    )
                active_state = self.retain_runtime_state(
                    rollout.state,
                    workspace=workspace,
                    lease=acquired_lease,
                    run_id=self._run_id(request),
                    owner_node_id=request.projected_state.parent_node_id,
                    purpose="mutation",
                )
                rollout = rollout.model_copy(update={"state": active_state})
                rollout, driver_debug_refs = self._persist_driver_debug_artifacts(request, rollout)
                self._sync_session_changes_to_workspace(request, rollout, workspace)
                transcript_ref = self._resolve_transcript(rollout)
                session_archive_ref = self._persist_provider_session_archive(request, rollout)
                (
                    changed_files_ref,
                    manifest_source,
                    changed_manifest,
                ) = self._resolve_changed_files_manifest(
                    request,
                    rollout,
                    workspace=workspace,
                    actual_execution_mode=actual_execution_mode,
                )
                transcript_digest_ref = self._resolve_transcript_digest(
                    request,
                    rollout,
                    None,
                    transcript_ref,
                )
                portable_state = self._resolve_portable_state(
                    request,
                    None,
                    transcript_digest_ref,
                    rollout_summary=rollout.summary,
                )
                materialization = self._resolve_child_materialization(
                    request,
                    workspace,
                    changed_files_ref,
                    changed_manifest,
                )
                runtime_state = self.retain_runtime_state(
                    self._enrich_runtime_state(
                        rollout.state,
                        workspace=workspace,
                        transcript_ref=transcript_ref,
                        session_archive_ref=session_archive_ref,
                        changed_files_ref=changed_files_ref,
                        manifest_source=manifest_source,
                        requested_inheritance_mode=requested_inheritance_mode,
                        actual_execution_mode=actual_execution_mode,
                        actual_workspace_inheritance=workspace_inheritance,
                        fallback_reason=fallback_reason or rollout.fallback_reason,
                        run_id=self._run_id(request),
                        owner_node_id=request.projected_state.parent_node_id,
                    ),
                    workspace=workspace,
                    lease=acquired_lease,
                    run_id=self._run_id(request),
                    owner_node_id=request.projected_state.parent_node_id,
                    purpose="mutation",
                )
            artifact_refs = [
                ref
                for ref in [
                    instruction_text_ref,
                    transcript_ref,
                    session_archive_ref,
                    changed_files_ref,
                    transcript_digest_ref,
                    *driver_debug_refs,
                ]
                if ref is not None
            ]
            return MutationResult(
                request_id=request.request_id,
                provider_kind=self.kind,
                status="ok",
                output_text=rollout.summary or request.instruction.content,
                child_materialization=materialization,
                child_portable_state=portable_state,
                child_runtime_state=runtime_state,
                artifact_refs=artifact_refs,
                usage=rollout.usage,
            )
        except Exception:
            if active_state is not None:
                self.release_runtime_state(active_state, final_lifecycle=ExecutionLifecycle.FAILED)
            elif acquired_lease is not None:
                self._release_workspace_lease(acquired_lease)
            raise

    def _resolve_workspace(
        self,
        request: MutationRequest,
    ) -> tuple[SessionWorkspaceLease, WorkspaceLease | None, str]:
        target_binding = self._target_binding(request)
        assessment = self._assessment_contract(request)
        workspace = self._metadata_workspace(request)
        acquired_lease: WorkspaceLease | None = None
        workspace_inheritance = "fresh"
        if workspace is None and self.workspace_manager is not None and target_binding is not None:
            plan = resolve_workspace_plan(
                target_binding=target_binding,
                assessment=assessment or AssessmentContract(kind="objective"),
                provider_kind=request.provider.kind,
                purpose="mutation",
                mutation_surface=request.projected_state.mutation_surface,
                preferred_strategy=self.config.preferred_workspace_strategy,
            )
            acquired_lease = self.workspace_manager.acquire(
                WorkspaceLeaseRequest(
                    run_id=self._run_id(request),
                    node_id=self._child_node_id(request),
                    purpose="mutation",
                    owner_node_id=request.projected_state.parent_node_id,
                    target_repo_root=target_binding.repo_root or target_binding.config_root,
                    plan=plan,
                )
            )
            workspace = SessionWorkspaceLease(
                workspace_id=acquired_lease.workspace_id,
                target_repo_root=(
                    acquired_lease.target_repo_root
                    or acquired_lease.workspace_root
                    or acquired_lease.root
                ),
                workspace_root=acquired_lease.workspace_root or acquired_lease.root,
                session_cwd=acquired_lease.session_cwd or acquired_lease.root,
                strategy=acquired_lease.strategy,
            )
            if plan.strategy == "artifact_only":
                self._materialize_artifact_only_workspace(request, workspace)

        if workspace is None:
            working_directory = self._working_directory_hint(request)
            workspace = SessionWorkspaceLease(
                workspace_id=f"workspace:{request.request_id}",
                target_repo_root=working_directory,
                workspace_root=working_directory,
                session_cwd=working_directory,
            )

        return (workspace, acquired_lease, workspace_inheritance)

    def _prepare_workspace_runtime(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
    ) -> None:
        workspace_root = Path(workspace.workspace_root).expanduser().resolve(strict=False)
        self._write_runtime_bundle(workspace_root, request)
        session_cwd = Path(workspace.session_cwd).expanduser().resolve(strict=False)
        if workspace_root == session_cwd:
            return
        self._sync_workspace_tree_to_session_cwd(workspace_root, session_cwd)
        self._write_runtime_bundle(session_cwd, request)

    def _write_runtime_bundle(
        self,
        runtime_root: Path,
        request: MutationRequest,
    ) -> None:
        runtime_root.mkdir(parents=True, exist_ok=True)
        self._write_workspace_semantic_templates(runtime_root)
        self._write_workspace_sandbox_config(runtime_root, request)
        self._write_workspace_python_runtime(runtime_root, request)
        self._write_workspace_budget_status(runtime_root, request)
        self._write_workspace_rollout_prompt_config(runtime_root)
        if self.config.driver in {"claude_code", "claude_code_tmux"}:
            self._write_workspace_claude_settings(runtime_root)
        if self.config.driver in {"codex_cli", "codex_tmux", "codex_exec"}:
            self._write_workspace_codex_hooks(runtime_root)
        self._write_workspace_recovery_guidance(runtime_root, request)

    def _sync_workspace_tree_to_session_cwd(
        self,
        workspace_root: Path,
        session_cwd: Path,
    ) -> None:
        preserve_names = {".claude-driver-config"}
        session_cwd.mkdir(parents=True, exist_ok=True)
        for existing in list(session_cwd.iterdir()):
            if existing.name in preserve_names:
                continue
            if (workspace_root / existing.name).exists():
                continue
            if existing.is_dir():
                shutil.rmtree(existing, ignore_errors=True)
            else:
                existing.unlink(missing_ok=True)
        for source in workspace_root.iterdir():
            if source.name in preserve_names:
                continue
            target = session_cwd / source.name
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def _write_workspace_semantic_templates(self, workspace_root: Path) -> None:
        evolve_dir = workspace_root / ".evolve"
        evolve_dir.mkdir(parents=True, exist_ok=True)
        summary_path = evolve_dir / "SUMMARY.yaml"
        if not summary_path.exists():
            summary_path.write_text(
                "\n".join(
                    [
                        "summary: >",
                        (
                            "  Update with a one-sentence self-contained description "
                            "of the current approach."
                        ),
                        "",
                        "mechanisms:",
                        "  - Update with key mechanism 1.",
                        "",
                        "risks:",
                        "  - Update with the main known weakness.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        diff_path = evolve_dir / "DIFF.yaml"
        if not diff_path.exists():
            diff_path.write_text(
                "\n".join(
                    [
                        "delta: >",
                        "  Update with a one-sentence parent-relative change.",
                        "",
                        "motivation: >",
                        "  Update with why this mutation was attempted.",
                        "",
                        "result:",
                        "  status: unknown",
                        "  note: >",
                        "    Update with the observed outcome if known.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    def _write_workspace_sandbox_config(
        self,
        workspace_root: Path,
        request: MutationRequest,
    ) -> None:
        payload = {
            "own_workspace": str(workspace_root.resolve()),
            "evaluator_dirs": [str(path) for path in self._evaluator_dirs(request)],
            "execution_policy": self._python_execution_policy().to_payload(),
        }
        query_context = self._query_context_payload(request)
        if query_context:
            payload["query_context"] = query_context
        (workspace_root / ".sandbox_config.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_workspace_python_runtime(
        self,
        workspace_root: Path,
        request: MutationRequest,
    ) -> None:
        write_python_runtime(
            workspace_root,
            policy=self._python_execution_policy(),
            real_python=self._runtime_python_executable(),
            evaluator_dirs=tuple(self._evaluator_dirs(request)),
        )

    def _write_workspace_claude_settings(self, workspace_root: Path) -> None:
        claude_dir = workspace_root / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        hook_command = workspace_hook_command()
        settings = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash|Read|Edit|Write|MultiEdit|Glob|Grep",
                        "hooks": [
                            {
                                "type": "command",
                                "command": hook_command,
                            }
                        ],
                    }
                ],
                "SessionStart": [{"hooks": [{"type": "command", "command": hook_command}]}],
                "UserPromptSubmit": [{"hooks": [{"type": "command", "command": hook_command}]}],
                "PostToolUse": [{"hooks": [{"type": "command", "command": hook_command}]}],
            }
        }
        (claude_dir / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_workspace_codex_hooks(self, workspace_root: Path) -> None:
        if repo_codex_hooks_path().exists():
            return
        write_codex_hooks_file(workspace_root / ".codex" / "hooks.json")

    def _write_workspace_rollout_prompt_config(self, workspace_root: Path) -> None:
        from scaling_evolve.algorithms.eve.rollout_prompts.default import (
            BudgetPrompt,
            PromptContext,
        )

        prompts: list[dict[str, object]] = []
        if self.config.budget_prompt and self.config.rollout_max_turns > 0:
            budget_prompt = BudgetPrompt()
            ctx = PromptContext(
                workspace=workspace_root,
                rollout_max_turns=self.config.rollout_max_turns,
            )
            prompts.append(
                {
                    "name": "budget",
                    "system_text": budget_prompt.system(ctx),
                    "user_text": budget_prompt.user(ctx),
                    "turn_template": budget_prompt.turn_template_source(),
                    "turn_format_kwargs": budget_prompt.turn_format_kwargs(ctx),
                }
            )
        hooks_dir = workspace_root / ".hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "prompts": prompts,
        }
        (hooks_dir / "rollout_prompts.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _python_execution_policy(self) -> PythonExecutionPolicy:
        managed = any(
            value is not None
            for value in (
                self.config.policy_profile,
                self.config.allow_network,
                self.config.allow_subprocess,
                self.config.allowed_env_vars,
            )
        )
        return PythonExecutionPolicy(
            managed=managed,
            allow_network=(
                self.config.allow_network if self.config.allow_network is not None else True
            ),
            allow_subprocess=(
                self.config.allow_subprocess if self.config.allow_subprocess is not None else True
            ),
            allowed_env_vars=tuple(self.config.allowed_env_vars or []),
        )

    def _runtime_python_executable(self) -> Path:
        candidates: list[Path] = []
        virtual_env = os.environ.get("VIRTUAL_ENV")
        if virtual_env:
            candidates.append(Path(virtual_env) / "bin" / "python")
        cwd = Path.cwd().resolve(strict=False)
        for root in (cwd, *cwd.parents):
            candidates.append(root / ".venv" / "bin" / "python")
        which_python = shutil.which("python")
        if which_python:
            candidates.append(Path(which_python))
        candidates.append(Path(sys.executable))

        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve(strict=False)
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.exists():
                return resolved
        return Path(sys.executable).expanduser().resolve(strict=False)

    def _write_workspace_recovery_guidance(
        self,
        workspace_root: Path,
        request: MutationRequest,
    ) -> None:
        if not self._workspace_recovery_guidance_enabled(request):
            return
        guide_path = workspace_root / "CLAUDE.md"
        start_marker = "<!-- scaling-evolve-recovery:start -->"
        end_marker = "<!-- scaling-evolve-recovery:end -->"
        section = "\n".join(
            [
                start_marker,
                "## Eve Recovery",
                "",
                "If you don't remember the current code state, read `./candidate.py`.",
                "If you need evolution history, read `.context_snapshot.json`.",
                end_marker,
                "",
            ]
        )
        existing = guide_path.read_text(encoding="utf-8") if guide_path.exists() else ""
        if start_marker in existing and end_marker in existing:
            prefix, remainder = existing.split(start_marker, maxsplit=1)
            _old_section, suffix = remainder.split(end_marker, maxsplit=1)
            updated = f"{prefix}{section}{suffix.lstrip()}"
        elif existing:
            separator = "\n" if existing.endswith("\n") else "\n\n"
            updated = f"{existing}{separator}{section}"
        else:
            updated = section
        guide_path.write_text(updated, encoding="utf-8")

    def _write_workspace_budget_status(
        self,
        workspace_root: Path,
        request: MutationRequest,
    ) -> None:
        payload = request.metadata.get("budget_status")
        if not isinstance(payload, Mapping):
            return
        budget_status = {str(key): value for key, value in payload.items()}
        (workspace_root / ".budget_status.json").write_text(
            json.dumps(budget_status, indent=2) + "\n",
            encoding="utf-8",
        )

    def _evaluator_dirs(self, request: MutationRequest) -> list[Path]:
        explicit = request.metadata.get("evaluator_dirs")
        if isinstance(explicit, list):
            configured = [
                Path(item).expanduser().resolve(strict=False)
                for item in explicit
                if isinstance(item, str) and item
            ]
            if configured:
                return configured
        return [(Path(__file__).resolve().parents[1] / "applications").resolve(strict=False)]

    def _child_node_id(self, request: MutationRequest) -> str:
        child_node_id = request.metadata.get("child_node_id")
        if isinstance(child_node_id, str) and child_node_id:
            return child_node_id
        return self._edge_id(request.request_id, request)

    def _query_context_payload(self, request: MutationRequest) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {}
        for metadata_key in ("run_id", "archive_name", "child_node_id"):
            value = request.metadata.get(metadata_key)
            if isinstance(value, str) and value:
                payload[metadata_key] = value
        for metadata_key in ("sqlite_path", "config_path"):
            value = request.metadata.get(metadata_key)
            if isinstance(value, str) and value:
                payload[metadata_key] = str(Path(value).expanduser().resolve(strict=False))
        parent_node_id = getattr(request.projected_state, "parent_node_id", None)
        if isinstance(parent_node_id, str) and parent_node_id:
            payload["parent_node_id"] = parent_node_id
        runtime_state = request.projected_state.runtime_state
        if runtime_state is not None:
            compact_count = runtime_state.metadata.get("compact_event_count")
            payload["compact_event_count"] = compact_count if isinstance(compact_count, int) else 0
            last_compact_timestamp = runtime_state.metadata.get("last_compact_timestamp")
            if isinstance(last_compact_timestamp, str) and last_compact_timestamp:
                payload["compact_boundary_timestamp"] = last_compact_timestamp
            last_compact_line_number = runtime_state.metadata.get("last_compact_line_number")
            if isinstance(last_compact_line_number, int) and last_compact_line_number > 0:
                payload["last_compact_line_number"] = last_compact_line_number
        return payload

    def _metadata_workspace(self, request: MutationRequest) -> SessionWorkspaceLease | None:
        for payload in (
            request.projected_state.metadata.get("workspace"),
            request.metadata.get("workspace"),
        ):
            workspace_payload = _as_mapping(payload)
            if workspace_payload is not None:
                return SessionWorkspaceLease.model_validate(workspace_payload)
        return None

    def _working_directory_hint(self, request: MutationRequest) -> str:
        metadata_hint = request.metadata.get("working_directory")
        if isinstance(metadata_hint, str):
            return metadata_hint
        projected_hint = request.projected_state.metadata.get("working_directory")
        if isinstance(projected_hint, str):
            return projected_hint
        return str(Path.cwd())

    def _target_binding(self, request: MutationRequest) -> TargetBinding | None:
        payload = _as_mapping(request.metadata.get("target_binding"))
        if payload is None:
            return None
        return TargetBinding.model_validate(payload)

    def _assessment_contract(self, request: MutationRequest) -> AssessmentContract | None:
        payload = _as_mapping(request.metadata.get("assessment_contract"))
        if payload is None:
            return None
        return AssessmentContract.model_validate(payload)

    def _render_session_instruction(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
        *,
        actual_execution_mode: str,
    ) -> str:
        return self._render_iteration_instruction(
            request,
            actual_execution_mode=actual_execution_mode,
            workspace_root=workspace.session_cwd,
        )

    def _uses_score_feedback_resume(self, request: MutationRequest) -> bool:
        resume_mode = request.metadata.get("session_resume_mode")
        if isinstance(resume_mode, str) and resume_mode == "score_feedback":
            return True
        return self._algorithm_name(request) == "scaling_evolve"

    def _render_score_feedback_instruction(self, request: MutationRequest) -> str:
        return self._render_iteration_instruction(request, actual_execution_mode="fork")

    def _render_iteration_instruction(
        self,
        request: MutationRequest,
        *,
        actual_execution_mode: str,
        workspace_root: str | None = None,
    ) -> str:
        lines: list[str] = []
        parent_score = request.metadata.get("parent_primary_score")
        if isinstance(parent_score, int | float):
            score_display = f"{float(parent_score):.2f}"
            if actual_execution_mode == "fork":
                lines.append(
                    f"Your previous work scored {score_display}. "
                    "Try a different approach to improve it."
                )
            else:
                lines.append(
                    f"The current program scores {score_display}. "
                    "Try a different approach to improve it."
                )
        task_instruction = getattr(request.instruction, "task_instruction", None)
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            task_instruction = request.instruction.content
        if isinstance(task_instruction, str) and task_instruction.strip():
            if lines:
                lines.append("")
            lines.extend(task_instruction.strip().splitlines())
        output_format = getattr(request.instruction, "output_format", None)
        if isinstance(output_format, str) and output_format.strip():
            if lines:
                lines.append("")
            lines.extend(output_format.strip().splitlines())
        workspace_note = getattr(request.instruction, "workspace_note", None)
        if isinstance(workspace_note, str) and workspace_note.strip():
            if lines:
                lines.append("")
            lines.extend(workspace_note.strip().splitlines())
        fork_warning = getattr(request.instruction, "fork_warning", None)
        if (
            actual_execution_mode == "fork"
            and isinstance(fork_warning, str)
            and fork_warning.strip()
        ):
            rendered = fork_warning.strip()
            if workspace_root:
                rendered = rendered.replace("{workspace_path}", workspace_root)
            if lines:
                lines.append("")
            lines.extend(rendered.splitlines())
        return "\n".join(lines).strip() or "Continue improving the candidate."

    def _append_session_constraints(
        self,
        instruction: str,
        request: MutationRequest,
        *,
        actual_execution_mode: str = "spawn",
    ) -> str:
        _ = (request, actual_execution_mode)
        return instruction.strip()

    def _algorithm_name(self, request: MutationRequest) -> str | None:
        algorithm = request.metadata.get("algorithm")
        return algorithm if isinstance(algorithm, str) and algorithm else None

    def _workspace_recovery_guidance_enabled(self, request: MutationRequest) -> bool:
        guidance = request.metadata.get("workspace_recovery_guidance")
        if isinstance(guidance, bool):
            return guidance
        return self._algorithm_name(request) == "scaling_evolve"

    def _materialize_artifact_only_workspace(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
    ) -> None:
        workspace_root = Path(workspace.workspace_root)
        workspace_root.mkdir(parents=True, exist_ok=True)
        for artifact in request.projected_state.artifacts:
            source_path = self._projected_artifact_path(artifact.ref, artifact.path)
            if source_path is None or not source_path.exists():
                continue
            destination = self._normalize_workspace_path(
                artifact.path or str(source_path),
                workspace,
            )
            if source_path.is_dir():
                shutil.copytree(source_path, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)

    def _projected_artifact_path(
        self,
        ref: ArtifactRef | None,
        path: str | None,
    ) -> Path | None:
        if ref is not None and self.artifact_store is not None:
            ref_path = self.artifact_store.path_for(ref)
            if ref_path.exists():
                return ref_path
        if path is None:
            return None
        return Path(path)

    def _execute_native_fork(
        self,
        request: MutationRequest,
        parent_state: RuntimeStateRef,
        workspace: SessionWorkspaceLease,
        acquired_lease: WorkspaceLease | None,
    ) -> tuple[SessionRollout, ArtifactRef | None]:
        parent_workspace_root = parent_state.workspace_root or parent_state.session_cwd
        parent_cwd = parent_state.session_cwd or parent_state.workspace_root
        if parent_workspace_root is None or parent_cwd is None:
            raise RuntimeError("Native session fork requires a parent workspace_root/session_cwd.")
        if Path(parent_workspace_root).resolve(strict=False) == Path(
            workspace.workspace_root
        ).resolve(strict=False):
            raise RuntimeError("Native session fork requires an isolated child workspace.")

        child_session_id = self.driver.fork_session(parent_state)
        _LOGGER.debug(
            "SESSION native fork step: fork_session parent_session=%s child_session=%s "
            "parent_workspace=%s child_workspace=%s",
            parent_state.session_id,
            child_session_id,
            parent_workspace_root,
            workspace.workspace_root,
        )
        self._fork_workspace_contents(parent_workspace_root, workspace, acquired_lease)
        # Reinstall child-specific runtime files because copytree(parent -> child)
        # can overwrite the sandbox/query metadata we wrote before the fork.
        self._prepare_workspace_runtime(request, workspace)
        session_instruction = self._render_session_instruction(
            request,
            workspace,
            actual_execution_mode="fork",
        )
        instruction_text_ref = self._store_session_instruction_text(
            request,
            session_instruction,
        )
        migrated_session_path = self.driver.migrate_session(
            parent_cwd=parent_cwd,
            child_cwd=workspace.session_cwd,
            session_id=child_session_id,
        )
        _LOGGER.debug(
            "SESSION native fork step: migrate_session child_session=%s migrated_path=%s",
            child_session_id,
            migrated_session_path,
        )
        child_state = parent_state.model_copy(
            update={
                "state_id": f"runtime:{child_session_id}",
                "session_id": child_session_id,
                "workspace_id": workspace.workspace_id,
                "target_repo_root": workspace.target_repo_root,
                "workspace_root": workspace.workspace_root,
                "session_cwd": workspace.session_cwd,
                "metadata": {
                    **parent_state.metadata,
                    "migrated_session_path": migrated_session_path,
                },
            }
        )
        _LOGGER.debug(
            "SESSION native fork step: resume child_session=%s workspace_files=%s",
            child_session_id,
            self._workspace_file_inventory(workspace),
        )
        return self.driver.resume(
            child_state, instruction=session_instruction
        ), instruction_text_ref

    def _fork_workspace_contents(
        self,
        source_root: str,
        workspace: SessionWorkspaceLease,
        lease: WorkspaceLease | None,
    ) -> None:
        source_path = Path(source_root).expanduser().resolve()
        destination_root = Path(workspace.workspace_root).expanduser().resolve()
        if source_path == destination_root:
            return
        if lease is not None and self.workspace_manager is not None:
            fork_workspace = getattr(self.workspace_manager, "fork_workspace", None)
            if callable(fork_workspace):
                fork_workspace(source_path, lease)
                return
        if destination_root.exists():
            shutil.rmtree(destination_root, ignore_errors=True)
        shutil.copytree(source_path, destination_root)

    def _materialize_reference_programs(
        self,
        projected_state: ProjectedState,
        workspace: SessionWorkspaceLease,
    ) -> dict[str, list[str]]:
        reference_paths = {
            "top_programs": self._reference_program_group_paths(projected_state.top_programs),
            "diverse_programs": self._reference_program_group_paths(
                projected_state.diverse_programs
            ),
            "inspiration_programs": self._reference_program_group_paths(
                projected_state.inspiration_programs
            ),
        }
        _LOGGER.debug(
            "SESSION reference programs: workspace=%s top=%s diverse=%s inspiration=%s",
            workspace.workspace_root,
            reference_paths["top_programs"],
            reference_paths["diverse_programs"],
            reference_paths["inspiration_programs"],
        )
        return reference_paths

    def _reference_program_group_paths(
        self,
        programs: list[ProjectedProgram],
    ) -> list[str]:
        reference_paths: list[str] = []
        for program in programs:
            if not isinstance(program.path, str) or not program.path:
                raise ValueError(
                    f"Projected reference program `{program.node_id}` is missing a filesystem path."
                )
            reference_paths.append(str(Path(program.path).expanduser().resolve(strict=False)))
        return reference_paths

    def _store_session_instruction_text(
        self,
        request: MutationRequest,
        session_instruction: str,
    ) -> ArtifactRef | None:
        artifact_store = self._artifact_store_for_request(request)
        if artifact_store is None:
            return None
        edge_id = self._edge_id(request.request_id, request)
        return artifact_store.put_text(
            ArtifactKind.INSTRUCTION_TEXT,
            session_instruction,
            filename=f"{edge_id}.session_instruction.txt",
            edge_id=edge_id,
        )

    def _instruction_candidate_path(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
        *,
        actual_execution_mode: str,
    ) -> str | None:
        primary_path = self._fallback_primary_path(
            request,
            workspace,
            actual_execution_mode=actual_execution_mode,
        )
        if primary_path is not None:
            return primary_path
        candidate_paths = self._fallback_changed_paths(
            request,
            workspace,
            actual_execution_mode=actual_execution_mode,
        )
        if candidate_paths:
            return candidate_paths[0]
        return None

    def _workspace_relative_path(
        self,
        path: str,
        workspace: SessionWorkspaceLease | Path,
    ) -> str:
        workspace_root = (
            Path(workspace.workspace_root)
            if isinstance(workspace, SessionWorkspaceLease)
            else workspace
        )
        relative = Path(path).relative_to(workspace_root)
        return f"./{relative.as_posix()}"

    def _workspace_file_inventory(self, workspace: SessionWorkspaceLease) -> list[str]:
        workspace_root = Path(workspace.workspace_root)
        if not workspace_root.exists():
            return []
        return sorted(
            str(path.relative_to(workspace_root))
            for path in workspace_root.rglob("*")
            if path.is_file()
        )

    def _can_native_fork(
        self,
        parent_state: RuntimeStateRef | None,
        workspace: SessionWorkspaceLease,
    ) -> bool:
        if parent_state is None or self.config.fork_mode != "native":
            return False
        capabilities = self.driver.capabilities()
        if not capabilities.supports_native_fork:
            return False
        parent_workspace = parent_state.workspace_root or parent_state.session_cwd
        if parent_workspace is None:
            return True
        if parent_workspace == workspace.workspace_root:
            return False
        return capabilities.supports_cross_workspace_fork

    def _resolve_transcript(self, rollout: SessionRollout) -> ArtifactRef | None:
        return rollout.transcript

    def _resolve_session_workspace(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
    ) -> SessionWorkspaceLease:
        workspace_root = Path(workspace.workspace_root).expanduser().resolve(strict=False)
        session_cwd = Path(workspace.session_cwd).expanduser().resolve(strict=False)
        if workspace_root != session_cwd:
            return workspace.model_copy(update={"session_cwd": str(session_cwd)})
        if not self._uses_stable_session_cwd(request):
            return workspace
        # Scaling Evolve keeps a lineage on a stable session slot so forks can
        # resume with the same conversational/cache state even as each child keeps
        # its own materialized workspace_root for artifact persistence.
        parent_state = request.projected_state.runtime_state
        if parent_state is not None:
            parent_session_cwd = parent_state.session_cwd
            parent_workspace_root = parent_state.workspace_root
            if (
                isinstance(parent_session_cwd, str)
                and parent_session_cwd
                and parent_session_cwd != parent_workspace_root
            ):
                return workspace.model_copy(update={"session_cwd": parent_session_cwd})
        slot_root = workspace_root.parent / ".session-slots" / workspace.workspace_id
        return workspace.model_copy(update={"session_cwd": str(slot_root)})

    def _uses_stable_session_cwd(self, request: MutationRequest) -> bool:
        explicit = request.metadata.get("stable_session_cwd")
        if isinstance(explicit, bool):
            return explicit
        projected = request.projected_state.metadata.get("stable_session_cwd")
        if isinstance(projected, bool):
            return projected
        return self._algorithm_name(request) == "scaling_evolve"

    def _sync_session_changes_to_workspace(
        self,
        request: MutationRequest,
        rollout: SessionRollout,
        workspace: SessionWorkspaceLease,
    ) -> None:
        workspace_root = Path(workspace.workspace_root).expanduser().resolve(strict=False)
        session_cwd = Path(workspace.session_cwd).expanduser().resolve(strict=False)
        if workspace_root == session_cwd:
            return
        raw_paths: list[str] = []
        if rollout.primary_path is not None:
            raw_paths.append(rollout.primary_path)
        raw_paths.extend(path for path in rollout.changed_paths if path not in raw_paths)
        if not raw_paths:
            raw_paths.extend(self._projected_workspace_paths(request, workspace))
        for raw_path in raw_paths:
            source = self._normalize_session_cwd_path(raw_path, workspace)
            destination = self._normalize_workspace_path(raw_path, workspace)
            if not source.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)

    def _normalize_session_cwd_path(
        self,
        path: str,
        workspace: SessionWorkspaceLease,
    ) -> Path:
        candidate = Path(path)
        session_cwd = Path(workspace.session_cwd)
        workspace_root = Path(workspace.workspace_root)
        if candidate.is_absolute():
            with suppress(ValueError):
                candidate.relative_to(session_cwd)
                return candidate
            with suppress(ValueError):
                return session_cwd / candidate.relative_to(workspace_root)
            return session_cwd / candidate.name
        return session_cwd / candidate

    def _persist_provider_session_archive(
        self,
        request: MutationRequest,
        rollout: SessionRollout,
    ) -> ArtifactRef | None:
        artifact_store = self._artifact_store_for_request(request)
        if artifact_store is None:
            return None
        raw_source_path = rollout.state.metadata.get("provider_transcript_path")
        if not isinstance(raw_source_path, str) or raw_source_path == "":
            return None
        source_path = Path(raw_source_path).expanduser()
        if not source_path.exists():
            _LOGGER.warning("Claude transcript archive missing at %s", source_path)
            return None
        try:
            return artifact_store.put_bytes(
                ArtifactKind.SESSION_ARCHIVE_JSONL,
                source_path.read_bytes(),
                filename=source_path.name,
                edge_id=self._edge_id(request.request_id, request),
                metadata={"source_path": str(source_path)},
            )
        except Exception as error:
            _LOGGER.warning("Failed to archive Claude transcript %s: %s", source_path, error)
            return None

    def _persist_driver_debug_artifacts(
        self,
        request: MutationRequest,
        rollout: SessionRollout,
    ) -> tuple[SessionRollout, list[ArtifactRef]]:
        artifact_store = self._artifact_store_for_request(request)
        metadata = dict(rollout.state.metadata)
        execution_payload = _as_mapping(metadata.get("driver_execution"))
        raw_stdout = metadata.pop("driver_stdout", None)
        raw_stderr = metadata.pop("driver_stderr", None)
        result_payload = _as_mapping(metadata.pop("driver_result", None))
        if execution_payload is None:
            sanitized_state = rollout.state.model_copy(update={"metadata": metadata})
            return rollout.model_copy(update={"state": sanitized_state}), []

        execution_metadata: dict[str, JSONValue] = dict(execution_payload)
        artifact_refs: list[ArtifactRef] = []
        edge_id = self._edge_id(request.request_id, request)

        if artifact_store is not None:
            if isinstance(raw_stdout, str):
                stdout_ref = artifact_store.put_text(
                    ArtifactKind.MODEL_RESPONSE_RAW_JSON,
                    raw_stdout,
                    filename=f"{edge_id}.claude_code.stdout.jsonl",
                    edge_id=edge_id,
                    metadata={
                        "driver": "claude_code",
                        "command": execution_metadata.get("command"),
                        "cwd": execution_metadata.get("cwd"),
                        "exit_code": execution_metadata.get("exit_code"),
                    },
                )
                artifact_refs.append(stdout_ref)
                execution_metadata["stdout_ref"] = stdout_ref.model_dump(mode="json")
            if result_payload is not None:
                result_ref = artifact_store.put_json(
                    ArtifactKind.MODEL_RESPONSE_PARSED_JSON,
                    dict(result_payload),
                    filename=f"{edge_id}.claude_code.result.json",
                    edge_id=edge_id,
                    metadata={"driver": "claude_code"},
                )
                artifact_refs.append(result_ref)
                execution_metadata["result_ref"] = result_ref.model_dump(mode="json")
            if isinstance(raw_stderr, str) and raw_stderr:
                stderr_ref = artifact_store.put_text(
                    ArtifactKind.FAILURE_SUMMARY_TXT,
                    raw_stderr,
                    filename=f"{edge_id}.claude_code.stderr.txt",
                    edge_id=edge_id,
                    metadata={"driver": "claude_code"},
                )
                artifact_refs.append(stderr_ref)
                execution_metadata["stderr_ref"] = stderr_ref.model_dump(mode="json")

        metadata["driver_execution"] = execution_metadata
        sanitized_state = rollout.state.model_copy(update={"metadata": metadata})
        return rollout.model_copy(update={"state": sanitized_state}), artifact_refs

    def _resolve_changed_files_manifest(
        self,
        request: MutationRequest,
        rollout: SessionRollout,
        workspace: SessionWorkspaceLease,
        *,
        actual_execution_mode: str,
    ) -> tuple[ArtifactRef | None, str, ChangedFilesManifest]:
        artifact_store = self._artifact_store_for_request(request)
        driver_paths = self._driver_changed_paths(rollout, workspace)
        has_driver_data = rollout.changed_files_manifest is not None or bool(driver_paths)
        manifest_source = "driver" if has_driver_data else "projection_fallback"
        changed_paths = driver_paths
        if not changed_paths:
            changed_paths = self._fallback_changed_paths(
                request,
                workspace,
                actual_execution_mode=actual_execution_mode,
            )
        primary_path = self._driver_primary_path(rollout, workspace) or (
            self._fallback_primary_path(
                request,
                workspace,
                actual_execution_mode=actual_execution_mode,
            )
            or (changed_paths[0] if changed_paths else None)
        )
        manifest = ChangedFilesManifest(
            files=list(changed_paths),
            primary_path=primary_path,
            changed_paths=list(changed_paths),
            workspace_strategy=workspace.strategy,
            manifest_source=manifest_source,
        )
        if artifact_store is None:
            if rollout.changed_files_manifest is not None:
                return rollout.changed_files_manifest, manifest_source, manifest
            return None, manifest_source, manifest
        filename = f"{self._edge_id(request.request_id, request)}.changed_files.manifest.json"
        return (
            artifact_store.put_json(
                ArtifactKind.CHANGED_FILES_MANIFEST_JSON,
                manifest.model_dump(mode="json"),
                filename=filename,
                edge_id=self._edge_id(request.request_id, request),
            ),
            manifest_source,
            manifest,
        )

    def _resolve_transcript_digest(
        self,
        request: MutationRequest,
        rollout: SessionRollout,
        snapshot: SessionSnapshot | None,
        transcript_ref: ArtifactRef | None,
    ) -> ArtifactRef | None:
        artifact_store = self._artifact_store_for_request(request)
        if snapshot is not None and snapshot.transcript_digest is not None:
            return snapshot.transcript_digest
        if transcript_ref is None or artifact_store is None:
            return None
        try:
            transcript_text = artifact_store.read_text(transcript_ref)
        except Exception:
            return None
        digest = build_transcript_digest(transcript_text)
        edge_id = self._edge_id(rollout.state.state_id, None)
        return artifact_store.put_json(
            ArtifactKind.TRANSCRIPT_DIGEST_JSON,
            digest,
            filename=f"{edge_id}.transcript_digest.json",
            edge_id=edge_id,
        )

    def _resolve_portable_state(
        self,
        request: MutationRequest,
        snapshot: SessionSnapshot | None,
        transcript_digest_ref: ArtifactRef | None,
        rollout_summary: str | None = None,
    ) -> PortableStateRef | None:
        artifact_store = self._artifact_store_for_request(request)
        if snapshot is not None and snapshot.portable_state is not None:
            if transcript_digest_ref is None:
                return snapshot.portable_state
            return snapshot.portable_state.model_copy(
                update={"transcript_digest_ref": transcript_digest_ref}
            )

        if artifact_store is None:
            if request.projected_state.portable_state is None and transcript_digest_ref is None:
                return None
            return PortableStateRef(
                state_id=f"portable:{request.request_id}",
                summary=rollout_summary or request.projected_state.summary,
                transcript_digest_ref=transcript_digest_ref,
            )

        lineage_summary_ref = artifact_store.put_json(
            ArtifactKind.LINEAGE_SUMMARY_JSON,
            build_lineage_summary(request.projected_state),
            filename=f"{self._edge_id(request.request_id, request)}.portable_lineage.json",
            edge_id=self._edge_id(request.request_id, request),
        )
        return PortableStateRef(
            state_id=f"portable:{request.request_id}",
            summary=rollout_summary or request.projected_state.summary,
            lineage_summary_ref=lineage_summary_ref,
            transcript_digest_ref=transcript_digest_ref,
            artifact=lineage_summary_ref,
        )

    def _resolve_child_materialization(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
        changed_files_ref: ArtifactRef | None,
        changed_manifest: ChangedFilesManifest | None,
    ) -> MaterializationRef | None:
        manifest = changed_manifest or ChangedFilesManifest(
            primary_path=None,
            changed_paths=self._projected_workspace_paths(request, workspace),
        )
        changed_paths = [path for path in manifest.changed_paths if Path(path).exists()]
        primary_path = manifest.primary_path
        if primary_path is None and changed_paths:
            primary_path = changed_paths[0]
        if primary_path is None or not Path(primary_path).exists():
            return None
        if primary_path not in changed_paths:
            changed_paths.insert(0, primary_path)
        return MaterializationRef(
            manifest_artifact=changed_files_ref,
            location=primary_path,
            metadata={
                "primary_path": primary_path,
                "changed_paths": list(changed_paths),
                "persist_from_workspace": True,
                "workspace_strategy": workspace.strategy,
                "workspace_root": workspace.workspace_root,
            },
        )

    def _driver_changed_paths(
        self,
        rollout: SessionRollout,
        workspace: SessionWorkspaceLease,
    ) -> list[str]:
        manifest = self._driver_changed_manifest(rollout)
        raw_paths = list(rollout.changed_paths)
        if not raw_paths and manifest is not None:
            raw_paths = list(manifest.changed_paths)
        return self._normalize_workspace_paths(raw_paths, workspace)

    def _driver_primary_path(
        self,
        rollout: SessionRollout,
        workspace: SessionWorkspaceLease,
    ) -> str | None:
        manifest = self._driver_changed_manifest(rollout)
        raw_path = rollout.primary_path or (manifest.primary_path if manifest is not None else None)
        if raw_path is None:
            return None
        return str(self._normalize_workspace_path(raw_path, workspace))

    def _driver_changed_manifest(self, rollout: SessionRollout) -> ChangedFilesManifest | None:
        if rollout.changed_files_manifest is None or self.artifact_store is None:
            return None
        try:
            return ChangedFilesManifest.model_validate(
                self.artifact_store.read_json(rollout.changed_files_manifest)
            )
        except Exception:
            return None

    def _projected_workspace_paths(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
    ) -> list[str]:
        paths: list[str] = []
        for artifact in request.projected_state.artifacts:
            source_path = self._projected_artifact_path(artifact.ref, artifact.path)
            if source_path is None:
                continue
            paths.append(
                str(
                    self._normalize_workspace_path(
                        artifact.path or str(source_path),
                        workspace,
                    )
                )
            )
        return paths

    def _fallback_changed_paths(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
        *,
        actual_execution_mode: str,
    ) -> list[str]:
        if actual_execution_mode == "fork":
            fork_paths = self._fork_rebased_parent_paths(request, workspace)
            if fork_paths:
                return fork_paths
        return self._projected_workspace_paths(request, workspace)

    def _fallback_primary_path(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
        *,
        actual_execution_mode: str,
    ) -> str | None:
        if actual_execution_mode == "fork":
            return self._fork_rebased_parent_primary_path(request, workspace)
        projected_paths = self._projected_workspace_paths(request, workspace)
        if projected_paths:
            return projected_paths[0]
        return None

    def _fork_rebased_parent_paths(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
    ) -> list[str]:
        parent_state = request.projected_state.runtime_state
        if parent_state is None:
            return []
        parent_manifest = self._runtime_state_changed_manifest(parent_state)
        if parent_manifest is None:
            return []
        rebased = [
            self._rebase_parent_workspace_path(path, parent_state, workspace)
            for path in parent_manifest.changed_paths
        ]
        return self._normalize_workspace_paths(rebased, workspace)

    def _fork_rebased_parent_primary_path(
        self,
        request: MutationRequest,
        workspace: SessionWorkspaceLease,
    ) -> str | None:
        parent_state = request.projected_state.runtime_state
        if parent_state is None:
            return None
        parent_manifest = self._runtime_state_changed_manifest(parent_state)
        if parent_manifest is None or parent_manifest.primary_path is None:
            return None
        rebased = self._rebase_parent_workspace_path(
            parent_manifest.primary_path,
            parent_state,
            workspace,
        )
        candidate = Path(rebased)
        if candidate.exists():
            return rebased
        return None

    def _runtime_state_changed_manifest(
        self,
        runtime_state: RuntimeStateRef,
    ) -> ChangedFilesManifest | None:
        payload = _as_mapping(runtime_state.metadata.get("changed_files_manifest_ref"))
        if payload is None or self.artifact_store is None:
            return None
        try:
            ref = ArtifactRef.model_validate(dict(payload))
            return ChangedFilesManifest.model_validate(self.artifact_store.read_json(ref))
        except Exception:
            return None

    def _artifact_store_for_request(self, request: MutationRequest) -> FSArtifactStore | None:
        if self.artifact_store is None:
            return None
        if bool(request.metadata.get("defer_lineage_registration")):
            return self.artifact_store.without_lineage_registration()
        return self.artifact_store

    def _rebase_parent_workspace_path(
        self,
        path: str,
        parent_state: RuntimeStateRef,
        workspace: SessionWorkspaceLease,
    ) -> str:
        candidate = Path(path)
        parent_root_value = parent_state.workspace_root or parent_state.session_cwd
        child_root = Path(workspace.workspace_root)
        if parent_root_value is not None:
            parent_root = Path(parent_root_value)
            if candidate.is_absolute():
                with suppress(ValueError):
                    relative = candidate.relative_to(parent_root)
                    return str(child_root / relative)
            return str(child_root / candidate)
        if candidate.is_absolute():
            return str(child_root / candidate.name)
        return str(child_root / candidate)

    def _normalize_workspace_paths(
        self,
        paths: list[str],
        workspace: SessionWorkspaceLease,
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(self._normalize_workspace_path(path, workspace))
            if resolved in seen:
                continue
            seen.add(resolved)
            normalized.append(resolved)
        return normalized

    def _normalize_workspace_path(
        self,
        path: str,
        workspace: SessionWorkspaceLease,
    ) -> Path:
        candidate = Path(path)
        workspace_root = Path(workspace.workspace_root)
        if candidate.is_absolute():
            with suppress(ValueError):
                candidate.relative_to(workspace_root)
                return candidate
            session_cwd = Path(workspace.session_cwd)
            with suppress(ValueError):
                return workspace_root / candidate.relative_to(session_cwd)
            target_repo_root = Path(workspace.target_repo_root)
            with suppress(ValueError):
                return workspace_root / candidate.relative_to(target_repo_root)
            return workspace_root / candidate.name
        return workspace_root / candidate

    def retain_runtime_state(
        self,
        state: RuntimeStateRef,
        *,
        workspace: SessionWorkspaceLease | None = None,
        lease: WorkspaceLease | None = None,
        run_id: str | None = None,
        owner_node_id: str | None = None,
        purpose: str | None = None,
        lifecycle: ExecutionLifecycle = ExecutionLifecycle.RUNNING,
    ) -> RuntimeStateRef:
        retained_lease = self._lease_for_retention(
            state,
            workspace=workspace,
            lease=lease,
            owner_node_id=owner_node_id,
            purpose=purpose,
        )
        if retained_lease is not None:
            retained_lease = retained_lease.model_copy(
                update={
                    "created_at": retained_lease.created_at or utc_now(),
                }
            )
            self._retained_leases[retained_lease.workspace_id] = retained_lease
        metadata = dict(state.metadata)
        return state.model_copy(
            update={
                "lifecycle": lifecycle,
                "workspace_id": state.workspace_id
                or (workspace.workspace_id if workspace else None),
                "target_repo_root": (
                    state.target_repo_root or (workspace.target_repo_root if workspace else None)
                ),
                "workspace_root": (
                    state.workspace_root or (workspace.workspace_root if workspace else None)
                ),
                "session_cwd": state.session_cwd or (workspace.session_cwd if workspace else None),
                "lease_owner_run_id": run_id or state.lease_owner_run_id,
                "lease_owner_node_id": owner_node_id or state.lease_owner_node_id,
                "lease_purpose": purpose or state.lease_purpose or "mutation",
                "lease_created_at": (
                    self._timestamp_text(retained_lease.created_at)
                    if retained_lease is not None
                    else state.lease_created_at
                ),
                "lease_detached_at": None,
                "lease_released_at": None,
                "metadata": metadata,
            }
        )

    def detach_runtime_state(
        self,
        state: RuntimeStateRef,
        *,
        lifecycle: ExecutionLifecycle = ExecutionLifecycle.DETACHED,
    ) -> RuntimeStateRef:
        if state.workspace_id is not None:
            self._retained_leases.pop(state.workspace_id, None)
        return state.model_copy(
            update={
                "lifecycle": lifecycle,
                "lease_detached_at": utc_now().isoformat(),
            }
        )

    def release_runtime_state(
        self,
        state: RuntimeStateRef,
        *,
        final_lifecycle: ExecutionLifecycle = ExecutionLifecycle.COMPLETED,
        snapshot: bool = False,
    ) -> RuntimeStateRef:
        if state.workspace_id is not None:
            self._retained_leases.pop(state.workspace_id, None)
        lease = self._lease_for_retention(state)
        if lease is not None:
            self._release_workspace_lease(lease, snapshot=snapshot)
        return state.model_copy(
            update={
                "lifecycle": final_lifecycle,
                "lease_released_at": utc_now().isoformat(),
            }
        )

    def close(self) -> None:
        leases = list(self._retained_leases.values())
        self._retained_leases.clear()
        for lease in leases:
            try:
                self._release_workspace_lease(lease, snapshot=True)
            except Exception:
                continue

    def _enrich_runtime_state(
        self,
        state: RuntimeStateRef,
        *,
        workspace: SessionWorkspaceLease,
        transcript_ref: ArtifactRef | None,
        session_archive_ref: ArtifactRef | None,
        changed_files_ref: ArtifactRef | None,
        manifest_source: str,
        requested_inheritance_mode: InheritanceMode,
        actual_execution_mode: str,
        actual_workspace_inheritance: str,
        fallback_reason: str | None,
        run_id: str,
        owner_node_id: str | None,
    ) -> RuntimeStateRef:
        metadata = dict(state.metadata)
        metadata.update(self._compact_runtime_metadata(metadata))
        if transcript_ref is not None:
            metadata["transcript_ref"] = transcript_ref.model_dump(mode="json")
        if session_archive_ref is not None:
            metadata["session_archive_ref"] = session_archive_ref.model_dump(mode="json")
        if changed_files_ref is not None:
            metadata["changed_files_manifest_ref"] = changed_files_ref.model_dump(mode="json")
        metadata["manifest_source"] = manifest_source
        metadata["requested_inheritance_mode"] = requested_inheritance_mode.value
        metadata["actual_execution_mode"] = actual_execution_mode
        metadata["actual_workspace_inheritance"] = actual_workspace_inheritance
        if fallback_reason is not None:
            metadata["fallback_reason"] = fallback_reason
        if workspace.strategy is not None:
            metadata["workspace_strategy"] = workspace.strategy
        metadata["budget_rollout_max_turns"] = self.config.rollout_max_turns
        return state.model_copy(
            update={
                "lifecycle": ExecutionLifecycle.RUNNING,
                "workspace_id": state.workspace_id or workspace.workspace_id,
                "target_repo_root": state.target_repo_root or workspace.target_repo_root,
                "workspace_root": state.workspace_root or workspace.workspace_root,
                "session_cwd": state.session_cwd or workspace.session_cwd,
                "lease_owner_run_id": run_id,
                "lease_owner_node_id": owner_node_id,
                "lease_purpose": state.lease_purpose or "mutation",
                "metadata": metadata,
            }
        )

    def _compact_runtime_metadata(self, metadata: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        transcript_path = metadata.get("provider_transcript_path")
        if not isinstance(transcript_path, str) or not transcript_path:
            return {"compact_event_count": 0}
        path = Path(transcript_path).expanduser()
        if not path.exists():
            _LOGGER.warning("Compact transcript missing at %s", path)
            return {"compact_event_count": 0}
        try:
            return compact_metadata_from_transcript(path)
        except Exception as error:
            _LOGGER.warning("Failed to parse compact events from %s: %s", path, error)
            return {"compact_event_count": 0}

    def _edge_id(self, fallback: str, request: MutationRequest | None) -> str:
        if request is not None:
            edge_id = request.metadata.get("edge_id")
            if isinstance(edge_id, str):
                return edge_id
        if ":" in fallback:
            return fallback.replace(":", "_")
        return fallback

    def _run_id(self, request: MutationRequest) -> str:
        run_id = request.metadata.get("run_id")
        if isinstance(run_id, str):
            return run_id
        return "run"

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in cast(list[object], value) if isinstance(item, str) and item]

    def _requested_inheritance_mode(self, request: MutationRequest) -> InheritanceMode:
        raw_mode = request.metadata.get("requested_inheritance_mode")
        if raw_mode == InheritanceMode.NATIVE.value:
            return InheritanceMode.NATIVE
        return InheritanceMode.SUMMARY_ONLY

    def _lease_for_retention(
        self,
        state: RuntimeStateRef,
        *,
        workspace: SessionWorkspaceLease | None = None,
        lease: WorkspaceLease | None = None,
        owner_node_id: str | None = None,
        purpose: str | None = None,
    ) -> WorkspaceLease | None:
        if lease is not None:
            return lease
        workspace_id = state.workspace_id or (
            workspace.workspace_id if workspace is not None else None
        )
        workspace_root = state.workspace_root or (
            workspace.workspace_root if workspace is not None else None
        )
        session_cwd = state.session_cwd or (
            workspace.session_cwd if workspace is not None else None
        )
        if workspace_id is None or workspace_root is None or session_cwd is None:
            return None
        return WorkspaceLease(
            workspace_id=workspace_id,
            root=workspace_root,
            owner_node_id=owner_node_id or state.lease_owner_node_id,
            purpose=purpose or state.lease_purpose,
            target_repo_root=state.target_repo_root
            or (workspace.target_repo_root if workspace is not None else None),
            workspace_root=workspace_root,
            session_cwd=session_cwd,
            created_at=utc_now(),
        )

    def _release_workspace_lease(self, lease: WorkspaceLease, *, snapshot: bool = False) -> None:
        if self.workspace_manager is None:
            return
        self.workspace_manager.release(lease, snapshot=snapshot)

    def _timestamp_text(self, value: object) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        return value if isinstance(value, str) else None
