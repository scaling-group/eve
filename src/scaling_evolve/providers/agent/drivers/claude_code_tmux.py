"""Interactive Claude Code tmux-backed session driver."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from scaling_evolve.core.engine import RuntimeStateRef
from scaling_evolve.core.mutation import ProviderUsage
from scaling_evolve.providers.agent.drivers._metadata import (
    TokenPricing,
    build_driver_execution_metadata,
    compute_cost,
    resolve_token_pricing,
)
from scaling_evolve.providers.agent.drivers._transcript import (
    archive_subagent_transcripts,
    archive_transcript,
)
from scaling_evolve.providers.agent.drivers._workspace import (
    changed_paths_from_tree,
    diff_patch_from_tree,
    read_workspace_tree,
)
from scaling_evolve.providers.agent.drivers.base import (
    SessionDriver,
    SessionDriverCapabilities,
    SessionRollout,
    SessionSeed,
    SessionSnapshot,
)
from scaling_evolve.providers.agent.drivers.codex_tmux import (
    CodexTmuxPanePool,
    _attempt_label,
    _display_value,
    _git_head,
    _jsonl_line_count,
    _load_json_line,
    _mapping,
    _string_metadata,
)
from scaling_evolve.providers.agent.tmux_runtime import (
    capture_pane_tail,
    kill_pane,
    launch_in_pane,
    pane_dead,
    pane_has_active_task,
    wait_for_pane_reset,
)

_DEFAULT_INSTRUCTION_ENTRYPOINT = (
    "Read {instruction_filename} in the current directory and follow it exactly."
)
_SIGNAL_FILENAME = ".claude-task-stopped"
_WORKSPACE_EXCLUDE_FILES = {_SIGNAL_FILENAME, ".claude-stop-hook.log"}


@dataclass(frozen=True)
class ClaudeCodeCompletionState:
    """Observed Claude-side completion state for one rollout turn."""

    session_jsonl: Path | None
    session_id: str | None
    session_jsonl_line_count: int
    rollout_max_turns_reached: bool = False
    observed_turns: int = 0


class ClaudeCodeTmuxSessionDriver(SessionDriver):
    """Interactive Claude Code driver backed by one pane or a shared pane pool."""

    def __init__(
        self,
        *,
        pane_id: str | None = None,
        pane_pool: CodexTmuxPanePool | None = None,
        run_root: str | Path,
        executable: str = "claude",
        model: str | None = None,
        effort_level: str | None = None,
        rollout_max_turns: int = 200,
        budget_prompt: bool = True,
        timeout_seconds: float = 900.0,
        instruction_filename: str = ".evolve-instruction.md",
        transcript_dirname: str = ".claude-driver-transcripts",
        role: str | None = None,
        setting_sources: tuple[str, ...] = ("project", "local"),
        disallowed_tools: tuple[str, ...] = (),
        dangerously_skip_permissions: bool = True,
        token_pricing: TokenPricing | None = None,
        pricing_table: Mapping[str, TokenPricing] | None = None,
        owns_pool: bool = False,
    ) -> None:
        if (pane_id is None) == (pane_pool is None):
            raise ValueError("Provide exactly one of pane_id or pane_pool.")
        self.pane_id = pane_id
        self.pane_pool = pane_pool
        self.run_root = Path(run_root).expanduser().resolve()
        self.executable = executable
        self.model = model
        self.effort_level = effort_level
        self.rollout_max_turns = rollout_max_turns
        self.budget_prompt = budget_prompt
        self.timeout_seconds = timeout_seconds
        self.instruction_filename = instruction_filename
        self.transcript_dirname = transcript_dirname
        self.role = role
        self.setting_sources = tuple(source for source in setting_sources if source)
        self.disallowed_tools = tuple(tool for tool in disallowed_tools if tool)
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.token_pricing = resolve_token_pricing(model, token_pricing, pricing_table)
        self.owns_pool = owns_pool

    def capabilities(self) -> SessionDriverCapabilities:
        return SessionDriverCapabilities(
            supports_native_fork=False,
            supports_cross_workspace_fork=False,
        )

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        workspace = seed.workspace
        if workspace is None:
            raise ValueError("ClaudeCodeTmuxSessionDriver requires a resolved workspace lease.")
        worktree_root = Path(workspace.session_cwd).resolve()
        return self._run_rollout(
            instruction=seed.instruction,
            worktree_root=worktree_root,
            workspace_id=workspace.workspace_id,
            target_repo_root=workspace.target_repo_root,
            workspace_root=workspace.workspace_root,
            session_cwd=workspace.session_cwd,
            session_id=None,
            state_id=f"runtime:{uuid4().hex}",
            existing_session_jsonl=None,
            line_count_before_launch=0,
            metadata={
                **dict(seed.display_context),
                "prompt_file": seed.prompt_file,
                "write_prompt_file": seed.write_prompt_file,
            },
        )

    def fork_session(self, parent: RuntimeStateRef) -> str:
        _ = parent
        raise NotImplementedError("claude_code_tmux is spawn-only; native fork is unsupported.")

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        _ = (parent_cwd, child_cwd, session_id)
        raise NotImplementedError("claude_code_tmux is spawn-only; migrate_session is unsupported.")

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        _ = (parent, instruction)
        raise NotImplementedError("claude_code_tmux is spawn-only; fork is unsupported.")

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        if not state.session_id:
            raise ValueError("claude_code_tmux resume requires a session_id.")

        worktree_root = Path(state.session_cwd or state.workspace_root or ".").resolve()
        existing_session_jsonl = self._resolve_live_session_path(
            worktree_root=worktree_root,
            session_id=state.session_id,
            metadata=state.metadata,
        )
        line_count_before_launch = _jsonl_line_count(existing_session_jsonl)
        return self._run_rollout(
            instruction=instruction or "continue",
            worktree_root=worktree_root,
            workspace_id=state.workspace_id,
            target_repo_root=state.target_repo_root,
            workspace_root=state.workspace_root,
            session_cwd=str(worktree_root),
            session_id=state.session_id,
            state_id=state.state_id,
            existing_session_jsonl=existing_session_jsonl,
            line_count_before_launch=line_count_before_launch,
            metadata=dict(state.metadata),
        )

    def snapshot(self, state: RuntimeStateRef) -> SessionSnapshot:
        _ = state
        raise NotImplementedError("claude_code_tmux is spawn-only; snapshot is unsupported.")

    def close(self) -> None:
        if self.owns_pool and self.pane_pool is not None:
            self.pane_pool.close()

    def _run_rollout(
        self,
        *,
        instruction: str,
        worktree_root: Path,
        workspace_id: str | None,
        target_repo_root: str | None,
        workspace_root: str | None,
        session_cwd: str | None,
        session_id: str | None,
        state_id: str,
        existing_session_jsonl: Path | None,
        line_count_before_launch: int,
        metadata: dict[str, object],
    ) -> SessionRollout:
        prompt_file = _string_metadata(metadata.get("prompt_file"))
        write_prompt_file = bool(metadata.get("write_prompt_file", True))
        instruction_path: Path | None = None
        if prompt_file is not None:
            instruction_path = worktree_root / prompt_file
        if write_prompt_file and instruction_path is not None:
            instruction_path.write_text(self._instruction_document(instruction), encoding="utf-8")
        elif instruction_path is not None and not instruction_path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {instruction_path}")

        before_tree = read_workspace_tree(
            worktree_root,
            exclude_dirs=self._workspace_exclude_dirs(),
            exclude_files=_WORKSPACE_EXCLUDE_FILES,
        )
        initial_head = _git_head(worktree_root)
        launch_started_ns = time.time_ns()
        signal_file = worktree_root / _SIGNAL_FILENAME
        signal_file.unlink(missing_ok=True)
        argv = self._argv(
            session_id=session_id,
            instruction=instruction,
            prompt_file=prompt_file,
        )
        pane_title, banner_lines = _visual_phase_metadata_cc(
            workspace_id=workspace_id,
            instruction_filename=prompt_file or "inline-prompt",
            role=self.role,
            display_context=metadata,
        )
        transcript_root = worktree_root / self.transcript_dirname
        transcript_root.mkdir(parents=True, exist_ok=True)
        active_pane_id = self._acquire_pane(
            preferred_pane_id=_string_metadata(metadata.get("pane_id"))
        )
        metadata["pane_id"] = active_pane_id
        if self.pane_pool is not None:
            metadata["pane_pool_session_name"] = self.pane_pool.session_name
        try:
            env = {"HOME": str(Path.home())}
            if self.effort_level is not None:
                env["CLAUDE_CODE_EFFORT_LEVEL"] = self.effort_level
            launch_in_pane(
                pane_id=active_pane_id,
                cwd=worktree_root,
                env=env,
                argv=argv,
                pane_title=pane_title,
                banner_lines=banner_lines,
            )
            completion = self._wait_for_completion(
                worktree_root=worktree_root,
                existing_session_jsonl=existing_session_jsonl,
                line_count_before_launch=line_count_before_launch,
                launch_started_ns=launch_started_ns,
                pane_id=active_pane_id,
            )
        finally:
            try:
                kill_pane(active_pane_id)
                wait_for_pane_reset(active_pane_id)
                if self.pane_pool is not None:
                    self.pane_pool.reset_idle_banner(active_pane_id)
            finally:
                self._release_pane(active_pane_id)

        final_head = _git_head(worktree_root)
        attempt_label = _attempt_label(prefix="resume" if session_id else "spawn")
        patch_path = transcript_root / f"{attempt_label}-diff.patch"
        after_tree = read_workspace_tree(
            worktree_root,
            exclude_dirs=self._workspace_exclude_dirs(),
            exclude_files=_WORKSPACE_EXCLUDE_FILES,
        )
        changed_paths = changed_paths_from_tree(before_tree, after_tree)
        patch_path.write_text(diff_patch_from_tree(before_tree, after_tree), encoding="utf-8")

        resolved_session_jsonl = completion.session_jsonl
        if resolved_session_jsonl is None or not resolved_session_jsonl.exists():
            # Claude's stop hook can fire before the session JSONL is fully flushed to the
            # workspace bucket. Give the bucket a short grace period before falling back to the
            # restored transcript metadata path.
            resolved_session_jsonl = _wait_for_latest_session_jsonl(
                worktree_root,
                after_mtime_ns=launch_started_ns,
            )
        if resolved_session_jsonl is None or not resolved_session_jsonl.exists():
            resolved_session_jsonl = self._resolve_live_session_path(
                worktree_root=worktree_root,
                session_id=completion.session_id or session_id or "",
                metadata=metadata,
            )
        summary: str | None = None
        usage: ProviderUsage | None = None
        session_archive_path: Path | None = None
        subagents_archive_path: Path | None = None
        if resolved_session_jsonl is not None and resolved_session_jsonl.exists():
            summary = _extract_last_summary(
                resolved_session_jsonl,
                from_line=line_count_before_launch,
            )
            usage_payload = _extract_usage_totals(
                resolved_session_jsonl,
                from_line=line_count_before_launch,
            )
            parsed_usage = ProviderUsage(
                input_tokens=usage_payload["input_tokens"],
                cache_read_tokens=usage_payload["cache_read_input_tokens"],
                output_tokens=usage_payload["output_tokens"],
                wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
                agent_turns=usage_payload["agent_turns"],
            )
            usage = parsed_usage.model_copy(
                update={"model_cost_usd": compute_cost(parsed_usage, self.token_pricing)}
            )
            session_archive_path, subagents_archive_path = _archive_session_transcript(
                resolved_session_jsonl,
                transcript_root=transcript_root,
            )
        if completion.rollout_max_turns_reached and usage is None:
            parsed_usage = ProviderUsage(
                wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
                agent_turns=completion.observed_turns,
            )
            usage = parsed_usage.model_copy(
                update={"model_cost_usd": compute_cost(parsed_usage, self.token_pricing)}
            )
        if completion.rollout_max_turns_reached and summary is None:
            summary = _rollout_max_turns_summary(completion.observed_turns)

        completion_payload = {
            "status": "ok",
            "summary": summary,
            "changed_files": changed_paths,
        }
        archived_completion = transcript_root / f"{attempt_label}-completion.json"
        archived_completion.write_text(
            json.dumps(completion_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for stale_key in (
            "attempt_root",
            "instruction_path",
            "completion_path",
            "diff_path",
            "provider_transcript_path",
            "provider_transcript_live_path",
            "provider_transcript_subagents_path",
            "initial_head",
            "final_head",
            "actual_execution_mode",
            "driver_execution",
        ):
            metadata.pop(stale_key, None)
        metadata.update(
            {
                "driver": "claude_code_tmux",
                "role": self.role,
                "attempt_root": str(transcript_root),
                "instruction_path": (
                    str(instruction_path) if instruction_path is not None else None
                ),
                "completion_path": str(archived_completion),
                "diff_path": str(patch_path),
                "provider_transcript_path": (
                    str(session_archive_path) if session_archive_path is not None else None
                ),
                "provider_transcript_live_path": (
                    str(resolved_session_jsonl) if resolved_session_jsonl is not None else None
                ),
                "provider_transcript_subagents_path": (
                    str(subagents_archive_path) if subagents_archive_path is not None else None
                ),
                "initial_head": initial_head,
                "final_head": final_head,
                "actual_execution_mode": "claude_code_tmux",
                "driver_execution": build_driver_execution_metadata(
                    driver="claude_code_tmux",
                    command=argv,
                    cwd=worktree_root,
                    exit_code=0,
                    rollout_max_turns=self.rollout_max_turns,
                    timeout_seconds=self.timeout_seconds,
                    model=self.model,
                    effort_level=self.effort_level,
                    result_subtype=(
                        "error_max_turns" if completion.rollout_max_turns_reached else "success"
                    ),
                    result_is_error=False,
                    accepted_partial_result=completion.rollout_max_turns_reached,
                    num_turns=(
                        usage.agent_turns if usage is not None else completion.observed_turns
                    ),
                    setting_sources=list(self.setting_sources),
                    disallowed_tools=list(self.disallowed_tools),
                    dangerously_skip_permissions=self.dangerously_skip_permissions,
                ),
                "prompt_file": prompt_file,
                "write_prompt_file": write_prompt_file,
            }
        )
        resolved_session_id = (
            completion.session_id
            or session_id
            or (resolved_session_jsonl.stem if resolved_session_jsonl is not None else None)
        )
        if resolved_session_id is None:
            resolved_session_id = f"session:{uuid4().hex}"
        state = RuntimeStateRef(
            state_id=state_id if session_id is not None else f"runtime:{uuid4().hex}",
            provider_kind="claude_code_tmux",
            session_id=resolved_session_id,
            workspace_id=workspace_id,
            target_repo_root=target_repo_root,
            workspace_root=workspace_root,
            session_cwd=session_cwd or str(worktree_root),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
        return SessionRollout(
            state=state,
            primary_path=changed_paths[0] if changed_paths else None,
            changed_paths=changed_paths,
            summary=summary,
            usage=usage,
        )

    def _argv(
        self,
        *,
        session_id: str | None,
        instruction: str,
        prompt_file: str | None,
    ) -> list[str]:
        command = [self.executable]
        if self.model:
            command.extend(["--model", self.model])
        if self.setting_sources:
            command.extend(["--setting-sources", ",".join(self.setting_sources)])
        if self.disallowed_tools:
            command.extend(["--disallowedTools", ",".join(self.disallowed_tools)])
        if self.dangerously_skip_permissions:
            command.append("--dangerously-skip-permissions")
        if session_id is not None:
            command.extend(["--resume", session_id])
        prompt = instruction.strip()
        command.append(prompt)
        return command

    def _instruction_document(self, instruction: str) -> str:
        return instruction.strip() + "\n"

    def _workspace_exclude_dirs(self) -> set[str]:
        return {
            ".git",
            ".claude-driver-transcripts",
            self.transcript_dirname,
            ".codex-driver-home",
            ".codex-driver-transcripts",
        }

    def _acquire_pane(self, *, preferred_pane_id: str | None) -> str:
        if self.pane_pool is not None:
            return self.pane_pool.acquire(preferred_pane_id=preferred_pane_id)
        assert self.pane_id is not None
        return self.pane_id

    def _release_pane(self, pane_id: str) -> None:
        if self.pane_pool is not None:
            self.pane_pool.release(pane_id)

    def _resolve_live_session_path(
        self,
        *,
        worktree_root: Path,
        session_id: str,
        metadata: dict[str, object],
    ) -> Path | None:
        live_path = _string_metadata(metadata.get("provider_transcript_live_path"))
        if live_path:
            candidate = Path(live_path).expanduser().resolve()
        elif session_id:
            candidate = _project_bucket_dir(worktree_root) / f"{session_id}.jsonl"
        else:
            return None
        if candidate.exists():
            return candidate
        archive_path = _string_metadata(metadata.get("provider_transcript_path"))
        archive_subagents_path = _string_metadata(
            metadata.get("provider_transcript_subagents_path")
        )
        if archive_path:
            _restore_session_transcript(
                archive_jsonl=Path(archive_path),
                live_jsonl=candidate,
                archive_subagents=(
                    Path(archive_subagents_path) if archive_subagents_path is not None else None
                ),
            )
            if candidate.exists():
                return candidate
        return None

    def _wait_for_completion(
        self,
        *,
        worktree_root: Path,
        existing_session_jsonl: Path | None,
        line_count_before_launch: int,
        launch_started_ns: int,
        pane_id: str,
    ) -> ClaudeCodeCompletionState:
        signal_file = worktree_root / _SIGNAL_FILENAME
        deadline = time.monotonic() + self.timeout_seconds
        session_jsonl = existing_session_jsonl
        processed_line_count = line_count_before_launch
        while time.monotonic() < deadline:
            if session_jsonl is None:
                session_jsonl = _latest_session_jsonl(
                    worktree_root,
                    after_mtime_ns=launch_started_ns,
                )
            if session_jsonl is not None and session_jsonl.exists():
                observed_turns = _extract_usage_totals(
                    session_jsonl,
                    from_line=line_count_before_launch,
                )["agent_turns"]
                processed_line_count = _jsonl_line_count(session_jsonl)
                if observed_turns >= self.rollout_max_turns:
                    return ClaudeCodeCompletionState(
                        session_jsonl=session_jsonl,
                        session_id=session_jsonl.stem,
                        session_jsonl_line_count=processed_line_count,
                        rollout_max_turns_reached=True,
                        observed_turns=observed_turns,
                    )
            if signal_file.exists():
                captured = capture_pane_tail(pane_id, tail_lines=80)
                if captured.strip() and not pane_has_active_task(captured):
                    return ClaudeCodeCompletionState(
                        session_jsonl=session_jsonl,
                        session_id=(session_jsonl.stem if session_jsonl is not None else None),
                        session_jsonl_line_count=processed_line_count,
                        observed_turns=(
                            _extract_usage_totals(
                                session_jsonl, from_line=line_count_before_launch
                            )["agent_turns"]
                            if session_jsonl is not None and session_jsonl.exists()
                            else 0
                        ),
                    )
            if pane_dead(pane_id):
                return ClaudeCodeCompletionState(
                    session_jsonl=session_jsonl,
                    session_id=(session_jsonl.stem if session_jsonl is not None else None),
                    session_jsonl_line_count=processed_line_count,
                    observed_turns=(
                        _extract_usage_totals(session_jsonl, from_line=line_count_before_launch)[
                            "agent_turns"
                        ]
                        if session_jsonl is not None and session_jsonl.exists()
                        else 0
                    ),
                )
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for Claude Code stop-hook completion.")


def _rollout_max_turns_summary(turns: int) -> str:
    if turns > 0:
        return f"Claude Code tmux reached rollout_max_turns after {turns} turns."
    return "Claude Code tmux reached rollout_max_turns before returning a final summary."


def _project_bucket_dir(worktree_root: Path) -> Path:
    resolved = worktree_root.expanduser().resolve()
    # Claude stores project buckets by slugifying the full absolute path, not by joining
    # path components verbatim. Dots, underscores, and path separators all become `-`.
    bucket_name = re.sub(r"[^A-Za-z0-9]", "-", str(resolved))
    return Path.home() / ".claude" / "projects" / bucket_name


def _latest_session_jsonl(worktree_root: Path, *, after_mtime_ns: int | None = None) -> Path | None:
    project_root = _project_bucket_dir(worktree_root)
    if not project_root.exists():
        return None
    candidates = [path for path in project_root.glob("*.jsonl") if path.is_file()]
    if after_mtime_ns is not None:
        candidates = [path for path in candidates if path.stat().st_mtime_ns >= after_mtime_ns]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _wait_for_latest_session_jsonl(
    worktree_root: Path,
    *,
    after_mtime_ns: int | None = None,
    timeout_seconds: float = 5.0,
) -> Path | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        candidate = _latest_session_jsonl(worktree_root, after_mtime_ns=after_mtime_ns)
        if candidate is not None and candidate.exists():
            return candidate
        time.sleep(0.1)
    return _latest_session_jsonl(
        worktree_root,
        after_mtime_ns=after_mtime_ns,
    ) or _latest_session_jsonl(worktree_root)


def _archive_session_transcript(
    session_jsonl: Path,
    *,
    transcript_root: Path,
) -> tuple[Path, Path | None]:
    archive_root = transcript_root / session_jsonl.stem
    archive_jsonl = archive_transcript(session_jsonl, archive_root, session_jsonl.stem)

    live_session_root = session_jsonl.parent / session_jsonl.stem
    live_subagents = live_session_root / "subagents"
    archive_subagents = archive_subagent_transcripts(
        live_subagents,
        transcript_root,
        session_jsonl.stem,
    )

    session_jsonl.unlink(missing_ok=True)
    if live_session_root.exists():
        shutil.rmtree(live_session_root, ignore_errors=True)
    _prune_if_empty(session_jsonl.parent)
    return archive_jsonl, archive_subagents


def _restore_session_transcript(
    *,
    archive_jsonl: Path,
    live_jsonl: Path,
    archive_subagents: Path | None,
) -> None:
    if not archive_jsonl.exists():
        return
    live_jsonl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_jsonl, live_jsonl)
    if archive_subagents is None or not archive_subagents.exists():
        return
    live_subagents = live_jsonl.parent / live_jsonl.stem / "subagents"
    if live_subagents.exists():
        shutil.rmtree(live_subagents)
    live_subagents.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(archive_subagents, live_subagents)


def _prune_if_empty(path: Path) -> None:
    current = path
    while True:
        try:
            current.rmdir()
        except OSError:
            return
        if current.parent == current:
            return
        current = current.parent


def _extract_last_summary(session_jsonl: Path, *, from_line: int) -> str | None:
    fallback: str | None = None
    summary: str | None = None
    for payload in _iter_session_payloads(session_jsonl, from_line=from_line):
        if payload.get("type") != "assistant":
            continue
        message = _mapping(payload.get("message"))
        text = _assistant_text(message)
        if text is None:
            continue
        fallback = text
        if message.get("stop_reason") == "end_turn":
            summary = text
    return summary or fallback


def _assistant_text(message: dict[str, object]) -> str | None:
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        block = _mapping(item)
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    if not parts:
        return None
    return "\n".join(parts)


def _extract_usage_totals(session_jsonl: Path, *, from_line: int) -> dict[str, int]:
    by_request: dict[str, dict[str, int]] = {}
    assistant_ids: set[str] = set()
    for payload in _iter_session_payloads(session_jsonl, from_line=from_line):
        if payload.get("type") != "assistant":
            continue
        assistant_id = _assistant_turn_id(payload)
        if assistant_id is not None:
            assistant_ids.add(assistant_id)
        request_id = _string_metadata(payload.get("requestId"))
        if request_id is None:
            continue
        message = _mapping(payload.get("message"))
        usage = _mapping(message.get("usage"))
        if not usage:
            continue
        by_request[request_id] = {
            "input_tokens": _int_value(usage.get("input_tokens")),
            "cache_read_input_tokens": _int_value(usage.get("cache_read_input_tokens")),
            "output_tokens": _int_value(usage.get("output_tokens")),
        }
    return {
        "input_tokens": sum(item["input_tokens"] for item in by_request.values()),
        "cache_read_input_tokens": sum(
            item["cache_read_input_tokens"] for item in by_request.values()
        ),
        "output_tokens": sum(item["output_tokens"] for item in by_request.values()),
        "agent_turns": len(assistant_ids),
    }


def _assistant_turn_id(payload: dict[str, object]) -> str | None:
    message = _mapping(payload.get("message"))
    message_id = _string_metadata(message.get("id"))
    if message_id is not None:
        return message_id
    request_id = _string_metadata(payload.get("requestId"))
    if request_id is not None:
        return request_id
    return None


def _iter_session_payloads(session_jsonl: Path, *, from_line: int) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for line in session_jsonl.read_text(encoding="utf-8").splitlines()[from_line:]:
        payload = _load_json_line(line)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _visual_phase_metadata_cc(
    *,
    workspace_id: str | None,
    instruction_filename: str,
    role: str | None = None,
    display_context: dict[str, object] | None = None,
) -> tuple[str, tuple[str, ...]]:
    context = display_context or {}
    iteration = _display_value(context.get("iteration"))
    worker_index = _display_value(context.get("worker_index"))
    title = f"Gen {iteration or '?'} | TASK | Slot {worker_index or '?'}"
    if role and role != "task":
        title = f"[{role}] {title}"
    banner_lines = [
        title,
        f"attempt: {workspace_id or 'workspace:unknown'}",
        f"instruction: {instruction_filename}",
        "signal: stop hook + capture-pane active-task guard",
    ]
    if role:
        banner_lines.insert(1, f"role: {role}")
    return title, tuple(banner_lines)
