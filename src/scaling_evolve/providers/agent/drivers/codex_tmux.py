"""Interactive Codex tmux-backed session drivers."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from scaling_evolve.core.engine import RuntimeStateRef
from scaling_evolve.core.mutation import ProviderUsage
from scaling_evolve.providers.agent.codex_hooks import repo_codex_hooks_path
from scaling_evolve.providers.agent.codex_isolation import (
    CodexLaunchConfig,
    IsolatedCodexHome,
    create_isolated_codex_home,
    extract_last_assistant_message,
    extract_session_id,
    extract_usage,
    latest_rollout_path,
    reset_isolated_codex_home,
    rollout_path_for_session,
)
from scaling_evolve.providers.agent.drivers._metadata import (
    TokenPricing,
    build_driver_execution_metadata,
    compute_cost,
    resolve_token_pricing,
)
from scaling_evolve.providers.agent.drivers._transcript import archive_transcript
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
from scaling_evolve.providers.agent.tmux_runtime import (
    TmuxPanePoolSession,
    create_pane_pool_session,
    kill_pane,
    kill_session,
    launch_in_pane,
    pane_dead,
    show_banner_in_pane,
    wait_for_pane_reset,
)

_DEFAULT_INSTRUCTION_ENTRYPOINT = (
    "Read {instruction_filename} in the current directory and follow it exactly."
)
_WORKSPACE_EXCLUDE_DIRS = {".codex-driver-home", ".codex-driver-transcripts", ".git"}


@dataclass(frozen=True)
class CodexCompletionState:
    """Observed provider-side completion state for one rollout turn."""

    session_jsonl: Path | None
    task_complete_payload: dict[str, object] | None
    session_jsonl_line_count: int
    rollout_max_turns_reached: bool = False
    observed_turns: int = 0


@dataclass
class CodexTmuxPanePool:
    """Thread-safe pool of fungible tmux panes for Codex sessions."""

    session_name: str
    pane_ids: tuple[str, ...]
    cwd: Path
    idle_title: str = "Codex Pool | idle"
    _available: list[str] = field(init=False)
    _in_use: set[str] = field(init=False, default_factory=set)
    _closed: bool = field(init=False, default=False)
    _condition: threading.Condition = field(
        init=False,
        default_factory=lambda: threading.Condition(threading.RLock()),
    )

    def __post_init__(self) -> None:
        self.cwd = self.cwd.expanduser().resolve()
        self._available = list(self.pane_ids)

    @classmethod
    def create(
        cls,
        *,
        session_name: str,
        cwd: str | Path,
        pane_count: int,
    ) -> CodexTmuxPanePool:
        session = create_pane_pool_session(
            session_name=session_name,
            cwd=Path(cwd).expanduser().resolve(),
            pane_count=pane_count,
        )
        pool = cls(
            session_name=session.session_name,
            pane_ids=session.pane_ids,
            cwd=Path(cwd),
        )
        pool.reset_idle_banners()
        return pool

    @classmethod
    def from_session(
        cls,
        *,
        session: TmuxPanePoolSession,
        cwd: str | Path,
    ) -> CodexTmuxPanePool:
        pool = cls(session_name=session.session_name, pane_ids=session.pane_ids, cwd=Path(cwd))
        pool.reset_idle_banners()
        return pool

    def acquire(self, *, preferred_pane_id: str | None = None) -> str:
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("Codex tmux pane pool is already closed.")
                pane_id = self._take_available(preferred_pane_id)
                if pane_id is not None:
                    self._in_use.add(pane_id)
                    return pane_id
                self._condition.wait(timeout=0.1)

    def release(self, pane_id: str) -> None:
        with self._condition:
            if self._closed:
                return
            if pane_id in self._in_use:
                self._in_use.remove(pane_id)
            if pane_id not in self._available:
                self._available.append(pane_id)
            self._condition.notify()

    def reset_idle_banner(self, pane_id: str) -> None:
        show_banner_in_pane(
            pane_id=pane_id,
            cwd=self.cwd,
            title=self.idle_title,
            lines=("available",),
        )

    def reset_idle_banners(self) -> None:
        for pane_id in self.pane_ids:
            self.reset_idle_banner(pane_id)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        kill_session(self.session_name)

    def _take_available(self, preferred_pane_id: str | None) -> str | None:
        if preferred_pane_id is not None and preferred_pane_id in self._available:
            self._available.remove(preferred_pane_id)
            return preferred_pane_id
        if not self._available:
            return None
        return self._available.pop(0)


class CodexTmuxSessionDriver(SessionDriver):
    """Interactive Codex driver backed by one pane or a shared pane pool."""

    def __init__(
        self,
        *,
        pane_id: str | None = None,
        pane_pool: CodexTmuxPanePool | None = None,
        run_root: str | Path,
        executable: str = "codex",
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "low",
        rollout_max_turns: int = 200,
        budget_prompt: bool = True,
        completion_filename: str = ".evolve-done.json",
        instruction_filename: str = ".evolve-instruction.md",
        timeout_seconds: float = 900.0,
        personality: str | None = None,
        role: str | None = None,
        approval_policy: str = "never",
        sandbox_mode: str = "workspace-write",
        web_search: str = "disabled",
        token_pricing: TokenPricing | None = None,
        pricing_table: Mapping[str, TokenPricing] | None = None,
        model_provider: str | None = None,
        model_providers: dict[str, dict[str, object]] | None = None,
        provider_env: dict[str, str] | None = None,
        owns_pool: bool = False,
    ) -> None:
        if (pane_id is None) == (pane_pool is None):
            raise ValueError("Provide exactly one of pane_id or pane_pool.")
        self.pane_id = pane_id
        self.pane_pool = pane_pool
        self.run_root = Path(run_root).expanduser().resolve()
        self.executable = executable
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.rollout_max_turns = rollout_max_turns
        self.budget_prompt = budget_prompt
        self.completion_filename = completion_filename
        self.instruction_filename = instruction_filename
        self.timeout_seconds = timeout_seconds
        self.personality = personality
        self.role = role
        self.approval_policy = approval_policy
        self.sandbox_mode = sandbox_mode
        self.web_search = web_search if web_search in {"disabled", "live", "cached"} else "disabled"
        self.token_pricing = resolve_token_pricing(model, token_pricing, pricing_table)
        self.model_provider = model_provider
        self.model_providers = dict(model_providers or {})
        self.provider_env = dict(provider_env or {})
        self.owns_pool = owns_pool

    def capabilities(self) -> SessionDriverCapabilities:
        return SessionDriverCapabilities(
            supports_native_fork=False,
            supports_cross_workspace_fork=False,
        )

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        workspace = seed.workspace
        if workspace is None:
            raise ValueError("CodexTmuxSessionDriver requires a resolved workspace lease.")
        worktree_root = Path(workspace.session_cwd).resolve()
        isolated_home = create_isolated_codex_home(
            home_root=self._home_root(worktree_root),
            source_auth_path=Path.home() / ".codex" / "auth.json",
            launch=self._launch_config(worktree_root),
        )
        return self._run_rollout(
            instruction=seed.instruction,
            worktree_root=worktree_root,
            workspace_id=workspace.workspace_id,
            target_repo_root=workspace.target_repo_root,
            workspace_root=workspace.workspace_root,
            session_cwd=workspace.session_cwd,
            isolated_home=isolated_home,
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
        raise NotImplementedError("codex_tmux is spawn-only; native fork is unsupported.")

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        _ = (parent_cwd, child_cwd, session_id)
        raise NotImplementedError("codex_tmux is spawn-only; migrate_session is unsupported.")

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        _ = (parent, instruction)
        raise NotImplementedError("codex_tmux is spawn-only; fork is unsupported.")

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        if not state.session_id:
            raise ValueError("codex_tmux resume requires a session_id.")

        worktree_root = Path(state.session_cwd or state.workspace_root or ".").resolve()
        isolated_home = create_isolated_codex_home(
            home_root=self._home_root(worktree_root),
            source_auth_path=Path.home() / ".codex" / "auth.json",
            launch=self._launch_config(worktree_root),
        )
        existing_session_jsonl = self._resolve_live_session_path(
            isolated_home=isolated_home,
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
            isolated_home=isolated_home,
            session_id=state.session_id,
            state_id=state.state_id,
            existing_session_jsonl=existing_session_jsonl,
            line_count_before_launch=line_count_before_launch,
            metadata=dict(state.metadata),
        )

    def snapshot(self, state: RuntimeStateRef) -> SessionSnapshot:
        _ = state
        raise NotImplementedError("codex_tmux is spawn-only; snapshot is unsupported.")

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
        isolated_home: IsolatedCodexHome,
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
        before_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        initial_head = _git_head(worktree_root)
        launch_started_ns = time.time_ns()
        argv = self._argv(
            worktree_root,
            session_id=session_id,
            instruction=instruction,
            prompt_file=prompt_file,
        )
        pane_title, banner_lines = _visual_phase_metadata(
            workspace_id=workspace_id,
            completion_filename=self.completion_filename,
            instruction_filename=prompt_file or "inline-prompt",
            role=self.role,
            display_context=metadata,
        )
        transcript_root = self._transcript_snapshot_root(worktree_root)
        transcript_root.mkdir(parents=True, exist_ok=True)
        active_pane_id = self._acquire_pane(
            preferred_pane_id=_string_metadata(metadata.get("pane_id")),
        )
        metadata["pane_id"] = active_pane_id
        if self.pane_pool is not None:
            metadata["pane_pool_session_name"] = self.pane_pool.session_name
        try:
            launch_in_pane(
                pane_id=active_pane_id,
                cwd=worktree_root,
                env={**isolated_home.env(), **self.provider_env},
                argv=argv,
                pane_title=pane_title,
                banner_lines=banner_lines,
            )
            completion = self._wait_for_completion(
                isolated_home=isolated_home,
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
        after_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        changed_paths = changed_paths_from_tree(before_tree, after_tree)
        patch_text = diff_patch_from_tree(before_tree, after_tree)
        patch_path.write_text(patch_text, encoding="utf-8")

        session_jsonl = completion.session_jsonl or latest_rollout_path(
            isolated_home.root,
            after_mtime_ns=launch_started_ns,
        )
        if session_jsonl is not None and not session_jsonl.exists():
            session_jsonl = None
        session_archive_path: Path | None = None
        usage: ProviderUsage | None = None
        fallback_summary: str | None = None
        if session_jsonl is not None:
            resolved_session_id = extract_session_id(session_jsonl) or session_id or uuid4().hex
            session_archive_path = archive_transcript(
                session_jsonl,
                transcript_root,
                resolved_session_id,
                timestamp_ns=time.time_ns(),
            )
            usage_payload = extract_usage(session_jsonl, from_line=line_count_before_launch)
            parsed_usage = ProviderUsage(
                input_tokens=usage_payload["input_tokens"],
                cache_read_tokens=usage_payload["cached_input_tokens"],
                output_tokens=usage_payload["output_tokens"],
                wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
                agent_turns=usage_payload["agent_turns"],
            )
            usage = parsed_usage.model_copy(
                update={"model_cost_usd": compute_cost(parsed_usage, self.token_pricing)}
            )
            fallback_summary = extract_last_assistant_message(session_jsonl)
        if completion.rollout_max_turns_reached and usage is None:
            parsed_usage = ProviderUsage(
                wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
                agent_turns=completion.observed_turns,
            )
            usage = parsed_usage.model_copy(
                update={"model_cost_usd": compute_cost(parsed_usage, self.token_pricing)}
            )
        isolated_home = reset_isolated_codex_home(
            home=isolated_home,
            source_auth_path=Path.home() / ".codex" / "auth.json",
            launch=self._launch_config(worktree_root),
        )

        summary = _summary_from_task_complete(completion.task_complete_payload) or fallback_summary
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
            "runtime_contract_path",
            "completion_path",
            "diff_path",
            "isolated_home",
            "codex_driver_home",
            "provider_transcript_path",
            "provider_transcript_live_path",
            "initial_head",
            "final_head",
            "actual_execution_mode",
            "driver_execution",
        ):
            metadata.pop(stale_key, None)
        metadata.update(
            {
                "driver": "codex_tmux",
                "role": self.role,
                "attempt_root": str(transcript_root),
                "instruction_path": (
                    str(instruction_path) if instruction_path is not None else None
                ),
                "completion_path": str(archived_completion),
                "diff_path": str(patch_path),
                "codex_driver_home": str(isolated_home.root),
                "provider_transcript_path": (
                    str(session_archive_path) if session_archive_path else None
                ),
                "provider_transcript_live_path": (
                    str(session_jsonl) if session_jsonl is not None else None
                ),
                "initial_head": initial_head,
                "final_head": final_head,
                "actual_execution_mode": "codex_tmux",
                "driver_execution": build_driver_execution_metadata(
                    driver="codex_tmux",
                    command=argv,
                    cwd=worktree_root,
                    exit_code=0,
                    rollout_max_turns=self.rollout_max_turns,
                    timeout_seconds=self.timeout_seconds,
                    model=self.model,
                    effort_level=self.reasoning_effort,
                    reasoning_effort=self.reasoning_effort,
                    result_subtype=(
                        "error_max_turns" if completion.rollout_max_turns_reached else "success"
                    ),
                    result_is_error=False,
                    accepted_partial_result=completion.rollout_max_turns_reached,
                    num_turns=(
                        usage.agent_turns if usage is not None else completion.observed_turns
                    ),
                    approval_policy=self.approval_policy,
                    sandbox_mode=self.sandbox_mode,
                    web_search=self.web_search,
                ),
                "prompt_file": prompt_file,
                "write_prompt_file": write_prompt_file,
            }
        )
        resolved_session_id = (
            extract_session_id(session_jsonl) if session_jsonl is not None else session_id
        )
        if resolved_session_id is None:
            resolved_session_id = f"session:{uuid4().hex}"
        state = RuntimeStateRef(
            state_id=state_id,
            provider_kind="codex_tmux",
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
        worktree_root: Path,
        *,
        session_id: str | None,
        instruction: str,
        prompt_file: str | None,
    ) -> list[str]:
        command = [
            self.executable,
            "--no-alt-screen",
            "-a",
            self.approval_policy,
            "-s",
            self.sandbox_mode,
            "-C",
            str(worktree_root),
        ]
        if self.model:
            command.extend(["-m", self.model])
        command.extend(self._config_override_args())
        if self.web_search == "live":
            command.append("--search")
        prompt = instruction.strip()
        if session_id is None:
            command.append(prompt)
            return command
        command.extend(["resume", session_id, prompt])
        return command

    def _instruction_document(self, instruction: str) -> str:
        return instruction.strip() + "\n"

    def _config_override_args(self) -> list[str]:
        overrides = [
            ("model_reasoning_effort", self.reasoning_effort),
            ("web_search", self.web_search),
        ]
        if self.personality is not None:
            overrides.append(("personality", self.personality))
        if self.model_provider is not None:
            overrides.append(("model_provider", self.model_provider))
        for provider_id, provider_config in self.model_providers.items():
            for key, value in provider_config.items():
                if value is None:
                    continue
                overrides.append((f"model_providers.{provider_id}.{key}", _toml_scalar(value)))

        args: list[str] = []
        for key, value in overrides:
            rendered = value if key.startswith("model_providers.") else json.dumps(value)
            args.extend(["-c", f"{key}={rendered}"])
        return args

    def _home_root(self, cwd: Path) -> Path:
        return cwd / ".codex-driver-home"

    def _launch_config(self, worktree_root: Path) -> CodexLaunchConfig:
        hooks_json_path = repo_codex_hooks_path()
        return CodexLaunchConfig(
            worktree_root=worktree_root,
            hooks_json_path=hooks_json_path,
            trusted_project_roots=(hooks_json_path.parent.parent,),
        )

    def _transcript_snapshot_root(self, cwd: Path) -> Path:
        return cwd / ".codex-driver-transcripts"

    def _resolve_live_session_path(
        self,
        *,
        isolated_home: IsolatedCodexHome,
        session_id: str,
        metadata: dict[str, object],
    ) -> Path | None:
        live_path = _string_metadata(metadata.get("provider_transcript_live_path"))
        if live_path is not None:
            candidate = Path(live_path).expanduser().resolve()
            if candidate.exists():
                return candidate
        return rollout_path_for_session(isolated_home.root, session_id=session_id)

    def _acquire_pane(self, *, preferred_pane_id: str | None) -> str:
        if self.pane_pool is not None:
            return self.pane_pool.acquire(preferred_pane_id=preferred_pane_id)
        assert self.pane_id is not None
        return self.pane_id

    def _release_pane(self, pane_id: str) -> None:
        if self.pane_pool is not None:
            self.pane_pool.release(pane_id)

    def _wait_for_completion(
        self,
        *,
        isolated_home: IsolatedCodexHome,
        existing_session_jsonl: Path | None,
        line_count_before_launch: int,
        launch_started_ns: int,
        pane_id: str,
    ) -> CodexCompletionState:
        deadline = time.monotonic() + self.timeout_seconds
        session_jsonl = existing_session_jsonl
        processed_line_count = line_count_before_launch
        root_turn_id: str | None = None
        while time.monotonic() < deadline:
            if session_jsonl is None:
                session_jsonl = latest_rollout_path(
                    isolated_home.root,
                    after_mtime_ns=launch_started_ns,
                )
                if session_jsonl is not None:
                    processed_line_count = 0
            if session_jsonl is not None and session_jsonl.exists():
                lines = session_jsonl.read_text(encoding="utf-8").splitlines()
                observed_turns = extract_usage(
                    session_jsonl,
                    from_line=line_count_before_launch,
                )["agent_turns"]
                if observed_turns >= self.rollout_max_turns:
                    return CodexCompletionState(
                        session_jsonl=session_jsonl,
                        task_complete_payload=None,
                        session_jsonl_line_count=len(lines),
                        rollout_max_turns_reached=True,
                        observed_turns=observed_turns,
                    )
                for line in lines[processed_line_count:]:
                    payload = _load_json_line(line)
                    if payload is None or payload.get("type") != "event_msg":
                        continue
                    event = _mapping(payload.get("payload"))
                    event_type = event.get("type")
                    event_turn_id = _string_metadata(event.get("turn_id"))
                    if (
                        root_turn_id is None
                        and event_type == "task_started"
                        and event_turn_id is not None
                    ):
                        root_turn_id = event_turn_id
                    if (
                        event_type == "task_complete"
                        and event_turn_id is not None
                        and (root_turn_id is None or event_turn_id == root_turn_id)
                    ):
                        return CodexCompletionState(
                            session_jsonl=session_jsonl,
                            task_complete_payload=event,
                            session_jsonl_line_count=len(lines),
                            observed_turns=observed_turns,
                        )
                processed_line_count = len(lines)
            if pane_dead(pane_id):
                return CodexCompletionState(
                    session_jsonl=session_jsonl,
                    task_complete_payload=None,
                    session_jsonl_line_count=processed_line_count,
                    observed_turns=(
                        extract_usage(session_jsonl, from_line=line_count_before_launch)[
                            "agent_turns"
                        ]
                        if session_jsonl is not None and session_jsonl.exists()
                        else 0
                    ),
                )
            time.sleep(0.2)
        raise TimeoutError("Timed out waiting for Codex `task_complete` event.")


def _rollout_max_turns_summary(turns: int) -> str:
    if turns > 0:
        return f"Codex tmux reached rollout_max_turns after {turns} turns."
    return "Codex tmux reached rollout_max_turns before returning a final summary."


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _string_metadata(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _display_value(value: object) -> str | None:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _is_git_repo_root(cwd: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    try:
        resolved_root = Path(completed.stdout.strip()).resolve()
    except OSError:
        return False
    return resolved_root == cwd.resolve()


def _attempt_label(*, prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def _git_head(cwd: Path) -> str | None:
    if not _is_git_repo_root(cwd):
        return None
    completed = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_patch(*, initial_head: str | None, final_head: str | None, cwd: Path) -> str:
    if not _is_git_repo_root(cwd):
        return ""
    if initial_head is not None and final_head is not None and initial_head != final_head:
        command = ["git", "-C", str(cwd), "diff", f"{initial_head}..{final_head}"]
    else:
        command = ["git", "-C", str(cwd), "diff"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout


def _summary_from_task_complete(payload: dict[str, object] | None) -> str | None:
    if payload is None:
        return None
    summary = payload.get("last_agent_message")
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


def _visual_phase_metadata(
    *,
    workspace_id: str | None,
    completion_filename: str,
    instruction_filename: str,
    role: str | None = None,
    display_context: dict[str, object] | None = None,
) -> tuple[str, tuple[str, ...]]:
    phase = "TASK" if completion_filename == ".evolve-done.json" else "EVAL"
    context = display_context or {}
    iteration = _display_value(context.get("iteration"))
    worker_index = _display_value(context.get("worker_index"))
    title = f"Gen {iteration or '?'} | {phase} | Slot {worker_index or '?'}"
    if role and role != "task":
        title = f"[{role}] {title}"
    banner_lines = [
        title,
        f"attempt: {workspace_id or 'workspace:unknown'}",
        f"instruction: {instruction_filename}",
        "signal: event_msg/task_complete",
    ]
    if role:
        banner_lines.insert(1, f"role: {role}")
    return title, tuple(banner_lines)


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value))


def _load_json_line(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _jsonl_line_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())
