"""Non-interactive Codex exec-backed session driver."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
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

_WORKSPACE_EXCLUDE_DIRS = {".codex-driver-home", ".codex-driver-transcripts", ".git"}


@dataclass(frozen=True)
class CodexExecStreamSummary:
    """Parsed summary fields from `codex exec --json` stdout."""

    session_id: str | None
    summary: str | None
    usage: dict[str, int]


@dataclass(frozen=True)
class CodexExecCommandResult:
    """Observed process result for one non-interactive Codex exec rollout."""

    completed: subprocess.CompletedProcess[str]
    rollout_max_turns_reached: bool = False
    observed_turns: int = 0


class CodexExecSessionDriver(SessionDriver):
    """Spawn-only Codex driver backed by `codex exec --json`."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        executable: str = "codex",
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "low",
        rollout_max_turns: int = 200,
        budget_prompt: bool = True,
        timeout_seconds: float = 900.0,
        personality: str | None = None,
        role: str | None = None,
        web_search: str = "disabled",
        token_pricing: TokenPricing | None = None,
        pricing_table: Mapping[str, TokenPricing] | None = None,
        model_provider: str | None = None,
        model_providers: dict[str, dict[str, object]] | None = None,
        provider_env: dict[str, str] | None = None,
    ) -> None:
        self.run_root = Path(run_root).expanduser().resolve()
        self.executable = executable
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.rollout_max_turns = rollout_max_turns
        self.budget_prompt = budget_prompt
        self.timeout_seconds = timeout_seconds
        self.personality = personality
        self.role = role
        self.web_search = web_search if web_search in {"disabled", "live", "cached"} else "disabled"
        self.token_pricing = resolve_token_pricing(model, token_pricing, pricing_table)
        self.model_provider = model_provider
        self.model_providers = dict(model_providers or {})
        self.provider_env = dict(provider_env or {})

    def capabilities(self) -> SessionDriverCapabilities:
        return SessionDriverCapabilities(
            supports_native_fork=False,
            supports_cross_workspace_fork=False,
        )

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        workspace = seed.workspace
        if workspace is None:
            raise ValueError("CodexExecSessionDriver requires a resolved workspace lease.")
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
        raise NotImplementedError("codex_exec is spawn-only; native fork is unsupported.")

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        _ = (parent_cwd, child_cwd, session_id)
        raise NotImplementedError("codex_exec is spawn-only; migrate_session is unsupported.")

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        _ = (parent, instruction)
        raise NotImplementedError("codex_exec is spawn-only; fork is unsupported.")

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        if not state.session_id:
            raise ValueError("codex_exec resume requires a session_id.")

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
        raise NotImplementedError("codex_exec is spawn-only; snapshot is unsupported.")

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
            instruction_path.write_text(instruction.strip() + "\n", encoding="utf-8")
        elif instruction_path is not None and not instruction_path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {instruction_path}")

        before_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        initial_head = _git_head(worktree_root)
        launch_started_ns = time.time_ns()
        argv = self._argv(
            worktree_root=worktree_root,
            session_id=session_id,
            instruction=instruction,
        )
        transcript_root = self._transcript_snapshot_root(worktree_root)
        transcript_root.mkdir(parents=True, exist_ok=True)
        driver_stdout_live_path = self._driver_stdout_live_path(worktree_root)
        raw_command_result = self._run_command(
            command=argv,
            cwd=worktree_root,
            env={**isolated_home.env(), **self.provider_env},
            stdout_live_path=driver_stdout_live_path,
        )
        command_result = (
            raw_command_result
            if isinstance(raw_command_result, CodexExecCommandResult)
            else CodexExecCommandResult(completed=raw_command_result)
        )
        completed = command_result.completed
        if completed.returncode != 0 and not command_result.rollout_max_turns_reached:
            raise RuntimeError(
                self._format_execution_failure(command=argv, cwd=worktree_root, completed=completed)
            )

        final_head = _git_head(worktree_root)
        after_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        changed_paths = changed_paths_from_tree(before_tree, after_tree)
        attempt_label = _attempt_label(prefix="resume" if session_id else "spawn")
        patch_path = transcript_root / f"{attempt_label}-diff.patch"
        patch_path.write_text(diff_patch_from_tree(before_tree, after_tree), encoding="utf-8")

        stream_summary = self._parse_stdout_jsonl(completed.stdout)
        live_session_path = self._resolve_session_jsonl(
            isolated_home=isolated_home,
            session_id=stream_summary.session_id or session_id,
            existing_session_jsonl=existing_session_jsonl,
            after_mtime_ns=launch_started_ns,
        )
        session_archive_path: Path | None = None
        fallback_summary: str | None = None
        usage_payload = dict(stream_summary.usage)
        usage_payload["agent_turns"] = max(
            int(usage_payload.get("agent_turns", 0) or 0),
            command_result.observed_turns,
        )
        if live_session_path is not None:
            resolved_session_id = (
                stream_summary.session_id
                or extract_session_id(live_session_path)
                or session_id
                or uuid4().hex
            )
            session_archive_path = archive_transcript(
                live_session_path,
                transcript_root,
                resolved_session_id,
                timestamp_ns=time.time_ns(),
            )
            fallback_summary = extract_last_assistant_message(live_session_path)
            fallback_usage = extract_usage(live_session_path, from_line=line_count_before_launch)
            usage_payload = {
                key: _int_value(usage_payload.get(key)) or fallback_usage.get(key, 0)
                for key in fallback_usage
            }
            usage_payload["agent_turns"] = max(
                int(usage_payload.get("agent_turns", 0) or 0),
                command_result.observed_turns,
            )

        isolated_home = reset_isolated_codex_home(
            home=isolated_home,
            source_auth_path=Path.home() / ".codex" / "auth.json",
            launch=self._launch_config(worktree_root),
        )

        summary = stream_summary.summary or fallback_summary
        if command_result.rollout_max_turns_reached and summary is None:
            summary = _rollout_max_turns_summary(command_result.observed_turns)
        completion_payload = {
            "status": "ok",
            "summary": summary,
            "changed_files": changed_paths,
        }
        completion_path = transcript_root / f"{attempt_label}-completion.json"
        completion_path.write_text(
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
            "driver_stdout_live_path",
            "driver_stdout",
            "driver_stderr",
            "initial_head",
            "final_head",
            "actual_execution_mode",
            "driver_execution",
        ):
            metadata.pop(stale_key, None)
        metadata.update(
            {
                "driver": "codex_exec",
                "role": self.role,
                "attempt_root": str(transcript_root),
                "instruction_path": str(instruction_path) if instruction_path is not None else None,
                "completion_path": str(completion_path),
                "diff_path": str(patch_path),
                "codex_driver_home": str(isolated_home.root),
                "provider_transcript_path": (
                    str(session_archive_path) if session_archive_path is not None else None
                ),
                "provider_transcript_live_path": (
                    str(live_session_path) if live_session_path is not None else None
                ),
                "driver_stdout_live_path": str(driver_stdout_live_path),
                "driver_stdout": completed.stdout,
                "driver_stderr": completed.stderr,
                "initial_head": initial_head,
                "final_head": final_head,
                "actual_execution_mode": "codex_exec",
                "driver_execution": build_driver_execution_metadata(
                    driver="codex_exec",
                    command=argv,
                    cwd=worktree_root,
                    exit_code=completed.returncode,
                    rollout_max_turns=self.rollout_max_turns,
                    timeout_seconds=self.timeout_seconds,
                    model=self.model,
                    effort_level=self.reasoning_effort,
                    reasoning_effort=self.reasoning_effort,
                    result_subtype=(
                        "error_max_turns" if command_result.rollout_max_turns_reached else "success"
                    ),
                    result_is_error=False,
                    accepted_partial_result=command_result.rollout_max_turns_reached,
                    num_turns=int(usage_payload.get("agent_turns", 0) or 0),
                    web_search=self.web_search,
                ),
                "prompt_file": prompt_file,
                "write_prompt_file": write_prompt_file,
            }
        )

        resolved_session_id = (
            stream_summary.session_id
            or (extract_session_id(live_session_path) if live_session_path is not None else None)
            or session_id
            or f"session:{uuid4().hex}"
        )
        parsed_usage = ProviderUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            cache_read_tokens=int(usage_payload.get("cached_input_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
            agent_turns=int(usage_payload.get("agent_turns", 0) or 0),
        )
        usage = parsed_usage.model_copy(
            update={"model_cost_usd": compute_cost(parsed_usage, self.token_pricing)}
        )
        state = RuntimeStateRef(
            state_id=state_id,
            provider_kind="codex_exec",
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
        worktree_root: Path,
        session_id: str | None,
        instruction: str,
    ) -> list[str]:
        prompt = instruction.strip()
        command = [self.executable, "exec"]
        if session_id is None:
            command.append(prompt)
        else:
            command.extend(["resume", session_id, prompt])
        command.extend(
            [
                "--dangerously-bypass-approvals-and-sandbox",
                "--json",
            ]
        )
        if session_id is None:
            command.extend(["-C", str(worktree_root)])
        if self.model:
            command.extend(["-m", self.model])
        command.extend(self._config_override_args())
        return command

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

    def _run_command(
        self,
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        stdout_live_path: Path,
    ) -> CodexExecCommandResult:
        stdout_live_path = stdout_live_path.expanduser()
        stdout_live_path.parent.mkdir(parents=True, exist_ok=True)
        process_env = os.environ.copy()
        process_env.update(dict(env))
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        limit_reached = threading.Event()
        observed_turns = {"count": 0}
        with stdout_live_path.open("w", encoding="utf-8") as stdout_handle:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=process_env,
            )

            def _stdout_worker() -> None:
                if process.stdout is None:
                    return
                for line in process.stdout:
                    stdout_handle.write(line)
                    stdout_handle.flush()
                    stdout_chunks.append(line)
                    if (
                        _record_exec_turn_from_stdout_line(line, observed_turns)
                        >= self.rollout_max_turns
                    ):
                        if limit_reached.is_set():
                            continue
                        limit_reached.set()
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            return

            def _stderr_worker() -> None:
                if process.stderr is None:
                    return
                stderr_text = process.stderr.read()
                if stderr_text:
                    stderr_chunks.append(stderr_text)

            stdout_thread = threading.Thread(target=_stdout_worker, daemon=True)
            stderr_thread = threading.Thread(target=_stderr_worker, daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            deadline = time.monotonic() + self.timeout_seconds
            terminate_deadline: float | None = None
            while process.poll() is None:
                now = time.monotonic()
                if limit_reached.is_set():
                    if terminate_deadline is None:
                        terminate_deadline = now + 1.0
                    elif now >= terminate_deadline:
                        process.kill()
                if now >= deadline:
                    process.kill()
                    stdout_thread.join(timeout=1.0)
                    stderr_thread.join(timeout=1.0)
                    stdout_handle.flush()
                    raise RuntimeError(
                        self._format_timeout_failure(
                            command=command,
                            cwd=cwd,
                            stdout_path=stdout_live_path,
                            stderr_text="".join(stderr_chunks),
                            timeout=self.timeout_seconds,
                        )
                    )
                time.sleep(0.05)

            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            stdout_handle.flush()

        completed = subprocess.CompletedProcess(
            args=command,
            returncode=int(process.returncode or 0),
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
        return CodexExecCommandResult(
            completed=completed,
            rollout_max_turns_reached=limit_reached.is_set(),
            observed_turns=observed_turns["count"],
        )

    @staticmethod
    def _parse_stdout_jsonl(stdout: str) -> CodexExecStreamSummary:
        session_id: str | None = None
        summary: str | None = None
        usage = _empty_usage()
        for line in stdout.splitlines():
            payload = _load_json_line(line)
            if payload is None:
                continue
            payload_type = _string_metadata(payload.get("type"))
            if payload_type == "thread.started":
                thread_id = _string_metadata(payload.get("thread_id"))
                if thread_id is not None:
                    session_id = thread_id
            elif payload_type == "item.completed":
                item = _mapping(payload.get("item"))
                if item.get("type") == "agent_message":
                    usage["agent_turns"] += 1
                    text = _string_metadata(item.get("text"))
                    if text is not None:
                        summary = text
            elif payload_type == "turn.completed":
                info = _mapping(payload.get("usage"))
                usage = {
                    "input_tokens": _int_value(info.get("input_tokens")),
                    "cached_input_tokens": _int_value(info.get("cached_input_tokens")),
                    "output_tokens": _int_value(info.get("output_tokens")),
                    "agent_turns": usage["agent_turns"],
                }
        return CodexExecStreamSummary(session_id=session_id, summary=summary, usage=usage)

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

    def _resolve_session_jsonl(
        self,
        *,
        isolated_home: IsolatedCodexHome,
        session_id: str | None,
        existing_session_jsonl: Path | None,
        after_mtime_ns: int,
    ) -> Path | None:
        if session_id is not None:
            found = rollout_path_for_session(isolated_home.root, session_id=session_id)
            if found is not None:
                return found
        if existing_session_jsonl is not None and existing_session_jsonl.exists():
            return existing_session_jsonl
        return latest_rollout_path(isolated_home.root, after_mtime_ns=after_mtime_ns)

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

    def _driver_stdout_live_path(self, cwd: Path) -> Path:
        filename = f"attempt-{time.time_ns()}-{uuid4().hex[:8]}-live.jsonl"
        return self._transcript_snapshot_root(cwd) / filename

    def _format_execution_failure(
        self,
        *,
        command: list[str],
        cwd: Path,
        completed: subprocess.CompletedProcess[str],
    ) -> str:
        return "\n".join(
            [
                "Codex exec failed.",
                f"cwd: {cwd}",
                f"exit_code: {completed.returncode}",
                f"command: {command}",
                f"stdout:\n{completed.stdout}",
                f"stderr:\n{completed.stderr}",
            ]
        )

    def _format_timeout_failure(
        self,
        *,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_text: str,
        timeout: float | None,
    ) -> str:
        return "\n".join(
            [
                "Codex exec timed out.",
                f"cwd: {cwd}",
                f"timeout_seconds: {timeout or self.timeout_seconds}",
                f"command: {command}",
                f"stdout:\n{_read_text_if_exists(stdout_path)}",
                f"stderr:\n{stderr_text}",
            ]
        )


def _attempt_label(*, prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def _git_head(cwd: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _load_json_line(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string_metadata(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _empty_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "agent_turns": 0,
    }


def _record_exec_turn_from_stdout_line(line: str, observed_turns: dict[str, int]) -> int:
    payload = _load_json_line(line.strip())
    if payload is None or _string_metadata(payload.get("type")) != "item.completed":
        return observed_turns["count"]
    item = _mapping(payload.get("item"))
    if item.get("type") == "agent_message":
        observed_turns["count"] += 1
    return observed_turns["count"]


def _rollout_max_turns_summary(turns: int) -> str:
    if turns > 0:
        return f"Codex exec reached rollout_max_turns after {turns} turns."
    return "Codex exec reached rollout_max_turns before returning a final summary."


def _jsonl_line_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value))


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _timeout_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return ""
