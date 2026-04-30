"""Claude Code CLI-backed session driver."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from scaling_evolve.core.engine import PortableStateRef, RuntimeStateRef
from scaling_evolve.core.enums import ExecutionLifecycle
from scaling_evolve.core.mutation import ProviderUsage
from scaling_evolve.core.storage.models import ArtifactRef
from scaling_evolve.providers.agent.drivers._metadata import (
    TokenPricing,
    build_driver_execution_metadata,
    compute_cost,
    resolve_token_pricing,
)
from scaling_evolve.providers.agent.drivers._subprocess import run_with_live_log
from scaling_evolve.providers.agent.drivers._transcript import archive_transcript
from scaling_evolve.providers.agent.drivers._workspace import (
    changed_paths_from_tree,
    read_workspace_tree,
)
from scaling_evolve.providers.agent.drivers.base import (
    SessionDriver,
    SessionDriverCapabilities,
    SessionRollout,
    SessionSeed,
    SessionSnapshot,
)
from scaling_evolve.providers.agent.session_log.claude_code_parser import (
    parse_claude_code_session,
)

Runner = Callable[[Sequence[str], str, dict[str, str] | None], subprocess.CompletedProcess[str]]
_LOGGER = logging.getLogger(__name__)
_NON_ERROR_RESULT_SUBTYPES = {"success", "error_max_turns"}
_WORKSPACE_EXCLUDE_DIRS = {
    ".git",
    ".claude-driver-config",
    ".claude-driver-transcripts",
    ".codex-driver-home",
    ".codex-driver-transcripts",
}
_WORKSPACE_EXCLUDE_FILES = {
    ".claude-stop-hook.log",
    ".claude-task-stopped",
}


class ClaudeCodeSessionDriver(SessionDriver):
    """Thin wrapper around the Claude Code CLI."""

    def __init__(
        self,
        *,
        executable: str = "claude",
        runner: Runner | None = None,
        capabilities: SessionDriverCapabilities | None = None,
        provider_base_url: str | None = None,
        api_key_env: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        dangerously_skip_permissions: bool = True,
        disallowed_tools: Sequence[str] = (),
        isolate_config: bool = False,
        temperature: float = 1.0,
        rollout_max_turns: int = 200,
        budget_prompt: bool = True,
        timeout_seconds: float = 900.0,
        context_window: int | None = None,
        auto_compact_pct: int | None = None,
        max_output_tokens: int | None = None,
        max_thinking_tokens: int | None = None,
        effort_level: str | None = None,
        disable_adaptive_thinking: bool | None = None,
        token_pricing: TokenPricing | None = None,
        pricing_table: Mapping[str, TokenPricing] | None = None,
        setting_sources: Sequence[str] = ("project", "local"),
        inherited_config_dir: str | None = None,
        openrouter_provider_pin: str | None = None,
        system_prompt_append: str | None = None,
    ) -> None:
        self.executable = executable
        self.runner = runner or self._default_runner
        self._capabilities = capabilities or SessionDriverCapabilities(
            supports_native_fork=True,
            supports_cross_workspace_fork=True,
        )
        self.provider_base_url = provider_base_url
        self.api_key_env = api_key_env
        self.auth_token = auth_token
        self.model = model
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.disallowed_tools = tuple(tool for tool in disallowed_tools if tool)
        self.isolate_config = isolate_config
        self.temperature = temperature
        self.rollout_max_turns = rollout_max_turns
        self.budget_prompt = budget_prompt
        self.timeout_seconds = timeout_seconds
        self.context_window = context_window
        self.auto_compact_pct = auto_compact_pct
        self.max_output_tokens = max_output_tokens
        self.max_thinking_tokens = max_thinking_tokens
        self.effort_level = effort_level
        self.disable_adaptive_thinking = disable_adaptive_thinking
        self.token_pricing = resolve_token_pricing(model, token_pricing, pricing_table)
        self.setting_sources = tuple(source for source in setting_sources if source)
        self.inherited_config_dir = inherited_config_dir
        self.openrouter_provider_pin = openrouter_provider_pin
        self.system_prompt_append = system_prompt_append
        self.include_hook_events = self._supports_include_hook_events(self.executable)
        if temperature != 1.0:
            _LOGGER.warning(
                "Claude Code CLI does not expose a documented temperature flag; "
                "configured session temperature is retained for config parity only."
            )
        if not self.include_hook_events:
            _LOGGER.info(
                "Claude Code CLI `%s` does not support --include-hook-events; "
                "continuing without hook-response streaming.",
                self.executable,
            )

    def capabilities(self) -> SessionDriverCapabilities:
        return self._capabilities

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        cwd = self._absolute_path(self._cwd(seed))
        workspace = seed.workspace
        generated_session_id = f"session:{uuid4().hex}"
        target_repo_root = (
            self._absolute_path(workspace.target_repo_root) if workspace is not None else None
        )
        workspace_root = (
            self._absolute_path(workspace.workspace_root) if workspace is not None else None
        )
        return self._execute_rollout(
            self._spawn_command(seed),
            cwd,
            default_state_id=f"runtime:{generated_session_id}",
            provider_kind="agent_fork",
            default_session_id=generated_session_id,
            workspace_id=workspace.workspace_id if workspace is not None else None,
            target_repo_root=target_repo_root,
            workspace_root=workspace_root,
            session_cwd=cwd,
        )

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        cwd = self._absolute_path(parent.session_cwd or parent.workspace_root or ".")
        child_session_id = self.fork_session(parent)
        child_state = parent.model_copy(
            update={
                "state_id": f"runtime:{child_session_id}",
                "session_id": child_session_id,
                "workspace_root": cwd,
                "session_cwd": cwd,
            }
        )
        return self.resume(child_state, instruction=instruction)

    def fork_session(self, parent: RuntimeStateRef) -> str:
        cwd = self._absolute_path(parent.session_cwd or parent.workspace_root or ".")
        command = self._fork_session_command(parent)
        completed = self._run_command(command, cwd)
        result_line = self._validated_result_line(completed, command=command, cwd=cwd)
        session_id = self._optional_string(result_line.get("session_id"))
        if session_id is None:
            raise RuntimeError("Claude Code fork_session did not return a child session_id.")
        return session_id

    def migrate_session(
        self,
        *,
        parent_cwd: str,
        child_cwd: str,
        session_id: str,
    ) -> str:
        if self._absolute_path(parent_cwd) == self._absolute_path(child_cwd):
            return str(self._provider_session_path(cwd=child_cwd, session_id=session_id))
        source_path = self._provider_session_path(cwd=parent_cwd, session_id=session_id)
        if not source_path.exists():
            raise FileNotFoundError(f"Claude session archive missing at {source_path}")
        target_path = self._provider_session_path(cwd=child_cwd, session_id=session_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(f"{target_path.suffix}.{uuid4().hex}.tmp")
        temp_path.write_bytes(source_path.read_bytes())
        temp_path.replace(target_path)
        return str(target_path)

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        cwd = self._absolute_path(state.session_cwd or state.workspace_root or ".")
        return self._execute_rollout(
            self._resume_command(state, instruction),
            cwd,
            default_state_id=state.state_id,
            provider_kind=state.provider_kind,
            default_session_id=state.session_id,
            workspace_id=state.workspace_id,
            target_repo_root=self._absolute_optional_path(state.target_repo_root),
            workspace_root=self._absolute_optional_path(state.workspace_root),
            session_cwd=cwd,
        )

    def snapshot(self, state: RuntimeStateRef) -> SessionSnapshot:
        cwd = self._absolute_path(state.session_cwd or state.workspace_root or ".")
        command = self._snapshot_command(state)
        completed = self._run_command(command, cwd)
        result_line = self._validated_result_line(
            completed,
            command=command,
            cwd=cwd,
        )
        return SessionSnapshot(
            state=self._runtime_state_from_result(
                result_line,
                default_state_id=state.state_id,
                provider_kind=state.provider_kind,
                default_session_id=state.session_id,
                workspace_id=state.workspace_id,
                target_repo_root=self._absolute_optional_path(state.target_repo_root),
                workspace_root=self._absolute_optional_path(state.workspace_root),
                session_cwd=cwd,
            ).model_copy(
                update={
                    "metadata": self._execution_metadata(
                        command=self._snapshot_command(state),
                        cwd=cwd,
                        completed=completed,
                        result_line=result_line,
                    )
                }
            ),
            portable_state=self._portable_state_from_payload(result_line.get("portable_state")),
            transcript_digest=self._artifact_from_payload(result_line.get("transcript_digest")),
            summary=self._summary_from_result(result_line),
        )

    def _base_flags(self) -> list[str]:
        command = [
            self.executable,
            "-p",
        ]
        if self.dangerously_skip_permissions:
            command.append("--dangerously-skip-permissions")
        command += [
            "--verbose",
            "--output-format",
            "stream-json",
            "--max-turns",
            str(self.rollout_max_turns),
        ]
        if self.setting_sources:
            command.extend(["--setting-sources", ",".join(self.setting_sources)])
        if self.include_hook_events:
            command.append("--include-hook-events")
        if self.model is not None:
            command.extend(["--model", self.model])
        if self.system_prompt_append:
            command.extend(["--append-system-prompt", self.system_prompt_append])
        return command

    def _post_prompt_flags(self) -> list[str]:
        """Flags that must appear AFTER the prompt argument."""

        flags: list[str] = []
        if self.disallowed_tools:
            flags.extend(["--disallowedTools", ",".join(self.disallowed_tools)])
        return flags

    @staticmethod
    @cache
    def _supports_include_hook_events(executable: str) -> bool:
        try:
            completed = subprocess.run(
                [executable, "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        help_text = "\n".join(
            part for part in (completed.stdout, completed.stderr) if isinstance(part, str)
        )
        return "--include-hook-events" in help_text

    def _spawn_command(self, seed: SessionSeed) -> list[str]:
        command = self._base_flags()
        command.append(seed.instruction)
        command.extend(self._post_prompt_flags())
        return command

    def _fork_session_command(self, parent: RuntimeStateRef) -> list[str]:
        command = self._base_flags()
        if parent.session_id is not None:
            command.extend(["--fork-session", "--resume", parent.session_id])
        command.append("Continue from the forked session.")
        command.extend(self._post_prompt_flags())
        return command

    def _resume_command(
        self,
        state: RuntimeStateRef,
        instruction: str | None,
    ) -> list[str]:
        command = self._base_flags()
        if state.session_id is not None:
            command.extend(["--resume", state.session_id])
        if instruction is not None:
            command.append(instruction)
        else:
            command.append("continue")
        command.extend(self._post_prompt_flags())
        return command

    def _snapshot_command(self, state: RuntimeStateRef) -> list[str]:
        command = self._base_flags()
        if state.session_id is not None:
            command.extend(["--resume", state.session_id])
        command.append("Summarize the current state of the session and list all changed files.")
        command.extend(self._post_prompt_flags())
        return command

    def _execute_rollout(
        self,
        command: Sequence[str],
        cwd: str,
        *,
        default_state_id: str,
        provider_kind: str,
        default_session_id: str | None,
        workspace_id: str | None,
        target_repo_root: str | None,
        workspace_root: str | None,
        session_cwd: str,
    ) -> SessionRollout:
        attempt_root_path = self._transcript_snapshot_root(cwd)
        attempt_root_path.mkdir(parents=True, exist_ok=True)
        before_tree = read_workspace_tree(
            Path(cwd),
            exclude_dirs=_WORKSPACE_EXCLUDE_DIRS,
            exclude_files=_WORKSPACE_EXCLUDE_FILES,
        )
        driver_stdout_live_path = self._driver_stdout_live_path(cwd)
        completed = self._run_command(command, cwd, driver_stdout_live_path=driver_stdout_live_path)
        result_line = self._validated_result_line(completed, command=command, cwd=cwd)
        session_id = self._optional_string(result_line.get("session_id")) or default_session_id
        transcript_live_path = self._provider_transcript_path(cwd=cwd, session_id=session_id)
        transcript_path = self._snapshot_provider_transcript(
            cwd=cwd,
            session_id=session_id,
            live_path=transcript_live_path,
        )
        resolved_num_turns = self._resolved_num_turns(
            result_line,
            transcript_path=transcript_path,
            session_id=session_id,
        )
        attempt_root = str(attempt_root_path)
        changed_paths = self._extract_changed_paths(result_line)
        if not changed_paths:
            changed_paths = self._extract_changed_paths_from_stream_json(completed.stdout)
        if not changed_paths:
            after_tree = read_workspace_tree(
                Path(cwd),
                exclude_dirs=_WORKSPACE_EXCLUDE_DIRS,
                exclude_files=_WORKSPACE_EXCLUDE_FILES,
            )
            changed_paths = changed_paths_from_tree(before_tree, after_tree)
        primary_path = self._optional_string(result_line.get("primary_path"))
        if primary_path is None and changed_paths:
            primary_path = changed_paths[0]
        state = self._runtime_state_from_result(
            result_line,
            default_state_id=default_state_id,
            provider_kind=provider_kind,
            default_session_id=session_id,
            workspace_id=workspace_id,
            target_repo_root=target_repo_root,
            workspace_root=workspace_root,
            session_cwd=session_cwd,
        ).model_copy(
            update={
                "metadata": self._execution_metadata(
                    command=command,
                    cwd=cwd,
                    completed=completed,
                    result_line=result_line,
                    num_turns=resolved_num_turns,
                    driver_stdout_live_path=driver_stdout_live_path,
                    transcript_path=transcript_path,
                    transcript_live_path=transcript_live_path,
                    attempt_root=attempt_root,
                )
            }
        )
        return SessionRollout(
            state=state,
            transcript=self._artifact_from_payload(result_line.get("transcript")),
            changed_files_manifest=self._artifact_from_payload(
                result_line.get("changed_files_manifest")
            ),
            primary_path=primary_path,
            changed_paths=changed_paths,
            summary=self._summary_from_result(result_line, num_turns=resolved_num_turns),
            usage=self._extract_usage(result_line, num_turns=resolved_num_turns),
        )

    def _cwd(self, seed: SessionSeed) -> str:
        if seed.workspace is not None:
            return seed.workspace.session_cwd
        return seed.working_directory or "."

    def _default_runner(
        self,
        command: Sequence[str],
        cwd: str,
        extra_env: dict[str, str] | None = None,
        *,
        stdout_live_path: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        resolved_env = dict(extra_env or {})
        with self._openrouter_pin_proxy(resolved_env):
            if stdout_live_path is None:
                invocation = list(command)
                if resolved_env:
                    config_dir = resolved_env.get("CLAUDE_CONFIG_DIR")
                    if isinstance(config_dir, str) and config_dir:
                        Path(config_dir).mkdir(parents=True, exist_ok=True)
                    invocation = [
                        "env",
                        *(f"{key}={value}" for key, value in resolved_env.items()),
                        *invocation,
                    ]
                return subprocess.run(
                    invocation,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )

            stdout_path = Path(stdout_live_path).expanduser()
            config_dir = resolved_env.get("CLAUDE_CONFIG_DIR")
            if isinstance(config_dir, str) and config_dir:
                Path(config_dir).mkdir(parents=True, exist_ok=True)
            return run_with_live_log(
                command=command,
                cwd=cwd,
                env=resolved_env,
                live_log_path=stdout_path,
                timeout=self.timeout_seconds,
            )

    def _run_command(
        self,
        command: Sequence[str],
        cwd: str,
        *,
        driver_stdout_live_path: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self._provider_env(cwd)
        try:
            if self._runner_uses_default_impl():
                completed = self._default_runner(
                    command,
                    cwd,
                    env,
                    stdout_live_path=driver_stdout_live_path,
                )
            else:
                completed = self.runner(command, cwd, env)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                self._format_timeout_failure(command=command, cwd=cwd, error=exc)
            ) from exc
        return self._completed_with_live_stdout(
            completed,
            driver_stdout_live_path=driver_stdout_live_path,
        )

    @contextmanager
    def _openrouter_pin_proxy(self, env: dict[str, str]) -> Iterator[None]:
        if not self._should_use_openrouter_pin_proxy():
            yield
            return
        port = self._reserve_loopback_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scaling_evolve.providers.agent.drivers.openrouter_proxy",
                "--listen-port",
                str(port),
                "--target-base-url",
                self.provider_base_url or "",
                "--pinned-provider",
                self.openrouter_provider_pin or "",
            ],
            cwd=Path.cwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
        try:
            self._wait_for_proxy(port)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _should_use_openrouter_pin_proxy(self) -> bool:
        return (
            isinstance(self.provider_base_url, str)
            and "openrouter.ai" in self.provider_base_url
            and isinstance(self.openrouter_provider_pin, str)
            and self.openrouter_provider_pin != ""
        )

    def _reserve_loopback_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _wait_for_proxy(self, port: int) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"OpenRouter pinning proxy on port {port} did not become ready.")

    def _provider_env(self, cwd: str) -> dict[str, str] | None:
        provider_auth_override_configured = any(
            value is not None for value in (self.provider_base_url, self.api_key_env)
        )
        if self.api_key_env is not None and not self.auth_token:
            raise ValueError(
                f"Missing auth token for Claude Code provider override `{self.api_key_env}`."
            )
        if provider_auth_override_configured and not self.auth_token:
            raise ValueError("Claude Code provider overrides require an auth token.")
        env = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        runtime_bin = Path(cwd) / ".agent-runtime" / "bin"
        if runtime_bin.exists():
            env["PATH"] = f"{runtime_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        if self.isolate_config:
            env["CLAUDE_CONFIG_DIR"] = str(self._isolated_config_dir(cwd))
        if self.provider_base_url is not None:
            env["ANTHROPIC_BASE_URL"] = self.provider_base_url
        if provider_auth_override_configured:
            # OpenRouter's Claude Code integration requires this to be explicitly blank so the
            # CLI does not fall back to cached or ambient Anthropic credentials.
            env["ANTHROPIC_API_KEY"] = ""
        if self.auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = self.auth_token
        if self.model is not None:
            env["ANTHROPIC_MODEL"] = self.model
            env["ANTHROPIC_SMALL_FAST_MODEL"] = self.model
        if self.context_window is not None:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(self.context_window)
        if self.auto_compact_pct is not None:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(self.auto_compact_pct)
        if self.max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_output_tokens)
        if self.max_thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)
        if self.effort_level is not None:
            env["CLAUDE_CODE_EFFORT_LEVEL"] = self.effort_level
        if self.disable_adaptive_thinking is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = (
                "1" if self.disable_adaptive_thinking else "0"
            )
        return env

    def _maybe_extract_result_line(self, stdout: str) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload_dict = cast(dict[str, Any], payload)
                if payload_dict.get("type") == "result":
                    result = payload_dict
        if result:
            return result

        text = stdout.strip()
        if text != "":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                if stdout.strip() == "":
                    return None
                raise ValueError(
                    "Claude Code driver expected a `type=result` line in stream-json output."
                ) from error
            if isinstance(payload, dict):
                payload_dict = cast(dict[str, Any], payload)
                if payload_dict.get("type") == "result":
                    return payload_dict
        return None

    def _validated_result_line(
        self,
        completed: subprocess.CompletedProcess[str],
        *,
        command: Sequence[str],
        cwd: str,
    ) -> dict[str, Any]:
        result_line = self._maybe_extract_result_line(completed.stdout)
        if result_line is None:
            raise RuntimeError(
                self._format_execution_failure(
                    command=command,
                    cwd=cwd,
                    completed=completed,
                    result_line=None,
                    reason="missing_result_line",
                )
            )
        if self._result_is_error(result_line):
            raise RuntimeError(
                self._format_execution_failure(
                    command=command,
                    cwd=cwd,
                    completed=completed,
                    result_line=result_line,
                    reason="error_result",
                )
            )
        if completed.returncode != 0:
            _LOGGER.warning(
                "Claude Code exited with code %s but returned usable subtype `%s` in %s",
                completed.returncode,
                self._optional_string(result_line.get("subtype")) or "unknown",
                cwd,
            )
        return result_line

    def _result_is_error(self, result: dict[str, Any]) -> bool:
        subtype = self._optional_string(result.get("subtype"))
        if subtype in _NON_ERROR_RESULT_SUBTYPES:
            return False
        is_error = result.get("is_error")
        if isinstance(is_error, bool):
            return is_error
        if subtype is None:
            return False
        return subtype not in _NON_ERROR_RESULT_SUBTYPES and subtype.startswith("error_")

    def _format_execution_failure(
        self,
        *,
        command: Sequence[str],
        cwd: str,
        completed: subprocess.CompletedProcess[str],
        result_line: dict[str, Any] | None,
        reason: str,
    ) -> str:
        subtype = self._optional_string(result_line.get("subtype")) if result_line else None
        return "\n".join(
            [
                f"Claude Code CLI failed ({reason}).",
                f"cwd: {cwd}",
                f"exit_code: {completed.returncode}",
                f"subtype: {subtype or 'n/a'}",
                f"command: {list(command)}",
                f"stdout:\n{completed.stdout}",
                f"stderr:\n{completed.stderr}",
            ]
        )

    def _format_timeout_failure(
        self,
        *,
        command: Sequence[str],
        cwd: str,
        error: subprocess.TimeoutExpired,
    ) -> str:
        return "\n".join(
            [
                "Claude Code CLI timed out.",
                f"cwd: {cwd}",
                f"timeout_seconds: {self.timeout_seconds}",
                f"command: {list(command)}",
                f"stdout:\n{self._timeout_text(error.stdout)}",
                f"stderr:\n{self._timeout_text(error.stderr)}",
            ]
        )

    def _extract_usage(
        self,
        result: dict[str, Any],
        *,
        num_turns: int | None = None,
    ) -> ProviderUsage:
        usage_payload = result.get("usage")
        usage = cast(dict[str, Any], usage_payload) if isinstance(usage_payload, dict) else {}
        duration_ms = result.get("duration_ms")
        wallclock_seconds = (
            float(duration_ms) / 1000.0 if isinstance(duration_ms, (int, float)) else 0.0
        )
        resolved_num_turns = (
            num_turns if num_turns is not None else self._safe_int(result.get("num_turns"))
        )
        parsed = ProviderUsage(
            input_tokens=self._safe_int(usage.get("input_tokens")),
            output_tokens=self._safe_int(usage.get("output_tokens")),
            cache_read_tokens=self._cache_read_tokens(usage),
            cache_creation_tokens=self._cache_creation_tokens(usage),
            model_cost_usd=self._safe_float(result.get("total_cost_usd")),
            wallclock_seconds=wallclock_seconds,
            agent_turns=resolved_num_turns,
        )
        return parsed.model_copy(
            update={"model_cost_usd": compute_cost(parsed, self.token_pricing)}
        )

    def _cache_read_tokens(self, usage: dict[str, Any]) -> int:
        direct = self._safe_int(usage.get("cache_read_input_tokens"))
        if direct > 0:
            return direct
        return self._safe_int(usage.get("cached_tokens"))

    def _cache_creation_tokens(self, usage: dict[str, Any]) -> int:
        direct = self._safe_int(usage.get("cache_creation_input_tokens"))
        if direct > 0:
            return direct
        return self._safe_int(usage.get("cache_write_tokens"))

    def _extract_changed_paths(self, result: dict[str, Any]) -> list[str]:
        changed_paths = self._string_list(result.get("changed_paths"))
        if changed_paths:
            return changed_paths
        return self._string_list(result.get("changed_files"))

    def _extract_changed_paths_from_stream_json(self, stdout: str) -> list[str]:
        changed_paths: list[str] = []
        seen: set[str] = set()
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or payload.get("type") != "user":
                continue
            tool_result = payload.get("tool_use_result")
            if not isinstance(tool_result, dict):
                continue
            file_path = self._optional_string(tool_result.get("filePath")) or self._optional_string(
                tool_result.get("file_path")
            )
            if file_path is None or file_path in seen:
                continue
            seen.add(file_path)
            changed_paths.append(file_path)
        return changed_paths

    def _runtime_state_from_result(
        self,
        result: dict[str, Any],
        *,
        default_state_id: str,
        provider_kind: str,
        default_session_id: str | None,
        workspace_id: str | None,
        target_repo_root: str | None,
        workspace_root: str | None,
        session_cwd: str,
    ) -> RuntimeStateRef:
        session_id = self._optional_string(result.get("session_id")) or default_session_id
        state_id = self._optional_string(result.get("state_id")) or default_state_id
        return RuntimeStateRef(
            state_id=state_id,
            provider_kind=provider_kind,
            lifecycle=ExecutionLifecycle(
                self._optional_string(result.get("lifecycle")) or "running"
            ),
            session_id=session_id,
            workspace_id=workspace_id,
            target_repo_root=target_repo_root,
            workspace_root=workspace_root,
            session_cwd=session_cwd,
        )

    def _execution_metadata(
        self,
        *,
        command: Sequence[str],
        cwd: str,
        completed: subprocess.CompletedProcess[str],
        result_line: dict[str, Any],
        num_turns: int | None = None,
        driver_stdout_live_path: str | None = None,
        transcript_path: str | None = None,
        transcript_live_path: str | None = None,
        attempt_root: str | None = None,
    ) -> dict[str, Any]:
        subtype = self._optional_string(result_line.get("subtype"))
        resolved_num_turns = (
            num_turns if num_turns is not None else self._safe_int(result_line.get("num_turns"))
        )
        return {
            "driver_execution": build_driver_execution_metadata(
                driver="claude_code",
                command=command,
                cwd=cwd,
                exit_code=completed.returncode,
                rollout_max_turns=self.rollout_max_turns,
                timeout_seconds=self.timeout_seconds,
                model=self.model,
                effort_level=self.effort_level,
                result_subtype=subtype,
                result_is_error=self._result_is_error(result_line),
                accepted_partial_result=subtype == "error_max_turns",
                num_turns=resolved_num_turns,
                setting_sources=list(self.setting_sources),
                disallowed_tools=list(self.disallowed_tools),
                dangerously_skip_permissions=self.dangerously_skip_permissions,
            ),
            "attempt_root": attempt_root,
            "driver_stdout_live_path": driver_stdout_live_path,
            "provider_transcript_path": transcript_path,
            "provider_transcript_live_path": transcript_live_path,
            "driver_stdout": completed.stdout,
            "driver_stderr": completed.stderr,
            "driver_result": result_line,
        }

    def _summary_from_result(
        self,
        result: dict[str, Any],
        *,
        num_turns: int | None = None,
    ) -> str | None:
        summary = self._optional_string(result.get("result")) or self._optional_string(
            result.get("summary")
        )
        if summary is not None:
            return summary
        subtype = self._optional_string(result.get("subtype"))
        if subtype != "error_max_turns":
            return None
        turns = num_turns if num_turns is not None else self._safe_int(result.get("num_turns"))
        if turns > 0:
            return f"Claude Code reached rollout_max_turns after {turns} turns."
        return "Claude Code reached rollout_max_turns before returning a final summary."

    def _resolved_num_turns(
        self,
        result: dict[str, Any],
        *,
        transcript_path: str | None,
        session_id: str | None,
    ) -> int:
        transcript_turns = self._transcript_turn_count(transcript_path, session_id=session_id)
        if transcript_turns > 0:
            return transcript_turns
        return self._safe_int(result.get("num_turns"))

    def _transcript_turn_count(self, transcript_path: str | None, *, session_id: str | None) -> int:
        if not transcript_path:
            return 0
        try:
            parsed = parse_claude_code_session(
                Path(transcript_path).expanduser(),
                session_id=session_id,
                effort=self.effort_level,
            )
        except Exception:
            return 0
        return len(parsed.turns) if parsed is not None else 0

    def _absolute_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve(strict=False))

    def _absolute_optional_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        return self._absolute_path(path)

    def _provider_transcript_path(self, *, cwd: str, session_id: str | None) -> str | None:
        if session_id is None:
            return None
        return str(self._provider_session_path(cwd=cwd, session_id=session_id))

    def _snapshot_provider_transcript(
        self,
        *,
        cwd: str,
        session_id: str | None,
        live_path: str | None,
    ) -> str | None:
        if session_id is None or live_path is None:
            return None
        source_path = Path(live_path).expanduser()
        if not source_path.exists():
            return None
        snapshot_root = self._transcript_snapshot_root(cwd)
        snapshot_path = archive_transcript(
            source_path,
            snapshot_root,
            session_id,
            timestamp_ns=time.time_ns(),
        )
        return str(snapshot_path)

    def _transcript_snapshot_root(self, cwd: str) -> Path:
        return Path(self._absolute_path(cwd)) / ".claude-driver-transcripts"

    def _driver_stdout_live_path(self, cwd: str) -> str:
        root = self._transcript_snapshot_root(cwd)
        filename = f"attempt-{time.time_ns()}-{uuid4().hex[:8]}-live.jsonl"
        return str(root / filename)

    def _provider_session_bucket(self, *, cwd: str) -> Path:
        encoded_cwd = "".join(character if character.isalnum() else "-" for character in cwd)
        return self._provider_config_dir(cwd) / "projects" / encoded_cwd

    def _isolated_config_dir(self, cwd: str) -> Path:
        return Path(self._absolute_path(cwd)) / ".claude-driver-config"

    def _provider_config_dir(self, cwd: str) -> Path:
        if self.isolate_config:
            return self._isolated_config_dir(cwd)
        if self.inherited_config_dir is not None:
            return Path(self._absolute_path(self.inherited_config_dir))
        return Path.home() / ".claude"

    def _provider_session_path(self, *, cwd: str, session_id: str) -> Path:
        return self._provider_session_bucket(cwd=cwd) / f"{session_id}.jsonl"

    def _completed_with_live_stdout(
        self,
        completed: subprocess.CompletedProcess[str],
        *,
        driver_stdout_live_path: str | None,
    ) -> subprocess.CompletedProcess[str]:
        if driver_stdout_live_path is None:
            return completed
        live_path = Path(driver_stdout_live_path).expanduser()
        stdout_text = completed.stdout
        if live_path.exists():
            live_text = self._read_text_if_exists(live_path)
            if not stdout_text:
                stdout_text = live_text
        elif stdout_text:
            live_path.parent.mkdir(parents=True, exist_ok=True)
            live_path.write_text(stdout_text, encoding="utf-8")
        if stdout_text == completed.stdout:
            return completed
        return subprocess.CompletedProcess(
            args=completed.args,
            returncode=completed.returncode,
            stdout=stdout_text,
            stderr=completed.stderr,
        )

    @staticmethod
    def _read_text_if_exists(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _runner_uses_default_impl(self) -> bool:
        return getattr(self.runner, "__func__", None) is ClaudeCodeSessionDriver._default_runner

    def _portable_state_from_payload(self, payload: object) -> PortableStateRef | None:
        if not isinstance(payload, dict):
            return None
        return PortableStateRef.model_validate(payload)

    def _artifact_from_payload(self, payload: object) -> ArtifactRef | None:
        if not isinstance(payload, dict):
            return None
        return ArtifactRef.model_validate(payload)

    def _optional_string(self, value: object) -> str | None:
        return value if isinstance(value, str) else None

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items = cast(list[object], value)
        return [item for item in items if isinstance(item, str)]

    @staticmethod
    def _safe_int(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0

    @staticmethod
    def _safe_float(value: object) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _timeout_text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        return ""
