"""Non-interactive OpenCode session driver."""

from __future__ import annotations

import json
import re
import subprocess
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
from scaling_evolve.providers.agent.drivers._subprocess import run_with_live_log
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
from scaling_evolve.providers.agent.runtime_env import prepend_agent_runtime_bins

_WORKSPACE_EXCLUDE_DIRS = {
    ".codex-driver-home",
    ".codex-driver-transcripts",
    ".git",
    ".opencode-driver-transcripts",
}
_SUCCESSFUL_FINISH_REASON = "stop"
_STEP_LIMIT_SUMMARY = re.compile(
    r"\bmaximum\b.{0,120}\bsteps\b.{0,120}\breached\b",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class OpenCodeStreamSummary:
    """Strictly parsed fields from one `opencode run --format json` stream."""

    session_id: str
    summary: str | None
    finish_reason: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    model_cost_usd: float
    agent_turns: int


class OpenCodeSessionDriver(SessionDriver):
    """Spawn and resume OpenCode sessions through its non-interactive CLI."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        executable: str = "opencode",
        model: str | None = None,
        variant: str | None = None,
        rollout_max_turns: int = 200,
        budget_prompt: bool = False,
        timeout_seconds: float = 900.0,
        role: str | None = None,
        system_prompt_file: str | Path | None = None,
        token_pricing: TokenPricing | None = None,
        pricing_table: Mapping[str, TokenPricing] | None = None,
        provider_env: dict[str, str] | None = None,
    ) -> None:
        if budget_prompt:
            raise ValueError(
                "opencode does not support the Codex hook-based budget prompt; "
                "set budget_prompt=false."
            )
        self.run_root = Path(run_root).expanduser().resolve()
        self.executable = executable
        self.model = model
        self.variant = variant
        self.rollout_max_turns = int(rollout_max_turns)
        if self.rollout_max_turns <= 0:
            raise ValueError("rollout_max_turns must be positive.")
        self.budget_prompt = False
        self.timeout_seconds = timeout_seconds
        self.role = role
        self.system_prompt_file = (
            Path(system_prompt_file).expanduser().resolve()
            if system_prompt_file is not None
            else None
        )
        self.token_pricing = resolve_token_pricing(model, token_pricing, pricing_table)
        self.provider_env = dict(provider_env or {})

    def capabilities(self) -> SessionDriverCapabilities:
        return SessionDriverCapabilities(
            supports_native_fork=False,
            supports_cross_workspace_fork=False,
        )

    def workspace_config_dir(self, worktree_root: Path) -> Path:
        """Return the task-local OpenCode config directory for a workspace."""

        return self._transcript_root(worktree_root) / "config"

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        workspace = seed.workspace
        if workspace is None:
            raise ValueError("OpenCodeSessionDriver requires a resolved workspace lease.")
        worktree_root = Path(workspace.session_cwd).resolve()
        return self._run_rollout(
            instruction=seed.instruction,
            worktree_root=worktree_root,
            workspace_id=workspace.workspace_id,
            target_repo_root=workspace.target_repo_root,
            workspace_root=workspace.workspace_root,
            session_cwd=workspace.session_cwd,
            expected_session_id=None,
            state_id=f"runtime:{uuid4().hex}",
            metadata={
                **dict(seed.display_context),
                "prompt_file": seed.prompt_file,
                "write_prompt_file": seed.write_prompt_file,
            },
        )

    def fork_session(self, parent: RuntimeStateRef) -> str:
        _ = parent
        raise NotImplementedError("opencode native fork is unsupported.")

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        _ = (parent_cwd, child_cwd, session_id)
        raise NotImplementedError("opencode cross-workspace session migration is unsupported.")

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        _ = (parent, instruction)
        raise NotImplementedError("opencode native fork is unsupported.")

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        if not state.session_id:
            raise ValueError("opencode resume requires an exact session_id.")
        session_cwd = state.session_cwd or state.workspace_root
        if not session_cwd:
            raise ValueError("opencode resume requires the original session workspace.")
        worktree_root = Path(session_cwd).resolve()
        return self._run_rollout(
            instruction=instruction or "continue",
            worktree_root=worktree_root,
            workspace_id=state.workspace_id,
            target_repo_root=state.target_repo_root,
            workspace_root=state.workspace_root,
            session_cwd=str(worktree_root),
            expected_session_id=state.session_id,
            state_id=state.state_id,
            metadata=dict(state.metadata),
        )

    def snapshot(self, state: RuntimeStateRef) -> SessionSnapshot:
        _ = state
        raise NotImplementedError("opencode session snapshots are unsupported.")

    def _run_rollout(
        self,
        *,
        instruction: str,
        worktree_root: Path,
        workspace_id: str | None,
        target_repo_root: str | None,
        workspace_root: str | None,
        session_cwd: str | None,
        expected_session_id: str | None,
        state_id: str,
        metadata: dict[str, object],
    ) -> SessionRollout:
        prompt_file = _string(metadata.get("prompt_file"))
        write_prompt_file = bool(metadata.get("write_prompt_file", True))
        instruction_path = worktree_root / prompt_file if prompt_file is not None else None
        if write_prompt_file and instruction_path is not None:
            instruction_path.write_text(instruction.strip() + "\n", encoding="utf-8")
        elif instruction_path is not None and not instruction_path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {instruction_path}")

        before_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        initial_head = _git_head(worktree_root)
        launch_started_ns = time.time_ns()
        command = self._argv(
            worktree_root=worktree_root,
            session_id=expected_session_id,
            instruction=instruction,
        )
        transcript_root = self._transcript_root(worktree_root)
        transcript_root.mkdir(parents=True, exist_ok=True)
        driver_stdout_live_path = self._driver_stdout_live_path(worktree_root)
        try:
            completed = self._run_command(
                command=command,
                cwd=worktree_root,
                env=self._rollout_env(worktree_root),
                stdout_live_path=driver_stdout_live_path,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                self._format_timeout_failure(
                    command=command,
                    cwd=worktree_root,
                    stdout=_timeout_text(error.output),
                    stderr=_timeout_text(error.stderr),
                )
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                self._format_execution_failure(
                    command=command,
                    cwd=worktree_root,
                    completed=completed,
                )
            )

        try:
            stream = self._parse_stdout_jsonl(
                completed.stdout,
                expected_session_id=expected_session_id,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"OpenCode returned an invalid JSON event stream: {error}"
            ) from error
        rollout_max_turns_reached = (
            stream.agent_turns >= self.rollout_max_turns
            and stream.summary is not None
            and _STEP_LIMIT_SUMMARY.search(stream.summary) is not None
        )

        final_head = _git_head(worktree_root)
        after_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        changed_paths = changed_paths_from_tree(before_tree, after_tree)
        attempt_label = _attempt_label(prefix="resume" if expected_session_id else "spawn")
        patch_path = transcript_root / f"{attempt_label}-diff.patch"
        patch_path.write_text(diff_patch_from_tree(before_tree, after_tree), encoding="utf-8")

        cumulative_path = self._cumulative_transcript_path(worktree_root, stream.session_id)
        self._append_cumulative_transcript(
            cumulative_path,
            stdout=completed.stdout,
            instruction=instruction,
            session_id=stream.session_id,
            cwd=worktree_root,
            timestamp_ms=launch_started_ns // 1_000_000,
            truncate=expected_session_id is None,
        )
        session_archive_path = archive_transcript(
            cumulative_path,
            transcript_root,
            stream.session_id,
            timestamp_ns=time.time_ns(),
        )

        raw_usage = ProviderUsage(
            input_tokens=stream.input_tokens,
            output_tokens=stream.output_tokens + stream.reasoning_tokens,
            cache_read_tokens=stream.cache_read_tokens,
            cache_creation_tokens=stream.cache_creation_tokens,
            model_cost_usd=stream.model_cost_usd,
            wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
            agent_turns=stream.agent_turns,
        )
        usage = raw_usage.model_copy(
            update={"model_cost_usd": compute_cost(raw_usage, self.token_pricing)}
        )
        completion_payload = {
            "status": "ok",
            "session_id": stream.session_id,
            "summary": stream.summary,
            "finish_reason": stream.finish_reason,
            "changed_files": changed_paths,
            "usage": usage.model_dump(mode="json"),
        }
        completion_path = transcript_root / f"{attempt_label}-completion.json"
        completion_path.write_text(
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
                "driver": "opencode",
                "role": self.role,
                "attempt_root": str(transcript_root),
                "instruction_path": (
                    str(instruction_path) if instruction_path is not None else None
                ),
                "completion_path": str(completion_path),
                "diff_path": str(patch_path),
                "provider_transcript_path": str(session_archive_path),
                "provider_transcript_live_path": str(cumulative_path),
                "driver_stdout_live_path": str(driver_stdout_live_path),
                "driver_stdout": completed.stdout,
                "driver_stderr": completed.stderr,
                "initial_head": initial_head,
                "final_head": final_head,
                "actual_execution_mode": "opencode",
                "driver_execution": build_driver_execution_metadata(
                    driver="opencode",
                    command=command,
                    cwd=worktree_root,
                    exit_code=completed.returncode,
                    rollout_max_turns=self.rollout_max_turns,
                    timeout_seconds=self.timeout_seconds,
                    model=self.model,
                    effort_level=self.variant,
                    result_subtype=("error_max_turns" if rollout_max_turns_reached else "success"),
                    result_is_error=False,
                    accepted_partial_result=rollout_max_turns_reached,
                    num_turns=stream.agent_turns,
                    variant=self.variant,
                    finish_reason=stream.finish_reason,
                ),
                "prompt_file": prompt_file,
                "write_prompt_file": write_prompt_file,
            }
        )
        state = RuntimeStateRef(
            state_id=state_id,
            provider_kind="opencode",
            session_id=stream.session_id,
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
            summary=stream.summary,
            usage=usage,
        )

    def _argv(
        self,
        *,
        worktree_root: Path,
        session_id: str | None,
        instruction: str,
    ) -> list[str]:
        command = [
            self.executable,
            "run",
            "--format",
            "json",
            "--auto",
            "--dir",
            str(worktree_root),
            "--agent",
            "build",
            "--thinking",
        ]
        if session_id is not None:
            command.extend(["--session", session_id])
        if self.model is not None:
            command.extend(["--model", self.model])
        if self.variant is not None:
            command.extend(["--variant", self.variant])
        command.append(instruction.strip())
        return command

    def _rollout_env(self, worktree_root: Path) -> dict[str, str]:
        env = dict(self.provider_env)
        runtime_root = self._transcript_root(worktree_root)
        home_root = runtime_root / "home"
        config_root = self.workspace_config_dir(worktree_root)
        config_path = config_root / "opencode.json"
        home_root.mkdir(parents=True, exist_ok=True)
        config_root.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}\n", encoding="utf-8")
        env.update(
            {
                "HOME": str(home_root),
                "OPENCODE_CONFIG": str(config_path),
                "OPENCODE_CONFIG_DIR": str(config_root),
                "OPENCODE_DB": str(runtime_root / "opencode.db"),
                "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
                "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                "XDG_CACHE_HOME": str(home_root / ".cache"),
                "XDG_CONFIG_HOME": str(home_root / ".config"),
                "XDG_DATA_HOME": str(home_root / ".local" / "share"),
                "XDG_STATE_HOME": str(home_root / ".local" / "state"),
            }
        )
        existing = env.get("OPENCODE_CONFIG_CONTENT")
        config: dict[str, object]
        if existing:
            try:
                payload = json.loads(existing)
            except json.JSONDecodeError as error:
                raise ValueError("OPENCODE_CONFIG_CONTENT must contain valid JSON.") from error
            if not isinstance(payload, dict):
                raise ValueError("OPENCODE_CONFIG_CONTENT must contain a JSON object.")
            config = dict(payload)
        else:
            config = {}
        workspace_instructions = worktree_root / "AGENTS.md"
        if workspace_instructions.is_file():
            config["instructions"] = [str(workspace_instructions)]
        agent = config.get("agent", {})
        if not isinstance(agent, dict):
            raise ValueError("OpenCode inline config `agent` must be an object.")
        build = agent.get("build", {})
        if not isinstance(build, dict):
            raise ValueError("OpenCode inline config `agent.build` must be an object.")
        build_config = {
            **build,
            "disable": False,
            "mode": "primary",
            "steps": self.rollout_max_turns,
        }
        if self.system_prompt_file is not None:
            if not self.system_prompt_file.is_file():
                raise FileNotFoundError(
                    f"OpenCode system prompt file does not exist: {self.system_prompt_file}"
                )
            prompt = self.system_prompt_file.read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError(f"OpenCode system prompt file is empty: {self.system_prompt_file}")
            build_config["prompt"] = prompt
        config["agent"] = {
            **agent,
            "build": build_config,
        }
        config["autoupdate"] = False
        config["share"] = "disabled"
        env["OPENCODE_AUTO_SHARE"] = "false"
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            config,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return prepend_agent_runtime_bins(env, workspace_root=worktree_root)

    def _run_command(
        self,
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        stdout_live_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        return run_with_live_log(
            command,
            cwd,
            env,
            stdout_live_path,
            self.timeout_seconds,
        )

    @staticmethod
    def _parse_stdout_jsonl(
        stdout: str,
        *,
        expected_session_id: str | None = None,
    ) -> OpenCodeStreamSummary:
        session_ids: set[str] = set()
        summary: str | None = None
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        model_cost_usd = 0.0
        agent_turns = 0

        for line_number, line in enumerate(stdout.splitlines(), start=1):
            if not line.strip():
                continue
            payload = _load_json_object(line, line_number=line_number)
            event_type = _required_string(payload, "type", line_number=line_number)
            session_id = _required_string(payload, "sessionID", line_number=line_number)
            session_ids.add(session_id)
            if event_type == "error":
                error = payload.get("error")
                rendered = json.dumps(error, ensure_ascii=False, sort_keys=True)
                raise RuntimeError(f"session {session_id} reported an error: {rendered}")
            if event_type not in {"step_start", "step_finish", "text", "reasoning", "tool_use"}:
                continue
            part = _required_mapping(payload, "part", line_number=line_number)
            if event_type == "step_finish":
                finish_reason = _required_string(part, "reason", line_number=line_number)
                tokens = _required_mapping(part, "tokens", line_number=line_number)
                cache = _required_mapping(tokens, "cache", line_number=line_number)
                input_tokens += _required_token(tokens, "input", line_number=line_number)
                output_tokens += _required_token(tokens, "output", line_number=line_number)
                reasoning_tokens += _required_token(tokens, "reasoning", line_number=line_number)
                cache_read_tokens += _required_token(cache, "read", line_number=line_number)
                cache_creation_tokens += _required_token(cache, "write", line_number=line_number)
                model_cost_usd += _required_number(part, "cost", line_number=line_number)
                agent_turns += 1
            elif event_type in {"text", "reasoning"}:
                text = _required_text(part, "text", line_number=line_number)
                if event_type == "text" and text.strip():
                    summary = text.strip()
            elif event_type == "tool_use":
                _required_string(part, "tool", line_number=line_number)
                _required_string(part, "callID", line_number=line_number)
                state = _required_mapping(part, "state", line_number=line_number)
                status = _required_string(state, "status", line_number=line_number)
                if status not in {"completed", "error"}:
                    raise ValueError(
                        f"line {line_number}: completed tool event has invalid status `{status}`."
                    )

        if not session_ids:
            raise ValueError("stream contains no OpenCode session ID.")
        if len(session_ids) != 1:
            raise ValueError(f"stream contains multiple session IDs: {sorted(session_ids)}")
        session_id = next(iter(session_ids))
        if expected_session_id is not None and session_id != expected_session_id:
            raise ValueError(
                f"resumed session ID mismatch: expected `{expected_session_id}`, "
                f"got `{session_id}`."
            )
        if agent_turns == 0:
            raise ValueError("stream contains no completed OpenCode step.")
        if finish_reason is None:
            raise ValueError("stream contains no OpenCode finish reason.")
        if finish_reason != _SUCCESSFUL_FINISH_REASON:
            raise RuntimeError(
                f"session {session_id} ended with incomplete finish reason `{finish_reason}`."
            )
        return OpenCodeStreamSummary(
            session_id=session_id,
            summary=summary,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            model_cost_usd=model_cost_usd,
            agent_turns=agent_turns,
        )

    def _append_cumulative_transcript(
        self,
        path: Path,
        *,
        stdout: str,
        instruction: str,
        session_id: str,
        cwd: Path,
        timestamp_ms: int,
        truncate: bool,
    ) -> None:
        context = {
            "type": "eve.rollout",
            "timestamp": timestamp_ms,
            "sessionID": session_id,
            "instruction": instruction,
            "model": self.model,
            "variant": self.variant,
            "role": self.role,
            "cwd": str(cwd),
        }
        mode = "w" if truncate else "a"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n")
            handle.write(stdout)
            if stdout and not stdout.endswith("\n"):
                handle.write("\n")

    def _transcript_root(self, cwd: Path) -> Path:
        return cwd / ".opencode-driver-transcripts"

    def _driver_stdout_live_path(self, cwd: Path) -> Path:
        filename = f"attempt-{time.time_ns()}-{uuid4().hex[:8]}-live.jsonl"
        return self._transcript_root(cwd) / filename

    def _cumulative_transcript_path(self, cwd: Path, session_id: str) -> Path:
        safe_session_id = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in session_id
        )
        return self._transcript_root(cwd) / f"session-{safe_session_id}.jsonl"

    @staticmethod
    def _format_execution_failure(
        *,
        command: list[str],
        cwd: Path,
        completed: subprocess.CompletedProcess[str],
    ) -> str:
        return "\n".join(
            [
                "OpenCode run failed.",
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
        stdout: str,
        stderr: str,
    ) -> str:
        return "\n".join(
            [
                "OpenCode run timed out.",
                f"cwd: {cwd}",
                f"timeout_seconds: {self.timeout_seconds}",
                f"command: {command}",
                f"stdout:\n{stdout}",
                f"stderr:\n{stderr}",
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


def _load_json_object(line: str, *, line_number: int) -> dict[str, object]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line_number}: invalid JSON: {error.msg}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: event must be a JSON object.")
    return payload


def _required_mapping(
    payload: dict[str, object],
    key: str,
    *,
    line_number: int,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: `{key}` must be an object.")
    return value


def _required_string(payload: dict[str, object], key: str, *, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"line {line_number}: `{key}` must be a non-empty string.")
    return value


def _required_text(payload: dict[str, object], key: str, *, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"line {line_number}: `{key}` must be a string.")
    return value


def _required_number(payload: dict[str, object], key: str, *, line_number: int) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"line {line_number}: `{key}` must be a number.")
    if value < 0:
        raise ValueError(f"line {line_number}: `{key}` cannot be negative.")
    return float(value)


def _required_token(payload: dict[str, object], key: str, *, line_number: int) -> int:
    value = _required_number(payload, key, line_number=line_number)
    if not value.is_integer():
        raise ValueError(f"line {line_number}: token count `{key}` must be an integer.")
    return int(value)


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _timeout_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""
