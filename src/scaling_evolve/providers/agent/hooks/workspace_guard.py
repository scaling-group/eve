"""Hook entrypoint for workspace sandboxing and rollout prompt injection."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

from scaling_evolve.providers.agent.turns import inspect_transcript_turn_state

_FILE_TOOLS = {
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "MultiEdit": "file_path",
}
_DIR_TOOLS = {
    "Glob": "path",
    "Grep": "path",
}
_ABS_PATH_RE = re.compile(r"(?<!\w)/(?:Users|home|tmp|var|etc|opt|usr|private)(?:/\S*)")
_LEGACY_ROLLOUT_CONFIG_FILENAME = ".agent_hooks.json"
_ROLLOUT_PROMPT_CONFIG_RELATIVE_PATH = Path(".hooks/rollout_prompts.json")


def _resolve(path_str: str, cwd: Path) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.expanduser().resolve(strict=False)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _find_config(cwd: Path) -> Path | None:
    for candidate in (cwd, *cwd.parents):
        config_path = candidate / ".sandbox_config.json"
        if config_path.exists():
            return config_path
    return None


def _load_config(cwd: Path) -> tuple[Path | None, list[Path]]:
    config_path = _find_config(cwd)
    if config_path is None:
        return (None, [])
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    own_workspace_raw = payload.get("own_workspace")
    own_workspace = (
        Path(own_workspace_raw).expanduser().resolve(strict=False)
        if isinstance(own_workspace_raw, str) and own_workspace_raw
        else None
    )
    evaluator_dirs = [
        Path(item).expanduser().resolve(strict=False)
        for item in payload.get("evaluator_dirs", [])
        if isinstance(item, str) and item
    ]
    return (own_workspace, evaluator_dirs)


def _emit_additional_context(hook_event_name: str, additional_context: str) -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(output))


def _is_evaluator_path(path: Path, evaluator_dirs: list[Path]) -> bool:
    return any(_is_under(path, evaluator_dir) for evaluator_dir in evaluator_dirs)


def _block(message: str) -> None:
    sys.stderr.write(f"Blocked: {message}\n")
    raise SystemExit(2)


def _emit_rollout_context(hook_input: dict[str, object]) -> str | None:
    event_name = _string(hook_input.get("hook_event_name"))
    if event_name not in {"SessionStart", "UserPromptSubmit", "PostToolUse"}:
        return None
    cwd = Path(str(hook_input.get("cwd", "."))).expanduser().resolve(strict=False)
    payload = _load_rollout_prompt_payload(cwd)
    if payload is None:
        return None
    prompts = _normalize_rollout_prompt_entries(payload)
    if not prompts:
        return None
    if event_name == "SessionStart":
        return _join_prompt_parts(_string(prompt.get("system_text")) for prompt in prompts)
    if event_name == "UserPromptSubmit":
        return _join_prompt_parts(_string(prompt.get("user_text")) for prompt in prompts)
    transcript_path = _transcript_path(hook_input)
    current_tool_use_id = _hook_tool_use_id(hook_input)
    if transcript_path is None or current_tool_use_id is None:
        return None
    turn_state = inspect_transcript_turn_state(transcript_path)
    latest_batch_ids = turn_state.latest_batch_tool_ids
    if not latest_batch_ids or latest_batch_ids[0] != current_tool_use_id:
        return None
    return _join_prompt_parts(
        _render_turn_prompt(prompt, turn_count=turn_state.turn_count) for prompt in prompts
    )


def _is_claude_background_task_output(path: Path) -> bool:
    text = path.as_posix()
    if "/tasks/" not in text:
        return False
    return "/tmp/claude-" in text or "/private/tmp/claude-" in text


def _load_rollout_prompt_payload(cwd: Path) -> dict[str, object] | None:
    config_path = _find_rollout_prompt_config(cwd)
    if config_path is None:
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _find_rollout_prompt_config(cwd: Path) -> Path | None:
    for candidate in (cwd, *cwd.parents):
        prompt_config = candidate / _ROLLOUT_PROMPT_CONFIG_RELATIVE_PATH
        if prompt_config.exists():
            return prompt_config
    for candidate in (cwd, *cwd.parents):
        legacy_config = candidate / _LEGACY_ROLLOUT_CONFIG_FILENAME
        if legacy_config.exists():
            return legacy_config
    return None


def _normalize_rollout_prompt_entries(payload: dict[str, object]) -> list[dict[str, object]]:
    version = payload.get("version")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        return []
    if version == 2:
        return [item for item in prompts if isinstance(item, dict)]
    bundle_rollout_max_turns = _positive_int(payload.get("rollout_max_turns"))
    normalized: list[dict[str, object]] = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        kind = _string(item.get("kind"))
        if kind == "budget":
            rollout_max_turns = (
                _positive_int(item.get("rollout_max_turns")) or bundle_rollout_max_turns
            )
            if rollout_max_turns is None:
                continue
            normalized.append(
                {
                    "name": "budget",
                    "system_text": None,
                    "user_text": (
                        "Turn budget enabled: this session has "
                        f"{rollout_max_turns} turns per rollout. "
                        "After each turn you will see "
                        f"`[Budget] N/{rollout_max_turns} turns remaining`. "
                        "Use that signal to pace your work - the current rollout will be "
                        "terminated when the budget runs out."
                    ),
                    "turn_template": (
                        "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining"
                    ),
                    "turn_format_kwargs": {"rollout_max_turns": rollout_max_turns},
                }
            )
        elif kind == "static":
            normalized.append(
                {
                    "name": "static",
                    "system_text": _string(item.get("system")),
                    "user_text": _string(item.get("user")),
                    "turn_template": _string(item.get("turn")),
                    "turn_format_kwargs": {},
                }
            )
    return normalized


def _render_turn_prompt(prompt: dict[str, object], *, turn_count: int) -> str | None:
    template = _string(prompt.get("turn_template"))
    if template is None:
        return None
    format_kwargs = prompt.get("turn_format_kwargs")
    extra_kwargs = format_kwargs if isinstance(format_kwargs, dict) else {}
    rollout_max_turns = _positive_int(extra_kwargs.get("rollout_max_turns"))
    turns_remaining = max(rollout_max_turns - turn_count, 0) if rollout_max_turns is not None else 0
    try:
        return template.format(turns_remaining=turns_remaining, **extra_kwargs).strip()
    except Exception:
        return None


def _transcript_path(hook_input: dict[str, object]) -> Path | None:
    raw_path = _string(hook_input.get("transcript_path")) or _string(
        hook_input.get("transcriptPath")
    )
    if raw_path is None:
        return None
    return Path(raw_path).expanduser().resolve(strict=False)


def _hook_tool_use_id(hook_input: dict[str, object]) -> str | None:
    candidates = (
        hook_input.get("tool_use_id"),
        hook_input.get("toolUseId"),
        hook_input.get("call_id"),
        hook_input.get("callId"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    tool_response = hook_input.get("tool_response")
    if isinstance(tool_response, dict):
        for key in ("call_id", "callId", "tool_use_id", "toolUseId"):
            candidate = tool_response.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _join_prompt_parts(parts) -> str | None:
    rendered = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    if not rendered:
        return None
    return "\n\n".join(rendered)


def _positive_int(value: object) -> int | None:
    if not isinstance(value, int):
        return None
    return value if value > 0 else None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _check_bash_command(
    command: str,
    *,
    cwd: Path,
    own_workspace: Path,
    evaluator_dirs: list[Path],
) -> None:
    raw_paths = _ABS_PATH_RE.findall(command)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        if (token.startswith("/") and not token.startswith("//")) or (
            "/" in token and not token.startswith("-")
        ):
            raw_paths.append(token)

    workspace_root = own_workspace.parent
    seen: set[str] = set()
    for raw_path in raw_paths:
        raw_path = raw_path.rstrip(";|&>)")
        if not raw_path or raw_path in seen:
            continue
        seen.add(raw_path)
        resolved = _resolve(raw_path, cwd)
        if _is_evaluator_path(resolved, evaluator_dirs):
            _block(f"Bash evaluator source is off-limits: {resolved}")
        if _is_claude_background_task_output(resolved):
            continue
        if _is_under(resolved, own_workspace):
            continue
        if _is_under(resolved, workspace_root):
            continue
        _block(f"Bash path is outside your workspace: {resolved}")


def main() -> None:
    try:
        hook_input = json.load(sys.stdin)
    except Exception:
        raise SystemExit(0) from None

    try:
        event_name = str(hook_input.get("hook_event_name", "PreToolUse"))
        cwd = Path(str(hook_input.get("cwd", "."))).expanduser().resolve(strict=False)
        if event_name == "PreToolUse":
            tool_name = str(hook_input.get("tool_name", ""))
            tool_input = hook_input.get("tool_input", {})
            own_workspace, evaluator_dirs = _load_config(cwd)
            if own_workspace is None:
                raise SystemExit(0)
            if tool_name == "Bash":
                command = tool_input.get("command", "")
                if isinstance(command, str) and command:
                    _check_bash_command(
                        command,
                        cwd=cwd,
                        own_workspace=own_workspace,
                        evaluator_dirs=evaluator_dirs,
                    )
                raise SystemExit(0)

            path_key = _FILE_TOOLS.get(tool_name) or _DIR_TOOLS.get(tool_name)
            if path_key is None:
                raise SystemExit(0)
            raw_path = tool_input.get(path_key)
            if not isinstance(raw_path, str) or raw_path == "":
                raise SystemExit(0)
            resolved = _resolve(raw_path, cwd)
            if _is_evaluator_path(resolved, evaluator_dirs):
                _block(f"{tool_name} evaluator source is off-limits: {resolved}")
            if _is_claude_background_task_output(resolved):
                raise SystemExit(0)
            workspace_root = own_workspace.parent
            if tool_name in {"Edit", "Write", "MultiEdit"} and not _is_under(
                resolved, own_workspace
            ):
                _block(f"{tool_name} path is outside your workspace: {resolved}")
            if (tool_name in _FILE_TOOLS or tool_name in _DIR_TOOLS) and not _is_under(
                resolved, workspace_root
            ):
                _block(f"{tool_name} path is outside workspaces: {resolved}")
            raise SystemExit(0)

        additional_context = _emit_rollout_context(hook_input)
        if additional_context is not None:
            _emit_additional_context(event_name, additional_context)
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
