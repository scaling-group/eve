from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from scaling_evolve.algorithms.eve.runtime.driver import build_driver
from scaling_evolve.algorithms.eve.workspace.runtime_hooks import (
    install_workspace_runtime_hooks,
)
from scaling_evolve.providers.agent.config import AgentProviderConfig
from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease
from scaling_evolve.providers.agent.drivers.opencode import (
    OpenCodeSessionDriver,
    OpenCodeStreamSummary,
)

_LIVE_OPENCODE = os.environ.get("SCALING_EVOLVE_RUN_LIVE_OPENCODE_TESTS") == "1"
_LIVE_MODEL = os.environ.get("SCALING_EVOLVE_OPENCODE_MODEL")


def _init_git_repo(worktree: Path, *, contents: str = "VALUE = 1\n") -> None:
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text(contents, encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)


def _lease(worktree: Path) -> SessionWorkspaceLease:
    return SessionWorkspaceLease(
        workspace_id="attempt-1",
        target_repo_root=str(worktree),
        workspace_root=str(worktree),
        session_cwd=str(worktree),
    )


def _event(event_type: str, session_id: str, part: dict[str, object]) -> str:
    return json.dumps(
        {
            "type": event_type,
            "timestamp": 1_700_000_000_000,
            "sessionID": session_id,
            "part": part,
        }
    )


def _successful_stream(
    session_id: str,
    summary: str,
    *,
    cost: float = 0.012,
    finish_reason: str = "stop",
) -> str:
    return (
        "\n".join(
            [
                _event("step_start", session_id, {"type": "step-start"}),
                _event(
                    "reasoning",
                    session_id,
                    {"type": "reasoning", "text": "Inspect the file."},
                ),
                _event(
                    "tool_use",
                    session_id,
                    {
                        "type": "tool",
                        "tool": "edit",
                        "callID": "call-1",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "candidate.py"},
                            "output": "Done",
                        },
                    },
                ),
                _event("text", session_id, {"type": "text", "text": summary}),
                _event(
                    "step_finish",
                    session_id,
                    {
                        "type": "step-finish",
                        "reason": finish_reason,
                        "cost": cost,
                        "tokens": {
                            "input": 11,
                            "output": 7,
                            "reasoning": 2,
                            "cache": {"read": 3, "write": 4},
                        },
                    },
                ),
            ]
        )
        + "\n"
    )


def test_opencode_parse_stdout_extracts_native_usage_and_cost() -> None:
    parsed = OpenCodeSessionDriver._parse_stdout_jsonl(
        _successful_stream("ses_123", "Finished."),
    )

    assert parsed == OpenCodeStreamSummary(
        session_id="ses_123",
        summary="Finished.",
        finish_reason="stop",
        input_tokens=11,
        output_tokens=7,
        reasoning_tokens=2,
        cache_read_tokens=3,
        cache_creation_tokens=4,
        model_cost_usd=pytest.approx(0.012),
        agent_turns=1,
    )


def test_opencode_spawn_and_exact_resume_collect_cumulative_transcript(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    workspace_instructions = worktree / "AGENTS.md"
    workspace_instructions.write_text("Use only EvE workspace instructions.\n", encoding="utf-8")
    system_prompt = tmp_path / "SYSTEM_PROMPT.md"
    system_prompt.write_text("Use the project workflow.\n", encoding="utf-8")
    shared_npm_cache = tmp_path / "shared-npm-cache"
    shared_xdg_cache = tmp_path / "shared-xdg-cache"
    monkeypatch.setenv("NPM_CONFIG_CACHE", str(shared_npm_cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(shared_xdg_cache))
    launches: list[list[str]] = []
    runtime_homes: list[Path] = []

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        launches.append(command)
        assert cwd == worktree
        inline = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert inline["agent"]["build"] == {
            "disable": False,
            "mode": "primary",
            "prompt": "Use the project workflow.",
            "steps": 4,
        }
        assert inline["provider"] == {"sentinel": {"npm": "test"}}
        assert inline["instructions"] == [str(workspace_instructions)]
        assert inline["autoupdate"] is False
        assert inline["share"] == "disabled"
        assert env["OPENCODE_AUTO_SHARE"] == "false"
        assert env["DEEPSEEK_API_KEY"] == "not-a-real-key"
        runtime_root = worktree / ".opencode-driver-transcripts"
        home_root = Path(env["HOME"])
        assert home_root.parent == driver.run_root
        config_root = home_root / ".config" / "opencode"
        assert env["OPENCODE_CONFIG"] == str(config_root / "opencode.json")
        assert env["OPENCODE_CONFIG_DIR"] == str(config_root)
        assert env["NPM_CONFIG_CACHE"] == str(shared_npm_cache)
        assert env["OPENCODE_DB"] == str(runtime_root / "opencode.db")
        assert env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "true"
        assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "true"
        assert env["XDG_CACHE_HOME"] == str(shared_xdg_cache)
        assert env["XDG_CONFIG_HOME"] == str(home_root / ".config")
        assert env["XDG_DATA_HOME"] == str(home_root / ".local" / "share")
        assert env["XDG_STATE_HOME"] == str(home_root / ".local" / "state")
        assert (config_root / "opencode.json").read_text(encoding="utf-8") == "{}\n"
        assert (config_root / "agents" / "eve-agent.md").read_text(encoding="utf-8") == (
            "Eve agent\n"
        )
        runtime_homes.append(home_root)
        residue = home_root / ".future-opencode-cache" / "new-version" / "payload"
        residue.parent.mkdir(parents=True)
        residue.write_text("cache\n", encoding="utf-8")
        summary = "Spawn finished." if len(launches) == 1 else "Resume finished."
        stdout = _successful_stream("ses_resume", summary)
        stdout_live_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_live_path.write_text(stdout, encoding="utf-8")
        value = 2 if len(launches) == 1 else 3
        (worktree / "candidate.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(OpenCodeSessionDriver, "_run_command", fake_run)
    driver = OpenCodeSessionDriver(
        run_root=tmp_path / "run-root",
        model="deepseek/deepseek-chat",
        variant="high",
        rollout_max_turns=4,
        system_prompt_file=system_prompt,
        provider_env={
            "DEEPSEEK_API_KEY": "not-a-real-key",
            "OPENCODE_AUTO_SHARE": "true",
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                {
                    "agent": {"build": {"disable": True, "mode": "subagent"}},
                    "provider": {"sentinel": {"npm": "test"}},
                    "share": "auto",
                }
            ),
        },
    )
    projected_agent = driver.workspace_config_dir(worktree) / "agents" / "eve-agent.md"
    projected_agent.parent.mkdir(parents=True)
    projected_agent.write_text("Eve agent\n", encoding="utf-8")

    spawned = driver.spawn(SessionSeed(instruction="Do the task", workspace=_lease(worktree)))
    resumed = driver.resume(spawned.state, instruction="Continue the task")

    common = [
        "opencode",
        "run",
        "--format",
        "json",
        "--auto",
        "--dir",
        str(worktree),
        "--agent",
        "build",
        "--thinking",
    ]
    assert launches[0] == [
        *common,
        "--model",
        "deepseek/deepseek-chat",
        "--variant",
        "high",
        "Do the task",
    ]
    assert launches[1] == [
        *common,
        "--session",
        "ses_resume",
        "--model",
        "deepseek/deepseek-chat",
        "--variant",
        "high",
        "Continue the task",
    ]
    assert spawned.state.session_id == resumed.state.session_id == "ses_resume"
    assert spawned.changed_paths == resumed.changed_paths == ["candidate.py"]
    assert resumed.summary == "Resume finished."
    assert resumed.usage is not None
    assert resumed.usage.input_tokens == 11
    assert resumed.usage.output_tokens == 9
    assert resumed.usage.cache_read_tokens == 3
    assert resumed.usage.cache_creation_tokens == 4
    assert resumed.usage.agent_turns == 1
    assert resumed.usage.model_cost_usd == pytest.approx(0.012)
    transcript = Path(resumed.state.metadata["provider_transcript_path"])
    transcript_text = transcript.read_text(encoding="utf-8")
    assert transcript_text.count('"type": "eve.rollout"') == 2
    assert "Spawn finished." in transcript_text
    assert "Resume finished." in transcript_text
    assert Path(resumed.state.metadata["diff_path"]).exists()
    assert Path(resumed.state.metadata["completion_path"]).exists()
    assert len(runtime_homes) == 2
    assert all(not home.exists() for home in runtime_homes)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("not-json\n", "invalid JSON"),
        (
            _event("text", "ses_incomplete", {"type": "text", "text": "partial"}),
            "no completed OpenCode step",
        ),
        (json.dumps({"type": "text", "part": {"text": "x"}}), "sessionID"),
        (
            _successful_stream("ses_other", "x"),
            "resumed session ID mismatch",
        ),
    ],
)
def test_opencode_json_stream_failures_are_explicit(stdout: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenCodeSessionDriver._parse_stdout_jsonl(
            stdout,
            expected_session_id="ses_expected" if "other" in stdout else None,
        )


@pytest.mark.parametrize("finish_reason", ["tool-calls", "unknown"])
def test_opencode_incomplete_final_finish_reason_is_fatal(finish_reason: str) -> None:
    stdout = _successful_stream(
        "ses_partial",
        "partial",
        finish_reason=finish_reason,
    )
    with pytest.raises(RuntimeError, match=f"incomplete finish reason `{finish_reason}`"):
        OpenCodeSessionDriver._parse_stdout_jsonl(stdout)


def test_opencode_accepts_intermediate_tool_call_finish() -> None:
    stdout = _successful_stream(
        "ses_tools",
        "Using a tool.",
        finish_reason="tool-calls",
    ) + _successful_stream("ses_tools", "Finished.")
    parsed = OpenCodeSessionDriver._parse_stdout_jsonl(stdout)
    assert parsed.finish_reason == "stop"
    assert parsed.agent_turns == 2


def test_opencode_rollout_turn_exhaustion_preserves_partial_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    driver = OpenCodeSessionDriver(
        run_root=tmp_path / "run-root",
        rollout_max_turns=2,
    )

    def exhausted(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        stdout = _successful_stream("ses_exhausted", "Still working.")
        stdout += _successful_stream(
            "ses_exhausted",
            "## Maximum Steps Reached\n\nWork is incomplete.",
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(OpenCodeSessionDriver, "_run_command", exhausted)

    rollout = driver.spawn(SessionSeed(instruction="Do it", workspace=_lease(worktree)))

    assert rollout.summary == "## Maximum Steps Reached\n\nWork is incomplete."
    assert rollout.state.metadata["driver_execution"]["result_subtype"] == "error_max_turns"
    assert rollout.state.metadata["driver_execution"]["accepted_partial_result"] is True


def test_opencode_accepts_completion_on_final_allowed_turn(monkeypatch, tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    driver = OpenCodeSessionDriver(
        run_root=tmp_path / "run-root",
        rollout_max_turns=2,
    )

    def complete(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        stdout = _successful_stream("ses_complete", "Still working.")
        stdout += _successful_stream("ses_complete", "Finished.")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(OpenCodeSessionDriver, "_run_command", complete)

    rollout = driver.spawn(SessionSeed(instruction="Do it", workspace=_lease(worktree)))

    assert rollout.summary == "Finished."


def test_opencode_accepts_structural_empty_text_parts() -> None:
    stdout = "\n".join(
        [
            _event("text", "ses_empty", {"type": "text", "text": ""}),
            _successful_stream("ses_empty", "Finished."),
        ]
    )
    assert OpenCodeSessionDriver._parse_stdout_jsonl(stdout).summary == "Finished."


def test_opencode_error_event_is_fatal() -> None:
    stdout = json.dumps(
        {
            "type": "error",
            "timestamp": 1_700_000_000_000,
            "sessionID": "ses_error",
            "error": {"name": "ProviderAuthError"},
        }
    )
    with pytest.raises(RuntimeError, match="ProviderAuthError"):
        OpenCodeSessionDriver._parse_stdout_jsonl(stdout)


def test_opencode_nonzero_exit_and_timeout_are_fatal(monkeypatch, tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    driver = OpenCodeSessionDriver(run_root=tmp_path / "run-root", timeout_seconds=0.1)

    runtime_homes: list[Path] = []

    def leave_runtime_residue(env) -> None:  # noqa: ANN001
        home_root = Path(env["HOME"])
        runtime_homes.append(home_root)
        (home_root / ".future-opencode-cache").mkdir()

    def failed(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        leave_runtime_residue(env)
        return subprocess.CompletedProcess(command, 2, "", "provider failed")

    monkeypatch.setattr(OpenCodeSessionDriver, "_run_command", failed)
    with pytest.raises(RuntimeError, match="OpenCode run failed"):
        driver.spawn(SessionSeed(instruction="Do it", workspace=_lease(worktree)))
    assert not runtime_homes[-1].exists()

    def timed_out(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        leave_runtime_residue(env)
        raise subprocess.TimeoutExpired(command, 0.1, output="partial", stderr="slow")

    monkeypatch.setattr(OpenCodeSessionDriver, "_run_command", timed_out)
    with pytest.raises(RuntimeError, match="OpenCode run timed out"):
        driver.spawn(SessionSeed(instruction="Do it", workspace=_lease(worktree)))
    assert not runtime_homes[-1].exists()


def test_opencode_rollout_env_projects_stored_auth_only_without_explicit_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_data_root = tmp_path / "host-data"
    source_auth = source_data_root / "opencode" / "auth.json"
    source_auth.parent.mkdir(parents=True)
    source_auth.write_text("stored credential\n", encoding="utf-8")
    (source_auth.parent / "session.json").write_text("host state\n", encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(source_data_root))
    driver = OpenCodeSessionDriver(run_root=tmp_path / "run-root")

    rollout_env = driver._rollout_env(
        tmp_path / "workspace",
        home_root=tmp_path / "runtime-home",
    )

    isolated_data_root = Path(rollout_env["XDG_DATA_HOME"])
    isolated_auth = isolated_data_root / "opencode" / "auth.json"
    assert isolated_auth.is_symlink()
    assert isolated_auth.resolve() == source_auth.resolve()
    assert not (isolated_data_root / "opencode" / "session.json").exists()
    isolated_auth.unlink()
    assert source_auth.read_text(encoding="utf-8") == "stored credential\n"

    explicit_driver = OpenCodeSessionDriver(
        run_root=tmp_path / "explicit-run-root",
        provider_env={"DEEPSEEK_API_KEY": "explicit credential"},
    )
    explicit_env = explicit_driver._rollout_env(
        tmp_path / "explicit-workspace",
        home_root=tmp_path / "explicit-runtime-home",
    )
    explicit_auth = Path(explicit_env["XDG_DATA_HOME"]) / "opencode" / "auth.json"
    assert explicit_env["DEEPSEEK_API_KEY"] == "explicit credential"
    assert not explicit_auth.exists()
    assert not explicit_auth.is_symlink()

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "missing-host-data"))
    missing_env = driver._rollout_env(
        tmp_path / "missing-workspace",
        home_root=tmp_path / "missing-runtime-home",
    )
    missing_auth = Path(missing_env["XDG_DATA_HOME"]) / "opencode" / "auth.json"
    assert not missing_auth.exists()
    assert not missing_auth.is_symlink()


def test_opencode_config_and_factory_use_backend_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"permission": {"bash": "deny"}}),
    )
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "host-opencode.json"))
    system_prompt = tmp_path / "SYSTEM_PROMPT.md"
    system_prompt.write_text("Use the project workflow.\n", encoding="utf-8")
    provider = AgentProviderConfig.model_validate(
        {
            "kind": "agent_fork",
            "driver": "opencode",
            "model": "deepseek/deepseek-chat",
            "variant": "high",
        }
    )
    driver = build_driver(
        {
            "driver": "opencode",
            "model": "deepseek/deepseek-chat",
            "variant": "high",
            "system_prompt_file": str(system_prompt),
            "rollout_max_turns": 6,
            "budget_prompt": False,
            "model_providers": {
                "deepseek": {
                    "env_key": "DEEPSEEK_API_KEY",
                }
            },
            "token_pricing": {
                "input_per_million": 0.25,
                "output_per_million": 1.0,
            },
        },
        run_root=tmp_path,
    )

    assert provider.executable == "opencode"
    assert provider.budget_prompt is False
    assert provider.variant == "high"
    assert isinstance(driver, OpenCodeSessionDriver)
    assert driver.executable == "opencode"
    assert driver.model == "deepseek/deepseek-chat"
    assert driver.variant == "high"
    assert driver.system_prompt_file == system_prompt.resolve()
    assert driver.rollout_max_turns == 6
    assert driver.provider_env == {"DEEPSEEK_API_KEY": "test-key"}
    assert driver.token_pricing is not None
    assert driver.token_pricing.input_per_million == pytest.approx(0.25)
    with tempfile.TemporaryDirectory(prefix="scaling-evolve-opencode-test-") as home_root:
        rollout_env = driver._rollout_env(
            tmp_path / "workspace",
            home_root=Path(home_root),
        )
        assert "permission" not in json.loads(rollout_env["OPENCODE_CONFIG_CONTENT"])
        assert rollout_env["OPENCODE_CONFIG"] != os.environ["OPENCODE_CONFIG"]


def test_driver_specific_options_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rollout_max_turns"):
        OpenCodeSessionDriver(run_root=tmp_path, rollout_max_turns=0)
    with pytest.raises(ValueError, match="enable_multi_agent"):
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "opencode",
                "enable_multi_agent": True,
            }
        )
    with pytest.raises(ValueError, match="variant"):
        AgentProviderConfig.model_validate(
            {
                "kind": "agent_fork",
                "driver": "codex_exec",
                "variant": "high",
            }
        )
    with pytest.raises(SystemExit, match="budget_prompt"):
        build_driver(
            {"driver": "opencode", "budget_prompt": True},
            run_root=tmp_path,
        )
    with pytest.raises(SystemExit, match="reasoning_effort"):
        build_driver(
            {
                "driver": "opencode",
                "budget_prompt": False,
                "reasoning_effort": "high",
            },
            run_root=tmp_path,
        )
    with pytest.raises(SystemExit, match="model_provider"):
        build_driver(
            {
                "driver": "opencode",
                "budget_prompt": False,
                "model_provider": "custom-codex-provider",
            },
            run_root=tmp_path,
        )
    with pytest.raises(SystemExit, match=r"model_providers\.deepseek\.base_url"):
        build_driver(
            {
                "driver": "opencode",
                "budget_prompt": False,
                "model_providers": {
                    "deepseek": {
                        "env_key": "DEEPSEEK_API_KEY",
                        "base_url": "https://example.invalid",
                    }
                },
            },
            run_root=tmp_path,
        )
    with pytest.raises(SystemExit, match="variant"):
        build_driver(
            {"driver": "codex_exec", "variant": "high"},
            run_root=tmp_path,
        )


@pytest.mark.live
@pytest.mark.skipif(
    not _LIVE_OPENCODE or not _LIVE_MODEL,
    reason=(
        "Set SCALING_EVOLVE_RUN_LIVE_OPENCODE_TESTS=1 and "
        "SCALING_EVOLVE_OPENCODE_MODEL to run live OpenCode tests."
    ),
)
def test_opencode_live_spawn_and_exact_resume(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    _init_git_repo(outer)
    worktree = outer / "workspace"
    worktree.mkdir()
    (worktree / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
    outer_skill = outer / ".agents" / "skills" / "outer-only" / "SKILL.md"
    outer_skill.parent.mkdir(parents=True)
    outer_skill.write_text(
        "---\nname: outer-only\ndescription: Must stay outside EvE.\n---\n",
        encoding="utf-8",
    )
    workspace_skill = worktree / "guidance" / "skills" / "eve-only" / "SKILL.md"
    workspace_skill.parent.mkdir(parents=True)
    workspace_skill.write_text(
        "---\nname: eve-only\ndescription: EvE-owned skill.\n---\n",
        encoding="utf-8",
    )
    driver = OpenCodeSessionDriver(
        run_root=tmp_path / "run-root",
        executable=os.environ.get("SCALING_EVOLVE_OPENCODE_EXECUTABLE", "opencode"),
        model=_LIVE_MODEL,
        rollout_max_turns=10,
        timeout_seconds=300,
    )
    install_workspace_runtime_hooks(worktree, driver=driver, prompt_specs=[])
    with tempfile.TemporaryDirectory(prefix="scaling-evolve-opencode-test-") as home_root:
        debug_env = os.environ.copy()
        debug_env.update(driver._rollout_env(worktree, home_root=Path(home_root)))
        debug = subprocess.run(
            [driver.executable, "debug", "skill"],
            cwd=worktree,
            env=debug_env,
            check=True,
            capture_output=True,
            text=True,
        )
    skill_names = {item["name"] for item in json.loads(debug.stdout)}
    assert "eve-only" in skill_names
    assert "outer-only" not in skill_names

    spawned = driver.spawn(
        SessionSeed(
            instruction=(
                "Open candidate.py, replace `VALUE = 1` with `VALUE = 2`, save the file, and stop."
            ),
            workspace=_lease(worktree),
        )
    )
    resumed = driver.resume(
        spawned.state,
        instruction=(
            "Continue this exact session. Replace `VALUE = 2` with `VALUE = 3`, "
            "save the file, and stop."
        ),
    )

    assert spawned.state.session_id == resumed.state.session_id
    assert (worktree / "candidate.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert resumed.changed_paths == ["candidate.py"]
    transcript = Path(resumed.state.metadata["provider_transcript_path"])
    assert transcript.read_text(encoding="utf-8").count('"type": "eve.rollout"') == 2
    runtime_root = worktree / ".opencode-driver-transcripts"
    assert not list(runtime_root.rglob("node_modules"))
    assert not (runtime_root / "home").exists()
