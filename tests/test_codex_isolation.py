from __future__ import annotations

from pathlib import Path

from scaling_evolve.providers.agent.codex_hooks import CODEX_HOOK_TRUST_EVENTS
from scaling_evolve.providers.agent.codex_isolation import (
    CodexLaunchConfig,
    create_isolated_codex_home,
    extract_last_assistant_message,
    extract_session_id,
    extract_usage,
    latest_rollout_path,
    reset_isolated_codex_home,
    rollout_path_for_session,
)


def test_create_isolated_codex_home_symlinks_auth_and_writes_minimal_config(tmp_path: Path) -> None:
    source_auth = tmp_path / "source-auth.json"
    source_auth.write_text('{"token":"abc"}\n', encoding="utf-8")

    isolated = create_isolated_codex_home(
        home_root=tmp_path / "home",
        source_auth_path=source_auth,
        launch=CodexLaunchConfig(worktree_root=tmp_path / "repo"),
    )

    config_text = isolated.config_path.read_text(encoding="utf-8")
    assert isolated.auth_path.is_symlink()
    assert isolated.auth_path.resolve() == source_auth.resolve()
    assert isolated.auth_path.read_text(encoding="utf-8") == '{"token":"abc"}\n'
    assert "[features]" in config_text
    assert "hooks = true" in config_text
    assert "codex_hooks = true" not in config_text
    assert "model = " not in config_text
    assert "model_reasoning_effort = " not in config_text
    assert "[projects." in config_text
    assert 'trust_level = "trusted"' in config_text


def test_create_isolated_codex_home_escapes_project_path(tmp_path: Path) -> None:
    isolated = create_isolated_codex_home(
        home_root=tmp_path / "home",
        source_auth_path=None,
        launch=CodexLaunchConfig(worktree_root=tmp_path / 'repo "quoted"'),
    )

    config_text = isolated.config_path.read_text(encoding="utf-8")
    assert "[projects." in config_text
    assert '\\"quoted\\"' in config_text


def test_create_isolated_codex_home_copies_trusted_hook_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    real_home = tmp_path / "real-home"
    config_path = real_home / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    hooks_json_path = tmp_path / "repo" / ".codex" / "hooks.json"
    hooks_json_path.parent.mkdir(parents=True)
    hooks_json_path.write_text("{}\n", encoding="utf-8")
    other_hooks_path = tmp_path / "other" / ".codex" / "hooks.json"
    entries = []
    for event in CODEX_HOOK_TRUST_EVENTS:
        entries.append(
            "\n".join(
                [
                    f'[hooks.state."{hooks_json_path.resolve()}:{event}:0:0"]',
                    f'trusted_hash = "sha256:{event}"',
                    "",
                ]
            )
        )
    entries.append(
        "\n".join(
            [
                f'[hooks.state."{other_hooks_path.resolve()}:pre_tool_use:0:0"]',
                'trusted_hash = "sha256:other"',
                "",
            ]
        )
    )
    config_path.write_text("".join(entries), encoding="utf-8")
    monkeypatch.setenv("HOME", str(real_home))

    isolated = create_isolated_codex_home(
        home_root=tmp_path / "isolated-home",
        source_auth_path=None,
        launch=CodexLaunchConfig(
            worktree_root=tmp_path / "workspace",
            hooks_json_path=hooks_json_path,
        ),
    )

    config_text = isolated.config_path.read_text(encoding="utf-8")
    for event in CODEX_HOOK_TRUST_EVENTS:
        assert f"{hooks_json_path.resolve()}:{event}:0:0" in config_text
        assert f'trusted_hash = "sha256:{event}"' in config_text
    assert f"[projects.{str((tmp_path / 'workspace').resolve())!r}]" not in config_text
    assert f'[projects."{(tmp_path / "workspace").resolve()}"]' in config_text
    assert f'[projects."{(tmp_path / "repo").resolve()}"]' in config_text
    assert str(other_hooks_path.resolve()) not in config_text


def test_rollout_helpers_extract_summary_and_usage(tmp_path: Path) -> None:
    session_path = tmp_path / ".codex" / "sessions" / "2026" / "04" / "03" / "rollout-test.jsonl"
    session_path.parent.mkdir(parents=True)
    session_path.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"session-123"}}',
                '{"type":"event_msg","payload":{"type":"agent_message","message":"first"}}',
                '{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":12,"cached_input_tokens":3,"output_tokens":4,"reasoning_output_tokens":1}}}}',
                '{"type":"event_msg","payload":{"type":"agent_message","message":"final summary"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert latest_rollout_path(tmp_path) == session_path
    assert rollout_path_for_session(tmp_path, session_id="session-123") == session_path
    assert extract_session_id(session_path) == "session-123"
    assert extract_last_assistant_message(session_path) == "final summary"
    assert extract_usage(session_path) == {
        "input_tokens": 12,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_output_tokens": 1,
        "agent_turns": 2,
    }
    assert extract_usage(session_path, from_line=2) == {
        "input_tokens": 12,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_output_tokens": 1,
        "agent_turns": 1,
    }


def test_reset_isolated_codex_home_prunes_runtime_residue(tmp_path: Path) -> None:
    source_auth = tmp_path / "source-auth.json"
    source_auth.write_text('{"token":"abc"}\n', encoding="utf-8")

    isolated = create_isolated_codex_home(
        home_root=tmp_path / "home",
        source_auth_path=source_auth,
        launch=CodexLaunchConfig(worktree_root=tmp_path / "repo"),
    )
    sessions_dir = isolated.codex_dir / "sessions" / "2026" / "04" / "05"
    sessions_dir.mkdir(parents=True)
    rollout = sessions_dir / "rollout-test.jsonl"
    rollout.write_text('{"type":"session_meta","payload":{"id":"session-1"}}\n', encoding="utf-8")
    (isolated.codex_dir / "cache").mkdir()
    (isolated.codex_dir / "models_cache.json").write_text("{}", encoding="utf-8")

    reset = reset_isolated_codex_home(
        home=isolated,
        source_auth_path=source_auth,
        launch=CodexLaunchConfig(worktree_root=tmp_path / "repo"),
    )

    assert reset.auth_path.is_symlink()
    assert rollout.exists()
    assert not (reset.codex_dir / "cache").exists()
    assert not (reset.codex_dir / "models_cache.json").exists()
    assert reset.config_path.read_text(encoding="utf-8").strip().endswith('trust_level = "trusted"')
