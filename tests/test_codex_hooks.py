from __future__ import annotations

import json
from pathlib import Path

import pytest

from scaling_evolve.providers.agent.codex_hooks import (
    CODEX_HOOK_TRUST_EVENTS,
    ensure_codex_hooks_trusted,
    find_repo_root,
    missing_hook_trust_events,
    write_repo_codex_hooks,
)


def test_write_repo_codex_hooks_creates_repo_level_hooks_json(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "package"
    nested.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    hooks_path = write_repo_codex_hooks(nested)

    assert find_repo_root(nested) == repo
    assert hooks_path == repo / ".codex" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))
    command = payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert f"PYTHONPATH={repo / 'src'}" in command
    assert "scaling_evolve.providers.agent.hooks.workspace_guard" in command


def test_missing_hook_trust_events_reads_hooks_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    hooks_path = repo / ".codex" / "hooks.json"
    config_path = _write_trust_config(
        tmp_path / "home" / ".codex" / "config.toml",
        hooks_path,
        events=CODEX_HOOK_TRUST_EVENTS[:-1],
    )

    assert missing_hook_trust_events(hooks_path, config_path=config_path) == ["user_prompt_submit"]


def test_ensure_codex_hooks_trusted_passes_with_all_entries(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    hooks_path = repo / ".codex" / "hooks.json"
    config_path = _write_trust_config(
        tmp_path / "home" / ".codex" / "config.toml",
        hooks_path,
        events=CODEX_HOOK_TRUST_EVENTS,
        repo_root=repo,
    )

    ensure_codex_hooks_trusted(repo, config_path=config_path)


def test_ensure_codex_hooks_trusted_exits_with_instructions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    hooks_path = repo / ".codex" / "hooks.json"
    config_path = _write_trust_config(
        tmp_path / "home" / ".codex" / "config.toml",
        hooks_path,
        events=("pre_tool_use",),
    )

    with pytest.raises(SystemExit) as exc_info:
        ensure_codex_hooks_trusted(repo, config_path=config_path)

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Codex hooks are not trusted for this repository." in stderr
    assert f"codex -C {repo}" in stderr
    assert "Then re-run Eve." in stderr


def _write_trust_config(
    config_path: Path,
    hooks_path: Path,
    *,
    events: tuple[str, ...],
    repo_root: Path | None = None,
) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    if repo_root is not None:
        body.append(f'[projects."{repo_root.resolve()}"]\ntrust_level = "trusted"\n\n')
    for event in events:
        body.append(
            "\n".join(
                [
                    f'[hooks.state."{hooks_path.resolve()}:{event}:0:0"]',
                    f'trusted_hash = "sha256:{event}"',
                    "",
                ]
            )
        )
    config_path.write_text("".join(body), encoding="utf-8")
    return config_path
