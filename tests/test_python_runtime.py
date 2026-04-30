"""Runtime policy tests for workspace-local Python execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scaling_evolve.providers.agent.python_runtime import (
    PythonExecutionPolicy,
    write_python_runtime,
)


def _write_sandbox_config(
    workspace: Path,
    *,
    evaluator_dir: Path,
    policy: dict[str, object],
) -> None:
    (workspace / ".sandbox_config.json").write_text(
        json.dumps(
            {
                "own_workspace": str(workspace),
                "evaluator_dirs": [str(evaluator_dir)],
                "execution_policy": policy,
            }
        ),
        encoding="utf-8",
    )


def _run_wrapped_python(
    workspace: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    wrapper = workspace / ".agent-runtime" / "bin" / "python"
    return subprocess.run(
        [str(wrapper), *args],
        cwd=workspace,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_wrapped_python_scrubs_environment_and_blocks_subprocess_and_network(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir()
    policy = PythonExecutionPolicy(
        managed=True,
        allow_network=False,
        allow_subprocess=False,
        allowed_env_vars=(),
    )
    _write_sandbox_config(workspace, evaluator_dir=evaluator_dir, policy=policy.to_payload())
    write_python_runtime(workspace, policy=policy, real_python=Path(sys.executable))

    env_result = _run_wrapped_python(
        workspace,
        "-c",
        "import os; print(os.getenv('SECRET_TOKEN', ''))",
        env={"SECRET_TOKEN": "top-secret"},
    )
    subprocess_result = _run_wrapped_python(
        workspace,
        "-c",
        "import subprocess; subprocess.run(['echo', 'hi'], check=True)",
    )
    network_result = _run_wrapped_python(
        workspace,
        "-c",
        "import socket; socket.create_connection(('127.0.0.1', 80), timeout=0.01)",
    )

    assert env_result.returncode == 0
    assert env_result.stdout.strip() == ""
    assert subprocess_result.returncode != 0
    assert "subprocess" in subprocess_result.stderr.lower()
    assert network_result.returncode != 0
    assert "network" in network_result.stderr.lower()


def test_wrapped_python_allows_local_writes_but_blocks_outside_writes_and_evaluator_reads(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir()
    secret = evaluator_dir / "secret.py"
    secret.write_text("SECRET = 1\n", encoding="utf-8")
    outside_file = tmp_path / "outside.txt"
    policy = PythonExecutionPolicy(
        managed=True,
        allow_network=False,
        allow_subprocess=False,
        allowed_env_vars=(),
    )
    _write_sandbox_config(workspace, evaluator_dir=evaluator_dir, policy=policy.to_payload())
    write_python_runtime(workspace, policy=policy, real_python=Path(sys.executable))

    local_result = _run_wrapped_python(
        workspace,
        "-c",
        "from pathlib import Path; Path('local.txt').write_text('ok', encoding='utf-8')",
    )
    outside_result = _run_wrapped_python(
        workspace,
        "-c",
        f"from pathlib import Path; Path({str(outside_file)!r}).write_text('no', encoding='utf-8')",
    )
    evaluator_result = _run_wrapped_python(
        workspace,
        "-c",
        f"print(open({str(secret)!r}, 'r', encoding='utf-8').read())",
    )

    assert local_result.returncode == 0
    assert (workspace / "local.txt").read_text(encoding="utf-8") == "ok"
    assert outside_result.returncode != 0
    assert "outside your workspace" in outside_result.stderr.lower()
    assert evaluator_result.returncode != 0
    assert "evaluator" in evaluator_result.stderr.lower()


def test_wrapped_python_blocks_link_and_symlink_to_outside_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret\n", encoding="utf-8")
    policy = PythonExecutionPolicy(
        managed=True,
        allow_network=False,
        allow_subprocess=False,
        allowed_env_vars=(),
    )
    _write_sandbox_config(workspace, evaluator_dir=evaluator_dir, policy=policy.to_payload())
    write_python_runtime(workspace, policy=policy, real_python=Path(sys.executable))

    link_result = _run_wrapped_python(
        workspace,
        "-c",
        (f"import os; os.link({str(outside_file)!r}, 'linked.txt')"),
    )
    symlink_result = _run_wrapped_python(
        workspace,
        "-c",
        (
            "from pathlib import Path; import os; "
            f"os.symlink({str(outside_file)!r}, 'linked.txt'); "
            "Path('linked.txt').write_text('mutated', encoding='utf-8')"
        ),
    )

    assert link_result.returncode != 0
    assert "outside your workspace" in link_result.stderr.lower()
    assert symlink_result.returncode != 0
    assert "outside your workspace" in symlink_result.stderr.lower()


def test_wrapped_python_allows_workspace_symlink_to_readable_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir()
    policy = PythonExecutionPolicy(
        managed=True,
        allow_network=False,
        allow_subprocess=False,
        allowed_env_vars=(),
    )
    _write_sandbox_config(workspace, evaluator_dir=evaluator_dir, policy=policy.to_payload())
    write_python_runtime(workspace, policy=policy, real_python=Path(sys.executable))

    symlink_result = _run_wrapped_python(
        workspace,
        "-c",
        (
            "from pathlib import Path; import os; "
            f"os.symlink({os.devnull!r}, 'null-link'); "
            "print(Path('null-link').is_symlink())"
        ),
    )

    assert symlink_result.returncode == 0
    assert symlink_result.stdout.strip() == "True"
