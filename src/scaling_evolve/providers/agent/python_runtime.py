"""Workspace-local Python runtime controls for benchmark-safe agent execution."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

RUNTIME_DIRNAME = ".agent-runtime"
PYTHON_POLICY_FILENAME = "python_policy.json"
_WRAPPER_NAMES = ("python", "python3")


@dataclass(frozen=True)
class PythonExecutionPolicy:
    """Resolved Python execution policy for workspace-local self-checks."""

    managed: bool
    allow_network: bool
    allow_subprocess: bool
    allowed_env_vars: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "managed": self.managed,
            "allow_network": self.allow_network,
            "allow_subprocess": self.allow_subprocess,
            "allowed_env_vars": list(self.allowed_env_vars),
        }


def _wrapper_source(*, real_python: Path) -> str:
    python_literal = repr(str(real_python))
    return textwrap.dedent(
        f"""\
        #!{real_python}
        from __future__ import annotations

        import fnmatch
        import json
        import os
        import sys
        from pathlib import Path

        REAL_PYTHON = {python_literal}
        RUNTIME_ROOT = Path(__file__).resolve().parents[1]
        POLICY_FILE = RUNTIME_ROOT / "{PYTHON_POLICY_FILENAME}"
        GUARD_DIR = str(RUNTIME_ROOT)
        ESSENTIAL_ENV_KEYS = (
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PATH",
            "PYTHONHOME",
            "PYTHONIOENCODING",
            "PY_COLORS",
            "TERM",
            "TMP",
            "TEMP",
            "TMPDIR",
            "TZ",
        )


        def _load_policy() -> dict[str, object]:
            if not POLICY_FILE.exists():
                return {{"managed": False}}
            try:
                return json.loads(POLICY_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {{"managed": False}}


        def _sanitized_env(policy: dict[str, object]) -> dict[str, str]:
            current = dict(os.environ)
            if not bool(policy.get("managed", False)):
                current["SCALING_EVOLVE_PYTHON_POLICY_FILE"] = str(POLICY_FILE)
                current["PYTHONDONTWRITEBYTECODE"] = "1"
                return current

            allowed_patterns = [
                item
                for item in policy.get("allowed_env_vars", [])
                if isinstance(item, str) and item
            ]
            env: dict[str, str] = {{}}
            for key in ESSENTIAL_ENV_KEYS:
                value = current.get(key)
                if value:
                    env[key] = value
            for key, value in current.items():
                if any(fnmatch.fnmatchcase(key, pattern) for pattern in allowed_patterns):
                    env[key] = value

            existing_pythonpath = current.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                GUARD_DIR
                if not existing_pythonpath
                else GUARD_DIR + os.pathsep + existing_pythonpath
            )
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["SCALING_EVOLVE_PYTHON_POLICY_FILE"] = str(POLICY_FILE)
            return env


        def main() -> None:
            policy = _load_policy()
            env = _sanitized_env(policy)
            os.execve(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]], env)


        if __name__ == "__main__":
            main()
        """
    )


def _sitecustomize_source() -> str:
    return textwrap.dedent(
        """\
        from __future__ import annotations

        import builtins
        import io
        import json
        import os
        import site
        import socket
        import subprocess
        import sys
        import sysconfig
        from pathlib import Path

        def _install_policy() -> None:
            policy_file = os.environ.get("SCALING_EVOLVE_PYTHON_POLICY_FILE")
            if not policy_file:
                return

            try:
                policy = json.loads(Path(policy_file).read_text(encoding="utf-8"))
            except Exception:
                policy = {}

            if not bool(policy.get("managed", False)):
                return

            own_workspace = Path(str(policy.get("own_workspace", "."))).expanduser().resolve(
                strict=False
            )
            workspace_root = own_workspace.parent
            evaluator_dirs = [
                Path(str(item)).expanduser().resolve(strict=False)
                for item in policy.get("evaluator_dirs", [])
                if isinstance(item, str) and item
            ]
            safe_device_reads = {
                Path("/dev/null"),
                Path("/dev/random"),
                Path("/dev/urandom"),
            }

            runtime_read_roots = {
                Path(__file__).resolve().parent,
                Path(sys.prefix).resolve(strict=False),
                Path(sys.base_prefix).resolve(strict=False),
            }
            for value in sysconfig.get_paths().values():
                if isinstance(value, str) and value:
                    runtime_read_roots.add(Path(value).expanduser().resolve(strict=False))
            for value in site.getsitepackages():
                if value:
                    runtime_read_roots.add(Path(value).expanduser().resolve(strict=False))

            def _is_under(path: Path, parent: Path) -> bool:
                try:
                    path.relative_to(parent)
                    return True
                except ValueError:
                    return False

            def _resolve_path(raw_path: str | os.PathLike[str]) -> Path:
                candidate = Path(os.fspath(raw_path))
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                return candidate.expanduser().resolve(strict=False)

            def _block(message: str) -> None:
                raise PermissionError(message)

            def _is_evaluator_path(path: Path) -> bool:
                return any(_is_under(path, evaluator_dir) for evaluator_dir in evaluator_dirs)

            def _allow_read(path: Path) -> bool:
                if _is_under(path, workspace_root):
                    return True
                if path in safe_device_reads:
                    return True
                return any(_is_under(path, runtime_root) for runtime_root in runtime_read_roots)

            def _check_file_access(path_like, *, write: bool) -> Path:
                resolved = _resolve_path(path_like)
                if _is_evaluator_path(resolved):
                    _block(f"Evaluator paths are off-limits: {resolved}")
                if write:
                    if not _is_under(resolved, own_workspace):
                        _block(f"Python write path is outside your workspace: {resolved}")
                    return resolved
                if not _allow_read(resolved):
                    _block(f"Python read path is outside permitted roots: {resolved}")
                return resolved

            orig_open = builtins.open
            orig_io_open = io.open
            orig_os_open = os.open
            orig_remove = os.remove
            orig_unlink = os.unlink
            orig_rename = os.rename
            orig_replace = os.replace
            orig_link = getattr(os, "link", None)
            orig_symlink = getattr(os, "symlink", None)
            orig_mkdir = os.mkdir
            orig_makedirs = os.makedirs
            orig_rmdir = os.rmdir
            orig_removedirs = os.removedirs

            def _mode_writes(mode: str) -> bool:
                return any(flag in mode for flag in ("w", "a", "x", "+"))

            def _guarded_open(file, mode="r", *args, **kwargs):
                if isinstance(file, int):
                    return orig_open(file, mode, *args, **kwargs)
                _check_file_access(file, write=_mode_writes(mode))
                return orig_open(file, mode, *args, **kwargs)

            def _guarded_io_open(file, mode="r", *args, **kwargs):
                if isinstance(file, int):
                    return orig_io_open(file, mode, *args, **kwargs)
                _check_file_access(file, write=_mode_writes(mode))
                return orig_io_open(file, mode, *args, **kwargs)

            def _guarded_os_open(file, flags, mode=0o777, *, dir_fd=None):
                write_flags = (
                    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
                )
                _check_file_access(file, write=bool(flags & write_flags))
                return orig_os_open(file, flags, mode, dir_fd=dir_fd)

            def _guarded_remove(path, *args, **kwargs):
                _check_file_access(path, write=True)
                return orig_remove(path, *args, **kwargs)

            def _guarded_unlink(path, *args, **kwargs):
                _check_file_access(path, write=True)
                return orig_unlink(path, *args, **kwargs)

            def _guarded_rename(src, dst, *args, **kwargs):
                _check_file_access(src, write=True)
                _check_file_access(dst, write=True)
                return orig_rename(src, dst, *args, **kwargs)

            def _guarded_replace(src, dst, *args, **kwargs):
                _check_file_access(src, write=True)
                _check_file_access(dst, write=True)
                return orig_replace(src, dst, *args, **kwargs)

            def _guarded_link(src, dst, *args, **kwargs):
                _check_file_access(src, write=True)
                _check_file_access(dst, write=True)
                if orig_link is None:
                    _block("os.link is unavailable in the managed Python runtime.")
                return orig_link(src, dst, *args, **kwargs)

            def _guarded_symlink(src, dst, *args, **kwargs):
                _check_file_access(src, write=False)
                _check_file_access(dst, write=True)
                if orig_symlink is None:
                    _block("os.symlink is unavailable in the managed Python runtime.")
                return orig_symlink(src, dst, *args, **kwargs)

            def _guarded_mkdir(path, *args, **kwargs):
                _check_file_access(path, write=True)
                return orig_mkdir(path, *args, **kwargs)

            def _guarded_makedirs(name, *args, **kwargs):
                _check_file_access(name, write=True)
                return orig_makedirs(name, *args, **kwargs)

            def _guarded_rmdir(path, *args, **kwargs):
                _check_file_access(path, write=True)
                return orig_rmdir(path, *args, **kwargs)

            def _guarded_removedirs(name, *args, **kwargs):
                _check_file_access(name, write=True)
                return orig_removedirs(name, *args, **kwargs)

            def _blocked_subprocess(*args, **kwargs):
                _block("Subprocess execution is disabled by the Python execution policy.")

            def _blocked_network(*args, **kwargs):
                _block("Network access is disabled by the Python execution policy.")

            builtins.open = _guarded_open
            io.open = _guarded_io_open
            os.open = _guarded_os_open
            os.remove = _guarded_remove
            os.unlink = _guarded_unlink
            os.rename = _guarded_rename
            os.replace = _guarded_replace
            if orig_link is not None:
                os.link = _guarded_link
            if orig_symlink is not None:
                os.symlink = _guarded_symlink
            os.mkdir = _guarded_mkdir
            os.makedirs = _guarded_makedirs
            os.rmdir = _guarded_rmdir
            os.removedirs = _guarded_removedirs

            if not bool(policy.get("allow_subprocess", True)):
                os.system = _blocked_subprocess
                subprocess.Popen = _blocked_subprocess
                subprocess.run = _blocked_subprocess
                subprocess.call = _blocked_subprocess
                subprocess.check_call = _blocked_subprocess
                subprocess.check_output = _blocked_subprocess
                for name in dir(os):
                    if name.startswith("exec") or name.startswith("spawn"):
                        setattr(os, name, _blocked_subprocess)

            if not bool(policy.get("allow_network", True)):
                socket.socket.connect = _blocked_network
                socket.socket.connect_ex = _blocked_network
                socket.create_connection = _blocked_network

        _install_policy()
        """
    )


def write_python_runtime(
    workspace_root: Path,
    *,
    policy: PythonExecutionPolicy,
    real_python: Path,
    evaluator_dirs: tuple[Path, ...] | None = None,
) -> None:
    """Write a workspace-local Python wrapper and policy payload."""

    workspace_root = workspace_root.expanduser().resolve(strict=False)
    runtime_root = workspace_root / RUNTIME_DIRNAME
    bin_dir = runtime_root / "bin"
    runtime_root.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    if evaluator_dirs is None:
        evaluator_dirs = ()
        config_path = workspace_root / ".sandbox_config.json"
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                evaluator_dirs = tuple(
                    Path(str(item)).expanduser().resolve(strict=False)
                    for item in payload.get("evaluator_dirs", [])
                    if isinstance(item, str) and item
                )

    payload = {
        **policy.to_payload(),
        "own_workspace": str(workspace_root),
        "evaluator_dirs": [str(path) for path in evaluator_dirs],
    }
    (runtime_root / PYTHON_POLICY_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (runtime_root / "sitecustomize.py").write_text(_sitecustomize_source(), encoding="utf-8")

    wrapper_source = _wrapper_source(real_python=real_python.expanduser().resolve(strict=False))
    for name in _WRAPPER_NAMES:
        wrapper_path = bin_dir / name
        wrapper_path.write_text(wrapper_source, encoding="utf-8")
        wrapper_path.chmod(0o755)
