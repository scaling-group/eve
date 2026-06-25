#!/usr/bin/env python3
"""Run the ICON remote-cluster workflow inside one PBS allocation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OrchestratorFailure(RuntimeError):
    """Base class for classified orchestrator failures."""

    status = "unclear"

    def __init__(self, stage: str, message: str, traceback_text: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.traceback_text = traceback_text


class CandidateFailure(OrchestratorFailure):
    status = "candidate_failure"


class InfraFailure(OrchestratorFailure):
    status = "infra_failure"


@dataclass
class OrchestratorConfig:
    task: str
    attempt_id: str
    staging_dir: Path
    remote_repo_path: Path
    remote_branch: str
    remote_worktree: Path
    editable_files: list[str]
    editable_folders: list[str]
    max_steps: int
    val_every: int
    save_every: int
    demo_nums_csv: str
    task_name_prefix: str
    bs: str
    lr: str
    num_workers: str
    data_dir: str
    log_dir: str
    train_timeout_seconds: int
    eval_timeout_seconds: int
    check_eval_timeout_seconds: int
    train_script_path: Path
    eval_script_path: Path
    import_runner: list[str]
    task_name: str


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: UP017


def load_meta(staging_dir: Path) -> dict[str, Any]:
    meta_path = staging_dir / "meta.json"
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised from shell
        raise InfraFailure(
            "overlay",
            f"Failed to read {meta_path}",
            traceback.format_exc(),
        ) from exc
    if not isinstance(loaded, dict):
        raise InfraFailure("overlay", f"Expected {meta_path} to contain a JSON object")
    return loaded


def default_import_runner(user: str) -> list[str]:
    configured_python = os.environ.get("EVE_REMOTE_PYTHON")
    if configured_python:
        return [configured_python]
    python_bin = Path(f"/scratch/{user}/envs/venvs/icon-core/bin/python")
    if python_bin.is_file():
        return [str(python_bin)]
    return ["python3"]


def build_config(*, task: str, staging_dir: Path, meta: dict[str, Any]) -> OrchestratorConfig:
    user = os.environ.get("USER", "unknown")
    attempt_id = str(meta.get("attempt_id") or "")
    if not attempt_id:
        raise InfraFailure("overlay", "meta.json did not contain attempt_id")

    remote_repo_path = Path(str(meta.get("remote_repo") or "~/repos/icon-core")).expanduser()
    remote_branch = str(meta.get("remote_branch") or "main")
    editable_files = [str(item) for item in meta.get("editable_files") or []]
    editable_folders = [str(item) for item in meta.get("editable_folders") or []]

    default_max_steps = 50 if task == "check" else 2000
    max_steps = int(meta.get("max_steps", default_max_steps))
    val_every = int(meta.get("val_every", max_steps))
    save_every = int(meta.get("save_every", max_steps))
    demo_nums_value = meta.get("demo_nums", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if isinstance(demo_nums_value, list):
        demo_nums_csv = ",".join(str(item) for item in demo_nums_value)
    else:
        demo_nums_csv = str(demo_nums_value)
    task_name_prefix = str(meta.get("task_name_prefix") or "eve_icon_evolve_iter")
    bs = str(meta.get("bs") or os.environ.get("BS") or "32")
    lr = str(meta.get("lr") or os.environ.get("LR") or "5e-4")
    num_workers = str(meta.get("num_workers") or os.environ.get("NUM_WORKERS") or "32")
    data_dir = str(
        meta.get("data_dir")
        or os.environ.get("DATA_DIR")
        or os.environ.get("EVE_REMOTE_DATA_DIR")
        or f"/scratch/{user}/data"
    )
    log_dir = str(meta.get("log_dir") or os.environ.get("LOG_DIR") or f"/scratch/{user}/logs")
    train_timeout_seconds = int(
        meta.get("train_timeout_seconds", 1200 if task == "check" else 10800)
    )
    eval_timeout_seconds = int(meta.get("eval_timeout_seconds", 3600))
    check_eval_timeout_seconds = int(meta.get("check_eval_timeout_seconds", 90))
    train_script_path = remote_repo_path / str(meta.get("train_script") or "scripts/evolve_iter.sh")
    eval_script_path = remote_repo_path / str(
        meta.get("eval_script") or "scripts/eval_context_length.sh"
    )
    remote_worktree = Path(f"/scratch/{user}/evolve_workspaces/{attempt_id}/repo")
    task_name = f"{task_name_prefix}_bs{bs}x2_lr{lr}_s{max_steps}_{attempt_id}"

    import_runner_value = meta.get("import_runner")
    if isinstance(import_runner_value, list) and import_runner_value:
        import_runner = [str(item) for item in import_runner_value]
    else:
        import_runner = default_import_runner(user)

    return OrchestratorConfig(
        task=task,
        attempt_id=attempt_id,
        staging_dir=staging_dir,
        remote_repo_path=remote_repo_path,
        remote_branch=remote_branch,
        remote_worktree=remote_worktree,
        editable_files=editable_files,
        editable_folders=editable_folders,
        max_steps=max_steps,
        val_every=val_every,
        save_every=save_every,
        demo_nums_csv=demo_nums_csv,
        task_name_prefix=task_name_prefix,
        bs=bs,
        lr=lr,
        num_workers=num_workers,
        data_dir=data_dir,
        log_dir=log_dir,
        train_timeout_seconds=train_timeout_seconds,
        eval_timeout_seconds=eval_timeout_seconds,
        check_eval_timeout_seconds=check_eval_timeout_seconds,
        train_script_path=train_script_path,
        eval_script_path=eval_script_path,
        import_runner=import_runner,
        task_name=task_name,
    )


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    stage: str,
    infra: bool = False,
) -> None:
    failure_type = InfraFailure if infra else CandidateFailure
    try:
        subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            timeout=timeout_seconds,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise failure_type(
            stage,
            f"{' '.join(command)} exceeded timeout after {timeout_seconds}s",
            None,
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise failure_type(
            stage,
            f"{' '.join(command)} exited with return code {exc.returncode}",
            None,
        ) from exc


def create_worktree(config: OrchestratorConfig) -> None:
    try:
        config.remote_worktree.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            ["git", "fetch", "origin", config.remote_branch],
            cwd=config.remote_repo_path,
            stage="worktree_create",
            infra=True,
        )
        run_command(
            [
                "git",
                "worktree",
                "add",
                str(config.remote_worktree),
                f"origin/{config.remote_branch}",
            ],
            cwd=config.remote_repo_path,
            stage="worktree_create",
            infra=True,
        )
    except FileExistsError as exc:
        raise InfraFailure(
            "worktree_create",
            f"Failed to prepare {config.remote_worktree}",
        ) from exc

    required_paths = [
        config.remote_worktree / "src/eval_context_length_batch.py",
        config.remote_worktree / config.train_script_path.relative_to(config.remote_repo_path),
    ]
    if config.task in {"evaluate", "check"}:
        required_paths.append(
            config.remote_worktree / config.eval_script_path.relative_to(config.remote_repo_path)
        )
    for path in required_paths:
        if not path.exists():
            raise InfraFailure(
                "worktree_create",
                f"Required path missing after worktree add: {path}",
            )


def overlay_inputs(config: OrchestratorConfig) -> None:
    inputs_root = config.staging_dir / "inputs"

    for rel_path in config.editable_files:
        source = inputs_root / rel_path
        target = config.remote_worktree / rel_path
        if not source.is_file():
            raise InfraFailure("overlay", f"Missing staged file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for rel_path in config.editable_folders:
        source = inputs_root / rel_path
        target = config.remote_worktree / rel_path
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            raise InfraFailure("overlay", f"Expected staged folder at {source}")


def verify_import(config: OrchestratorConfig) -> None:
    command = [
        *config.import_runner,
        "-c",
        (
            "import os; "
            "import src.eval_context_length_batch as module; "
            "print(f'CWD={os.getcwd()}'); "
            "print(f'FILE={module.__file__}')"
        ),
    ]
    run_command(command, cwd=config.remote_worktree, timeout_seconds=300, stage="import_check")


def run_training(config: OrchestratorConfig) -> None:
    train_env = os.environ.copy()
    train_env.update(
        {
            "PBS_O_WORKDIR": str(config.remote_worktree),
            "TASK_TAG": config.attempt_id,
            "MAX_STEPS": str(config.max_steps),
            "VAL_EVERY": str(config.val_every),
            "SAVE_EVERY": str(config.save_every),
            "BS": config.bs,
            "LR": config.lr,
            "NUM_WORKERS": config.num_workers,
            "DATA_DIR": config.data_dir,
            "LOG_DIR": config.log_dir,
        }
    )
    train_script = config.remote_worktree / config.train_script_path.relative_to(
        config.remote_repo_path
    )
    run_command(
        ["bash", str(train_script)],
        cwd=config.remote_worktree,
        env=train_env,
        timeout_seconds=config.train_timeout_seconds,
        stage="training",
    )


def locate_checkpoint(config: OrchestratorConfig) -> str:
    run_root = Path(config.log_dir) / config.task_name / "runs"
    candidates = sorted(run_root.glob("*/checkpoints/last.ckpt"))
    if not candidates:
        raise CandidateFailure(
            "checkpoint_verify",
            f"Training completed without producing last.ckpt under {run_root}",
        )
    return str(candidates[-1])


def run_eval(config: OrchestratorConfig, checkpoint_path: str) -> tuple[str, dict[str, Any]]:
    summary_path = config.staging_dir / "eval_summary.json"
    eval_env = os.environ.copy()
    eval_env.update(
        {
            "PBS_O_WORKDIR": str(config.remote_worktree),
            "CKPT": checkpoint_path,
            "DEMO_NUMS": config.demo_nums_csv,
            "OUT_DIR": str(config.staging_dir / "eval_context"),
            "SUMMARY_OUT": str(summary_path),
            "BS": config.bs,
            "DATA_DIR": config.data_dir,
        }
    )
    eval_script = config.remote_worktree / config.eval_script_path.relative_to(
        config.remote_repo_path
    )
    run_command(
        ["bash", str(eval_script)],
        cwd=config.remote_worktree,
        env=eval_env,
        timeout_seconds=config.eval_timeout_seconds,
        stage="eval",
    )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateFailure(
            "eval",
            f"Failed to read {summary_path}",
            traceback.format_exc(),
        ) from exc

    score_metric = str(summary.get("score_metric") or "mean_d1_d10")
    score_components = dict(summary.get("score_components") or {})
    score_components.setdefault("score_metric", score_metric)
    if "optimized_score" not in score_components and "score" in summary:
        score_components["optimized_score"] = float(summary["score"])
    return score_metric, score_components


def _extract_max_demo_num(demo_nums_csv: str) -> int:
    values = [chunk.strip() for chunk in demo_nums_csv.replace(",", " ").split() if chunk.strip()]
    if not values:
        raise CandidateFailure("eval", "meta demo_nums was empty for check eval smoke")
    try:
        return int(values[-1])
    except ValueError as exc:
        raise CandidateFailure(
            "eval",
            f"Failed to parse max demo_num from {demo_nums_csv!r}",
            traceback.format_exc(),
        ) from exc


def _assert_finite_numbers(payload: Any, *, stage: str, field_path: str = "summary") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            _assert_finite_numbers(value, stage=stage, field_path=f"{field_path}.{key}")
        return
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_finite_numbers(value, stage=stage, field_path=f"{field_path}[{index}]")
        return
    if isinstance(payload, (int, float)) and not math.isfinite(float(payload)):
        raise CandidateFailure(stage, f"Non-finite numeric value detected at {field_path}")


def run_eval_smoke(config: OrchestratorConfig, checkpoint_path: str) -> None:
    demo_num = _extract_max_demo_num(config.demo_nums_csv)
    summary_path = config.staging_dir / "check_eval_summary.json"
    eval_env = os.environ.copy()
    eval_env.update(
        {
            "PBS_O_WORKDIR": str(config.remote_worktree),
            "CKPT": checkpoint_path,
            "DEMO_NUMS": str(demo_num),
            "OUT_DIR": str(config.staging_dir / "check_eval_context"),
            "SUMMARY_OUT": str(summary_path),
            "BS": config.bs,
            "DATA_DIR": config.data_dir,
        }
    )
    try:
        run_command(
            [
                "bash",
                str(
                    config.remote_worktree
                    / config.eval_script_path.relative_to(config.remote_repo_path)
                ),
            ],
            cwd=config.remote_worktree,
            env=eval_env,
            timeout_seconds=config.check_eval_timeout_seconds,
            stage="eval",
        )
    except CandidateFailure as exc:
        raise CandidateFailure(
            "eval",
            f"max-demo eval smoke failed at demo_num={demo_num}: {exc.message}",
            exc.traceback_text,
        ) from exc

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateFailure(
            "eval",
            f"Failed to read {summary_path} for demo_num={demo_num}",
            traceback.format_exc(),
        ) from exc

    _assert_finite_numbers(summary, stage="eval")


def cleanup_worktree(config: OrchestratorConfig | None) -> None:
    if config is None:
        return
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(config.remote_worktree)],
            cwd=str(config.remote_repo_path),
            check=False,
        )
        subprocess.run(["git", "worktree", "prune"], cwd=str(config.remote_repo_path), check=False)
    except Exception:
        pass


def write_result_json(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, result_path)


def result_payload(*, task: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": task,
        "status": "unclear",
        "attempt_id": "",
        "started_at": timestamp_utc(),
        "finished_at": None,
        "pbs_job_id": os.environ.get("PBS_JOBID", ""),
        "remote_worktree": "",
        "checkpoint_path": None,
        "score_metric": None,
        "score_components": None,
        "error": None,
        "host": socket.gethostname(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("check", "evaluate"), required=True)
    parser.add_argument("--staging-dir", required=True)
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir)
    result_path = staging_dir / "result.json"
    result = result_payload(task=args.task)
    config: OrchestratorConfig | None = None

    try:
        meta = load_meta(staging_dir)
        config = build_config(task=args.task, staging_dir=staging_dir, meta=meta)
        result["attempt_id"] = config.attempt_id
        result["remote_worktree"] = str(config.remote_worktree)

        create_worktree(config)
        overlay_inputs(config)
        verify_import(config)
        run_training(config)
        checkpoint_path = locate_checkpoint(config)
        result["checkpoint_path"] = checkpoint_path

        if args.task == "evaluate":
            score_metric, score_components = run_eval(config, checkpoint_path)
            result["score_metric"] = score_metric
            result["score_components"] = score_components
        else:
            run_eval_smoke(config, checkpoint_path)

        result["status"] = "pass"
    except OrchestratorFailure as exc:
        result["status"] = exc.status
        result["error"] = {
            "stage": exc.stage,
            "message": exc.message,
            "traceback": exc.traceback_text,
        }
    except Exception as exc:  # pragma: no cover - exercised from shell
        result["status"] = "unclear"
        result["error"] = {
            "stage": "result_write",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        cleanup_worktree(config)
        result["finished_at"] = timestamp_utc()

    try:
        write_result_json(result_path, result)
    except Exception:  # pragma: no cover - exercised from shell
        print(traceback.format_exc(), file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
