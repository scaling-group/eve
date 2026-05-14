"""Probe one Eve agent role without running the full loop."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve.runtime.driver import build_role_drivers
from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease

_SOLVER_PROBE_ENTRYPOINT = "\n".join(
    [
        "Read `README.md` first.",
        "This is a solver smoke probe, not a full optimization run.",
        "Inspect `guidance/`, `output/`, and the predefined `check-runner` sub-agent.",
        "Make one tiny safe improvement to `output/candidate.py`, run the check workflow once,",
        "write the required completion file, and stop immediately.",
    ]
)
_OPTIMIZER_PROBE_ENTRYPOINT = "\n".join(
    [
        "Read `README.md` first.",
        "This is an optimizer smoke probe, not a full optimization run.",
        "Inspect the reference example directories, `guidance/`, and `output/`.",
        "Make one tiny documentation improvement inside `output/`,",
        "write the required completion file, and stop immediately.",
    ]
)
_EVAL_PROBE_ENTRYPOINT = "\n".join(
    [
        "This is an evaluation smoke probe.",
        "Run the evaluation instructions below exactly once.",
        "Write `logs/evaluate/score.yaml`, write the required completion file,",
        "and stop immediately.",
    ]
)
_EVAL_EVALUATION_MD = """# Probe Evaluation

From the repository root:

```bash
python3 evaluate.py
```

Write the score card to `logs/evaluate/score.yaml`.
Save any extra notes under `logs/evaluate/`.
"""


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("solver", "eval", "optimizer"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    config_file = Path(args.config).resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_file.parent), version_base=None):
        cfg = compose(config_name=config_file.stem, overrides=args.overrides)
    raw = OmegaConf.to_container(cfg, resolve=True)

    repo_root = Path(__file__).resolve().parents[4]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    uid = uuid4().hex[:12]
    run_id = f"run-{ts}-probe-{args.agent}-{uid}"
    run_root = repo_root / ".runs" / "eve" / "probe" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    workers = int(raw.get("loop", {}).get("n_workers_phase2", 1) or 1)
    drivers = build_role_drivers(
        raw.get("driver", {}),
        run_root=run_root,
        workers=workers,
    )
    try:
        if args.agent == "solver":
            payload = _probe_solver(run_root / "solver_probe", drivers.solver_driver)
        elif args.agent == "optimizer":
            payload = _probe_optimizer(run_root / "optimizer_probe", drivers.optimizer_driver)
        else:
            payload = _probe_eval(run_root / "eval_probe", drivers.eval_driver_factory())
    finally:
        drivers.close()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _probe_solver(workspace: Path, driver) -> dict[str, object]:
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "guidance").mkdir(parents=True)
    (workspace / "output").mkdir(parents=True)
    _write(
        workspace / "README.md",
        "\n".join(
            [
                "# Solver Probe",
                "",
                "- Editable file: `output/candidate.py`.",
                "- **MANDATORY:** Run the predefined `check-runner`"
                " from the workspace root before you stop.",
            ]
        )
        + "\n",
    )
    _write(workspace / "guidance" / "notes.md", "# Probe Guidance\nKeep changes tiny.\n")
    _write(
        workspace / ".claude" / "agents" / "check-runner.md",
        "\n".join(
            [
                "---",
                "name: check-runner",
                'description: "Run the solver smoke check and report PASS or FAIL."',
                "tools: Bash, Read",
                "---",
                "",
                "python3 -m py_compile candidate.py",
            ]
        )
        + "\n",
    )
    _write(
        workspace / ".codex" / "agents" / "check-runner.toml",
        "\n".join(
            [
                'name = "check-runner"',
                'description = "Run the solver smoke check and report PASS or FAIL."',
                "",
                'developer_instructions = """python3 -m py_compile candidate.py',
                '"""',
            ]
        )
        + "\n",
    )
    _write(
        workspace / "output" / "candidate.py",
        "\n".join(
            [
                "VALUE = 1",
                "",
                "def score() -> int:",
                "    return VALUE",
            ]
        )
        + "\n",
    )

    rollout = driver.spawn(
        SessionSeed(
            instruction=_SOLVER_PROBE_ENTRYPOINT,
            workspace=_workspace_lease(workspace, target_repo_root=workspace / "output"),
            prompt_file="README.md",
            write_prompt_file=False,
        )
    )
    return {
        "agent": "solver",
        "workspace": str(workspace),
        "summary": rollout.summary,
        "changed_paths": rollout.changed_paths,
        "completion_path": rollout.state.metadata.get("completion_path"),
        "candidate": (workspace / "output" / "candidate.py").read_text(encoding="utf-8"),
        "pane_pool_session_name": rollout.state.metadata.get("pane_pool_session_name"),
    }


def _probe_optimizer(workspace: Path, driver) -> dict[str, object]:
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "guidance").mkdir(parents=True)
    (workspace / "examples" / "optimizer-a").mkdir(parents=True)
    (workspace / "output").mkdir(parents=True)
    _write(
        workspace / "README.md",
        "\n".join(
            [
                "# Optimizer Probe",
                "",
                "- Edit only files under `output/`.",
                "- Keep changes tiny and documentation-focused.",
            ]
        )
        + "\n",
    )
    _write(workspace / "guidance" / "notes.md", "# Probe Guidance\nPrefer clarity.\n")
    _write(
        workspace / "examples" / "optimizer-a" / "APPROACH.md",
        "# Reference\nUse quick checks.\n",
    )
    _write(workspace / "output" / "APPROACH.md", "# Approach\nStart simple.\n")

    rollout = driver.spawn(
        SessionSeed(
            instruction=_OPTIMIZER_PROBE_ENTRYPOINT,
            workspace=_workspace_lease(workspace, target_repo_root=workspace / "output"),
            prompt_file="README.md",
            write_prompt_file=False,
        )
    )
    return {
        "agent": "optimizer",
        "workspace": str(workspace),
        "summary": rollout.summary,
        "changed_paths": rollout.changed_paths,
        "completion_path": rollout.state.metadata.get("completion_path"),
        "approach": (workspace / "output" / "APPROACH.md").read_text(encoding="utf-8"),
        "pane_pool_session_name": rollout.state.metadata.get("pane_pool_session_name"),
    }


def _probe_eval(workspace: Path, driver) -> dict[str, object]:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "logs" / "evaluate").mkdir(parents=True)
    _write(
        workspace / "candidate.py",
        "VALUE = 1\n",
    )
    _write(
        workspace / "evaluate.py",
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "score = 1.25",
                "Path('logs/evaluate').mkdir(parents=True, exist_ok=True)",
                "Path('logs/evaluate/summary.txt').write_text('probe ok\\n', encoding='utf-8')",
                "print(score)",
            ]
        )
        + "\n",
    )

    rollout = driver.spawn(
        SessionSeed(
            instruction=_EVAL_PROBE_ENTRYPOINT + "\n\n" + _EVAL_EVALUATION_MD,
            workspace=_workspace_lease(workspace),
            prompt_file="evaluation.md",
            write_prompt_file=True,
        )
    )
    score_path = workspace / "logs" / "evaluate" / "score.yaml"
    score_payload = (
        yaml.safe_load(score_path.read_text(encoding="utf-8")) if score_path.exists() else None
    )
    return {
        "agent": "eval",
        "workspace": str(workspace),
        "summary": rollout.summary,
        "score": score_payload,
        "completion_path": rollout.state.metadata.get("completion_path"),
        "logs": sorted(
            str(path.relative_to(workspace)) for path in (workspace / "logs").rglob("*")
        ),
        "pane_pool_session_name": rollout.state.metadata.get("pane_pool_session_name"),
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _workspace_lease(
    workspace: Path,
    *,
    target_repo_root: Path | None = None,
) -> SessionWorkspaceLease:
    resolved_workspace = workspace.resolve()
    resolved_target_repo_root = (target_repo_root or workspace).resolve()
    return SessionWorkspaceLease(
        workspace_id=f"probe:{resolved_workspace.name}",
        target_repo_root=str(resolved_target_repo_root),
        workspace_root=str(resolved_workspace),
        session_cwd=str(resolved_workspace),
    )


if __name__ == "__main__":
    raise SystemExit(main())
