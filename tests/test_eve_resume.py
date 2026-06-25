from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve.factory import EveFactory
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.evaluators.elo import ScalarEloEvaluator
from scaling_evolve.algorithms.eve.populations.optimizer_population import OptimizerPopulation
from scaling_evolve.algorithms.eve.populations.samplers.uniform import UniformSampler
from scaling_evolve.algorithms.eve.populations.solver_population import SolverPopulation
from scaling_evolve.algorithms.eve.runtime.resume import (
    RESUME_ARCHIVE_SUBDIR,
    EveCheckpoint,
    ResumeError,
    prepare_resume,
    read_checkpoint,
    write_checkpoint,
)
from scaling_evolve.algorithms.eve.runtime.snapshots import (
    SNAPSHOT_FAMILIES,
    snapshot_root,
    write_lineage_snapshots,
)
from scaling_evolve.algorithms.eve.workflow.loop import Eve
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2BatchRunner, Phase2Result
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore


def _init_marker_db(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
        conn.execute("INSERT INTO marker(value) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def _write_marker(path: Path, *, value: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("DELETE FROM marker")
        conn.execute("INSERT INTO marker(value) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def _read_marker(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM marker").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


def _single_resume_archive(run_root: Path) -> Path:
    archives = sorted((run_root / RESUME_ARCHIVE_SUBDIR).glob("resume_*"))
    assert len(archives) == 1
    return archives[0]


def _loop_config(run_root: Path, *, max_iterations: int, enable_resume: bool = True) -> object:
    return OmegaConf.create(
        {
            "workspace_root": str(run_root),
            "artifact_root": str(run_root / "artifacts"),
            "solver_db_path": str(run_root / "solver_lineage.db"),
            "optimizer_db_path": str(run_root / "optimizer_lineage.db"),
            "run_id": "run-deterministic",
            "label": "",
            "max_iterations": max_iterations,
            "n_workers_phase2": 1,
            "n_solver_examples_phase2": 1,
            "n_optimizer_examples_phase2": 1,
            "boundary_repair_attempts": 0,
            "enable_resume": enable_resume,
            "produce_optimizer_in_phase2": 0,
        }
    )


def _make_stub_loop(
    run_root: Path,
    *,
    max_iterations: int,
    seed_entries: bool = True,
    enable_resume: bool = True,
) -> tuple[Eve, SQLiteLineageStore, SQLiteLineageStore]:
    config = _loop_config(
        run_root,
        max_iterations=max_iterations,
        enable_resume=enable_resume,
    )
    solver_store = SQLiteLineageStore(run_root / "solver_lineage.db")
    optimizer_store = SQLiteLineageStore(run_root / "optimizer_lineage.db")
    solver_pop = SolverPopulation(
        solver_store,
        FSArtifactStore(run_root / "artifacts", run_id="run-deterministic_solver"),
        run_id="run-deterministic_solver",
        config=config,
    )
    optimizer_pop = OptimizerPopulation(
        optimizer_store,
        FSArtifactStore(run_root / "artifacts", run_id="run-deterministic_optimizer"),
        run_id="run-deterministic_optimizer",
        config=config,
    )
    if seed_entries:
        solver_pop.add(
            PopulationEntry(
                id="solver_seed",
                files={"candidate.py": "seed\n"},
                score={"score": 0.0},
                logs={},
            )
        )
        optimizer_pop.add(
            PopulationEntry(
                id="optimizer_seed",
                files={"README.md": "optimizer\n"},
                score={"elo": 1500.0},
                logs={},
            )
        )
    loop = Eve(
        run_id="run-deterministic",
        solver_pop=solver_pop,
        optimizer_pop=optimizer_pop,
        solver_workspace_builder=SimpleNamespace(_rng=None),
        solver_driver=object(),
        solver_evaluator=object(),
        config=config,
        optimizer_evaluator=ScalarEloEvaluator(),
        phase2_optimizer_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_solver_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_prefill_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_optimizer_examples_sampler=UniformSampler(replacement_mode="no_replacement"),
        phase2_produced_optimizer_sampler=UniformSampler(replacement_mode="no_replacement"),
    )
    return loop, solver_store, optimizer_store


def _canonical_entries(run_root: Path) -> tuple[tuple[str, object, object, object], ...]:
    solver_store = SQLiteLineageStore(run_root / "solver_lineage.db")
    artifact_store = FSArtifactStore(run_root / "artifacts", run_id="run-deterministic_solver")
    population = SolverPopulation(
        solver_store,
        artifact_store,
        run_id="run-deterministic_solver",
        config=_loop_config(run_root, max_iterations=0),
    )
    try:
        entries = [
            (entry.id, entry.files, entry.logs, entry.score)
            for entry in sorted(population.entries(), key=lambda item: item.id)
        ]
    finally:
        solver_store.close()
    assert entries
    return tuple(entries)


def _install_deterministic_phase2(monkeypatch, *, fail_at: int | None = None) -> None:
    def _fake_phase2_run(self):  # noqa: ANN001
        if fail_at is not None and self.iteration == fail_at:
            run_root = Path(str(self.solver_pop._config.workspace_root))  # noqa: SLF001
            self.solver_pop.add(
                PopulationEntry(
                    id=f"solver_partial_iter_{self.iteration:06d}",
                    files={"candidate.py": "partial\n"},
                    score={"score": -1.0},
                    logs={},
                )
            )
            (run_root / f"step_{self.iteration}").mkdir()
            (run_root / "solver_workspaces" / f"stub_step_{self.iteration}_partial").mkdir(
                parents=True
            )
            raise RuntimeError("simulated interruption")
        optimizer = self.optimizer_pop.entries()[0]
        produced_solver = PopulationEntry(
            id=f"solver_iter_{self.iteration:06d}",
            files={"candidate.py": f"iteration {self.iteration}\n"},
            score={"score": float(self.iteration)},
            logs={"summary.txt": f"iteration {self.iteration}\n"},
        )
        return [
            Phase2Result(
                optimizer=optimizer,
                produced_solver=produced_solver,
                workspace_id=f"stub_step_{self.iteration}",
            )
        ]

    monkeypatch.setattr(Phase2BatchRunner, "run", _fake_phase2_run)


def _immutable_readme(version: str) -> str:
    return "\n".join(
        [
            f"# immutable version {version}",
            "",
            "{editable_files_block}",
            "{editable_folders_block}",
            "{solver_examples_block}",
            "{optimizer_examples_block}",
        ]
    )


def _factory_reload_config(
    root: Path,
    *,
    immutable_root: Path,
    prompt_root: Path,
    initial_guidance: Path,
) -> object:
    sampler_target = "scaling_evolve.algorithms.eve.populations.samplers.uniform.UniformSampler"
    return OmegaConf.create(
        {
            "run_id": "run-reload",
            "run_root": str(root / "run"),
            "loop": {
                "workspace_root": str(root / "run"),
                "artifact_root": str(root / "run" / "artifacts"),
                "solver_db_path": str(root / "run" / "solver_lineage.db"),
                "optimizer_db_path": str(root / "run" / "optimizer_lineage.db"),
                "max_iterations": 1,
                "n_workers_phase2": 1,
                "n_solver_examples_phase2": 1,
                "n_optimizer_examples_phase2": 1,
                "produce_optimizer_in_phase2": 0,
                "boundary_repair_attempts": 0,
                "enable_resume": True,
                "sampling": {
                    "phase1_optimizer_population": {
                        "_target_": sampler_target,
                        "replacement_mode": "no_replacement",
                    },
                    "phase1_solver_population": {
                        "_target_": sampler_target,
                        "replacement_mode": "no_replacement",
                    },
                    "solver_workspace_prefill": {
                        "_target_": sampler_target,
                        "replacement_mode": "no_replacement",
                    },
                    "phase2_optimizer_examples": {
                        "_target_": sampler_target,
                        "replacement_mode": "no_replacement",
                    },
                    "phase2_produced_optimizers": {
                        "_target_": sampler_target,
                        "replacement_mode": "no_replacement",
                    },
                },
            },
            "optimizer": {
                "initial_guidance": str(initial_guidance.relative_to(root)),
                "immutable_renderer": {
                    "_target_": (
                        "scaling_evolve.algorithms.eve.workspace.immutable_renderers.default."
                        "DefaultRenderer"
                    ),
                },
                "workers": {
                    "selection": "random",
                    "items": [
                        {
                            "name": "normal",
                            "weight": 1.0,
                            "immutable": str(immutable_root.relative_to(root)),
                            "prompt": str(prompt_root.relative_to(root)),
                        }
                    ],
                },
                "evaluation": {
                    "_target_": (
                        "scaling_evolve.algorithms.eve.populations.evaluators.elo."
                        "ScalarEloEvaluator"
                    ),
                    "initial_score": {"elo": 1500.0},
                },
            },
        }
    )


def test_checkpoint_round_trips_atomically(tmp_path: Path) -> None:
    checkpoint = EveCheckpoint(
        run_id="run-demo",
        last_completed_iteration=3,
        max_iterations=5,
    )

    write_checkpoint(tmp_path, checkpoint)

    assert read_checkpoint(tmp_path) == checkpoint
    assert not (tmp_path / "checkpoint.json.tmp").exists()


def test_prepare_resume_rolls_back_and_archives_interrupted_iteration(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    solver_db = run_root / "solver_lineage.db"
    optimizer_db = run_root / "optimizer_lineage.db"
    _init_marker_db(solver_db, value=1)
    _init_marker_db(optimizer_db, value=101)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=1,
    )
    _write_marker(solver_db, value=2)
    _write_marker(optimizer_db, value=102)
    (run_root / "step_2").mkdir()
    (run_root / "solver_workspaces" / "20260621_step_2_partial").mkdir(parents=True)
    (run_root / "solver_workspaces" / "20260621_step_20_other").mkdir(parents=True)
    (run_root / "evaluation_workspaces" / "20260621_step_2_partial").mkdir(parents=True)
    write_checkpoint(
        run_root,
        EveCheckpoint(run_id="run-demo", last_completed_iteration=1, max_iterations=3),
    )

    plan = prepare_resume(run_root)

    assert plan.run_id == "run-demo"
    assert plan.start_iteration == 1
    assert _read_marker(solver_db) == 1
    assert _read_marker(optimizer_db) == 101
    assert not (run_root / "step_2").exists()
    assert not (run_root / "solver_workspaces" / "20260621_step_2_partial").exists()
    assert not (run_root / "solver_workspaces" / "20260621_step_20_other").exists()
    assert not (run_root / "evaluation_workspaces" / "20260621_step_2_partial").exists()

    archive = _single_resume_archive(run_root)
    assert (archive / "step_2").is_dir()
    assert (archive / "solver_workspaces" / "20260621_step_2_partial").is_dir()
    assert (archive / "solver_workspaces" / "20260621_step_20_other").is_dir()
    assert (archive / "evaluation_workspaces" / "20260621_step_2_partial").is_dir()
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "run-demo"
    assert manifest["anchor_iteration"] == 1
    assert {
        (item["kind"], item["iteration"], item["source"], item["archive"])
        for item in manifest["moved_paths"]
    } == {
        ("step_dir", 2, "step_2", "step_2"),
        (
            "solver_workspace",
            2,
            "solver_workspaces/20260621_step_2_partial",
            "solver_workspaces/20260621_step_2_partial",
        ),
        (
            "solver_workspace",
            20,
            "solver_workspaces/20260621_step_20_other",
            "solver_workspaces/20260621_step_20_other",
        ),
        (
            "evaluation_workspace",
            2,
            "evaluation_workspaces/20260621_step_2_partial",
            "evaluation_workspaces/20260621_step_2_partial",
        ),
    }


def test_prepare_resume_can_use_explicit_iteration_anchor(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    solver_db = run_root / "solver_lineage.db"
    optimizer_db = run_root / "optimizer_lineage.db"
    _init_marker_db(solver_db, value=1)
    _init_marker_db(optimizer_db, value=101)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=1,
    )
    _write_marker(solver_db, value=2)
    _write_marker(optimizer_db, value=102)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=2,
    )
    _write_marker(solver_db, value=3)
    _write_marker(optimizer_db, value=103)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=3,
    )
    _write_marker(solver_db, value=4)
    _write_marker(optimizer_db, value=104)
    for step in (2, 3, 4):
        (run_root / f"step_{step}").mkdir(parents=True)
    (run_root / "solver_workspaces" / "stub_step_2_keep").mkdir(parents=True)
    (run_root / "solver_workspaces" / "stub_step_3_remove").mkdir(parents=True)
    (run_root / "solver_workspaces" / "stub_step_20_remove").mkdir(parents=True)
    (run_root / "evaluation_workspaces" / "stub_step_3_remove").mkdir(parents=True)
    write_checkpoint(
        run_root,
        EveCheckpoint(run_id="run-demo", last_completed_iteration=3, max_iterations=5),
    )

    plan = prepare_resume(run_root, resume_iteration=2)

    assert plan.run_id == "run-demo"
    assert plan.start_iteration == 2
    assert _read_marker(solver_db) == 2
    assert _read_marker(optimizer_db) == 102
    assert (run_root / "step_2").exists()
    assert not (run_root / "step_3").exists()
    assert not (run_root / "step_4").exists()
    assert (run_root / "solver_workspaces" / "stub_step_2_keep").exists()
    assert not (run_root / "solver_workspaces" / "stub_step_3_remove").exists()
    assert not (run_root / "solver_workspaces" / "stub_step_20_remove").exists()
    assert not (run_root / "evaluation_workspaces" / "stub_step_3_remove").exists()
    archive = _single_resume_archive(run_root)
    assert not (archive / "step_2").exists()
    assert (archive / "step_3").is_dir()
    assert (archive / "step_4").is_dir()
    assert (archive / "solver_workspaces" / "stub_step_3_remove").is_dir()
    assert (archive / "solver_workspaces" / "stub_step_20_remove").is_dir()
    assert (archive / "evaluation_workspaces" / "stub_step_3_remove").is_dir()


def test_prepare_resume_creates_separate_archives_for_repeated_resume(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    solver_db = run_root / "solver_lineage.db"
    optimizer_db = run_root / "optimizer_lineage.db"
    _init_marker_db(solver_db, value=1)
    _init_marker_db(optimizer_db, value=101)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=1,
    )
    write_checkpoint(
        run_root,
        EveCheckpoint(run_id="run-demo", last_completed_iteration=1, max_iterations=3),
    )

    (run_root / "solver_workspaces" / "first_step_2_failed").mkdir(parents=True)
    prepare_resume(run_root)
    (run_root / "solver_workspaces" / "second_step_2_failed").mkdir(parents=True)
    prepare_resume(run_root)

    archives = sorted((run_root / RESUME_ARCHIVE_SUBDIR).glob("resume_*"))
    assert [path.name.split("__", 1)[0] for path in archives] == ["resume_0001", "resume_0002"]
    assert (archives[0] / "solver_workspaces" / "first_step_2_failed").is_dir()
    assert (archives[1] / "solver_workspaces" / "second_step_2_failed").is_dir()


def test_prepare_resume_rejects_invalid_iteration_anchor(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_checkpoint(
        run_root,
        EveCheckpoint(run_id="run-demo", last_completed_iteration=2, max_iterations=4),
    )

    with pytest.raises(ResumeError, match="after the last completed iteration"):
        prepare_resume(run_root, resume_iteration=3)

    with pytest.raises(ResumeError, match="non-negative"):
        prepare_resume(run_root, resume_iteration=-1)

    write_checkpoint(
        run_root,
        EveCheckpoint(run_id="run-demo", last_completed_iteration=4, max_iterations=4),
    )
    with pytest.raises(ResumeError, match="less than max_iterations"):
        prepare_resume(run_root, resume_iteration=4)


def test_prepare_resume_requires_selected_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_checkpoint(
        run_root,
        EveCheckpoint(run_id="run-demo", last_completed_iteration=1, max_iterations=3),
    )

    with pytest.raises(ResumeError, match="Missing snapshot"):
        prepare_resume(run_root, resume_iteration=1)


def test_write_lineage_snapshots_preserves_existing_anchor(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    solver_db = run_root / "solver_lineage.db"
    optimizer_db = run_root / "optimizer_lineage.db"
    _init_marker_db(solver_db, value=1)
    _init_marker_db(optimizer_db, value=101)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=1,
    )

    _write_marker(solver_db, value=2)
    _write_marker(optimizer_db, value=102)
    write_lineage_snapshots(
        run_root=run_root,
        solver_db_path=solver_db,
        optimizer_db_path=optimizer_db,
        anchor_iteration=1,
    )

    snapshots = snapshot_root(run_root)
    assert _read_marker(snapshots / "solver_lineage_iter_1.db") == 1
    assert _read_marker(snapshots / "optimizer_lineage_iter_1.db") == 101


def test_loop_keeps_all_iteration_snapshots(tmp_path: Path, monkeypatch) -> None:
    _install_deterministic_phase2(monkeypatch)
    loop, solver_store, optimizer_store = _make_stub_loop(
        tmp_path / "run",
        max_iterations=4,
    )
    try:
        loop.run()
    finally:
        solver_store.close()
        optimizer_store.close()

    root = snapshot_root(tmp_path / "run")
    for family in SNAPSHOT_FAMILIES:
        assert [path.name for path in sorted(root.glob(f"{family.prefix}*.db"))] == [
            f"{family.prefix}0.db",
            f"{family.prefix}1.db",
            f"{family.prefix}2.db",
            f"{family.prefix}3.db",
            f"{family.prefix}4.db",
        ]


def test_loop_can_disable_resume_persistence(tmp_path: Path, monkeypatch) -> None:
    _install_deterministic_phase2(monkeypatch)
    loop, solver_store, optimizer_store = _make_stub_loop(
        tmp_path / "run",
        max_iterations=1,
        enable_resume=False,
    )
    try:
        loop.run()
    finally:
        solver_store.close()
        optimizer_store.close()

    assert read_checkpoint(tmp_path / "run") is None
    snapshots = snapshot_root(tmp_path / "run")
    assert not snapshots.exists() or not any(snapshots.iterdir())


def test_deterministic_stub_resume_matches_clean_run(tmp_path: Path, monkeypatch) -> None:
    """Resume must reproduce a clean run's solver population exactly.

    Determinism here is achieved entirely test-side: stable pre-seeded ids plus a
    stubbed Phase 2 that returns fixed results. The production loop needs no
    determinism hooks for this. If this test ever turns flaky, fix it in the stub,
    do not add determinism switches to the production code.
    """
    clean_root = tmp_path / "clean"
    _install_deterministic_phase2(monkeypatch)
    clean_loop, clean_solver_store, clean_optimizer_store = _make_stub_loop(
        clean_root,
        max_iterations=4,
    )
    try:
        clean_loop.run()
    finally:
        clean_solver_store.close()
        clean_optimizer_store.close()

    interrupted_root = tmp_path / "interrupted"
    _install_deterministic_phase2(monkeypatch, fail_at=3)
    interrupted_loop, interrupted_solver_store, interrupted_optimizer_store = _make_stub_loop(
        interrupted_root,
        max_iterations=4,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted_loop.run()
    interrupted_solver_store.close()
    interrupted_optimizer_store.close()

    assert (interrupted_root / "step_3").exists()
    assert _canonical_entries(interrupted_root) != _canonical_entries(clean_root)

    plan = prepare_resume(interrupted_root)
    assert plan.start_iteration == 2
    assert not (interrupted_root / "step_3").exists()

    _install_deterministic_phase2(monkeypatch)
    resumed_loop, resumed_solver_store, resumed_optimizer_store = _make_stub_loop(
        interrupted_root,
        max_iterations=4,
        seed_entries=False,
    )
    try:
        resumed_loop.run(start_iteration=plan.start_iteration)
    finally:
        resumed_solver_store.close()
        resumed_optimizer_store.close()

    assert _canonical_entries(interrupted_root) == _canonical_entries(clean_root)
    checkpoint = read_checkpoint(interrupted_root)
    assert checkpoint is not None
    assert checkpoint.run_id == "run-deterministic"
    assert checkpoint.last_completed_iteration == 4


def test_factory_reload_reads_updated_immutable_files(tmp_path: Path) -> None:
    immutable_root = tmp_path / "immutable"
    prompt_root = tmp_path / "prompt"
    initial_guidance = tmp_path / "initial_guidance"
    immutable_root.mkdir()
    prompt_root.mkdir()
    initial_guidance.mkdir()
    (immutable_root / "README.md").write_text(_immutable_readme("v1"), encoding="utf-8")
    (prompt_root / "ENTRYPOINT.md").write_text("entrypoint\n", encoding="utf-8")
    (prompt_root / "BOUNDARY_REPAIR.md").write_text("repair\n", encoding="utf-8")
    (prompt_root / "budget").mkdir()
    (prompt_root / "budget" / "USER.md").write_text(
        "Budget: {rollout_max_turns}\n",
        encoding="utf-8",
    )
    (prompt_root / "budget" / "TURN.md").write_text(
        "Remaining: {turns_remaining}\n",
        encoding="utf-8",
    )
    (initial_guidance / "README.md").write_text("seed optimizer\n", encoding="utf-8")
    config = _factory_reload_config(
        tmp_path,
        immutable_root=immutable_root,
        prompt_root=prompt_root,
        initial_guidance=initial_guidance,
    )

    first = EveFactory.from_config(
        config,
        solver_evaluator=object(),
        solver_driver=object(),
        task_problem=SimpleNamespace(),
        search_root=tmp_path,
    )
    try:
        assert (
            "version v1"
            in first.loop.solver_workspace_builder.worker_configs[0].immutable_files["README.md"]
        )
    finally:
        first.close()

    (immutable_root / "README.md").write_text(_immutable_readme("v2"), encoding="utf-8")

    second = EveFactory.from_config(
        config,
        solver_evaluator=object(),
        solver_driver=object(),
        task_problem=SimpleNamespace(),
        search_root=tmp_path,
    )
    try:
        assert (
            "version v2"
            in second.loop.solver_workspace_builder.worker_configs[0].immutable_files["README.md"]
        )
    finally:
        second.close()
