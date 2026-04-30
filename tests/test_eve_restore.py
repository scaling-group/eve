from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.optimizer_population import (
    OptimizerPopulation,
)
from scaling_evolve.algorithms.eve.populations.solver_population import SolverPopulation
from scaling_evolve.algorithms.eve.runtime.restore import (
    parse_restore_spec,
    resolve_restore_run_root,
    restore_populations_from_run,
)
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore


def _score(value: float) -> object:
    return {"score": value, "summary": f"score={value}"}


def _config(root: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "workspace_root": str(root),
            "run_id": "test-run",
            "label": "",
            "artifact_root": str(root / "artifacts"),
            "solver_db_path": str(root / "solver_lineage.db"),
            "optimizer_db_path": str(root / "optimizer_lineage.db"),
        }
    )


def test_restore_populations_from_previous_run_rewrites_artifacts_into_target_run(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_run"
    source_root.mkdir()
    source_solver_store = SQLiteLineageStore(source_root / "solver_lineage.db")
    source_optimizer_store = SQLiteLineageStore(source_root / "optimizer_lineage.db")
    source_artifact_store = FSArtifactStore(source_root / "artifacts", run_id="source_solver")
    source_solver_population = SolverPopulation(
        source_solver_store,
        source_artifact_store,
        run_id="source_solver",
        config=_config(source_root),
    )
    source_optimizer_population = OptimizerPopulation(
        source_optimizer_store,
        source_artifact_store,
        run_id="source_optimizer",
        config=_config(source_root),
    )
    source_solver_population.add(
        PopulationEntry(
            id="solver_old",
            files={"candidate.py": "print('old solver')\n"},
            score=_score(1.5),
            logs={"evaluate/summary.txt": "old solver\n"},
        )
    )
    source_optimizer_population.add(
        PopulationEntry(
            id="optimizer_old",
            files={"README.md": "# old optimizer\n"},
            score=_score(1520.0),
            logs={"step_1_solver_old/score.yaml": ("score:\n  score: 1.5\n  summary: score=1.5\n")},
        )
    )

    target_root = tmp_path / "target_run"
    target_root.mkdir()
    target_solver_store = SQLiteLineageStore(target_root / "solver_lineage.db")
    target_optimizer_store = SQLiteLineageStore(target_root / "optimizer_lineage.db")
    target_artifact_store = FSArtifactStore(target_root / "artifacts", run_id="target_solver")
    target_solver_population = SolverPopulation(
        target_solver_store,
        target_artifact_store,
        run_id="target_solver",
        config=_config(target_root),
    )
    target_optimizer_population = OptimizerPopulation(
        target_optimizer_store,
        target_artifact_store,
        run_id="target_optimizer",
        config=_config(target_root),
    )

    try:
        report = restore_populations_from_run(
            source_root / "solver_workspaces" / "nested",
            solver_population=target_solver_population,
            optimizer_population=target_optimizer_population,
        )

        assert report.source_run_root == source_root
        assert report.solver.entries_restored == 1
        assert report.optimizer.entries_restored == 1

        restored_solver = target_solver_population.entries()
        restored_optimizer = target_optimizer_population.entries()
        assert [entry.id for entry in restored_solver] == ["solver_old"]
        assert [entry.id for entry in restored_optimizer] == ["optimizer_old"]
        artifact_files = list((target_root / "artifacts").rglob("*"))
        assert any(path.is_file() for path in artifact_files)
        assert all(source_root not in path.parents for path in artifact_files)
    finally:
        source_solver_store.close()
        source_optimizer_store.close()
        target_solver_store.close()
        target_optimizer_store.close()


def test_resolve_restore_run_root_accepts_nested_workspace_path(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    nested = run_root / "solver_workspaces" / "20260406_seed"
    nested.mkdir(parents=True)
    (run_root / "solver_lineage.db").write_text("", encoding="utf-8")
    (run_root / "optimizer_lineage.db").write_text("", encoding="utf-8")

    resolved = resolve_restore_run_root(nested)

    assert resolved == run_root


def test_restore_populations_from_multiple_sources_is_additive(tmp_path: Path) -> None:
    target_root = tmp_path / "target_run"
    target_root.mkdir()
    target_solver_store = SQLiteLineageStore(target_root / "solver_lineage.db")
    target_optimizer_store = SQLiteLineageStore(target_root / "optimizer_lineage.db")
    target_artifact_store = FSArtifactStore(target_root / "artifacts", run_id="target_solver")
    target_solver_population = SolverPopulation(
        target_solver_store,
        target_artifact_store,
        run_id="target_solver",
        config=_config(target_root),
    )
    target_optimizer_population = OptimizerPopulation(
        target_optimizer_store,
        target_artifact_store,
        run_id="target_optimizer",
        config=_config(target_root),
    )

    source_specs = [
        ("source_run_a", "solver_a", "optimizer_a"),
        ("source_run_b", "solver_b", "optimizer_b"),
    ]
    source_stores: list[SQLiteLineageStore] = []
    try:
        for root_name, solver_id, optimizer_id in source_specs:
            source_root = tmp_path / root_name
            source_root.mkdir()
            source_solver_store = SQLiteLineageStore(source_root / "solver_lineage.db")
            source_optimizer_store = SQLiteLineageStore(source_root / "optimizer_lineage.db")
            source_stores.extend([source_solver_store, source_optimizer_store])
            source_artifact_store = FSArtifactStore(source_root / "artifacts", run_id=solver_id)
            SolverPopulation(
                source_solver_store,
                source_artifact_store,
                run_id=solver_id,
                config=_config(source_root),
            ).add(
                PopulationEntry(
                    id=solver_id,
                    files={"candidate.py": f"print('{solver_id}')\n"},
                    score=_score(1.0),
                    logs={},
                )
            )
            OptimizerPopulation(
                source_optimizer_store,
                source_artifact_store,
                run_id=optimizer_id,
                config=_config(source_root),
            ).add(
                PopulationEntry(
                    id=optimizer_id,
                    files={"README.md": f"# {optimizer_id}\n"},
                    score=_score(1500.0),
                    logs={},
                )
            )

            restore_populations_from_run(
                source_root,
                solver_population=target_solver_population,
                optimizer_population=target_optimizer_population,
            )

        assert {entry.id for entry in target_solver_population.entries()} == {
            "solver_a",
            "solver_b",
        }
        assert {entry.id for entry in target_optimizer_population.entries()} == {
            "optimizer_a",
            "optimizer_b",
        }
    finally:
        for store in source_stores:
            store.close()
        target_solver_store.close()
        target_optimizer_store.close()


def test_restore_populations_can_filter_specific_solver_and_optimizer_ids(tmp_path: Path) -> None:
    source_root = tmp_path / "source_run"
    source_root.mkdir()
    source_solver_store = SQLiteLineageStore(source_root / "solver_lineage.db")
    source_optimizer_store = SQLiteLineageStore(source_root / "optimizer_lineage.db")
    source_artifact_store = FSArtifactStore(source_root / "artifacts", run_id="source_solver")
    solver_population = SolverPopulation(
        source_solver_store,
        source_artifact_store,
        run_id="source_solver",
        config=_config(source_root),
    )
    optimizer_population = OptimizerPopulation(
        source_optimizer_store,
        source_artifact_store,
        run_id="source_optimizer",
        config=_config(source_root),
    )
    solver_population.add(
        PopulationEntry(id="solver_keep", files={"a.py": "a\n"}, score=_score(1.0), logs={})
    )
    solver_population.add(
        PopulationEntry(id="solver_drop", files={"b.py": "b\n"}, score=_score(2.0), logs={})
    )
    optimizer_population.add(
        PopulationEntry(id="optimizer_keep", files={"x.md": "x\n"}, score=_score(1500.0), logs={})
    )
    optimizer_population.add(
        PopulationEntry(id="optimizer_drop", files={"y.md": "y\n"}, score=_score(1490.0), logs={})
    )

    target_root = tmp_path / "target_run"
    target_root.mkdir()
    target_solver_store = SQLiteLineageStore(target_root / "solver_lineage.db")
    target_optimizer_store = SQLiteLineageStore(target_root / "optimizer_lineage.db")
    target_artifact_store = FSArtifactStore(target_root / "artifacts", run_id="target_solver")
    target_solver_population = SolverPopulation(
        target_solver_store,
        target_artifact_store,
        run_id="target_solver",
        config=_config(target_root),
    )
    target_optimizer_population = OptimizerPopulation(
        target_optimizer_store,
        target_artifact_store,
        run_id="target_optimizer",
        config=_config(target_root),
    )

    try:
        restore_populations_from_run(
            source_root,
            solver_population=target_solver_population,
            optimizer_population=target_optimizer_population,
            solver_ids=("solver_keep",),
            optimizer_ids=("optimizer_keep",),
        )
        assert {entry.id for entry in target_solver_population.entries()} == {"solver_keep"}
        assert {entry.id for entry in target_optimizer_population.entries()} == {"optimizer_keep"}
    finally:
        source_solver_store.close()
        source_optimizer_store.close()
        target_solver_store.close()
        target_optimizer_store.close()


def test_parse_restore_spec_accepts_filtered_mapping() -> None:
    spec = parse_restore_spec(
        {
            "path": "/tmp/demo-run",
            "solver_ids": ["solver_a", "solver_b"],
            "optimizer_ids": "optimizer_x",
        }
    )

    assert spec.path == Path("/tmp/demo-run")
    assert spec.solver_ids == ("solver_a", "solver_b")
    assert spec.optimizer_ids == ("optimizer_x",)
