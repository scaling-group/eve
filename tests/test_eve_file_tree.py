from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.populations.solver_population import SolverPopulation
from scaling_evolve.algorithms.eve.workspace.file_tree import (
    read_file_tree,
    write_file_tree,
)
from scaling_evolve.storage.artifacts import FSArtifactStore
from scaling_evolve.storage.sqlite import SQLiteLineageStore


def test_portable_file_tree_round_trips_binary_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    text_payload = "# evaluation notes\n"
    binary_payload = b"\x89PNG\r\n\x1a\n\x00binary-payload\xff"

    (source / "README.md").write_text(text_payload, encoding="utf-8")
    (source / "preview.png").write_bytes(binary_payload)

    portable = read_file_tree(source)

    assert portable["README.md"] == text_payload
    assert portable["preview.png"].startswith("__scaling_evolve_binary_v1__:")

    restored = tmp_path / "restored"
    write_file_tree(restored, portable)

    assert (restored / "README.md").read_text(encoding="utf-8") == text_payload
    assert (restored / "preview.png").read_bytes() == binary_payload


def test_solver_population_persists_portable_binary_logs(tmp_path: Path) -> None:
    eval_logs = tmp_path / "evaluate_logs"
    eval_logs.mkdir()
    image_bytes = b"\x00\x01binary-artifact\x02\x03"
    (eval_logs / "score.yaml").write_text("score: 1.0\nsummary: ok\n", encoding="utf-8")
    (eval_logs / "preview.png").write_bytes(image_bytes)

    store = SQLiteLineageStore(tmp_path / "solver.sqlite3")
    artifact_store = FSArtifactStore(tmp_path / "artifacts", run_id="solver_run")
    population = SolverPopulation(
        store,
        artifact_store,
        run_id="solver_run",
        config=OmegaConf.create({}),
    )

    try:
        population.add(
            PopulationEntry(
                id="solver_a",
                files={"candidate.py": "print('ok')\n"},
                score={"score": 1.0, "summary": "ok"},
                logs={
                    f"evaluate/{path}": content
                    for path, content in read_file_tree(eval_logs).items()
                },
            )
        )

        restored = population.entries()[0]

        assert restored.logs["evaluate/score.yaml"] == "score: 1.0\nsummary: ok\n"
        assert restored.logs["evaluate/preview.png"].startswith("__scaling_evolve_binary_v1__:")

        materialized = tmp_path / "materialized_logs"
        write_file_tree(
            materialized,
            {
                path.removeprefix("evaluate/"): content
                for path, content in restored.logs.items()
                if path.startswith("evaluate/")
            },
        )
        assert (materialized / "preview.png").read_bytes() == image_bytes
    finally:
        store.close()
