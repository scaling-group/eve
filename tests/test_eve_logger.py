from __future__ import annotations

from types import SimpleNamespace

from scaling_evolve.algorithms.eve.logger.wandb import WandbEveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4Result


class _FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, object] = {}
        self.url = "https://wandb.example/run/test"


class _FakeSdk:
    def __init__(self) -> None:
        self.run = _FakeRun()
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.metrics: list[tuple[str, str | None, str | None]] = []
        self.finished = False
        self.tables: list[_FakeTable] = []

    def init(self, **kwargs: object) -> _FakeRun:
        _ = kwargs
        return self.run

    def define_metric(
        self,
        name: str,
        step_metric: str | None = None,
        summary: str | None = None,
    ) -> None:
        self.metrics.append((name, step_metric, summary))

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        self.logged.append((payload, step))

    def Table(self, columns: list[str] | None = None) -> _FakeTable:  # noqa: N802
        table = _FakeTable(columns=columns or [])
        self.tables.append(table)
        return table

    def finish(self) -> None:
        self.finished = True


class _FakeTable:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.data: list[list[object]] = []

    def add_data(self, *row: object) -> None:
        self.data.append(list(row))


def test_wandb_eve_logger_disabled_skips_sdk_load(monkeypatch) -> None:
    def _unexpected_load() -> None:
        raise AssertionError("wandb SDK should not load when logger is disabled")

    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.logger.wandb._load_wandb_sdk",
        _unexpected_load,
    )

    logger = WandbEveLogger(
        run_id="run-disabled",
        full_config={"logger": {"enabled": False}},
        enabled=False,
        excluded_score_fields=[],
    )

    assert logger._sdk is None
    assert logger._run is None


def test_wandb_eve_logger_logs_phase2_and_phase4(monkeypatch) -> None:
    fake_sdk = _FakeSdk()
    monkeypatch.setattr(
        "scaling_evolve.algorithms.eve.logger.wandb._load_wandb_sdk",
        lambda: fake_sdk,
    )

    logger = WandbEveLogger(
        run_id="run-123",
        full_config={"logger": {"enabled": True}},
        enabled=True,
        project="demo",
        excluded_score_fields=["info"],
    )

    phase2_results = [
        Phase2Result(
            optimizer=PopulationEntry(id="optimizer-a", files={}, score={"elo": 1400.0}, logs={}),
            sampled_solvers=[],
            prefill_solver=None,
            produced_solver=PopulationEntry(
                id="solver-a",
                files={},
                score={
                    "score": 1.25,
                    "info": {"target": 3.0},
                    "information": {"still_logged": 9.0},
                },
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id="optimizer-phase2-a",
                files={"APPROACH.md": "phase2-a"},
                score={"elo": 1412.0, "note": "phase2 optimizer a"},
                logs={},
            ),
            rollouts=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        cache_read_tokens=1,
                        cache_creation_tokens=0,
                        agent_turns=2,
                        model_cost_usd=0.4,
                        wallclock_seconds=4.0,
                    )
                )
            ],
            workspace_id="workspace-a",
        ),
        Phase2Result(
            optimizer=PopulationEntry(id="optimizer-b", files={}, score={"elo": 1450.0}, logs={}),
            sampled_solvers=[],
            prefill_solver=None,
            produced_solver=PopulationEntry(
                id="solver-b",
                files={},
                score={"score": 2.5},
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id="optimizer-phase2-b",
                files={"APPROACH.md": "phase2-b"},
                score={"elo": 1398.0},
                logs={},
            ),
            rollouts=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=20,
                        output_tokens=8,
                        cache_read_tokens=2,
                        cache_creation_tokens=3,
                        agent_turns=4,
                        model_cost_usd=0.6,
                        wallclock_seconds=6.0,
                    )
                )
            ],
            workspace_id="workspace-b",
        ),
    ]
    phase4_results = [
        Phase4Result(
            lead_optimizer=PopulationEntry(
                id="optimizer-lead",
                files={"APPROACH.md": "lead"},
                score={"elo": 1500.0},
                logs={},
            ),
            sampled_optimizers=[
                PopulationEntry(id="optimizer-a", files={}, score={"elo": 1400.0}, logs={}),
                PopulationEntry(id="optimizer-b", files={}, score={"elo": 1450.0}, logs={}),
            ],
            prefill_optimizer=PopulationEntry(
                id="optimizer-prefill",
                files={"APPROACH.md": "prefill"},
                score={"elo": 1425.0},
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id="optimizer-new",
                files={"APPROACH.md": "new"},
                score={"elo": 1425.0},
                logs={},
            ),
            rollouts=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=7,
                        output_tokens=3,
                        cache_read_tokens=0,
                        cache_creation_tokens=1,
                        agent_turns=1,
                        model_cost_usd=0.75,
                        wallclock_seconds=2.0,
                    )
                )
            ],
            workspace_id="workspace-123",
        ),
        Phase4Result(
            lead_optimizer=PopulationEntry(
                id="optimizer-lead-2",
                files={"APPROACH.md": "lead-2"},
                score={"elo": 1490.0},
                logs={},
            ),
            sampled_optimizers=[
                PopulationEntry(id="optimizer-a", files={}, score={"elo": 1400.0}, logs={}),
            ],
            prefill_optimizer=PopulationEntry(
                id="optimizer-prefill-2",
                files={"APPROACH.md": "prefill-2"},
                score={"elo": 1410.0},
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id="optimizer-new-2",
                files={"APPROACH.md": "new-2"},
                score={"elo": 1410.0},
                logs={},
            ),
            rollouts=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=11,
                        output_tokens=4,
                        cache_read_tokens=2,
                        cache_creation_tokens=0,
                        agent_turns=2,
                        model_cost_usd=0.25,
                        wallclock_seconds=5.0,
                    )
                )
            ],
            workspace_id="workspace-456",
        ),
    ]

    logger.on_iteration(
        iteration=3,
        solver_entries=[
            PopulationEntry(id="solver-seed", files={}, score={"score": 0.5}, logs={}),
            PopulationEntry(id="solver-a", files={}, score={"score": 1.25}, logs={}),
            PopulationEntry(id="solver-b", files={}, score={"score": 2.5}, logs={}),
            PopulationEntry(id="solver-c", files={}, score={"score": 1.8}, logs={}),
            PopulationEntry(id="solver-d", files={}, score={"score": 1.9}, logs={}),
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-seed", files={}, score={"elo": 1300.0}, logs={}),
            PopulationEntry(id="optimizer-a", files={}, score={"elo": 1400.0}, logs={}),
            PopulationEntry(id="optimizer-b", files={}, score={"elo": 1450.0}, logs={}),
            PopulationEntry(id="optimizer-new", files={}, score={"elo": 1425.0}, logs={}),
        ],
        phase2_results=phase2_results,
        phase4_results=phase4_results,
    )
    logger.finish(
        solver_entries=[
            PopulationEntry(id="solver-a", files={}, score={"score": 1.25}, logs={}),
            PopulationEntry(id="solver-b", files={}, score={"score": 2.5}, logs={}),
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-a", files={}, score={"elo": 1400.0}, logs={}),
            PopulationEntry(id="optimizer-new", files={}, score={"elo": 1425.0}, logs={}),
        ],
        iterations_completed=8,
    )

    assert fake_sdk.logged
    payload, step = fake_sdk.logged[-1]
    assert step == 3
    assert payload["phase2/scores/score"] == [1.25, 2.5]
    assert payload["phase2/iteration/mean/score"] == 1.875
    assert payload["phase2/iteration/max/score"] == 2.5
    assert payload["phase2/cumulative/max/score"] == 2.5
    assert payload["usage/iteration/input_tokens"] == 48.0
    assert payload["usage/iteration/output_tokens"] == 20.0
    assert payload["usage/iteration/cache_read_tokens"] == 5.0
    assert payload["usage/iteration/cache_creation_tokens"] == 4.0
    assert payload["usage/iteration/agent_turns"] == 9.0
    assert payload["usage/iteration/wallclock_seconds"] == 17.0
    assert payload["usage/cumulative/input_tokens"] == 48.0
    assert payload["usage/cumulative/output_tokens"] == 20.0
    assert payload["usage/cumulative/cache_read_tokens"] == 5.0
    assert payload["usage/cumulative/cache_creation_tokens"] == 4.0
    assert payload["usage/cumulative/agent_turns"] == 9.0
    assert payload["usage/cumulative/wallclock_seconds"] == 17.0
    assert payload["usage/phase2/input_tokens"] == 30
    assert payload["usage/phase2/output_tokens"] == 13
    assert payload["usage/phase2/cache_read_tokens"] == 3
    assert payload["usage/phase2/cache_creation_tokens"] == 3
    assert payload["usage/phase2/agent_turns"] == 6
    assert payload["usage/phase2/wallclock_seconds"] == 10.0
    assert payload["usage/phase2/model_cost_usd"] == 1.0
    assert payload["usage/phase4/model_cost_usd"] == 1.0
    assert payload["phase4/scores/elo"] == [1425.0, 1410.0]
    assert payload["phase4/iteration/mean/elo"] == 1417.5
    assert payload["phase4/iteration/max/elo"] == 1425.0
    assert payload["phase4/cumulative/max/elo"] == 1425.0
    assert payload["usage/phase4/input_tokens"] == 18
    assert payload["usage/phase4/output_tokens"] == 7
    assert payload["usage/phase4/agent_turns"] == 3
    assert payload["population/solver_size"] == 5
    assert payload["population/optimizer_size"] == 4
    assert payload["usage/iteration/model_cost_usd"] == 2.0
    assert payload["usage/cumulative/phase2/model_cost_usd"] == 1.0
    assert payload["usage/cumulative/phase4/model_cost_usd"] == 1.0
    assert payload["usage/cumulative/model_cost_usd"] == 2.0
    phase2_solver_table = payload["tables/phase2_solvers"]
    assert isinstance(phase2_solver_table, _FakeTable)
    assert "primary_score" not in phase2_solver_table.columns
    assert "score_json" in phase2_solver_table.columns
    assert "score" not in phase2_solver_table.columns
    assert "information/still_logged" not in phase2_solver_table.columns
    assert "solver_id" in phase2_solver_table.columns
    assert len(phase2_solver_table.data) == 2
    score_json_idx = phase2_solver_table.columns.index("score_json")
    solver_id_idx = phase2_solver_table.columns.index("solver_id")
    assert phase2_solver_table.data[0][solver_id_idx] == "solver-a"
    assert phase2_solver_table.data[0][score_json_idx] == (
        '{"info": {"target": 3.0}, "information": {"still_logged": 9.0}, "score": 1.25}'
    )
    phase2_optimizer_table = payload["tables/phase2_optimizers"]
    assert isinstance(phase2_optimizer_table, _FakeTable)
    assert "optimizer_id" in phase2_optimizer_table.columns
    assert len(phase2_optimizer_table.data) == 2
    phase2_optimizer_score_json_idx = phase2_optimizer_table.columns.index("score_json")
    phase2_optimizer_id_idx = phase2_optimizer_table.columns.index("optimizer_id")
    assert phase2_optimizer_table.data[0][phase2_optimizer_id_idx] == "optimizer-phase2-a"
    assert phase2_optimizer_table.data[0][phase2_optimizer_score_json_idx] == (
        '{"elo": 1412.0, "note": "phase2 optimizer a"}'
    )
    assert phase2_optimizer_table.data[1][phase2_optimizer_score_json_idx] == '{"elo": 1398.0}'
    phase4_optimizer_table = payload["tables/phase4_optimizers"]
    assert isinstance(phase4_optimizer_table, _FakeTable)
    assert "primary_score" not in phase4_optimizer_table.columns
    assert "score_json" in phase4_optimizer_table.columns
    assert "elo" not in phase4_optimizer_table.columns
    assert "optimizer_id" in phase4_optimizer_table.columns
    assert len(phase4_optimizer_table.data) == 2
    phase4_score_json_idx = phase4_optimizer_table.columns.index("score_json")
    assert phase4_optimizer_table.data[0][phase4_score_json_idx] == '{"elo": 1425.0}'
    assert phase4_optimizer_table.data[1][phase4_score_json_idx] == '{"elo": 1410.0}'
    assert fake_sdk.run.summary["total_iterations"] == 8
    assert fake_sdk.finished is True
