"""Eve: two-population co-evolution algorithm.

Maintains a solver population T and an optimizer population O that co-evolve
across an explicit loop:

  Phase 1 — Sample optimizers I from O and solver examples J from T.
  Phase 2 — Each optimizer in I produces a new solver candidate,
            optionally producing a revised optimizer in the same step (parallelized).
  Phase 3 — Pairwise Elo tournament updates optimizer ratings in O.

Public surface:

    EveFactory   — assembles and runs the loop
    Eve          — the co-evolution loop itself
    PopulationEntry     — unified entry type for both populations
"""

from scaling_evolve.algorithms.eve.factory import EveFactory
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.runtime.imports import (
    ImportResult,
    ImportSpec,
    import_populations_from_run,
    parse_import_spec,
)
from scaling_evolve.algorithms.eve.workflow.loop import Eve

__all__ = [
    "Eve",
    "EveFactory",
    "PopulationEntry",
    "RepoTaskProblem",
    "ImportSpec",
    "ImportResult",
    "parse_import_spec",
    "import_populations_from_run",
]
