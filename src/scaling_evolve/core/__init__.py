"""Core data model surface shared by Eve runtime components."""

from scaling_evolve.core.engine import ExecutionLifecycle, PortableState, RuntimeState
from scaling_evolve.core.evaluation import EvalStatus, EvaluationResult, ScoreCard
from scaling_evolve.core.mutation import (
    MutationInstruction,
    MutationInstructionLike,
    MutationRequest,
    MutationResult,
    ProviderSpec,
)
from scaling_evolve.core.node import ArtifactRef, EdgeRecord, Node, NodeRecord

__all__ = [
    "ArtifactRef",
    "EdgeRecord",
    "EvaluationResult",
    "EvalStatus",
    "ExecutionLifecycle",
    "MutationInstruction",
    "MutationInstructionLike",
    "MutationRequest",
    "MutationResult",
    "Node",
    "NodeRecord",
    "PortableState",
    "ProviderSpec",
    "RuntimeState",
    "ScoreCard",
]
