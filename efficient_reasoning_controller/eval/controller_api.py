from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMDesignedMethod(ABC):
    """
    A complete reasoning-time method for the 2D probing environment.

    The method is responsible for the whole control algorithm:
    probing, stopping, pruning, voting, and final answer selection.
    """

    NAME = "unnamed_method"

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def solve(self, question) -> str:
        """
        Run the full reasoning-control algorithm on one question
        and return the final answer.
        """
        pass

    def description(self) -> str:
        if not self.config:
            return self.NAME
        return f"{self.NAME} | config={self.config}"


class MethodStrategyAdapter:
    """
    Adapter for compatibility with the old evaluation pipeline.

    Old solvers expect a 'branch_strategy' object with:
      - execute(question)
      - description()
    This adapter wraps a full method and exposes that interface.
    """

    def __init__(self, method: LLMDesignedMethod):
        self.method = method

    def execute(self, question) -> str:
        return self.method.solve(question)

    def description(self) -> str:
        return self.method.description()


class MethodTraceRecorder:
    """
    Lightweight trace recorder for storing step-level method states.
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, **kwargs):
        self.steps.append(dict(kwargs))

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.steps)