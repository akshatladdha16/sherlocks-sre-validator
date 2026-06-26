from __future__ import annotations

from abc import ABC, abstractmethod

from sre_bench.schema import RCAOutput, Scenario


class BaseAgent(ABC):
    """
    All agents implement this interface.
    The runner calls diagnose() and expects RCAOutput back.
    Internal complexity is handled by each agent implementation.
    all agents to be created in future shall inherit from this base agnet and folow the same input output schema for all described methods.
    """

    name: str

    @abstractmethod
    def diagnose(self, scenario: Scenario) -> RCAOutput:
        """
        Given a full Scenario, return a structured RCAOutput.
        Implementations should use _safe_diagnose() at call sites to avoid run crashes.
        """
        raise NotImplementedError

    def _safe_diagnose(self, scenario: Scenario) -> RCAOutput:
        """Wrap diagnose() with error handling for the runner."""
        try:
            return self.diagnose(scenario)
        except Exception as exc:  # pragma: no cover - defensive fallback
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Exception: {type(exc).__name__}: {exc}",
            )
