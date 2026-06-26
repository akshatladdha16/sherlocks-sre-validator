"""Agent implementations for SRE-Bench."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sre_bench.agents.multi_agent import MultiAgentCrewAgent
    from sre_bench.agents.react_agent import ReActAgent
    from sre_bench.agents.simple_llm import SimpleLLMAgent

__all__ = ["SimpleLLMAgent", "ReActAgent", "MultiAgentCrewAgent"]


def __getattr__(name: str) -> Any:
    if name == "SimpleLLMAgent":
        from sre_bench.agents.simple_llm import SimpleLLMAgent

        return SimpleLLMAgent
    if name == "ReActAgent":
        from sre_bench.agents.react_agent import ReActAgent

        return ReActAgent
    if name == "MultiAgentCrewAgent":
        from sre_bench.agents.multi_agent import MultiAgentCrewAgent

        return MultiAgentCrewAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
