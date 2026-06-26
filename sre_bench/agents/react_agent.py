from __future__ import annotations

import json
import os
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from sre_bench.agents.base import BaseAgent
from sre_bench.schema import RCAOutput, Scenario


REACT_SYSTEM_PROMPT = """You are a senior SRE investigating a production incident.

You have access to these tools:
{tools}

Available tool names: {tool_names}

Use them to gather evidence before concluding. Always call get_deploy_history first.
Then search logs for errors. Then check metrics.

When you have enough evidence, think step by step and form a conclusion.

ALERT: {alert}
SLACK CONTEXT: {slack}

Use this format:
Thought: what do I need to investigate?
Action: tool name
Action Input: query string
Observation: tool result
... (repeat until confident)
Thought: I now have enough evidence to conclude
Final Answer: JSON with keys: root_cause, contributing_factors, recommended_action, confidence, reasoning_trace

Question: {input}
Thought: {agent_scratchpad}
"""


def make_tools(scenario: Scenario) -> list[Any]:
    @tool
    def search_logs(query: str) -> str:
        """
        Search application logs for lines matching the query string.
        Returns matching log lines or 'No matching logs found'.
        Use this to find errors, stack traces, or specific service mentions.
        """
        matches = [line for line in scenario.context.logs if query.lower() in line.lower()]
        return "\n".join(matches) if matches else "No matching logs found."

    @tool
    def get_metrics(service_or_metric: str) -> str:
        """
        Get current metric values. Optionally filter by service or metric name.
        Returns key-value metric pairs relevant to the query.
        Use this to check CPU, memory, latency, error rates, connection counts.
        """
        query = service_or_metric.strip()
        if query == "*":
            selected = scenario.context.metrics
        else:
            selected = {
                key: value
                for key, value in scenario.context.metrics.items()
                if query.lower() in key.lower()
            }

        if not selected:
            available = ", ".join(scenario.context.metrics.keys())
            return (
                f"No metrics found matching '{service_or_metric}'. "
                f"Available: [{available}]"
            )

        return "\n".join(f"{key}: {value}" for key, value in selected.items())

    @tool
    def get_deploy_history() -> str:
        """
        Get recent deployment history. Always call this first.
        Returns deployments with time offsets and change summaries.
        """
        if not scenario.context.recent_deploys:
            return "No recent deployments in the last 4 hours."
        return "\n".join(
            f"{deploy.time_offset}: [{deploy.service}] {deploy.change_summary}"
            for deploy in scenario.context.recent_deploys
        )

    return [search_logs, get_metrics, get_deploy_history]


class ReActAgent(BaseAgent):
    name = "react_langchain"

    def __init__(
        self,
        model: str | None = None,
        llm: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.llm = llm or ChatOpenAI(model=self.model, temperature=0.2)
        self.verbose = verbose

    def diagnose(self, scenario: Scenario) -> RCAOutput:
        tools = make_tools(scenario)
        prompt = PromptTemplate.from_template(REACT_SYSTEM_PROMPT)
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=8,
            handle_parsing_errors=True,
            verbose=self.verbose,
        )

        try:
            result = executor.invoke(
                {
                    "input": "Investigate this incident and provide the final RCA JSON.",
                    "alert": scenario.context.alert_message,
                    "slack": scenario.context.slack_context,
                }
            )
            raw_output = str(result.get("output", "")).strip()
            if not raw_output:
                raise ValueError("ReAct agent returned empty output")
            return self._parse_final_answer(raw_output)
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Exception: {type(exc).__name__}: {exc}",
            )

    def _parse_final_answer(self, raw_output: str) -> RCAOutput:
        try:
            return RCAOutput.model_validate_json(raw_output)
        except Exception:
            pass

        json_blob = self._extract_json_object(raw_output)
        if not json_blob:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=(
                    "Failed to parse ReAct final answer as JSON. "
                    f"Raw output: {raw_output}"
                ),
            )

        try:
            parsed = json.loads(json_blob)
            return RCAOutput.model_validate(parsed)
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Failed to parse extracted JSON: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start : end + 1]
