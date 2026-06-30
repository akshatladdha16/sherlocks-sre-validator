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
Use no more than 5 tool calls before concluding.

ALERT: {alert}
SLACK CONTEXT: {slack}

Use this format:
Thought: what do I need to investigate?
Action: tool name
Action Input: query string (for get_deploy_history use "*")
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
        query_text = str(query or "").strip().lower()
        if not query_text:
            return "No matching logs found."
        matches = [line for line in scenario.context.logs if query_text in line.lower()]
        return "\n".join(matches) if matches else "No matching logs found."

    @tool
    def get_metrics(service_or_metric: str) -> str:
        """
        Get current metric values. Optionally filter by service or metric name.
        Returns key-value metric pairs relevant to the query.
        Use this to check CPU, memory, latency, error rates, connection counts.
        """
        query = str(service_or_metric or "").strip()
        if query.lower() in {"none", "null"}:
            query = "*"
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
    def get_deploy_history(query: str = "*") -> str:
        """
        Get recent deployment history. Always call this first.
        Input is ignored; pass '*' as Action Input.
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
            max_iterations=12,
            handle_parsing_errors=(
                "Invalid format. Use Thought/Action/Action Input/Observation. "
                "Always provide Action Input and end with Final Answer JSON."
            ),
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
            return self._parse_final_answer(raw_output, scenario)
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Exception: {type(exc).__name__}: {exc}",
            )

    def _parse_final_answer(self, raw_output: str, scenario: Scenario) -> RCAOutput:
        if "Agent stopped due to iteration limit" in raw_output:
            return self._synthesize_without_tools(
                scenario=scenario,
                note="ReAct loop reached iteration/time limit before final answer.",
            )

        try:
            return RCAOutput.model_validate_json(raw_output)
        except Exception:
            pass

        json_blob = self._extract_json_object(raw_output)
        if not json_blob:
            return self._synthesize_without_tools(
                scenario=scenario,
                note=f"Failed to parse ReAct final answer as JSON. Raw output: {raw_output}",
            )

        try:
            parsed = json.loads(json_blob)
            return self._validate_rca_payload(parsed)
        except Exception as exc:
            return self._synthesize_without_tools(
                scenario=scenario,
                note=f"Failed to parse extracted JSON: {type(exc).__name__}: {exc}",
            )

    def _synthesize_without_tools(self, scenario: Scenario, note: str) -> RCAOutput:
        logs = "\n".join(scenario.context.logs)
        metrics = "\n".join(
            f"{key}: {value}" for key, value in scenario.context.metrics.items()
        )
        deploys = "\n".join(
            f"{deploy.time_offset}: [{deploy.service}] {deploy.change_summary}"
            for deploy in scenario.context.recent_deploys
        ) or "No recent deployments in the last 4 hours."

        fallback_prompt = (
            "You are a senior SRE. Return JSON only with keys: "
            "root_cause, contributing_factors, recommended_action, confidence, reasoning_trace.\n\n"
            f"ALERT: {scenario.context.alert_message}\n"
            f"SLACK: {scenario.context.slack_context}\n\n"
            f"LOGS:\n{logs}\n\n"
            f"METRICS:\n{metrics}\n\n"
            f"DEPLOYS:\n{deploys}\n\n"
            f"CONTEXT: {note}"
        )
        try:
            response = self.llm.invoke(fallback_prompt)
            content = str(getattr(response, "content", "")).strip()
            if not content:
                raise ValueError("Fallback synthesis returned empty content")

            try:
                return RCAOutput.model_validate_json(content)
            except Exception:
                json_blob = self._extract_json_object(content)
                if not json_blob:
                    raise ValueError("Fallback synthesis did not return JSON")
                return self._validate_rca_payload(json.loads(json_blob))
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Fallback synthesis failed: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _validate_rca_payload(payload: dict[str, Any]) -> RCAOutput:
        normalized = dict(payload)

        confidence = normalized.get("confidence", 0.0)
        if isinstance(confidence, str):
            lowered = confidence.strip().lower()
            if lowered in {"high", "very high"}:
                confidence = 0.85
            elif lowered in {"medium", "moderate"}:
                confidence = 0.55
            elif lowered in {"low", "very low"}:
                confidence = 0.25
            else:
                try:
                    confidence = float(lowered)
                except ValueError:
                    confidence = 0.0
        normalized["confidence"] = float(confidence)

        factors = normalized.get("contributing_factors", [])
        if isinstance(factors, str):
            factors = [factors]
        normalized["contributing_factors"] = [str(item) for item in factors]

        reasoning = normalized.get("reasoning_trace", "")
        if isinstance(reasoning, list):
            reasoning = " ".join(str(item) for item in reasoning)
        normalized["reasoning_trace"] = str(reasoning)

        normalized["root_cause"] = str(normalized.get("root_cause", "Agent failed to produce output"))
        normalized["recommended_action"] = str(normalized.get("recommended_action", "N/A"))

        return RCAOutput.model_validate(normalized)

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start : end + 1]
