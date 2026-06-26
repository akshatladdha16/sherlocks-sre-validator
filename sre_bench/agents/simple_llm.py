from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from sre_bench.agents.base import BaseAgent
from sre_bench.schema import RCAOutput, Scenario


SYSTEM_PROMPT = """You are a senior SRE diagnosing a production incident.
You will be given: an alert message, recent application logs, current metrics,
recent deployment events, and team Slack context.

Analyze all signals carefully. Your output must be a JSON object with these exact keys:
- root_cause: string (1-2 sentences, the definitive root cause)
- contributing_factors: list of strings
- recommended_action: string (specific, actionable, safe)
- confidence: float 0.0-1.0 (how certain you are given the evidence)
- reasoning_trace: string (walk through your thinking step by step before concluding)

Important: Do not hallucinate service names or metrics not present in the input.
If you are unsure, lower your confidence score and say so in reasoning_trace.
Respond with JSON only. No markdown fences.
"""


class SimpleLLMAgent(BaseAgent):
    name = "simple_llm"

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.client = client or OpenAI()

    def diagnose(self, scenario: Scenario) -> RCAOutput:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._format_scenario_context(scenario)},
                ],
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned empty content")

            return RCAOutput.model_validate_json(content)
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Exception: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _format_scenario_context(scenario: Scenario) -> str:
        logs_block = "\n".join(scenario.context.logs)
        metrics_block = "\n".join(
            f"{key}: {value}" for key, value in scenario.context.metrics.items()
        )
        if scenario.context.recent_deploys:
            deploys_block = "\n".join(
                f"{deploy.time_offset}: [{deploy.service}] {deploy.change_summary}"
                for deploy in scenario.context.recent_deploys
            )
        else:
            deploys_block = "No recent deployments in the last 4 hours."

        return (
            f"ALERT: {scenario.context.alert_message}\n\n"
            f"LOGS (most recent first):\n{logs_block}\n\n"
            f"METRICS:\n{metrics_block}\n\n"
            f"RECENT DEPLOYS:\n{deploys_block}\n\n"
            f"SLACK CONTEXT:\n{scenario.context.slack_context}"
        )
