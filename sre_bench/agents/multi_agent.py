from __future__ import annotations

import json
import os
from typing import Any

from crewai import Agent, Crew, Process, Task

from sre_bench.agents.base import BaseAgent
from sre_bench.schema import RCAOutput, Scenario


class MultiAgentCrewAgent(BaseAgent):
    name = "multi_agent_crewai"

    def __init__(self, model: str | None = None, llm: Any | None = None, verbose: bool = False) -> None:
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.llm = llm
        self.verbose = verbose

    def diagnose(self, scenario: Scenario) -> RCAOutput:
        try:
            log_analyst = Agent(
                role="Log analyst",
                goal="Identify all errors, exceptions, and anomalies in the application logs",
                backstory=(
                    "Expert at reading distributed system logs. Finds patterns, "
                    "correlates timestamps, identifies the first error in a chain."
                ),
                llm=self.llm or self.model,
                verbose=self.verbose,
            )
            metrics_analyst = Agent(
                role="Metrics analyst",
                goal="Identify which metrics are anomalous and what they indicate about system health",
                backstory=(
                    "SRE specialist in telemetry. Knows normal baselines and can identify "
                    "anomalies by magnitude and timing."
                ),
                llm=self.llm or self.model,
                verbose=self.verbose,
            )
            deploy_analyst = Agent(
                role="Deploy analyst",
                goal="Correlate recent deployments with the incident timeline and identify causal changes",
                backstory=(
                    "Deployment expert. Always checks what changed before an incident and "
                    "tests timing correlation carefully."
                ),
                llm=self.llm or self.model,
                verbose=self.verbose,
            )
            synthesizer = Agent(
                role="SRE synthesizer",
                goal="Combine findings from logs, metrics, and deploys into a definitive RCA",
                backstory=(
                    "Principal SRE who resolves conflicting signals and proposes safe, "
                    "specific remediation actions."
                ),
                llm=self.llm or self.model,
                verbose=self.verbose,
            )

            logs_input = "\n".join(scenario.context.logs)
            metrics_input = "\n".join(
                f"{key}: {value}" for key, value in scenario.context.metrics.items()
            )
            deploy_input = (
                "\n".join(
                    f"{deploy.time_offset}: [{deploy.service}] {deploy.change_summary}"
                    for deploy in scenario.context.recent_deploys
                )
                if scenario.context.recent_deploys
                else "No recent deployments in the last 4 hours."
            )

            log_task = Task(
                description=(
                    "Analyze these logs and identify key findings, a timeline of errors, "
                    f"and any stack traces.\nAlert context: {scenario.context.alert_message}\n"
                    f"Logs:\n{logs_input}"
                ),
                expected_output="Structured findings from logs with timeline and likely first failure signal.",
                agent=log_analyst,
            )
            metrics_task = Task(
                description=(
                    "Analyze these metrics and identify anomalies, severity, and likely impacted components.\n"
                    f"Alert context: {scenario.context.alert_message}\nMetrics:\n{metrics_input}"
                ),
                expected_output="Structured metrics findings with anomaly list and interpretation.",
                agent=metrics_analyst,
            )
            deploy_task = Task(
                description=(
                    "Analyze recent deploys and correlate timing with incident onset. "
                    "Return most suspicious change and plausible mechanism.\n"
                    f"Alert context: {scenario.context.alert_message}\n"
                    f"Slack context: {scenario.context.slack_context}\n"
                    f"Recent deploys:\n{deploy_input}"
                ),
                expected_output="Most suspicious deploy with timing argument and mechanism.",
                agent=deploy_analyst,
            )
            synth_task = Task(
                description=(
                    "Synthesize findings and output JSON only with keys: "
                    "root_cause, contributing_factors, recommended_action, confidence, reasoning_trace. "
                    "Set confidence as a float between 0.0 and 1.0. "
                    "Set reasoning_trace as a single string (not a list)."
                ),
                expected_output=(
                    "JSON object with keys root_cause, contributing_factors, "
                    "recommended_action, confidence, reasoning_trace"
                ),
                agent=synthesizer,
                context=[log_task, metrics_task, deploy_task],
            )

            crew = Crew(
                agents=[log_analyst, metrics_analyst, deploy_analyst, synthesizer],
                tasks=[log_task, metrics_task, deploy_task, synth_task],
                process=Process.sequential,
                verbose=self.verbose,
            )

            result = crew.kickoff()
            raw_output = str(result).strip()
            if not raw_output:
                raise ValueError("Crew returned empty output")
            return self._parse_synthesizer_output(raw_output)
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Exception: {type(exc).__name__}: {exc}",
            )

    def _parse_synthesizer_output(self, raw_output: str) -> RCAOutput:
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
                    "Failed to parse multi-agent output as JSON. "
                    f"Raw output: {raw_output}"
                ),
            )

        try:
            parsed = json.loads(json_blob)
            return self._validate_rca_payload(parsed)
        except Exception as exc:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Failed to parse extracted JSON: {type(exc).__name__}: {exc}",
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
