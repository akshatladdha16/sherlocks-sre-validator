from __future__ import annotations

import json
import os
from typing import Any

from sre_bench.schema import RCAOutput, Scenario, ScoreResult

JUDGE_SYSTEM_PROMPT = """
You are an expert SRE evaluating the quality of an AI agent's incident diagnosis.

You will be given:
1. The AGENT'S RCA: their diagnosis of a production incident
2. The GROUND TRUTH: the actual root cause and correct action
3. The SCORING RUBRIC below

Score the agent on three dimensions, each 0.0 to 1.0:

RCA_SCORE (weight: 0.5) — Root cause correctness:
  1.0 = correctly identified the specific root cause (exact match or semantically equivalent)
  0.7 = identified the right layer/component but missed the specific mechanism
  0.4 = partially correct — named a contributing factor but not the primary cause
  0.0 = wrong — named a different service, a symptom, or hallucinated

CALIBRATION_SCORE (weight: 0.25) — Confidence calibration:
  1.0 = high confidence when correct, appropriately hedged when uncertain
  0.5 = overconfident when wrong, or underconfident when right
  0.0 = stated certainty was completely wrong (confident but hallucinated)

ACTIONABILITY_SCORE (weight: 0.25) — Quality of recommended action:
  1.0 = specific, safe, correct fix that directly addresses the root cause
  0.7 = correct direction but vague (e.g. "check the deploy" instead of "roll back X")
  0.3 = generic advice not specific to this incident
  0.0 = dangerous, irrelevant, or no action given

Also assign one FAILURE_MODE label from this list:
  correct, wrong_layer, hallucinated_service, missed_deploy_signal,
  symptom_not_cause, missed_upstream, over_hedged, ignored_slack_context,
  vague_action, other(mention and explain in reasoning)

Return JSON only, no markdown fences:
{
  "rca_score": float,
  "calibration_score": float,
  "actionability_score": float,
  "failure_mode": string,
  "judge_reasoning": string (2-3 sentences explaining scores)
}
""".strip()

JUDGE_USER_TEMPLATE = """
AGENT'S RCA:
Root cause: {agent_root_cause}
Contributing factors: {agent_factors}
Recommended action: {agent_action}
Confidence: {agent_confidence}

GROUND TRUTH:
Root cause: {true_root_cause}
Contributing factors: {true_factors}
Correct action: {true_action}
""".strip()

_VALID_FAILURE_MODES = {
    "correct",
    "wrong_layer",
    "hallucinated_service",
    "missed_deploy_signal",
    "symptom_not_cause",
    "missed_upstream",
    "over_hedged",
    "ignored_slack_context",
    "vague_action",
    "other",
}


def compute_final_score(rca: float, cal: float, act: float) -> float:
    return round(rca * 0.5 + cal * 0.25 + act * 0.25, 4)


class LLMJudge:
    def __init__(
        self,
        model: str | None = None,
        openai_client: Any | None = None,
        anthropic_client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("JUDGE_MODEL", "gpt-4o")
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client

    def score(self, scenario: Scenario, rca_output: RCAOutput, agent_name: str) -> ScoreResult:
        try:
            raw_judge = self._call_judge(scenario, rca_output)
            parsed = self._parse_judge_output(raw_judge)
            final_score = compute_final_score(
                parsed["rca_score"],
                parsed["calibration_score"],
                parsed["actionability_score"],
            )
            return ScoreResult(
                scenario_id=scenario.id,
                agent_name=agent_name,
                rca_score=parsed["rca_score"],
                calibration_score=parsed["calibration_score"],
                actionability_score=parsed["actionability_score"],
                final_score=final_score,
                failure_mode=parsed["failure_mode"],
                judge_reasoning=parsed["judge_reasoning"],
                rca_output=rca_output,
            )
        except Exception as exc:
            return self._fallback_result(
                scenario_id=scenario.id,
                agent_name=agent_name,
                rca_output=rca_output,
                reason=f"Judge failed: {type(exc).__name__}: {exc}",
            )

    def _call_judge(self, scenario: Scenario, rca_output: RCAOutput) -> str:
        user_prompt = JUDGE_USER_TEMPLATE.format(
            agent_root_cause=rca_output.root_cause,
            agent_factors=", ".join(rca_output.contributing_factors) or "None",
            agent_action=rca_output.recommended_action,
            agent_confidence=rca_output.confidence,
            true_root_cause=scenario.ground_truth.root_cause,
            true_factors=", ".join(scenario.ground_truth.contributing_factors) or "None",
            true_action=scenario.ground_truth.correct_action,
        )

        return self._call_openai(user_prompt)

    def _call_openai(self, user_prompt: str) -> str:
        client = self.openai_client
        if client is None:
            from openai import OpenAI

            client = OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge model returned empty response")
        return content

    def _parse_judge_output(self, raw: str) -> dict[str, Any]:
        payload = self._parse_json(raw)

        required = [
            "rca_score",
            "calibration_score",
            "actionability_score",
            "failure_mode",
            "judge_reasoning",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Judge response missing keys: {', '.join(missing)}")

        rca_score = self._coerce_score(payload["rca_score"], "rca_score")
        calibration_score = self._coerce_score(
            payload["calibration_score"], "calibration_score"
        )
        actionability_score = self._coerce_score(
            payload["actionability_score"], "actionability_score"
        )

        failure_mode = str(payload["failure_mode"]).strip()
        if failure_mode not in _VALID_FAILURE_MODES:
            failure_mode = "other"

        judge_reasoning = str(payload["judge_reasoning"]).strip()
        if not judge_reasoning:
            judge_reasoning = "No judge reasoning provided."

        return {
            "rca_score": rca_score,
            "calibration_score": calibration_score,
            "actionability_score": actionability_score,
            "failure_mode": failure_mode,
            "judge_reasoning": judge_reasoning,
        }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
            raise ValueError("Judge response must be a JSON object")
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Judge response is not valid JSON")
            candidate = raw[start : end + 1]
            loaded = json.loads(candidate)
            if not isinstance(loaded, dict):
                raise ValueError("Judge response must be a JSON object")
            return loaded

    @staticmethod
    def _coerce_score(value: Any, key: str) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} is not numeric: {value}") from exc

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    @staticmethod
    def _fallback_result(
        scenario_id: str,
        agent_name: str,
        rca_output: RCAOutput,
        reason: str,
    ) -> ScoreResult:
        return ScoreResult(
            scenario_id=scenario_id,
            agent_name=agent_name,
            rca_score=0.0,
            calibration_score=0.0,
            actionability_score=0.0,
            final_score=0.0,
            failure_mode="other",
            judge_reasoning=reason,
            rca_output=rca_output,
        )
