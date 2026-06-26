from __future__ import annotations

import json
from types import SimpleNamespace

from sre_bench.evaluators.judge import LLMJudge, compute_final_score
from sre_bench.schema import RCAOutput, Scenario


def _build_scenario() -> Scenario:
    payload = {
        "id": "unit_test_scenario",
        "category": "database",
        "difficulty": "easy",
        "context": {
            "alert_message": "db slow",
            "logs": ["2026-01-01 WARN slow query"],
            "metrics": {"db_avg_query_time": "2400ms", "api_p99_latency": "4200ms"},
            "recent_deploys": [
                {
                    "time_offset": "-90m",
                    "service": "orders-api",
                    "change_summary": "new search endpoint",
                }
            ],
            "slack_context": "Search endpoint looks hot",
        },
        "ground_truth": {
            "root_cause": "Missing index on orders.status",
            "contributing_factors": ["high query load"],
            "correct_action": "Create index on orders(status)",
        },
    }
    return Scenario.model_validate(payload)


def _build_output(root_cause: str, confidence: float = 0.9) -> RCAOutput:
    return RCAOutput(
        root_cause=root_cause,
        contributing_factors=["high query load"],
        recommended_action="Create index on orders(status)",
        confidence=confidence,
        reasoning_trace="Evidence summary",
    )


def _make_openai_client(response_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
                )
            )
        )
    )


def test_perfect_rca_scores_high() -> None:
    scenario = _build_scenario()
    output = _build_output("Missing index on orders.status", confidence=0.95)
    judge_payload = json.dumps(
        {
            "rca_score": 1.0,
            "calibration_score": 1.0,
            "actionability_score": 1.0,
            "failure_mode": "correct",
            "judge_reasoning": "Root cause and action match exactly.",
        }
    )
    judge = LLMJudge(openai_client=_make_openai_client(judge_payload))

    score = judge.score(scenario=scenario, rca_output=output, agent_name="simple_llm")

    assert score.final_score == 1.0
    assert score.failure_mode == "correct"


def test_wrong_rca_scores_low() -> None:
    scenario = _build_scenario()
    output = _build_output("Redis memory leak in cache cluster", confidence=0.95)
    judge_payload = json.dumps(
        {
            "rca_score": 0.0,
            "calibration_score": 0.0,
            "actionability_score": 0.1,
            "failure_mode": "hallucinated_service",
            "judge_reasoning": "Diagnosis is unrelated to the scenario evidence.",
        }
    )
    judge = LLMJudge(openai_client=_make_openai_client(judge_payload))

    score = judge.score(scenario=scenario, rca_output=output, agent_name="simple_llm")

    assert score.final_score <= 0.1
    assert score.failure_mode == "hallucinated_service"


def test_judge_json_parse_failure_returns_zero_scores() -> None:
    scenario = _build_scenario()
    output = _build_output("Missing index on orders.status")
    judge = LLMJudge(openai_client=_make_openai_client("not valid json"))

    score = judge.score(scenario=scenario, rca_output=output, agent_name="simple_llm")

    assert score.final_score == 0.0
    assert score.rca_score == 0.0
    assert score.calibration_score == 0.0
    assert score.actionability_score == 0.0
    assert score.failure_mode == "other"


def test_compute_final_score_weighting() -> None:
    score = compute_final_score(0.8, 0.4, 0.6)
    assert score == 0.65
