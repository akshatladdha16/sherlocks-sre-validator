from __future__ import annotations

import json
from pathlib import Path
from sre_bench.runner import run_evaluation
from sre_bench.schema import RCAOutput, Scenario, ScoreResult


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def _safe_diagnose(self, scenario: Scenario) -> RCAOutput:
        return RCAOutput(
            root_cause=f"diagnosed by {self.name} for {scenario.id}",
            contributing_factors=["factor-a"],
            recommended_action="apply fix",
            confidence=0.6,
            reasoning_trace="mock trace",
        )


def _load_one_scenario() -> Scenario:
    payload = json.loads(Path("scenarios/k8s_oom_easy.json").read_text(encoding="utf-8"))
    return Scenario.model_validate(payload)


def test_run_evaluation_dry_run_returns_score_result(tmp_path: Path) -> None:
    scenario = _load_one_scenario()
    agents = [
        _FakeAgent("simple_llm"),
        _FakeAgent("react_langchain"),
        _FakeAgent("multi_agent_crewai"),
    ]

    class _FakeJudge:
        def score(self, **_: object) -> ScoreResult:
            return ScoreResult(
                scenario_id=scenario.id,
                agent_name="simple_llm",
                rca_score=0.9,
                calibration_score=0.7,
                actionability_score=0.8,
                final_score=0.825,
                failure_mode="correct",
                judge_reasoning="mock judge",
                rca_output=RCAOutput(
                    root_cause="x",
                    contributing_factors=[],
                    recommended_action="y",
                    confidence=0.5,
                    reasoning_trace="z",
                ),
            )

    results, raw_outputs = run_evaluation(
        agents=agents,
        scenarios=[scenario],
        output_path=tmp_path / "run.json",
        dry_run=True,
        no_judge=False,
        judge=_FakeJudge(),
    )

    assert len(results) == 1
    assert raw_outputs == []
    assert results[0].final_score == 0.825
