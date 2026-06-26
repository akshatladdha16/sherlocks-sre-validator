from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DeployEvent(BaseModel):
    time_offset: str
    service: str
    change_summary: str


class ScenarioContext(BaseModel):
    alert_message: str
    logs: list[str]
    metrics: dict[str, str]
    recent_deploys: list[DeployEvent]
    slack_context: str


class GroundTruth(BaseModel):
    root_cause: str
    contributing_factors: list[str]
    correct_action: str


class Scenario(BaseModel):
    id: str
    category: Literal["kubernetes", "database", "deploy_regression", "cascade"]
    difficulty: Literal["easy", "hard"]
    context: ScenarioContext
    ground_truth: GroundTruth


class RCAOutput(BaseModel):
    root_cause: str
    contributing_factors: list[str]
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trace: str


FailureMode = Literal[
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
]


class ScoreResult(BaseModel):
    scenario_id: str
    agent_name: str
    rca_score: float = Field(ge=0.0, le=1.0)
    calibration_score: float = Field(ge=0.0, le=1.0)
    actionability_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    failure_mode: FailureMode
    judge_reasoning: str
    rca_output: RCAOutput


class AggregatedResult(BaseModel):
    agent_name: str
    category: str
    difficulty: str
    mean_final_score: float
    mean_rca_score: float
    dominant_failure_mode: str
    n_scenarios: int
