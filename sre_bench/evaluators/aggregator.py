from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Mapping

from sre_bench.schema import AggregatedResult, Scenario, ScoreResult


def aggregate(
    results: list[ScoreResult],
    scenarios_by_id: Mapping[str, Scenario] | None = None,
) -> dict[str, Any]:
    """
    Groups results by (agent_name, category, difficulty) and returns report-friendly data.
    """
    if not results:
        return {
            "leaderboard": [],
            "breakdown": {},
            "heatmap": {},
            "failure_modes": {},
            "worst_scenarios": {},
            "best_scenarios": {},
            "aggregated_rows": [],
            "global": {
                "n_results": 0,
                "mean_final_score": 0.0,
                "most_common_failure_mode": "other",
            },
        }

    by_agent: dict[str, list[ScoreResult]] = defaultdict(list)
    by_agent_cat_diff: dict[tuple[str, str, str], list[ScoreResult]] = defaultdict(list)
    by_agent_cat: dict[tuple[str, str], list[ScoreResult]] = defaultdict(list)

    for result in results:
        category, difficulty = _resolve_meta(result.scenario_id, scenarios_by_id)
        by_agent[result.agent_name].append(result)
        by_agent_cat_diff[(result.agent_name, category, difficulty)].append(result)
        by_agent_cat[(result.agent_name, category)].append(result)

    leaderboard = _build_leaderboard(by_agent)
    failure_modes = _build_failure_mode_summary(by_agent)
    worst_scenarios, best_scenarios = _build_best_worst(by_agent)
    breakdown, aggregated_rows = _build_breakdown(by_agent_cat_diff)
    heatmap = _build_heatmap(by_agent_cat)

    global_failure_counts = Counter(r.failure_mode for r in results)
    global_common_mode = global_failure_counts.most_common(1)[0][0]

    return {
        "leaderboard": leaderboard,
        "breakdown": breakdown,
        "heatmap": heatmap,
        "failure_modes": failure_modes,
        "worst_scenarios": worst_scenarios,
        "best_scenarios": best_scenarios,
        "aggregated_rows": [row.model_dump() for row in aggregated_rows],
        "global": {
            "n_results": len(results),
            "mean_final_score": round(mean(r.final_score for r in results), 4),
            "most_common_failure_mode": global_common_mode,
        },
    }


def _build_leaderboard(by_agent: dict[str, list[ScoreResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_name, items in by_agent.items():
        fail_counts = Counter(item.failure_mode for item in items)
        top_failure = fail_counts.most_common(1)[0][0] if fail_counts else "other"
        rows.append(
            {
                "agent_name": agent_name,
                "mean_final_score": round(mean(item.final_score for item in items), 4),
                "mean_rca_score": round(mean(item.rca_score for item in items), 4),
                "mean_calibration_score": round(
                    mean(item.calibration_score for item in items), 4
                ),
                "mean_actionability_score": round(
                    mean(item.actionability_score for item in items), 4
                ),
                "top_failure_mode": top_failure,
                "n_scenarios": len(items),
            }
        )
    return sorted(rows, key=lambda row: row["mean_final_score"], reverse=True)


def _build_failure_mode_summary(
    by_agent: dict[str, list[ScoreResult]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for agent_name, items in by_agent.items():
        counts = Counter(item.failure_mode for item in items)
        total = len(items)
        dominant = counts.most_common(1)[0][0] if counts else "other"
        summary[agent_name] = {
            "counts": dict(counts),
            "rates": {
                mode: round(count / total, 4) for mode, count in counts.items()
            },
            "dominant_failure_mode": dominant,
        }
    return summary


def _build_best_worst(
    by_agent: dict[str, list[ScoreResult]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    worst: dict[str, list[dict[str, Any]]] = {}
    best: dict[str, list[dict[str, Any]]] = {}
    for agent_name, items in by_agent.items():
        ordered = sorted(items, key=lambda item: item.final_score)
        worst[agent_name] = [
            {
                "scenario_id": item.scenario_id,
                "final_score": item.final_score,
                "failure_mode": item.failure_mode,
            }
            for item in ordered[:2]
        ]
        best[agent_name] = [
            {
                "scenario_id": item.scenario_id,
                "final_score": item.final_score,
                "failure_mode": item.failure_mode,
            }
            for item in reversed(ordered[-2:])
        ]
    return worst, best


def _build_breakdown(
    by_agent_cat_diff: dict[tuple[str, str, str], list[ScoreResult]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], list[AggregatedResult]]:
    breakdown: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    rows: list[AggregatedResult] = []

    for (agent_name, category, difficulty), items in by_agent_cat_diff.items():
        fail_counts = Counter(item.failure_mode for item in items)
        dominant = fail_counts.most_common(1)[0][0] if fail_counts else "other"

        row = AggregatedResult(
            agent_name=agent_name,
            category=category,
            difficulty=difficulty,
            mean_final_score=round(mean(item.final_score for item in items), 4),
            mean_rca_score=round(mean(item.rca_score for item in items), 4),
            dominant_failure_mode=dominant,
            n_scenarios=len(items),
        )
        rows.append(row)
        breakdown[agent_name][category][difficulty] = row.model_dump()

    return breakdown, sorted(
        rows,
        key=lambda row: (row.agent_name, row.category, row.difficulty),
    )


def _build_heatmap(
    by_agent_cat: dict[tuple[str, str], list[ScoreResult]],
) -> dict[str, dict[str, float]]:
    heatmap: dict[str, dict[str, float]] = defaultdict(dict)
    for (agent_name, category), items in by_agent_cat.items():
        heatmap[agent_name][category] = round(mean(item.final_score for item in items), 4)
    return dict(heatmap)


def _resolve_meta(
    scenario_id: str,
    scenarios_by_id: Mapping[str, Scenario] | None,
) -> tuple[str, str]:
    if scenarios_by_id and scenario_id in scenarios_by_id:
        scenario = scenarios_by_id[scenario_id]
        return scenario.category, scenario.difficulty

    difficulty = "hard" if scenario_id.endswith("_hard") else "easy"
    if scenario_id.startswith("k8s_"):
        return "kubernetes", difficulty
    if scenario_id.startswith("db_"):
        return "database", difficulty
    if scenario_id.startswith("deploy_"):
        return "deploy_regression", difficulty
    if scenario_id.startswith("cascade_"):
        return "cascade", difficulty
    return "unknown", difficulty
