from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from jinja2 import Template

from sre_bench.evaluators.aggregator import aggregate
from sre_bench.schema import Scenario, ScoreResult


REPORT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SRE-Bench: Agent Leaderboard</title>
  <style>
    :root {
      --bg: #f7f9fc;
      --panel: #ffffff;
      --ink: #15233a;
      --muted: #546179;
      --line: #dbe4f0;
      --good: #1c7c54;
      --mid: #d08700;
      --bad: #c73e1d;
      --accent: #1f6feb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 45%);
    }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1, h2 { margin: 0 0 12px; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.25rem; margin-top: 28px; }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .meta-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }
    .meta-key { color: var(--muted); font-size: 0.85rem; }
    .meta-val { font-weight: 600; margin-top: 4px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-top: 12px;
      overflow-x: auto;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      min-width: 760px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 10px;
      vertical-align: top;
      font-size: 0.95rem;
    }
    th { color: var(--muted); font-size: 0.82rem; letter-spacing: 0.02em; text-transform: uppercase; }
    .score-pill {
      display: inline-block;
      min-width: 52px;
      text-align: center;
      border-radius: 999px;
      padding: 4px 8px;
      color: white;
      font-weight: 700;
      font-size: 0.82rem;
    }
    .good { background: var(--good); }
    .mid { background: var(--mid); }
    .bad { background: var(--bad); }
    .heat-cell {
      border-radius: 8px;
      color: white;
      text-align: center;
      font-weight: 700;
      padding: 8px;
    }
    .bars { display: grid; gap: 8px; }
    .bar-row { display: grid; grid-template-columns: 220px 1fr auto; gap: 10px; align-items: center; }
    .bar-track { background: #e9eef5; border-radius: 999px; height: 14px; overflow: hidden; }
    .bar-fill { background: linear-gradient(90deg, #69b3ff, #1f6feb); height: 100%; }
    ul.findings { margin: 8px 0 0 20px; }
    details summary { cursor: pointer; color: var(--accent); }
    .small { color: var(--muted); font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SRE-Bench: Agent Leaderboard</h1>
    <div class="small">Benchmarking RCA quality for agentic incident diagnosis</div>

    <div class="meta">
      <div class="meta-card"><div class="meta-key">Run timestamp</div><div class="meta-val">{{ run_timestamp }}</div></div>
      <div class="meta-card"><div class="meta-key">Scenarios evaluated</div><div class="meta-val">{{ n_scenarios }}</div></div>
      <div class="meta-card"><div class="meta-key">Evaluations</div><div class="meta-val">{{ n_results }}</div></div>
      <div class="meta-card"><div class="meta-key">Judge model</div><div class="meta-val">{{ judge_model }}</div></div>
    </div>

    <h2>1) Leaderboard</h2>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Agent</th>
            <th>Overall</th>
            <th>RCA</th>
            <th>Calibration</th>
            <th>Actionability</th>
            <th>Top Failure Mode</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {% for row in leaderboard %}
          <tr>
            <td>{{ row.agent_name }}</td>
            <td><span class="score-pill {{ row.final_class }}">{{ row.mean_final_score }}</span></td>
            <td>{{ row.mean_rca_score }}</td>
            <td>{{ row.mean_calibration_score }}</td>
            <td>{{ row.mean_actionability_score }}</td>
            <td>{{ row.top_failure_mode }}</td>
            <td>{{ row.n_scenarios }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <h2>2) Score Heatmap (Agent x Category)</h2>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Agent</th>
            {% for c in categories %}<th>{{ c }}</th>{% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for row in heatmap_rows %}
          <tr>
            <td>{{ row.agent_name }}</td>
            {% for cell in row.cells %}
              <td>{% if cell.value is not none %}<div class="heat-cell" style="background: {{ cell.color }}">{{ cell.value }}</div>{% else %}-{% endif %}</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <h2>3) Failure Mode Breakdown</h2>
    <div class="panel">
      {% for agent in failure_mode_rows %}
        <h3>{{ agent.agent_name }}</h3>
        <div class="bars">
          {% for mode in agent.modes %}
          <div class="bar-row">
            <div>{{ mode.name }}</div>
            <div class="bar-track"><div class="bar-fill" style="width: {{ mode.rate_pct }}%"></div></div>
            <div>{{ mode.count }} ({{ mode.rate_pct }}%)</div>
          </div>
          {% endfor %}
        </div>
      {% endfor %}
    </div>

    <h2>4) Key Findings</h2>
    <div class="panel">
      <ul class="findings">
        {% for finding in key_findings %}<li>{{ finding }}</li>{% endfor %}
      </ul>
    </div>

    <h2>5) Scenario Detail Table</h2>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Category</th>
            <th>Difficulty</th>
            <th>Agent</th>
            <th>Final</th>
            <th>Failure Mode</th>
            <th>Judge Reasoning</th>
          </tr>
        </thead>
        <tbody>
          {% for row in scenario_rows %}
          <tr>
            <td>{{ row.scenario_id }}</td>
            <td>{{ row.category }}</td>
            <td>{{ row.difficulty }}</td>
            <td>{{ row.agent_name }}</td>
            <td>{{ row.final_score }}</td>
            <td>{{ row.failure_mode }}</td>
            <td>
              <details>
                <summary>View reasoning</summary>
                <div>{{ row.judge_reasoning }}</div>
              </details>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
""".strip()


def _score_class(score: float) -> str:
    if score < 0.4:
        return "bad"
    if score < 0.7:
        return "mid"
    return "good"


def _heat_color(score: float | None) -> str:
    if score is None:
        return "#9aa5b1"
    if score < 0.4:
        return "#c73e1d"
    if score < 0.7:
        return "#d08700"
    return "#1c7c54"


def _load_scenarios_map(path: Path = Path("scenarios")) -> dict[str, Scenario]:
    if not path.exists():
        return {}
    out: dict[str, Scenario] = {}
    for scenario_file in sorted(path.glob("*.json")):
        payload = json.loads(scenario_file.read_text(encoding="utf-8"))
        scenario = Scenario.model_validate(payload)
        out[scenario.id] = scenario
    return out


def _build_key_findings(agg: dict[str, Any], scenario_rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    leaderboard = agg.get("leaderboard", [])
    if leaderboard:
        best = leaderboard[0]
        findings.append(
            f"{best['agent_name']} leads overall with mean final score {best['mean_final_score']}."
        )

    heatmap = agg.get("heatmap", {})
    best_pair: tuple[str, str, float] | None = None
    worst_pair: tuple[str, str, float] | None = None
    for agent_name, cat_map in heatmap.items():
        for category, score in cat_map.items():
            if best_pair is None or score > best_pair[2]:
                best_pair = (agent_name, category, score)
            if worst_pair is None or score < worst_pair[2]:
                worst_pair = (agent_name, category, score)
    if best_pair:
        findings.append(
            f"Best category performance: {best_pair[0]} on {best_pair[1]} ({best_pair[2]:.4f})."
        )
    if worst_pair:
        findings.append(
            f"Weakest category performance: {worst_pair[0]} on {worst_pair[1]} ({worst_pair[2]:.4f})."
        )

    global_data = agg.get("global", {})
    common_mode = global_data.get("most_common_failure_mode")
    if common_mode:
        findings.append(f"Most common failure mode overall: {common_mode}.")

    if scenario_rows:
        hard_scores = [r["final_score"] for r in scenario_rows if r["difficulty"] == "hard"]
        easy_scores = [r["final_score"] for r in scenario_rows if r["difficulty"] == "easy"]
        if hard_scores and easy_scores:
            findings.append(
                "Difficulty gap: "
                f"easy mean {mean(easy_scores):.4f} vs hard mean {mean(hard_scores):.4f}."
            )

    return findings[:5]


def _to_report_context(run_data: dict[str, Any]) -> dict[str, Any]:
    meta = run_data.get("meta", {})
    results_payload = run_data.get("results")
    if not results_payload:
        raise ValueError("Input file has no judged results. Run without --no-judge for report.")

    results = [ScoreResult.model_validate(item) for item in results_payload]
    scenarios_map = _load_scenarios_map()
    agg = meta.get("aggregate") or aggregate(results, scenarios_by_id=scenarios_map)

    leaderboard = []
    for row in agg.get("leaderboard", []):
        leaderboard.append(
            {
                **row,
                "final_class": _score_class(float(row["mean_final_score"])),
            }
        )

    categories = sorted({k for m in agg.get("heatmap", {}).values() for k in m.keys()})
    heatmap_rows = []
    for agent_name in sorted(agg.get("heatmap", {}).keys()):
        cat_scores = agg["heatmap"][agent_name]
        cells = []
        for category in categories:
            value = cat_scores.get(category)
            cells.append(
                {
                    "value": None if value is None else f"{value:.4f}",
                    "color": _heat_color(value),
                }
            )
        heatmap_rows.append({"agent_name": agent_name, "cells": cells})

    failure_mode_rows = []
    failure_modes = agg.get("failure_modes", {})
    for agent_name in sorted(failure_modes.keys()):
        counts = failure_modes[agent_name].get("counts", {})
        modes = []
        for mode, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            total = sum(counts.values()) or 1
            rate_pct = round(count * 100 / total, 1)
            modes.append({"name": mode, "count": count, "rate_pct": rate_pct})
        failure_mode_rows.append({"agent_name": agent_name, "modes": modes})

    scenario_rows = []
    for result in results:
        scenario = scenarios_map.get(result.scenario_id)
        category = scenario.category if scenario else "unknown"
        difficulty = scenario.difficulty if scenario else "unknown"
        scenario_rows.append(
            {
                "scenario_id": result.scenario_id,
                "category": category,
                "difficulty": difficulty,
                "agent_name": result.agent_name,
                "final_score": f"{result.final_score:.4f}",
                "failure_mode": result.failure_mode,
                "judge_reasoning": result.judge_reasoning,
            }
        )
    scenario_rows.sort(key=lambda row: (row["scenario_id"], row["agent_name"]))

    key_findings = _build_key_findings(agg=agg, scenario_rows=[
        {
            "difficulty": row["difficulty"],
            "final_score": float(row["final_score"]),
        }
        for row in scenario_rows
    ])

    return {
        "run_timestamp": meta.get("timestamp", "unknown"),
        "judge_model": meta.get("judge_model", "not recorded"),
        "n_scenarios": len({r.scenario_id for r in results}),
        "n_results": len(results),
        "leaderboard": leaderboard,
        "categories": categories,
        "heatmap_rows": heatmap_rows,
        "failure_mode_rows": failure_mode_rows,
        "key_findings": key_findings,
        "scenario_rows": scenario_rows,
    }


def generate_report(input_path: Path, output_path: Path) -> Path:
    run_data = json.loads(input_path.read_text(encoding="utf-8"))
    context = _to_report_context(run_data)
    html = Template(REPORT_TEMPLATE).render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
