from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from sre_bench.evaluators.aggregator import aggregate
from sre_bench.evaluators.judge import LLMJudge
from sre_bench.schema import RCAOutput, Scenario, ScoreResult

app = typer.Typer(help="SRE-Bench CLI")
console = Console()


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"run_{stamp}.json"


def load_scenarios(path: Path = Path("scenarios")) -> list[Scenario]:
    scenario_files = sorted(path.glob("*.json"))
    scenarios: list[Scenario] = []
    for file_path in scenario_files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        scenarios.append(Scenario.model_validate(payload))
    return scenarios


def select_scenarios(all_scenarios: list[Scenario], selector: str) -> list[Scenario]:
    if selector.strip().lower() == "all":
        return all_scenarios
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    chosen = [scenario for scenario in all_scenarios if scenario.id in requested]
    missing = sorted(requested - {scenario.id for scenario in chosen})
    if missing:
        raise ValueError(f"Unknown scenarios: {', '.join(missing)}")
    return chosen


def build_agent_registry() -> dict[str, Any]:
    from sre_bench.agents import MultiAgentCrewAgent, ReActAgent, SimpleLLMAgent

    return {
        "simple_llm": SimpleLLMAgent,
        "react_langchain": ReActAgent,
        "multi_agent_crewai": MultiAgentCrewAgent,
    }


def instantiate_agents(selector: str) -> list[Any]:
    registry = build_agent_registry()
    if selector.strip().lower() == "all":
        names = list(registry.keys())
    else:
        names = [item.strip() for item in selector.split(",") if item.strip()]

    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown agents: {', '.join(unknown)}")

    return [registry[name]() for name in names]


def score_threshold_color(score: float) -> str:
    if score < 0.4:
        return "red"
    if score < 0.7:
        return "yellow"
    return "green"


def save_trace_files(
    output_file: Path,
    scenario: Scenario,
    agent_name: str,
    rca_output: RCAOutput,
) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stamp}_{scenario.id}_{agent_name}"
    trace_root = output_file.parent / "traces"
    full_path = trace_root / "full" / f"{base_name}.txt"
    summary_path = trace_root / "summary" / f"{base_name}.json"

    full_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    full_text = (
        f"timestamp: {stamp}\n"
        f"scenario_id: {scenario.id}\n"
        f"agent_name: {agent_name}\n"
        f"root_cause: {rca_output.root_cause}\n"
        f"recommended_action: {rca_output.recommended_action}\n"
        f"confidence: {rca_output.confidence}\n\n"
        f"reasoning_trace:\n{rca_output.reasoning_trace}\n"
    )
    full_path.write_text(full_text, encoding="utf-8")

    summary_payload = {
        "timestamp": stamp,
        "scenario_id": scenario.id,
        "agent_name": agent_name,
        "root_cause": rca_output.root_cause,
        "contributing_factors": rca_output.contributing_factors,
        "recommended_action": rca_output.recommended_action,
        "confidence": rca_output.confidence,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def run_evaluation(
    agents: list[Any],
    scenarios: list[Scenario],
    output_path: Path,
    dry_run: bool = False,
    no_judge: bool = False,
    judge: LLMJudge | None = None,
) -> tuple[list[ScoreResult], list[dict[str, Any]]]:
    results: list[ScoreResult] = []
    raw_outputs: list[dict[str, Any]] = []
    score_judge = judge or LLMJudge()

    for scenario in scenarios:
        for agent in agents:
            if dry_run:
                console.rule(f"Dry Run: {scenario.id} x {agent.name}")
                console.print(f"[bold]Alert:[/bold] {scenario.context.alert_message}")

            rca_output = agent._safe_diagnose(scenario)
            save_trace_files(output_path, scenario, agent.name, rca_output)

            if no_judge:
                raw_outputs.append(
                    {
                        "scenario_id": scenario.id,
                        "agent_name": agent.name,
                        "rca_output": rca_output.model_dump(),
                    }
                )
                if dry_run:
                    console.print_json(data=raw_outputs[-1])
                    return [], raw_outputs
                continue

            score = score_judge.score(
                scenario=scenario,
                rca_output=rca_output,
                agent_name=agent.name,
            )
            results.append(score)

            color = score_threshold_color(score.final_score)
            console.print(
                f"[{color}][{scenario.id}] [{agent.name}] -> score: "
                f"{score.final_score:.2f} (failure: {score.failure_mode})[/{color}]"
            )

            if dry_run:
                console.print_json(data=score.model_dump())
                return results, []

    return results, raw_outputs


def save_results(
    output_path: Path,
    results: list[ScoreResult],
    raw_outputs: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"meta": run_meta}
    if results:
        payload["results"] = [item.model_dump() for item in results]
    if raw_outputs:
        payload["raw_outputs"] = raw_outputs
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary_table(results: list[ScoreResult], scenarios: list[Scenario]) -> None:
    if not results:
        console.print("[yellow]No judged scores to summarize.[/yellow]")
        return

    scenario_map = {scenario.id: scenario for scenario in scenarios}
    grouped: dict[tuple[str, str], list[float]] = {}
    for result in results:
        category = scenario_map[result.scenario_id].category
        key = (category, result.agent_name)
        grouped.setdefault(key, []).append(result.final_score)

    categories = sorted({scenario.category for scenario in scenarios})
    agents = sorted({result.agent_name for result in results})

    table = Table(title="Score Summary by Category")
    table.add_column("Category")
    for agent in agents:
        table.add_column(agent)

    for category in categories:
        row = [category]
        for agent in agents:
            values = grouped.get((category, agent), [])
            if not values:
                row.append("-")
                continue
            avg = sum(values) / len(values)
            color = score_threshold_color(avg)
            row.append(f"[{color}]{avg:.2f}[/{color}]")
        table.add_row(*row)

    console.print(table)


@app.command("run")
def run_command(
    agents: str = typer.Option("all", help="Comma-separated agent names or 'all'."),
    scenarios: str = typer.Option("all", help="Comma-separated scenario IDs or 'all'."),
    output: Path = typer.Option(default_factory=default_output_path),
    dry_run: bool = typer.Option(False, help="Run one scenario x one agent and print details."),
    no_judge: bool = typer.Option(False, help="Skip LLM judge and save raw RCA outputs only."),
) -> None:
    load_dotenv()

    all_scenarios = load_scenarios()
    selected_scenarios = select_scenarios(all_scenarios, scenarios)
    selected_agents = instantiate_agents(agents)

    scored_results, raw_outputs = run_evaluation(
        agents=selected_agents,
        scenarios=selected_scenarios,
        output_path=output,
        dry_run=dry_run,
        no_judge=no_judge,
    )

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agents": [agent.name for agent in selected_agents],
        "scenarios": [scenario.id for scenario in selected_scenarios],
        "dry_run": dry_run,
        "no_judge": no_judge,
    }
    if scored_results:
        meta["aggregate"] = aggregate(
            scored_results,
            scenarios_by_id={scenario.id: scenario for scenario in selected_scenarios},
        )

    save_results(output, scored_results, raw_outputs, run_meta=meta)

    if scored_results:
        print_summary_table(scored_results, selected_scenarios)
    console.print(f"[green]Saved run output to {output}[/green]")


@app.command("report")
def report_command(input: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    try:
        from sre_bench.report import generate_report
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "Report generator not implemented yet. Add sre_bench/report.py first."
        ) from exc

    generate_report(input_path=input, output_path=output)
    console.print(f"[green]Saved HTML report to {output}[/green]")


if __name__ == "__main__":
    app()
