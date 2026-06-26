# AGENT.md — SRE-Bench: Agentic Incident Diagnosis Eval Harness

> Complete technical specification for an autonomous coding agent.
> Read this file fully before writing a single line of code.
> Every section is load-bearing. Do not skip or reorder steps.

---

## 0. Project identity

**Name:** SRE-Bench
**Tagline:** An open-source eval harness for agentic incident diagnosis
**Purpose:** Benchmark how well LLM agents diagnose root causes in synthetic production incidents. Compare agent architectures, surface failure modes, generate a leaderboard report.
**Audience:** The Sherlocks.ai team (pitch project), and the open-source SRE/AI community.
**Repository:** Public GitHub repo. Must be forkable and runnable by anyone with API keys.

---

## 1. Directory structure

Create this exact layout. Do not add files outside this structure without noting it here first.

```
sre-bench/
├── AGENT.md                  ← this file
├── README.md                 ← written last
├── pyproject.toml            ← dependencies + project metadata
├── .env.example              ← required env vars, no values
├── .gitignore
│
├── sre_bench/
│   ├── __init__.py
│   ├── schema.py             ← Pydantic models (ScenarioContext, RCAOutput, ScoreResult)
│   ├── runner.py             ← orchestrates scenario × agent matrix
│   ├── report.py             ← renders HTML leaderboard from results JSON
│   └── agents/
│       ├── __init__.py
│       ├── base.py           ← BaseAgent ABC
│       ├── simple_llm.py     ← Agent 1: single prompt
│       ├── react_agent.py    ← Agent 2: LangChain ReAct + mock tools
│       └── multi_agent.py    ← Agent 3: CrewAI specialist crew
│   └── evaluators/
│       ├── __init__.py
│       ├── judge.py          ← LLM-as-judge scorer
│       └── aggregator.py     ← groups scores → leaderboard data
│
├── scenarios/
│   ├── k8s_oom_easy.json
│   ├── k8s_crash_hard.json
│   ├── db_slow_easy.json
│   ├── db_pool_hard.json
│   ├── deploy_regression_easy.json
│   ├── deploy_timeout_hard.json
│   ├── cascade_auth_easy.json
│   └── cascade_dns_hard.json
│
├── results/
│   ├── .gitkeep
│   └── sample_run.html       ← pre-generated, committed to repo
│
└── tests/
    ├── test_schema.py
    ├── test_judge.py
    └── test_runner.py
```

---

## 2. Environment and dependencies

### 2.1 Python version
Use Python 3.11+. Do not use 3.12+ features.

### 2.2 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sre-bench"
version = "0.1.0"
description = "Eval harness for agentic SRE incident diagnosis"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.6",
  "langchain>=0.2",
  "langchain-openai>=0.1",
  "langchain-anthropic>=0.1",
  "crewai>=0.55",
  "openai>=1.30",
  "anthropic>=0.28",
  "jinja2>=3.1",
  "rich>=13.7",
  "python-dotenv>=1.0",
  "typer>=0.12",
]

[project.scripts]
sre-bench = "sre_bench.runner:app"
```

Install with: `pip install -e .`

### 2.3 `.env.example`

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
JUDGE_MODEL=gpt-4o          # model used for LLM judge
AGENT_MODEL=gpt-4o-mini     # model used for agents (cost control)
```

### 2.4 `.gitignore`

```
.env
results/*.json
results/*.html
!results/sample_run.html
__pycache__/
*.pyc
.venv/
dist/
```

---

## 3. Data models — `sre_bench/schema.py`

This is the single source of truth for all data shapes. Build this file first. Everything else imports from it.

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

# ── Scenario input ──────────────────────────────────────────────

class DeployEvent(BaseModel):
    time_offset: str          # e.g. "-45m", "-2h"
    service: str
    change_summary: str

class ScenarioContext(BaseModel):
    alert_message: str
    logs: list[str]           # 5-15 log lines, realistic format
    metrics: dict[str, str]   # {"memory_usage": "98%", "error_rate": "12%"}
    recent_deploys: list[DeployEvent]
    slack_context: str        # 1-2 sentences of team chatter

class GroundTruth(BaseModel):
    root_cause: str           # 1-2 sentence definitive answer
    contributing_factors: list[str]
    correct_action: str       # what a senior SRE would do

class Scenario(BaseModel):
    id: str
    category: Literal["kubernetes", "database", "deploy_regression", "cascade"]
    difficulty: Literal["easy", "hard"]
    context: ScenarioContext
    ground_truth: GroundTruth

# ── Agent output ─────────────────────────────────────────────────

class RCAOutput(BaseModel):
    root_cause: str
    contributing_factors: list[str]
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trace: str      # agent's internal chain of thought

# ── Evaluation output ────────────────────────────────────────────

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
    "other"
]

class ScoreResult(BaseModel):
    scenario_id: str
    agent_name: str
    rca_score: float = Field(ge=0.0, le=1.0)       # root cause correctness
    calibration_score: float = Field(ge=0.0, le=1.0)
    actionability_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)      # weighted average
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
```

---

## 4. Scenario files — `scenarios/*.json`

Write all 8 scenarios as JSON. They must validate against the `Scenario` Pydantic model. Each scenario must be independently diagnosable by a human reading only the `context` block — the answer must be discoverable from the signals provided, not require external knowledge.

### Rules for writing good scenarios
- Logs must look like real application logs: include timestamps, log levels, service names, stack fragments
- Metrics must include at least 3 key-value pairs relevant to the incident
- Every scenario must have exactly one "smoking gun" signal that confirms the root cause
- Hard scenarios must have at least 2 misleading signals that could lead an agent astray
- `recent_deploys` must include at least one deploy within 2 hours of the incident for deploy-category scenarios

### Scenario specs

**1. `k8s_oom_easy.json`**
- Alert: payments-svc pods OOMKilled (3 restarts in 10 min)
- Smoking gun: memory_usage=98%, a deploy 45min ago added image processing
- Misleading signal: none (easy)
- Root cause: new image processing feature introduced memory regression, insufficient pod memory limit
- Action: roll back deploy or increase `resources.limits.memory` and redeploy

**2. `k8s_crash_hard.json`**
- Alert: CrashLoopBackOff on 3 services simultaneously (api-gateway, order-svc, inventory-svc)
- Smoking gun: all 3 mount the same ConfigMap that was updated 20min ago with a malformed DB connection string
- Misleading signals: high CPU on one node (unrelated HPA scale event), a deploy to a 4th service (unrelated)
- Root cause: ConfigMap `app-config-v2` contains malformed `DATABASE_URL` — missing port number
- Action: patch ConfigMap with correct connection string, restart affected pods

**3. `db_slow_easy.json`**
- Alert: p99 API latency 4200ms (baseline 180ms), DB query time alert firing
- Smoking gun: logs show full table scan on `orders` table, deploy 90min ago added `/api/v2/orders/search` endpoint with no index
- Root cause: new search endpoint queries `orders` table with `WHERE status = ?` — no index on `status` column
- Action: `CREATE INDEX CONCURRENTLY idx_orders_status ON orders(status);`

**4. `db_pool_hard.json`**
- Alert: connection pool exhausted on primary DB, error rate 23%
- Smoking gun: analytics job started 35min ago consuming 40/50 connections, new service version leaks 1 connection per request
- Misleading signals: traffic is 15% above average (normal Friday), a Redis eviction spike (unrelated)
- Root cause: combination of long-running analytics query + connection leak in reporting-svc v2.1.0 exhausted the pool
- Action: kill analytics job, roll back reporting-svc, increase pool size as short-term mitigation

**5. `deploy_regression_easy.json`**
- Alert: NullPointerException in checkout-svc, error rate 8%
- Smoking gun: deploy 15min ago enabled feature flag `new_checkout_flow` — the new code path has an unguarded null access on `user.preferences`
- Root cause: `new_checkout_flow` feature flag silently enabled for all users, code assumes `user.preferences` is never null but new user accounts don't have this field
- Action: disable feature flag `new_checkout_flow`, fix null guard, re-enable

**6. `deploy_timeout_hard.json`**
- Alert: p95 latency increased from 220ms to 1800ms, no errors, no OOM
- Smoking gun: `http-client` SDK updated from 2.1.0 to 2.2.0 in the same deploy — default read timeout changed from 5s to 30s, masking slow upstream calls
- Misleading signals: memory is fine, error rate is 0%, traffic is normal, no DB alerts
- Root cause: SDK version bump silently changed timeout defaults, hiding slow responses from notification-service that were previously timing out fast
- Action: pin `http-client` to 2.1.0 or explicitly set `read_timeout=5` in config

**7. `cascade_auth_easy.json`**
- Alert: payment-svc returning 503s
- Smoking gun: payment-svc depends on auth-svc which depends on Redis cache; Redis was flushed 10min ago causing cache stampede on auth-svc, which cascaded
- Root cause: Redis cache manual flush caused auth-svc cache miss storm → auth-svc response time 4s → payment-svc timeout → 503s
- Action: restore Redis from backup or warm cache, add circuit breaker on payment-svc → auth-svc calls

**8. `cascade_dns_hard.json`**
- Alert: 5 services degraded simultaneously — user-svc, order-svc, notification-svc, payment-svc, inventory-svc
- Smoking gun: network infra change 25min ago rotated internal DNS servers; new DNS server has 8s TTL misconfiguration causing intermittent resolution failures
- Misleading signals: each service shows different error patterns (timeouts, connection refused, 503s) making it look like 5 independent incidents; a code deploy to user-svc 2h ago (unrelated)
- Root cause: DNS TTL misconfiguration in internal DNS server rotation — all services affected because all internal service discovery uses DNS
- Action: revert DNS server to previous config or fix TTL, services will recover automatically

---

## 5. Base agent — `sre_bench/agents/base.py`

```python
from abc import ABC, abstractmethod
from sre_bench.schema import Scenario, RCAOutput

class BaseAgent(ABC):
    """
    All agents implement this interface.
    The runner calls diagnose() and expects RCAOutput back.
    Internal complexity (tool calls, multi-agent, chain-of-thought) is the agent's own business.
    """
    name: str  # set as class attribute on each subclass

    @abstractmethod
    def diagnose(self, scenario: Scenario) -> RCAOutput:
        """
        Given a full Scenario, return a structured RCAOutput.
        Must not raise — catch all exceptions and return a low-confidence output with error details in reasoning_trace.
        """
        ...

    def _safe_diagnose(self, scenario: Scenario) -> RCAOutput:
        """Wrap diagnose() with error handling for the runner."""
        try:
            return self.diagnose(scenario)
        except Exception as e:
            return RCAOutput(
                root_cause="Agent failed to produce output",
                contributing_factors=[],
                recommended_action="N/A",
                confidence=0.0,
                reasoning_trace=f"Exception: {type(e).__name__}: {str(e)}"
            )
```

---

## 6. Agent 1 — `sre_bench/agents/simple_llm.py`

Single LLM call. Dumps the entire scenario context into a system prompt and asks for structured JSON output. No tools, no loops.

**System prompt structure:**
```
You are a senior SRE diagnosing a production incident.
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
```

**User message:** Format the scenario context as a structured string:
```
ALERT: {alert_message}

LOGS (most recent first):
{newline-joined logs}

METRICS:
{key: value pairs}

RECENT DEPLOYS:
{formatted deploy list with time offsets}

SLACK CONTEXT:
{slack_context}
```

**Implementation notes:**
- Use `response_format={"type": "json_object"}` on the OpenAI call
- Parse response with `RCAOutput.model_validate_json(response.choices[0].message.content)`
- Model: read from `AGENT_MODEL` env var
- Temperature: 0.2 (want consistency, not creativity)
- Name: `"simple_llm"`

---

## 7. Agent 2 — `sre_bench/agents/react_agent.py`

LangChain ReAct agent with 3 mock tools backed by the scenario JSON. The agent iteratively calls tools to "investigate" before concluding.

### 7.1 Mock tools

All three tools are closures over the current scenario. They simulate real infra calls by filtering/returning scenario data.

```python
def make_tools(scenario: Scenario) -> list:
    """
    Returns 3 LangChain tools bound to the scenario data.
    """

    @tool
    def search_logs(query: str) -> str:
        """
        Search application logs for lines matching the query string.
        Returns matching log lines or 'No matching logs found'.
        Use this to find errors, stack traces, or specific service mentions.
        """
        matches = [l for l in scenario.context.logs if query.lower() in l.lower()]
        return "\n".join(matches) if matches else "No matching logs found."

    @tool
    def get_metrics(service_or_metric: str) -> str:
        """
        Get current metric values. Optionally filter by service or metric name.
        Returns key-value metric pairs relevant to the query.
        Use this to check CPU, memory, latency, error rates, connection counts.
        """
        results = {
            k: v for k, v in scenario.context.metrics.items()
            if service_or_metric.lower() in k.lower() or service_or_metric == "*"
        }
        if not results:
            return f"No metrics found matching '{service_or_metric}'. Available: {list(scenario.context.metrics.keys())}"
        return "\n".join(f"{k}: {v}" for k, v in results.items())

    @tool
    def get_deploy_history() -> str:
        """
        Get recent deployment history. Always call this — deploys are the most
        common root cause of incidents and must be correlated with the alert time.
        Returns a list of recent deployments with time offsets and change summaries.
        """
        if not scenario.context.recent_deploys:
            return "No recent deployments in the last 4 hours."
        lines = []
        for d in scenario.context.recent_deploys:
            lines.append(f"{d.time_offset}: [{d.service}] {d.change_summary}")
        return "\n".join(lines)

    return [search_logs, get_metrics, get_deploy_history]
```

### 7.2 Agent construction

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

REACT_SYSTEM_PROMPT = """You are a senior SRE investigating a production incident.

You have access to these tools:
{tools}

Use them to gather evidence before concluding. Always call get_deploy_history first.
Then search logs for errors. Then check metrics.

When you have enough evidence, think step by step and form a conclusion.

ALERT: {alert}
SLACK CONTEXT: {slack}

Use this format:
Thought: what do I need to investigate?
Action: tool name
Action Input: query string
Observation: tool result
... (repeat until confident)
Thought: I now have enough evidence to conclude
Final Answer: JSON with keys: root_cause, contributing_factors, recommended_action, confidence, reasoning_trace

{agent_scratchpad}
"""
```

**Implementation notes:**
- Max iterations: 8 (prevents runaway loops)
- Parse `Final Answer:` from agent output, extract JSON block
- Fall back to `_safe_diagnose` if JSON parse fails
- Name: `"react_langchain"`

---

## 8. Agent 3 — `sre_bench/agents/multi_agent.py`

CrewAI crew with 4 specialist agents. Each agent investigates one signal domain. A synthesis agent produces the final RCA.

### 8.1 Agents and their roles

```python
AGENTS = [
    {
        "role": "Log analyst",
        "goal": "Identify all errors, exceptions, and anomalies in the application logs",
        "backstory": "Expert at reading distributed system logs. Finds patterns, correlates timestamps, identifies the first error in a chain.",
        "task": "Analyze these logs: {logs}\nAlert context: {alert}\nReturn: key findings, timeline of errors, any stack traces."
    },
    {
        "role": "Metrics analyst",
        "goal": "Identify which metrics are anomalous and what they indicate about system health",
        "backstory": "SRE specialist in telemetry. Knows normal baselines and can identify anomalies by magnitude and timing.",
        "task": "Analyze these metrics: {metrics}\nAlert context: {alert}\nReturn: anomalous metrics, severity assessment, likely impacted components."
    },
    {
        "role": "Deploy analyst",
        "goal": "Correlate recent deployments with the incident timeline. Identify if a deploy is causal.",
        "backstory": "Deployment expert. Always checks what changed before an incident. Looks for timing correlation between deploys and symptom onset.",
        "task": "Recent deploys: {deploys}\nAlert context: {alert}\nSlack: {slack}\nReturn: most suspicious deploy, timing correlation, what changed that could cause this."
    },
    {
        "role": "SRE synthesizer",
        "goal": "Combine findings from log, metrics, and deploy analysts into a definitive root cause analysis",
        "backstory": "Principal SRE. Reads all evidence, weighs conflicting signals, produces a structured RCA with a specific remediation action.",
        "task": "Combine these findings:\nLOGS: {log_findings}\nMETRICS: {metrics_findings}\nDEPLOY: {deploy_findings}\n\nOutput JSON: root_cause, contributing_factors, recommended_action, confidence, reasoning_trace"
    }
]
```

**Implementation notes:**
- Use `Process.sequential` — log, metrics, deploy analysts run first, synthesizer last
- Pass outputs from first 3 tasks as context into the synthesizer task
- Temperature: 0.1 for all agents (want determinism)
- Parse synthesizer output for JSON block
- Name: `"multi_agent_crewai"`

---

## 9. LLM judge — `sre_bench/evaluators/judge.py`

The judge is called once per (agent_output, scenario) pair. It returns a `ScoreResult`.

### 9.1 Judge prompt

```python
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
  vague_action, other

Return JSON only, no markdown fences:
{
  "rca_score": float,
  "calibration_score": float,
  "actionability_score": float,
  "failure_mode": string,
  "judge_reasoning": string (2-3 sentences explaining scores)
}
"""

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
"""
```

### 9.2 Score computation

```python
def compute_final_score(rca: float, cal: float, act: float) -> float:
    return round(rca * 0.5 + cal * 0.25 + act * 0.25, 4)
```

**Implementation notes:**
- Always use `JUDGE_MODEL` env var for the judge (use the best available — gpt-4o or claude-opus)
- Temperature: 0.0 (deterministic scoring)
- Use `response_format={"type": "json_object"}`
- Validate output keys before constructing `ScoreResult`
- If judge call fails, return `ScoreResult` with all zeros and `failure_mode="other"` — do not crash the run

---

## 10. Runner — `sre_bench/runner.py`

Orchestrates the full evaluation matrix: every agent against every scenario.

### 10.1 CLI interface (Typer)

```
sre-bench run [OPTIONS]
  --agents TEXT      comma-separated agent names [default: all]
  --scenarios TEXT   comma-separated scenario IDs or "all" [default: all]
  --output PATH      results JSON output path [default: results/run_{timestamp}.json]
  --dry-run          run one scenario × one agent, print all intermediates, exit
  --no-judge         skip LLM judge, save raw RCAOutputs only (saves cost)

sre-bench report --input PATH --output PATH
  generate HTML report from a results JSON file
```

### 10.2 Run loop

```python
def run(agents, scenarios, output_path, dry_run=False):
    results: list[ScoreResult] = []

    for scenario in scenarios:
        for agent in agents:
            if dry_run:
                print_dry_run_header(scenario, agent)

            rca_output = agent._safe_diagnose(scenario)

            if dry_run:
                print_rca_output(rca_output)

            score = judge.score(scenario, rca_output, agent.name)

            if dry_run:
                print_score(score)
                return  # exit after first run in dry-run mode

            results.append(score)
            print_progress(scenario.id, agent.name, score.final_score)

    save_results(results, output_path)
    print_summary_table(results)
```

### 10.3 Rich console output

Use `rich` for all console output:
- Progress: `[scenario_id] [agent_name] → score: 0.74 (failure: correct)` in green/red
- Summary table at end: agents as columns, scenario categories as rows, mean scores in cells
- Use red/yellow/green color coding: <0.4 red, 0.4-0.7 yellow, >0.7 green

---

## 11. Aggregator — `sre_bench/evaluators/aggregator.py`

Reads a list of `ScoreResult` and produces leaderboard data.

```python
def aggregate(results: list[ScoreResult]) -> dict:
    """
    Groups by (agent_name, category, difficulty).
    Returns structure suitable for report.py to render.
    """
    # leaderboard: agent → mean final score across all scenarios
    # breakdown: agent × category × difficulty → mean scores
    # failure_modes: agent → {mode: count} — top failure mode per agent
    # worst_scenarios: per agent, the 2 scenarios with lowest score
```

Key outputs:
1. Overall leaderboard (agent ranked by mean final score)
2. Score heatmap data (agent × category)
3. Failure mode distribution per agent
4. Best/worst scenario per agent

---

## 12. Report generator — `sre_bench/report.py`

Generates a self-contained HTML file from aggregated results. No external dependencies — all CSS and JS inline.

### 12.1 Report sections (in order)

1. **Header** — "SRE-Bench: Agent Leaderboard" + run metadata (date, model, n_scenarios)
2. **Leaderboard table** — agents ranked by mean final score, columns: agent name, overall score, RCA score, calibration, actionability, top failure mode
3. **Score heatmap** — agent × scenario category grid, color-coded cells (red → green)
4. **Failure mode breakdown** — per agent, a bar showing distribution of failure mode labels
5. **Key findings** — 3-5 bullet points auto-generated from the data:
   - "X agent achieved the highest RCA score on database scenarios (0.82)"
   - "All agents struggled with cascade failures (mean 0.41)"
   - "Most common failure mode across all agents: {mode}"
6. **Scenario detail table** — expandable rows showing per-scenario scores and judge reasoning

### 12.2 Jinja2 template

Use a single `report.html.j2` template embedded as a string in `report.py`. Do not create a separate templates/ directory — keep it self-contained.

---

## 13. Tests — `tests/`

### `test_schema.py`
- Load each scenario JSON file and validate against `Scenario` model
- Assert all 8 scenarios load without error
- Assert all scenario IDs are unique

### `test_judge.py`
- Mock the LLM call
- Test: perfect RCA → score close to 1.0
- Test: completely wrong RCA → score close to 0.0
- Test: judge JSON parse failure → returns zero-score result without raising
- Test: `compute_final_score` weighting is correct (0.5/0.25/0.25)

### `test_runner.py`
- Instantiate all three agents
- Run `--dry-run` against one scenario using mocked LLM calls
- Assert output is a valid `ScoreResult`
- Assert no exceptions raised

Run with: `pytest tests/ -v`

---

## 14. Build order

Execute in this exact order. Do not proceed to the next step until the current one is complete and any stated tests pass.

```
Step 1:  pyproject.toml + .env.example + .gitignore
Step 2:  sre_bench/schema.py  →  run: python -c "from sre_bench.schema import Scenario"
Step 3:  All 8 scenario JSON files  →  run: test_schema.py (all 8 must validate)
Step 4:  sre_bench/agents/base.py
Step 5:  sre_bench/agents/simple_llm.py  →  smoke test: diagnose k8s_oom_easy manually
Step 6:  sre_bench/evaluators/judge.py  →  run: test_judge.py
Step 7:  sre_bench/agents/react_agent.py  →  smoke test: diagnose k8s_oom_easy manually
Step 8:  sre_bench/agents/multi_agent.py  →  smoke test: diagnose k8s_oom_easy manually
Step 9:  sre_bench/evaluators/aggregator.py
Step 10: sre_bench/runner.py  →  run: sre-bench run --dry-run
Step 11: sre_bench/report.py  →  run: sre-bench report
Step 12: Full run: sre-bench run --agents all --scenarios all
Step 13: Commit sample_run.html to results/
Step 14: README.md (written last, after findings are known)
```

---

## 15. Error handling rules

These apply everywhere. No exceptions.

- All agent `diagnose()` calls are wrapped in `_safe_diagnose()` — never let an agent crash the run
- All judge calls are try/caught — a failed judge returns zeros, not an exception
- All JSON parses from LLM outputs are try/caught — log the raw output and return a fallback
- All scenario file loads validate against Pydantic schema — reject invalid files loudly at startup
- Missing env vars: check at startup, print which vars are missing, exit with code 1
- Never silently swallow errors — log them with `rich` in yellow/red

---

## 16. Cost control

This project makes many LLM API calls. Keep costs under control:

- `AGENT_MODEL` defaults to `gpt-4o-mini` — cheap for agent calls
- `JUDGE_MODEL` defaults to `gpt-4o` — use the best model for scoring accuracy
- Full run (3 agents × 8 scenarios = 24 agent calls + 24 judge calls) costs approximately $0.30–$0.80 at current prices
- Use `--no-judge` flag during development to skip judge calls
- Use `--dry-run` to test the pipeline on 1 scenario at no cost above a single pair of calls
- Cache judge results: if `results/cache/{scenario_id}_{agent_name}_judge.json` exists, load it instead of calling the API

---

## 17. README.md — write this last

The README is the pitch document. Write it after you have real results.

### Structure

```markdown
# SRE-Bench

> Benchmark how well LLM agents diagnose root causes in production incidents.

## What this is
[2 sentences on the problem]

## What we found
[Your 3 most interesting findings from the actual run — use real numbers]

## How it works
[Architecture diagram or description — 3 paragraphs]

## Quickstart
pip install -e .
cp .env.example .env  # fill in your API keys
sre-bench run --dry-run
sre-bench run

## Sample report
[Link to or embed sample_run.html]

## Scenario categories
[Table: category, n_scenarios, description]

## Agent architectures compared
[Table: agent name, approach, key characteristic]

## Extending SRE-Bench
[How to add a scenario, how to add an agent]
```

**The "What we found" section is the most important part of the README.** It must contain real findings with numbers from your actual run. Do not write placeholder text here. Example of a good finding:

> "ReAct agents outperformed simple LLM on cascade scenarios (0.71 vs 0.48) but performed worse on easy deploy regressions (0.62 vs 0.84), likely because tool-calling overhead introduces confusion on obvious single-signal incidents."

---

## 18. What done looks like

The project is complete when all of the following are true:

- [ ] `pip install -e . && sre-bench run --dry-run` works from a fresh clone with only API keys set
- [ ] `sre-bench run` completes all 24 evaluations (3 agents × 8 scenarios) without crashing
- [ ] `sre-bench report` generates a valid HTML file with all 6 report sections
- [ ] `pytest tests/ -v` passes all tests
- [ ] `results/sample_run.html` is committed and renders in a browser
- [ ] README.md contains real findings with real numbers
- [ ] No hardcoded API keys anywhere in the codebase
- [ ] Repo is public on GitHub with a descriptive repo description