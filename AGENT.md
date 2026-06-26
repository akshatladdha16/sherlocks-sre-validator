## Company overview 
Sherlocks.ai just raised ₹7.5 crore (~$900K) in pre-seed on June 23rd, literally days ago. Led by SenseAI Ventures with participation from Uppekha, they're using the capital to strengthen product and scale go-to-market in North America. The team is only ~7 people, which means every hire is high-leverage. Businessnewsthisweek
## Product deep dive 
The core insight is that MTTR for production incidents averages 3–4 hours, largely because diagnosis remains manual, fragmented, and dependent on human coordination across multiple tools. Sherlocks deploys specialized AI agents across databases, Kubernetes, networks, CI/CD pipelines, and cloud infrastructure — working in concert to form hypotheses, correlate signals, and converge on root causes within minutes. Their differentiator is the "Awareness Graph" — a combination of telemetry, infrastructure state, incident history and team knowledge that lets the agent use historical context rather than treating every alert as an entirely new investigation

## What problem are we solving?
Sherlocks.ai replaces manual incident diagnosis with AI agents. But here's the thing: they don't know how good their agents actually are. They know agent success rate went from 35% to 74.8% — but what drives that number? Which failure modes dominate? Do different LLMs or prompting strategies do better on database incidents vs K8s incidents?

There's no systematic framework for measuring agent RCA quality. Every experiment is ad-hoc. This is exactly what the Research Engineer role asks you to fix.

## The real-world pain
When a production incident fires, the agent has ~5 minutes to return a useful RCA. If it hallucinates a root cause, engineers waste 20+ minutes chasing the wrong thing — worse than no answer.

## The measurement gap
You can't improve what you can't measure. Without an eval harness, every agent change is a guess. You need ground truth: "for this incident, the correct root cause is X — how close did the agent get?"

## Why this is hard
1. Incidents are one-off events. You can't replay a real K8s OOMKill on demand. So evaluation needs synthetic scenarios — fake but realistic incident contexts with known ground truth.

2. RCA quality is fuzzy. "The database was slow because of a missing index" is a correct answer but so is "the slow query was triggered by a deploy at 14:32." Evaluating LLM outputs requires more than string matching — you need semantic scoring.

3. Multi-signal correlation. Real incidents span logs + metrics + deployment events + Slack history simultaneously. An agent that only reads logs misses the signal hiding in the deploy timeline.


Architecture: data flow and agent harness diagram stays in ./arch-plan folder. review them. 

## The key design decisions, explained:
- The BaseAgent abstract class is the hinge everything swings on. It has a single method signature — diagnose(scenario: ScenarioContext) -> RCAOutput — and every agent variant, no matter how complex internally, must return the same Pydantic model. This is what lets the runner swap agents without caring about internals, and it mirrors exactly how Sherlocks would plug in a new model or strategy in production.
- The mock tool layer is what makes your ReActAgent feel real without needing actual infra. When LangChain's ReAct loop calls search_logs("heap space"), that function just filters scenario["context"]["logs"] and returns matching lines. When it calls get_deploys(), it returns scenario["context"]["recent_deploys"]. The agent doesn't know it's working with fake data — and that's the point. You're testing the reasoning, not the data retrieval.
- The LLM judge prompt is the most important thing you'll write in this whole project. It needs three things: the RCA the agent produced, the ground truth root cause from the scenario JSON, and a rubric telling it exactly what a score of 1.0, 0.5, and 0.0 looks like on each dimension. The judge must return structured JSON only — use response_format={"type": "json_object"} on the OpenAI call, or a strict system prompt with Claude. The failure label is what turns this from a number into a story — that's the thing Sherlocks will actually care about.
- The aggregation step is simpler than it sounds: just group results.json by (agent_name, scenario_category, difficulty) and compute mean scores. That gives you the leaderboard table naturally.


## Three things that will make your repo stand out from any other eval project:
- The first is the failure mode taxonomy. Don't just say "ReActAgent scored 0.4 on cascade failures." Say "ReActAgent consistently fails cascade scenarios because it stops investigating once it finds the first symptom — it never checks whether that symptom has an upstream cause. This is the early_stop failure mode." Name things. Patterns with names are actionable.
- The second is a --dry-run flag that runs the full pipeline on one scenario with one agent and prints every intermediate step — the scenario context, the agent's reasoning trace, the judge's raw output, and the final score. This makes the project feel like a real tool, not just a script.
- The third is a results/sample_run.html committed to the repo. Anyone who looks at your GitHub should be able to see what the output looks like without running anything. Put your most interesting finding in the report title — "ReAct agents miss upstream causes in 67% of cascade failures" is better than "SRE-Bench results."

