# ProbeAI v2: multi-agent build guide (Phases 1 and 2)

**Goal:** turn the system under test from one ReAct agent into a small team of agents, then add the metrics that show where a multi-agent system breaks.

**Headline question the work answers:** do more agents actually help, and what do they cost?

**Time:** about 8 to 12 hours. One step per sitting is fine.

---

## How to work with Claude Code on this

1. **One step per prompt.** Paste one prompt below, let it finish, check the result, commit. Don't paste the whole plan at once.
2. **Ask for a plan first.** Each prompt asks Claude Code to explain its approach before editing. Read it. Push back if it wants to rewrite things that already work.
3. **Protect what works.** The existing 7 metrics, the RAG path, and the single-agent path must keep working. Every step ends with a check for that.
4. **Commit after every step.** If a step goes sideways, you can roll back to the last good commit.
5. **Watch API cost.** Multi-agent runs make several LLM calls per question. Smoke tests use 3 scenarios. Full runs use a subset of about 18, not all 40.

---

## Setup (10 minutes)

In a terminal, inside your `probe-ai` folder:

```bash
git checkout -b multi-agent
claude
```

Then paste this first, so Claude Code knows the plan:

> Read CLAUDE.md, README.md, src/agent/graph.py, src/agent/tools.py, src/agent/memory.py, src/evaluation/runner.py, and src/evaluation/metrics/tool_selection.py. Then read docs/multi-agent-plan.md. Summarize in 10 bullets how the current agent path works end to end, and what you think will be hardest about Phase 1. Don't change any files yet.

Check its summary against what you know. If it misunderstands something, correct it now. That's cheaper than fixing code later.

---

## Phase 1: build the multi-agent system

### Step 0: Make the model configurable (15 minutes)

> The model ID "claude-sonnet-4-5-20250929" is hard-coded in several places (graph.py, runner.py, and possibly others). Move it into src/config.py as a setting with an env var override (e.g. LLM_MODEL), and use that setting everywhere. Add it to .env.example. Show me every file you plan to touch before editing. Don't change behavior otherwise.

**Check:** `grep -rn "claude-sonnet-4-5" src scripts` returns nothing. Run `python scripts/smoke_test_rag.py` and `python scripts/smoke_test_agent.py`. Both still work.

**Commit:** `git commit -am "Make LLM model configurable"`

### Step 1: A trace that records which agent did what (45 minutes)

> Create src/multi_agent/trace.py with a MultiAgentTrace that extends or wraps the existing AgentTrace in src/agent/memory.py. Each step should record: agent_name, step type (route, tool_call, handoff, draft, verify, final), input, output, tokens, and timestamp. Keep the existing properties working (tools_used, num_tool_calls, tool_calls, to_dict) so the existing tool_selection and task_success metrics work unchanged on a MultiAgentTrace. Add properties for: agents_used, handoff_count, llm_call_count, and route_sequence. Plan first, then implement, then add pytest tests in tests/test_multi_agent_trace.py that build a fake trace by hand (no API calls).

**Check:** `pytest tests/test_multi_agent_trace.py` passes.

**Commit.**

### Step 2: The agents (2 to 3 hours)

> Build a supervisor-plus-specialists system in src/multi_agent/ using LangGraph's StateGraph. Reuse the existing tools in src/agent/tools.py. Don't duplicate them.
>
> Agents:
> - supervisor: reads the question, writes a short plan of steps, routes each step to one specialist, and decides when to stop
> - retrieval_agent: semantic search only (search_movies, get_movie_details)
> - structured_agent: precise lookups and filters (search_by_director, search_by_cast, search_by_genre, filter_movies)
> - analyst_agent: no database tools; compares and aggregates what other agents returned
> - verifier_agent: checks the draft answer against the evidence gathered so far; returns approve, or revise with a reason. Allow at most one revise loop.
>
> Rules:
> - Every agent follows the same grounding rule as the current system prompt: only answer from tool results.
> - Cap total steps (e.g. 12) so a confused supervisor can't loop forever. When the cap is hit, return the best answer so far and record that the cap was hit.
> - Record every step in MultiAgentTrace.
> - Expose run_multi_agent(question) -> MultiAgentResult, mirroring run_agent in src/agent/graph.py.
> - Don't add new dependencies unless you tell me why first.
>
> Plan first: show me the graph (nodes, edges, state fields) as a short diagram in text before writing code.

**Check:** read the graph it proposes. Does the supervisor route steps, or does it just call everything? Routing is what you're testing, so it has to be a real decision.

**Commit.**

### Step 3: Plug it into the eval runner (45 minutes)

> Add a multi-agent path to src/evaluation/runner.py, mirroring _run_single_agent: a _run_single_multi_agent function that wraps MultiAgentResult into a PipelineResult, builds context_texts from tool outputs, and stashes the trace on pipeline_result._agent_trace so existing metrics work. Add a use_multi_agent option to run(), and a --multi-agent flag to scripts/evaluate.py. Record model as "<model> (multi-agent)". Don't change the RAG or single-agent paths.

**Commit.**

### Step 4: Smoke test (30 minutes)

> Create scripts/smoke_test_multi_agent.py that runs 3 scenarios from scenarios/library.json (one multi_field, one aggregation, one multi_hop) through the multi-agent system and prints, for each: the answer, the route sequence, handoff count, LLM call count, total tokens, latency, and whether the verifier approved or revised. Then run it.

**Check:** run it. Read one full trace by hand. Does the route make sense for the question? If the supervisor sends everything to one agent, tune the supervisor prompt before moving on.

**Phase 1 done when:** RAG, single-agent, and multi-agent paths all run through `scripts/evaluate.py`, and the smoke test output makes sense to you.

**Commit and push.**

---

## Phase 2: metrics that show where multi-agent breaks

### Step 5: Expected routes in scenarios (45 minutes)

> Add an optional field expected_route (list of specialist names, e.g. ["structured_agent", "analyst_agent"]) to the Scenario dataclass in src/evaluation/runner.py with a default of an empty list, so existing scenarios still load. Update scenarios/schema.md. Then propose expected_route values for every multi_field, aggregation, and multi_hop scenario in scenarios/library.json, and show me the list for approval before writing them.

**Check:** read its proposed routes. You're the judge of what "correct routing" means. Change any you disagree with before it writes them.

**Commit.**

### Step 6: Mechanical metrics, no LLM calls (1.5 hours)

> Add these metrics under src/evaluation/metrics/, each following the existing MetricFn interface (scenario, pipeline_result) -> dict, reading the trace from pipeline_result._agent_trace and returning None scores with a reason when there's no multi-agent trace:
> 1. routing.py: routing_accuracy. Compare the specialists actually used to scenario.expected_route: precision, recall, and an exact-order match flag.
> 2. coordination.py: handoff_count, loop_detected (same agent called with the same input twice, or step cap hit), steps_to_answer.
> 3. cost.py: llm_call_count, total_tokens, latency_ms, and tokens per correct answer (use the existing task_success result if present).
> 4. verifier.py: verifier_action (approve or revise), and whether a revision changed the final answer.
> Add pytest tests for each using hand-built fake traces. No API calls in tests.

**Check:** `pytest` passes, including the old tests.

**Commit.**

### Step 7: Error attribution, LLM-as-judge (1.5 hours)

> Add src/evaluation/metrics/attribution.py. Only for scenarios where the answer is wrong (use existing task_success or hallucination scores), ask Claude to read the full multi-agent trace plus the expected answer and return JSON: culprit_agent (one of the agent names, or "data" if the needed info wasn't in the corpus), failure_type (wrong_route, missing_evidence, bad_tool_args, reasoning_error, verifier_missed, other), and a one-sentence reason. Also return error_propagated: true if a wrong claim from a specialist's output appears in the final answer. Follow the structure of the existing LLM-as-judge in faithfulness.py. Use a low temperature. Cache judge results by scenario ID and trace hash so reruns don't pay twice.

**Check:** run it on 2 or 3 wrong answers and read the verdicts. Do you agree? An LLM judge is only useful if you'd trust it.

**Commit.**

### Step 8: The three-way comparison (1 hour plus run time)

> Add scripts/compare_modes.py that runs the same scenario subset (multi_field, aggregation, multi_hop, adversarial; about 18 scenarios) through RAG, single agent, and multi-agent, with all relevant metrics. Save to results/mode_comparison.json and print one table per mode with: accuracy (task_success), faithfulness, hallucination, routing accuracy (multi-agent only), avg LLM calls, avg tokens, avg latency, and a failure count by culprit_agent. Add a --limit option so I can test on 3 scenarios first.

**Check:** run with `--limit 3` first. Then run the full subset once.

**Phase 2 done when:** you have one table comparing the three modes, and you can answer: where did multi-agent win, where did it lose, what did it cost, and which agent caused most failures?

**Commit and push.**

---

## After Phase 2: write it up (feeds Session 5)

Answer these in one line each, from your own results:
1. Did multi-agent beat single-agent on multi-hop and aggregation? By how much?
2. What did it cost in LLM calls, tokens, and latency?
3. Which agent caused most failures, and what type?
4. Did the verifier catch real errors, or mostly approve?
5. What would you change first?

That's the core of your updated ProbeAI story.

---

## Later (Phases 3 and 4, optional)

- Planted errors to measure verifier catch rate and false alarms
- Indirect prompt injection: a poisoned movie record with hidden instructions, to see whether it spreads between agents
- Tool failure injection: a tool that errors or times out, to test recovery
- Dashboard and README updates with the comparison results
