# Multi-agent findings

Answers to the five questions in `multi-agent-plan.md`, from real runs on the
18-scenario hard subset (multi_field 5, aggregation 4, multi_hop 4, adversarial 5).
Raw results are in `results/`. Each number names its source file.

**Read these with the caveats at the bottom.** n=18 and single runs: one scenario
is 5.5 points of accuracy, and the same answer was scored 1.0 and 0.5 by the judge
on different runs.

## 1. Did multi-agent beat single-agent on multi-hop and aggregation?

**No.** Task success, tool-fixed code (`mode_comparison_fixed_tools.json`):

| Category | RAG | Single agent | Multi-agent |
|---|---|---|---|
| multi_hop (4) | 0.75 | **1.00** | 0.88 |
| aggregation (4) | 0.75 | **1.00** | **1.00** |
| multi_field (5) | 0.40 | 0.80 | **0.90** |
| adversarial (5) | 0.80 | 1.00 | 1.00 |
| **Overall (18)** | 0.67 | **0.94** | **0.94** |

Both agent modes beat RAG by about 27 points. Multi-agent tied the single agent
overall and was one scenario worse on multi_hop. After the later prompt and budget
changes, multi-agent scored 0.89 overall (`mode_comparison_v3.json`), and the two
dropped scenarios were a wrong expected answer (sc_010, since fixed) and judge variance.

## 2. What did it cost?

Multi-agent vs single agent (`mode_comparison_fixed_tools.json`, and v3 for the
later multi-agent code):

| | Single agent | Multi-agent |
|---|---|---|
| LLM calls per query | 2.5 | 8.1 |
| Tokens per query | 12.3K | 32K to 37K |
| Latency | 7.3 s | 25 s to 32 s |
| Cost per query | $0.041 | $0.12 to $0.13 |

About **3x the calls, tokens and cost, and 3x to 4x the latency**, for no accuracy gain here.
The token figure includes runaway runs. Before the cap, one scenario (sc_012) used
394K tokens; after the fix it used 12.8K. Excluding that runaway, multi-agent averaged
15.7K tokens against 8.0K for the single agent, roughly 2x.

## 3. Which agent caused most failures, and what type?

Attribution (LLM-as-judge, wrong answers only):

- **First run:** `structured_agent` / `bad_tool_args` was the top culprit (3 failures).
  Following those traces led to a real bug: `search_by_director("Leonardo DiCaprio")`
  returned 22 films, because the match crossed line boundaries and found the name in the
  Cast line. Fixing it cut tokens 40% to 50% on director and cast lookups.
- **Latest full run (`mode_comparison_v3.json`):** `data / other` (3), `structured_agent`
  (3: two bad_tool_args, one reasoning_error), `analyst_agent` (1).

Two lessons. The "agent" blamed was often a **tool or test-data problem**, not a
reasoning problem. And "data / other" failures were frequently **errors in our own
expected answers** (sc_010 listed Titanic as an action film; the data says it isn't).
Attribution found the tool bug, but its verdicts need a human check.

## 4. Did the verifier catch real errors, or mostly approve?

In normal runs it mostly approved (1 revise in 18 on the first run, 3 in 18 on v3).
That could mean good drafts or a weak checker, so I tested it directly with planted
errors (`scripts/planted_errors.py`, 10 saved cases with their real evidence):

| Test | Result |
|---|---|
| Obvious errors (swap a director, change a number, append an unsupported claim) | **27 of 27 caught** |
| Subtler errors (near-miss numbers, dropped list items, swapped years) | **19 of 23 caught** (82.6%) |
| Untouched correct drafts flagged | 1 of 10 (10% false alarms) |

It is reliable on wrong values: all 8 swapped years were caught, including off-by-one.
It is weaker on **omissions** (3 of 5 dropped list items) and **counts stated in the
answer** (missed 2 to 3 and 8 to 9). Its one false alarm was itself a mistake: it
treated "above 7.5" as including 7.5.

## 5. What would I change first?

1. **Keep the single agent as the default and add the verifier as a final check.**
   The verifier catches wrong values at about 2K tokens a call, while the full
   multi-agent system costs about 3x more for no accuracy gain on this task.
2. **Add an aggregate tool** (for example, count movies per director). Questions like
   "which director has the most movies" cannot be answered with the current tools. One
   such run spent 325K tokens searching before the token budget was added.
3. **Fix the evaluation, not just the system.** Several "failures" were wrong expected
   answers, and the hallucination judge flags 0.5 to 1.0 on correct or refusing
   answers. Before trusting a number, check the scenario and the judge.

Already done in this branch: anchored lookups, per-turn tool-round cap, run-level
token budget (60K), single-agent round cap (10), production-company filter.

## Caveats

- **Sample size.** 18 scenarios, one run each. Differences of one or two scenarios are
  not findings.
- **Judge noise.** The same sc_010 answer scored 1.0 and then 0.5. Many expected answers
  say "Varies", which makes task_success lenient.
- **Hallucination judge** flags 0.5 to 1.0 on answers task_success scores correct, and on
  plain refusals (sc_015), so those figures are unreliable.
- **Routing accuracy** (0.74 to 0.86) is low partly because my expected routes
  over-require the analyst on questions the structured agent answers alone.
- **The token budget is only mock-tested.** The runaway did not recur in the targeted
  rerun, so it has not fired on a real run.
- **Planted errors** used deterministic edits on 10 answers and only 4 distinct
  director names; the clean-draft false-alarm rate rests on 10 drafts.
- Cost totals are estimates from model pricing, not billed amounts.
