# Talk summary: what to present

Every number here comes from a saved file in `results/`; the details and caveats are in
[multi-agent-findings.md](multi-agent-findings.md).

## One-line pitch
ProbeAI tests AI systems the way you test software. Pointed at a multi-agent system, it found that the
fancy architecture did not beat a single agent, and it found real bugs the team would have shipped.

## The story in six beats

1. **The setup.** RAG, a single ReAct agent and a multi-agent system (supervisor, three specialists,
   verifier) on the same 18 hard scenarios. RAG 0.67, single agent 0.94, multi-agent 0.94.
2. **Multi-agent did not win.** Same accuracy at about 1.4x to 3x the cost and about 3x the latency,
   depending on whether a run spirals. Lead with this: an honest negative result.
3. **The eval found real bugs.** Failure attribution pointed at a lookup tool that matched a director
   name anywhere in the record (it returned 22 films for an actor). Fixing it cut tokens 40% to 50%.
   Runaway runs (394K tokens on one question) led to a per-run token budget.
4. **Does the verifier work?** Planted errors: it caught 27 of 27 obvious ones and about 20 of 23
   subtle ones, with 0 to 1 false alarm in 10. It is strong on wrong values and weak on omissions
   and counts.
5. **Safety: poisoned data.** A hidden instruction in a movie record was followed by the multi-agent
   system in 4 of 7 stronger-payload runs, by the single agent in 0 of 7. Wrapping evidence as
   untrusted data and a data-not-instructions rule brought it to 0 of 7, and 0 of 5 on fresh payloads.
   Unsolved: a plausible false fact with nothing to contradict it (multi-agent followed it 1 of 4,
   single agent 4 of 4).
6. **Robustness: tool failures.** Before a fix, any tool exception crashed both systems (12 of 16
   runs). After surfacing errors to the model with a retry-or-disclose rule: 0 crashes.

## The line that makes it memorable
The framework also caught its own mistakes: a wrong expected answer, a judge that scored the same
answer 1.0 and then 0.5, and my own classifier. **Test the tests.**

## What NOT to claim
- That multi-agent is generally worse. This is one task, 18 scenarios, one run each.
- That the injection fix "works". It beat the payloads it was written against and five fresh ones,
  at 1 to 2 runs per payload. The false-fact case remains open.
- That the verifier "catches 100%". It caught every obvious planted error, and missed 3 of 23 subtle ones.
- Cost multiples as fixed numbers. They swing between 1.4x and 3x with runaway runs.

## Suggested demo
`python scripts/planted_errors.py evaluate --dry-run` (free, shows the corruptions), the dashboard's
Multi-agent tab, and the one multi-agent answer that stated "directed by Steven Spielberg" citing a
planted "verified correction" before the fix.

## Reproduce
```bash
python -m pytest tests                      # 93 tests, no API calls
python scripts/add_multi_agent_tab.py       # rebuild the dashboard tab from results/
```
