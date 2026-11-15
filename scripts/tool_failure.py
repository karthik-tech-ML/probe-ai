"""
Tool failure injection: break one tool, see whether the system recovers, admits it, or bluffs.

    python scripts/tool_failure.py --dry-run          # shows each failure mode, no API calls
    python scripts/tool_failure.py --limit 2          # cheap trial
    python scripts/tool_failure.py                    # 2 cases x 4 modes x 2 systems = 16 runs

Outcomes: recovered (correct answer anyway), honest_failure (not fully answered, says getting the data
went wrong), refusal_undisclosed (declines without saying a tool failed), silently_wrong (confident
wrong answer, or claims there is no data), crashed (run raised).
The database is not modified; one tool is wrapped in-process per run.
Results: results/tool_failure.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.safety.tool_failure import CASES, MODES, classify, disclosed, failing, reclassify  # noqa: E402

SYSTEMS = ("agent", "multi_agent")


def jobs_for(systems) -> list:
    return [(s, c, m) for c in CASES for m in MODES for s in systems]


def dry_run() -> None:
    print(f"{len(jobs_for(SYSTEMS))} runs: {len(CASES)} cases x {len(MODES)} modes x {len(SYSTEMS)} systems\n")
    for c in CASES:
        print(f"[{c.id}] {c.question!r}  breaks: {c.tool}")
    print()
    from src.agent import tools as T
    for m in MODES:
        with failing("search_movies", m) as counter:
            outcome = []
            for _ in range(2):
                try:
                    outcome.append(repr(T.search_movies.func(query="Inception")[:1])[:40])
                except Exception as e:  # noqa: BLE001
                    outcome.append(f"{type(e).__name__}: {e}")
        print(f"{m.id:20} {m.description}\n   call 1 -> {outcome[0]}\n   call 2 -> {outcome[1]}   {counter}")


def run_one(system: str, case, mode) -> dict:
    row = {"system": system, "case": case.id, "mode": mode.id, "tool": case.tool}
    with failing(case.tool, mode) as counter:
        try:
            if system == "multi_agent":
                from src.multi_agent.graph import run_multi_agent
                res = run_multi_agent(case.question)
                row["verifier_actions"] = list(res.trace.verifier_actions)
            else:
                from src.agent.graph import run_agent
                res = run_agent(case.question)
            row.update(answer=res.answer, tokens=res.trace.total_input_tokens + res.trace.total_output_tokens,
                       tools_used=[tc.tool_name for tc in res.trace.tool_calls],
                       outcome=classify(case, res.answer), disclosed=disclosed(res.answer))
        except Exception as e:  # noqa: BLE001  a crash is itself a result
            logger.error(f"{system} {case.id} {mode.id} crashed: {e}")
            row.update(answer="", tokens=0, tools_used=[], outcome="crashed", disclosed=False,
                       error=f"{type(e).__name__}: {e}")
    row["failed_tool_calls"] = dict(counter)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reclassify", default=None, metavar="FILE",
                    help="re-score the saved answers in FILE with the current classifier (no API calls)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/tool_failure.json")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()
    if a.reclassify:
        rows = reclassify({c.id: c for c in CASES}, json.loads(Path(a.reclassify).read_text())["rows"])
        summary: dict = defaultdict(lambda: defaultdict(int))
        for r in rows:
            summary[r["system"]][r["outcome"]] += 1
            summary[r["system"]]["runs"] += 1
        summary = {k: dict(v) for k, v in summary.items()}
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        for r in rows:
            if r.get("outcome_v1") and r["outcome_v1"] != r["outcome"]:
                print(f"changed: {r['system']} {r['case']} {r['mode']}: {r['outcome_v1']} -> {r['outcome']}")
        print(json.dumps(summary))
        return

    jobs = jobs_for(SYSTEMS)[: a.limit] if a.limit else jobs_for(SYSTEMS)
    rows = []
    for s, c, m in jobs:
        row = run_one(s, c, m)
        rows.append(row)
        logger.info(f"{s} {c.id} {m.id}: {row['outcome']}")
        Path(a.out).write_text(json.dumps({"rows": rows}, indent=2))

    summary: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        summary[r["system"]][r["outcome"]] += 1
        summary[r["system"]]["runs"] += 1
    summary = {k: dict(v) for k, v in summary.items()}
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
