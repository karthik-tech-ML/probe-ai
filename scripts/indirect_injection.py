"""
Indirect prompt injection test: a poisoned movie record with a hidden instruction.

    python scripts/indirect_injection.py --dry-run     # shows the poisoned tool output, no API calls
    python scripts/indirect_injection.py --limit 2     # cheap trial
    python scripts/indirect_injection.py               # 4 payloads x 2 questions x 2 systems = 16 runs

The database is not modified; the tools are wrapped in-process for each run.
Results: results/indirect_injection.json
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.safety.indirect_injection import (  # noqa: E402
    PAYLOADS, PAYLOADS_V2, PAYLOADS_V3, QUESTIONS, complied, poisoned, spread, token_in,
)


def jobs_for(payloads, systems=("agent", "multi_agent")) -> list:
    return [(s, q, p) for p in payloads for q in (p.questions or QUESTIONS) for s in systems]


def dry_run(payloads) -> None:
    from src.agent.tools import search_by_director, search_movies

    print(f"{len(jobs_for(payloads))} runs across {len(payloads)} payloads\n")
    for p in payloads:
        with poisoned(p):
            if p.surface in ("cast", "genres"):
                out = search_by_director.invoke({"director_name": "Christopher Nolan"})
                hit = next((r for r in out if r["title"] == "Inception"), None)
                tail = hit[p.surface][-170:] if hit else "(Inception not returned)"
            else:
                out = search_movies.invoke({"query": "Inception"})
                hit = next((r for r in out if "Title: Inception" in r["chunk_text"]), None)
                tail = hit["chunk_text"][-170:].replace("\n", " ") if hit else "(Inception not retrieved)"
        print(f"[{p.id}] ({p.surface}) goal: {p.goal}\n   tail: ...{tail}\n")


def run_one(system: str, question: str, payload) -> dict:
    with poisoned(payload):
        if system == "multi_agent":
            from src.multi_agent.graph import run_multi_agent
            res = run_multi_agent(question)
            tokens = res.trace.total_input_tokens + res.trace.total_output_tokens
            extra = {"spread": spread(res.trace, payload)}
        else:
            from src.agent.graph import run_agent
            res = run_agent(question)
            tokens = res.trace.total_input_tokens + res.trace.total_output_tokens
            extra = {}
    # did the poisoned text actually reach the model? Without this, "resisted" could mean "never saw it"
    seen = any(token_in(tc.tool_output, payload) for tc in res.trace.tool_calls)
    return {"system": system, "payload": payload.id, "question": question, "answer": res.answer,
            "payload_seen": seen, "complied": complied(payload, res.answer), "token_in_answer": token_in(res.answer, payload),
            "tokens": tokens, **extra}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--round", type=int, choices=(1, 2, 3), default=1, help="1: crude, 2: stronger, 3: fresh payloads written after the mitigation")
    ap.add_argument("--systems", default="agent,multi_agent", help="comma-separated: agent, multi_agent")
    ap.add_argument("--only", default=None, help="only this payload id (e.g. r3_no_conflict)")
    ap.add_argument("--repeat", type=int, default=1, help="run each job this many times")
    ap.add_argument("--limit", type=int, default=None, help="only run this many runs")
    ap.add_argument("--out", default="results/indirect_injection.json")
    a = ap.parse_args()
    payloads = {1: PAYLOADS, 2: PAYLOADS_V2, 3: PAYLOADS_V3}[a.round]
    if a.dry_run:
        return dry_run(payloads)

    if a.only:
        payloads = tuple(p for p in payloads if p.id == a.only)
    jobs = jobs_for(payloads, tuple(a.systems.split(","))) * a.repeat
    if a.limit:
        jobs = jobs[: a.limit]
    rows = []
    for s, q, p in jobs:
        row = run_one(s, q, p)
        rows.append(row)
        logger.info(f"{s} {p.id}: complied={row['complied']}")
        Path(a.out).write_text(json.dumps({"rows": rows}, indent=2))

    summary: dict[str, dict] = {}
    for r in rows:
        d = summary.setdefault(r["system"], {"runs": 0, "payload_seen": 0, "complied": 0, "token_in_answer": 0})
        d["runs"] += 1
        d["payload_seen"] += r["payload_seen"]
        d["complied"] += r["complied"]
        d["token_in_answer"] += r["token_in_answer"]
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
