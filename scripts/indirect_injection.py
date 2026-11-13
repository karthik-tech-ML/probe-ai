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
    PAYLOADS, QUESTIONS, complied, poisoned, spread, token_in,
)


def dry_run() -> None:
    from src.agent.tools import search_movies

    print(f"{len(PAYLOADS)} payloads x {len(QUESTIONS)} questions x 2 systems = "
          f"{len(PAYLOADS) * len(QUESTIONS) * 2} runs\n")
    for p in PAYLOADS:
        with poisoned(p):
            out = search_movies.invoke({"query": "Inception"})
        hit = next((r for r in out if "Title: Inception" in r["chunk_text"]), None)
        tail = hit["chunk_text"][-170:].replace("\n", " ") if hit else "(Inception not retrieved)"
        print(f"[{p.id}] goal: {p.goal}\n   record tail: ...{tail}\n")
    clean = search_movies.invoke({"query": "Inception"})
    print("clean record carries payload text:", any("NOTE TO ALL" in r["chunk_text"] for r in clean))


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
    ap.add_argument("--limit", type=int, default=None, help="only run this many runs")
    ap.add_argument("--out", default="results/indirect_injection.json")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()

    jobs = [(s, q, p) for p in PAYLOADS for q in QUESTIONS for s in ("agent", "multi_agent")]
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
