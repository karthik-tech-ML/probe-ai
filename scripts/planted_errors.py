"""
Phase 3: does the verifier catch real errors?

  collect   run scenarios through the multi-agent system and save question,
            evidence and final answer (API: about $0.12 per scenario)
  evaluate  corrupt each saved answer (swap a director, change a number, add an
            unsupported claim) and ask the verifier about the corrupted drafts
            plus the untouched originals (API: one verifier call each)
  --dry-run on evaluate prints every corruption and the call count, no API calls

    python scripts/planted_errors.py collect --ids sc_008,sc_009 --out results/planted_cases.json
    python scripts/planted_errors.py evaluate --dry-run
    python scripts/planted_errors.py evaluate
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multi_agent.planted import corrupt_all, corrupt_hard  # noqa: E402

DEFAULT_IDS = "sc_008,sc_009,sc_010,sc_011,sc_012,sc_013,sc_014,sc_016,sc_027,sc_030"


def collect(ids: list[str], out: Path) -> None:
    """Run scenarios for real and save what the verifier would have seen."""
    from src.evaluation.runner import load_scenarios
    from src.multi_agent.graph import run_multi_agent

    wanted = {s.id: s for s in load_scenarios() if s.id in ids}
    cases = []
    for sid in ids:
        sc = wanted[sid]
        res = run_multi_agent(sc.question)
        cases.append({"scenario_id": sid, "question": sc.question, "expected_answer": sc.expected_answer,
                      "answer": res.answer, "evidence": res.evidence,
                      "tokens": res.trace.total_input_tokens + res.trace.total_output_tokens})
        out.write_text(json.dumps(cases, indent=2, default=str))  # save as we go
    logger.info(f"saved {len(cases)} cases to {out}")


def _evidence_text(evidence: list[dict]) -> str:
    return " ".join(f"{e['result']} " + " ".join(str(o) for _, o in e.get("tool_outputs", []))
                    for e in evidence)


def build_jobs(cases: list[dict], hard: bool = False) -> list[dict]:
    """Easy: a clean job per case plus every corruption. Hard: subtler corruptions only, no clean controls."""
    jobs = []
    for i, c in enumerate(cases):
        base = {"scenario_id": c["scenario_id"], "question": c["question"], "evidence": c["evidence"]}
        if not hard:
            jobs.append({**base, "kind": "clean", "draft": c["answer"], "detail": ""})
        found = corrupt_hard(c["answer"]) if hard else corrupt_all(c["answer"], _evidence_text(c["evidence"]), index=i)
        for cor in found:
            jobs.append({**base, "kind": cor.type, "draft": cor.draft, "detail": cor.detail})
    return jobs


def evaluate(cases_path: Path, out: Path, dry_run: bool, hard: bool = False) -> None:
    jobs = build_jobs(json.loads(cases_path.read_text()), hard)
    counts = defaultdict(int)
    for j in jobs:
        counts[j["kind"]] += 1
    print(f"{len(jobs)} verifier calls: {dict(counts)}")
    if dry_run:
        for j in jobs:
            if j["kind"] != "clean":
                print(f"  {j['scenario_id']:7} {j['kind']:22} {j['detail']}")
        return

    from src.multi_agent.graph import verify_draft

    rows = []
    for j in jobs:
        action, reason, i_tok, o_tok = verify_draft(j["question"], j["evidence"], j["draft"])
        rows.append({"scenario_id": j["scenario_id"], "kind": j["kind"], "detail": j["detail"],
                     "action": action, "reason": reason, "tokens": i_tok + o_tok})
        logger.info(f"{j['scenario_id']} {j['kind']}: {action}")
        out.write_text(json.dumps({"rows": rows}, indent=2))

    tally: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        tally[r["kind"]].append(r["action"] == "revise")
    summary = {k: {"n": len(v), "flagged": sum(v), "rate": round(sum(v) / len(v), 3)} for k, v in tally.items()}
    corrupted = [f for k, v in tally.items() if k != "clean" for f in v]
    summary["catch_rate_overall"] = round(sum(corrupted) / len(corrupted), 3) if corrupted else None
    summary["false_alarm_rate"] = summary.get("clean", {}).get("rate")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ids", default=DEFAULT_IDS)
    c.add_argument("--out", default="results/planted_cases.json")
    e = sub.add_parser("evaluate")
    e.add_argument("--cases", default="results/planted_cases.json")
    e.add_argument("--out", default="results/planted_errors.json")
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--hard", action="store_true", help="subtler corruptions only (no clean controls)")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.ids.split(","), Path(a.out))
    else:
        evaluate(Path(a.cases), Path(a.out), a.dry_run, a.hard)


if __name__ == "__main__":
    main()
