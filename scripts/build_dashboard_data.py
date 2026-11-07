"""
Convert eval runner JSON results to the dashboard flat format.

Usage:
    python scripts/build_dashboard_data.py \
        --rag results/rag_full_real.json \
        --agent results/agent_full_real.json \
        --out dashboard/dist/data/eval-results.json

Either --rag or --agent is optional; the other side of the dashboard is
left unchanged if the flag is omitted.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _sources_list(r: dict) -> list[str]:
    """Build 'Title (sim)' strings from retrieved_titles + similarity_scores."""
    titles = r.get("retrieved_titles") or []
    sims   = r.get("similarity_scores") or []
    out = []
    for title, sim in zip(titles, sims):
        out.append(f"{title} ({sim:.4f})")
    return out


def _safe_score(s: dict, key: str):
    """
    Return the numeric score value or None.

    When a metric scorer hits a 429/error, the runner stores the exception
    as {"error": "..."} instead of a float. Detect that and return None so
    the dashboard renders it as n/a rather than crashing.
    """
    v = s.get(key)
    if isinstance(v, dict):
        return None   # error dict from a failed scorer call
    return v


def _convert_rag_result(r: dict) -> dict:
    s = r.get("scores", {})
    entry = {
        "id":                  r["scenario_id"],
        "question":            r["question"],
        "expected":            r.get("expected_answer", ""),
        "answer":              r.get("answer", ""),
        "category":            r["category"],
        "difficulty":          r.get("difficulty", ""),
        # retrieval metrics
        "recall":              _safe_score(s, "recall_at_k"),
        "context_precision":   _safe_score(s, "context_precision"),
        "mrr":                 _safe_score(s, "mrr"),
        "ndcg":                _safe_score(s, "ndcg"),
        # generation quality
        "faithfulness":        _safe_score(s, "faithfulness_score"),
        "grounding":           _safe_score(s, "grounding_score"),
        "hallucination":       _safe_score(s, "hallucination_score"),
        "answer_relevancy":    _safe_score(s, "answer_relevancy"),
        "context_relevancy":   _safe_score(s, "context_relevancy"),
        # operational
        "latency_ms":          int(r.get("latency_ms", 0)),
        "latency_pass":        s.get("latency_pass"),
        "sources":             _sources_list(r),
        "input_tokens":        _safe_score(s, "input_tokens"),
        "output_tokens":       _safe_score(s, "output_tokens"),
        "total_tokens":        _safe_score(s, "total_tokens"),
        "cost_usd":            _safe_score(s, "cost_usd"),
    }
    return entry


def _convert_agent_result(r: dict) -> dict:
    s = r.get("scores", {})

    # tool information is in the scores dict from tool_selection scorer
    tools_used = s.get("tool_selection_tools_used") or []
    tool_calls = s.get("tool_selection_num_calls") or 0

    entry = {
        "id":                  r["scenario_id"],
        "question":            r["question"],
        "expected":            r.get("expected_answer", ""),
        "answer":              r.get("answer", ""),
        "category":            r["category"],
        "difficulty":          r.get("difficulty", ""),
        # agent-specific
        "task_success":        _safe_score(s, "task_success_score"),
        "tool_selection":      _safe_score(s, "tool_selection_score"),
        # generation quality (shared metrics)
        "faithfulness":        _safe_score(s, "faithfulness_score"),
        "grounding":           _safe_score(s, "grounding_score"),
        "hallucination":       _safe_score(s, "hallucination_score"),
        "answer_relevancy":    _safe_score(s, "answer_relevancy"),
        # operational
        "latency_ms":          int(r.get("latency_ms", 0)),
        "latency_pass":        s.get("latency_pass"),
        "tool_calls":          tool_calls,
        "tools_used":          tools_used,
        "input_tokens":        _safe_score(s, "input_tokens"),
        "output_tokens":       _safe_score(s, "output_tokens"),
        "total_tokens":        _safe_score(s, "total_tokens"),
        "cost_usd":            _safe_score(s, "cost_usd"),
    }
    return entry


def _load_and_convert(path: Path, mode: str) -> dict:
    """
    Load a runner JSON file and return a dict of category → [entries].
    mode: "rag" or "agent"
    """
    raw = json.loads(path.read_text())
    results = raw.get("results", [])

    by_category: dict[str, list] = defaultdict(list)
    for r in results:
        if mode == "rag":
            entry = _convert_rag_result(r)
        else:
            entry = _convert_agent_result(r)
        by_category[r["category"]].append(entry)

    # Sort each category by scenario id for stable ordering
    for cat in by_category:
        by_category[cat].sort(key=lambda e: e["id"])

    return dict(by_category)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard data from eval results")
    parser.add_argument("--rag",   type=str, help="path to RAG eval JSON")
    parser.add_argument("--agent", type=str, help="path to agent eval JSON")
    parser.add_argument("--out",   type=str,
                        default="dashboard/dist/data/eval-results.json",
                        help="output path (default: dashboard/dist/data/eval-results.json)")
    args = parser.parse_args()

    if not args.rag and not args.agent:
        parser.error("provide at least one of --rag or --agent")

    out_path = Path(args.out)

    # start from existing file so we only overwrite the sides we have data for
    if out_path.exists():
        dashboard = json.loads(out_path.read_text())
    else:
        dashboard = {}

    if args.rag:
        rag_path = Path(args.rag)
        if not rag_path.exists():
            raise FileNotFoundError(f"RAG results not found: {rag_path}")
        print(f"loading RAG results from {rag_path}")
        dashboard["rag"] = _load_and_convert(rag_path, "rag")
        rag_counts = {cat: len(v) for cat, v in dashboard["rag"].items()}
        print(f"  → {sum(rag_counts.values())} RAG entries across {rag_counts}")

    if args.agent:
        agent_path = Path(args.agent)
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent results not found: {agent_path}")
        print(f"loading agent results from {agent_path}")
        dashboard["agent"] = _load_and_convert(agent_path, "agent")
        agent_counts = {cat: len(v) for cat, v in dashboard["agent"].items()}
        print(f"  → {sum(agent_counts.values())} agent entries across {agent_counts}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dashboard, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
