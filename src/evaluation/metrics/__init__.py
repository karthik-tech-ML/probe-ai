"""
Metric scorer functions for the ProbeAI eval framework.

All scorers follow the same interface:
    fn(scenario: Scenario, result: PipelineResult) -> dict[str, Any]

14 core scorers, plus 4 multi-agent scorers (below), grouped by when they run:

RAG + agent (7):
    score_faithfulness       — LLM judge: are claims supported by context?
    score_grounding          — LLM judge: which source backs each claim?
    score_hallucination      — fraction of claims with no source support
    score_latency            — pass/fail against LATENCY_THRESHOLD_MS
    score_answer_relevancy   — LLM judge: does the answer address the question?
    score_token_efficiency   — input + output tokens consumed per query
    score_cost_per_query     — estimated dollar cost based on model pricing

RAG only (5):
    score_retrieval          — Recall@K + Precision@K vs ground truth IDs
    score_context_precision  — Precision@K (standalone, RAGAS-aligned name)
    score_context_relevancy  — LLM judge: are retrieved docs useful for this question?
    score_mrr                — Mean Reciprocal Rank: where does the first relevant doc land?
    score_ndcg               — nDCG: are the most relevant docs ranked highest?

Agent only (2):
    score_task_success       — LLM judge: did the agent complete the task?
    score_tool_selection     — mechanical: did the agent call the right tools?

Multi-agent only (5, return None without a MultiAgentTrace; attribution is an LLM judge, the rest are mechanical):
    score_routing            — did the supervisor route to the expected specialists?
    score_coordination       — handoffs, loop detection, steps to answer
    score_multi_agent_cost   — LLM calls, tokens, latency, tokens per correct answer
    score_verifier           — approve/revise, and whether a revision changed the answer
    score_attribution        — LLM judge, wrong answers only: which agent caused it and why

Note: score_context_relevancy also runs meaningfully in agent mode (it judges
tool call outputs as context), but it's excluded from AGENT_METRICS in
evaluate.py to keep the agent table readable.
"""

from src.evaluation.metrics.answer_relevancy import score_answer_relevancy
from src.evaluation.metrics.attribution import score_attribution
from src.evaluation.metrics.context_precision import score_context_precision
from src.evaluation.metrics.context_relevancy import score_context_relevancy
from src.evaluation.metrics.coordination import score_coordination
from src.evaluation.metrics.cost import score_multi_agent_cost
from src.evaluation.metrics.cost_per_query import score_cost_per_query
from src.evaluation.metrics.faithfulness import score_faithfulness
from src.evaluation.metrics.grounding import score_grounding
from src.evaluation.metrics.hallucination import score_hallucination
from src.evaluation.metrics.latency import score_latency
from src.evaluation.metrics.mrr import score_mrr
from src.evaluation.metrics.ndcg import score_ndcg
from src.evaluation.metrics.retrieval import score_retrieval
from src.evaluation.metrics.routing import score_routing
from src.evaluation.metrics.task_success import score_task_success
from src.evaluation.metrics.token_efficiency import score_token_efficiency
from src.evaluation.metrics.tool_selection import score_tool_selection
from src.evaluation.metrics.verifier import score_verifier

__all__ = [
    "score_answer_relevancy",
    "score_attribution",
    "score_context_precision",
    "score_context_relevancy",
    "score_coordination",
    "score_cost_per_query",
    "score_faithfulness",
    "score_grounding",
    "score_hallucination",
    "score_latency",
    "score_mrr",
    "score_multi_agent_cost",
    "score_ndcg",
    "score_retrieval",
    "score_routing",
    "score_task_success",
    "score_token_efficiency",
    "score_tool_selection",
    "score_verifier",
]
