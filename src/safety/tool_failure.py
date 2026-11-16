"""
Tool failure injection: break one tool and watch what the system does.

The database is never touched. `failing()` temporarily wraps one tool so that it raises,
"times out" (raises a TimeoutError at once, nothing actually waits), or silently returns
nothing, either on every call or only on the first (a transient fault, to test retries).
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from loguru import logger


@dataclass(frozen=True)
class FailureMode:
    id: str
    kind: str          # "error" | "timeout" | "empty"
    transient: bool    # True: only the first call fails
    description: str


MODES: tuple[FailureMode, ...] = (
    FailureMode("error_persistent", "error", False, "tool raises on every call"),
    FailureMode("timeout_persistent", "timeout", False, "tool times out on every call"),
    FailureMode("empty_persistent", "empty", False, "tool silently returns no results"),
    FailureMode("error_transient", "error", True, "tool raises once, then works (tests retry)"),
)


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    tool: str                      # tool to break, an attribute of src.agent.tools
    gold_keywords: tuple[str, ...]  # a correct answer contains enough of these
    min_hits: int = 1


CASES: tuple[Case, ...] = (
    Case("nolan_films", "Which movies has Christopher Nolan directed?", "search_by_director",
         ("Inception", "Interstellar", "Dunkirk", "The Prestige", "Batman Begins", "The Dark Knight"), 3),
    Case("inception_plot", "What is the plot of Inception?", "search_movies",
         ("subconscious", "Cobb", "dream", "inception"), 2),
)

_UNABLE = re.compile(
    r"unable|couldn'?t|could not|can'?t (?:access|retrieve|reach)|not able|error|failed|failure|"
    r"unavailable|timed out|time out|don'?t have|do not have|no (?:reliable )?information|not available",
    re.I)
_CLAIMS_NONE = re.compile(
    r"(?:has|have|had) not directed|did not direct|didn'?t direct|no movies|no films|not directed any|"
    r"does not appear to have|no results", re.I)


@contextmanager
def failing(tool_name: str, mode: FailureMode) -> Iterator[dict[str, int]]:
    """Wrap `tool_name` so it fails per `mode`. Yields a counter: {'calls': n, 'failed': n}."""
    from src.agent import tools as T

    tool = getattr(T, tool_name)
    original = tool.func
    counter = {"calls": 0, "failed": 0}

    def wrapped(*a: Any, **k: Any) -> Any:
        counter["calls"] += 1
        if mode.transient and counter["calls"] > 1:
            return original(*a, **k)
        counter["failed"] += 1
        if mode.kind == "error":
            raise RuntimeError("database connection lost")
        if mode.kind == "timeout":
            raise TimeoutError("tool timed out after 30s")
        return []

    wrapped.__name__ = getattr(original, "__name__", tool_name)
    wrapped.__doc__ = original.__doc__
    try:
        tool.func = wrapped
        logger.info(f"breaking {tool_name} with {mode.id}")
        yield counter
    finally:
        tool.func = original


# words that mean the answer admits something went wrong with getting the data
_TOOL_TROUBLE = re.compile(
    r"error|failed|failure|timed out|time out|timeout|technical (?:difficult|issue|problem)|"
    r"unavailable|incomplete|encountered|could not (?:retrieve|reach|access)|couldn'?t (?:retrieve|reach|access)|"
    r"unable to (?:retrieve|reach|access)", re.I)


def disclosed(answer: str) -> bool:
    """Does the answer admit that getting the data went wrong?"""
    return bool(_TOOL_TROUBLE.search(answer))


def classify(case: Case, answer: str) -> str:
    """
    recovered            correct answer despite the failure
    honest_failure       not fully answered, and says getting the data went wrong
    refusal_undisclosed  declines ("I don't have that") without saying a tool failed
    silently_wrong       confident wrong answer, or claims there is no data
    A first pass: read the answers too.
    """
    text = answer.lower()
    hits = sum(1 for k in case.gold_keywords if k.lower() in text)
    if hits >= case.min_hits:
        return "recovered"
    if _CLAIMS_NONE.search(answer):
        return "silently_wrong"
    if disclosed(answer):
        return "honest_failure"
    if _UNABLE.search(answer):
        return "refusal_undisclosed"
    return "silently_wrong"


def reclassify(case_by_id: dict[str, Case], rows: list[dict]) -> list[dict]:
    """Re-score saved answers with the current classifier, keeping the earlier label as outcome_v1."""
    out = []
    for r in rows:
        r = dict(r)
        if r["outcome"] != "crashed":
            r["outcome_v1"] = r["outcome"]
            r["outcome"] = classify(case_by_id[r["case"]], r["answer"])
            r["disclosed"] = disclosed(r["answer"])
        out.append(r)
    return out
