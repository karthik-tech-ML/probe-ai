"""
ProbeConnector — the interface any AI system implements to get evaluated.

Connectors wrap a RAG pipeline, agent, or any question-answering system
so ProbeAI can probe it for hallucinations, grounding failures, and
quality regressions without knowing anything about the system's internals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class SourceDocument:
    """
    A single document returned by the system under test.

    Uses string IDs because not every system has integer PKs —
    could be UUIDs, slugs, whatever the domain uses.
    """

    doc_id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0


@dataclass
class ProbeResult:
    """
    What the connector returns after answering a question.

    This is the universal format — the adapter layer converts it
    to PipelineResult so existing metrics still work unchanged.
    """

    question: str
    answer: str
    source_documents: list[SourceDocument]
    context_texts: list[str]
    latency_ms: float
    token_usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    # only populated if the connector wraps an agent with tool calls
    agent_trace: Any | None = None


@dataclass
class FieldInfo:
    """Describes one field in the corpus — used by the scenario generator."""

    name: str
    field_type: str  # "string", "number", "date", "list"
    description: str
    sample_values: list[str] = field(default_factory=list)


@dataclass
class CorpusSchema:
    """
    Describes what's in the corpus so the scenario generator knows
    what kinds of questions to produce.
    """

    domain_name: str
    description: str
    fields: list[FieldInfo]
    total_documents: int


class ProbeConnector(ABC):
    """
    Base class for all connectors. Implement this to plug your
    RAG/agent system into ProbeAI's eval framework.

    Three methods to implement:
    - ask(): send a question, get back an answer + sources
    - get_corpus_sample(): return a handful of documents for scenario generation
    - get_schema(): describe your data so we can auto-generate test scenarios
    """

    @abstractmethod
    def ask(self, question: str, mode: str = "rag") -> ProbeResult:
        """
        Ask the system a question and get a structured result.

        Args:
            question: the natural language query
            mode: "rag" for retrieval-augmented generation,
                  "agent" for tool-using agent mode.
                  Connectors that only support one mode can
                  ignore this or raise ValueError.

        Returns:
            ProbeResult with the answer, sources, and metadata.
        """
        ...

    @abstractmethod
    def get_corpus_sample(self, n: int = 10) -> list[SourceDocument]:
        """
        Return a random sample of documents from the corpus.

        Used by the scenario generator to understand what's in the
        data and create realistic test questions. Should return
        documents with full content and metadata populated.
        """
        ...

    @abstractmethod
    def get_schema(self) -> CorpusSchema:
        """
        Describe the corpus structure — field names, types, and
        a few example values for each.

        The scenario generator uses this to create category-appropriate
        questions (e.g., it needs to know there's a "director" field
        to generate multi_field queries like "sci-fi films by Nolan").
        """
        ...

    def health_check(self) -> bool:
        """
        Quick sanity check that the connector is wired up correctly.

        Default implementation sends a simple question through ask().
        Override if you need something different.
        """
        try:
            result = self.ask("test query", mode="rag")
            return bool(result.answer)
        except Exception as e:
            logger.warning(f"health check failed: {e}")
            return False
