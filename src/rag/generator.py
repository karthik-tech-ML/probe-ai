"""
Answer generation via Claude API. Takes a question + retrieved chunks
and produces a grounded answer. The system prompt is intentionally
strict — we want the model to refuse rather than hallucinate.
"""

from dataclasses import dataclass

import anthropic
from loguru import logger

from src.config import settings
from src.database.vector_store import SearchResult

_SYSTEM_PROMPT = """\
You are a movie knowledge assistant. Your ONLY source of information is the \
context provided below. Follow these rules strictly:

1. ONLY use facts that appear in the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
say "I don't have enough information to answer that based on the available data."
3. When citing facts (director, cast, budget, etc.), stick to exactly what the \
context says. Do not embellish or infer beyond what is written.
4. If the question asks about a movie or topic not covered in the context, say so.
5. Keep answers concise and direct. No filler.
"""


def _build_context_block(chunks: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    sections = []
    for i, chunk in enumerate(chunks, 1):
        sections.append(f"[Source {i}]\n{chunk.chunk_text}")
    return "\n\n".join(sections)


@dataclass
class GenerationResult:
    answer: str
    model: str
    usage: dict  # token counts for cost tracking


def generate(
    question: str,
    chunks: list[SearchResult],
    model: str = "claude-sonnet-4-5-20250929",
) -> GenerationResult:
    """
    Call Claude with the question and retrieved context.

    Returns the answer text plus model/usage metadata for
    downstream eval and cost tracking.
    """
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set — add it to your .env file"
        )

    context = _build_context_block(chunks)

    user_message = f"""Context:
{context}

Question: {question}"""

    logger.debug(f"calling {model} with {len(chunks)} context chunks")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    logger.info(
        f"generated answer ({usage['input_tokens']}in/{usage['output_tokens']}out tokens)"
    )

    return GenerationResult(answer=answer, model=model, usage=usage)
