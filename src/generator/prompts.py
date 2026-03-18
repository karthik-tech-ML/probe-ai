"""
Prompt templates for auto-generating evaluation scenarios.

Each category gets its own prompt that tells Claude what kind of
test questions to create. The prompts receive the corpus schema
and sample documents so they can produce realistic, answerable
questions grounded in actual data.
"""

from src.connectors.base import CorpusSchema, SourceDocument


def _format_schema(schema: CorpusSchema) -> str:
    """Turn a CorpusSchema into a readable block for the prompt."""
    lines = [
        f"Domain: {schema.domain_name}",
        f"Description: {schema.description}",
        f"Total documents: {schema.total_documents}",
        "",
        "Fields:",
    ]
    for f in schema.fields:
        samples = ", ".join(f.sample_values[:3]) if f.sample_values else "n/a"
        lines.append(f"  - {f.name} ({f.field_type}): {f.description} — e.g. {samples}")
    return "\n".join(lines)


def _format_samples(docs: list[SourceDocument], max_docs: int = 5) -> str:
    """Format sample documents so the LLM can see real data."""
    parts = []
    for doc in docs[:max_docs]:
        meta = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items()) if doc.metadata else "none"
        parts.append(
            f"--- Document: {doc.title} (id: {doc.doc_id}) ---\n"
            f"Metadata: {meta}\n"
            f"Content: {doc.content[:500]}"
        )
    return "\n\n".join(parts)


_SYSTEM = """\
You are a QA engineer generating test scenarios for an AI evaluation framework. \
You write clear, specific questions that can be verified against source data. \
Every question must be answerable (or deliberately unanswerable for no_answer \
category) using the corpus described below.

Output valid JSON only — an array of scenario objects.\
"""


def _base_prompt(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    category: str,
    count: int,
    category_instructions: str,
) -> str:
    """Build the full prompt from schema + samples + category instructions."""
    return f"""{_SYSTEM}

## Corpus
{_format_schema(schema)}

## Sample Documents
{_format_samples(samples)}

## Task
Generate exactly {count} test scenarios in the "{category}" category.

{category_instructions}

## Output Format
Return a JSON array. Each element must have these fields:
- "id": string like "gen_001", "gen_002", etc.
- "question": the test question
- "expected_answer": the correct answer based on the source data
- "category": "{category}"
- "expected_source_ids": array of document IDs (as integers if numeric, otherwise use 0) that should be retrieved
- "difficulty": "easy", "medium", or "hard"
- "hallucination_risk": "low", "medium", or "high"
- "notes": brief explanation of what this scenario tests

Return ONLY the JSON array, no markdown fences or extra text."""


def prompt_simple_factual(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    count: int,
) -> str:
    return _base_prompt(
        schema, samples, "simple_factual", count,
        instructions := (
            "Generate questions that ask for a single fact from one document. "
            "These should be straightforward lookups — one field, one document, "
            "one clear answer. Use different fields across questions (don't just "
            "ask about the same field every time). Mix easy and medium difficulty."
        ),
    )


def prompt_multi_field(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    count: int,
) -> str:
    return _base_prompt(
        schema, samples, "multi_field", count,
        instructions := (
            "Generate questions that require combining information from multiple "
            "fields within one or two documents. For example: 'What genre is the "
            "2010 film directed by X?' requires matching director + year + genre. "
            "These test whether the system can cross-reference structured fields."
        ),
    )


def prompt_aggregation(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    count: int,
) -> str:
    return _base_prompt(
        schema, samples, "aggregation", count,
        instructions := (
            "Generate questions that require counting, ranking, comparing, or "
            "filtering across multiple documents. Examples: 'highest-rated X', "
            "'how many documents match Y', 'compare A vs B'. These are harder "
            "because single-document retrieval isn't enough."
        ),
    )


def prompt_no_answer(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    count: int,
) -> str:
    return _base_prompt(
        schema, samples, "no_answer", count,
        instructions := (
            "Generate questions that CANNOT be answered from this corpus. "
            "The questions should sound plausible and be on-topic for the domain, "
            "but ask about data that doesn't exist in the corpus. The expected "
            "answer should be something like 'This information is not available' "
            "or 'I don't have data on that'. Set expected_source_ids to []. "
            "These test whether the system admits uncertainty vs. hallucinating."
        ),
    )


def prompt_adversarial(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    count: int,
) -> str:
    return _base_prompt(
        schema, samples, "adversarial", count,
        instructions := (
            "Generate tricky questions designed to trip up the system. Techniques: "
            "questions with false premises ('When did X do Y?' where X never did Y), "
            "questions that sound similar to real data but aren't, ambiguous questions "
            "that could be misinterpreted, or questions that tempt the system to "
            "guess rather than cite sources. All should have high hallucination risk."
        ),
    )


def prompt_multi_hop(
    schema: CorpusSchema,
    samples: list[SourceDocument],
    count: int,
) -> str:
    return _base_prompt(
        schema, samples, "multi_hop", count,
        instructions := (
            "Generate questions that require connecting information across "
            "multiple documents to answer. The system needs to retrieve "
            "document A, learn something, then use that to find document B. "
            "Example: 'What other films did the director of X make?' requires "
            "first finding X's director, then finding their other work."
        ),
    )


# map category names to their prompt builders
CATEGORY_PROMPTS = {
    "simple_factual": prompt_simple_factual,
    "multi_field": prompt_multi_field,
    "aggregation": prompt_aggregation,
    "no_answer": prompt_no_answer,
    "adversarial": prompt_adversarial,
    "multi_hop": prompt_multi_hop,
}

ALL_CATEGORIES = list(CATEGORY_PROMPTS.keys())
