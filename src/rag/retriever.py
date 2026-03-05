"""
Question → embed → vector search → ranked results.

This is the glue between the user's natural language query and the
vector store. Uses the same embedding model as ingestion so the
vectors live in the same space.
"""

from dataclasses import dataclass

from loguru import logger

from src.config import settings
from src.database.vector_store import SearchResult, search
from src.ingestion.embedder import embed_texts


@dataclass
class RetrievalResult:
    query: str
    results: list[SearchResult]


def retrieve(
    query: str,
    top_k: int | None = None,
) -> RetrievalResult:
    """
    Embed a question and find the most relevant movie chunks.

    Returns a RetrievalResult with the original query and ranked
    SearchResults (highest similarity first).
    """
    if top_k is None:
        top_k = settings.default_top_k

    logger.debug(f"retrieving top-{top_k} for: {query!r}")

    # embed with the same model used at ingestion time
    query_vec = embed_texts([query])[0].tolist()

    results = search(query_vec, top_k=top_k)

    logger.info(
        f"retrieved {len(results)} chunks for query "
        f"(top: {results[0].title}, sim={results[0].similarity})"
        if results
        else f"no results for query: {query!r}"
    )

    return RetrievalResult(query=query, results=results)


def _extract_title_hint(query: str) -> str | None:
    """
    Try to pull a movie title out of the question. Not bulletproof —
    just needs to work for the common "What is the X of MovieName?"
    patterns. Returns None if we can't find one.
    """
    import re

    patterns = [
        # "plot of Avatar", "budget of The Dark Knight"
        r"(?:plot|story|budget|revenue|runtime|rating|genre|cast|director|overview|synopsis)"
        r"\s+(?:of|for)\s+(?:the\s+(?:movie|film)\s+)?(.+?)[\?\.\!]?\s*$",
        # "about the movie Avatar"
        r"(?:about|regarding)\s+(?:the\s+)?(?:movie|film)\s+(.+?)[\?\.\!]?\s*$",
        # "movie called Avatar"
        r"(?:movie|film)\s+(?:called|named|titled)\s+(.+?)[\?\.\!]?\s*$",
    ]
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            title = m.group(1).strip().strip("\"'")
            if len(title) > 1:
                return title
    return None


def retrieve_with_title_boost(
    query: str,
    top_k: int | None = None,
) -> RetrievalResult:
    """
    Same as retrieve(), but reshapes the query to match the chunk
    format before embedding. When we can extract a movie title from
    the question, we prepend "Title: X" and the chunk field headers
    so the embedding lands closer to the actual movie chunk.

    This fixes cases where a generic title like "Avatar" drifts
    toward thematically similar movies instead of the exact match.
    """
    if top_k is None:
        top_k = settings.default_top_k

    title_hint = _extract_title_hint(query)

    if title_hint:
        # mirror the chunk template so the embedding sits in the
        # same region of vector space as the ingested document
        boosted_query = (
            f"Title: {title_hint}\n"
            f"Director: \nCast: \nGenres: \n"
            f"Plot: {query}"
        )
        logger.debug(f"title boost: extracted {title_hint!r} from query")
    else:
        # no title found — fall back to plain embedding
        boosted_query = query
        logger.debug("title boost: no title extracted, using original query")

    query_vec = embed_texts([boosted_query])[0].tolist()
    results = search(query_vec, top_k=top_k)

    logger.info(
        f"retrieved (boosted) {len(results)} chunks for query "
        f"(top: {results[0].title}, sim={results[0].similarity})"
        if results
        else f"no results for query: {query!r}"
    )

    return RetrievalResult(query=query, results=results)
