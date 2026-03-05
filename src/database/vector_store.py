"""
Vector similarity search against pgvector. Uses cosine distance (<=>) so
lower distance = more similar. Since embeddings are L2-normalized at
ingestion time, cosine distance and dot product give the same ranking.
"""

from dataclasses import dataclass

from loguru import logger
from sqlalchemy import Float

from src.config import settings
from src.database.connection import get_session
from src.database.models import MovieChunk


@dataclass
class SearchResult:
    movie_id: int
    title: str
    chunk_text: str
    # cosine similarity: 1.0 = identical, 0.0 = orthogonal
    similarity: float


def search(
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[SearchResult]:
    """
    Find the top-K most similar movie chunks by cosine distance.

    Takes a pre-computed query embedding (384-dim list of floats).
    Returns results sorted by similarity, highest first.
    """
    if top_k is None:
        top_k = settings.default_top_k

    session = get_session()

    try:
        # <=> is pgvector's cosine distance operator
        # returns a float between 0 (identical) and 2 (opposite)
        cosine_dist = MovieChunk.embedding.op("<=>", return_type=Float)(
            query_embedding
        ).label("distance")

        rows = (
            session.query(MovieChunk, cosine_dist)
            .order_by(cosine_dist)
            .limit(top_k)
            .all()
        )

        results = [
            SearchResult(
                movie_id=chunk.movie_id,
                title=chunk.title,
                chunk_text=chunk.chunk_text,
                # flip distance → similarity so higher = better
                similarity=round(1.0 - distance, 4),
            )
            for chunk, distance in rows
        ]

        logger.debug(
            f"search returned {len(results)} results, "
            f"top match: {results[0].title} ({results[0].similarity:.4f})"
            if results
            else "search returned 0 results"
        )

        return results

    finally:
        session.close()
