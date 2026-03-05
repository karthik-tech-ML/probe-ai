"""
Agent tools for structured movie queries.

These solve the multi_field and aggregation retrieval problem:
single-shot vector search can't handle "all movies by director X
in genre Y" because it only returns top-K nearest neighbors.
These tools let the agent do precise filtering against the
actual data stored in chunk_text.

All structured queries parse fields from chunk_text since
that's our only data store — we didn't create separate SQL
columns for director, genre, etc.
"""

import re
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from src.database.connection import get_session
from src.database.models import MovieChunk
from src.rag.retriever import retrieve


def _parse_chunk_field(chunk_text: str, field: str) -> str:
    """Pull a field value from the chunk text (e.g. 'Director: Christopher Nolan')."""
    pattern = rf"^{field}:\s*(.+)$"
    m = re.search(pattern, chunk_text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_budget_revenue(chunk_text: str) -> dict[str, int]:
    """Extract budget and revenue as ints from the chunk format."""
    result = {"budget": 0, "revenue": 0}
    # format: "Budget: $160,000,000 | Revenue: $825,532,764"
    # or just "Budget: $160,000,000" or "Revenue: $825,532,764"
    budget_m = re.search(r"Budget:\s*\$([0-9,]+)", chunk_text)
    revenue_m = re.search(r"Revenue:\s*\$([0-9,]+)", chunk_text)
    if budget_m:
        result["budget"] = int(budget_m.group(1).replace(",", ""))
    if revenue_m:
        result["revenue"] = int(revenue_m.group(1).replace(",", ""))
    return result


def _parse_rating(chunk_text: str) -> float:
    """Extract rating as a float from 'Rating: 8.1/10 (13752 votes)'."""
    m = re.search(r"Rating:\s*([0-9.]+)/10", chunk_text)
    return float(m.group(1)) if m else 0.0


def _parse_year(chunk_text: str) -> int:
    """Extract release year as an int."""
    m = re.search(r"Release Year:\s*(\d{4})", chunk_text)
    return int(m.group(1)) if m else 0


def _chunk_to_summary(chunk: MovieChunk) -> dict[str, Any]:
    """Turn a MovieChunk into a compact dict the agent can reason over."""
    financials = _parse_budget_revenue(chunk.chunk_text)
    return {
        "movie_id": chunk.movie_id,
        "title": chunk.title,
        "director": _parse_chunk_field(chunk.chunk_text, "Director"),
        "cast": _parse_chunk_field(chunk.chunk_text, "Cast"),
        "genres": _parse_chunk_field(chunk.chunk_text, "Genres"),
        "release_year": _parse_year(chunk.chunk_text),
        "budget": financials["budget"],
        "revenue": financials["revenue"],
        "rating": _parse_rating(chunk.chunk_text),
    }


@tool
def search_movies(query: str) -> list[dict]:
    """Semantic search for movies. Use this when you need to find movies
    by topic, plot description, or general concept. Returns top 5 results
    ranked by relevance."""
    logger.debug(f"tool:search_movies query={query!r}")
    result = retrieve(query, top_k=5)
    return [
        {
            "movie_id": r.movie_id,
            "title": r.title,
            "similarity": r.similarity,
            "chunk_text": r.chunk_text,
        }
        for r in result.results
    ]


@tool
def search_by_director(director_name: str) -> list[dict]:
    """Find all movies by a specific director. Use this when the question
    asks about a director's filmography or movies directed by someone."""
    logger.debug(f"tool:search_by_director name={director_name!r}")
    session = get_session()
    try:
        # case-insensitive search within chunk_text
        rows = (
            session.query(MovieChunk)
            .filter(MovieChunk.chunk_text.ilike(f"%Director: %{director_name}%"))
            .all()
        )
        results = [_chunk_to_summary(r) for r in rows]
        logger.debug(f"tool:search_by_director found {len(results)} movies")
        return results
    finally:
        session.close()


@tool
def search_by_genre(genre: str) -> list[dict]:
    """Find movies in a specific genre. Returns up to 30 results sorted by
    rating (highest first). For broader filtering, use filter_movies instead
    which supports sorting and multi-field constraints."""
    logger.debug(f"tool:search_by_genre genre={genre!r}")
    session = get_session()
    try:
        rows = (
            session.query(MovieChunk)
            .filter(MovieChunk.chunk_text.ilike(f"%Genres:%{genre}%"))
            .all()
        )
        results = [_chunk_to_summary(r) for r in rows]
        # sort by rating descending and cap to avoid context blowups
        results.sort(key=lambda m: m.get("rating", 0), reverse=True)
        total = len(results)
        results = results[:30]
        logger.debug(f"tool:search_by_genre found {total} movies, returning {len(results)}")
        return results
    finally:
        session.close()


@tool
def search_by_cast(actor_name: str) -> list[dict]:
    """Find all movies featuring a specific actor. Use this when the
    question asks about an actor's filmography or movies they appeared in."""
    logger.debug(f"tool:search_by_cast actor={actor_name!r}")
    session = get_session()
    try:
        rows = (
            session.query(MovieChunk)
            .filter(MovieChunk.chunk_text.ilike(f"%Cast:%{actor_name}%"))
            .all()
        )
        results = [_chunk_to_summary(r) for r in rows]
        logger.debug(f"tool:search_by_cast found {len(results)} movies")
        return results
    finally:
        session.close()


@tool
def filter_movies(
    genre: str | None = None,
    min_budget: int | None = None,
    max_budget: int | None = None,
    min_revenue: int | None = None,
    min_rating: float | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    sort_by: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Flexible filter across multiple movie fields at once. Use this when
    the question has multiple constraints like 'comedy movies after 2010
    with rating above 7'. Pass only the filters you need.

    sort_by: optional field to sort results by. Options: 'revenue', 'budget',
        'rating', 'year'. Prefix with '-' for descending (default), e.g. '-revenue'
        sorts highest first. Without prefix sorts ascending.
    limit: max number of results to return (default 20, max 50). Use with sort_by
        to get "top N" results efficiently."""
    logger.debug(
        f"tool:filter_movies genre={genre} budget={min_budget}-{max_budget} "
        f"rev>={min_revenue} rating>={min_rating} year={year_from}-{year_to} "
        f"sort={sort_by} limit={limit}"
    )
    # clamp limit to avoid context window blowups
    limit = min(limit, 50)

    session = get_session()
    try:
        query = session.query(MovieChunk)

        # start with genre filter in SQL if provided (narrows the scan)
        if genre:
            query = query.filter(MovieChunk.chunk_text.ilike(f"%Genres:%{genre}%"))

        rows = query.all()

        # apply the numeric filters in Python (parsed from chunk_text)
        results = []
        for row in rows:
            financials = _parse_budget_revenue(row.chunk_text)
            rating = _parse_rating(row.chunk_text)
            year = _parse_year(row.chunk_text)

            if min_budget is not None and financials["budget"] < min_budget:
                continue
            if max_budget is not None and financials["budget"] > max_budget:
                continue
            if min_revenue is not None and financials["revenue"] < min_revenue:
                continue
            if min_rating is not None and rating < min_rating:
                continue
            if year_from is not None and year < year_from:
                continue
            if year_to is not None and year > year_to:
                continue

            results.append(_chunk_to_summary(row))

        # sorting — lets the agent ask for "top N by revenue" etc.
        if sort_by:
            descending = sort_by.startswith("-")
            field = sort_by.lstrip("-")
            sort_keys = {
                "revenue": lambda m: m.get("revenue", 0),
                "budget": lambda m: m.get("budget", 0),
                "rating": lambda m: m.get("rating", 0),
                "year": lambda m: m.get("release_year", 0),
            }
            if field in sort_keys:
                results.sort(key=sort_keys[field], reverse=descending)

        total_matched = len(results)
        results = results[:limit]

        logger.debug(
            f"tool:filter_movies matched {total_matched} movies, "
            f"returning {len(results)}"
        )

        # tell the agent if results were truncated so it knows there's more
        if total_matched > limit:
            return results + [
                {"_note": f"showing {limit} of {total_matched} matches — add more filters or adjust limit"}
            ]
        return results
    finally:
        session.close()


@tool
def get_movie_details(tmdb_id: int) -> dict:
    """Get full details for a specific movie by its TMDB ID. Use this
    when you already know the movie ID and need its complete info."""
    logger.debug(f"tool:get_movie_details id={tmdb_id}")
    session = get_session()
    try:
        row = session.query(MovieChunk).filter_by(movie_id=tmdb_id).first()
        if not row:
            return {"error": f"no movie found with tmdb_id={tmdb_id}"}
        return {
            "movie_id": row.movie_id,
            "title": row.title,
            "chunk_text": row.chunk_text,
        }
    finally:
        session.close()


# all tools in one list for the agent to use
ALL_TOOLS = [
    search_movies,
    search_by_director,
    search_by_genre,
    search_by_cast,
    filter_movies,
    get_movie_details,
]
