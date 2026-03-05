"""
Turn each movie row into a single composite text chunk that combines
structured metadata + plot overview. One chunk per movie — no sub-chunking
needed since overviews are short enough to fit in one embedding.
"""

import pandas as pd
from loguru import logger


def _format_budget_revenue(budget: float, revenue: float) -> str:
    """Format budget/revenue as dollar amounts, skip zeros (means unknown)."""
    parts = []
    if budget and budget > 0:
        parts.append(f"Budget: ${budget:,.0f}")
    if revenue and revenue > 0:
        parts.append(f"Revenue: ${revenue:,.0f}")
    return " | ".join(parts) if parts else ""


def _format_year(release_date: str | None) -> str:
    """Pull the year out of a YYYY-MM-DD date string."""
    if pd.isna(release_date) or not release_date:
        return "Unknown"
    return release_date[:4]


def build_chunk(row: pd.Series) -> str:
    """
    Build a single text chunk from a movie row. Follows the composite
    format from CLAUDE.md so all movie info lives in one retrievable unit.
    """
    lines = [f"Title: {row['title']}"]

    if row.get("director"):
        lines.append(f"Director: {row['director']}")

    if row.get("top_cast"):
        lines.append(f"Cast: {', '.join(row['top_cast'])}")

    if row.get("genre_names"):
        lines.append(f"Genres: {', '.join(row['genre_names'])}")

    lines.append(f"Release Year: {_format_year(row.get('release_date'))}")

    budget_rev = _format_budget_revenue(
        row.get("budget", 0), row.get("revenue", 0)
    )
    if budget_rev:
        lines.append(budget_rev)

    runtime = row.get("runtime")
    if pd.notna(runtime) and runtime:
        lines.append(f"Runtime: {int(runtime)} minutes")

    vote_avg = row.get("vote_average")
    vote_count = row.get("vote_count")
    if pd.notna(vote_avg) and pd.notna(vote_count):
        lines.append(f"Rating: {vote_avg}/10 ({int(vote_count)} votes)")

    if row.get("keyword_names"):
        lines.append(f"Keywords: {', '.join(row['keyword_names'])}")

    if row.get("company_names"):
        lines.append(f"Production: {', '.join(row['company_names'])}")

    overview = row.get("overview", "")
    if pd.notna(overview) and overview:
        lines.append(f"Plot: {overview}")

    return "\n".join(lines)


def build_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Build chunks for every movie in the DataFrame.

    Returns a list of dicts with:
      - movie_id: int (TMDB id)
      - title: str
      - chunk_text: str (the composite document)
    """
    chunks = []
    for _, row in df.iterrows():
        chunks.append(
            {
                "movie_id": int(row["id"]),
                "title": row["title"],
                "chunk_text": build_chunk(row),
            }
        )

    logger.info(f"built {len(chunks)} chunks")
    return chunks
