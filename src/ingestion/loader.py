"""
Load TMDB 5000 CSVs, merge them, and parse all the JSON-encoded columns
into plain Python lists/strings so downstream code doesn't have to care
about the raw format.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# all the columns that are stored as JSON strings in the CSV
_JSON_COLUMNS = [
    "genres",
    "keywords",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "cast",
    "crew",
]


def _safe_parse_json(val: Any) -> list[dict]:
    """Parse a JSON string into a list of dicts, or return [] on failure."""
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_names(records: list[dict]) -> list[str]:
    """Pull the 'name' field from each dict in a list."""
    return [r["name"] for r in records if "name" in r]


def _extract_director(crew: list[dict]) -> str | None:
    """Find the director in the crew list. Returns None if missing."""
    for member in crew:
        if member.get("job") == "Director":
            return member["name"]
    return None


def _extract_top_cast(cast: list[dict], n: int = 5) -> list[str]:
    """Top N cast members sorted by billing order."""
    sorted_cast = sorted(cast, key=lambda c: c.get("order", 999))
    return [c["name"] for c in sorted_cast[:n]]


def load_tmdb(
    movies_path: str | Path,
    credits_path: str | Path,
) -> pd.DataFrame:
    """
    Load and merge TMDB movies + credits CSVs.

    Parses all JSON columns and adds convenience columns:
      - genre_names, keyword_names, company_names: list[str]
      - director: str | None
      - top_cast: list[str] (top 5 billed actors)

    Returns a merged DataFrame with 1 row per movie.
    """
    movies_path = Path(movies_path)
    credits_path = Path(credits_path)

    logger.info(f"loading movies from {movies_path}")
    movies = pd.read_csv(movies_path)
    logger.info(f"loading credits from {credits_path}")
    credits = pd.read_csv(credits_path)

    logger.info(
        f"loaded {len(movies)} movies, {len(credits)} credits rows"
    )

    # merge on movie ID — movies uses 'id', credits uses 'movie_id'
    # credits also has a 'title' column that collides, so we suffix it
    df = pd.merge(
        movies,
        credits,
        left_on="id",
        right_on="movie_id",
        suffixes=("", "_credits"),
    )
    logger.info(f"merged into {len(df)} rows")

    # parse every JSON column in-place
    for col in _JSON_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_json)

    # pull out the useful bits into clean columns
    df["genre_names"] = df["genres"].apply(_extract_names)
    df["keyword_names"] = df["keywords"].apply(_extract_names)
    df["company_names"] = df["production_companies"].apply(_extract_names)
    df["director"] = df["crew"].apply(_extract_director)
    df["top_cast"] = df["cast"].apply(_extract_top_cast)

    # drop rows with no overview — can't build a useful chunk without one
    before = len(df)
    df = df.dropna(subset=["overview"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"dropped {dropped} rows with missing overview")

    df = df.reset_index(drop=True)
    logger.info(f"final dataset: {len(df)} movies ready for chunking")

    return df
