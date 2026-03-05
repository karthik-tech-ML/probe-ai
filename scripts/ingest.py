"""
Full ingestion pipeline: load CSVs → build chunks → embed → store in pgvector.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --movies data/raw/tmdb_5000_movies.csv --credits data/raw/tmdb_5000_credits.csv
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

# make sure project root is on the path when running as a script
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.database.connection import get_session, init_db
from src.database.models import MovieChunk
from src.ingestion.chunker import build_chunks
from src.ingestion.embedder import embed_chunks
from src.ingestion.loader import load_tmdb

# default paths relative to project root
DEFAULT_MOVIES = _root / "data" / "raw" / "tmdb_5000_movies.csv"
DEFAULT_CREDITS = _root / "data" / "raw" / "tmdb_5000_credits.csv"


def store_chunks(chunks: list[dict], batch_size: int = 500) -> int:
    """
    Upsert chunks into the movie_chunks table. Uses movie_id to
    avoid duplicates on re-runs — existing rows get updated.
    """
    session = get_session()
    stored = 0

    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            for chunk in batch:
                # check if this movie already has a row
                existing = (
                    session.query(MovieChunk)
                    .filter_by(movie_id=chunk["movie_id"])
                    .first()
                )

                if existing:
                    existing.title = chunk["title"]
                    existing.chunk_text = chunk["chunk_text"]
                    existing.embedding = chunk["embedding"]
                else:
                    session.add(
                        MovieChunk(
                            movie_id=chunk["movie_id"],
                            title=chunk["title"],
                            chunk_text=chunk["chunk_text"],
                            embedding=chunk["embedding"],
                        )
                    )
                stored += 1

            session.commit()
            logger.info(f"committed batch {i // batch_size + 1} ({stored}/{len(chunks)})")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest TMDB data into pgvector")
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    parser.add_argument("--credits", type=Path, default=DEFAULT_CREDITS)
    args = parser.parse_args()

    t0 = time.perf_counter()

    # 1. set up the DB (creates extension + tables if needed)
    logger.info("initializing database")
    init_db()

    # 2. load and merge CSVs
    df = load_tmdb(args.movies, args.credits)

    # 3. build composite text chunks
    chunks = build_chunks(df)

    # 4. embed all chunks
    logger.info("embedding chunks (this takes a minute on CPU)")
    chunks = embed_chunks(chunks, batch_size=64)

    # 5. store in pgvector
    logger.info("storing chunks in pgvector")
    count = store_chunks(chunks)

    elapsed = time.perf_counter() - t0
    logger.info(f"done — {count} movies ingested in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
