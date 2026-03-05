"""
Quick sanity check on the ingested data.
Run after scripts/ingest.py to verify everything landed correctly.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func

from src.database.connection import get_session
from src.database.models import EMBEDDING_DIM, MovieChunk

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def main() -> None:
    session = get_session()

    # 1. total row count
    total = session.query(func.count(MovieChunk.id)).scalar()
    check("row count", total == 4800, f"{total} rows")

    # 2. three random movies
    print("\n  --- sample chunks ---")
    all_ids = [r[0] for r in session.query(MovieChunk.id).all()]
    sample_ids = random.sample(all_ids, min(3, len(all_ids)))
    for sid in sample_ids:
        row = session.get(MovieChunk, sid)
        print(f"\n  [{row.title}]")
        # first 3 lines of the chunk so we don't flood the terminal
        for line in row.chunk_text.split("\n")[:3]:
            print(f"    {line}")
        print("    ...")
    print()

    # 3. embedding dimensions
    sample = session.query(MovieChunk).first()
    emb_len = len(sample.embedding) if sample is not None and sample.embedding is not None else 0
    check("embedding dim", emb_len == EMBEDDING_DIM, f"{emb_len}")

    # 4. null embeddings
    null_embeddings = (
        session.query(func.count(MovieChunk.id))
        .filter(MovieChunk.embedding.is_(None))
        .scalar()
    )
    check("no null embeddings", null_embeddings == 0, f"{null_embeddings} nulls")

    # 5. empty chunk texts
    empty_chunks = (
        session.query(func.count(MovieChunk.id))
        .filter((MovieChunk.chunk_text.is_(None)) | (MovieChunk.chunk_text == ""))
        .scalar()
    )
    check("no empty chunks", empty_chunks == 0, f"{empty_chunks} empty")

    # 6. Inception has Christopher Nolan
    inception = (
        session.query(MovieChunk).filter_by(movie_id=27205).first()
    )
    has_nolan = inception is not None and "Christopher Nolan" in inception.chunk_text
    check(
        "Inception → Christopher Nolan",
        has_nolan,
        "found" if has_nolan else "missing",
    )

    session.close()

    # summary
    passed = sum(1 for _, p, _ in results if p)
    total_checks = len(results)
    print(f"\n  {passed}/{total_checks} checks passed", end="")
    if passed == total_checks:
        print(" ✓")
    else:
        print(" ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
