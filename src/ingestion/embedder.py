"""
Embed chunk texts using sentence-transformers. Loads the model once
and reuses it — the model is small enough (~80MB) to keep in memory.
"""

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load so we don't pay startup cost on import."""
    global _model
    if _model is None:
        logger.info(f"loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings into vectors.

    Returns an (N, 384) numpy array of float32 embeddings.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        normalize_embeddings=True,
    )
    logger.info(f"embedded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Takes chunks from chunker.build_chunks() and adds an 'embedding'
    field (list[float]) to each one. Returns the same list, mutated.
    """
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_texts(texts, batch_size=batch_size)

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec.tolist()

    return chunks
