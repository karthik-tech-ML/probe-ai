from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase

# all-MiniLM-L6-v2 produces 384-dim vectors
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class MovieChunk(Base):
    """One row per movie — stores the composite text chunk + its embedding."""

    __tablename__ = "movie_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))

    def __repr__(self) -> str:
        return f"<MovieChunk movie_id={self.movie_id} title={self.title!r}>"
