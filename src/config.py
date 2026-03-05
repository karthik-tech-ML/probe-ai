from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# .env lives at the project root, one level up from src/
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)


class Settings(BaseSettings):
    # -- database --
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/probeai",
        description="postgres connection string with pgvector extension",
    )

    # -- anthropic --
    anthropic_api_key: str = Field(default="")

    # -- embeddings --
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # -- rag --
    default_top_k: int = Field(default=5)
    latency_threshold_ms: int = Field(default=2000)

    # -- eval --
    eval_judge_model: str = Field(default="claude-sonnet-4-5-20250929")

    model_config = {"env_file": str(_env_path), "env_file_encoding": "utf-8"}


# single instance everything imports from
settings = Settings()
