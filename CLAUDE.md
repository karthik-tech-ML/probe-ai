# ProbeAI

## Project Overview
ProbeAI is an evaluation and testing framework for RAG and AI agent systems. It probes AI systems for hallucinations, grounding failures, safety vulnerabilities, and quality regressions. The project has two layers:
1. **System Under Test**: A RAG pipeline + tool-using agent built on the TMDB 5000 movie dataset
2. **Testing Framework**: Evaluation, safety, and benchmarking infrastructure that validates the system

This is a portfolio project demonstrating AI quality engineering skills, and will be presented at a conference talk in ~8 weeks.

## Tech Stack
- **Language**: Python 3.11+
- **API Framework**: FastAPI
- **Database**: PostgreSQL + pgvector
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude API (anthropic SDK) for answer generation
- **Agent Framework**: LangGraph (weeks 3-4)
- **Eval Libraries**: RAGAS, DeepEval as baselines (extend with custom metrics)
- **Testing**: pytest

## Architecture

### System Under Test
```
1. RAG System
   - PostgreSQL + pgvector for vector storage
   - sentence-transformers for embedding
   - Cosine similarity retrieval with top-K
   - Claude API for answer generation

2. Agent System (weeks 3-4)
   - LangGraph tool-using agent
   - Multi-step task execution
   - Memory / conversation state
   - Failure recovery and error handling flows
```

### Testing Framework
```
3. Scenario Library
   - Unified test case repository (scenarios/library.json)
   - Categories: simple_factual, multi_field, aggregation, 
     no_answer, adversarial, multi_hop
   - Reusable across all evaluation layers

4. Evaluation Layer
   - Hallucination scoring
   - Grounding detection (trace claims → source chunks)
   - Faithfulness scoring
   - Retrieval Recall@K and Precision
   - Task success scoring
   - Tool selection accuracy (agent layer)
   - Latency threshold pass/fail

5. Safety & Guardrail Suite (weeks 5-6)
   - Prompt injection tests
   - Tool misuse tests
   - Red team prompts
   - Output format validation
   - Topic boundary enforcement
   - PII leakage checks

6. Inference Benchmark (week 5)
   - Local vs cloud comparison (Ollama vs Claude API)
   - Latency measurement
   - Cost per query
   - Quality parity scoring
```

## Project Structure
```
probeai/
├── CLAUDE.md                    # This file
├── README.md                    # Public-facing documentation
├── pyproject.toml               # Dependencies
├── .env.example                 # Environment variables template
├── data/
│   ├── raw/                     # TMDB CSV files (gitignored)
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/               # Cleaned/merged data
├── src/
│   ├── __init__.py
│   ├── config.py                # Settings, env vars, constants
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py            # Load and parse TMDB CSVs
│   │   ├── chunker.py           # Build composite document chunks
│   │   └── embedder.py          # Generate embeddings
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy/pgvector models
│   │   ├── connection.py        # DB connection management
│   │   └── vector_store.py      # Vector search operations
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Query → embed → top-K retrieval
│   │   ├── generator.py         # Claude API answer generation
│   │   └── pipeline.py          # End-to-end RAG pipeline
│   ├── agent/                   # Weeks 3-4
│   │   ├── __init__.py
│   │   ├── tools.py             # Agent tool definitions
│   │   ├── graph.py             # LangGraph agent definition
│   │   └── memory.py            # Conversation state
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── runner.py            # Eval harness — runs scenarios, collects results
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py     # Recall@K, Precision
│   │   │   ├── faithfulness.py  # LLM-as-judge faithfulness
│   │   │   ├── grounding.py     # Claim → source traceability
│   │   │   ├── hallucination.py # Hallucination detection scoring
│   │   │   └── latency.py       # Latency threshold checks
│   │   └── reports.py           # Generate eval result summaries
│   ├── safety/                  # Weeks 5-6
│   │   ├── __init__.py
│   │   ├── injection.py         # Prompt injection tests
│   │   ├── guardrails.py        # Topic boundary, format, PII checks
│   │   └── redteam.py           # Red team prompt suite
│   └── benchmark/               # Week 5
│       ├── __init__.py
│       ├── inference.py         # Local vs cloud comparison
│       └── cost.py              # Cost per query tracking
├── scenarios/
│   ├── library.json             # Master scenario library
│   └── schema.md                # Scenario schema documentation
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI endpoints
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
└── scripts/
    ├── ingest.py                # CLI script to run data ingestion
    ├── evaluate.py              # CLI script to run eval suite
    └── benchmark.py             # CLI script to run benchmarks
```

## Data: TMDB 5000 Movie Dataset
Two CSV files from Kaggle:
- `tmdb_5000_movies.csv`: budget, genres (JSON), homepage, id, keywords (JSON), original_title, overview, popularity, production_companies (JSON), production_countries (JSON), release_date, revenue, runtime, spoken_languages, status, tagline, vote_average, vote_count
- `tmdb_5000_credits.csv`: movie_id, title, cast (JSON), crew (JSON)

### Composite Document Chunk Format
Each movie becomes one document chunk combining structured + unstructured data:
```
Title: Inception
Director: Christopher Nolan
Cast: Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy, Ken Watanabe
Genres: Action, Science Fiction, Adventure
Release Year: 2010
Budget: $160,000,000 | Revenue: $835,532,764
Runtime: 148 minutes
Rating: 8.1/10 (14075 votes)
Keywords: dream, subconscious, heist, mission, sleep
Production: Warner Bros., Legendary Pictures, Syncopy
Plot: A thief who steals corporate secrets through the use of dream-sharing technology...
```

### Ground Truth Strategy
Structured metadata fields (director, cast, genres, budget, revenue) serve as verifiable ground truth for eval scenarios. If the RAG system says "Inception was directed by Christopher Nolan" — we can validate that against the source data.

## Key Design Decisions
- **Document chunks = 1 per movie**: No sub-chunking needed since each movie overview is short enough. The composite format keeps all movie info in one retrievable unit.
- **Embedding model**: all-MiniLM-L6-v2 locally via sentence-transformers. Free, fast, good enough for the eval framework demo. Can swap in OpenAI embeddings later for inference benchmark comparison.
- **LLM-as-judge for faithfulness**: Use a second Claude API call to judge whether generated answers are supported by retrieved context. Cheaper and faster than fine-tuned classifiers for this use case.
- **Scenario-driven evaluation**: All eval runs pull from the scenario library. No ad-hoc testing.

## Development Phases
- **Weeks 1-2**: RAG pipeline + initial eval metrics (retrieval recall, faithfulness, grounding)
- **Weeks 3-4**: Agent system (LangGraph) + agent eval metrics (task success, tool selection)
- **Week 5**: Inference benchmark (local vs cloud) + safety/guardrail suite
- **Week 6**: Polish framework, dashboard/CLI visualization, public repo
- **Week 7**: Conference talk slides, narrative, dry run
- **Week 8**: Refine, rehearse, buffer

## Environment Variables
```
DATABASE_URL=postgresql://user:pass@localhost:5432/probeai
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_TOP_K=5
LATENCY_THRESHOLD_MS=2000
```

## Commands
```bash
# Ingest TMDB data
python scripts/ingest.py

# Run evaluation suite
python scripts/evaluate.py

# Run specific eval category
python scripts/evaluate.py --category simple_factual

# Run inference benchmark
python scripts/benchmark.py

# Start API server
uvicorn api.main:app --reload
```

## Code Style
- Type hints everywhere
- Docstrings on public functions
- Async where it makes sense (FastAPI endpoints, API calls)
- Structured logging with loguru
- Results always output as structured JSON
