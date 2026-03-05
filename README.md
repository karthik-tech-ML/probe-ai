# ProbeAI

> Probe AI systems for hallucinations, grounding failures, and quality regressions.

## What is this?

ProbeAI is **not** another RAG application. It's the **testing infrastructure** that validates one.

The project builds a movie knowledge RAG pipeline (using the TMDB 5000 dataset) as the system under test, then wraps it with a comprehensive evaluation, safety, and benchmarking framework — demonstrating what production-grade AI quality engineering looks like.

## Why?

Most teams evaluate AI outputs by eyeballing them. This project answers: *"How do you systematically test AI systems the way we test traditional software?"*

## Architecture

```
┌─────────────────────────────────────────────────┐
│              SYSTEM UNDER TEST                   │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │  RAG Pipeline │    │  Tool-Using Agent    │   │
│  │  pgvector     │    │  LangGraph           │   │
│  │  Claude API   │    │  Multi-step tasks    │   │
│  └──────────────┘    └──────────────────────┘   │
├─────────────────────────────────────────────────┤
│              TESTING FRAMEWORK                   │
│  ┌────────────┐ ┌───────────┐ ┌──────────────┐ │
│  │ Scenario   │ │ Eval      │ │ Safety &     │ │
│  │ Library    │ │ Layer     │ │ Guardrails   │ │
│  │            │ │           │ │              │ │
│  │ 30+ test   │ │ Recall@K  │ │ Injection    │ │
│  │ scenarios  │ │ Grounding │ │ Red team     │ │
│  │ 6 category │ │ Faithful  │ │ PII checks   │ │
│  │ types      │ │ Halluc.   │ │ Guardrails   │ │
│  └────────────┘ └───────────┘ └──────────────┘ │
│  ┌──────────────────────────────────────────┐   │
│  │         Inference Benchmark              │   │
│  │   Local vs Cloud · Latency · Cost        │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Tech Stack

- **Python** · FastAPI · PostgreSQL + pgvector
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude API (Anthropic)
- **Agent**: LangGraph
- **Eval**: Custom metrics + RAGAS/DeepEval baselines
- **Data**: TMDB 5000 Movie Dataset

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/probeai.git
cd probeai
pip install -e .

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and database URL

# Ingest TMDB data
python scripts/ingest.py

# Run evaluation suite
python scripts/evaluate.py

# Run specific category
python scripts/evaluate.py --category adversarial
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Retrieval Recall@K | Are the correct source documents in the top-K results? |
| Retrieval Precision | What fraction of retrieved documents are relevant? |
| Faithfulness | Does the answer only contain info from retrieved context? |
| Grounding | Can each claim be traced to a specific source chunk? |
| Hallucination Score | Does the answer contain fabricated information? |
| Latency Pass/Fail | Does the response meet the latency threshold? |
| Task Success | Did the agent complete the multi-step task correctly? |
| Tool Selection Accuracy | Did the agent pick the right tools? |

## Scenario Categories

- **simple_factual**: Single-fact questions with verifiable answers
- **multi_field**: Questions requiring multiple fields or documents
- **aggregation**: Comparison and ranking across documents
- **no_answer**: Questions about data NOT in the corpus
- **adversarial**: False premises and hallucination traps
- **multi_hop**: Multi-step reasoning across documents

## License

MIT
