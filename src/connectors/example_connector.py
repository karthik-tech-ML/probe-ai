"""
Example connector — copy this file as a starting point for your own.

This shows a hypothetical e-commerce product search system. Replace
the placeholder logic with your actual RAG/agent calls. The three
methods you need to implement:

1. ask()              — send a question, get an answer + sources
2. get_corpus_sample() — return some sample documents for scenario generation
3. get_schema()        — describe your data fields so we can auto-generate tests

Once you've built your connector, register it in src/connectors/__init__.py
and you're ready to run:

    python scripts/evaluate.py --connector your_connector
    python scripts/generate_scenarios.py --connector your_connector
"""

import time

from src.connectors.base import (
    CorpusSchema,
    FieldInfo,
    ProbeConnector,
    ProbeResult,
    SourceDocument,
)


class ExampleProductConnector(ProbeConnector):
    """
    Example: wrapping a product search RAG system.

    Replace the fake responses below with calls to your actual
    pipeline — whatever answers questions about your data.
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        # store whatever config your system needs
        self.api_url = api_url

    def ask(self, question: str, mode: str = "rag") -> ProbeResult:
        """
        Send a question to your system and wrap the response.

        In a real connector you'd call your API here. Something like:
            response = requests.post(f"{self.api_url}/ask", json={"question": question})
            data = response.json()

        Then map the response fields into ProbeResult.
        """
        t0 = time.perf_counter()

        # --- replace this block with your actual API call ---
        answer = f"Placeholder answer for: {question}"
        sources = [
            {
                "id": "prod-001",
                "name": "Wireless Headphones",
                "description": "Noise-cancelling over-ear headphones...",
                "price": 149.99,
                "category": "Electronics",
                "similarity": 0.87,
            }
        ]
        model_used = "your-model-name"
        tokens = {"input_tokens": 500, "output_tokens": 200}
        # --- end of placeholder ---

        latency_ms = (time.perf_counter() - t0) * 1000

        # map your system's response into SourceDocuments
        source_docs = [
            SourceDocument(
                doc_id=str(s["id"]),
                title=s["name"],
                content=s["description"],
                metadata={k: v for k, v in s.items() if k not in ("id", "name", "description")},
                similarity=s.get("similarity", 0.0),
            )
            for s in sources
        ]

        return ProbeResult(
            question=question,
            answer=answer,
            source_documents=source_docs,
            context_texts=[doc.content for doc in source_docs],
            latency_ms=round(latency_ms, 1),
            token_usage=tokens,
            model=model_used,
        )

    def get_corpus_sample(self, n: int = 10) -> list[SourceDocument]:
        """
        Return some representative documents from your corpus.

        The scenario generator uses these to craft realistic questions.
        More variety = better test coverage. Pull from different
        categories, price ranges, etc.
        """
        # --- replace with a real DB query or API call ---
        return [
            SourceDocument(
                doc_id="prod-001",
                title="Wireless Headphones",
                content="Noise-cancelling over-ear headphones with 30-hour battery life.",
                metadata={"price": 149.99, "category": "Electronics", "brand": "AudioMax"},
            ),
            SourceDocument(
                doc_id="prod-002",
                title="Running Shoes",
                content="Lightweight mesh running shoes with cushioned sole.",
                metadata={"price": 89.99, "category": "Footwear", "brand": "SpeedStep"},
            ),
        ]

    def get_schema(self) -> CorpusSchema:
        """
        Describe what fields exist in your data.

        The scenario generator reads this to know what kinds of
        questions it can ask — e.g., if you have a "price" field,
        it can generate aggregation questions like "cheapest product
        in electronics category."
        """
        return CorpusSchema(
            domain_name="products",
            description="E-commerce product catalog with pricing, categories, and reviews.",
            fields=[
                FieldInfo(
                    name="name",
                    field_type="string",
                    description="Product name",
                    sample_values=["Wireless Headphones", "Running Shoes", "Coffee Maker"],
                ),
                FieldInfo(
                    name="category",
                    field_type="string",
                    description="Product category",
                    sample_values=["Electronics", "Footwear", "Kitchen"],
                ),
                FieldInfo(
                    name="price",
                    field_type="number",
                    description="Price in USD",
                    sample_values=["149.99", "89.99", "79.99"],
                ),
                FieldInfo(
                    name="brand",
                    field_type="string",
                    description="Brand name",
                    sample_values=["AudioMax", "SpeedStep", "BrewPro"],
                ),
                FieldInfo(
                    name="description",
                    field_type="string",
                    description="Product description",
                    sample_values=["Noise-cancelling over-ear headphones..."],
                ),
            ],
            total_documents=5000,
        )
