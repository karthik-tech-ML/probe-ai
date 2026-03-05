"""
Smoke test for the full RAG pipeline.
Runs 3 queries and prints answer, sources, similarity scores, and latency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import ask

QUERIES = [
    "Who directed Inception?",
    "What is the plot of Oppenheimer?",
    "What genre is Toy Story?",
]


def main() -> None:
    for i, question in enumerate(QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"  Test {i}: {question}")
        print(f"{'='*60}")

        result = ask(question, top_k=5)

        print(f"\n  Answer:\n    {result.answer}\n")

        print(f"  Sources:")
        for j, src in enumerate(result.sources, 1):
            print(f"    {j}. {src.title} (similarity: {src.similarity})")

        print(f"\n  Latency: {result.latency_ms:.0f}ms")
        print(
            f"  Tokens:  {result.generation.usage['input_tokens']}in / "
            f"{result.generation.usage['output_tokens']}out"
        )
        print(f"  Model:   {result.generation.model}")


if __name__ == "__main__":
    main()
