"""
Run the ProbeAI inference benchmark.

Compares cloud (Claude API) vs local (Ollama) inference using the same
retrieved context. Both backends get identical chunks from pgvector —
we're only measuring how well each model generates grounded answers.

Usage:
    python scripts/benchmark.py --backend local
    python scripts/benchmark.py --backend local --category simple_factual
    python scripts/benchmark.py --backend both --out results/bench.json
    python scripts/benchmark.py --backend local --model llama3.2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.benchmark.comparison import compare, print_comparison
from src.benchmark.inference import run_benchmark
from src.benchmark.local_backend import check_ollama, list_models
from src.evaluation.runner import load_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProbeAI inference benchmark — cloud vs local"
    )
    parser.add_argument(
        "--backend",
        choices=["cloud", "local", "both"],
        default="local",
        help="which backend(s) to benchmark (default: local)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="filter to a specific scenario category",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="override the model name (e.g. llama3.2, mistral)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="path to write full JSON results",
    )
    args = parser.parse_args()

    # figure out which backends to run
    run_cloud = args.backend in ("cloud", "both")
    run_local = args.backend in ("local", "both")

    # preflight checks
    if run_local:
        if not check_ollama():
            logger.error(
                "Ollama is not running. Start it with: ollama serve"
            )
            sys.exit(1)
        available = list_models()
        target_model = args.model or "llama3.2"
        # ollama model names can have :latest suffix
        has_model = any(
            m == target_model or m.startswith(f"{target_model}:")
            for m in available
        )
        if not has_model:
            logger.error(
                f"Model {target_model!r} not found in Ollama. "
                f"Available: {available}. Pull it with: ollama pull {target_model}"
            )
            sys.exit(1)
        logger.info(f"Ollama ready, model: {target_model}")

    if run_cloud:
        from src.config import settings
        if not settings.anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY not set — can't run cloud backend")
            sys.exit(1)

    # load scenarios
    scenarios = load_scenarios(category=args.category)
    logger.info(f"loaded {len(scenarios)} scenarios")

    # run benchmarks
    cloud_results = None
    local_results = None

    if run_cloud:
        logger.info("--- Running cloud benchmark ---")
        cloud_results = run_benchmark(
            scenarios, "cloud", top_k=args.top_k, model=args.model
        )

    if run_local:
        logger.info("--- Running local benchmark ---")
        local_results = run_benchmark(
            scenarios, "local", top_k=args.top_k, model=args.model
        )

    # build comparison report
    report = compare(cloud_results, local_results)
    print_comparison(report)

    # write JSON if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json())
        logger.info(f"results written to {out_path}")


if __name__ == "__main__":
    main()
