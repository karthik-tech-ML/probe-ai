"""
Generate evaluation scenarios from any connected system.

Usage:
    python scripts/generate_scenarios.py --connector tmdb
    python scripts/generate_scenarios.py --connector tmdb --categories simple_factual,multi_field
    python scripts/generate_scenarios.py --connector tmdb --count 10 --out scenarios/generated.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.connectors import get_connector
from src.generator.prompts import ALL_CATEGORIES
from src.generator.scenario_generator import generate_scenarios, scenarios_to_library


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation scenarios from a connected system"
    )
    parser.add_argument(
        "--connector", type=str, required=True,
        help="connector name (e.g. 'tmdb')",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help=f"comma-separated categories to generate (default: all). "
             f"Available: {', '.join(ALL_CATEGORIES)}",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="number of scenarios per category (default: 5)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-5-20250929",
        help="Claude model to use for generation",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="path to write generated scenarios JSON",
    )
    args = parser.parse_args()

    # parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # get the connector
    connector = get_connector(args.connector)
    logger.info(f"using connector: {args.connector}")

    # generate
    scenarios = generate_scenarios(
        connector=connector,
        categories=categories,
        count_per_category=args.count,
        model=args.model,
    )

    if not scenarios:
        logger.error("no scenarios generated")
        sys.exit(1)

    # wrap in library format
    schema = connector.get_schema()
    library = scenarios_to_library(scenarios, domain_name=schema.domain_name)

    # output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(library, indent=2))
        logger.info(f"wrote {len(scenarios)} scenarios to {out_path}")
    else:
        print(json.dumps(library, indent=2))

    # summary
    by_category = {}
    for s in scenarios:
        cat = s["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    print(f"\nGenerated {len(scenarios)} scenarios:", file=sys.stderr)
    for cat, n in sorted(by_category.items()):
        print(f"  {cat}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
