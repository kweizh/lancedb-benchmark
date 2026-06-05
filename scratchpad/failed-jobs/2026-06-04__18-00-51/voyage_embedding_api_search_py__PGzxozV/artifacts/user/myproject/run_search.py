"""
CLI entrypoint for semantic product search.

Usage:
    python3 run_search.py --query "<text>" --k <int>

Prints a JSON array of result dicts (id, description) ordered by relevance.
"""

import argparse
import json
import sys

from solution import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic product search via Voyage AI + LanceDB")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--k", type=int, required=True, help="Number of results to return")
    args = parser.parse_args()

    results = search(args.query, args.k)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
