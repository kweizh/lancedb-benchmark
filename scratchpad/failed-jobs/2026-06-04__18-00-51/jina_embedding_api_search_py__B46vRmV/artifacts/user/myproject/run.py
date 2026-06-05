#!/usr/bin/env python3
"""
CLI entrypoint for the headline semantic search system.

Usage:
    python3 run.py --build
        Rebuild the LanceDB index from headlines.json.

    python3 run.py "<query>" [--k <int>]
        Search the index and print a JSON array of k results to stdout.
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="News headline semantic search (Jina v3 + LanceDB)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query string.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build (or rebuild) the vector index from headlines.json.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="retrieval.query",
        help="Jina task parameter for query embedding (default: retrieval.query).",
    )

    args = parser.parse_args()

    if args.build:
        from solution import build_index
        build_index()
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    from solution import search
    results = search(args.query, k=args.k, task=args.task)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
