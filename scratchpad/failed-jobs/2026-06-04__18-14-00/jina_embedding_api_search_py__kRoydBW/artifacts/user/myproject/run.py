"""
CLI entrypoint for the news headline semantic search system.

Usage:
    python3 run.py --build
    python3 run.py "<query>" [--k <int>]
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over news headlines."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query string.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build (or rebuild) the LanceDB index from headlines.json.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )

    args = parser.parse_args()

    if args.build:
        from solution import build_index
        build_index()
        return

    if args.query is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    from solution import search
    results = search(args.query, k=args.k)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
