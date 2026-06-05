#!/usr/bin/env python3
"""CLI entrypoint for headline semantic search."""

import argparse
import json
import sys

from solution import build_index, search


def main() -> None:
    parser = argparse.ArgumentParser(description="News headline semantic search")
    parser.add_argument("--build", action="store_true", help="Build the search index")
    parser.add_argument("query", nargs="?", default=None, help="Search query string")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.build:
        build_index()
        return

    if args.query is None:
        parser.error("A query string is required when --build is not used")

    results = search(args.query, k=args.k)
    print(json.dumps(results))


if __name__ == "__main__":
    main()