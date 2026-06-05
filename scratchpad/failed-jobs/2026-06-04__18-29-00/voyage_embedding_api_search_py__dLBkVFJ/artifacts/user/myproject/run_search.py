#!/usr/bin/env python3
"""CLI entrypoint for semantic product search."""

import argparse
import json
import sys

from solution import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic product search")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--k", type=int, required=True, help="Number of results")
    args = parser.parse_args()

    results = search(args.query, args.k)
    print(json.dumps(results))
    sys.stdout.flush()


if __name__ == "__main__":
    main()