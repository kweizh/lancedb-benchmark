import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from solution import build_index, search

def main():
    parser = argparse.ArgumentParser(description="News Headline Semantic Search CLI")
    parser.add_argument("--build", action="store_true", help="Rebuild the index")
    parser.add_argument("query", nargs="?", default=None, help="The query string to search for")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.build:
        build_index()
    else:
        if args.query is None:
            parser.error("The query argument is required unless --build is specified.")
        results = search(args.query, k=args.k)
        print(json.dumps(results))

if __name__ == "__main__":
    main()
