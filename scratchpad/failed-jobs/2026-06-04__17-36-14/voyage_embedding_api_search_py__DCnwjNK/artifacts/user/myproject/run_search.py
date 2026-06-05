import argparse
import json
import sys
from solution import search

def main():
    parser = argparse.ArgumentParser(description="Run semantic product search.")
    parser.add_argument("--query", type=str, required=True, help="The search query text.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    
    args = parser.parse_args()
    
    try:
        results = search(args.query, args.k)
        # Print a single JSON array to stdout
        print(json.dumps(results))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
