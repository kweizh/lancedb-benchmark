import argparse
import json
from solution import build_index, search

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--build", action="store_true", help="Build index")
    
    args = parser.parse_args()
    
    if args.build:
        build_index()
    elif args.query:
        results = search(args.query, k=args.k)
        print(json.dumps(results))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
