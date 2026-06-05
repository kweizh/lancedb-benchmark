import argparse
import json
from solution import search

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    
    results = search(args.query, args.k)
    print(json.dumps(results))

if __name__ == "__main__":
    main()
