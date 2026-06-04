import os
import sys
import json
import argparse
import lancedb

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tantivy-backed Full-Text Search")
    parser.add_argument("--query", type=str, required=True, help="The raw Tantivy query string")
    parser.add_argument("--k", type=int, required=True, help="The number of results to return")
    args = parser.parse_args()

    # Read ZEALT_RUN_ID environment variable
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("Error: ZEALT_RUN_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    db_dir = "/home/user/myproject/lancedb_data"
    os.makedirs(db_dir, exist_ok=True)
    
    table_name = f"docs_{run_id}"

    # Connect to LanceDB
    db = lancedb.connect(db_dir)

    # Idempotent table creation and index construction
    if table_name in db.table_names():
        table = db.open_table(table_name)
    else:
        seed_path = "/home/user/myproject/seed/docs.json"
        if not os.path.exists(seed_path):
            print(f"Error: Seed file not found at {seed_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(seed_path, "r") as f:
            data = json.load(f)
        
        table = db.create_table(table_name, data=data)
        table.create_fts_index(["title", "body"], use_tantivy=True)

    # Perform search using the Tantivy-backed FTS index
    hits = table.search(args.query, query_type="fts").limit(args.k).to_list()

    # Format the results to match criteria (strip internal score/distance columns)
    results = []
    for hit in hits:
        results.append({
            "id": int(hit["id"]),
            "title": str(hit["title"]),
            "body": str(hit["body"])
        })

    # Print JSON list of results to stdout
    print(json.dumps(results))

if __name__ == "__main__":
    main()
