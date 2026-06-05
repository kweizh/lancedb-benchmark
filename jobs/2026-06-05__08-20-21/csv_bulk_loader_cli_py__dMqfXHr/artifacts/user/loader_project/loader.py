import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Any

import lancedb
import pyarrow as pa
from openai import OpenAI

# Configuration
DB_PATH = "/home/user/loader_project/lance_db"
MODEL = "text-embedding-3-small"

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    
    # Workaround for httpx/openai conflict where OpenAI passes 'proxies' to httpx.Client
    # which is not supported in newer httpx versions.
    import httpx
    return OpenAI(api_key=api_key, http_client=httpx.Client())

def get_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    response = client.embeddings.create(
        input=texts,
        model=MODEL
    )
    return [data.embedding for data in response.data]

def ingest(csv_path: str, table_name: str, text_col: str, batch_size: int):
    """Ingest CSV data into LanceDB with embeddings."""
    client = get_openai_client()
    db = lancedb.connect(DB_PATH)
    
    # Read CSV data
    rows = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    total_rows = len(rows)
    print(f"Read {total_rows} rows from {csv_path}", file=sys.stderr)
    
    data_to_ingest = []
    
    for i in range(0, total_rows, batch_size):
        batch = rows[i:i + batch_size]
        batch_texts = [row[text_col] for row in batch]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}...", file=sys.stderr)
        embeddings = get_embeddings(client, batch_texts)
        
        for row, emb in zip(batch, embeddings):
            # Ensure proper types for LanceDB/Arrow if necessary
            # The requirements specify id, title, body, category, published
            ingest_row = {
                "id": int(row["id"]),
                "title": row["title"],
                "body": row["body"],
                "category": row["category"],
                "published": row["published"],
                "vector": emb
            }
            data_to_ingest.append(ingest_row)
            
    # Create or overwrite the table
    if table_name in db.table_names():
        db.drop_table(table_name)
    
    db.create_table(table_name, data=data_to_ingest)
    print(f"Successfully ingested {len(data_to_ingest)} rows into table '{table_name}'", file=sys.stderr)

def search(table_name: str, query: str, k: int):
    """Search for top-k matches in a LanceDB table."""
    client = get_openai_client()
    db = lancedb.connect(DB_PATH)
    
    try:
        table = db.open_table(table_name)
    except Exception as e:
        print(f"Error opening table '{table_name}': {e}", file=sys.stderr)
        sys.exit(1)
        
    # Embed the query
    query_vector = get_embeddings(client, [query])[0]
    
    # Run search
    results = table.search(query_vector).limit(k).to_list()
    
    # Format results
    formatted_results = []
    for res in results:
        # Distance is returned as '_distance' by LanceDB
        score = res.get("_distance", 0.0)
        formatted_results.append({
            "id": int(res["id"]),
            "title": res["title"],
            "category": res["category"],
            "published": res["published"],
            "score": float(score)
        })
        
    output = {
        "query": query,
        "k": k,
        "results": formatted_results
    }
    
    print(json.dumps(output))

def main():
    parser = argparse.ArgumentParser(description="CSV Bulk Loader CLI for LanceDB")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--csv", required=True, help="Path to CSV file")
    ingest_parser.add_argument("--table", required=True, help="Table name")
    ingest_parser.add_argument("--text-col", required=True, help="Column to embed")
    ingest_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embeddings")
    
    # Search subcommand
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--table", required=True, help="Table name")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest(args.csv, args.table, args.text_col, args.batch_size)
    elif args.command == "search":
        search(args.table, args.query, args.k)

if __name__ == "__main__":
    main()
