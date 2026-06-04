import argparse
import os
import sys
import json
import pandas as pd
import lancedb
from openai import OpenAI

DB_PATH = "/home/user/loader_project/lance_db"

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is missing", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)

def get_embeddings(texts, client):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def ingest(args):
    client = get_openai_client()
    db = lancedb.connect(DB_PATH)
    
    csv_path = args.csv
    table_name = args.table
    text_col = args.text_col
    batch_size = args.batch_size
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    
    # Process in batches
    data = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        texts = batch[text_col].astype(str).tolist()
        
        # Get embeddings
        embeddings = get_embeddings(texts, client)
        
        # Prepare records
        records = batch.to_dict(orient="records")
        for record, emb in zip(records, embeddings):
            record["vector"] = emb
            data.append(record)
            
    # Create or overwrite table
    db.create_table(table_name, data=data, mode="overwrite")
    print(f"Successfully ingested {len(df)} rows into {table_name}", file=sys.stderr)

def search(args):
    client = get_openai_client()
    db = lancedb.connect(DB_PATH)
    
    table_name = args.table
    query = args.query
    k = args.k
    
    # Embed query
    query_vector = get_embeddings([query], client)[0]
    
    # Search
    table = db.open_table(table_name)
    results = table.search(query_vector).limit(k).to_list()
    
    # Format output
    output_results = []
    for r in results:
        score = r.get("_distance", 0.0)
        output_results.append({
            "id": int(r["id"]),
            "title": str(r["title"]),
            "category": str(r["category"]),
            "published": str(r["published"]),
            "score": float(score)
        })
        
    output = {
        "query": query,
        "k": k,
        "results": output_results
    }
    
    print(json.dumps(output))

def main():
    parser = argparse.ArgumentParser(description="CSV Bulk Loader CLI for LanceDB")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--csv", required=True, help="Path to CSV file")
    ingest_parser.add_argument("--table", required=True, help="LanceDB table name")
    ingest_parser.add_argument("--text-col", required=True, help="Column to embed")
    ingest_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding")
    
    # Search subcommand
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--table", required=True, help="LanceDB table name")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest(args)
    elif args.command == "search":
        search(args)

if __name__ == "__main__":
    main()
