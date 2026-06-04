#!/usr/bin/env python3
import argparse
import sys
import os
import json
import pandas as pd
import pyarrow as pa
import lancedb
import httpx
from openai import OpenAI

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, http_client=httpx.Client())

def handle_ingest(args):
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading CSV file from {args.csv}...", file=sys.stderr)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate columns
    required_cols = ["id", "title", "body", "category", "published"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing from the CSV.", file=sys.stderr)
            sys.exit(1)

    if args.text_col not in df.columns:
        print(f"Error: Designated text column '{args.text_col}' not found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Handle missing values and convert types defensively
    df["id"] = df["id"].astype(int)
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["published"] = df["published"].fillna("").astype(str)

    # Database setup
    db_dir = "/home/user/loader_project/lance_db"
    os.makedirs(db_dir, exist_ok=True)
    db = lancedb.connect(db_dir)

    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("body", pa.string()),
        pa.field("category", pa.string()),
        pa.field("published", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1536))
    ])

    print(f"Creating table '{args.table}' in LanceDB...", file=sys.stderr)
    tbl = db.create_table(args.table, schema=schema, mode="overwrite")

    # OpenAI Client
    client = get_openai_client()

    total_rows = len(df)
    batch_size = args.batch_size
    if batch_size <= 0:
        print("Error: Batch size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting ingestion of {total_rows} rows in batches of {batch_size}...", file=sys.stderr)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Prepare text to embed
        texts = batch_df[args.text_col].tolist()
        # Clean empty texts to prevent OpenAI API errors
        processed_texts = [t if t.strip() != "" else " " for t in texts]

        print(f"Embedding rows {start_idx} to {end_idx}...", file=sys.stderr)
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=processed_texts
            )
            embeddings = [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error calling OpenAI Embeddings API: {e}", file=sys.stderr)
            sys.exit(1)

        if len(embeddings) != len(processed_texts):
            print(f"Error: Number of embeddings returned ({len(embeddings)}) does not match batch size ({len(processed_texts)}).", file=sys.stderr)
            sys.exit(1)

        # Construct batch records
        batch_records = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            batch_records.append({
                "id": int(row["id"]),
                "title": str(row["title"]),
                "body": str(row["body"]),
                "category": str(row["category"]),
                "published": str(row["published"]),
                "vector": embeddings[idx]
            })

        tbl.add(batch_records)
        print(f"Successfully added batch to table.", file=sys.stderr)

    print(f"Ingestion complete. Table '{args.table}' contains {len(tbl)} rows.", file=sys.stderr)

def handle_search(args):
    db_dir = "/home/user/loader_project/lance_db"
    if not os.path.exists(db_dir):
        print(f"Error: Database directory '{db_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    db = lancedb.connect(db_dir)
    if args.table not in db.table_names():
        print(f"Error: Table '{args.table}' does not exist in LanceDB.", file=sys.stderr)
        sys.exit(1)

    tbl = db.open_table(args.table)

    # OpenAI Client
    client = get_openai_client()

    # Clean query text
    query_text = args.query if args.query.strip() != "" else " "

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query_text]
        )
        query_vector = response.data[0].embedding
    except Exception as e:
        print(f"Error calling OpenAI Embeddings API: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform search
    try:
        search_results = tbl.search(query_vector).limit(args.k).to_list()
    except Exception as e:
        print(f"Error performing search in LanceDB: {e}", file=sys.stderr)
        sys.exit(1)

    # Format results to match acceptance criteria
    results_list = []
    for row in search_results:
        results_list.append({
            "id": int(row["id"]),
            "title": str(row["title"]),
            "category": str(row["category"]),
            "published": str(row["published"]),
            "score": float(row["_distance"])
        })

    output = {
        "query": args.query,
        "k": args.k,
        "results": results_list
    }

    # Print ONLY the JSON to stdout
    print(json.dumps(output, indent=2))

def main():
    parser = argparse.ArgumentParser(description="CSV Bulk Loader CLI for LanceDB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest parser
    parser_ingest = subparsers.add_parser("ingest", help="Ingest CSV into LanceDB table")
    parser_ingest.add_argument("--csv", required=True, help="Path to the source CSV file")
    parser_ingest.add_argument("--table", required=True, help="Name of the LanceDB table")
    parser_ingest.add_argument("--text-col", required=True, help="Designated text column to embed")
    parser_ingest.add_argument("--batch-size", required=True, type=int, help="Batch size for embedding and ingestion")

    # Search parser
    parser_search = subparsers.add_parser("search", help="Semantic search on LanceDB table")
    parser_search.add_argument("--table", required=True, help="Name of the LanceDB table")
    parser_search.add_argument("--query", required=True, help="Query string to search for")
    parser_search.add_argument("--k", required=True, type=int, help="Number of top results to return")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "search":
        handle_search(args)

if __name__ == "__main__":
    main()
