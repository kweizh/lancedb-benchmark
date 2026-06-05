import os
import json
import time
import sys
import lancedb
import pyarrow as pa
import voyageai

# Read environment variables
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
ZEALT_RUN_ID = os.environ.get("ZEALT_RUN_ID")

if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY environment variable is not set.")
if not ZEALT_RUN_ID:
    raise ValueError("ZEALT_RUN_ID environment variable is not set.")

DB_PATH = "/home/user/myproject/lancedb_data"
TABLE_NAME = f"products_{ZEALT_RUN_ID}"
CATALOG_PATH = "/home/user/myproject/products.json"

_client = None

def get_voyage_client():
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _client

def embed_with_retry(client, texts, model, input_type, max_retries=5, initial_delay=20):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return client.embed(texts, model=model, input_type=input_type)
        except voyageai.error.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Rate limit hit. Retrying in {delay} seconds...", file=sys.stderr)
            time.sleep(delay)
            delay *= 1.5
        except Exception as e:
            raise e

def get_or_create_table():
    # Ensure database directory exists
    os.makedirs(DB_PATH, exist_ok=True)
    db = lancedb.connect(DB_PATH)
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    
    # Otherwise, create the table
    # Read the products catalog
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Catalogue file not found at {CATALOG_PATH}")
        
    with open(CATALOG_PATH, "r") as f:
        products = json.load(f)
        
    descriptions = [p["description"] for p in products]
    
    # Embed all descriptions
    client = get_voyage_client()
    response = embed_with_retry(client, descriptions, model="voyage-3", input_type="document")
    embeddings = response.embeddings
    
    # Define PyArrow schema
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("description", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])
    
    # Prepare data for insertion
    data = []
    for product, vector in zip(products, embeddings):
        data.append({
            "id": str(product["id"]),
            "description": str(product["description"]),
            "vector": vector
        })
        
    # Create the table in LanceDB
    table = db.create_table(TABLE_NAME, data=data, schema=schema)
    return table

def search(query: str, k: int) -> list[dict]:
    # Ensure table is initialized and get it
    table = get_or_create_table()
    
    # Embed the query
    client = get_voyage_client()
    response = embed_with_retry(client, [query], model="voyage-3", input_type="query")
    query_vector = response.embeddings[0]
    
    # Perform vector search
    results = table.search(query_vector).limit(k).to_list()
    
    # Format and return results
    formatted_results = []
    for row in results:
        formatted_results.append({
            "id": row["id"],
            "description": row["description"]
        })
    return formatted_results
