import os
import json
import lancedb
import pyarrow as pa
import voyageai

def init_db():
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID not set")
    
    db_path = "/home/user/myproject/lancedb_data"
    table_name = f"products_{run_id}"
    
    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        return db.open_table(table_name)
    
    # Otherwise, create the table
    products_file = "/home/user/myproject/products.json"
    with open(products_file, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    # Embed the descriptions
    client = voyageai.Client() # Uses VOYAGE_API_KEY from env
    descriptions = [p["description"] for p in products]
    
    result = client.embed(descriptions, model="voyage-3", input_type="document")
    embeddings = result.embeddings
    
    # Prepare data for LanceDB
    data = []
    for p, emb in zip(products, embeddings):
        data.append({
            "id": str(p["id"]),
            "description": p["description"],
            "vector": emb
        })
    
    # Define schema
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("description", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])
    
    table = db.create_table(table_name, data=data, schema=schema)
    return table

def search(query: str, k: int) -> list[dict]:
    table = init_db()
    client = voyageai.Client()
    
    # Embed query
    result = client.embed([query], model="voyage-3", input_type="query")
    query_vec = result.embeddings[0]
    
    # Search
    results = table.search(query_vec).limit(k).to_list()
    
    # Format output
    output = []
    for r in results:
        output.append({
            "id": r["id"],
            "description": r["description"]
        })
        
    return output
