import os
import json
import cohere
import lancedb
import numpy as np
from lancedb.pydantic import LanceModel, Vector

# Configuration
CORPUS_PATH = "/home/user/myproject/corpus.json"
DB_PATH = "/home/user/myproject/lancedb_data"
MODEL_NAME = "embed-multilingual-v3.0"
DIMENSION = 1024

def get_table_name():
    zealt_run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"multilingual_{zealt_run_id}"

class Schema(LanceModel):
    concept_id: int
    language: str
    text: str
    vector: Vector(DIMENSION)

def build_index():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")
    
    co = cohere.Client(api_key)
    
    with open(CORPUS_PATH, 'r') as f:
        corpus = json.load(f)
    
    texts = [row['text'] for row in corpus]
    
    # Embed documents
    response = co.embed(
        texts=texts,
        model=MODEL_NAME,
        input_type="search_document",
        embedding_types=["float"]
    )
    
    # Cohere SDK v5+ returns response.embeddings.float_
    # Cohere SDK v4- returns response.embeddings
    if hasattr(response.embeddings, 'float_'):
        embeddings = response.embeddings.float_
    elif hasattr(response.embeddings, 'float'):
        embeddings = response.embeddings.float
    else:
        embeddings = response.embeddings

    # Prepare data for LanceDB
    data = []
    for i, row in enumerate(corpus):
        data.append({
            "concept_id": row["concept_id"],
            "language": row["language"],
            "text": row["text"],
            "vector": embeddings[i]
        })
    
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()
    
    if table_name in db.table_names():
        db.drop_table(table_name)
    
    db.create_table(table_name, data=data, schema=Schema)

def cross_lingual_search(query: str, k: int = 3) -> list[dict]:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")
    
    co = cohere.Client(api_key)
    
    # Embed query
    response = co.embed(
        texts=[query],
        model=MODEL_NAME,
        input_type="search_query",
        embedding_types=["float"]
    )
    
    if hasattr(response.embeddings, 'float_'):
        query_vector = response.embeddings.float_[0]
    elif hasattr(response.embeddings, 'float'):
        query_vector = response.embeddings.float[0]
    else:
        query_vector = response.embeddings[0]
    
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()
    
    if table_name not in db.table_names():
        # Implicitly build index if it doesn't exist
        build_index()
    
    table = db.open_table(table_name)
    
    results = table.search(query_vector).limit(k).to_list()
    
    # Format output to include only required keys
    formatted_results = []
    for res in results:
        formatted_results.append({
            "concept_id": res["concept_id"],
            "language": res["language"],
            "text": res["text"]
        })
    
    return formatted_results
