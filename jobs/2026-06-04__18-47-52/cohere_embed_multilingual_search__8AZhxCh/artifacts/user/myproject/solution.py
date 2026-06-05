import os
import json
import cohere
import lancedb
import pyarrow as pa

def get_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"multilingual_{run_id}"

def build_index():
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    
    with open("/home/user/myproject/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    texts = [row["text"] for row in corpus]
    
    # We embed in batches if needed, but 90 rows is well within Cohere's limits (which is usually 96).
    response = co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    
    embeddings = response.embeddings
    
    data = []
    for row, emb in zip(corpus, embeddings):
        data.append({
            "concept_id": row["concept_id"],
            "language": row["language"],
            "text": row["text"],
            "vector": emb
        })
        
    schema = pa.schema([
        pa.field("concept_id", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])
    
    db = lancedb.connect("/home/user/myproject/lancedb_data/")
    table_name = get_table_name()
    db.create_table(table_name, data=data, schema=schema, mode="overwrite")

def cross_lingual_search(query: str, k: int = 3) -> list[dict]:
    db = lancedb.connect("/home/user/myproject/lancedb_data/")
    table_name = get_table_name()
    
    if table_name not in db.table_names():
        build_index()
        
    table = db.open_table(table_name)
    
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    response = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    
    query_vector = response.embeddings[0]
    
    results = table.search(query_vector, vector_column_name="vector").limit(k).to_list()
    
    output = []
    for r in results:
        output.append({
            "concept_id": r["concept_id"],
            "language": r["language"],
            "text": r["text"]
        })
    return output
