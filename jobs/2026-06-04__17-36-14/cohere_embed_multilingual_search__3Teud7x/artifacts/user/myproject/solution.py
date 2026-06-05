import os
import json
import cohere
import lancedb
import pyarrow as pa

# Path configuration
DB_PATH = "/home/user/myproject/lancedb_data"
CORPUS_PATH = "/home/user/myproject/corpus.json"

def get_table_name() -> str:
    """Helper to get the run-scoped table name."""
    zealt_run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"multilingual_{zealt_run_id}"

def build_index() -> None:
    """Reads corpus.json, embeds texts using Cohere embed-multilingual-v3.0, and stores them in LanceDB."""
    # Read corpus
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus file not found at {CORPUS_PATH}")
    
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Initialize Cohere client
    cohere_key = os.environ.get("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY environment variable is not set")
    co = cohere.Client(cohere_key)

    # Extract all texts to embed
    texts = [row["text"] for row in corpus]

    # Embed using embed-multilingual-v3.0 with input_type="search_document"
    res = co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )

    # Extract embeddings list-of-lists format robustly
    if hasattr(res, "embeddings"):
        if isinstance(res.embeddings, list):
            embeddings = res.embeddings
        elif hasattr(res.embeddings, "float_") and res.embeddings.float_ is not None:
            embeddings = res.embeddings.float_
        elif hasattr(res.embeddings, "float") and res.embeddings.float is not None:
            embeddings = res.embeddings.float
        else:
            raise ValueError("Unknown embeddings structure in Cohere response")
    else:
        raise ValueError("Cohere response has no 'embeddings' attribute")

    # Format data for LanceDB ingestion
    rows = []
    for row, emb in zip(corpus, embeddings):
        rows.append({
            "concept_id": int(row["concept_id"]),
            "language": str(row["language"]),
            "text": str(row["text"]),
            "vector": [float(v) for v in emb]
        })

    # Define PyArrow schema explicitly
    schema = pa.schema([
        pa.field("concept_id", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])

    # Connect to LanceDB and create/overwrite table
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()
    
    if table_name in db.table_names():
        db.drop_table(table_name)
    
    tbl = db.create_table(table_name, schema=schema)
    tbl.add(rows)

def cross_lingual_search(query: str, k: int = 3) -> list[dict]:
    """Embeds the query and searches the LanceDB table for the top-k nearest neighbors."""
    # Initialize database connection
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()

    # Build index lazily if table does not exist
    if table_name not in db.table_names():
        build_index()

    tbl = db.open_table(table_name)

    # Initialize Cohere client
    cohere_key = os.environ.get("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY environment variable is not set")
    co = cohere.Client(cohere_key)

    # Embed query using embed-multilingual-v3.0 with input_type="search_query"
    res = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )

    # Extract query embedding vector
    if hasattr(res, "embeddings"):
        if isinstance(res.embeddings, list):
            query_vector = res.embeddings[0]
        elif hasattr(res.embeddings, "float_") and res.embeddings.float_ is not None:
            query_vector = res.embeddings.float_[0]
        elif hasattr(res.embeddings, "float") and res.embeddings.float is not None:
            query_vector = res.embeddings.float[0]
        else:
            raise ValueError("Unknown embeddings structure in Cohere response")
    else:
        raise ValueError("Cohere response has no 'embeddings' attribute")

    # Perform LanceDB search using cosine metric (since embeddings are L2 normalized)
    search_results = tbl.search(query_vector).metric("cosine").limit(k).to_list()

    # Format the results
    output = []
    for item in search_results:
        output.append({
            "concept_id": int(item["concept_id"]),
            "language": str(item["language"]),
            "text": str(item["text"])
        })
    return output
