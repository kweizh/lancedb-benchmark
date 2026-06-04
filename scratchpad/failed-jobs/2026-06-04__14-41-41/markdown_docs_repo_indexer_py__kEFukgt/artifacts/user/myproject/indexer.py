import os
import sys
import lancedb
import pyarrow as pa
from openai import OpenAI

def parse_markdown_file(file_path: str, relative_path: str) -> list[dict]:
    """
    Parses a markdown file and splits it into sections by '## ' headers.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.splitlines()
    doc_title = ""
    
    # Extract doc title (the single top-level # Title)
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break
            
    sections = []
    current_section_title = None
    current_section_lines = []
    
    for line in lines:
        if line.startswith("## "):
            if current_section_title is not None:
                sections.append({
                    "repo_path": relative_path,
                    "doc_title": doc_title,
                    "section_title": current_section_title,
                    "content": "\n".join(current_section_lines).strip()
                })
            current_section_title = line[3:].strip()
            current_section_lines = []
        else:
            if current_section_title is not None:
                current_section_lines.append(line)
                
    if current_section_title is not None:
        sections.append({
            "repo_path": relative_path,
            "doc_title": doc_title,
            "section_title": current_section_title,
            "content": "\n".join(current_section_lines).strip()
        })
        
    return sections

def get_all_sections(docs_dir: str) -> list[dict]:
    """
    Walks docs_dir recursively and parses all .md files.
    """
    all_sections = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, docs_dir)
                sections = parse_markdown_file(full_path, relative_path)
                all_sections.extend(sections)
    return all_sections

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using OpenAI text-embedding-3-small.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
        
    client = OpenAI(**kwargs)
    
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def build_index():
    """
    Runs the full indexing process: walks, parses, embeds, and persists to LanceDB.
    """
    print("Starting indexing process...")
    docs_dir = "/app/docs/"
    db_dir = "/home/user/myproject/lancedb/"
    
    # Ensure database directory exists
    os.makedirs(db_dir, exist_ok=True)
    
    # Get run id and table name
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    table_name = f"docs_sections_{run_id}"
    print(f"Using table name: {table_name}")
    
    # Walk and parse docs
    sections = get_all_sections(docs_dir)
    print(f"Found {len(sections)} sections in total.")
    
    if not sections:
        print("No sections found to index.")
        return
        
    # Prepare texts for embedding
    # We concatenate section_title and content for richer semantics
    texts_to_embed = []
    for s in sections:
        texts_to_embed.append(f"{s['section_title']}\n\n{s['content']}")
        
    print("Generating embeddings via OpenAI...")
    embeddings = get_embeddings(texts_to_embed)
    print("Embeddings generated successfully.")
    
    # Combine sections with their embeddings
    data = []
    for s, emb in zip(sections, embeddings):
        data.append({
            "repo_path": s["repo_path"],
            "doc_title": s["doc_title"],
            "section_title": s["section_title"],
            "content": s["content"],
            "embedding": emb
        })
        
    # Create LanceDB table with schema
    schema = pa.schema([
        pa.field("repo_path", pa.string()),
        pa.field("doc_title", pa.string()),
        pa.field("section_title", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 1536))
    ])
    
    db = lancedb.connect(db_dir)
    # Mode overwrite ensures idempotency
    db.create_table(table_name, data=data, schema=schema, mode="overwrite")
    print(f"Table '{table_name}' created and populated successfully in LanceDB.")

def search(query: str, k: int) -> list[dict]:
    """
    Exposes a search API to find top-k most relevant sections for a query.
    """
    db_dir = "/home/user/myproject/lancedb/"
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    table_name = f"docs_sections_{run_id}"
    
    db = lancedb.connect(db_dir)
    # Check if table exists
    if table_name not in db.table_names():
        raise ValueError(f"Table '{table_name}' does not exist. Please run indexer.py first to build the index.")
        
    table = db.open_table(table_name)
    
    # Embed query
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
        
    client = OpenAI(**kwargs)
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Search table
    # We order from most to least semantically relevant (using cosine distance)
    results = table.search(query_embedding).metric("cosine").limit(k).to_list()
    
    output = []
    for row in results:
        output.append({
            "repo_path": row["repo_path"],
            "doc_title": row["doc_title"],
            "section_title": row["section_title"],
            "score": float(row["_distance"])
        })
    return output

if __name__ == "__main__":
    build_index()
