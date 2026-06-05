import os
import re
import lancedb
import pandas as pd
from openai import OpenAI
from pathlib import Path

# Configuration
DOCS_DIR = "/app/docs/"
DB_PATH = "/home/user/myproject/lancedb/"
EMBEDDING_MODEL = "text-embedding-3-small"

def get_table_name():
    run_id = os.getenv("ZEALT_RUN_ID", "default")
    return f"docs_sections_{run_id}"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def parse_markdown(file_path: Path, relative_path: str):
    content = file_path.read_text()
    
    # Extract Title
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else "Untitled"
    
    # Split by sections (## Section)
    sections = []
    # Find all ## headers and their positions
    header_matches = list(re.finditer(r'^##\s+(.+)$', content, re.MULTILINE))
    
    for i, match in enumerate(header_matches):
        section_title = match.group(1).strip()
        start_pos = match.end()
        end_pos = header_matches[i+1].start() if i + 1 < len(header_matches) else len(content)
        
        section_content = content[start_pos:end_pos].strip()
        
        if section_content:
            sections.append({
                "repo_path": relative_path,
                "doc_title": doc_title,
                "section_title": section_title,
                "content": section_content
            })
            
    return sections

def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

def index_docs():
    all_sections = []
    docs_path = Path(DOCS_DIR)
    
    for md_file in docs_path.rglob("*.md"):
        relative_path = str(md_file.relative_to(docs_path))
        sections = parse_markdown(md_file, relative_path)
        all_sections.extend(sections)
    
    if not all_sections:
        print("No sections found to index.")
        return

    # Compute embeddings
    for section in all_sections:
        # Concatenate title and content for better context
        text_to_embed = f"{section['section_title']}\n{section['content']}"
        section['embedding'] = get_embedding(text_to_embed)
    
    # Persist to LanceDB
    db = lancedb.connect(DB_PATH)
    
    # Create or overwrite table
    # Make it idempotent: if it exists, we can overwrite or just leave it. 
    # Requirement says: "when the script is re-run with the same ZEALT_RUN_ID, it should not error out and should leave the table queryable."
    # Overwriting is a safe way to ensure it's queryable and updated.
    table_name = get_table_name()
    db.create_table(table_name, data=all_sections, mode="overwrite")
    print(f"Indexed {len(all_sections)} sections into table '{table_name}'")

def search(query: str, k: int) -> list[dict]:
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()
    table = db.open_table(table_name)
    
    query_embedding = get_embedding(query)
    
    results = table.search(query_embedding).limit(k).to_list()
    
    formatted_results = []
    for res in results:
        formatted_results.append({
            "repo_path": res["repo_path"],
            "doc_title": res["doc_title"],
            "section_title": res["section_title"],
            "score": float(res["_distance"]) # LanceDB returns _distance by default
        })
    
    return formatted_results

if __name__ == "__main__":
    index_docs()
