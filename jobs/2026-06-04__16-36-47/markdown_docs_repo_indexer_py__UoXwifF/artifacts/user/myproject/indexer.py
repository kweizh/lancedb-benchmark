import os
import re
import lancedb
from openai import OpenAI

LANCEDB_PATH = "/home/user/myproject/lancedb/"
DOCS_DIR = "/app/docs/"

def get_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"docs_sections_{run_id}"

def parse_markdown(filepath, repo_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    doc_title = ""
    title_match = re.search(r'^#\s+(.*)', content, re.MULTILINE)
    if title_match:
        doc_title = title_match.group(1).strip()

    sections = []
    parts = re.split(r'^##\s+', content, flags=re.MULTILINE)
    
    for part in parts[1:]:
        lines = part.split('\n', 1)
        section_title = lines[0].strip()
        section_content = lines[1].strip() if len(lines) > 1 else ""
        sections.append({
            "repo_path": repo_path,
            "doc_title": doc_title,
            "section_title": section_title,
            "content": section_content
        })
    return sections

def index_docs():
    sections = []
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                repo_path = os.path.relpath(filepath, DOCS_DIR)
                sections.extend(parse_markdown(filepath, repo_path))

    if not sections:
        return

    texts_to_embed = [f"{s['section_title']}\n{s['content']}" for s in sections]
    
    embeddings = []
    batch_size = 100
    client = OpenAI()
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([d.embedding for d in res.data])

    for i, s in enumerate(sections):
        s['embedding'] = embeddings[i]

    db = lancedb.connect(LANCEDB_PATH)
    table_name = get_table_name()
    
    if table_name in db.table_names():
        db.drop_table(table_name)
        
    db.create_table(table_name, data=sections)

def search(query: str, k: int) -> list[dict]:
    client = OpenAI()
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = res.data[0].embedding

    db = lancedb.connect(LANCEDB_PATH)
    table_name = get_table_name()
    table = db.open_table(table_name)

    results = table.search(query_embedding).metric("cosine").limit(k).to_list()
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            "repo_path": r["repo_path"],
            "doc_title": r["doc_title"],
            "section_title": r["section_title"],
            "score": float(1.0 - r.get("_distance", 0.0))
        })
    return formatted_results

if __name__ == "__main__":
    index_docs()
