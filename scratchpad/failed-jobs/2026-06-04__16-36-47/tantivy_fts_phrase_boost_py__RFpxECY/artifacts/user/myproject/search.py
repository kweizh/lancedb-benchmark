import argparse
import json
import os
import lancedb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    db_path = "/home/user/myproject/lancedb_data"
    db = lancedb.connect(db_path)

    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"docs_{run_id}"

    if table_name not in db.table_names():
        with open("/home/user/myproject/seed/docs.json", "r") as f:
            data = json.load(f)
        
        table = db.create_table(table_name, data=data)
        table.create_fts_index(["title", "body"], use_tantivy=True)
    else:
        table = db.open_table(table_name)

    results = table.search(args.query, query_type="fts").limit(args.k).to_list()
    
    clean_results = []
    for r in results:
        clean_results.append({
            "id": r["id"],
            "title": r["title"],
            "body": r["body"]
        })

    print(json.dumps(clean_results))

if __name__ == "__main__":
    main()
