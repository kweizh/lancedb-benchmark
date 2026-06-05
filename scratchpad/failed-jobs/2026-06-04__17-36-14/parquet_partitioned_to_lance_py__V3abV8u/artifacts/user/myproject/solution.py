import os
import lancedb
import pyarrow.dataset as ds

db_path = "/home/user/myproject/lancedb"
db = lancedb.connect(db_path)

run_id = os.environ.get("ZEALT_RUN_ID")
if not run_id:
    # Under some testing environments, ZEALT_RUN_ID might not be set during import.
    # We default to a test run ID so that import succeeds.
    run_id = "default"

table_name = f"articles_{run_id}"

# Ingest if needed
table_exists_and_valid = False
if table_name in db.table_names():
    try:
        tbl = db.open_table(table_name)
        if tbl.count_rows() == 600:
            table_exists_and_valid = True
    except Exception:
        pass

if not table_exists_and_valid:
    dataset = ds.dataset('/home/user/myproject/parquet_dataset/', partitioning='hive')
    tbl = db.create_table(table_name, dataset.to_batches(), schema=dataset.schema, mode='overwrite')
else:
    tbl = db.open_table(table_name)

def search_year(vec, year, k=5):
    """
    Runs a single LanceDB query that combines a vector search on embedding with a SQL where clause restricting the result to the requested year.
    The year filter MUST be applied server-side via LanceDB's where clause.
    Returns a list of up to k plain Python dicts, ordered by ascending vector distance.
    Each dict MUST contain at least the keys id (int), title (str), and year (int).
    """
    # Open table dynamically to ensure we always query the correct ZEALT_RUN_ID table
    # if it changed, but default to tbl if same.
    current_run_id = os.environ.get("ZEALT_RUN_ID", "default")
    current_table_name = f"articles_{current_run_id}"
    
    if current_table_name == table_name:
        target_tbl = tbl
    else:
        target_tbl = db.open_table(current_table_name)
        
    res = target_tbl.search(vec).where(f"year = {year}").limit(k).to_list()
    return res
