import os
import pyarrow.dataset as ds
import lancedb

def _init():
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"articles_{run_id}"
    db_path = "/home/user/myproject/lancedb"
    
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    
    if table_name in db.table_names():
        table = db.open_table(table_name)
        if len(table) == 600:
            return table
            
    dataset = ds.dataset("/home/user/myproject/parquet_dataset", format="parquet", partitioning="hive")
    table = db.create_table(table_name, data=dataset.scanner().to_reader(), mode="overwrite")
    return table

_init()

def search_year(vec, year, k=5):
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"articles_{run_id}"
    db_path = "/home/user/myproject/lancedb"
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    
    results = table.search(vec).where(f"year = {year}").limit(k).to_list()
    return results
