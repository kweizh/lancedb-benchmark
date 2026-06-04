import os
import lancedb

def test_search():
    db = lancedb.connect("/home/user/myproject/data")
    table_name = os.environ.get("LANCE_TABLE", "articles")
    table = db.open_table(table_name)
    res = table.head(1).to_pylist()
    print(res)

if __name__ == "__main__":
    test_search()
