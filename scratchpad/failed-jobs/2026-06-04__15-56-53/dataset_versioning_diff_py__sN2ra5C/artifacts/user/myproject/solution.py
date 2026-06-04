import lancedb

DB_PATH = "/data/lancedb"
TABLE_NAME = "customers"
NON_ID_COLS = ["name", "price", "category"]


def diff_versions(v_a: int, v_b: int) -> dict:
    """
    Compare two versions of the 'customers' LanceDB table.

    Returns a dict with:
      - added:    list of int ids present in v_b but not v_a
      - removed:  list of int ids present in v_a but not v_b
      - modified: list of {"id": int, "old": {...}, "new": {...}} for rows
                  present in both versions whose non-id columns differ
    """
    # Open two independent connections so in-place checkout calls don't
    # interfere with each other, and the on-disk table is never modified.
    db_a = lancedb.connect(DB_PATH)
    tbl_a = db_a.open_table(TABLE_NAME)
    tbl_a.checkout(v_a)
    df_a = tbl_a.to_pandas()

    db_b = lancedb.connect(DB_PATH)
    tbl_b = db_b.open_table(TABLE_NAME)
    tbl_b.checkout(v_b)
    df_b = tbl_b.to_pandas()

    ids_a = set(df_a["id"].tolist())
    ids_b = set(df_b["id"].tolist())

    added = sorted(ids_b - ids_a)
    removed = sorted(ids_a - ids_b)

    # Build id-keyed lookup dicts for rows present in both versions
    common_ids = ids_a & ids_b

    rows_a = df_a[df_a["id"].isin(common_ids)].set_index("id")
    rows_b = df_b[df_b["id"].isin(common_ids)].set_index("id")

    modified = []
    for row_id in common_ids:
        row_old = rows_a.loc[row_id]
        row_new = rows_b.loc[row_id]

        changed = False
        for col in NON_ID_COLS:
            val_old = row_old[col]
            val_new = row_new[col]
            # Use string comparison for non-numeric; float comparison otherwise
            if isinstance(val_old, float) or isinstance(val_new, float):
                if float(val_old) != float(val_new):
                    changed = True
                    break
            else:
                if val_old != val_new:
                    changed = True
                    break

        if changed:
            old_vals = {}
            new_vals = {}
            for col in NON_ID_COLS:
                v = row_old[col]
                old_vals[col] = float(v) if isinstance(v, float) else v
                v = row_new[col]
                new_vals[col] = float(v) if isinstance(v, float) else v

            modified.append({
                "id": int(row_id),
                "old": old_vals,
                "new": new_vals,
            })

    modified.sort(key=lambda x: x["id"])

    return {
        "added": [int(i) for i in added],
        "removed": [int(i) for i in removed],
        "modified": modified,
    }
