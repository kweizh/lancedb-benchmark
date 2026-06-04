import lancedb
import pandas as pd


def diff_versions(v_a: int, v_b: int) -> dict:
    """Compare two versions of the customers table and return the diff.

    Returns a dict with keys:
      - added: list of ids present in v_b but absent from v_a
      - removed: list of ids present in v_a but absent from v_b
      - modified: list of dicts describing rows whose non-id columns changed
    """
    db = lancedb.connect("/data/lancedb")
    table = db.open_table("customers")

    # Read version v_a
    table.checkout(v_a)
    df_a = table.to_pandas()

    # Read version v_b
    table.checkout(v_b)
    df_b = table.to_pandas()

    # Restore to latest so we don't leave the table in a checked-out state
    table.checkout_latest()

    # Index by id for fast lookup
    df_a_indexed = df_a.set_index("id")
    df_b_indexed = df_b.set_index("id")

    ids_a = set(df_a_indexed.index)
    ids_b = set(df_b_indexed.index)

    added = sorted(ids_b - ids_a)
    removed = sorted(ids_a - ids_b)
    common = sorted(ids_a & ids_b)

    non_id_cols = [c for c in df_a.columns if c != "id"]

    modified = []
    for row_id in common:
        row_a = df_a_indexed.loc[row_id]
        row_b = df_b_indexed.loc[row_id]

        # If there are duplicate ids, .loc returns a DataFrame/Series;
        # take the first entry if needed.
        if isinstance(row_a, pd.DataFrame):
            row_a = row_a.iloc[0]
        if isinstance(row_b, pd.DataFrame):
            row_b = row_b.iloc[0]

        old_vals = {c: row_a[c] for c in non_id_cols}
        new_vals = {c: row_b[c] for c in non_id_cols}

        # Compare values — treat numeric as float with tolerance
        differs = False
        for c in non_id_cols:
            v_old = old_vals[c]
            v_new = new_vals[c]
            if isinstance(v_old, float) or isinstance(v_new, float):
                if abs(float(v_old) - float(v_new)) > 1e-9:
                    differs = True
                    break
            else:
                if v_old != v_new:
                    differs = True
                    break

        if differs:
            # Convert numpy types to native Python types for JSON serialization
            modified.append({
                "id": int(row_id),
                "old": {c: _to_native(old_vals[c]) for c in non_id_cols},
                "new": {c: _to_native(new_vals[c]) for c in non_id_cols},
            })

    return {
        "added": [int(x) for x in added],
        "removed": [int(x) for x in removed],
        "modified": modified,
    }


def _to_native(val):
    """Convert numpy/pandas scalar to a native Python type for JSON serialization."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.str_,)):
        return str(val)
    return val