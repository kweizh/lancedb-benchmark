import importlib.util
import os
import sys

import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"
DELTA_PATH = "/app/delta_data/products"


def _run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set."
    return rid


@pytest.fixture(scope="module")
def db():
    import lancedb

    return lancedb.connect(LANCEDB_DIR)


@pytest.fixture(scope="module")
def ground_truth():
    """Recompute expected diff between Delta versions 0 and 2 from the source table."""
    from deltalake import DeltaTable

    dt0 = DeltaTable(DELTA_PATH, version=0)
    dt2 = DeltaTable(DELTA_PATH, version=2)

    df0 = dt0.to_pyarrow_table().to_pandas()
    df2 = dt2.to_pyarrow_table().to_pandas()

    ids0 = set(int(x) for x in df0["id"].tolist())
    ids2 = set(int(x) for x in df2["id"].tolist())

    added = sorted(ids2 - ids0)
    removed = sorted(ids0 - ids2)

    cat0 = {int(r["id"]): r["category"] for _, r in df0.iterrows()}
    cat2 = {int(r["id"]): r["category"] for _, r in df2.iterrows()}
    modified = sorted(i for i in (ids0 & ids2) if cat0[i] != cat2[i])
    return {"added": added, "removed": removed, "modified": modified}


def test_products_table_exists(db):
    rid = _run_id()
    name = f"products_{rid}"
    names = db.table_names()
    assert name in names, f"Expected LanceDB table {name}, found tables: {names}"


def test_products_table_row_count(db):
    rid = _run_id()
    name = f"products_{rid}"
    tbl = db.open_table(name)
    assert tbl.count_rows() == 800, f"Expected 800 rows in {name}, got {tbl.count_rows()}."


def test_products_table_ids_complete(db):
    rid = _run_id()
    tbl = db.open_table(f"products_{rid}")
    df = tbl.to_pandas()
    ids = set(int(x) for x in df["id"].tolist())
    assert ids == set(range(800)), "Expected ids 0..799 in migrated products table."


def test_products_vector_is_fixed_size_list_float32_32(db):
    rid = _run_id()
    tbl = db.open_table(f"products_{rid}")
    schema = tbl.schema
    field = schema.field("vector")
    assert pa.types.is_fixed_size_list(field.type), (
        f"vector column must be fixed_size_list, got {field.type}"
    )
    assert field.type.list_size == 32, (
        f"vector column must have list_size 32, got {field.type.list_size}"
    )
    assert field.type.value_type == pa.float32(), (
        f"vector value type must be float32, got {field.type.value_type}"
    )


def test_migration_audit_table_exists(db):
    rid = _run_id()
    name = f"migration_audit_{rid}"
    assert name in db.table_names(), (
        f"Expected LanceDB table {name}, found tables: {db.table_names()}"
    )


def test_migration_audit_schema(db):
    rid = _run_id()
    tbl = db.open_table(f"migration_audit_{rid}")
    field_names = set(tbl.schema.names)
    required = {"id", "change", "version_a", "version_b"}
    assert required.issubset(field_names), (
        f"Expected columns {required} in migration_audit, got {field_names}."
    )


def test_migration_audit_versions(db):
    rid = _run_id()
    tbl = db.open_table(f"migration_audit_{rid}")
    df = tbl.to_pandas()
    assert (df["version_a"] == 0).all(), "All audit rows must have version_a == 0."
    assert (df["version_b"] == 2).all(), "All audit rows must have version_b == 2."


def test_migration_audit_content_matches_ground_truth(db, ground_truth):
    rid = _run_id()
    tbl = db.open_table(f"migration_audit_{rid}")
    df = tbl.to_pandas()

    actual_added = sorted(int(x) for x in df[df["change"] == "added"]["id"].tolist())
    actual_removed = sorted(int(x) for x in df[df["change"] == "removed"]["id"].tolist())
    actual_modified = sorted(int(x) for x in df[df["change"] == "modified"]["id"].tolist())

    assert actual_added == ground_truth["added"], (
        f"Audit 'added' ids mismatch.\nExpected: {ground_truth['added'][:10]}... ({len(ground_truth['added'])} total)"
        f"\nGot: {actual_added[:10]}... ({len(actual_added)} total)"
    )
    assert actual_removed == ground_truth["removed"], (
        f"Audit 'removed' ids mismatch. Expected {ground_truth['removed']}, got {actual_removed}."
    )
    assert actual_modified == ground_truth["modified"], (
        f"Audit 'modified' ids mismatch.\nExpected: {ground_truth['modified'][:10]}... ({len(ground_truth['modified'])} total)"
        f"\nGot: {actual_modified[:10]}... ({len(actual_modified)} total)"
    )


def test_solution_historical_compare_callable(ground_truth):
    sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location(
        "solution", os.path.join(PROJECT_DIR, "solution.py")
    )
    assert spec is not None, "solution.py module spec could not be loaded."
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "historical_compare"), (
        "solution.py must expose a callable named historical_compare."
    )
    result = module.historical_compare(DELTA_PATH, 0, 2)
    assert isinstance(result, dict), f"historical_compare must return a dict, got {type(result)}."
    for key in ("added", "removed", "modified"):
        assert key in result, f"historical_compare result missing key '{key}'."
        assert isinstance(result[key], list), f"result['{key}'] must be a list."

    assert sorted(int(x) for x in result["added"]) == ground_truth["added"], (
        "historical_compare 'added' does not match Delta ground truth."
    )
    assert sorted(int(x) for x in result["removed"]) == ground_truth["removed"], (
        "historical_compare 'removed' does not match Delta ground truth."
    )
    assert sorted(int(x) for x in result["modified"]) == ground_truth["modified"], (
        "historical_compare 'modified' does not match Delta ground truth."
    )
