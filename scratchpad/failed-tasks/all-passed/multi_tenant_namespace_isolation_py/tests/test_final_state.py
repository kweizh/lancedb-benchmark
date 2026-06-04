import os
import sys
import importlib.util
import tempfile

import numpy as np
import pytest
import lancedb


PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")

TENANTS = ["acme", "globex", "umbrella"]
ROWS_PER_TENANT = 30
EMBED_DIM = 32
SEED = 2026


def _load_solution_module():
    assert os.path.isfile(SOLUTION_PATH), (
        f"solution.py was not found at {SOLUTION_PATH}; the candidate must create it."
    )
    spec = importlib.util.spec_from_file_location(
        "candidate_solution", SOLUTION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solution"] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "TenantDB"), (
        "solution.py must export a class named `TenantDB`."
    )
    return module


def _generate_rows(tenant_id, rng):
    rows = []
    for i in range(ROWS_PER_TENANT):
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32).tolist()
        rows.append(
            {
                "id": f"{tenant_id}-{i:03d}",
                "text": f"document {i} for tenant {tenant_id}",
                "embedding": vec,
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    return rows


@pytest.fixture(scope="module")
def candidate_module():
    return _load_solution_module()


@pytest.fixture()
def fresh_setup(tmp_path, candidate_module):
    """Create a fresh LanceDB connection and seed all three tenants."""
    db_dir = tmp_path / "tenant_db"
    db_dir.mkdir()
    conn = lancedb.connect(str(db_dir))

    TenantDB = candidate_module.TenantDB

    rng = np.random.default_rng(SEED)
    tenant_rows = {}
    tenants = {}
    for tenant_id in TENANTS:
        rows = _generate_rows(tenant_id, rng)
        tenant_rows[tenant_id] = rows
        wrapper = TenantDB(conn, tenant_id)
        wrapper.create_documents_table(rows)
        tenants[tenant_id] = wrapper

    return {
        "conn": conn,
        "tenants": tenants,
        "rows": tenant_rows,
        "TenantDB": TenantDB,
    }


def test_tenants_are_in_physically_separate_tables(fresh_setup):
    """Each tenant's rows must be stored in its own table, not commingled."""
    conn = fresh_setup["conn"]
    names = list(conn.table_names())
    # There must be at least one distinct table per tenant.
    matched = {}
    for tenant_id in TENANTS:
        candidates = [n for n in names if tenant_id in n]
        assert len(candidates) >= 1, (
            f"Expected at least one table whose name encodes tenant '{tenant_id}', "
            f"found tables: {names}"
        )
        matched[tenant_id] = candidates[0]

    # All matched tables must be distinct (no shared table).
    assert len(set(matched.values())) == len(TENANTS), (
        f"Tenants must occupy distinct tables, got mapping: {matched}"
    )

    # Each matched table should contain exactly ROWS_PER_TENANT rows.
    for tenant_id, table_name in matched.items():
        tbl = conn.open_table(table_name)
        df = tbl.to_pandas()
        assert len(df) == ROWS_PER_TENANT, (
            f"Tenant '{tenant_id}' table '{table_name}' should have "
            f"{ROWS_PER_TENANT} rows, found {len(df)}."
        )
        ids = set(df["id"].tolist())
        expected_ids = {row["id"] for row in fresh_setup["rows"][tenant_id]}
        assert ids == expected_ids, (
            f"Tenant '{tenant_id}' table '{table_name}' has unexpected ids; "
            f"missing={expected_ids - ids}, extra={ids - expected_ids}"
        )
        # Reject cross-tenant ids appearing inside another tenant's physical table.
        for other in TENANTS:
            if other == tenant_id:
                continue
            other_prefix = f"{other}-"
            leaked = [i for i in ids if i.startswith(other_prefix)]
            assert not leaked, (
                f"Table '{table_name}' for tenant '{tenant_id}' contains ids "
                f"from tenant '{other}': {leaked}"
            )


def test_list_tenants_returns_sorted_ids(fresh_setup):
    TenantDB = fresh_setup["TenantDB"]
    listed = TenantDB.list_tenants(fresh_setup["conn"])
    assert isinstance(listed, list), (
        f"TenantDB.list_tenants must return a list, got {type(listed).__name__}."
    )
    assert listed == sorted(TENANTS), (
        f"TenantDB.list_tenants(connection) must return {sorted(TENANTS)}, "
        f"got {listed}."
    )


def test_search_is_isolated_per_tenant(fresh_setup):
    rows = fresh_setup["rows"]
    tenants = fresh_setup["tenants"]
    for tenant_id in TENANTS:
        # Use one of this tenant's own vectors as the query.
        query_vec = rows[tenant_id][0]["embedding"]
        results = tenants[tenant_id].search(query_vec, k=10)
        assert isinstance(results, list), (
            f"search() must return a list, got {type(results).__name__}."
        )
        assert len(results) > 0, (
            f"search() returned no results for tenant '{tenant_id}'."
        )
        assert len(results) <= 10, (
            f"search(k=10) returned more than 10 rows for tenant '{tenant_id}'."
        )
        for row in results:
            assert "id" in row and "text" in row, (
                "Each search result row must contain at least `id` and `text`, "
                f"got keys: {list(row.keys())}"
            )
            assert row["id"].startswith(f"{tenant_id}-"), (
                f"search() for tenant '{tenant_id}' returned foreign id "
                f"'{row['id']}'."
            )


def test_add_documents_rejects_cross_tenant_id_collision(fresh_setup):
    rng = np.random.default_rng(SEED + 1)
    rows = fresh_setup["rows"]
    tenants = fresh_setup["tenants"]
    conn = fresh_setup["conn"]

    foreign_id = rows["acme"][0]["id"]  # acme-000 is owned by acme.
    bad_vec = rng.standard_normal(EMBED_DIM).astype(np.float32).tolist()
    bad_row = {
        "id": foreign_id,
        "text": "attempted cross-tenant write",
        "embedding": bad_vec,
        "created_at": "2026-02-02T00:00:00Z",
    }

    # globex tries to claim an id that already belongs to acme.
    with pytest.raises(PermissionError):
        tenants["globex"].add_documents([bad_row])

    # globex's table must not have changed.
    names = list(conn.table_names())
    globex_tables = [n for n in names if "globex" in n]
    assert globex_tables, "globex table disappeared after rejected write."
    tbl = conn.open_table(globex_tables[0])
    assert len(tbl.to_pandas()) == ROWS_PER_TENANT, (
        "globex table row count must be unchanged after a rejected add_documents."
    )

    # A legitimate non-colliding write must succeed and grow the table.
    good_vec = rng.standard_normal(EMBED_DIM).astype(np.float32).tolist()
    good_row = {
        "id": "globex-new-1",
        "text": "legitimate new globex row",
        "embedding": good_vec,
        "created_at": "2026-02-02T00:00:00Z",
    }
    tenants["globex"].add_documents([good_row])
    tbl = conn.open_table(globex_tables[0])
    assert len(tbl.to_pandas()) == ROWS_PER_TENANT + 1, (
        "globex table should grow by exactly one row after a legitimate "
        "add_documents call."
    )


def test_delete_tenant_isolates_other_tenants(fresh_setup):
    TenantDB = fresh_setup["TenantDB"]
    conn = fresh_setup["conn"]
    tenants = fresh_setup["tenants"]
    rows = fresh_setup["rows"]

    tenants["globex"].delete_tenant()

    # globex must no longer appear in list_tenants.
    listed = TenantDB.list_tenants(conn)
    assert "globex" not in listed, (
        f"After delete_tenant(), 'globex' must not be listed; got {listed}."
    )
    assert sorted(listed) == ["acme", "umbrella"], (
        f"After dropping globex, remaining tenants must be ['acme', 'umbrella']; "
        f"got {listed}."
    )

    # globex's underlying table must be gone.
    names = list(conn.table_names())
    assert not any("globex" in n for n in names), (
        f"globex's physical table must be removed from the connection; "
        f"table_names() returned: {names}"
    )

    # Remaining tenants must still work correctly.
    for tenant_id in ["acme", "umbrella"]:
        query_vec = rows[tenant_id][0]["embedding"]
        results = tenants[tenant_id].search(query_vec, k=5)
        assert len(results) > 0, (
            f"Surviving tenant '{tenant_id}' must still return search results."
        )
        for row in results:
            assert row["id"].startswith(f"{tenant_id}-"), (
                f"Surviving tenant '{tenant_id}' returned a foreign id "
                f"'{row['id']}'."
            )
