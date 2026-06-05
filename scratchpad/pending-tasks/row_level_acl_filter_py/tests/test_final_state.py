"""Final-state verification for the row_level_acl_filter_py LanceDB task."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "data")
SOLUTION_FILE = os.path.join(PROJECT_DIR, "solution.py")

CANONICAL_USERS = [
    {"user_id": "user_00", "roles": ["team:team_0", "team:team_1"]},
    {"user_id": "user_01", "roles": ["team:team_2"]},
    {"user_id": "user_02", "roles": []},
    {"user_id": "user_03", "roles": ["team:team_0", "team:team_3", "team:team_5"]},
    {"user_id": "user_04", "roles": ["admin"]},
]
TOP_K = 10


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set in the environment for verification."
    return rid


def _table_name() -> str:
    return f"documents_{_run_id()}"


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    assert os.path.isfile(SOLUTION_FILE), f"Missing {SOLUTION_FILE}."
    # Force a fresh import so the test does not pick up a stale cached module.
    if "solution" in sys.modules:
        del sys.modules["solution"]
    mod = importlib.import_module("solution")
    assert hasattr(mod, "ACLSearch"), "solution.py must expose ACLSearch."
    return mod


@pytest.fixture(scope="session")
def seeded_rows():
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(_table_name())
    df = tbl.to_pandas()
    assert len(df) == 300, f"Expected 300 seeded rows, found {len(df)}."
    # Materialize a numpy vector matrix for brute force.
    vectors = np.stack([np.asarray(v, dtype=np.float32) for v in df["vector"].tolist()])
    return df, vectors


@pytest.fixture(scope="session")
def canonical_queries():
    rng = np.random.default_rng(7777)
    return rng.standard_normal((5, 32)).astype(np.float32)


def _allowed_teams(roles):
    return {r.split(":", 1)[1] for r in roles if r.startswith("team:")}


def _brute_force_visible_topk(df, vectors, query_vec, user_id, roles, k):
    allowed = _allowed_teams(roles)
    visibility = df["visibility"].to_numpy()
    owner_id = df["owner_id"].to_numpy()
    team_id = df["team_id"].to_numpy()

    public_mask = visibility == "public"
    private_mask = (visibility == "private") & (owner_id == user_id)
    if allowed:
        team_mask = (visibility == "team") & np.isin(team_id, list(allowed))
    else:
        team_mask = np.zeros(len(df), dtype=bool)
    visible_mask = public_mask | private_mask | team_mask
    visible_idx = np.flatnonzero(visible_mask)
    if visible_idx.size == 0:
        return [], visible_mask
    diff = vectors[visible_idx] - query_vec[None, :]
    dists = np.einsum("ij,ij->i", diff, diff)
    ids = df["id"].to_numpy()[visible_idx]
    # Sort by (distance ASC, id ASC) for a deterministic tie-break.
    order = np.lexsort((ids, dists))
    top_local = order[:k]
    top_ids = ids[top_local].tolist()
    return top_ids, visible_mask


def test_solution_uses_server_side_filtering():
    """Guardrail: the source must show prefilter=True and a `.where(` call."""
    src = Path(SOLUTION_FILE).read_text()
    assert "prefilter=True" in src, (
        "solution.py must call LanceDB with prefilter=True to enforce server-side ACL."
    )
    assert ".where(" in src, "solution.py must use .where(...) to filter on LanceDB."


def test_where_clause_contains_public_branch(solution_module):
    for user in CANONICAL_USERS:
        inst = solution_module.ACLSearch(user["user_id"], user["roles"])
        clause = inst.build_where_clause()
        assert isinstance(clause, str) and clause.strip(), (
            f"build_where_clause() must return a non-empty SQL string (user={user})."
        )
        assert "visibility" in clause and "'public'" in clause, (
            f"WHERE clause must include the public-visibility disjunct. Got: {clause}"
        )


def test_where_clause_user_specific_predicates(solution_module):
    # U1: has team roles + a user_id that may match owner_id rows.
    u1 = CANONICAL_USERS[0]
    c1 = solution_module.ACLSearch(u1["user_id"], u1["roles"]).build_where_clause()
    assert "'user_00'" in c1, f"WHERE for U1 must mention 'user_00'. Got: {c1}"
    assert "'team_0'" in c1 and "'team_1'" in c1, (
        f"WHERE for U1 must reference allowed teams. Got: {c1}"
    )

    # U3: no roles. team predicate must collapse — team_id literal MUST NOT appear.
    u3 = CANONICAL_USERS[2]
    c3 = solution_module.ACLSearch(u3["user_id"], u3["roles"]).build_where_clause()
    # No team literals like 'team_X' may leak into the clause.
    for i in range(10):
        assert f"'team_{i}'" not in c3, (
            f"WHERE for empty-role user must NOT reference team_id literals. Got: {c3}"
        )

    # U5: 'admin' role only -> behave like empty team set.
    u5 = CANONICAL_USERS[4]
    c5 = solution_module.ACLSearch(u5["user_id"], u5["roles"]).build_where_clause()
    for i in range(10):
        assert f"'team_{i}'" not in c5, (
            f"WHERE for admin-only user must NOT reference team_id literals. Got: {c5}"
        )


def test_search_returns_brute_force_topk_set(
    solution_module, seeded_rows, canonical_queries
):
    df, vectors = seeded_rows
    for user, qv in zip(CANONICAL_USERS, canonical_queries):
        inst = solution_module.ACLSearch(user["user_id"], user["roles"])
        results = inst.search(qv.tolist(), TOP_K)
        assert isinstance(results, list), (
            f"search(...) must return a list, got {type(results)} for user {user}."
        )
        for row in results:
            for key in ("id", "owner_id", "visibility", "team_id"):
                assert key in row, (
                    f"Each result dict must contain '{key}'. Got: {row}"
                )
        returned_ids = sorted(int(r["id"]) for r in results)
        gt_ids, _ = _brute_force_visible_topk(
            df, vectors, qv, user["user_id"], user["roles"], TOP_K
        )
        assert set(returned_ids) == set(gt_ids), (
            f"Top-{TOP_K} id set mismatch for user {user['user_id']}.\n"
            f"  returned={returned_ids}\n  expected={sorted(gt_ids)}"
        )


def test_zero_leakage(solution_module, seeded_rows, canonical_queries):
    df, vectors = seeded_rows
    df_indexed = df.set_index("id")
    for user, qv in zip(CANONICAL_USERS, canonical_queries):
        inst = solution_module.ACLSearch(user["user_id"], user["roles"])
        results = inst.search(qv.tolist(), TOP_K)
        allowed = _allowed_teams(user["roles"])
        for row in results:
            seeded = df_indexed.loc[int(row["id"])]
            vis = str(seeded["visibility"])
            owner = str(seeded["owner_id"])
            tid = str(seeded["team_id"])
            if vis == "private":
                assert owner == user["user_id"], (
                    f"LEAK: private row {row['id']} owned by {owner!r} "
                    f"returned to user {user['user_id']!r}."
                )
            elif vis == "team":
                assert tid in allowed, (
                    f"LEAK: team row {row['id']} on team {tid!r} returned to user "
                    f"{user['user_id']!r} whose allowed teams are {allowed!r}."
                )
            elif vis != "public":
                pytest.fail(f"Unexpected visibility value {vis!r} on row {row['id']}.")


def test_empty_roles_returns_only_public(
    solution_module, seeded_rows, canonical_queries
):
    df, vectors = seeded_rows
    # user_99 does not appear in any owner_id in the seeded data.
    assert "user_99" not in set(df["owner_id"].unique()), (
        "Seed precondition broken: user_99 should not be an owner_id."
    )
    qv = canonical_queries[0]
    inst = solution_module.ACLSearch("user_99", [])
    results = inst.search(qv.tolist(), 50)
    assert len(results) > 0, "Empty-role search should still return public rows."
    for row in results:
        assert row["visibility"] == "public", (
            f"Empty-role user must only receive public rows; got {row}"
        )

    # Verify exact match against brute-force top-50 of the public subset.
    _, vectors = seeded_rows
    gt_ids, _ = _brute_force_visible_topk(df, vectors, qv, "user_99", [], 50)
    returned_ids = sorted(int(r["id"]) for r in results)
    assert set(returned_ids) == set(gt_ids), (
        f"Empty-role top-50 id set mismatch.\n"
        f"  returned={returned_ids}\n  expected={sorted(gt_ids)}"
    )
