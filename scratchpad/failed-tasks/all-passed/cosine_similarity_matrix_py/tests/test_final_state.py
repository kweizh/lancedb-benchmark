import importlib
import os
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"


@pytest.fixture(scope="module")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    mod = importlib.import_module("solution")
    return mod


@pytest.fixture(scope="module")
def similarity_matrix(solution_module):
    assert hasattr(solution_module, "similarity_matrix"), (
        "solution.py must expose a top-level `similarity_matrix` function."
    )
    S = solution_module.similarity_matrix()
    assert isinstance(S, np.ndarray), (
        f"similarity_matrix() must return a numpy.ndarray, got {type(S).__name__}."
    )
    return S


@pytest.fixture(scope="module")
def lancedb_table():
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    table_name = os.environ["LANCEDB_TABLE"]
    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)
    return tbl


@pytest.fixture(scope="module")
def table_rows(lancedb_table):
    rows = lancedb_table.to_pandas()
    rows = rows.sort_values("id").reset_index(drop=True)
    return rows


def test_matrix_shape_and_dtype(similarity_matrix):
    assert similarity_matrix.shape == (200, 200), (
        f"similarity_matrix() must have shape (200, 200), got {similarity_matrix.shape}."
    )
    assert similarity_matrix.dtype.kind == "f", (
        f"similarity_matrix() must have a floating-point dtype, got {similarity_matrix.dtype}."
    )
    assert not np.isnan(similarity_matrix).any(), "similarity_matrix() contains NaNs."
    assert not np.isinf(similarity_matrix).any(), "similarity_matrix() contains inf."


def test_matrix_is_symmetric(similarity_matrix):
    assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-3), (
        "similarity_matrix() must be symmetric within atol=1e-3."
    )


def test_matrix_diagonal_is_one(similarity_matrix):
    diag = np.diag(similarity_matrix)
    assert np.allclose(diag, 1.0, atol=1e-3), (
        f"Diagonal entries must all be ~1.0; got min={diag.min()}, max={diag.max()}."
    )


def test_matrix_value_range(similarity_matrix):
    assert similarity_matrix.min() >= -1.0 - 1e-3, (
        f"All entries must be >= -1.0, got min={similarity_matrix.min()}."
    )
    assert similarity_matrix.max() <= 1.0 + 1e-3, (
        f"All entries must be <= 1.0, got max={similarity_matrix.max()}."
    )


def test_intra_class_mean_matches_numpy(solution_module, similarity_matrix, table_rows):
    labels = table_rows["label"].to_numpy()
    n = similarity_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    for L in range(5):
        idx = np.where(labels == L)[0]
        assert len(idx) > 0, f"No rows with label {L} found in seeded table."
        sub = similarity_matrix[np.ix_(idx, idx)]
        sub_off = sub[~np.eye(len(idx), dtype=bool)]
        expected = float(sub_off.mean())
        actual = float(solution_module.intra_class_mean(int(L)))
        assert abs(actual - expected) < 1e-3, (
            f"intra_class_mean({L}) = {actual}; expected {expected} (within 1e-3)."
        )


def test_cluster_structure_preserved(solution_module, similarity_matrix, table_rows):
    labels = table_rows["label"].to_numpy()
    n = similarity_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    global_mean = float(similarity_matrix[off_diag_mask].mean())
    passed = 0
    margins = {}
    for L in range(5):
        intra = float(solution_module.intra_class_mean(int(L)))
        margins[L] = intra - global_mean
        if intra > global_mean + 0.05:
            passed += 1
    assert passed >= 4, (
        f"At least 4/5 labels must have intra_class_mean exceeding the global "
        f"off-diagonal mean by >0.05. global_mean={global_mean}, margins={margins}."
    )


def test_matrix_matches_direct_lancedb_cosine(similarity_matrix, lancedb_table, table_rows):
    """Provenance: compare a sample of candidate entries against fresh LanceDB cosine searches."""
    rng = np.random.default_rng(7)
    sample_ids = rng.choice(200, size=20, replace=False)
    id_to_idx = {int(r["id"]): i for i, r in table_rows.iterrows()}
    vectors_by_id = {int(r["id"]): np.asarray(r["vector"], dtype=np.float32) for _, r in table_rows.iterrows()}

    for i in sample_ids:
        qvec = vectors_by_id[int(i)]
        results = (
            lancedb_table.search(qvec)
            .distance_type("cosine")
            .limit(200)
            .to_list()
        )
        assert len(results) == 200, (
            f"LanceDB returned {len(results)} rows for id={i}; expected 200."
        )
        # Verify candidate matrix matches LanceDB's reported similarity for each neighbour.
        for r in results:
            j = int(r["id"])
            expected_sim = 1.0 - float(r["_distance"])
            candidate_sim = float(similarity_matrix[int(i), j])
            assert abs(candidate_sim - expected_sim) < 1e-3, (
                f"S[{i},{j}]={candidate_sim} does not match LanceDB cosine "
                f"similarity {expected_sim} (|delta|>1e-3). This indicates the "
                f"matrix was not derived from LanceDB cosine search results."
            )
