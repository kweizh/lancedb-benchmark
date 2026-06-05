import importlib.util
import os
import sys

import numpy as np
import pyarrow as pa
import pytest


PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
PCA_MODEL_PATH = "/app/pca_model.npz"


@pytest.fixture(scope="session")
def run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID env var must be set for verification."
    return rid


@pytest.fixture(scope="session")
def db():
    import lancedb

    return lancedb.connect(LANCEDB_DIR)


@pytest.fixture(scope="session")
def new_table_name(run_id):
    return f"articles_pca_{run_id}"


@pytest.fixture(scope="session")
def src_tbl(db):
    return db.open_table("articles")


@pytest.fixture(scope="session")
def pca_tbl(db, new_table_name):
    assert new_table_name in db.table_names(), (
        f"Expected new PCA table '{new_table_name}' in {db.table_names()}."
    )
    return db.open_table(new_table_name)


@pytest.fixture(scope="session")
def pca_model():
    assert os.path.isfile(PCA_MODEL_PATH), (
        f"PCA model artifact missing at {PCA_MODEL_PATH}."
    )
    data = np.load(PCA_MODEL_PATH)
    assert set(data.files) == {"components", "mean"}, (
        f"/app/pca_model.npz must contain exactly arrays 'components' and 'mean'; got {data.files}."
    )
    components = np.asarray(data["components"], dtype=np.float64)
    mean = np.asarray(data["mean"], dtype=np.float64)
    assert components.shape == (16, 128), (
        f"components must have shape (16, 128); got {components.shape}."
    )
    assert mean.shape == (128,), f"mean must have shape (128,); got {mean.shape}."
    return components, mean


@pytest.fixture(scope="session")
def src_df(src_tbl):
    df = src_tbl.to_pandas()
    return df


@pytest.fixture(scope="session")
def src_vectors(src_df):
    # Convert embedding column to a 2-d numpy array (n, 128).
    vecs = np.stack([np.asarray(v, dtype=np.float64) for v in src_df["embedding"].tolist()])
    assert vecs.shape == (600, 128), f"Source vectors must be (600, 128); got {vecs.shape}."
    return vecs


@pytest.fixture(scope="session")
def solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"Missing {SOLUTION_PATH}."
    spec = importlib.util.spec_from_file_location("candidate_solution", SOLUTION_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Make sure candidate code can import its own helpers from /home/user/myproject.
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "search"), "solution.py must define a top-level `search` callable."
    return mod


# ---------------------------------------------------------------------------
# Step 1: New table exists, has 600 rows, correct schema.
# ---------------------------------------------------------------------------


def test_new_table_has_600_rows(pca_tbl):
    n = pca_tbl.count_rows()
    assert n == 600, f"PCA table must have exactly 600 rows; got {n}."


def test_new_table_schema(pca_tbl):
    schema = pca_tbl.schema
    fields = {f.name: f.type for f in schema}
    for required in ("id", "title", "embedding", "original_id"):
        assert required in fields, (
            f"PCA table schema missing required column '{required}'; got columns {list(fields)}."
        )

    emb_type = fields["embedding"]
    assert pa.types.is_fixed_size_list(emb_type), (
        f"PCA 'embedding' must be a fixed-size list; got {emb_type}."
    )
    assert emb_type.list_size == 16, (
        f"PCA 'embedding' must have list_size==16; got {emb_type.list_size}."
    )
    value_type = emb_type.value_type
    assert pa.types.is_floating(value_type), (
        f"PCA 'embedding' element type must be a float type; got {value_type}."
    )
    # Specifically expect float32 for compactness.
    assert pa.types.is_float32(value_type), (
        f"PCA 'embedding' element type must be float32; got {value_type}."
    )


def test_new_table_original_id_covers_source_ids(pca_tbl, src_df):
    df = pca_tbl.to_pandas()
    src_ids = set(int(x) for x in src_df["id"].tolist())
    pca_ids = set(int(x) for x in df["original_id"].tolist())
    assert pca_ids == src_ids, (
        f"PCA 'original_id' set must equal source 'id' set. "
        f"missing={src_ids - pca_ids}, extra={pca_ids - src_ids}."
    )


# ---------------------------------------------------------------------------
# Step 2: PCA model artifact is valid (shape checked in fixture).
# ---------------------------------------------------------------------------


def test_pca_model_artifact_loadable(pca_model):
    components, mean = pca_model
    # Sanity: components rows are roughly unit-norm (PCA components_ are orthonormal).
    norms = np.linalg.norm(components, axis=1)
    # Allow some slack in case the candidate scaled differently — but they should not be zero.
    assert np.all(norms > 1e-3), f"PCA components have near-zero rows: norms={norms}."


# ---------------------------------------------------------------------------
# Step 3: Re-projection consistency — applying the saved PCA to a source vector
# must reproduce the stored 16-d vector in the new table.
# ---------------------------------------------------------------------------


def test_reprojection_matches_stored(pca_tbl, src_df, pca_model):
    components, mean = pca_model
    # Pick the row with the smallest original_id present in both tables.
    pca_df = pca_tbl.to_pandas()
    chosen_id = int(pca_df["original_id"].min())

    src_row = src_df[src_df["id"] == chosen_id]
    assert len(src_row) == 1, f"Could not locate source row id={chosen_id}."
    src_vec = np.asarray(src_row["embedding"].iloc[0], dtype=np.float64)

    pca_row = pca_df[pca_df["original_id"] == chosen_id]
    assert len(pca_row) == 1, f"Could not locate PCA row original_id={chosen_id}."
    stored = np.asarray(pca_row["embedding"].iloc[0], dtype=np.float64)
    assert stored.shape == (16,), f"Stored embedding must be 16-d; got {stored.shape}."

    proj = (src_vec - mean) @ components.T
    # Compare directly; sign convention of components must match what the candidate stored.
    assert np.allclose(proj, stored, atol=1e-3), (
        f"Re-projected vector does not match stored vector for original_id={chosen_id}. "
        f"max_abs_diff={np.max(np.abs(proj - stored))}."
    )


# ---------------------------------------------------------------------------
# Step 4 & 5: Candidate's search() returns the correct shape AND has substantial
# overlap with the brute-force 128-d ground-truth top-5.
# ---------------------------------------------------------------------------


def _make_query():
    rng = np.random.default_rng(7)
    return rng.standard_normal(128).astype(np.float32)


def test_search_returns_correct_shape(solution_module):
    query = _make_query()
    result = solution_module.search(query, 5)
    assert isinstance(result, list), f"search() must return a list; got {type(result).__name__}."
    assert len(result) == 5, f"search(.., k=5) must return 5 results; got {len(result)}."
    for i, item in enumerate(result):
        assert isinstance(item, dict), f"result[{i}] must be a dict; got {type(item).__name__}."
        for required_key in ("id", "title", "original_id"):
            assert required_key in item, (
                f"result[{i}] missing required key '{required_key}'. keys={list(item)}."
            )
        assert isinstance(item["id"], int), f"result[{i}]['id'] must be int."
        assert isinstance(item["title"], str), f"result[{i}]['title'] must be str."
        assert isinstance(item["original_id"], int), f"result[{i}]['original_id'] must be int."


def test_search_overlaps_with_128d_ground_truth(solution_module, src_df, src_vectors):
    query = _make_query()
    # Ground truth in original 128-d space using L2 distance.
    diffs = src_vectors - query.astype(np.float64)
    dists = np.einsum("ij,ij->i", diffs, diffs)
    order = np.argsort(dists)[:5]
    gt_top5 = set(int(src_df["id"].iloc[i]) for i in order)

    result = solution_module.search(query, 5)
    pca_top5 = set(int(item["original_id"]) for item in result)

    overlap = pca_top5 & gt_top5
    assert len(overlap) >= 3, (
        f"PCA-space top-5 must overlap 128-d ground-truth top-5 by at least 3; "
        f"overlap={overlap}, pca_top5={pca_top5}, gt_top5={gt_top5}."
    )


# ---------------------------------------------------------------------------
# Step 6: Verifier re-projects the query and runs a direct PCA-space search.
# The set of top-5 IDs must equal the candidate's search() top-5 set.
# ---------------------------------------------------------------------------


def test_verifier_direct_pca_search_matches_candidate(
    solution_module, pca_tbl, pca_model
):
    query = _make_query()
    components, mean = pca_model
    qp = (query.astype(np.float64) - mean) @ components.T

    direct = pca_tbl.search(qp.astype(np.float32)).limit(5).to_list()
    direct_ids = set(int(row["original_id"]) for row in direct)

    cand = solution_module.search(query, 5)
    cand_ids = set(int(item["original_id"]) for item in cand)

    assert direct_ids == cand_ids, (
        "Candidate's search() top-5 IDs must equal verifier's direct PCA-space top-5 IDs. "
        f"direct={direct_ids}, candidate={cand_ids}."
    )
