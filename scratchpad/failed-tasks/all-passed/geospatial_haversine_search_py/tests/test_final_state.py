import json
import math
import os
import subprocess

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
SEARCH_SCRIPT = os.path.join(PROJECT_DIR, "search.py")
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
QUERY_VEC_A = os.path.join(PROJECT_DIR, "query_vec_A.json")
QUERY_VEC_B = os.path.join(PROJECT_DIR, "query_vec_B.json")
OUT_A = os.path.join(PROJECT_DIR, "out_A.json")
OUT_B = os.path.join(PROJECT_DIR, "out_B.json")

# These constants MUST mirror environment/seed_pois.py exactly.
N_ROWS = 240
SEED = 2026
CATEGORIES = ["restaurant", "cafe", "museum", "park", "shop"]
EARTH_RADIUS_KM = 6371.0


def _generate_reference_dataset():
    """Reconstruct the exact pre-seeded POI corpus deterministically."""
    rng = np.random.default_rng(SEED)
    lats = rng.uniform(37.0, 38.0, N_ROWS)
    lons = rng.uniform(-123.0, -122.0, N_ROWS)
    cats = rng.choice(CATEGORIES, N_ROWS)
    embeddings = rng.standard_normal((N_ROWS, 32)).astype(np.float32)
    ids = np.arange(N_ROWS, dtype=np.int32)
    names = np.array([f"POI-{i:04d}" for i in ids])
    return ids, names, lats, lons, cats, embeddings


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km (Earth radius = 6371.0)."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def _reference_topk(center_lat, center_lon, radius_km, category, query_vector, top_k):
    ids, names, lats, lons, cats, embeddings = _generate_reference_dataset()
    qv = np.asarray(query_vector, dtype=np.float32)
    dist_km = _haversine_km(center_lat, center_lon, lats, lons)
    mask_radius = dist_km <= radius_km
    mask_cat = cats == category
    mask = mask_radius & mask_cat
    sel_ids = ids[mask]
    sel_emb = embeddings[mask]
    sel_dist_km = dist_km[mask]
    # L2 vector distance
    diffs = sel_emb - qv
    vec_dist = np.sqrt(np.sum(diffs * diffs, axis=1))
    # Sort by ascending vec_dist; tie-break by id for determinism
    order = np.lexsort((sel_ids, vec_dist))
    order = order[:top_k]
    return [int(sel_ids[i]) for i in order]


def _run_search(args):
    return subprocess.run(
        ["python3", SEARCH_SCRIPT, *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=300,
    )


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _check_entry_schema(entry):
    expected_keys = {"id", "name", "category", "lat", "lon", "distance_km", "vector_distance"}
    assert expected_keys.issubset(entry.keys()), (
        f"Entry missing keys. Expected {expected_keys}, got {set(entry.keys())}"
    )
    assert isinstance(entry["id"], int), f"id must be int, got {type(entry['id'])}"
    assert isinstance(entry["name"], str), f"name must be str, got {type(entry['name'])}"
    assert isinstance(entry["category"], str), f"category must be str, got {type(entry['category'])}"
    assert isinstance(entry["lat"], (int, float)), f"lat must be number"
    assert isinstance(entry["lon"], (int, float)), f"lon must be number"
    assert isinstance(entry["distance_km"], (int, float)), f"distance_km must be number"
    assert isinstance(entry["vector_distance"], (int, float)), f"vector_distance must be number"


@pytest.fixture(scope="session", autouse=True)
def cleanup_outputs():
    for p in (OUT_A, OUT_B):
        if os.path.isfile(p):
            os.remove(p)
    yield


def test_search_script_exists():
    assert os.path.isfile(SEARCH_SCRIPT), f"Expected candidate script at {SEARCH_SCRIPT}."


def test_query_a_restaurant_top5():
    qv = _read_json(QUERY_VEC_A)
    result = _run_search([
        "--center-lat", "37.50",
        "--center-lon", "-122.50",
        "--radius-km", "25.0",
        "--category", "restaurant",
        "--query-vector-path", QUERY_VEC_A,
        "--top-k", "5",
        "--output", OUT_A,
    ])
    assert result.returncode == 0, (
        f"Query A search.py exited with {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert os.path.isfile(OUT_A), f"Output file {OUT_A} was not created."
    payload = _read_json(OUT_A)
    assert "results" in payload, f"Output JSON missing 'results' key. Got: {payload}"
    actual_ids = [e["id"] for e in payload["results"]]
    expected_ids = _reference_topk(37.50, -122.50, 25.0, "restaurant", qv, 5)
    assert actual_ids == expected_ids, (
        f"Query A ranked ids mismatch.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )
    for entry in payload["results"]:
        _check_entry_schema(entry)
        assert entry["category"] == "restaurant", (
            f"Query A entry id={entry['id']} has category {entry['category']}; expected 'restaurant'."
        )
        assert entry["distance_km"] <= 25.0 + 1e-6, (
            f"Query A entry id={entry['id']} distance_km={entry['distance_km']} exceeds radius 25.0."
        )
    vds = [e["vector_distance"] for e in payload["results"]]
    assert vds == sorted(vds), f"Query A results not sorted by vector_distance ascending: {vds}"


def test_query_b_cafe_top7():
    qv = _read_json(QUERY_VEC_B)
    result = _run_search([
        "--center-lat", "37.30",
        "--center-lon", "-122.10",
        "--radius-km", "25.0",
        "--category", "cafe",
        "--query-vector-path", QUERY_VEC_B,
        "--top-k", "7",
        "--output", OUT_B,
    ])
    assert result.returncode == 0, (
        f"Query B search.py exited with {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert os.path.isfile(OUT_B), f"Output file {OUT_B} was not created."
    payload = _read_json(OUT_B)
    assert "results" in payload, f"Output JSON missing 'results' key. Got: {payload}"
    actual_ids = [e["id"] for e in payload["results"]]
    expected_ids = _reference_topk(37.30, -122.10, 25.0, "cafe", qv, 7)
    assert actual_ids == expected_ids, (
        f"Query B ranked ids mismatch.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )
    for entry in payload["results"]:
        _check_entry_schema(entry)
        assert entry["category"] == "cafe", (
            f"Query B entry id={entry['id']} has category {entry['category']}; expected 'cafe'."
        )
        assert entry["distance_km"] <= 25.0 + 1e-6, (
            f"Query B entry id={entry['id']} distance_km={entry['distance_km']} exceeds radius 25.0."
        )
    vds = [e["vector_distance"] for e in payload["results"]]
    assert vds == sorted(vds), f"Query B results not sorted by vector_distance ascending: {vds}"
