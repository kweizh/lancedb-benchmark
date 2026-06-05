import importlib
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

PROJECT_DIR = "/home/user/myproject"
DB_PATH = os.path.join(PROJECT_DIR, "data", "db")
METRICS_URL = "http://127.0.0.1:9100/metrics"
EXPECTED_BUCKETS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]


@pytest.fixture(scope="session")
def run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set in the verification environment."
    return rid


@pytest.fixture(scope="session")
def table_name(run_id):
    return f"documents_{run_id}"


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    sys.modules.pop("solution", None)
    mod = importlib.import_module("solution")
    return mod


@pytest.fixture(scope="session")
def lancedb_table(table_name):
    import lancedb

    db = lancedb.connect(DB_PATH)
    assert table_name in db.table_names(), (
        f"Seeded table {table_name} not found in {DB_PATH}; got {db.table_names()}."
    )
    return db.open_table(table_name)


@pytest.fixture(scope="session")
def search_and_workload(solution_module, lancedb_table, table_name):
    # Public surface checks
    assert hasattr(solution_module, "Search"), "solution.py must expose a `Search` class."
    assert hasattr(solution_module, "start_metrics_server"), (
        "solution.py must expose a `start_metrics_server(port=9100)` function."
    )

    search = solution_module.Search(lancedb_table, table_name)
    for meth in ("vector_search", "fts_search", "hybrid_search"):
        assert hasattr(search, meth) and callable(getattr(search, meth)), (
            f"Search must expose a callable method `{meth}`."
        )

    # Start metrics exposition.
    solution_module.start_metrics_server(9100)

    # Wait until /metrics is reachable.
    deadline = time.time() + 10
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(METRICS_URL, timeout=2)
            if r.status_code == 200:
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.25)
    else:
        raise AssertionError(
            f"Could not reach Prometheus /metrics on {METRICS_URL} within 10s: {last_err}"
        )

    rng = np.random.default_rng(2026)

    counts = {"vector": 0, "fts": 0, "hybrid": 0}
    for _ in range(8):
        qvec = rng.standard_normal(32).astype("float32")
        out = search.vector_search(qvec, 5)
        assert out is not None, "vector_search returned None."
        counts["vector"] += 1
    for _ in range(6):
        out = search.fts_search("lance", 5)
        assert out is not None, "fts_search returned None."
        counts["fts"] += 1
    for _ in range(6):
        qvec = rng.standard_normal(32).astype("float32")
        out = search.hybrid_search(qvec, "lance", 5)
        assert out is not None, "hybrid_search returned None."
        counts["hybrid"] += 1

    assert sum(counts.values()) == 20, f"Expected to run 20 queries, ran {counts}."
    return counts


def _scrape_metrics():
    r = requests.get(METRICS_URL, timeout=5)
    assert r.status_code == 200, f"GET {METRICS_URL} returned {r.status_code}."
    body = r.text
    assert body.strip(), "Prometheus /metrics body is empty."
    return body


def _families_by_name(body):
    families = {}
    for fam in text_string_to_metric_families(body):
        families[fam.name] = fam
    return families


def _find_counter_family(families, exposed_name):
    """Return the counter family whose samples expose `exposed_name`.

    `text_string_to_metric_families` strips the `_total` suffix from counter
    family names (so `lancedb_query_total` samples appear under family name
    `lancedb_query`). Accept either spelling.
    """
    if exposed_name in families:
        return families[exposed_name]
    base = exposed_name[: -len("_total")] if exposed_name.endswith("_total") else exposed_name
    return families.get(base)


def test_counter_total_and_labels(search_and_workload, table_name):
    counts = search_and_workload
    body = _scrape_metrics()
    # Sanity: the raw exposition must mention the `lancedb_query_total` sample.
    assert "lancedb_query_total{" in body or "lancedb_query_total " in body, (
        "Raw /metrics body must expose Counter samples named `lancedb_query_total`."
    )
    families = _families_by_name(body)
    fam = _find_counter_family(families, "lancedb_query_total")
    assert fam is not None, (
        f"Counter family for `lancedb_query_total` not exposed at /metrics; saw {list(families)}."
    )
    assert fam.type == "counter", (
        f"`lancedb_query_total` must be a counter, got type={fam.type}."
    )

    total = 0.0
    per_qtype = defaultdict(float)
    for sample in fam.samples:
        if not sample.name.endswith("_total"):
            continue
        labels = sample.labels
        assert set(labels.keys()) == {"query_type", "table"}, (
            f"Counter sample labels must be exactly {{query_type, table}}, got {labels}."
        )
        assert labels["table"] == table_name, (
            f"Counter sample table label must be {table_name}, got {labels['table']}."
        )
        per_qtype[labels["query_type"]] += sample.value
        total += sample.value

    assert total == 20.0, f"Expected counter total == 20, got {total}."
    for qt, expected in counts.items():
        assert per_qtype.get(qt, 0.0) == float(expected), (
            f"Counter for query_type={qt!r} must be {expected}, got {per_qtype.get(qt, 0.0)}."
        )


def test_histogram_buckets_and_counts(search_and_workload, table_name):
    counts = search_and_workload
    body = _scrape_metrics()
    families = _families_by_name(body)
    assert "lancedb_query_duration_seconds" in families, (
        "Histogram family `lancedb_query_duration_seconds` not exposed at /metrics."
    )
    fam = families["lancedb_query_duration_seconds"]
    assert fam.type == "histogram", (
        f"`lancedb_query_duration_seconds` must be a histogram, got type={fam.type}."
    )

    # Group samples by (query_type, table) label key
    by_group = defaultdict(list)
    for sample in fam.samples:
        labels = dict(sample.labels)
        # all samples must have query_type and table labels
        assert "query_type" in labels and "table" in labels, (
            f"Histogram sample missing required labels query_type/table: {labels}"
        )
        assert labels["table"] == table_name, (
            f"Histogram table label must be {table_name}, got {labels['table']}."
        )
        key = (labels["query_type"], labels["table"])
        by_group[key].append(sample)

    expected_keys = {("vector", table_name), ("fts", table_name), ("hybrid", table_name)}
    assert expected_keys.issubset(set(by_group.keys())), (
        f"Histogram is missing label groups; expected {expected_keys}, got {set(by_group.keys())}."
    )

    for key, samples in by_group.items():
        bucket_les = []
        count_val = None
        inf_bucket_val = None
        for s in samples:
            if s.name.endswith("_bucket"):
                le = s.labels.get("le")
                if le == "+Inf":
                    inf_bucket_val = s.value
                else:
                    bucket_les.append(float(le))
            elif s.name.endswith("_count"):
                count_val = s.value
        for edge in EXPECTED_BUCKETS:
            assert any(abs(b - edge) < 1e-9 for b in bucket_les), (
                f"Expected bucket edge {edge} missing from histogram for {key}; got {bucket_les}."
            )
        assert count_val is not None, f"Histogram _count missing for {key}."
        assert inf_bucket_val is not None, f"Histogram +Inf bucket missing for {key}."
        assert inf_bucket_val == count_val, (
            f"Histogram +Inf bucket ({inf_bucket_val}) must equal _count ({count_val}) for {key}."
        )
        qtype = key[0]
        assert count_val == float(counts[qtype]), (
            f"Histogram _count for {key} must equal call count {counts[qtype]}, got {count_val}."
        )


def test_gauge_table_rows(search_and_workload, table_name):
    body = _scrape_metrics()
    families = _families_by_name(body)
    assert "lancedb_table_rows" in families, (
        "Gauge family `lancedb_table_rows` not exposed at /metrics."
    )
    fam = families["lancedb_table_rows"]
    assert fam.type == "gauge", (
        f"`lancedb_table_rows` must be a gauge, got type={fam.type}."
    )
    matched = None
    for sample in fam.samples:
        if sample.labels.get("table") == table_name:
            matched = sample.value
            break
    assert matched is not None, (
        f"No `lancedb_table_rows` sample with table={table_name} found."
    )
    assert matched == 200.0, (
        f"Gauge `lancedb_table_rows{{table={table_name}}}` must equal 200, got {matched}."
    )


def test_metrics_stable_on_second_scrape(search_and_workload, table_name):
    counts = search_and_workload
    # Scrape twice, ~0.5s apart, ensure counter total and gauge are unchanged.
    body1 = _scrape_metrics()
    time.sleep(0.5)
    body2 = _scrape_metrics()

    def counter_total(body):
        fam = _find_counter_family(_families_by_name(body), "lancedb_query_total")
        assert fam is not None, "Counter family `lancedb_query_total` missing on re-scrape."
        return sum(s.value for s in fam.samples if s.name.endswith("_total"))

    def gauge_val(body):
        fam = _families_by_name(body)["lancedb_table_rows"]
        for s in fam.samples:
            if s.labels.get("table") == table_name:
                return s.value
        return None

    assert counter_total(body1) == 20.0
    assert counter_total(body2) == 20.0
    assert gauge_val(body1) == 200.0
    assert gauge_val(body2) == 200.0
