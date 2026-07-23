import hashlib
import json
import math
import os
import re
import subprocess
import time

import pytest

PROJECT_DIR = "/home/user/hybrid_bridge"
LANCEDB_DIR = "/home/user/hybrid_bridge/data/lancedb"
RUN_SCRIPT = os.path.join(PROJECT_DIR, "run.py")

CLICKHOUSE_HOST = "127.0.0.1"
CLICKHOUSE_PORT = 8123

EMBED_DIM = 32
TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---------------------------------------------------------------------------
# Shared helpers: the verifier recomputes the expected result independently and
# directly from the live LanceDB table and ClickHouse server, following the
# documented pipeline. It does NOT hardcode any expected numbers.
# ---------------------------------------------------------------------------


def _embed(text):
    vec = [0.0] * EMBED_DIM
    for token in TOKEN_RE.findall(text.lower()):
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _clickhouse_client():
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username="default",
        password="",
        database="default",
    )


def _open_documents():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    return db.open_table("documents")


def _fts_candidate_ids(table, text):
    n = table.count_rows()
    rows = (
        table.search(text, query_type="fts")
        .select(["id"])
        .limit(max(n, 1))
        .to_list()
    )
    return {int(r["id"]) for r in rows}


def _vector_distances(table, query_vector):
    n = table.count_rows()
    rows = (
        table.search(query_vector)
        .select(["id"])
        .limit(max(n, 1))
        .to_list()
    )
    return {int(r["id"]): float(r["_distance"]) for r in rows}


def _clickhouse_metrics(client, doc_ids, window_start, window_end):
    metrics = {}
    if not doc_ids:
        return metrics
    id_list = ",".join(str(int(d)) for d in sorted(doc_ids))
    window = (
        f"ts >= toDateTime('{window_start}') AND ts < toDateTime('{window_end}')"
    )

    base = client.query(
        f"SELECT doc_id, count() AS c, quantileExact(0.95)(value) AS p95 "
        f"FROM events WHERE doc_id IN ({id_list}) AND {window} "
        f"GROUP BY doc_id"
    ).result_rows
    for doc_id, c, p95 in base:
        metrics.setdefault(int(doc_id), {})
        metrics[int(doc_id)]["events_in_window"] = int(c)
        metrics[int(doc_id)]["p95_value"] = float(p95)

    peak = client.query(
        f"SELECT doc_id, max(cnt) FROM ("
        f"SELECT doc_id, toStartOfHour(ts) AS h, count() AS cnt "
        f"FROM events WHERE doc_id IN ({id_list}) AND {window} "
        f"GROUP BY doc_id, h) GROUP BY doc_id"
    ).result_rows
    for doc_id, mx in peak:
        metrics.setdefault(int(doc_id), {})
        metrics[int(doc_id)]["peak_hour_count"] = int(mx)

    prem = client.query(
        f"SELECT e.doc_id, sum(e.value) FROM events e "
        f"INNER JOIN users u ON e.user_id = u.user_id "
        f"WHERE e.doc_id IN ({id_list}) AND {window} AND u.tier = 'premium' "
        f"GROUP BY e.doc_id"
    ).result_rows
    for doc_id, s in prem:
        metrics.setdefault(int(doc_id), {})
        metrics[int(doc_id)]["premium_value_sum"] = float(s)

    return metrics


def _expected_ranked(text, window_start, window_end):
    table = _open_documents()
    client = _clickhouse_client()

    candidates = _fts_candidate_ids(table, text)
    query_vector = _embed(text)
    distances = _vector_distances(table, query_vector)
    metrics = _clickhouse_metrics(client, candidates, window_start, window_end)

    rows = []
    for doc_id in candidates:
        m = metrics.get(doc_id, {})
        events_in_window = int(m.get("events_in_window", 0))
        premium_value_sum = float(m.get("premium_value_sum", 0.0))
        p95_value = float(m.get("p95_value", 0.0))
        peak_hour_count = int(m.get("peak_hour_count", 0))
        vector_distance = float(distances.get(doc_id, 0.0))
        score = round(premium_value_sum / (1.0 + vector_distance), 6)
        rows.append(
            {
                "doc_id": doc_id,
                "events_in_window": events_in_window,
                "premium_value_sum": premium_value_sum,
                "p95_value": p95_value,
                "peak_hour_count": peak_hour_count,
                "vector_distance": vector_distance,
                "score": score,
            }
        )

    rows.sort(key=lambda r: (-r["score"], r["doc_id"]))
    return rows


def _run_cli(query, query_name):
    query_path = os.path.join(PROJECT_DIR, f"{query_name}.json")
    output_path = os.path.join(PROJECT_DIR, f"{query_name}_result.json")
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(query_path, "w") as f:
        json.dump(query, f)

    result = subprocess.run(
        ["python3", RUN_SCRIPT, "--query-file", query_path, "--output", output_path],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    assert result.returncode == 0, (
        f"run.py exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert os.path.isfile(output_path), (
        f"Expected output file {output_path} was not created."
    )
    with open(output_path) as f:
        data = json.load(f)
    return data


REQUIRED_KEYS = {
    "doc_id",
    "events_in_window",
    "premium_value_sum",
    "p95_value",
    "peak_hour_count",
    "vector_distance",
    "score",
}


def _assert_matches(actual, expected_ranked, top):
    assert isinstance(actual, list), f"Output must be a JSON array, got {type(actual)}."
    expected = expected_ranked[:top]
    assert len(actual) == len(expected), (
        f"Expected {len(expected)} result rows (top={top}), got {len(actual)}."
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert isinstance(a, dict), f"Element {i} must be an object."
        assert set(a.keys()) == REQUIRED_KEYS, (
            f"Element {i} must have exactly keys {sorted(REQUIRED_KEYS)}, "
            f"got {sorted(a.keys())}."
        )
        assert int(a["doc_id"]) == e["doc_id"], (
            f"Ordering/selection mismatch at position {i}: "
            f"expected doc_id {e['doc_id']}, got {a['doc_id']}."
        )
        assert int(a["events_in_window"]) == e["events_in_window"], (
            f"events_in_window mismatch for doc_id {e['doc_id']}: "
            f"expected {e['events_in_window']}, got {a['events_in_window']}."
        )
        assert int(a["peak_hour_count"]) == e["peak_hour_count"], (
            f"peak_hour_count mismatch for doc_id {e['doc_id']}: "
            f"expected {e['peak_hour_count']}, got {a['peak_hour_count']}."
        )
        assert abs(float(a["premium_value_sum"]) - e["premium_value_sum"]) <= 1e-4, (
            f"premium_value_sum mismatch for doc_id {e['doc_id']}: "
            f"expected {e['premium_value_sum']}, got {a['premium_value_sum']}."
        )
        assert abs(float(a["p95_value"]) - e["p95_value"]) <= 1e-4, (
            f"p95_value mismatch for doc_id {e['doc_id']}: "
            f"expected {e['p95_value']}, got {a['p95_value']}."
        )
        assert abs(float(a["vector_distance"]) - e["vector_distance"]) <= 1e-4, (
            f"vector_distance mismatch for doc_id {e['doc_id']}: "
            f"expected {e['vector_distance']}, got {a['vector_distance']}."
        )
        assert abs(float(a["score"]) - e["score"]) <= 1e-6, (
            f"score mismatch for doc_id {e['doc_id']}: "
            f"expected {e['score']}, got {a['score']}."
        )


def _clickhouse_reachable():
    try:
        client = _clickhouse_client()
        return int(client.query("SELECT 1").result_rows[0][0]) == 1
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session", autouse=True)
def _ensure_clickhouse():
    if _clickhouse_reachable():
        return
    # Fallback: start the pre-seeded local ClickHouse server if it is not running.
    try:
        subprocess.Popen(
            [
                "runuser",
                "-u",
                "clickhouse",
                "--",
                "clickhouse-server",
                "--config-file=/etc/clickhouse-server/config.xml",
                "--daemon",
                "--pid-file=/run/clickhouse-server/clickhouse-server.pid",
            ]
        )
    except Exception:  # noqa: BLE001
        pass
    for _ in range(60):
        if _clickhouse_reachable():
            return
        time.sleep(1)
    pytest.fail(
        f"ClickHouse server not reachable on {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}."
    )


def test_run_script_exists():
    assert os.path.isfile(RUN_SCRIPT), f"Expected the CLI script at {RUN_SCRIPT}."


def test_typical_hybrid_query():
    query = {
        "text": "wireless bluetooth audio",
        "window_start": "2024-01-01 00:00:00",
        "window_end": "2024-01-31 00:00:00",
        "top": 5,
    }
    expected = _expected_ranked(query["text"], query["window_start"], query["window_end"])
    assert len(expected) >= 1, "Test data issue: FTS recall set is empty for the query."
    actual = _run_cli(query, "query1")
    _assert_matches(actual, expected, query["top"])


def test_zero_backfill_window():
    # A far-future window contains no seeded events, so every recalled document
    # must be back-filled with zeros and appear in the output ordered by doc_id.
    query = {
        "text": "wireless bluetooth audio",
        "window_start": "2025-06-01 00:00:00",
        "window_end": "2025-06-01 01:00:00",
        "top": 8,
    }
    expected = _expected_ranked(query["text"], query["window_start"], query["window_end"])
    actual = _run_cli(query, "query2")
    _assert_matches(actual, expected, query["top"])

    for row in actual:
        assert int(row["events_in_window"]) == 0, (
            "Expected zero events for an empty window, "
            f"got {row['events_in_window']} for doc_id {row['doc_id']}."
        )
        assert float(row["premium_value_sum"]) == 0.0
        assert float(row["p95_value"]) == 0.0
        assert int(row["peak_hour_count"]) == 0
        assert abs(float(row["score"])) <= 1e-9, (
            f"Score must be 0.0 when there is no premium value, got {row['score']}."
        )


def test_top_truncation_and_ordering():
    query = {
        "text": "audio device sound",
        "window_start": "2024-01-01 00:00:00",
        "window_end": "2024-01-31 00:00:00",
        "top": 3,
    }
    expected = _expected_ranked(query["text"], query["window_start"], query["window_end"])
    assert len(expected) >= 3, "Test data issue: fewer than 3 documents recalled."
    actual = _run_cli(query, "query3")
    assert len(actual) == 3, f"Expected exactly 3 results for top=3, got {len(actual)}."
    _assert_matches(actual, expected, query["top"])


def test_premium_join_correctness():
    query = {
        "text": "wireless bluetooth audio",
        "window_start": "2024-01-01 00:00:00",
        "window_end": "2024-01-31 00:00:00",
        "top": 5,
    }
    actual = _run_cli(query, "query4")
    client = _clickhouse_client()
    for row in actual:
        doc_id = int(row["doc_id"])
        all_tier_sum = client.query(
            f"SELECT sum(value) FROM events WHERE doc_id = {doc_id} "
            f"AND ts >= toDateTime('{query['window_start']}') "
            f"AND ts < toDateTime('{query['window_end']}')"
        ).result_rows[0][0]
        all_tier_sum = float(all_tier_sum) if all_tier_sum is not None else 0.0
        premium_sum = float(row["premium_value_sum"])
        # premium-only sum must never exceed the all-tier sum for the same window.
        assert premium_sum <= all_tier_sum + 1e-4, (
            f"premium_value_sum ({premium_sum}) exceeds the all-tier sum "
            f"({all_tier_sum}) for doc_id {doc_id}; the users JOIN / tier filter is wrong."
        )
