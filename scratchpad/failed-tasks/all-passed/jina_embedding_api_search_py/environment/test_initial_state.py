import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
HEADLINES_PATH = os.path.join(PROJECT_DIR, "headlines.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_httpx_importable():
    import httpx  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_headlines_fixture_exists():
    assert os.path.isfile(HEADLINES_PATH), (
        f"Headline fixture {HEADLINES_PATH} does not exist."
    )


def test_headlines_fixture_shape():
    with open(HEADLINES_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list), "headlines.json must be a JSON array."
    assert len(data) == 50, f"Expected 50 headlines, got {len(data)}."

    topics = {}
    ids = set()
    for entry in data:
        assert isinstance(entry, dict), "Each headline entry must be a dict."
        for key in ("id", "topic", "headline"):
            assert key in entry, f"Headline entry missing key '{key}': {entry}"
        assert isinstance(entry["id"], int), "id must be an integer."
        assert isinstance(entry["topic"], str) and entry["topic"], "topic must be non-empty str."
        assert isinstance(entry["headline"], str) and entry["headline"], "headline must be non-empty str."
        topics.setdefault(entry["topic"], 0)
        topics[entry["topic"]] += 1
        ids.add(entry["id"])

    assert len(ids) == 50, "Headline ids must be unique."
    assert len(topics) == 5, f"Expected 5 distinct topics, got {len(topics)}: {sorted(topics)}"
    for topic, count in topics.items():
        assert count == 10, f"Topic '{topic}' has {count} headlines (expected 10)."


def test_anchor_headlines_present():
    """The three rigged anchor headlines that the verifier targets must be in the corpus."""
    with open(HEADLINES_PATH) as f:
        data = json.load(f)
    headlines = {entry["id"]: entry for entry in data}

    expected = {
        1: ("finance", "Federal Reserve Raises Benchmark Interest Rate by 25 Basis Points"),
        11: ("sports", "American Swimmer Wins Olympic Gold Medal in 200m Freestyle"),
        21: ("space", "NASA Spacecraft Successfully Lands on Surface of Mars"),
    }
    for hid, (topic, headline) in expected.items():
        assert hid in headlines, f"Anchor headline id {hid} missing from corpus."
        assert headlines[hid]["topic"] == topic, (
            f"Anchor headline id {hid} should have topic '{topic}', got '{headlines[hid]['topic']}'."
        )
        assert headlines[hid]["headline"] == headline, (
            f"Anchor headline id {hid} text mismatch: {headlines[hid]['headline']!r}"
        )


def test_jina_api_key_env():
    assert os.environ.get("JINA_API_KEY"), "JINA_API_KEY env var must be set in the container."


def test_zealt_run_id_env():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID env var must be set in the container."
