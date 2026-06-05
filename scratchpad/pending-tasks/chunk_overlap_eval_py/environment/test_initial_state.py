import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DOCS_DIR = "/app/docs"
ANCHORS_PATH = "/app/anchors.json"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_openai_importable():
    import openai  # noqa: F401


def test_langchain_text_splitters_importable():
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_docs_dir_exists():
    assert os.path.isdir(DOCS_DIR), f"Docs directory {DOCS_DIR} does not exist."


def test_three_source_docs_present():
    files = sorted(os.listdir(DOCS_DIR))
    txt_files = [f for f in files if f.endswith(".txt")]
    assert len(txt_files) == 3, f"Expected exactly 3 .txt files in {DOCS_DIR}, found {txt_files}."


@pytest.mark.parametrize("name", ["doc_1.txt", "doc_2.txt", "doc_3.txt"])
def test_source_doc_size(name):
    path = os.path.join(DOCS_DIR, name)
    assert os.path.isfile(path), f"Missing source doc {path}."
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n = len(text)
    assert 3000 <= n <= 5000, f"{name} length {n} not in [3000, 5000]."


def test_anchors_json_present_and_schema():
    assert os.path.isfile(ANCHORS_PATH), f"Anchors fixture {ANCHORS_PATH} missing."
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchors = json.load(f)
    assert isinstance(anchors, list), "anchors.json must be a JSON list."
    assert len(anchors) == 10, f"Expected 10 anchor queries, got {len(anchors)}."
    for i, a in enumerate(anchors):
        for k in ("query", "doc_id", "start", "end"):
            assert k in a, f"Anchor #{i} missing field {k!r}."
        assert isinstance(a["query"], str) and a["query"], f"Anchor #{i} has empty query."
        assert a["doc_id"] in {"doc_1", "doc_2", "doc_3"}, (
            f"Anchor #{i} doc_id {a['doc_id']!r} not in {{doc_1, doc_2, doc_3}}."
        )
        assert isinstance(a["start"], int) and isinstance(a["end"], int), (
            f"Anchor #{i} start/end must be ints."
        )
        assert 0 <= a["start"] < a["end"], f"Anchor #{i} has invalid span."


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set at runtime."


def test_zealt_run_id_present():
    rid = os.environ.get("ZEALT_RUN_ID", "")
    assert rid, "ZEALT_RUN_ID must be set at runtime."
