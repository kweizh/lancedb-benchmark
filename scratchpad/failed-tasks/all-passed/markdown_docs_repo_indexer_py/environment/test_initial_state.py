import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DOCS_DIR = "/app/docs"

EXPECTED_DOCS = [
    "auth-guide.md",
    "api-reference.md",
    "deployment.md",
    "performance.md",
    "monitoring.md",
    "migrations.md",
]


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_docs_dir_exists():
    assert os.path.isdir(DOCS_DIR), f"Docs directory {DOCS_DIR} does not exist."


def test_docs_dir_has_six_markdown_files():
    files = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".md"))
    assert set(files) == set(EXPECTED_DOCS), (
        f"Expected docs dir to contain exactly {sorted(EXPECTED_DOCS)}, got {files}."
    )


@pytest.mark.parametrize("name", EXPECTED_DOCS)
def test_each_doc_has_one_title_and_multiple_sections(name):
    path = os.path.join(DOCS_DIR, name)
    assert os.path.isfile(path), f"Expected markdown file {path} to exist."
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    title_lines = [ln for ln in text.splitlines() if ln.startswith("# ") and not ln.startswith("## ")]
    section_lines = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert len(title_lines) == 1, (
        f"Expected exactly one '# Title' line in {path}, got {len(title_lines)}."
    )
    assert len(section_lines) >= 3, (
        f"Expected at least 3 '## Section' headers in {path}, got {len(section_lines)}."
    )


def test_lancedb_importable():
    importlib.import_module("lancedb")


def test_openai_importable():
    importlib.import_module("openai")


def test_markdown_it_py_importable():
    importlib.import_module("markdown_it")


def test_openai_api_key_set():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in the task environment."


def test_zealt_run_id_set():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID must be set in the task environment."
