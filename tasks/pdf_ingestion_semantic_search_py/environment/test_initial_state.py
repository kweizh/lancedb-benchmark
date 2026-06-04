import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
CORPUS_DIR = "/app/corpus"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_corpus_dir_exists():
    assert os.path.isdir(CORPUS_DIR), f"Corpus directory {CORPUS_DIR} does not exist."


def test_corpus_has_three_pdfs():
    pdfs = sorted(f for f in os.listdir(CORPUS_DIR) if f.lower().endswith(".pdf"))
    assert pdfs == ["alpha.pdf", "bravo.pdf", "charlie.pdf"], (
        f"Expected corpus to contain alpha.pdf, bravo.pdf, charlie.pdf; got {pdfs}."
    )


@pytest.mark.parametrize("name", ["alpha.pdf", "bravo.pdf", "charlie.pdf"])
def test_corpus_pdf_nonempty(name):
    path = os.path.join(CORPUS_DIR, name)
    assert os.path.isfile(path), f"Expected corpus PDF {path} to exist."
    assert os.path.getsize(path) > 200, f"Corpus PDF {path} is suspiciously small."


def test_lancedb_importable():
    importlib.import_module("lancedb")


def test_openai_importable():
    importlib.import_module("openai")


def test_pypdf_importable():
    importlib.import_module("pypdf")


def test_openai_api_key_set():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in the task environment."


def test_zealt_run_id_set():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID must be set in the task environment."
