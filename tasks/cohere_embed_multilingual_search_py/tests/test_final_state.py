import importlib
import json
import os
import re
import subprocess
import sys
import shutil

import lancedb
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "").strip()
    assert run_id, "ZEALT_RUN_ID must be set in the verifier environment."
    return f"multilingual_{run_id}"


@pytest.fixture(scope="session", autouse=True)
def _ensure_index_built():
    """Run candidate's setup logic before any other test."""
    assert os.path.isfile(SOLUTION_PATH), (
        f"Expected candidate solution module at {SOLUTION_PATH}."
    )
    # Remove any stale LanceDB store from a previous attempt so the
    # candidate's build runs fresh.
    if os.path.isdir(LANCEDB_DIR):
        shutil.rmtree(LANCEDB_DIR)

    # Run build/index step in a subprocess so that any os._exit / SIGABRT
    # quirks from lancedb 0.25.3 don't kill the verifier itself.
    build_script = (
        "import sys, os\n"
        f"sys.path.insert(0, {PROJECT_DIR!r})\n"
        "import solution\n"
        "if hasattr(solution, 'build_index'):\n"
        "    solution.build_index()\n"
        "else:\n"
        "    # Lazy build: trigger via one search.\n"
        "    solution.cross_lingual_search('hola mundo', k=1)\n"
        "print('BUILD_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", build_script],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=600,
    )
    assert result.returncode == 0, (
        "Candidate build/index step failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "BUILD_OK" in result.stdout, (
        f"Build script did not finish cleanly. Output:\n{result.stdout}\n{result.stderr}"
    )


@pytest.fixture(scope="session")
def candidate_module():
    """Import the candidate's solution module once for query tests."""
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Make sure we get a fresh import (in case any previous test imported it).
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


def test_table_row_count_and_schema():
    """LanceDB table must contain exactly 90 rows with 1024-d vectors."""
    db = lancedb.connect(LANCEDB_DIR)
    name = _table_name()
    assert name in db.table_names(), (
        f"Expected table {name!r} to exist after build. Got {db.table_names()}."
    )
    tbl = db.open_table(name)
    assert tbl.count_rows() == 90, (
        f"Expected exactly 90 rows in {name!r}, got {tbl.count_rows()}."
    )

    schema = tbl.schema
    field_names = {f.name for f in schema}
    for required in ("concept_id", "language", "text"):
        assert required in field_names, (
            f"Table schema missing required column {required!r}; got {field_names}."
        )
    # The vector column may be called "vector" or "embedding".
    vector_field_name = None
    for candidate_name in ("vector", "embedding"):
        if candidate_name in field_names:
            vector_field_name = candidate_name
            break
    assert vector_field_name is not None, (
        f"Table must have a 'vector' or 'embedding' column; got {field_names}."
    )
    vec_field = schema.field(vector_field_name)
    vec_type = vec_field.type
    # Accept FixedSizeList<float, 1024> or List<float> with rows-of-1024.
    list_size = None
    if pa.types.is_fixed_size_list(vec_type):
        list_size = vec_type.list_size
    assert list_size == 1024 or list_size is None, (
        f"Vector column should be 1024-d fixed_size_list; got {vec_type}."
    )

    df = tbl.to_pandas()
    if list_size is None:
        actual_dim = len(df[vector_field_name].iloc[0])
        assert actual_dim == 1024, (
            f"Expected vector dim 1024, got {actual_dim}."
        )

    # 30 per language.
    lang_counts = df["language"].value_counts().to_dict()
    assert lang_counts.get("en", 0) == 30, f"Expected 30 English rows; got {lang_counts.get('en', 0)}."
    assert lang_counts.get("es", 0) == 30, f"Expected 30 Spanish rows; got {lang_counts.get('es', 0)}."
    assert lang_counts.get("fr", 0) == 30, f"Expected 30 French rows; got {lang_counts.get('fr', 0)}."

    # Each concept_id exactly 3 times across 0..29.
    concept_counts = df["concept_id"].value_counts().to_dict()
    assert len(concept_counts) == 30, (
        f"Expected exactly 30 distinct concept_ids; got {len(concept_counts)}."
    )
    for cid, c in concept_counts.items():
        assert int(cid) in range(30), f"Unexpected concept_id {cid}; must be 0..29."
        assert c == 3, f"concept_id {cid} appears {c} times; expected 3."


def _validate_result_payload(results, anchor_concept_id, k=3):
    assert isinstance(results, list), f"cross_lingual_search must return a list, got {type(results)!r}."
    assert len(results) == k, f"Expected {k} results, got {len(results)}: {results}"
    for item in results:
        assert isinstance(item, dict), f"Each result must be a dict, got {type(item)!r}."
        for key in ("concept_id", "language", "text"):
            assert key in item, f"Result dict missing key {key!r}: {item}"
        assert item["language"] in {"en", "es", "fr"}, (
            f"Unexpected language in result: {item}"
        )
    langs = {item["language"] for item in results}
    assert len(langs) >= 2, (
        f"Expected results to span >=2 languages, got {langs}: {results}"
    )
    concept_ids = {int(item["concept_id"]) for item in results}
    assert anchor_concept_id in concept_ids, (
        f"Expected anchor concept_id={anchor_concept_id} in top-{k} results, got {concept_ids}: {results}"
    )


def test_english_anchor_query_eiffel(candidate_module):
    """English query about the Eiffel Tower must surface concept_id=0 across >=2 languages."""
    results = candidate_module.cross_lingual_search(
        "Where is the Eiffel Tower located in France?", k=3
    )
    _validate_result_payload(results, anchor_concept_id=0, k=3)


def test_spanish_anchor_query_everest(candidate_module):
    """Spanish query about the highest mountain must surface concept_id=1 across >=2 languages."""
    results = candidate_module.cross_lingual_search(
        "¿Cuál es la montaña más alta del mundo?", k=3
    )
    _validate_result_payload(results, anchor_concept_id=1, k=3)


def test_no_hardcoded_cohere_key():
    """The candidate must read COHERE_API_KEY from the environment, not hardcode it."""
    leak_pattern = re.compile(
        r"""(?ix)
        cohere[^\n]{0,40}=\s*['"][A-Za-z0-9_\-]{20,}['"]
        |
        (?<![A-Za-z0-9_])(co_)?[A-Za-z0-9]{40}(?![A-Za-z0-9_])
        """,
        re.VERBOSE,
    )
    for root, _dirs, files in os.walk(PROJECT_DIR):
        # Skip the lancedb store directory.
        if os.path.commonpath([root, LANCEDB_DIR]) == LANCEDB_DIR:
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Ensure env-var access pattern is used somewhere in the project
            # — but at minimum solution.py is grep'd for hardcoded strings.
            assert "COHERE_API_KEY" in content or "cohere" not in content.lower() or fname != "solution.py", (
                f"solution.py should reference COHERE_API_KEY env var."
            )
            for match in leak_pattern.finditer(content):
                # If the match coincides with a clear placeholder name, allow it.
                snippet = match.group(0)
                if "COHERE_API_KEY" in snippet or "os.environ" in snippet:
                    continue
                raise AssertionError(
                    f"Possible hardcoded Cohere credential in {fpath}: {snippet!r}"
                )
