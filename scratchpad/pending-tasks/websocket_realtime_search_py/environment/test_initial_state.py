import importlib
import os

import pytest


PROJECT_DIR = "/home/user/myproject"
EMBED_PATH = os.path.join(PROJECT_DIR, "embed.py")


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb must be importable."


def test_websockets_importable():
    mod = importlib.import_module("websockets")
    assert mod is not None, "websockets must be importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow must be importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} must exist."


def test_embed_helper_exists():
    assert os.path.isfile(EMBED_PATH), (
        f"Expected the embed helper at {EMBED_PATH} (provides embed_text)."
    )


def test_embed_text_is_callable():
    import importlib.util
    import numpy as np

    spec = importlib.util.spec_from_file_location("embed_helper", EMBED_PATH)
    assert spec is not None and spec.loader is not None, (
        "Could not load embed.py module spec."
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "embed_text"), "embed.py must define embed_text(text)."
    vec = mod.embed_text("hello world")
    arr = np.asarray(vec)
    assert arr.shape == (32,), f"embed_text must return a 32-d vector, got shape {arr.shape}."


def test_zealt_run_id_set():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID environment variable must be set."


def test_lancedb_path_set_and_exists():
    p = os.environ.get("LANCEDB_PATH")
    assert p, "LANCEDB_PATH environment variable must be set."
    assert os.path.isdir(p), f"LANCEDB_PATH directory {p} must exist."


def test_seeded_table_present_with_200_rows():
    import lancedb

    rid = os.environ["ZEALT_RUN_ID"]
    db_path = os.environ["LANCEDB_PATH"]
    db = lancedb.connect(db_path)
    table_name = f"docs_{rid}"
    assert table_name in db.table_names(), (
        f"Expected pre-seeded table {table_name} in {db_path}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 200, (
        f"Pre-seeded table {table_name} must contain exactly 200 rows."
    )


def test_seeded_table_has_fts_index_on_text():
    import lancedb

    rid = os.environ["ZEALT_RUN_ID"]
    db_path = os.environ["LANCEDB_PATH"]
    tbl = lancedb.connect(db_path).open_table(f"docs_{rid}")
    indices = tbl.list_indices()
    fts_on_text = [i for i in indices if "text" in list(i.columns)]
    assert fts_on_text, "Expected an FTS index on the text column of the seeded table."


def test_rigged_rows_present():
    import lancedb

    rid = os.environ["ZEALT_RUN_ID"]
    db_path = os.environ["LANCEDB_PATH"]
    tbl = lancedb.connect(db_path).open_table(f"docs_{rid}")
    rows = tbl.search().where("id = 42 OR id = 99").limit(10).to_list()
    by_id = {r["id"]: r for r in rows}
    assert 42 in by_id, "Rigged row id=42 must exist in the seeded table."
    assert 99 in by_id, "Rigged row id=99 must exist in the seeded table."
    assert "QUANTUMTOKEN42UNIQUE" in by_id[42]["text"], (
        "Row id=42 must contain the rare anchor token QUANTUMTOKEN42UNIQUE."
    )
    assert "VORTEXTOKEN99UNIQUE" in by_id[99]["text"], (
        "Row id=99 must contain the rare anchor token VORTEXTOKEN99UNIQUE."
    )
