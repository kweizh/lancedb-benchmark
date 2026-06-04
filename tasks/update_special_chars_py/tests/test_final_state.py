import json
import os

import pytest


OUTPUT_FILE = "/workspace/output/notes_after.json"
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "notes"

EXPECTED_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
UPDATED_BODY_ID2 = "I'm good"
UPDATED_BODY_ID4 = "It's a test"
UPDATED_AUTHOR_ID6 = "O'Brien"


@pytest.fixture(scope="module")
def output_rows():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Expected output file {OUTPUT_FILE} to exist after the task runs."
    )
    with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover
            pytest.fail(f"{OUTPUT_FILE} is not valid JSON: {exc}")
    assert isinstance(data, list), (
        f"{OUTPUT_FILE} must contain a JSON list, got {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def table_rows():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"LanceDB table '{TABLE_NAME}' was not found in {LANCEDB_URI}; "
        f"available tables: {table_names}."
    )
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    assert "id" in df.columns and "author" in df.columns and "body" in df.columns, (
        f"Table '{TABLE_NAME}' is missing required columns; got {list(df.columns)}."
    )
    rows = (
        df[["id", "author", "body"]]
        .sort_values("id")
        .to_dict(orient="records")
    )
    # Coerce numpy types to plain Python types for stable comparison.
    coerced = []
    for r in rows:
        coerced.append(
            {
                "id": int(r["id"]),
                "author": str(r["author"]),
                "body": str(r["body"]),
            }
        )
    return coerced


def test_output_file_row_count_and_ids(output_rows):
    assert len(output_rows) == 8, (
        f"Expected 8 rows in {OUTPUT_FILE}, got {len(output_rows)}."
    )
    ids = [row.get("id") for row in output_rows]
    assert ids == EXPECTED_IDS, (
        f"Expected ids {EXPECTED_IDS} in ascending order, got {ids}."
    )


def test_output_file_row_schema(output_rows):
    for row in output_rows:
        assert set(row.keys()) >= {"id", "author", "body"}, (
            f"Row {row} is missing one of required keys id/author/body."
        )
        assert isinstance(row["id"], int), (
            f"Row id must be an int, got {type(row['id']).__name__} for row {row}."
        )
        assert isinstance(row["author"], str), (
            f"Row author must be a string, got {type(row['author']).__name__} for row {row}."
        )
        assert isinstance(row["body"], str), (
            f"Row body must be a string, got {type(row['body']).__name__} for row {row}."
        )


def test_output_id2_body_has_apostrophe(output_rows):
    row = next(r for r in output_rows if r["id"] == 2)
    assert row["body"] == UPDATED_BODY_ID2, (
        f"Row id=2 must have body == {UPDATED_BODY_ID2!r}, got {row['body']!r}."
    )


def test_output_id4_body_has_apostrophe(output_rows):
    row = next(r for r in output_rows if r["id"] == 4)
    assert row["body"] == UPDATED_BODY_ID4, (
        f"Row id=4 must have body == {UPDATED_BODY_ID4!r}, got {row['body']!r}."
    )


def test_output_id6_author_has_apostrophe(output_rows):
    row = next(r for r in output_rows if r["id"] == 6)
    assert row["author"] == UPDATED_AUTHOR_ID6, (
        f"Row id=6 must have author == {UPDATED_AUTHOR_ID6!r}, got {row['author']!r}."
    )


def test_output_unmodified_rows_unchanged(output_rows):
    # The candidate may pick any deterministic seeded strings for the untouched rows,
    # but they MUST not contain apostrophes (we only inject apostrophes via the three updates).
    for row in output_rows:
        if row["id"] in {2, 4, 6}:
            continue
        assert "'" not in row["author"], (
            f"Row id={row['id']} author={row['author']!r} unexpectedly contains an apostrophe; "
            "only the three target rows should have apostrophes."
        )
        assert "'" not in row["body"], (
            f"Row id={row['id']} body={row['body']!r} unexpectedly contains an apostrophe; "
            "only the three target rows should have apostrophes."
        )


def test_table_matches_output_file(table_rows, output_rows):
    assert len(table_rows) == len(output_rows), (
        f"Table row count ({len(table_rows)}) does not match output file row count "
        f"({len(output_rows)})."
    )
    # Compare on (id, author, body) — table must match the file exactly.
    table_index = {r["id"]: r for r in table_rows}
    for row in output_rows:
        rid = row["id"]
        assert rid in table_index, (
            f"id={rid} present in output file but missing from LanceDB table."
        )
        tbl_row = table_index[rid]
        assert tbl_row["author"] == row["author"], (
            f"author mismatch for id={rid}: table={tbl_row['author']!r}, "
            f"file={row['author']!r}."
        )
        assert tbl_row["body"] == row["body"], (
            f"body mismatch for id={rid}: table={tbl_row['body']!r}, "
            f"file={row['body']!r}."
        )


def test_table_id2_body(table_rows):
    row = next(r for r in table_rows if r["id"] == 2)
    assert row["body"] == UPDATED_BODY_ID2, (
        f"In LanceDB table, row id=2 body must be {UPDATED_BODY_ID2!r}, got {row['body']!r}."
    )


def test_table_id4_body(table_rows):
    row = next(r for r in table_rows if r["id"] == 4)
    assert row["body"] == UPDATED_BODY_ID4, (
        f"In LanceDB table, row id=4 body must be {UPDATED_BODY_ID4!r}, got {row['body']!r}."
    )


def test_table_id6_author(table_rows):
    row = next(r for r in table_rows if r["id"] == 6)
    assert row["author"] == UPDATED_AUTHOR_ID6, (
        f"In LanceDB table, row id=6 author must be {UPDATED_AUTHOR_ID6!r}, got {row['author']!r}."
    )
