import json
import os

PROJECT_DIR = "/home/user/myproject"
CORPUS_PATH = os.path.join(PROJECT_DIR, "corpus.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_cohere_sdk_importable():
    import cohere  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_corpus_file_exists():
    assert os.path.isfile(CORPUS_PATH), (
        f"Pre-baked corpus file {CORPUS_PATH} is missing."
    )


def test_corpus_has_90_rows_with_required_schema():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    assert isinstance(rows, list), "corpus.json must be a JSON array."
    assert len(rows) == 90, f"corpus.json must contain exactly 90 rows, got {len(rows)}."

    languages = {"en": 0, "es": 0, "fr": 0}
    concept_ids = {}
    for row in rows:
        assert isinstance(row, dict), "Each corpus entry must be a dict."
        for key in ("concept_id", "language", "text"):
            assert key in row, f"Corpus row is missing required key '{key}': {row}"
        assert row["language"] in languages, (
            f"Unexpected language {row['language']!r} in corpus row {row}."
        )
        languages[row["language"]] += 1
        cid = row["concept_id"]
        assert isinstance(cid, int) and 0 <= cid <= 29, (
            f"concept_id must be an int in 0..29, got {cid!r}."
        )
        concept_ids.setdefault(cid, set()).add(row["language"])
        assert isinstance(row["text"], str) and row["text"].strip(), (
            f"text must be a non-empty string for row {row}."
        )

    for lang, count in languages.items():
        assert count == 30, (
            f"Expected exactly 30 entries for language {lang!r}, got {count}."
        )

    assert len(concept_ids) == 30, (
        f"Expected exactly 30 distinct concept_ids, got {len(concept_ids)}."
    )
    for cid, langs in concept_ids.items():
        assert langs == {"en", "es", "fr"}, (
            f"concept_id {cid} must have all three languages, got {langs}."
        )


def test_cohere_api_key_env_present():
    assert os.environ.get("COHERE_API_KEY"), (
        "COHERE_API_KEY must be set in the environment so the candidate can call Cohere."
    )
