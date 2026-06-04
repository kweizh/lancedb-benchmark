import importlib
import os


WORKSPACE = "/workspace"
DEFAULT_DB_DIR = "/workspace/db"
OUTPUT_DIR = "/workspace/output"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb package is not importable in the environment."


def test_lancedb_embeddings_registry_importable():
    embeddings = importlib.import_module("lancedb.embeddings")
    assert hasattr(embeddings, "get_registry"), (
        "lancedb.embeddings.get_registry is not available; the embedding registry "
        "API is required for this task."
    )


def test_lancedb_pydantic_importable():
    pyd = importlib.import_module("lancedb.pydantic")
    assert hasattr(pyd, "LanceModel") and hasattr(pyd, "Vector"), (
        "lancedb.pydantic.LanceModel/Vector are required to define the schema."
    )


def test_openai_sdk_importable():
    openai = importlib.import_module("openai")
    assert openai is not None, "openai SDK is required by the LanceDB OpenAI registry."


def test_pyarrow_importable():
    pa = importlib.import_module("pyarrow")
    assert pa is not None, "pyarrow is required by lancedb."


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_default_db_directory_exists():
    assert os.path.isdir(DEFAULT_DB_DIR), (
        f"Default LanceDB directory {DEFAULT_DB_DIR} does not exist; the harness "
        "should pre-create an empty directory for the candidate to use."
    )


def test_output_directory_exists():
    assert os.path.isdir(OUTPUT_DIR), (
        f"Output directory {OUTPUT_DIR} does not exist; the candidate writes "
        "registry_results.json here."
    )


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), (
        "OPENAI_API_KEY must be set in the task environment so the LanceDB "
        "OpenAI embedding registry can call the real API."
    )
