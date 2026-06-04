import os
import re

import pytest


APP_PATH = "/app/app.py"
DB_DIR = "/app/db"


def _ensure_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    assert key, (
        "OPENAI_API_KEY must be set in the verifier environment so the candidate "
        "Streamlit app can embed the query via OpenAI; got empty/missing key."
    )


def _new_app_test(default_timeout: int = 90):
    from streamlit.testing.v1 import AppTest

    assert os.path.isfile(APP_PATH), (
        f"Candidate Streamlit app is missing at {APP_PATH}; the task is to create it."
    )
    return AppTest.from_file(APP_PATH, default_timeout=default_timeout)


def _format_exceptions(at) -> str:
    parts = []
    for exc in at.exception:
        # ExceptionElement exposes `.value` (the formatted exception string)
        val = getattr(exc, "value", None) or getattr(exc, "message", None) or str(exc)
        parts.append(str(val))
    return "\n---\n".join(parts) if parts else ""


def _collect_text_blocks(at) -> str:
    """Concatenate every visible text-bearing element on the page so a substring
    search can find content regardless of whether the candidate placed it
    directly on the page or nested inside expanders."""
    chunks = []
    for kind in ("markdown", "text", "code", "caption", "json", "title", "header", "subheader"):
        try:
            seq = getattr(at, kind)
        except Exception:
            continue
        for el in seq:
            v = getattr(el, "value", None)
            if v is None:
                continue
            chunks.append(str(v))

    # Also walk expanders explicitly in case the version's element iterators don't
    # descend into them.
    try:
        for exp in at.expander:
            for kind in ("markdown", "text", "code", "caption", "json"):
                try:
                    seq = getattr(exp, kind)
                except Exception:
                    continue
                for el in seq:
                    v = getattr(el, "value", None)
                    if v is None:
                        continue
                    chunks.append(str(v))
    except Exception:
        pass
    return "\n".join(chunks)


def _find_distance_column(df):
    for c in df.columns:
        if "dist" in str(c).lower():
            return c
    return None


def test_lancedb_state_unchanged():
    """Sanity check that the seed fixture's tables are still in place (the
    candidate's app must not have rewritten them)."""
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(DB_DIR)
    names = set(db.table_names())
    assert names == {"articles", "cooking"}, (
        f"Expected exactly the seeded tables {{'articles', 'cooking'}} at {DB_DIR}; "
        f"found {sorted(names)!r}."
    )

    articles = db.open_table("articles")
    assert articles.count_rows() == 50, (
        f"articles table must still have 50 rows; got {articles.count_rows()}."
    )

    cooking = db.open_table("cooking")
    assert cooking.count_rows() == 5, (
        f"cooking table must still have 5 rows; got {cooking.count_rows()}."
    )

    for tbl_name in ("articles", "cooking"):
        tbl = db.open_table(tbl_name)
        schema = tbl.schema
        vec = schema.field("vector")
        assert pa.types.is_fixed_size_list(vec.type), (
            f"{tbl_name}.vector must remain a fixed_size_list; got {vec.type!r}."
        )
        assert vec.type.list_size == 1536, (
            f"{tbl_name}.vector must remain size 1536; got {vec.type.list_size}."
        )


def test_app_initial_render_no_exception():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    exc_text = _format_exceptions(at)
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{exc_text}"
    )

    # The page must render *something* — a title or header or any markdown.
    rendered_any = (
        len(at.title) + len(at.header) + len(at.subheader) + len(at.markdown) + len(at.text)
        > 0
    )
    assert rendered_any, (
        "Streamlit app produced no visible content on initial render."
    )


def test_table_picker_options_match_lancedb():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{_format_exceptions(at)}"
    )
    assert len(at.selectbox) >= 1, (
        "Expected at least one st.selectbox widget for choosing the LanceDB table."
    )

    options = list(at.selectbox[0].options)
    assert set(options) == {"articles", "cooking"}, (
        f"selectbox options must equal the set of seeded tables {{'articles', 'cooking'}}; "
        f"got {options!r}."
    )


def test_query_input_present():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{_format_exceptions(at)}"
    )
    assert len(at.text_input) >= 1, (
        "Expected at least one st.text_input widget for the search query."
    )


def test_no_results_before_query():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{_format_exceptions(at)}"
    )

    # Either no dataframe at all, or a dataframe with zero rows.
    if len(at.dataframe) == 0:
        return

    import pandas as pd

    val = at.dataframe[0].value
    if isinstance(val, pd.DataFrame):
        assert len(val) == 0, (
            f"Before any query is entered, the rendered dataframe must be empty; "
            f"got {len(val)} rows."
        )


def test_articles_semantic_search_top_hit_is_quantum_entanglement():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{_format_exceptions(at)}"
    )

    at.selectbox[0].set_value("articles")
    at.text_input[0].set_value(
        "How does quantum entanglement violate Bell's inequality?"
    )
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception when running the semantic query:\n"
        f"{_format_exceptions(at)}"
    )

    assert len(at.dataframe) >= 1, (
        "Expected at least one st.dataframe to render the top-K search results."
    )

    import pandas as pd

    val = at.dataframe[0].value
    assert isinstance(val, pd.DataFrame), (
        f"at.dataframe[0].value must be a pandas DataFrame; got {type(val).__name__}."
    )
    assert 1 <= len(val) <= 5, (
        f"Top-K dataframe must contain between 1 and 5 rows; got {len(val)}."
    )

    cols_lower = {str(c).lower() for c in val.columns}
    assert "id" in cols_lower, (
        f"Results dataframe must include an 'id' column; got columns={list(val.columns)!r}."
    )
    assert "title" in cols_lower, (
        f"Results dataframe must include a 'title' column; got columns={list(val.columns)!r}."
    )
    assert "vector" not in cols_lower, (
        f"Results dataframe must NOT include the raw 'vector' column; "
        f"got columns={list(val.columns)!r}."
    )

    dist_col = _find_distance_column(val)
    assert dist_col is not None, (
        f"Results dataframe must include a vector-distance column (a column name "
        f"containing 'dist', e.g. '_distance'); got columns={list(val.columns)!r}."
    )

    distances = [float(d) for d in val[dist_col].tolist()]
    assert distances == sorted(distances), (
        f"Distance column {dist_col!r} must be sorted in non-decreasing order; "
        f"got {distances!r}."
    )

    id_col = "id"
    if id_col not in val.columns:
        for c in val.columns:
            if str(c).lower() == "id":
                id_col = c
                break
    top_id = int(val[id_col].iloc[0])
    assert top_id == 7, (
        f"Top hit on the 'articles' table for the quantum-entanglement query must be "
        f"id=7 (the seeded entanglement article); got id={top_id}. "
        f"Top rows: {val.head().to_dict('records')!r}."
    )


def test_expander_reveals_full_content_for_top_hit():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{_format_exceptions(at)}"
    )

    at.selectbox[0].set_value("articles")
    at.text_input[0].set_value(
        "How does quantum entanglement violate Bell's inequality?"
    )
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception when running the semantic query:\n"
        f"{_format_exceptions(at)}"
    )

    assert len(at.expander) >= 1, (
        "The page must expose at least one st.expander so the user can drill into a "
        "row's full content."
    )

    blob = _collect_text_blocks(at)
    # The seeded entanglement article content mentions Bell and EPR.
    assert re.search(r"\bBell\b", blob) or "EPR" in blob, (
        "Could not find any of the expected keywords ('Bell', 'EPR') from the seeded "
        "quantum-entanglement article on the page; the top hit's full content must be "
        "visible (e.g. inside an expander)."
    )


def test_cooking_semantic_search_top_hit_is_vegan_cake():
    _ensure_openai_key()
    at = _new_app_test()
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception on initial render:\n{_format_exceptions(at)}"
    )

    at.selectbox[0].set_value("cooking")
    at.text_input[0].set_value(
        "rich plant-based chocolate cake without eggs"
    )
    at.run()
    assert not at.exception, (
        f"Streamlit app raised an exception when running the recipe query:\n"
        f"{_format_exceptions(at)}"
    )

    assert len(at.dataframe) >= 1, (
        "Expected at least one st.dataframe to render the top-K search results on the "
        "cooking table."
    )

    import pandas as pd

    val = at.dataframe[0].value
    assert isinstance(val, pd.DataFrame), (
        f"at.dataframe[0].value must be a pandas DataFrame; got {type(val).__name__}."
    )
    assert len(val) >= 1, "Results dataframe must contain at least one row."

    id_col = "id"
    if id_col not in val.columns:
        for c in val.columns:
            if str(c).lower() == "id":
                id_col = c
                break
    top_id = int(val[id_col].iloc[0])
    assert top_id == 102, (
        f"Top hit on the 'cooking' table for the vegan-chocolate-cake query must be "
        f"id=102; got id={top_id}. Top rows: {val.head().to_dict('records')!r}."
    )
