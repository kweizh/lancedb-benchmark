import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"


@pytest.fixture(scope="session")
def agent_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import importlib

    # Ensure a fresh import to pick up the candidate's latest agent.py
    if "agent" in sys.modules:
        del sys.modules["agent"]
    module = importlib.import_module("agent")
    return module


def test_agent_module_exposes_search_tool(agent_module):
    tool = getattr(agent_module, "search_internal_kb", None)
    assert tool is not None, (
        "agent.search_internal_kb is not defined. The module must expose a LangChain "
        "tool named 'search_internal_kb'."
    )
    # LangChain tools have a `.name` attribute and `.invoke(...)` method.
    name = getattr(tool, "name", None)
    assert name == "search_internal_kb", (
        f"Expected tool .name == 'search_internal_kb', got {name!r}. "
        "Define the tool with @tool decorator from langchain_core.tools."
    )
    assert callable(getattr(tool, "invoke", None)), (
        "agent.search_internal_kb must be a LangChain tool (it should have an "
        ".invoke() method)."
    )


def test_search_tool_invocable_directly(agent_module):
    tool = agent_module.search_internal_kb
    out = tool.invoke({"query": "What transactions does Helios support?"})
    assert isinstance(out, str), (
        f"search_internal_kb must return a string, got {type(out).__name__}"
    )
    assert out.strip(), "search_internal_kb returned an empty string."
    assert "ACID" in out.upper(), (
        "Direct tool invocation should surface the seeded Helios fact mentioning "
        f"'ACID'. Got: {out!r}"
    )


def test_ask_helios_acid(agent_module):
    assert callable(getattr(agent_module, "ask", None)), (
        "agent.ask must be a callable function."
    )
    result = agent_module.ask(
        "What kind of transactions does the Helios database support?"
    )
    assert isinstance(result, dict), (
        f"ask(...) must return a dict, got {type(result).__name__}"
    )
    assert "answer" in result and isinstance(result["answer"], str), (
        f"ask(...) result must include a string 'answer'. Got: {result!r}"
    )
    assert "tool_calls" in result and isinstance(result["tool_calls"], list), (
        f"ask(...) result must include a list 'tool_calls'. Got: {result!r}"
    )
    assert "search_internal_kb" in result["tool_calls"], (
        "Agent must invoke the search_internal_kb tool when asked a product "
        f"question. tool_calls={result['tool_calls']!r}"
    )
    assert "ACID" in result["answer"].upper(), (
        "Agent answer should contain the seeded Helios fact 'ACID'. "
        f"Got: {result['answer']!r}"
    )


def test_ask_aurora_retention(agent_module):
    result = agent_module.ask(
        "What is the maximum data retention period for Aurora analytics?"
    )
    assert isinstance(result, dict)
    assert "search_internal_kb" in result.get("tool_calls", []), (
        "Agent must invoke search_internal_kb for the Aurora question. "
        f"tool_calls={result.get('tool_calls')!r}"
    )
    assert "365" in result.get("answer", ""), (
        "Agent answer for Aurora retention should contain '365'. "
        f"Got: {result.get('answer')!r}"
    )


def test_ask_borealis_price(agent_module):
    result = agent_module.ask("How much does Borealis CRM cost per seat?")
    assert isinstance(result, dict)
    assert "search_internal_kb" in result.get("tool_calls", []), (
        "Agent must invoke search_internal_kb for the Borealis question. "
        f"tool_calls={result.get('tool_calls')!r}"
    )
    assert "99" in result.get("answer", ""), (
        "Agent answer for Borealis seat price should contain '99'. "
        f"Got: {result.get('answer')!r}"
    )
