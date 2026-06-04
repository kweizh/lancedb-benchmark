# LangChain Agent with a Custom LanceDB Search Tool

## Background
Build a LangChain agent that can answer questions about three fictional internal products — **Helios database**, **Aurora analytics**, and **Borealis CRM** — by calling a custom search tool backed by a pre-populated LanceDB knowledge base.

A LanceDB table named `internal_kb` has already been seeded inside the container at `/home/user/myproject/db` with about 40 short documents covering the three products, embedded with real OpenAI `text-embedding-3-small` vectors. Your job is to wire those vectors into a LangChain tool, give the tool to an agent that uses `gpt-4o-mini`, and expose a clean Python entrypoint.

## Requirements
- Implement `/home/user/myproject/agent.py` that exposes a function `ask(question: str) -> dict`.
- `ask(...)` must return a dictionary with at least:
  - `"answer"`: a string containing the agent's final natural-language answer.
  - `"tool_calls"`: a list of strings, each being the name of a tool that the agent invoked while answering the question.
- The module must define a LangChain tool (a function decorated with `@tool`) named `search_internal_kb` that takes a single `query: str` argument and returns a concatenated string of the top relevant snippets from the `internal_kb` LanceDB table.
- The agent must be built so that for any product question, it actually invokes `search_internal_kb` rather than answering from parametric knowledge.
- Use the OpenAI chat model `gpt-4o-mini` for the agent's LLM.

## Implementation Hints
- The LanceDB database lives at `/home/user/myproject/db`. The table is named `internal_kb` and uses real OpenAI `text-embedding-3-small` embeddings (1536 dimensions) in a `vector` column. Each row also has `id`, `product`, and `text` columns.
- For the tool body, embed the incoming query with the same `text-embedding-3-small` model and run a vector search on the table; return the top-K results as a single string.
- Construct the agent with the standard LangChain "tool calling" / "OpenAI tools" agent helpers from `langchain.agents`, paired with `langchain-openai`'s `ChatOpenAI`. Wrap the agent in `AgentExecutor` and configure it so that intermediate tool calls are recoverable.
- `ask(...)` should extract the list of tool names that were invoked from the executor's intermediate steps (or from the final response payload) and return them in the order they happened.
- `OPENAI_API_KEY` is available in the environment.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 -c "import agent, json; print(json.dumps(agent.ask('What is Helios?')))"`
- The module `agent` must be importable from `/home/user/myproject` and expose:
  - `search_internal_kb` — a callable LangChain tool (function decorated with `@tool`) accepting a single `query: str` argument and returning a string.
  - `ask(question: str) -> dict` — returns a dict with keys `answer` (str) and `tool_calls` (list of str).
- For any reasonable factual question about Helios, Aurora, or Borealis, `ask(...)` must:
  - Cause the agent to actually invoke `search_internal_kb` at least once (so `tool_calls` must contain `"search_internal_kb"`).
  - Return an `answer` string that contains the specific product fact present in the seeded knowledge base.

