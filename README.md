# Research Agent

A command-line and HTTP research assistant built with Claude, LangGraph and FastAPI.
It answers questions by searching the web, reading the results, and keeping track of
the conversation.

## What it does

Ask it a question and it decides for itself whether it needs to search. If it does,
it calls a search tool, reads the results, and may search again before answering.
This is the ReAct pattern: the model reasons about what it needs, acts by calling a
tool, observes the result, and repeats until it can answer.

The model never runs any code itself. It returns a structured request such as
*"call `web_search` with query X"*, the Python loop executes the function locally,
and the result is sent back to the model as the next message.

## Tools

| Tool | What it does |
|---|---|
| `web_search` | Searches DuckDuckGo and returns the top results with their URLs. |
| `read_file` | Reads a local text file so the agent can summarise or analyse it. |

Each tool returns a **string**, including when it fails. A failed search returns
`"Search failed: ..."` rather than raising, so the model can read what went wrong
and try a different query instead of crashing the loop.

## Running it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

**Command line:**

```bash
python agent.py
```

Type `new` to start a fresh conversation, `quit` to exit.

**HTTP API:**

```bash
uvicorn api:app --reload
```

Open <http://127.0.0.1:8000/docs> for an interactive page to test it.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What happened in Syria today?", "session_id": "test-1"}'
```

The response includes the answer, the tools the agent used, and how many steps it
took:

```json
{
  "answer": "...",
  "session_id": "test-1",
  "tool_calls": [{"name": "web_search", "args": {"query": "Syria news today"}}],
  "steps": 4
}
```

## Conversation memory

History is stored per `session_id` using LangGraph's `MemorySaver`. Two requests
with the same id continue the same conversation; a different id starts a new one.

Memory is held in RAM, so it is lost when the server restarts. Swapping
`MemorySaver` for `SqliteSaver` would make it persistent.

## Things I ran into

**The agent trusted its training data over its tools.** Asked about a recent match,
it reported a result from the previous season as upcoming, and told me the current
date was "in the future". The model has no clock, so when search results disagreed
with its internal sense of time it assumed the search was wrong. The fix was to put
today's date in the system prompt and stamp it onto the search output, next to the
evidence the model is reading.

**A failing tool is worse than a crashing one.** The search library had been renamed
and its old endpoints had stopped working. It returned an empty list instead of an
error, so the agent kept inventing plausible explanations for why it found nothing.
I only found the cause by calling the tool directly in a separate script, outside the
agent. The agent's description of its own failure was a guess, not a diagnosis.

**Results are only as current as their sources.** The agent is reliable on news and
market data, where pages carry publication dates, and much weaker on things like
sports fixtures, where the same evergreen page is republished across seasons. This
is a property of the sources, not of the model.

## Known limitations

- `read_file` can reach any path on the machine. It should be restricted to the
  project directory, with `.env` blocked, before this is run anywhere shared.
- No rate limiting or authentication on the API.
- DuckDuckGo throttles aggressively under repeated queries.
- Search returns short snippets only; a page-fetching tool would give fuller answers.

## Built with

Claude (Anthropic API) · LangChain · LangGraph · FastAPI · Pydantic · ddgs
