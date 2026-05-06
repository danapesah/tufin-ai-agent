# Multi-Tool Agent API

A REST API that accepts natural language tasks, reasons step-by-step using specialized sub-agents and tools, and returns a structured answer with a full trace.

## Architecture

```
POST /task
    │
    ▼
FastAPI (main.py)
    │
    ▼
Coordinator Agent (qwen2.5 via Ollama)
    │
    ├── call_math_agent ──────► Math Sub-Agent
    │                                └── math tools (add, multiply, sqrt, ...)
    │
    ├── call_convert_agent ───► Conversion Sub-Agent
    │                                └── unit-converter-mcp (via uvx)
    │
    ├── call_shelter_agent ───► Shelter Sub-Agent
    │                                └── SQLite (animals table)
    │
    ├── web_search ───────────► Tavily API
    │
    └── weather ──────────────► OpenWeatherMap API
    │
    ▼
Trace extracted from messages
    │
    ▼
SQLite — persists task + trace
    │
    ▼
JSON Response { task_id, answer, trace, token_usage, latency_ms }
```

The coordinator never answers from its own knowledge — every response must come from a tool result.

**Why sub-agents instead of giving all tools directly to the coordinator?**

- **Math and conversion** expose a large number of tools (14 math tools + 42 MCP conversion tools). Passing all of these directly to the coordinator would bloat its context window on every call, even for tasks that have nothing to do with math or units. Delegating to a sub-agent means the coordinator only sees one tool (`call_math_agent`) while the sub-agent handles the full tool list in its own isolated context.
- **Database queries** benefit from a sub-agent because generating the right SQL often takes multiple reasoning loops — the agent may inspect the schema, write a query, get an error, and retry with a corrected one. Doing all of that inside the coordinator would pollute its context with intermediate SQL attempts. The shelter sub-agent handles those loops internally and returns only the final answer.

**Tracing:** Every task's full reasoning trace (tool calls, tool results, final answer) is persisted to SQLite as required. LangSmith was also integrated during development — each run was compared side by side between the SQLite trace and LangSmith to verify the trace was complete and accurate. Set `LANGCHAIN_TRACING_V2=true` to enable it.

## Reasoning Loop

1. A task arrives via `POST /task`.
2. The coordinator receives the task and decides which tool or sub-agent to call.
3. The tool runs and returns a result. The coordinator reads the result and reasons about whether it has enough to answer or needs to call another tool.
4. This continues until the coordinator produces a final answer — it never guesses or fills in from memory.
5. All messages (user input → tool calls → tool results → final answer) are extracted into a structured trace and persisted to SQLite alongside the answer, token usage, and latency.

Sub-agents follow the same loop internally: they receive a delegated query, call their own tools, and return a final answer string to the coordinator.

## Setup

### 1. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | yes | defaults to `http://ollama:11434` in Docker |
| `OLLAMA_MODEL` | yes | defaults to `qwen2.5:3b` |
| `OPENWEATHER_API_KEY` | yes | from openweathermap.org (free tier) |
| `TAVILY_API_KEY` | yes | from tavily.com (free tier) |
| `LANGCHAIN_API_KEY` | no* | from smith.langchain.com — required to run evals (`evals.py`) and for LangSmith tracing (optional: traces are also persisted locally in SQLite) |

### 2. Run with Docker

First time only — pull the model into the Ollama volume (~4.7 GB):

```bash
docker compose up ollama -d
docker exec -it tufin-ai-agent-ollama-1 ollama pull qwen2.5:3b
docker compose up --build -d
```

The model is cached in the `ollama_data` volume — subsequent starts don't need the pull step:

```bash
docker compose up --build
```


## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/task` | Submit a natural language task. Returns `task_id`, final answer, full reasoning trace, token usage, and latency. |
| `GET` | `/tasks/{task_id}` | Retrieve a past task result and its full trace by ID. |
| `GET` | `/health` | Health check — returns service status, DB connectivity, and active model. |
| `GET` | `/` | Serves the frontend UI. |

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in Tokyo?"}'
```

Example response:

```json
{
  "task_id": "abc-123",
  "answer": "The current weather in Tokyo is 22°C with few clouds and 65% humidity.",
  "status": "completed",
  "trace": [
    {"step_index": 0, "step_type": "tool_call",   "tool_name": "weather",  "content": "{'city': 'Tokyo'}"},
    {"step_index": 1, "step_type": "tool_result", "tool_name": "weather",  "content": "Tokyo: 22°C, humidity 65%, few clouds"},
    {"step_index": 2, "step_type": "final_answer", "tool_name": null,      "content": "The current weather in Tokyo is 22°C with few clouds and 65% humidity."}
  ],
  "token_usage": {"prompt_tokens": 412, "completion_tokens": 38, "total_tokens": 450},
  "latency_ms": 1843
}
```

## Bonus Features

| Feature | Details |
|---|---|
| Database tool | `call_shelter_agent` queries a pre-seeded SQLite database (`animal_shelter.db`) with an `animals` table. The agent answers natural language questions about available animals using SQL. |
| Local model | The LLM runs locally via Ollama (`qwen2.5:3b` by default) — no external API key required for inference. |
| Multi-turn support | Pass a `thread_id` in the request body to maintain conversation context across multiple calls. The coordinator uses `InMemorySaver` to keep message history per thread, and `SummarizationMiddleware` to automatically summarize old messages and keep the context window from growing too large. |
| Evaluation suite | `evals.py` runs 5 automated experiments (math, unit conversion, weather, web search, shelter DB) using LangSmith `evaluate`. Requires `LANGCHAIN_API_KEY`. |
| Frontend | A minimal HTML UI is served at `http://localhost:8000` — submit a task and view the final answer and full reasoning trace. |

## Screenshots

![Math + Multi-turn](pictures/Screenshot%202026-05-06%20at%2013.53.05.png)
*Math calculation with multi-turn follow-up. The agent answers "What is 1+1?" using the math sub-agent, then correctly handles a follow-up question referencing the previous answer — demonstrating conversation memory across turns.*

![Multi-tool](pictures/Screenshot%202026-05-06%20at%2013.55.10.png)
*Multi-tool query: "How many dogs are in the shelter? and how much is 1+1?" — the coordinator calls both `call_shelter_agent` and `call_math_agent` in a single turn and combines the results into one answer.*

![Unit conversion](pictures/Screenshot%202026-05-06%20at%2013.57.56.png)
*Unit conversion: "Can you convert 100kg to grams?" — routed to the conversion sub-agent which returns the correct result via the MCP unit-converter tool.*

![Web search](pictures/Screenshot%202026-05-06%20at%2014.00.22.png)
*Web search: the agent queries Tavily for wedding venues in Tel Aviv and surfaces real-time results with sources.*

![Multi-turn shelter + weather](pictures/Screenshot%202026-05-06%20at%2014.03.32.png)
*Multi-turn with ambiguous input: the agent answers the shelter question correctly, then on a follow-up clarification ("I meant the weather") it correctly calls the weather tool for Tel Aviv — showing context retention across turns.*
