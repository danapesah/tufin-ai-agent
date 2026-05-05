# Multi-Tool Agent API

A general-purpose AI agent REST API built with LangChain and FastAPI. The agent accepts natural language tasks, reasons step-by-step using tools, and returns a final answer with a full structured trace of every reasoning step.

## Architecture



```
HTTP Request (POST /task)
    │
    ▼
FastAPI (main.py)
    │
    ▼
Coordinator Agent (gpt-4o)
    │
    ├── 1. update_state — determines which specialists are needed
    │
    ├── call_math_agent ──────► Math Sub-Agent (gpt-4o)
    │                                │
    │                                └──► TinyFn MCP /mcp/math/ (52 tools)
    │
    ├── call_convert_agent ───► Convert Sub-Agent (gpt-4o)
    │                                │
    │                                └──► TinyFn MCP /mcp/convert/ (42 tools)
    │
    ├── call_weather_agent ───► Weather Sub-Agent (gpt-4o)
    │                                │
    │                                └──► OpenWeatherMap API
    │
    └── call_web_search_agent ► Web Search Sub-Agent (gpt-4o)
                                     │
                                     └──► Tavily / DuckDuckGo
    │
    ▼
Trace extracted from result["messages"]
    │
    ▼
SQLite (database.py) — persists task + trace
    │
    ▼
JSON Response { task_id, answer, trace, token_usage, latency_ms }

Observability: all LLM calls and tool invocations are auto-traced to LangSmith
```

**Observability:** LangSmith provides automatic full tracing when `LANGCHAIN_TRACING_V2=true` is set. Every LLM call, tool call, token count, and latency is sent to [smith.langchain.com](https://smith.langchain.com) with no code changes. A local trace is also persisted in SQLite for the `GET /tasks/{task_id}` endpoint.

## Agent Reasoning Loop

1. The user submits a natural language task via `POST /task`.
2. `create_agent` runs a ReAct-style loop:
   - The LLM (gpt-4o) receives the task and decides which tool(s) to call.
   - Tools are executed and results returned to the LLM.
   - The LLM reasons about the results and either calls more tools or produces a final answer.
3. The full `result["messages"]` list is walked to build a structured trace.
4. The task, answer, trace, token usage, and latency are persisted to SQLite.

## Setup

### 1. Clone and configure environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

Required keys:
- `OPENAI_API_KEY` — from platform.openai.com
- `OPENWEATHER_API_KEY` — from openweathermap.org (free tier)
- `LANGCHAIN_API_KEY` — from smith.langchain.com (for LangSmith tracing)

Optional:
- `TAVILY_API_KEY` — from tavily.com (web search; falls back to DuckDuckGo if not set)

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

### 3. Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API is available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/task` | Submit a task |
| `GET` | `/tasks/{task_id}` | Retrieve a past task and its trace |
| `GET` | `/health` | Health check |

### POST /task

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in Paris?"}'
```

## Example Tasks and Traces

### 1. Calculator

**Task:** `"What is 144 / 12 + 5 ** 2?"`

**Trace:**
```json
[
  {"step_index": 0, "step_type": "tool_call",    "tool_name": "calculator", "content": "{'expression': '144 / 12 + 5 ** 2'}"},
  {"step_index": 1, "step_type": "tool_result",  "tool_name": "calculator", "content": "37.0"},
  {"step_index": 2, "step_type": "final_answer",  "tool_name": null,         "content": "144 / 12 + 5² = 12 + 25 = 37."}
]
```

### 2. Weather

**Task:** `"What is the current weather in Tokyo?"`

**Trace:**
```json
[
  {"step_index": 0, "step_type": "tool_call",    "tool_name": "weather", "content": "{'city': 'Tokyo'}"},
  {"step_index": 1, "step_type": "tool_result",  "tool_name": "weather", "content": "Tokyo: 22°C (feels like 21°C), humidity 65%, few clouds"},
  {"step_index": 2, "step_type": "final_answer",  "tool_name": null,      "content": "The current weather in Tokyo is 22°C with few clouds and 65% humidity."}
]
```

### 3. Unit Conversion

**Task:** `"Convert 100 miles to kilometers"`

**Trace:**
```json
[
  {"step_index": 0, "step_type": "tool_call",    "tool_name": "unit_converter", "content": "{'value': 100, 'from_unit': 'mi', 'to_unit': 'km'}"},
  {"step_index": 1, "step_type": "tool_result",  "tool_name": "unit_converter", "content": "160.9344"},
  {"step_index": 2, "step_type": "final_answer",  "tool_name": null,             "content": "100 miles is equal to approximately 160.93 kilometers."}
]
```

### 4. Web Search

**Task:** `"What is LangGraph and why is it useful?"`

**Trace:**
```json
[
  {"step_index": 0, "step_type": "tool_call",    "tool_name": "web_search", "content": "{'query': 'LangGraph what is it and why useful'}"},
  {"step_index": 1, "step_type": "tool_result",  "tool_name": "web_search", "content": "- https://langchain-ai.github.io/langgraph/: LangGraph is a library for building stateful, multi-actor applications..."},
  {"step_index": 2, "step_type": "final_answer",  "tool_name": null,         "content": "LangGraph is a framework built on top of LangChain for creating stateful agent workflows..."}
]
```

### 5. Multi-Tool (Weather + Unit Conversion)

**Task:** `"What is the temperature in London in Fahrenheit?"`

**Trace:**
```json
[
  {"step_index": 0, "step_type": "tool_call",    "tool_name": "weather",        "content": "{'city': 'London'}"},
  {"step_index": 1, "step_type": "tool_result",  "tool_name": "weather",        "content": "London: 15°C (feels like 13°C), humidity 72%, overcast clouds"},
  {"step_index": 2, "step_type": "tool_call",    "tool_name": "unit_converter", "content": "{'value': 15, 'from_unit': 'C', 'to_unit': 'F'}"},
  {"step_index": 3, "step_type": "tool_result",  "tool_name": "unit_converter", "content": "59.0"},
  {"step_index": 4, "step_type": "final_answer",  "tool_name": null,             "content": "The current temperature in London is 15°C, which is 59°F."}
]
```
