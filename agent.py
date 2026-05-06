import asyncio
import threading
import time
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from config import settings
from tools import weather, web_search, query_db, MATH_TOOLS


async def build_agent():
    chat_model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    # ── MCP clients ─────────────────────────────────────────────────────────
    convert_client = MultiServerMCPClient(
        {
            "unit-converter": {
                "transport": "stdio",
                "command": "uvx",
                "args": ["unit-converter-mcp"],
            }
        }
    )

    convert_tools = await convert_client.get_tools()

    # ── Sub-agents ─────────────────────────────────────────────────────────
    shelter_agent = create_agent(
        model=chat_model,
        tools=[query_db],
        system_prompt="""You are an animal shelter database specialist. Query the SQLite database to answer questions about the animals available for adoption.
        The database has a single table called 'animals' with columns: id, name, type, color, age.
        The 'type' column contains one of: 'dog', 'cat', or 'bird'.
        Before writing data queries, always discover the schema first if unsure.
        If a query fails, inspect the error and retry with a corrected query — do not come back empty-handed.
        Present your results clearly: list each animal's name, type, color, and age.
        If asked for counts or summaries, compute them with SQL aggregations rather than fetching all rows.""",
    )

    math_agent = create_agent(
        model=chat_model,
        tools=MATH_TOOLS,
        system_prompt="""You are a math specialist. Compute ONLY what the user asked — nothing else.
        Do not invent additional operations or go beyond the given input.
        - Use the most appropriate math tool for the operation
        - Return the result immediately once computed""",
    )

    convert_agent = create_agent(
        model=chat_model,
        tools=convert_tools,
        system_prompt="""You are a unit conversion specialist. Convert between units using your tools.
        You are not allowed to ask any more follow up questions, you must perform the conversion based on the input provided.
        - Use the exact conversion tool that matches the requested unit types
        - Return the converted value with the target unit clearly stated
        If the conversion requires chaining multiple steps, do so iteratively.
        After each tool result, briefly explain what the result means before deciding your next step.
        Once you have the final result, return it immediately.""",
    )

    # ── Wrapper tools ────────────────────────────────────────────────────────
    @tool
    async def call_math_agent(query: str) -> str:
        """Delegate a math calculation to the math specialist agent."""
        response = await math_agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    async def call_convert_agent(query: str) -> str:
        """Delegate a unit conversion to the conversion specialist agent."""
        response = await convert_agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    async def call_shelter_agent(query: str) -> str:
        """Delegate an animal shelter question to the shelter specialist agent."""
        response = await shelter_agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    # ── Coordinator ─────────────────────────────────────────────────────────
    coordinator = create_agent(
        model=chat_model,
        tools=[call_math_agent, call_convert_agent, call_shelter_agent, web_search, weather],
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=chat_model,
                trigger=("tokens", 3000),
                keep=("messages", 4),
            )
        ],
        system_prompt="""You are a general-purpose assistant with access to specialist tools.
        RULE 1: If the user's question can be answered by one of the tools above, you MUST use the tool. Never answer tool-related questions from your own knowledge.
        RULE 2: If a tool returns no data or fails, respond with "I could not retrieve that information." — do not guess.
        RULE 3: If the question is NOT related to any of the tools (e.g. greetings, general conversation, clarification questions), you may answer directly from your own knowledge.
        RULE 4: When calling a tool, always pass arguments as plain strings. For example: query="What is 1+1?" — never pass objects, dicts, or JSON structures as argument values.""",
    )

    return coordinator


_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_agent = None


def _get_sync_agent():
    global _bg_loop, _bg_agent
    if _bg_loop is None:
        _bg_loop = asyncio.new_event_loop()
        threading.Thread(target=_bg_loop.run_forever, daemon=True).start()
        _bg_agent = asyncio.run_coroutine_threadsafe(build_agent(), _bg_loop).result()
    return _bg_agent, _bg_loop


def run_agent_sync(task: str, thread_id: str | None = None) -> dict[str, Any]:
    """Synchronous entry point — safe to call from any thread, no asyncio required."""
    agent, loop = _get_sync_agent()
    tid = thread_id or str(uuid.uuid4())
    return asyncio.run_coroutine_threadsafe(run_task(agent, task, tid), loop).result()


async def run_task(agent, task: str, thread_id: str = "default") -> dict[str, Any]:
    start = time.perf_counter()
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        config,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    messages = result["messages"]
    answer = messages[-1].content
    trace = _extract_trace(messages)
    token_usage = _extract_token_usage(messages[-1])

    return {
        "answer": answer,
        "trace": trace,
        "latency_ms": latency_ms,
        "token_usage": token_usage,
    }


def _extract_trace(messages: list) -> list[dict]:
    steps = []

    if messages and isinstance(messages[0], HumanMessage):
        steps.append({
            "step_index": 0,
            "step_type": "user_input",
            "content": str(messages[0].content),
            "tool_name": None,
            "msg_type": type(messages[0]).__name__,
        })

    for msg in messages[1:]:
        if isinstance(msg, HumanMessage):
            steps.append({
                "step_index": len(steps),
                "step_type": "user_input",
                "content": str(msg.content),
                "tool_name": None,
                "msg_type": type(msg).__name__,
            })
        elif isinstance(msg, AIMessage):
            if msg.content and msg.tool_calls:
                steps.append({
                    "step_index": len(steps),
                    "step_type": "ai_reasoning",
                    "content": str(msg.content),
                    "tool_name": None,
                    "msg_type": type(msg).__name__,
                })
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "step_index": len(steps),
                        "step_type": "tool_call",
                        "content": str(tc.get("args", {})),
                        "tool_name": tc.get("name"),
                        "msg_type": type(msg).__name__,
                    })
            elif msg.content:
                steps.append({
                    "step_index": len(steps),
                    "step_type": "final_answer",
                    "content": msg.content,
                    "tool_name": None,
                    "msg_type": type(msg).__name__,
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "step_index": len(steps),
                "step_type": "tool_result",
                "content": str(msg.content),
                "tool_name": msg.name,
                "msg_type": type(msg).__name__,
            })
    return steps


def _extract_token_usage(last_message: AIMessage) -> dict:
    meta = getattr(last_message, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
