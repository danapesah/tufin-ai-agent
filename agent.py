import time
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

from config import settings
from tools import weather, web_search, query_db, MATH_TOOLS


class TaskState(AgentState):
    math_query: str
    convert_query: str
    shelter_query: str


async def build_agent():
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
        model=settings.model,
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
        model=settings.model,
        tools=MATH_TOOLS,
        system_prompt="""You are a math specialist. Perform the requested mathematical calculation using your tools.
        You are not allowed to ask any more follow up questions, you must compute the answer based on the input provided.
        - Use the most appropriate math tool for the operation
        - Show the result clearly
        You may need to make multiple tool calls to break down complex expressions.
        After each tool result, briefly explain what the result means before deciding your next step.
        Once you have the final result, return it immediately.""",
    )

    convert_agent = create_agent(
        model=settings.model,
        tools=convert_tools,
        system_prompt="""You are a unit conversion specialist. Convert between units using your tools.
        You are not allowed to ask any more follow up questions, you must perform the conversion based on the input provided.
        - Use the exact conversion tool that matches the requested unit types
        - Return the converted value with the target unit clearly stated
        If the conversion requires chaining multiple steps, do so iteratively.
        After each tool result, briefly explain what the result means before deciding your next step.
        Once you have the final result, return it immediately.""",
    )

    # ── update_state tool ───────────────────────────────────────────────────
    @tool
    def update_state(
        math_query: str,
        convert_query: str,
        shelter_query: str,
        runtime: ToolRuntime,
    ) -> str:
        """Update the task state with queries for math, unit-conversion, and shelter specialists.
        Call this first, alone, before delegating to any specialist.
        Use an empty string for specialists that are not needed for this task.
        If all fields are empty strings, the user input is unclear — do not update state, ask the user to clarify."""
        if not any([math_query, convert_query, shelter_query]):
            return Command(
                update={
                    "messages": [ToolMessage(
                        "All fields are empty — user input is unclear. Ask the user to clarify their request.",
                        tool_call_id=runtime.tool_call_id,
                    )],
                }
            )
        return Command(
            update={
                "math_query": math_query,
                "convert_query": convert_query,
                "shelter_query": shelter_query,
                "messages": [ToolMessage("State updated successfully.", tool_call_id=runtime.tool_call_id)],
            }
        )

    # ── Wrapper tools (closures over sub-agents, read from state) ──────────
    @tool
    async def call_math_agent(runtime: ToolRuntime) -> str:
        """Delegate the math calculation to the math specialist agent."""
        query = runtime.state["math_query"]
        response = await math_agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    async def call_convert_agent(runtime: ToolRuntime) -> str:
        """Delegate the unit conversion to the conversion specialist agent."""
        query = runtime.state["convert_query"]
        response = await convert_agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    # ── Coordinator ─────────────────────────────────────────────────────────
    coordinator = create_agent(
        model=settings.model,
        tools=[update_state, call_math_agent, call_convert_agent, web_search, weather],
        state_schema=TaskState,
        system_prompt="""You are a general-purpose assistant with access to specialist agents and direct tools.
        Analyze the user's request and determine what is needed:
        - For math or unit conversion: call update_state first (alone, before any other tool call) with the relevant queries, then delegate to call_math_agent or call_convert_agent.
        You must use at least one tool — never answer directly without using a tool.
        Not all tools need to be used in every run — only call the ones relevant to the task.
        After receiving each result, briefly explain what you learned before deciding your next step.
        Once you have all results, combine them into a clear final answer for the user.""",
    )

    return coordinator


async def run_task(agent, task: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content=task)],
            "math_query": "",
            "convert_query": "",
        }
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
        if isinstance(msg, AIMessage):
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
