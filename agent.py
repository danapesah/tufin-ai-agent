import time
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

from config import settings
from tools import weather, web_search, MATH_TOOLS


class TaskState(AgentState):
    math_query: str
    convert_query: str
    search_query: str
    weather_city: str


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

    web_search_agent = create_agent(
        model=settings.model,
        tools=[web_search],
        system_prompt="""You are a web search specialist. Search the web to answer the given query.
        You are not allowed to ask any more follow up questions, you must find the best answer based on the following criteria:
        - Accuracy (most reliable and recent sources)
        - Relevance (directly answers the query)
        You may need to make multiple searches to iteratively find the best answer.
        After each search result, briefly explain what you found and whether it answers the query before deciding your next step.
        You have a suggested limit of 5 web searches. Count every web_search call you make.
        After 5 searches, stop and summarize the best information you have found so far.""",
    )

    weather_agent = create_agent(
        model=settings.model,
        tools=[weather],
        system_prompt="""You are a weather specialist. Fetch and report the current weather for the requested city.
        You are not allowed to ask any more follow up questions, you must retrieve and report the weather based on the city provided.
        - Report temperature, feels-like temperature, humidity, and conditions
        - Present the information clearly and concisely
        After receiving the weather data, briefly explain what you see in the result before returning your answer.
        Once you have the weather data, return it immediately.""",
    )

    # ── update_state tool ───────────────────────────────────────────────────
    @tool
    def update_state(
        math_query: str,
        convert_query: str,
        search_query: str,
        weather_city: str,
        runtime: ToolRuntime,
    ) -> str:
        """Update the task state with queries for each specialist.
        Call this first, alone, before delegating to any specialist.
        Use an empty string for specialists that are not needed for this task.
        If all fields are empty strings, the user input is unclear — do not update state, ask the user to clarify."""
        if not any([math_query, convert_query, search_query, weather_city]):
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
                "search_query": search_query,
                "weather_city": weather_city,
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

    @tool
    async def call_web_search_agent(runtime: ToolRuntime) -> str:
        """Delegate the web search to the web search specialist agent."""
        query = runtime.state["search_query"]
        response = await web_search_agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    async def call_weather_agent(runtime: ToolRuntime) -> str:
        """Delegate the weather lookup to the weather specialist agent."""
        city = runtime.state["weather_city"]
        response = await weather_agent.ainvoke({"messages": [HumanMessage(content=city)]})
        return response["messages"][-1].content

    # ── Coordinator ─────────────────────────────────────────────────────────
    coordinator = create_agent(
        model=settings.model,
        tools=[update_state, call_math_agent, call_convert_agent, call_web_search_agent, call_weather_agent],
        state_schema=TaskState,
        system_prompt="""You are a general-purpose assistant with access to specialist agents.
        First, analyze the user's request and determine which specialists are needed.
        When you have all the information, call update_state with the relevant queries.
        This tool must be called alone, without any other tool calls. It must complete and return before you proceed.
        If update_state reports that all fields are empty, the user input is unclear — ask the user to clarify and do not delegate to any specialist.
        Once the state is updated successfully, delegate only to the specialists that are needed.
        You must call at least one specialist — never answer directly without delegating.
        Not all specialists need to be called in every run — only call the ones relevant to the task.
        After receiving each specialist's result, briefly explain what you learned from it before deciding your next step.
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
            "search_query": "",
            "weather_city": "",
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
        })

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            if msg.content and msg.tool_calls:
                steps.append({
                    "step_index": len(steps),
                    "step_type": "ai_reasoning",
                    "content": str(msg.content),
                    "tool_name": None,
                })
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "step_index": len(steps),
                        "step_type": "tool_call",
                        "content": str(tc.get("args", {})),
                        "tool_name": tc.get("name"),
                    })
            elif msg.content:
                steps.append({
                    "step_index": len(steps),
                    "step_type": "final_answer",
                    "content": msg.content,
                    "tool_name": None,
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "step_index": len(steps),
                "step_type": "tool_result",
                "content": str(msg.content),
                "tool_name": msg.name,
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
