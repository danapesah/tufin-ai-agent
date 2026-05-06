from typing import Optional
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=2000, example="What is the weather in Tokyo?")
    thread_id: str = Field(default="default", example="user-123")


class TraceStep(BaseModel):
    step_index: int
    step_type: str  # "tool_call" | "tool_result" | "final_answer" | "ai_reasoning" | "user_input"
    content: str
    tool_name: Optional[str] = None
    agent: Optional[str] = None  # sub-agent name when step originates from inside a wrapper tool
    msg_type: Optional[str] = None  # "HumanMessage" | "AIMessage" | "ToolMessage"


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class TaskResponse(BaseModel):
    task_id: str
    input: str
    answer: str
    status: str
    trace: list[TraceStep]
    model: str
    token_usage: TokenUsage
    latency_ms: int
    created_at: str


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    model: str
    timestamp: str
