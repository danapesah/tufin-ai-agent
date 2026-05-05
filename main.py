import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from agent import build_agent, run_task
from config import settings
from database import get_task, init_db, save_task
from schemas import HealthResponse, TaskRequest, TaskResponse, TokenUsage, TraceStep


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.agent = await build_agent()
    yield


app = FastAPI(title="Multi-Tool Agent API", version="1.0.0", lifespan=lifespan)


@app.post("/task", response_model=TaskResponse, status_code=201)
async def submit_task(req: TaskRequest, request: Request):
    task_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        result = await run_task(request.app.state.agent, req.task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    record = {
        "task_id": task_id,
        "input": req.task,
        "answer": result["answer"],
        "trace": result["trace"],
        "status": "completed",
        "model": settings.model,
        "total_tokens": result["token_usage"].get("total_tokens"),
        "prompt_tokens": result["token_usage"].get("prompt_tokens"),
        "completion_tokens": result["token_usage"].get("completion_tokens"),
        "latency_ms": result["latency_ms"],
        "created_at": created_at,
    }
    save_task(record)

    return TaskResponse(
        task_id=task_id,
        input=req.task,
        answer=result["answer"],
        status="completed",
        trace=[TraceStep(**s) for s in result["trace"]],
        model=settings.model,
        token_usage=TokenUsage(**result["token_usage"]),
        latency_ms=result["latency_ms"],
        created_at=created_at,
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task_by_id(task_id: str):
    record = get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return TaskResponse(
        task_id=record["task_id"],
        input=record["input"],
        answer=record["answer"],
        status=record["status"],
        trace=[TraceStep(**s) for s in record["trace"]],
        model=record["model"],
        token_usage=TokenUsage(
            prompt_tokens=record["prompt_tokens"],
            completion_tokens=record["completion_tokens"],
            total_tokens=record["total_tokens"],
        ),
        latency_ms=record["latency_ms"],
        created_at=record["created_at"],
    )


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        conn = sqlite3.connect(settings.database_url)
        conn.execute("SELECT 1")
        conn.close()
        db_connected = True
    except Exception:
        db_connected = False

    return HealthResponse(
        status="ok" if db_connected else "degraded",
        db_connected=db_connected,
        model=settings.model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
