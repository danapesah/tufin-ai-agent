import json
import sqlite3
from contextlib import contextmanager
from typing import Optional

from config import settings


def init_db() -> None:
    import os
    os.makedirs(os.path.dirname(settings.database_url) or ".", exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id           TEXT PRIMARY KEY,
                input             TEXT NOT NULL,
                answer            TEXT NOT NULL,
                trace             TEXT NOT NULL,
                status            TEXT NOT NULL,
                model             TEXT NOT NULL,
                total_tokens      INTEGER,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                latency_ms        INTEGER,
                created_at        TEXT NOT NULL
            )
        """)


@contextmanager
def _connect():
    conn = sqlite3.connect(settings.database_url)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_task(record: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO tasks VALUES (
                :task_id, :input, :answer, :trace, :status, :model,
                :total_tokens, :prompt_tokens, :completion_tokens, :latency_ms, :created_at
            )""",
            {**record, "trace": json.dumps(record["trace"])},
        )


def get_task(task_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["trace"] = json.loads(result["trace"])
    return result
