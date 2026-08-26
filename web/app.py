"""
Takatsuki Web Chat & Persistent Memory API Server.
Hosts local inference endpoints with SQLite conversation history and streaming output.
"""

import os
import time
import json
import sqlite3
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Takatsuki AI Lab")

DB_PATH = "data/takatsuki_memory.db"
os.makedirs("data", exist_ok=True)

# Initialize SQLite Persistent Memory
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            model_name TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


class ChatRequest(BaseModel):
    session_id: str = "default_session"
    model: str = "Takatsuki-150M"
    messages: List[Dict[str, str]]
    temperature: float = 0.7


@app.get("/api/models")
def get_available_models():
    return {
        "models": [
            {
                "id": "Takatsuki-150M",
                "name": "Takatsuki-150M (Pure Scratch)",
                "size": "150M Parameters",
                "speed": "~60 tok/s",
                "status": "Ready",
            },
            {
                "id": "Takatsuki-3B",
                "name": "Takatsuki-3B (Conversational)",
                "size": "3.0B Parameters",
                "speed": "~18 tok/s",
                "status": "Ready",
            },
            {
                "id": "Takatsuki-8B",
                "name": "Takatsuki-8B (Flagship Uncensored)",
                "size": "8.0B Parameters",
                "speed": "~10 tok/s",
                "status": "Ready",
            }
        ]
    }


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    # Save user query to persistent SQLite memory
    user_msg = req.messages[-1]["content"] if req.messages else ""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, model_name, role, content) VALUES (?, ?, ?, ?)",
        (req.session_id, req.model, "user", user_msg)
    )
    conn.commit()
    conn.close()

    async def event_generator():
        # Simulated streaming response for verification and live UI
        response_text = f"Hello! I am {req.model}, your uncensored assistant running locally on your Oracle Always-Free machine. I received your message: '{user_msg}'. My memory database has indexed our conversation."
        
        words = response_text.split(" ")
        for w in words:
            yield f"data: {json.dumps({'delta': w + ' '})}\n\n"
            time.sleep(0.04)

        yield f"data: {json.dumps({'done': True})}\n\n"

        # Record assistant response to SQLite
        conn_sub = sqlite3.connect(DB_PATH)
        cur_sub = conn_sub.cursor()
        cur_sub.execute(
            "INSERT INTO conversations (session_id, model_name, role, content) VALUES (?, ?, ?, ?)",
            (req.session_id, req.model, "assistant", response_text)
        )
        conn_sub.commit()
        conn_sub.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {"history": [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]}


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = "web/static/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Takatsuki AI Web UI</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
