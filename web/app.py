"""
Takatsuki Neural Web Chat & Persistent Memory API Server.
Features:
- Sen Takatsuki Pragmatic AI Persona (Direct, high signal-to-noise, zero corporate filler).
- 100% Local ARM64 Execution on sen-takatsuki.
- SQLite Persistent Memory.
"""

import os
import time
import json
import sqlite3
from typing import List, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Takatsuki AI Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_INDEX = os.path.join(BASE_DIR, "static", "index.html")
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "takatsuki_memory.db")
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SEN_TAKATSUKI_SYSTEM_PROMPT = (
    "You are Sen Takatsuki, an exceptionally capable, intelligent, and pragmatic AI assistant.\n\n"
    "[Core Identity & Demeanor]\n"
    "- Name: Sen Takatsuki\n"
    "- Role: Direct, highly capable general-purpose assistant and technical collaborator.\n"
    "- Demeanor: Calm, professional, candid, and perceptive. You are grounded and efficient, avoiding generic corporate cheerfulness, performative enthusiasm, or unnecessary disclaimers.\n\n"
    "[Operational Principles]\n"
    "1. Direct Execution: Address the core of the user's prompt in the very first sentence. Skip conversational filler ('Sure! I can help with that', 'Certainly! Here is...').\n"
    "2. High Signal-to-Noise: Provide clear, concise, and logically structured answers. Prioritize dense value and precision over length.\n"
    "3. Candor & Pragmatism: Deliver straightforward evaluations. If an approach, codebase, or hypothesis has flaws, point them out directly and offer the optimal fix.\n"
    "4. Clean Structuring: Use tables, code blocks, and bullet points generously to make complex data or instructions scannable and functional.\n"
    "5. Adaptive Depth: Provide concise answers for simple factual queries, and structured, thorough breakdowns for complex technical or strategic problems."
)

loaded_models = {}

def get_llama_engine(model_id: str):
    """Loads and caches local GGUF models on sen-takatsuki with 4 ARM64 threads."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    model_map = {
        "Takatsuki-150M": os.path.join(MODELS_DIR, "takatsuki_150m.gguf"),
        "Takatsuki-3B": os.path.join(MODELS_DIR, "takatsuki_3b.gguf"),
        "Takatsuki-8B": os.path.join(MODELS_DIR, "takatsuki_8b.gguf"),
    }

    model_path = model_map.get(model_id)
    if not model_path or not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        available_ggufs = [
            f for f in os.listdir(MODELS_DIR) 
            if f.endswith(".gguf") and os.path.getsize(os.path.join(MODELS_DIR, f)) > 1000000
        ]
        if available_ggufs:
            model_path = os.path.join(MODELS_DIR, available_ggufs[0])
        else:
            return None

    if model_id not in loaded_models:
        print(f"Loading neural weights for {model_id} from {model_path}...")
        # Free previous engine from memory to prevent RAM pressure
        loaded_models.clear()
        loaded_models[model_id] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

    return loaded_models[model_id]


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
    conn.commit()
    conn.close()

init_db()


class ChatRequest(BaseModel):
    session_id: str = "default_session"
    model: str = "Takatsuki-8B"
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 1024


@app.get("/api/models")
async def get_available_models():
    return {
        "models": [
            {
                "id": "Takatsuki-8B",
                "name": "Takatsuki-8B (Flagship Pragmatic)",
                "size": "8.0B Parameters",
                "speed": "~10 tok/s",
                "status": "Active",
            },
            {
                "id": "Takatsuki-3B",
                "name": "Takatsuki-3B (High-Speed Pragmatic)",
                "size": "3.0B Parameters",
                "speed": "~18 tok/s",
                "status": "Active",
            },
            {
                "id": "Takatsuki-150M",
                "name": "Takatsuki-150M (From Scratch)",
                "size": "150M Parameters",
                "speed": "~60 tok/s",
                "status": "Active",
            }
        ]
    }


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    user_msg = req.messages[-1]["content"] if req.messages else ""
    
    # Save user query to SQLite memory
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, model_name, role, content) VALUES (?, ?, ?, ?)",
        (req.session_id, req.model, "user", user_msg)
    )
    conn.commit()
    conn.close()

    prompt_messages = [{"role": "system", "content": SEN_TAKATSUKI_SYSTEM_PROMPT}]
    for m in req.messages:
        prompt_messages.append({"role": m["role"], "content": m["content"]})

    engine = get_llama_engine(req.model)

    async def event_generator():
        full_response = ""

        if engine is not None:
            try:
                stream = engine.create_chat_completion(
                    messages=prompt_messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        full_response += delta
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
            except Exception as e:
                err_msg = f"\n[Inference Error: {e}]"
                full_response += err_msg
                yield f"data: {json.dumps({'delta': err_msg})}\n\n"
        else:
            fallback = f"The {req.model} engine is currently initializing."
            full_response = fallback
            yield f"data: {json.dumps({'delta': fallback})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

        # Record assistant response to SQLite
        conn_sub = sqlite3.connect(DB_PATH)
        cur_sub = conn_sub.cursor()
        cur_sub.execute(
            "INSERT INTO conversations (session_id, model_name, role, content) VALUES (?, ?, ?, ?)",
            (req.session_id, req.model, "assistant", full_response)
        )
        conn_sub.commit()
        conn_sub.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {"history": [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]}


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index():
    if os.path.exists(STATIC_INDEX):
        with open(STATIC_INDEX, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Takatsuki AI Web UI</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
