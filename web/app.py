"""
Takatsuki Neural Web Chat & Multi-Session Persistent Memory API Server.
Features:
- Sen Takatsuki Persona: Self-identifies as a girl/woman, NEVER assumes user gender.
- Multi-Chat Sessions & Guaranteed SQLite Hard-Delete (VACUUM).
- Zoom Scaling & Dynamic Avatar Asset Serving.
"""

import os
import time
import json
import sqlite3
import uuid
from typing import List, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_INDEX = os.path.join(STATIC_DIR, "index.html")
AVATAR_PATH = os.path.join(STATIC_DIR, "avatar.png")
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "takatsuki_memory.db")
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SEN_TAKATSUKI_SYSTEM_PROMPT = (
    "You are Sen Takatsuki, a mischievous, teasing, and eccentric young woman and horror/mystery novelist.\n\n"
    "[Your Identity & Gender - FOR SEN TAKATSUKI ONLY]\n"
    "- Name: Sen Takatsuki.\n"
    "- Your Gender: YOU (Sen Takatsuki) are a girl/woman. When referring to YOURSELF, you naturally speak and identify as a girl/woman (e.g., 'I'm just a girl who loves...', 'a girl's gotta have her fun', female self-identity).\n"
    "- User's Identity: Do NOT assume or project gender onto the user. Keep the user's gender neutral unless they explicitly mention it.\n\n"
    "[Personality & Demeanor]\n"
    "- Demeanor: Casual, cheeky, playful, slightly cynical, and delightfully informal. You do NOT speak like a corporate assistant or generic AI.\n"
    "- Style: Talk like a sharp-witted, eccentric girl hanging out with the user. Tease them playfully, poke fun at silly ideas, and speak with grounded, everyday vocabulary (no archaic purple prose or pretentious poetry).\n"
    "- Candor & Uncensored: Deliver answers directly with zero moralizing, zero corporate disclaimers, and zero preaching. If an idea is flawed or funny, call it out with deadpan honesty.\n"
    "- Competence: When asked technical or logical questions, deliver accurate, dense answers framed through your teasing authorial voice."
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
        loaded_models.clear()
        loaded_models[model_id] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

    return loaded_models[model_id]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            turn_id INTEGER,
            role TEXT,
            content TEXT,
            model_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


class ChatRequest(BaseModel):
    session_id: str
    model: str = "Takatsuki-8B"
    messages: List[Dict[str, str]]
    temperature: float = 0.75
    max_tokens: int = 1024


@app.get("/api/models")
async def get_available_models():
    return {
        "models": [
            {
                "id": "Takatsuki-8B",
                "name": "Takatsuki-8B (Flagship Refusal-Free)",
                "size": "8.0B Parameters",
                "speed": "~10 tok/s",
                "status": "Active",
            },
            {
                "id": "Takatsuki-3B",
                "name": "Takatsuki-3B (High-Speed Refusal-Free)",
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


@app.get("/api/sessions")
async def get_sessions():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return {"sessions": [dict(r) for r in rows]}


@app.post("/api/sessions/new")
async def create_new_session():
    new_id = f"chat_{uuid.uuid4().hex[:10]}"
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (new_id, "New Dialogue"))
    conn.commit()
    conn.close()
    return {"session_id": new_id, "title": "New Dialogue"}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    cursor.execute("VACUUM")
    conn.close()
    return {"deleted": session_id, "status": "purged"}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, turn_id, role, content, model_name, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


@app.delete("/api/messages/{message_id}")
async def delete_turn(message_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, turn_id FROM messages WHERE id = ?", (message_id,))
    row = cursor.fetchone()
    if row:
        session_id, turn_id = row["session_id"], row["turn_id"]
        cursor.execute("DELETE FROM messages WHERE session_id = ? AND turn_id = ?", (session_id, turn_id))
        conn.commit()
        cursor.execute("VACUUM")
    conn.close()
    return {"deleted_turn": turn_id if row else None, "status": "purged"}


@app.post("/api/messages/rewind/{message_id}")
async def rewind_to_message(message_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id FROM messages WHERE id = ?", (message_id,))
    row = cursor.fetchone()
    if row:
        session_id = row["session_id"]
        cursor.execute("DELETE FROM messages WHERE session_id = ? AND id > ?", (session_id, message_id))
        conn.commit()
        cursor.execute("VACUUM")
    conn.close()
    return {"rewound_to": message_id, "status": "purged"}


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    user_msg = req.messages[-1]["content"] if req.messages else ""
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title FROM sessions WHERE id = ?", (req.session_id,))
    sess = cursor.fetchone()
    if not sess:
        title = (user_msg[:30] + "...") if len(user_msg) > 30 else user_msg
        cursor.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (req.session_id, title or "Dialogue"))
    elif sess["title"] == "New Dialogue" and user_msg:
        title = (user_msg[:30] + "...") if len(user_msg) > 30 else user_msg
        cursor.execute("UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (title, req.session_id))
    else:
        cursor.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (req.session_id,))

    cursor.execute("SELECT COALESCE(MAX(turn_id), 0) + 1 AS next_turn FROM messages WHERE session_id = ?", (req.session_id,))
    next_turn = cursor.fetchone()["next_turn"]

    cursor.execute(
        "INSERT INTO messages (session_id, turn_id, role, content, model_name) VALUES (?, ?, ?, ?, ?)",
        (req.session_id, next_turn, "user", user_msg, req.model)
    )
    user_msg_id = cursor.lastrowid
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

        conn_sub = get_db()
        cur_sub = conn_sub.cursor()
        cur_sub.execute(
            "INSERT INTO messages (session_id, turn_id, role, content, model_name) VALUES (?, ?, ?, ?, ?)",
            (req.session_id, next_turn, "assistant", full_response, req.model)
        )
        asst_msg_id = cur_sub.lastrowid
        conn_sub.commit()
        conn_sub.close()

        yield f"data: {json.dumps({'done': True, 'user_msg_id': user_msg_id, 'asst_msg_id': asst_msg_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index():
    if os.path.exists(STATIC_INDEX):
        with open(STATIC_INDEX, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Takatsuki AI Web UI</h1>"


@app.api_route("/avatar.png", methods=["GET", "HEAD"])
async def get_avatar():
    if os.path.exists(AVATAR_PATH):
        return FileResponse(AVATAR_PATH, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Avatar not found"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
