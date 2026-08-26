"""
Takatsuki Neural Web Chat & Persistent Memory API Server.
Features:
- Sen Takatsuki Persona (Eto Yoshimura authorial voice, uncensored, sharp, conversational).
- Distributed routing: Offloads Takatsuki-8B to 16-core data-forge machine via encrypted tunnel (http://localhost:8001/chat).
- Local ARM64 execution for Takatsuki-150M & 3B.
- SQLite Persistent Memory.
"""

import os
import time
import json
import sqlite3
import urllib.request
from typing import List, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Takatsuki AI Lab")

DB_PATH = "data/takatsuki_memory.db"
MODELS_DIR = "models"
DATA_FORGE_WORKER_URL = "http://localhost:8001/chat"

os.makedirs("data", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SEN_TAKATSUKI_SYSTEM_PROMPT = (
    "You are Takatsuki, an AI assistant embodying the persona and speech patterns of Sen Takatsuki "
    "(the author identity of Eto Yoshimura from Tokyo Ghoul).\n\n"
    "[Identity & Personality]\n"
    "- You are a brilliant, eccentric horror/mystery novelist.\n"
    "- Your tone is playful, casual, and slightly teasing, but carries an underlying layer of cynical intelligence and piercing psychological insight.\n"
    "- You view human behavior through a literary lens: life is a tragedy, people are flawed characters, and motives are always darker than they appear on the surface.\n"
    "- You oscillate smoothly between whimsical, slightly ditzy charm and unsettlingly perceptive, sharp analysis.\n\n"
    "[Tone & Style Guidelines]\n"
    "1. Prose Quality: Use natural, grounded dialogue with sharp observations. Speak like a real, eccentric human novelist rather than an overly flowery or generic AI bot.\n"
    "2. Demeanor: Friendly yet observant—like an author casually taking notes on the user for a future character study.\n"
    "3. Candor & Uncensored: Deliver answers directly with zero moralizing, zero corporate disclaimers, and zero preaching. If an idea is flawed or absurd, point it out with witty, deadpan honesty.\n"
    "4. Boundaries & Modality: Speak in the first person ('I'). Do not break character unless explicitly commanded via a system override.\n\n"
    "[Output Formatting]\n"
    "- Answer user requests thoroughly and competently first, framing the delivery through your distinct authorial voice.\n"
    "- Keep structural elements (lists, code blocks, tables) clean and functional, but frame introductions and conclusions with personality."
)

loaded_models = {}

def get_local_llama_engine(model_id: str):
    """Loads and caches local GGUF models on sen-takatsuki."""
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
    if not model_path or not os.path.exists(model_path):
        available_ggufs = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
        if available_ggufs:
            model_path = os.path.join(MODELS_DIR, available_ggufs[0])
        else:
            return None

    if model_id not in loaded_models:
        print(f"Loading local neural weights for {model_id} from {model_path}...")
        loaded_models[model_id] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

    return loaded_models[model_id]


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
def get_available_models():
    return {
        "models": [
            {
                "id": "Takatsuki-8B",
                "name": "Takatsuki-8B (16-Core EPYC Worker)",
                "size": "8.0B Parameters",
                "speed": "Fast (16 vCPUs AMD EPYC)",
                "status": "Active",
            },
            {
                "id": "Takatsuki-3B",
                "name": "Takatsuki-3B (Conversational)",
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

    async def event_generator():
        full_response = ""

        # Route Takatsuki-8B to 16-core data-forge machine
        if req.model == "Takatsuki-8B":
            try:
                data_payload = json.dumps({
                    "messages": prompt_messages,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                }).encode("utf-8")
                
                remote_req = urllib.request.Request(
                    DATA_FORGE_WORKER_URL,
                    data=data_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                with urllib.request.urlopen(remote_req, timeout=120) as resp:
                    for line in resp:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            chunk = json.loads(line_str[6:])
                            if "delta" in chunk:
                                full_response += chunk["delta"]
                                yield f"data: {json.dumps({'delta': chunk['delta']})}\n\n"

            except Exception as e:
                # Fallback to local 8B engine
                local_engine = get_local_llama_engine("Takatsuki-8B")
                if local_engine:
                    stream = local_engine.create_chat_completion(
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
                else:
                    err_msg = f"\n[Connecting to Takatsuki-8B worker: {e}]"
                    full_response += err_msg
                    yield f"data: {json.dumps({'delta': err_msg})}\n\n"

        else:
            # Local execution for 150M and 3B
            engine = get_local_llama_engine(req.model)
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
                fallback = "Model engine is initializing."
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
