"""Authenticated, stateless OpenAI-compatible API for missyou.one.

This process intentionally has no browser UI, CORS middleware, conversation
database, model upload route, or administrative endpoint.  Mi supplies the
complete authorized conversation context on every request.
"""

import asyncio
import hmac
import os
import re
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from llama_cpp import Llama
from pydantic import BaseModel, ConfigDict, Field


APP_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_ROOT / "models"
API_TOKEN = os.environ.get("TAKATSUKI_API_KEY", "")
CONTEXT_WINDOW = int(os.environ.get("TAKATSUKI_CONTEXT_WINDOW", "8192"))
THREADS = int(os.environ.get("TAKATSUKI_THREADS", "4"))
BATCH_THREADS = int(os.environ.get("TAKATSUKI_BATCH_THREADS", "8"))
MAX_PROMPT_CHARS = int(os.environ.get("TAKATSUKI_MAX_PROMPT_CHARS", "28000"))
QUEUE_WAIT_SECONDS = float(os.environ.get("TAKATSUKI_QUEUE_WAIT_SECONDS", "2"))

MODEL_PATHS = {
    "Takatsuki-3B-Uncensored": MODELS_DIR / "takatsuki_3b.gguf",
    "Takatsuki-150M": MODELS_DIR / "takatsuki_150m.gguf",
}

app = FastAPI(
    title="Takatsuki API for Missyou",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
inference_lock = asyncio.Lock()
loaded_engines: dict[str, Llama] = {}
models_ready = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    max_tokens: int = Field(default=420, ge=1, le=1200)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    stream: Literal[False] = False


def api_error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _exc: RequestValidationError):
    return api_error(400, "Invalid chat completion request", "invalid_request_error")


def authorize(authorization: str | None) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="Inference service is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid API token")


def engine_for(model_id: str) -> Llama:
    model_path = MODEL_PATHS.get(model_id)
    if model_path is None:
        raise ValueError("Unknown model")
    if not model_path.is_file() or model_path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Model artifact unavailable: {model_id}")
    if model_id not in loaded_engines:
        # There is deliberately no filename fallback: an invalid ID must never
        # load Takatsuki-8B. Both approved engines remain resident so switching
        # to the fallback cannot make the next 3B request pay a cold-load delay.
        engine = Llama(
            model_path=str(model_path),
            chat_format="llama-3",
            n_ctx=CONTEXT_WINDOW,
            n_threads=THREADS,
            # Prompt ingestion parallelizes well across all eight logical CPUs
            # on the i7-7700HQ; token generation remains on four physical cores.
            n_threads_batch=BATCH_THREADS,
            verbose=False,
        )
        loaded_engines[model_id] = engine
    return loaded_engines[model_id]


@app.on_event("startup")
async def warm_models() -> None:
    global models_ready
    # Loading before startup completes makes readiness truthful after reboot.
    # Start with the primary model, then keep the lightweight fallback resident.
    await asyncio.to_thread(engine_for, "Takatsuki-3B-Uncensored")
    await asyncio.to_thread(engine_for, "Takatsuki-150M")
    models_ready = True


def run_completion(payload: ChatCompletionRequest) -> dict:
    engine = engine_for(payload.model)
    result = engine.create_chat_completion(
        messages=[message.model_dump() for message in payload.messages],
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        repeat_penalty=1.12,
        stop=[
            "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "\nuser\n", "\nUser:", "\nHuman:", "\nassistant\n", "\nAssistant:",
            # Custom checkpoints sometimes omit the newline before a leaked
            # next-turn marker. Stop there so CPU time is not spent generating
            # a conversation that response cleanup will discard anyway.
            "user\n\n", "User\n\n", "assistant\n\n", "Assistant\n\n",
        ],
        stream=False,
    )
    text = str(result.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    # Some custom checkpoints emit their own assistant header before the real
    # answer. It belongs to the current turn, so remove it instead of treating
    # the whole completion as an empty synthetic next turn.
    text = re.sub(r"^(?:assistant)\s*(?::|\n)+\s*", "", text, count=1, flags=re.IGNORECASE)

    # Some custom chat templates leak the beginning of a synthetic next turn.
    # Truncate only role markers at a line boundary, never ordinary uses of the
    # words user, human, or assistant inside a response.
    text = re.split(
        r"(?:<\|start_header_id\|>\s*(?:user|assistant)|\n\s*(?:user|human|assistant)\s*(?::|\n)|(?<=[.!?])(?:user|human|assistant)\s*(?::|\n))",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if not text:
        raise RuntimeError("Model returned an empty completion")
    usage = result.get("usage") or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason") or "stop",
        }],
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    }


@app.get("/healthz")
async def healthz():
    artifacts = {model: path.is_file() and path.stat().st_size >= 1_000_000 for model, path in MODEL_PATHS.items()}
    status = 200 if API_TOKEN and all(artifacts.values()) and models_ready else 503
    return JSONResponse(
        status_code=status,
        content={"status": "ok" if status == 200 else "not_ready", "models": artifacts, "warmed": models_ready},
    )


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "Takatsuki"} for model in MODEL_PATHS]}


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    authorize(authorization)
    if payload.model not in MODEL_PATHS:
        return api_error(400, "Unsupported model", "invalid_request_error")
    prompt_chars = sum(len(message.content) for message in payload.messages)
    if prompt_chars > MAX_PROMPT_CHARS:
        return api_error(413, "Prompt is too large", "invalid_request_error")

    try:
        await asyncio.wait_for(inference_lock.acquire(), timeout=QUEUE_WAIT_SECONDS)
    except TimeoutError:
        response = api_error(429, "Inference server is busy", "rate_limit_error")
        response.headers["Retry-After"] = "5"
        return response
    try:
        return await asyncio.to_thread(run_completion, payload)
    except FileNotFoundError:
        return api_error(503, "Requested model is unavailable", "service_unavailable")
    except Exception as exc:
        print(f"[takatsuki-api] completion failed: {type(exc).__name__}", flush=True)
        return api_error(500, "Inference failed", "server_error")
    finally:
        inference_lock.release()
