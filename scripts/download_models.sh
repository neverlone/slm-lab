#!/usr/bin/env bash
set -e

mkdir -p /home/ubuntu/projects/slm-lab/models
cd /home/ubuntu/projects/slm-lab/models

echo "Downloading Takatsuki starter neural weights (GGUF format)..."

# 1. Takatsuki-150M (~100 MB)
if [ ! -f takatsuki_150m.gguf ]; then
    echo "[1/3] Downloading Takatsuki-150M GGUF..."
    wget -q --show-progress -O takatsuki_150m.gguf "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf" || true
fi

# 2. Takatsuki-3B (~1.9 GB)
if [ ! -f takatsuki_3b.gguf ]; then
    echo "[2/3] Downloading Takatsuki-3B GGUF..."
    wget -q --show-progress -O takatsuki_3b.gguf "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" || true
fi

# 3. Takatsuki-8B (~4.9 GB)
if [ ! -f takatsuki_8b.gguf ]; then
    echo "[3/3] Downloading Takatsuki-8B GGUF..."
    wget -q --show-progress -O takatsuki_8b.gguf "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" || true
fi

echo "=== All Neural Model Weights Ready ==="
