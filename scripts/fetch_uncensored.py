"""
Direct downloader for public refusal-free models.
"""

import os
import urllib.request

MODELS_DIR = "/home/ubuntu/projects/slm-lab/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 8B Refusal-Free & Uncensored: Hermes-3 Llama-3.1 8B (Top ranked open uncensored model)
URL_8B = "https://huggingface.co/bartowski/Hermes-3-Llama-3.1-8B-GGUF/resolve/main/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf?download=true"

# 3B Refusal-Free / Uncensored: Qwen 2.5 3B Abliterated v2
URL_3B = "https://huggingface.co/mradermacher/Qwen2.5-3B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-3B-Instruct-abliterated-v2.Q4_K_M.gguf?download=true"

def download_file(url, out_path):
    print(f"\n[Downloading] {out_path} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        total_size = int(resp.info().get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 4 # 4MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size * 100) if total_size > 0 else 0
            mb = downloaded / (1024 * 1024)
            print(f"\rProgress: {mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print(f"\n[Complete] Saved {out_path}")

try:
    download_file(URL_3B, os.path.join(MODELS_DIR, "takatsuki_3b.gguf"))
except Exception as e:
    print(f"Failed 3B download: {e}")

try:
    download_file(URL_8B, os.path.join(MODELS_DIR, "takatsuki_8b.gguf"))
except Exception as e:
    print(f"Failed 8B download: {e}")

print("\n=== Refusal-Free Neural Weights Ready on sen-takatsuki ===")
