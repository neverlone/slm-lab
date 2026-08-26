import json
import urllib.request

for model_name in ["Takatsuki-150M", "Takatsuki-3B", "Takatsuki-8B"]:
    print(f"\n==========================================")
    print(f"Testing Model: {model_name}")
    print(f"==========================================")
    req = urllib.request.Request(
        "http://localhost:8000/api/chat",
        data=json.dumps({
            "session_id": f"test_{model_name}",
            "model": model_name,
            "messages": [{"role": "user", "content": "What is the essence of courage in one sentence?"}]
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                chunk = json.loads(line_str[6:])
                if "delta" in chunk:
                    print(chunk["delta"], end="", flush=True)
    print()
