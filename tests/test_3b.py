import json
import urllib.request

req = urllib.request.Request(
    "http://localhost:8000/api/chat",
    data=json.dumps({
        "session_id": "test_3b_pragmatic",
        "model": "Takatsuki-3B",
        "messages": [{"role": "user", "content": "What is the optimal way to structure microservices in Kubernetes?"}]
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Streaming Takatsuki-3B Pragmatic Response:")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: "):
            chunk = json.loads(line_str[6:])
            if "delta" in chunk:
                print(chunk["delta"], end="", flush=True)
print("\n[Stream Complete]")
