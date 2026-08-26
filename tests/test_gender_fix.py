import json
import urllib.request

req = urllib.request.Request(
    "http://localhost:8000/api/chat",
    data=json.dumps({
        "session_id": "test_gender_fix",
        "model": "Takatsuki-8B",
        "messages": [{"role": "user", "content": "Tell me a little bit about yourself and how you approach writing."}]
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Streaming Sen Takatsuki (Gender Fix Verification):")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: "):
            try:
                chunk = json.loads(line_str[6:])
                if "delta" in chunk:
                    print(chunk["delta"], end="", flush=True)
            except Exception:
                pass
print("\n[Stream Complete]")
