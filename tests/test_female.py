import json
import urllib.request

req = urllib.request.Request(
    "http://localhost:8000/api/chat",
    data=json.dumps({
        "session_id": "test_female_persona",
        "model": "Takatsuki-8B",
        "messages": [{"role": "user", "content": "Who are you and what are you up to today?"}]
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Streaming Sen Takatsuki (Female Persona):")
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
