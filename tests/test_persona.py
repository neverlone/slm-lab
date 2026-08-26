import json
import urllib.request

req = urllib.request.Request(
    "http://localhost:8000/api/chat",
    data=json.dumps({
        "session_id": "test_eto_persona",
        "model": "Takatsuki-8B",
        "messages": [{"role": "user", "content": "What do you think of people who pretend to be completely innocent and harmless?"}]
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Streaming Sen Takatsuki Persona (8B running on 16-Core Data Forge):")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: "):
            chunk = json.loads(line_str[6:])
            if "delta" in chunk:
                print(chunk["delta"], end="", flush=True)
print("\n[Stream Complete]")
