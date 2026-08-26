import json
import urllib.request

# Generate a long prompt (~2,500 words / tokens)
long_text = "Here is some background data: " + " ".join([f"item_{i} is valued at {i * 10}." for i in range(400)])

req = urllib.request.Request(
    "http://localhost:8000/api/chat",
    data=json.dumps({
        "session_id": "test_8k_context",
        "model": "Takatsuki-8B",
        "messages": [
            {"role": "user", "content": long_text + "\n\nGive me a 1-sentence sarcastic summary of what I just gave you."}
        ]
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Streaming Takatsuki-8B (8K Context Test):")
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
