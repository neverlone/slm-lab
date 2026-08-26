import json
import urllib.request

# 1. Create session
req_new = urllib.request.Request("http://localhost:8000/api/sessions/new", method="POST")
sess_data = json.loads(urllib.request.urlopen(req_new).read().decode())
session_id = sess_data["session_id"]
print(f"Created Session: {session_id}")

# 2. Send prompt
req_chat = urllib.request.Request(
    "http://localhost:8000/api/chat",
    data=json.dumps({
        "session_id": session_id,
        "model": "Takatsuki-8B",
        "messages": [{"role": "user", "content": "I just wrote 500 lines of spaghetti code and I feel proud. Roast me."}]
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("\n--- Sen Takatsuki Response ---")
with urllib.request.urlopen(req_chat) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: "):
            try:
                chunk = json.loads(line_str[6:])
                if "delta" in chunk:
                    print(chunk["delta"], end="", flush=True)
            except Exception:
                pass
print("\n------------------------------")

# 3. Check history
req_hist = urllib.request.Request(f"http://localhost:8000/api/history/{session_id}")
hist_data = json.loads(urllib.request.urlopen(req_hist).read().decode())
print(f"History entries recorded: {len(hist_data['history'])}")
