import requests, json, os, sys

url = "http://127.0.0.1:7863/api/music/generate"
body = {
    "style": "English, warm piano pop, expressive female voice, 88 BPM",
    "lyrics": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n[Chorus]\nLet the day come into view\nEvery road begins with you",
    "model": "yue2",
    "cot": "full"
}

print("Sending request to YuE-2 (this takes 30-120s for GPU inference)...", flush=True)
sys.stdout.flush()

resp = requests.post(url, json=body, timeout=900)
print(f"Status: {resp.status_code}", flush=True)
ct = resp.headers.get("Content-Type", "")
print(f"Content-Type: {ct}", flush=True)
print(f"Content-Length: {len(resp.content)} bytes", flush=True)

# 尝试 JSON
try:
    data = resp.json()
    print("JSON response!", flush=True)
    text = json.dumps(data, indent=2, default=str)
    print(text[:3000], flush=True)
except Exception as e:
    print(f"Not JSON ({e}), trying raw...", flush=True)
    if "audio" in ct or "wav" in ct:
        with open("test_out.wav", "wb") as f:
            f.write(resp.content)
        print(f"Saved test_out.wav ({len(resp.content)} bytes)", flush=True)
    else:
        print(f"First 200 bytes hex:", flush=True)
        print(resp.content[:200].hex(), flush=True)
        # 看看能不能解析成 MIDI
        if resp.content[:4] == b"MThd":
            print("IT'S A MIDI FILE!", flush=True)
            with open("test_out.mid", "wb") as f:
                f.write(resp.content)
            print("Saved test_out.mid", flush=True)
