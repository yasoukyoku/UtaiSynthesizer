# -*- coding: utf-8 -*-
"""Check remote Qwen key prefix direction."""
import json
import struct
import urllib.request

URL = "https://hf-mirror.com/ACE-Step/Ace-Step1.5/resolve/main/Qwen3-Embedding-0.6B/model.safetensors"
HDR_UA = {"User-Agent": "Mozilla/5.0"}

req = urllib.request.Request(URL, headers={**HDR_UA, "Range": "bytes=0-7"})
with urllib.request.urlopen(req, timeout=60) as r:
    n = struct.unpack("<Q", r.read(8))[0]
req = urllib.request.Request(URL, headers={**HDR_UA, "Range": f"bytes=8-{8 + n - 1}"})
with urllib.request.urlopen(req, timeout=60) as r:
    h = json.loads(r.read(n))
keys = sorted(k for k in h if k != "__metadata__")
print("count:", len(keys))
print("first 5:", keys[:5])
print("last 3:", keys[-3:])
