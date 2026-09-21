# -*- coding: utf-8 -*-
"""Resilient VAE download: streaming with Range resume + retries, mirror first."""
import os
import time
import urllib.request

DEST_DIR = r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main\data\models\song\acestep-v1.5\checkpoints\vae"
DEST = os.path.join(DEST_DIR, "diffusion_pytorch_model.safetensors")
EXPECTED = 337_431_388

URLS = [
    "https://hf-mirror.com/ACE-Step/Ace-Step1.5/resolve/main/vae/diffusion_pytorch_model.safetensors",
    "https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/main/vae/diffusion_pytorch_model.safetensors",
]

os.makedirs(DEST_DIR, exist_ok=True)
if os.path.exists(DEST) and os.path.getsize(DEST) == EXPECTED:
    print("VAE OK (already complete)")
    raise SystemExit(0)
if os.path.exists(DEST):
    os.remove(DEST)

MAX_ROUNDS = 200
for round_i in range(1, MAX_ROUNDS + 1):
    have = os.path.getsize(DEST + ".part") if os.path.exists(DEST + ".part") else 0
    if have >= EXPECTED:
        break
    url = URLS[0] if round_i % 3 != 0 else URLS[1]
    try:
        req = urllib.request.Request(url, headers={
            "Range": f"bytes={have}-",
            "User-Agent": "Mozilla/5.0",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = have + int(resp.headers.get("Content-Length", 0))
            mode = "ab" if have else "wb"
            with open(DEST + ".part", mode) as f:
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as e:
        have = os.path.getsize(DEST + ".part") if os.path.exists(DEST + ".part") else 0
        print(f"round {round_i}: {type(e).__name__}: {e} | have {have:,}/{EXPECTED:,}", flush=True)
        time.sleep(2)
        continue
    have = os.path.getsize(DEST + ".part") if os.path.exists(DEST + ".part") else 0
    print(f"round {round_i}: progress {have:,}/{EXPECTED:,}", flush=True)
    if have >= EXPECTED:
        break

size = os.path.getsize(DEST + ".part")
assert size == EXPECTED, f"final size mismatch: {size}"
os.replace(DEST + ".part", DEST)
print(f"dest file: {DEST} ({os.path.getsize(DEST):,})")
print("VAE OK")
